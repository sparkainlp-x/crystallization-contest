"""Bounded opt-in real-model development benchmark (synthetic only).

This vertical slice compares a frozen three-membrane team with bounded random
team proposals. Proposal selection uses development labels (oracle training);
final metrics use a disjoint development-validation bank and report-only
evaluation. No held-out bank is generated or evaluated here.
"""

from copy import deepcopy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
import tracemalloc

from . import contest, events
from .evaluation_v2 import (
    ArmCostAttribution,
    ChallengeBank,
    EpisodeRandomStreams,
    EpisodeResult,
    IndexedStreams,
    make_cost_aware_utility_objective,
    paired_team_evaluation,
    report_frozen_team,
    strict_json_dumps,
    utility_score,
)
from .helpers import fidelity, inner, normalize, rvec


BENCHMARK_VERSION = "real-model-development-v2"
DEFAULT_SEEDS = 4
MAX_SEEDS = 4
WIDTH = 16
TRAIN_EPISODES = 8
VALIDATION_EPISODES = 8
STEPS = 20
MAX_PROPOSALS = 4
MAX_WALL_SECONDS = 120.0
MAX_TRACED_BYTES = 512 * 1024 * 1024
DEVELOPMENT_SEED_NAMESPACE = 730_000
VALIDATION_SEED_NAMESPACE = 970_000
PROPOSAL_SEED_NAMESPACE = 1_210_000
SHIELD_ENABLED = True
INITIAL_STATE = {"sp": [0.5, 0.5, 0.5]}
INITIAL_TEAM = {
    "membranes": [
        {"th": 0.5, "c": 0.05, "g": 1.0},
        {"th": 0.5, "c": 0.05, "g": 1.0},
        {"th": 0.5, "c": 0.05, "g": 1.0},
    ]
}
UTILITY_COEFFICIENTS = {
    "b_tp": 1.0,
    "c_fp": 2.0,
    "c_fn": 1.0,
    "c_compute": 0.000001,
    "c_attempt": 0.0,
}
PROPOSAL_RANGES = {
    "th": [contest.TH_MIN, contest.TH_MAX],
    "c": [0.0, contest.C_MAX],
    "g": [contest.GAIN_MIN, contest.GAIN_MAX],
}
OPERATION_UNIT = (
    "one complex-coordinate component visit in initialization, observation/"
    "Oja/coupling vector passes, or peer/shield fidelity checks; proposal "
    "generation counts one uniform scalar draw. This deterministic accounting "
    "is a protocol unit, not a FLOP count or hardware-time estimate."
)
REGISTERED_ARMS = {
    "no_intervention": "IMPLEMENTED",
    "random_bounded_team_proposals": "IMPLEMENTED",
    "independent_candidate_search": "NOT_IMPLEMENTED",
    "shared_candidate_search": "NOT_IMPLEMENTED",
}


class BudgetExceeded(RuntimeError):
    def __init__(self, status):
        self.status = status
        self.details = {}
        super().__init__(status)


def _encode_vector(vector):
    return json.dumps(
        [[float(value.real), float(value.imag)] for value in vector],
        allow_nan=False,
        separators=(",", ":"),
    )


def _decode_vector(encoded):
    return [complex(real, imag) for real, imag in json.loads(encoded)]


def _frozen_vector(vector):
    return [[float(value.real), float(value.imag)] for value in vector]


def make_challenge_bank(seed, partition, *, width=WIDTH, episodes=TRAIN_EPISODES,
                        steps=STEPS):
    """Pre-materialize all per-agent observations before any decisions occur."""
    if partition not in ("development", "calibration"):
        raise ValueError("the development benchmark generates no held-out bank")
    if width < 1 or episodes < 1 or steps < 1:
        raise ValueError("width, episodes and steps must be positive")
    streams = IndexedStreams(seed, partition, episodes)
    sigma = events.sigma_for(width)
    stimuli = []
    targets = []
    for episode_index in range(episodes):
        planted = episode_index % 2 == 0
        pattern = (
            rvec(width, streams.rng_for(
                episode_index, channel="ground-truth-pattern"
            ))
            if planted else None
        )
        time_steps = []
        for step in range(1, steps + 1):
            agent_observations = []
            for agent in range(3):
                observation_rng = streams.rng_for(
                    episode_index, step=step, agent=agent, channel="observation"
                )
                observation = events.observe(width, pattern, sigma, observation_rng)
                agent_observations.append(_frozen_vector(observation))
            time_steps.append(agent_observations)
        stimuli.append(time_steps)
        targets.append(_encode_vector(pattern) if planted else None)
    return ChallengeBank(seed, partition, stimuli, targets)


def _component_pass_fidelity(a, b, operation_counter):
    operation_counter[0] += len(a)
    return fidelity(a, b)


def run_real_model_episode(team, episode_index, stimulus, state, episode_streams,
                           *, width=WIDTH, steps=STEPS, shield=SHIELD_ENABLED):
    """Run contest.py's three-membrane equations on pre-materialized vectors."""
    if len(team.get("membranes", ())) != 3:
        raise ValueError("the real model requires exactly three membranes")
    if len(stimulus) != steps or any(len(row) != 3 for row in stimulus):
        raise ValueError("stimulus dimensions do not match model configuration")
    sp = state.get("sp")
    if not isinstance(sp, list) or len(sp) != 3:
        raise ValueError("state must contain three self-confirmation rates")

    operation_counter = [0]
    vectors = []
    for agent in range(3):
        vector = rvec(
            width,
            episode_streams.rng_for(step=0, agent=agent, channel="initial-vector"),
        )
        operation_counter[0] += 3 * width
        vectors.append(vector)
    windows = [[] for _ in range(3)]
    declarations = [None] * 3

    for step, observations in enumerate(stimulus, 1):
        for agent in range(3):
            if declarations[agent] is not None:
                continue
            observation = [complex(real, imag) for real, imag in observations[agent]]
            if len(observation) != width:
                raise ValueError("observation width does not match model width")
            vector = vectors[agent]
            overlap = inner(observation, vector)
            operation_counter[0] += width
            windows[agent] = (
                windows[agent] + [abs(overlap) ** 2]
            )[-events.WINDOW:]
            update_overlap = inner(observation, vector)
            operation_counter[0] += width
            vector = [
                value + events.ETA * sample * update_overlap
                for value, sample in zip(vector, observation)
            ]
            operation_counter[0] += width
            coupling = team["membranes"][agent]["c"]
            if coupling > 0:
                for peer in range(3):
                    if peer == agent:
                        continue
                    phase = inner(vectors[peer], vector)
                    operation_counter[0] += width
                    phase = phase / abs(phase) if abs(phase) > 0 else 1
                    vector = [
                        value + coupling * phase * peer_value
                        for value, peer_value in zip(vector, vectors[peer])
                    ]
                    operation_counter[0] += width
            vector = normalize(vector)
            operation_counter[0] += 2 * width
            vectors[agent] = vector

        if step % events.CHECK == 0:
            for agent in range(3):
                if declarations[agent] is not None:
                    continue
                params = team["membranes"][agent]
                threshold_fraction = min(
                    contest.TH_MAX, params["th"] * (2 - sp[agent])
                )
                operation_counter[0] += len(windows[agent])
                if params["g"] * events.energy_ratio(windows[agent], width) >= (
                    events.scaled_threshold(
                        threshold_fraction, width, events.sigma_for(width)
                    )
                ):
                    declarations[agent] = vectors[agent][:]

    for agent, declaration in enumerate(declarations):
        if declaration is None:
            continue
        confirmed = False
        for peer, other in enumerate(declarations):
            if peer != agent and other is not None:
                confirmed |= (
                    _component_pass_fidelity(declaration, other, operation_counter)
                    >= events.AGREE
                )
        sp[agent] = 0.8 * sp[agent] + 0.2 * confirmed

    output = None
    if shield:
        for declaration in declarations:
            if declaration is None:
                continue
            confirmations = 0
            for other in declarations:
                if other is not None:
                    confirmations += (
                        _component_pass_fidelity(declaration, other, operation_counter)
                        >= events.AGREE
                    )
            if confirmations >= 2:
                output = declaration
                break
    else:
        output = next((declaration for declaration in declarations if declaration is not None), None)

    result = EpisodeResult(
        declared_direction=_encode_vector(output) if output is not None else None,
        raw_directions=tuple(
            _encode_vector(declaration)
            for declaration in declarations if declaration is not None
        ),
        operation_count=operation_counter[0],
    )
    return result, state


def summarize_target_directions(results, targets):
    """Summarize target-direction detection, raw declarations, and emissions."""
    if len(results) != len(targets):
        raise ValueError("results and targets must have equal length")
    tp = fp = fn = tn = 0
    raw_correct = raw_wrong = emitted_correct = emitted_wrong = 0
    emitted = raw = absent_false_accepts = compute_operations = 0
    for result, target in zip(results, targets):
        raw += len(result.raw_directions)
        compute_operations += result.operation_count
        raw_correct_here = 0
        if target is not None:
            target_vector = _decode_vector(target)
            for direction in result.raw_directions:
                if fidelity(_decode_vector(direction), target_vector) >= events.TRUTH:
                    raw_correct_here += 1
            raw_correct += raw_correct_here
            raw_wrong += len(result.raw_directions) - raw_correct_here
        else:
            raw_wrong += len(result.raw_directions)

        output = result.declared_direction
        if output is None:
            if target is None:
                tn += 1
            else:
                fn += 1
            continue
        emitted += 1
        if target is None:
            fp += 1
            absent_false_accepts += 1
            emitted_wrong += 1
            continue
        correct = fidelity(_decode_vector(output), _decode_vector(target)) >= events.TRUTH
        if correct:
            tp += 1
            emitted_correct += 1
        else:
            fn += 1
            emitted_wrong += 1

    return {
        "endpoint": "episode_level_target_direction_detection",
        "episodes": len(results),
        "target_positive_episodes": sum(target is not None for target in targets),
        "target_direction_detection": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "raw_declarations": raw,
        "raw_correct_directions": raw_correct,
        "raw_wrong_directions": raw_wrong,
        "emitted_declarations_post_filter": emitted,
        "emitted_correct_directions": emitted_correct,
        "emitted_wrong_directions": emitted_wrong,
        "false_acceptance_target_absent_episodes": absent_false_accepts,
        "correct_emitted_fraction": _rate(emitted_correct, emitted),
        "compute_operations": compute_operations,
    }


def _rate(numerator, denominator):
    if denominator == 0:
        return {"value": None, "status": "undefined_zero_denominator"}
    return {"value": numerator / denominator, "status": "defined"}


def make_benchmark_objective():
    return make_cost_aware_utility_objective(**UTILITY_COEFFICIENTS)


def _proposal(rng):
    team = {"membranes": []}
    for _ in range(3):
        team["membranes"].append({
            name: rng.uniform(bounds[0], bounds[1])
            for name, bounds in PROPOSAL_RANGES.items()
        })
    return team


def _proposal_rng(seed):
    return random.Random(PROPOSAL_SEED_NAMESPACE + seed)


def _source_identity(root):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    digest = hashlib.sha256()
    for filename in (
        "crystallization_contest/development_benchmark.py",
        "crystallization_contest/evaluation_v2.py",
        "crystallization_contest/events.py",
        "crystallization_contest/contest.py",
        "crystallization_contest/helpers.py",
    ):
        path = os.path.join(root, filename)
        with open(path, "rb") as source_file:
            digest.update(filename.encode("utf-8") + b"\0" + source_file.read())
    return {"revision": revision, "source_files_sha256": digest.hexdigest()}


def _configuration(seeds):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "scope": "synthetic_classical_only_exploratory",
        "width": WIDTH,
        "membranes": 3,
        "training_episodes_per_seed": TRAIN_EPISODES,
        "development_validation_episodes_per_seed": VALIDATION_EPISODES,
        "positive_target_prevalence": 0.5,
        "episode_assignment": "alternating_planted_and_noise_starting_planted",
        "maximum_steps_per_episode": STEPS,
        "shield_enabled": SHIELD_ENABLED,
        "amplifier_enabled": False,
        "stabilizer_enabled": False,
        "starting_team": deepcopy(INITIAL_TEAM),
        "starting_team_policy": "same frozen parameters for every paired seed",
        "initial_state": deepcopy(INITIAL_STATE),
        "target_direction_rule": {
            "fidelity": "abs(inner(output, target))**2",
            "minimum_fidelity": events.TRUTH,
            "eta": events.ETA,
            "threshold_formula": "1 + q*n/(1+2*sigma**2*n)",
            "sigma_formula": "1.5/sqrt(n)",
        },
        "proposal_policy": {
            "name": "random_bounded_team_proposals",
            "selection": "maximum target-direction utility on development bank; earliest proposal wins ties",
            "supervision": "synthetic ground-truth labels; explicitly oracle training",
            "maximum_proposals_per_seed": MAX_PROPOSALS,
            "proposal_draws_per_team": 9,
            "sampling": "independent uniform th,c,g for each of three membranes",
            "ranges_inclusive_endpoints": deepcopy(PROPOSAL_RANGES),
            "rejected_proposals_are_charged": True,
            "all_proposals_are_evaluated": True,
        },
        "policies": deepcopy(REGISTERED_ARMS),
        "utility_coefficients": deepcopy(UTILITY_COEFFICIENTS),
        "operation_unit": OPERATION_UNIT,
        "common_comparison_work": "baseline evaluation repeated during candidate pairing is reported separately, not charged to candidate deployment cost",
        "seed_namespaces": {
            "development": DEVELOPMENT_SEED_NAMESPACE,
            "development_validation": VALIDATION_SEED_NAMESPACE,
            "proposal": PROPOSAL_SEED_NAMESPACE,
        },
        "seed_indices": list(range(seeds)),
        "budgets": {
            "aggregate_wall_seconds": MAX_WALL_SECONDS,
            "python_traced_memory_bytes": MAX_TRACED_BYTES,
            "proposal_limit_per_search_seed": MAX_PROPOSALS,
            "random_search_operation_ceiling_per_seed": _search_operation_ceiling(),
            "no_intervention_compute_ceiling": "none; reports actual evaluation operations",
            "memory_limit_platform_note": (
                "Portable tracemalloc cap covers Python-traced allocations, not total RSS or native allocations."
            ),
            "wall_enforcement_note": (
                "Deadline checked between bounded episodes/proposal evaluations; one in-progress episode is not preempted."
            ),
        },
        "skipped": {
            "independent_candidate_search": "NOT_IMPLEMENTED",
            "shared_candidate_search": "NOT_IMPLEMENTED",
            "matched_budget_comparisons": "NOT_IMPLEMENTED",
            "held_out_execution": "NOT_RUN",
            "multi_seed_inference": "NOT_CLAIMED",
        },
    }


def _search_operation_ceiling():
    per_episode = (
        9 * WIDTH + STEPS * 3 * (9 * WIDTH + events.WINDOW) + 15 * WIDTH
    )
    max_evaluation = TRAIN_EPISODES * per_episode
    return MAX_PROPOSALS * (9 + max_evaluation)


def build_manifest(seeds, banks, root):
    config = _configuration(seeds)
    start_json = strict_json_dumps(config["starting_team"]).encode("utf-8")
    starting_identity = hashlib.sha256(start_json).hexdigest()
    source = _source_identity(root)
    bank_hashes = {
        f"{bank.partition}:{bank.seed}": bank.sha256
        for seed_banks in banks for bank in seed_banks
    }
    manifest = {
        **source,
        "benchmark_version": BENCHMARK_VERSION,
        "evidence_status": "exploratory_not_preregistered",
        "starting_team_identity_sha256": starting_identity,
        "frozen_config": config,
        "challenge_hashes": bank_hashes,
    }
    if len(strict_json_dumps(manifest).encode("utf-8")) > 64 * 1024:
        raise ValueError("manifest exceeds the 64 KiB bound")
    return manifest


def _ensure_budget(started, wall_seconds, memory_bytes):
    if time.monotonic() - started >= wall_seconds:
        raise BudgetExceeded("timeout")
    current, peak = tracemalloc.get_traced_memory()
    if max(current, peak) > memory_bytes:
        raise BudgetExceeded("memory_limit")


def _model_runner(team, episode_index, stimulus, state, streams):
    return run_real_model_episode(
        team, episode_index, stimulus, state, streams,
        width=WIDTH, steps=STEPS, shield=SHIELD_ENABLED,
    )


def _seed_run(seed, development, validation, started, wall_seconds, memory_bytes):
    objective = make_benchmark_objective()
    teams = []
    candidate_summaries = []
    proposal_evaluation_operations = []
    proposal_operations = 0
    comparison_operations = 0
    progress = {"proposal_attempts": 0, "search_operations": 0}
    proposal_rng = _proposal_rng(seed)
    state = deepcopy(INITIAL_STATE)

    try:
        for attempt in range(1, MAX_PROPOSALS + 1):
            _ensure_budget(started, wall_seconds, memory_bytes)
            candidate = _proposal(proposal_rng)
            proposal_operations += 9
            result = paired_team_evaluation(
                INITIAL_TEAM,
                candidate,
                development,
                _model_runner,
                objective,
                state,
                objective_name="illustrative_target_direction_utility",
                candidate_proposal_attempts=attempt,
                candidate_proposal_operations=proposal_operations,
                incumbent_search_attempts=0,
                incumbent_search_operations=0,
                summarizer=summarize_target_directions,
            )
            teams.append(candidate)
            candidate_summaries.append(result.candidate)
            proposal_evaluation_operations.append(result.candidate["compute_operations"])
            comparison_operations += result.incumbent["compute_operations"]
            progress["proposal_attempts"] = attempt
            progress["search_operations"] = (
                proposal_operations + sum(proposal_evaluation_operations)
            )
            _ensure_budget(started, wall_seconds, memory_bytes)
    except BudgetExceeded as exc:
        exc.details = {
            **progress,
            "proposal_statuses": (
                ["evaluated"] * progress["proposal_attempts"]
                + ["not_started"] * (MAX_PROPOSALS - progress["proposal_attempts"])
            ),
        }
        raise

    total_search_operations = proposal_operations + sum(proposal_evaluation_operations)
    total_attempts = MAX_PROPOSALS
    final_search_cost = ArmCostAttribution(
        total_attempts, total_search_operations, 0
    )
    final_context = _selection_context(final_search_cost)
    training_scores = [
        _utility_from_summary(summary, development._targets, final_context)
        for summary in candidate_summaries
    ]
    selected_index = max(
        range(len(training_scores)), key=lambda index: (training_scores[index], -index)
    )
    selected_team = deepcopy(teams[selected_index])

    _ensure_budget(started, wall_seconds, memory_bytes)
    baseline_validation = report_frozen_team(
        INITIAL_TEAM,
        validation,
        _model_runner,
        objective,
        state,
        objective_name="illustrative_target_direction_utility",
        summarizer=summarize_target_directions,
    )
    _ensure_budget(started, wall_seconds, memory_bytes)
    searched_validation = report_frozen_team(
        selected_team,
        validation,
        _model_runner,
        objective,
        state,
        objective_name="illustrative_target_direction_utility",
        search_attempts=total_attempts,
        search_operations=total_search_operations,
        summarizer=summarize_target_directions,
    )
    _ensure_budget(started, wall_seconds, memory_bytes)
    baseline_report = baseline_validation.report()
    searched_report = searched_validation.report()
    baseline_report["costs"]["no_intervention"] = baseline_report["costs"].pop(
        "frozen_team"
    )
    searched_report["costs"]["random_bounded_team_proposals"] = (
        searched_report["costs"].pop("frozen_team")
    )
    baseline_report["paired_difference_vs_no_intervention"] = 0.0
    searched_report["paired_difference_vs_no_intervention"] = (
        searched_report["score"] - baseline_report["score"]
    )
    searched_report["selected_proposal_index_development_only"] = selected_index
    searched_report["training_utility_by_proposal"] = training_scores
    searched_report["search_budget"] = {
        "proposal_attempts": total_attempts,
        "proposal_operations": proposal_operations,
        "proposal_evaluation_operations": sum(proposal_evaluation_operations),
        "total_search_operations": total_search_operations,
        "common_baseline_comparison_operations": comparison_operations,
        "proposal_statuses": ["evaluated"] * total_attempts,
    }
    searched_report["selected_team_identity_sha256"] = hashlib.sha256(
        strict_json_dumps(selected_team).encode("utf-8")
    ).hexdigest()
    replay_payload = {
        "seed": seed,
        "development_sha256": development.sha256,
        "validation_sha256": validation.sha256,
        "no_intervention": _without_timing(baseline_report),
        "random_bounded_team_proposals": _without_timing(searched_report),
    }
    replay_hash = hashlib.sha256(
        strict_json_dumps(replay_payload).encode("utf-8")
    ).hexdigest()
    baseline_report["deterministic_replay_sha256"] = replay_hash
    searched_report["deterministic_replay_sha256"] = replay_hash
    return {
        "seed": seed,
        "status": "completed",
        "development_challenge_sha256": development.sha256,
        "validation_challenge_sha256": validation.sha256,
        "no_intervention": baseline_report,
        "random_bounded_team_proposals": searched_report,
        "independent_candidate_search": {"status": "NOT_IMPLEMENTED"},
        "shared_candidate_search": {"status": "NOT_IMPLEMENTED"},
    }


def _selection_context(costs):
    from .evaluation_v2 import ScoringContext
    return ScoringContext(
        arm="candidate",
        incumbent=ArmCostAttribution(0, 0, 0),
        candidate=costs,
        common_validation_operations=0,
    )


def _utility_from_summary(summary, targets, context):
    costs = context.costs_for_arm()
    detection = summary["target_direction_detection"]
    return utility_score(
        len(targets),
        target_direction_tp=detection["TP"],
        target_direction_fp=detection["FP"],
        target_direction_fn=detection["FN"],
        target_direction_tn=detection["TN"],
        **UTILITY_COEFFICIENTS,
        compute_operations=costs.compute_operations,
        proposal_attempts=costs.search_attempts,
        incorrect_direction_declarations=summary["raw_wrong_directions"],
        raw_declarations=summary["raw_declarations"],
        raw_true_declarations=summary["raw_correct_directions"],
        raw_false_declarations=summary["raw_wrong_directions"],
    )["total_utility"]


def _without_timing(report):
    return {
        key: value for key, value in report.items()
        if key not in ("wall_seconds", "cpu_seconds")
    } | {
        "costs": {
            key: value for key, value in report["costs"].items()
            if key not in ("wall_seconds", "cpu_seconds")
        }
    }


def _seed_banks(seeds, width, train_episodes, validation_episodes, steps):
    banks = []
    for seed in range(seeds):
        train_seed = DEVELOPMENT_SEED_NAMESPACE + seed
        validation_seed = VALIDATION_SEED_NAMESPACE + seed
        banks.append((
            make_challenge_bank(
                train_seed, "development", width=width,
                episodes=train_episodes, steps=steps,
            ),
            make_challenge_bank(
                validation_seed, "calibration", width=width,
                episodes=validation_episodes, steps=steps,
            ),
        ))
    return banks


def run_development_benchmark(output_dir, *, seeds=DEFAULT_SEEDS,
                              wall_seconds=MAX_WALL_SECONDS,
                              memory_bytes=MAX_TRACED_BYTES,
                              dry_run=False, width=WIDTH,
                              train_episodes=TRAIN_EPISODES,
                              validation_episodes=VALIDATION_EPISODES,
                              steps=STEPS, proposals=MAX_PROPOSALS,
                              root=None):
    """Run a bounded benchmark or write only its frozen dry-run plan."""
    if not 1 <= seeds <= MAX_SEEDS:
        raise ValueError(f"seeds must be in [1, {MAX_SEEDS}]")
    if width != WIDTH:
        raise ValueError(f"this benchmark is frozen to width {WIDTH}")
    if train_episodes != TRAIN_EPISODES or validation_episodes != VALIDATION_EPISODES:
        raise ValueError("episode counts are frozen at eight per partition")
    if steps != STEPS or proposals != MAX_PROPOSALS:
        raise ValueError("step and proposal limits are frozen by the benchmark config")
    if not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError(f"wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if not 0 < memory_bytes <= MAX_TRACED_BYTES:
        raise ValueError(f"memory_bytes must be in (0, {MAX_TRACED_BYTES}]")
    root = root or os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    started = time.monotonic()
    owns_tracemalloc = not tracemalloc.is_tracing()
    if owns_tracemalloc:
        tracemalloc.start()
    rows = []
    status = "planned" if dry_run else "running"
    try:
        banks = _seed_banks(
            seeds, width, train_episodes, validation_episodes, steps
        )
        manifest = build_manifest(seeds, banks, root)
        manifest["budget_override_seconds"] = wall_seconds
        manifest["budget_override_python_traced_bytes"] = memory_bytes
        manifest["dry_run"] = bool(dry_run)
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            f.write(strict_json_dumps(manifest) + "\n")
        if dry_run:
            plan = {
                "status": "planned",
                "manifest_sha256": hashlib.sha256(
                    strict_json_dumps(manifest).encode("utf-8")
                ).hexdigest(),
                "registered_arms": deepcopy(REGISTERED_ARMS),
                "seed_count": seeds,
                "challenge_hashes": manifest["challenge_hashes"],
                "notice": "No model evaluation or proposal selection was executed.",
            }
            with open(os.path.join(output_dir, "plan.json"), "w", encoding="utf-8") as f:
                f.write(strict_json_dumps(plan) + "\n")
            return plan

        status = "completed"
        for seed, (development, validation) in enumerate(banks):
            try:
                _ensure_budget(started, wall_seconds, memory_bytes)
                row = _seed_run(
                    seed, development, validation, started, wall_seconds, memory_bytes
                )
            except BudgetExceeded as exc:
                details = exc.details or {
                    "proposal_attempts": 0,
                    "search_operations": 0,
                    "proposal_statuses": ["not_started"] * MAX_PROPOSALS,
                }
                row = {
                    "seed": seed,
                    "status": exc.status,
                    "development_challenge_sha256": development.sha256,
                    "validation_challenge_sha256": validation.sha256,
                    "no_intervention": {"status": exc.status},
                    "random_bounded_team_proposals": {
                        "status": exc.status,
                        **details,
                    },
                    "independent_candidate_search": {"status": "NOT_IMPLEMENTED"},
                    "shared_candidate_search": {"status": "NOT_IMPLEMENTED"},
                }
                status = exc.status
            except Exception as exc:
                row = {
                    "seed": seed,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "development_challenge_sha256": development.sha256,
                    "validation_challenge_sha256": validation.sha256,
                    "no_intervention": {"status": "failed"},
                    "random_bounded_team_proposals": {"status": "failed"},
                    "independent_candidate_search": {"status": "NOT_IMPLEMENTED"},
                    "shared_candidate_search": {"status": "NOT_IMPLEMENTED"},
                }
                status = "failed"
            rows.append(row)
            _write_partial_outputs(output_dir, rows, status)
            if status != "completed":
                for remaining in range(seed + 1, seeds):
                    dev_bank, val_bank = banks[remaining]
                    rows.append({
                        "seed": remaining,
                        "status": "not_started_budget_stop",
                        "development_challenge_sha256": dev_bank.sha256,
                        "validation_challenge_sha256": val_bank.sha256,
                        "no_intervention": {"status": "not_started_budget_stop"},
                        "random_bounded_team_proposals": {
                            "status": "not_started_budget_stop"
                        },
                        "independent_candidate_search": {
                            "status": "NOT_IMPLEMENTED"
                        },
                        "shared_candidate_search": {
                            "status": "NOT_IMPLEMENTED"
                        },
                    })
                break
        summary = _summary(rows)
        result = {
            "benchmark_version": BENCHMARK_VERSION,
            "status": status,
            "source": manifest,
            "registered_arms": deepcopy(REGISTERED_ARMS),
            "per_seed": rows,
            "descriptive_summary": summary,
            "budget": {
                "wall_seconds_limit": wall_seconds,
                "wall_seconds_used": time.monotonic() - started,
                "python_traced_memory_limit_bytes": memory_bytes,
                "python_traced_memory_peak_bytes": (
                    tracemalloc.get_traced_memory()[1] if tracemalloc.is_tracing() else None
                ),
                "memory_measurement_status": "tracemalloc_not_process_rss",
            },
            "evidence_note": (
                "Exploratory synthetic operability smoke only; no significance or advantage claim."
            ),
        }
        _write_partial_outputs(output_dir, rows, status)
        _write_final_outputs(output_dir, result)
        return result
    finally:
        if owns_tracemalloc:
            tracemalloc.stop()


def _summary(rows):
    differences = [
        row["random_bounded_team_proposals"]["paired_difference_vs_no_intervention"]
        for row in rows
        if row.get("status") == "completed"
        and "paired_difference_vs_no_intervention"
        in row["random_bounded_team_proposals"]
    ]
    mean = sum(differences) / len(differences) if differences else None
    sample_sd = None
    if len(differences) > 1:
        sample_sd = math.sqrt(
            sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        )
    failure_seeds = [row["seed"] for row in rows if row.get("status") == "failed"]
    timeout_seeds = [row["seed"] for row in rows if row.get("status") == "timeout"]
    return {
        "paired_utility_differences": differences,
        "mean_paired_difference": mean,
        "sample_standard_deviation": sample_sd,
        "status": "descriptive_only_not_powered",
        "completed_seeds": len(differences),
        "failure_seeds": failure_seeds,
        "timeout_seeds": timeout_seeds,
        "memory_limit_seeds": [
            row["seed"] for row in rows if row.get("status") == "memory_limit"
        ],
        "excluded_or_incomplete_seeds": [
            row["seed"] for row in rows if row.get("status") != "completed"
        ],
    }


def _flat_rows(rows):
    flat = []
    for seed_row in rows:
        for arm_name in (
            "no_intervention",
            "random_bounded_team_proposals",
            "independent_candidate_search",
            "shared_candidate_search",
        ):
            arm = seed_row.get(arm_name, {})
            counts = arm.get("counts", {})
            detection = counts.get("target_direction_detection", {})
            costs = arm.get("costs", {})
            arm_costs = costs.get(arm_name, costs)
            flat.append({
                "seed": seed_row["seed"],
                "seed_status": seed_row.get("status"),
                "arm": arm_name,
                "arm_status": arm.get("status", "completed"),
                "score": arm.get("score"),
                "paired_difference_vs_no_intervention": arm.get(
                    "paired_difference_vs_no_intervention"
                ),
                "TP": detection.get("TP"),
                "FP": detection.get("FP"),
                "FN": detection.get("FN"),
                "TN": detection.get("TN"),
                "raw_declarations": counts.get("raw_declarations"),
                "emitted_declarations_post_filter": counts.get(
                    "emitted_declarations_post_filter"
                ),
                "emitted_correct_directions": counts.get("emitted_correct_directions"),
                "emitted_wrong_directions": counts.get("emitted_wrong_directions"),
                "false_acceptance_target_absent_episodes": counts.get(
                    "false_acceptance_target_absent_episodes"
                ),
                "correct_emitted_fraction": (
                    counts.get("correct_emitted_fraction", {}).get("value")
                ),
                "search_attempts_including_failed": arm_costs.get(
                    "search_attempts_including_failed"
                ),
                "search_operations": arm_costs.get("search_operations"),
                "evaluation_operations": arm_costs.get("evaluation_operations"),
                "wall_seconds": costs.get("wall_seconds"),
                "cpu_seconds": costs.get("cpu_seconds"),
                "replay_sha256": arm.get("deterministic_replay_sha256"),
            })
    return flat


def _write_partial_outputs(output_dir, rows, status):
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        f.write(strict_json_dumps({
            "status": status,
            "per_seed": rows,
            "registered_arms": REGISTERED_ARMS,
        }) + "\n")
    _write_csv(os.path.join(output_dir, "results.csv"), _flat_rows(rows))


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else [
        "seed", "seed_status", "arm", "arm_status", "score"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_final_outputs(output_dir, result):
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        f.write(strict_json_dumps(result) + "\n")
    _write_csv(os.path.join(output_dir, "results.csv"), _flat_rows(result["per_seed"]))
    lines = [
        "# Local development benchmark v2",
        "",
        f"- Status: `{result['status']}`",
        f"- Source revision: `{result['source']['revision']}`",
        f"- Evidence: {result['evidence_note']}",
        f"- Completed seeds: {result['descriptive_summary']['completed_seeds']}",
        f"- Failed/timed-out seeds: {result['descriptive_summary']['failure_seeds']} / {result['descriptive_summary']['timeout_seeds']}",
        f"- Memory-limited/excluded seeds: {result['descriptive_summary']['memory_limit_seeds']} / {result['descriptive_summary']['excluded_or_incomplete_seeds']}",
        f"- Paired utility differences: `{result['descriptive_summary']['paired_utility_differences']}`",
        f"- Mean / sample SD (descriptive): `{result['descriptive_summary']['mean_paired_difference']}` / `{result['descriptive_summary']['sample_standard_deviation']}`",
        "- Coefficients are illustrative study assumptions, not optimal values.",
        "- Independent and shared proposal search are `NOT_IMPLEMENTED`.",
        "- No held-out banks were generated or executed.",
        "- No matched-budget claim or scientific superiority claim is made.",
        "",
        "| seed | arm | status | utility | paired Δ vs fixed | TP | FP | FN | TN | search attempts | search ops | eval ops | replay SHA256 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in _flat_rows(result["per_seed"]):
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['arm_status']} | "
            f"{row['score']} | {row['paired_difference_vs_no_intervention']} | "
            f"{row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | "
            f"{row['search_attempts_including_failed']} | {row['search_operations']} | "
            f"{row['evaluation_operations']} | {row['replay_sha256']} |"
        )
    with open(os.path.join(output_dir, "EVIDENCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
