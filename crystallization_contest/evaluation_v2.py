"""Opt-in scoring and paired-evaluation primitives (synthetic protocol v2 core).

This module does not change the legacy experiment or implement proposal
generation. Evaluators receive pre-materialized stimuli and indexed random
streams, never challenge labels. The labels are used only after evaluation to
score returned directions.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import time


PROTOCOL_VERSION = "v2-core"
PARTITIONS = ("development", "calibration", "held-out")
TIE_POLICY = "incumbent"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("w must be finite and in [0, 1]")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("w must be finite and in [0, 1]") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("w must be finite and in [0, 1]")
    return value


def _score_inputs(episodes, w, joint_true, true_counts, false_counts, agent, fcost):
    episodes = _integer("episodes", episodes, 1)
    w = _weight(w)
    joint_true = _integer("joint_true", joint_true)
    fcost = _finite_nonnegative("fcost", fcost)
    try:
        true_counts = tuple(true_counts)
        false_counts = tuple(false_counts)
    except TypeError as exc:
        raise ValueError("true_counts and false_counts must be count vectors") from exc
    if not true_counts or len(true_counts) != len(false_counts):
        raise ValueError("true_counts and false_counts must have equal, positive length")
    agent = _integer("agent", agent)
    if agent >= len(true_counts):
        raise ValueError("agent index is outside the count vectors")
    for i, count in enumerate(true_counts):
        _integer(f"true_counts[{i}]", count)
        if count > episodes:
            raise ValueError(f"true_counts[{i}] exceeds episodes")
    for i, count in enumerate(false_counts):
        _integer(f"false_counts[{i}]", count)
        if count > episodes:
            raise ValueError(f"false_counts[{i}] exceeds episodes")
        if true_counts[i] + count > episodes:
            raise ValueError(f"true_counts[{i}] + false_counts[{i}] exceeds episodes")
    if joint_true > episodes or any(joint_true > count for count in true_counts):
        raise ValueError("joint_true must be bounded by episodes and every true count")
    return episodes, w, joint_true, true_counts, false_counts, agent, fcost


def _cooperative_breakdown(version, episodes, w, joint_true, true_counts,
                           false_counts, agent, fcost, normalized):
    max_true = max(true_counts)
    leaders = tuple(i for i, count in enumerate(true_counts) if count == max_true)
    rank_share = 1.0 / len(leaders) if max_true > 0 and agent in leaders else 0.0
    try:
        joint_value = joint_true / episodes if normalized else joint_true
        joint_contribution = w * joint_value
        rank_contribution = (1.0 - w) * rank_share
        penalty = -fcost * false_counts[agent] / episodes
        score = joint_contribution + rank_contribution + penalty
    except OverflowError as exc:
        raise ValueError("score is not finite") from exc
    if not math.isfinite(score):
        raise ValueError("score is not finite")
    return {
        "version": version,
        "agent": agent,
        "score": score,
        "contributions": {
            "joint_true": joint_contribution,
            "rank_bonus": rank_contribution,
            "false_event_penalty": penalty,
        },
        "denominators": {
            "joint_true": episodes if normalized else 1,
            "rank_bonus_leaders": len(leaders) if max_true > 0 else None,
            "false_events": episodes,
        },
        "counts": {
            "episodes": episodes,
            "joint_true_episodes": joint_true,
            "true_events_by_agent": list(true_counts),
            "raw_false_declarations_by_agent": list(false_counts),
        },
        "parameters": {
            "w": w,
            "fcost": fcost,
            "rank_leaders": list(leaders) if max_true > 0 else [],
            "rank_bonus_zero_reason": "all_true_counts_zero" if max_true == 0 else None,
        },
    }


def legacy_cooperative_score(episodes, w, joint_true, true_counts, false_counts,
                             agent, fcost=0.0):
    """Return a contribution breakdown identical to contest.py's historical score.

    The historical joint term is an episode count, not a rate. Consequently
    ``w=0.5`` does not give the joint and rank terms equal numerical scale.
    """
    values = _score_inputs(
        episodes, w, joint_true, true_counts, false_counts, agent, fcost
    )
    return _cooperative_breakdown(
        "legacy-contest-v1", *values, normalized=False
    )


def normalized_cooperative_score(episodes, w, joint_true, true_counts,
                                 false_counts, agent, fcost=0.0):
    """Score S_i = w*(J/E) + (1-w)*B_i - fcost*(F_i/E)."""
    values = _score_inputs(
        episodes, w, joint_true, true_counts, false_counts, agent, fcost
    )
    return _cooperative_breakdown(
        "normalized-cooperative-v2", *values, normalized=True
    )


def _metric(numerator, denominator):
    if denominator == 0:
        return {"value": None, "status": "undefined_zero_denominator"}
    return {"value": numerator / denominator, "status": "defined"}


def utility_score(episodes, *, tp, fp, fn, tn, b_tp, c_fp, c_fn,
                  c_compute, c_attempt, compute_operations, proposal_attempts,
                  incorrect_direction_declarations=0, raw_declarations=0,
                  raw_true_declarations=0, raw_false_declarations=0,
                  wall_seconds=None, cpu_seconds=None):
    """Evaluate target-detection utility with explicit costs and operation units.

    TP/FP/FN/TN are episode-level target-detection outcomes. A wrong-direction
    declaration on a target-present episode is an FN and is counted separately;
    no direction penalty is silently added. Raw membrane declarations are kept
    in separate fields. ``compute_operations`` must include all proposal,
    search, and evaluation work charged to this result; do not also price the
    same work through ``c_attempt`` unless intentional. Validation-only overhead
    should be reported separately and excluded from deployable-search cost.
    """
    episodes = _integer("episodes", episodes, 1)
    counts = {
        "TP": _integer("tp", tp),
        "FP": _integer("fp", fp),
        "FN": _integer("fn", fn),
        "TN": _integer("tn", tn),
    }
    if sum(counts.values()) != episodes:
        raise ValueError("TP + FP + FN + TN must equal episodes")
    coefficients = {
        "b_tp": _finite_nonnegative("b_tp", b_tp),
        "c_fp": _finite_nonnegative("c_fp", c_fp),
        "c_fn": _finite_nonnegative("c_fn", c_fn),
        "c_compute": _finite_nonnegative("c_compute", c_compute),
        "c_attempt": _finite_nonnegative("c_attempt", c_attempt),
    }
    compute_operations = _integer("compute_operations", compute_operations)
    proposal_attempts = _integer("proposal_attempts", proposal_attempts)
    incorrect_direction_declarations = _integer(
        "incorrect_direction_declarations", incorrect_direction_declarations
    )
    raw_declarations = _integer("raw_declarations", raw_declarations)
    raw_true_declarations = _integer("raw_true_declarations", raw_true_declarations)
    raw_false_declarations = _integer("raw_false_declarations", raw_false_declarations)
    if raw_true_declarations + raw_false_declarations != raw_declarations:
        raise ValueError("raw true and false declarations must sum to raw declarations")
    if incorrect_direction_declarations > raw_false_declarations:
        raise ValueError("incorrect-direction declarations exceed raw false declarations")
    for name, value in (("wall_seconds", wall_seconds), ("cpu_seconds", cpu_seconds)):
        if value is not None:
            _finite_nonnegative(name, value)

    try:
        components = {
            "true_positive_reward": coefficients["b_tp"] * counts["TP"],
            "false_positive_penalty": -coefficients["c_fp"] * counts["FP"],
            "false_negative_penalty": -coefficients["c_fn"] * counts["FN"],
            "compute_penalty": -coefficients["c_compute"] * compute_operations,
            "attempt_penalty": -coefficients["c_attempt"] * proposal_attempts,
        }
        total = sum(components.values())
    except OverflowError as exc:
        raise ValueError("utility is not finite") from exc
    if not math.isfinite(total):
        raise ValueError("utility is not finite")
    return {
        "version": "target-utility-v2",
        "total_utility": total,
        "per_episode_utility": total / episodes,
        "per_episode_denominator": episodes,
        "raw_counts": {
            **counts,
            "incorrect_direction_declarations": incorrect_direction_declarations,
            "raw_declarations": raw_declarations,
            "raw_true_declarations": raw_true_declarations,
            "raw_false_declarations": raw_false_declarations,
            "compute_operations": compute_operations,
            "proposal_attempts": proposal_attempts,
        },
        "coefficients": coefficients,
        "units": {
            "b_tp": "utility per true-positive target-detection episode",
            "c_fp": "utility per false-positive target-detection episode",
            "c_fn": "utility per false-negative target-detection episode",
            "c_compute": "utility per deterministic operation",
            "c_attempt": "utility per proposal attempt",
            "wall_seconds": "seconds, reported separately and not charged",
            "cpu_seconds": "seconds, reported separately and not charged",
        },
        "component_totals": components,
        "metrics": {
            "precision": _metric(counts["TP"], counts["TP"] + counts["FP"]),
            "recall": _metric(counts["TP"], counts["TP"] + counts["FN"]),
            "specificity": _metric(counts["TN"], counts["TN"] + counts["FP"]),
        },
        "timing": {"wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds},
    }


def strict_json_dumps(value):
    """Serialize strict JSON; non-finite floats raise instead of becoming NaN."""
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(value[key]) for key in sorted(value)}
    raise ValueError("value is not bounded JSON data")


def _freeze(value):
    value = _json_value(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    return value


@dataclass(frozen=True)
class IndexedStreams:
    """Indexed random channels without any reference to challenge stimuli/labels."""
    seed: int
    partition: str

    episodes: int

    def __post_init__(self):
        _integer("seed", self.seed)
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        _integer("episodes", self.episodes, 1)

    def rng_for(self, episode, *, step=0, agent=0, channel="environment"):
        """Return a deterministic RNG isolated to one exogenous index/channel."""
        episode = _integer("episode", episode)
        step = _integer("step", step)
        agent = _integer("agent", agent)
        if episode >= self.episodes or not isinstance(channel, str) or not channel:
            raise ValueError("invalid indexed random-stream key")
        key = strict_json_dumps(
            [self.seed, self.partition, episode, step, agent, channel]
        ).encode("utf-8")
        return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


@dataclass(frozen=True)
class ChallengeBank:
    """Immutable exogenous stimuli plus labels hidden from the episode runner."""

    seed: int
    partition: str
    stimuli: tuple
    _targets: tuple = field(repr=False)
    sha256: str = field(init=False)

    def __post_init__(self):
        _integer("seed", self.seed)
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        stimuli = tuple(_freeze(item) for item in self.stimuli)
        targets = tuple(self._targets)
        if not stimuli or len(stimuli) != len(targets):
            raise ValueError("stimuli and targets must have equal positive length")
        if any(target is not None and (not isinstance(target, str) or not target)
               for target in targets):
            raise ValueError("targets must be nonempty direction IDs or None")
        payload = {
            "protocol": PROTOCOL_VERSION,
            "seed": self.seed,
            "partition": self.partition,
            "stimuli": _json_value(stimuli),
            "targets": list(targets),
        }
        digest = hashlib.sha256(strict_json_dumps(payload).encode("utf-8")).hexdigest()
        object.__setattr__(self, "stimuli", stimuli)
        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "sha256", digest)

    @property
    def episodes(self):
        return len(self.stimuli)

    @property
    def indexed_streams(self):
        return IndexedStreams(self.seed, self.partition, self.episodes)

    def rng_for(self, episode, *, step=0, agent=0, channel="environment"):
        """Return a deterministic RNG isolated to one exogenous index/channel."""
        return self.indexed_streams.rng_for(
            episode, step=step, agent=agent, channel=channel
        )


def protocol_manifest(bank, frozen_config):
    """Create a small frozen manifest; no evidence of preregistration is implied."""
    if not isinstance(bank, ChallengeBank):
        raise TypeError("bank must be a ChallengeBank")
    config = _json_value(frozen_config)
    config_text = strict_json_dumps(config)
    if len(config_text.encode("utf-8")) > 4096:
        raise ValueError("frozen configuration exceeds the 4096-byte manifest bound")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "seed_partition": bank.partition,
        "seed": bank.seed,
        "episodes": bank.episodes,
        "challenge_sha256": bank.sha256,
        "frozen_config": config,
        "configuration_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "evidence_status": "exploratory_unregistered",
    }
    if len(strict_json_dumps(manifest).encode("utf-8")) > 8192:
        raise ValueError("protocol manifest exceeds the 8192-byte bound")
    return manifest


@dataclass(frozen=True)
class EpisodeResult:
    """One final system direction and raw declarations from an episode."""

    declared_direction: str | None
    raw_directions: tuple
    operation_count: int

    def __post_init__(self):
        _integer("operation_count", self.operation_count)
        raw = tuple(self.raw_directions)
        if any(not isinstance(direction, str) or not direction for direction in raw):
            raise ValueError("raw directions must be nonempty strings")
        if self.declared_direction is not None:
            if not isinstance(self.declared_direction, str) or not self.declared_direction:
                raise ValueError("declared_direction must be a nonempty string or None")
            if self.declared_direction not in raw:
                raise ValueError("final direction must be one of the raw declarations")
        object.__setattr__(self, "raw_directions", raw)


def _arm_summary(results, targets):
    tp = fp = fn = tn = 0
    raw_true = raw_false = incorrect_direction = operations = 0
    for result, target in zip(results, targets):
        if target is None:
            if result.declared_direction is None:
                tn += 1
            else:
                fp += 1
            raw_false += len(result.raw_directions)
        else:
            if result.declared_direction == target:
                tp += 1
            else:
                fn += 1
            raw_true += sum(direction == target for direction in result.raw_directions)
            wrong = sum(direction != target for direction in result.raw_directions)
            raw_false += wrong
            incorrect_direction += wrong
        operations += result.operation_count
    return {
        "episodes": len(results),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "raw_declarations": raw_true + raw_false,
        "raw_true_declarations": raw_true,
        "raw_false_declarations": raw_false,
        "incorrect_direction_declarations": incorrect_direction,
        "compute_operations": operations,
    }


@dataclass(frozen=True)
class PairedEvaluation:
    """Full-team transaction result; evaluation state is intentionally discarded."""

    accepted_team: object = field(repr=False)
    accepted: bool
    objective_name: str
    incumbent_score: float
    candidate_score: float
    incumbent: dict
    candidate: dict
    challenge_sha256: str
    candidate_proposal_attempts: int
    candidate_proposal_operations: int
    common_validation_operations: int
    wall_seconds: float
    cpu_seconds: float

    def report(self):
        """Return JSON-safe diagnostics without the potentially arbitrary team."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "objective_name": self.objective_name,
            "tie_policy": TIE_POLICY,
            "accepted": self.accepted,
            "incumbent_score": self.incumbent_score,
            "candidate_score": self.candidate_score,
            "incumbent": self.incumbent,
            "candidate": self.candidate,
            "challenge_sha256": self.challenge_sha256,
            "costs": {
                "candidate_proposal_attempts": self.candidate_proposal_attempts,
                "candidate_proposal_operations": self.candidate_proposal_operations,
                "incumbent_evaluation_operations": self.incumbent["compute_operations"],
                "candidate_evaluation_operations": self.candidate["compute_operations"],
                "common_validation_operations": self.common_validation_operations,
                "wall_seconds": self.wall_seconds,
                "cpu_seconds": self.cpu_seconds,
            },
            "evaluation_state_policy": "independent deep-copy per arm; discard after comparison",
        }


def paired_team_evaluation(incumbent_team, candidate_team, bank, runner, objective,
                           initial_state, *, objective_name, candidate_proposal_attempts,
                           candidate_proposal_operations=0,
                           common_validation_operations=0):
    """Evaluate complete incumbent/candidate teams on the same immutable bank.

    ``runner(team_copy, stimulus, state_copy, rng_for)`` returns
    ``(EpisodeResult, next_state)``. It receives no target labels. Both arms start
    from separate copies of the same initial state; all returned evaluation state
    is discarded. The candidate team is accepted atomically only for a strictly
    higher finite scalar objective. Ties deterministically keep the incumbent.
    """
    if not isinstance(bank, ChallengeBank):
        raise TypeError("bank must be a ChallengeBank")
    if not callable(runner) or not callable(objective):
        raise TypeError("runner and objective must be callable")
    if not isinstance(objective_name, str) or not objective_name:
        raise ValueError("objective_name must be a nonempty string")
    candidate_proposal_attempts = _integer(
        "candidate_proposal_attempts", candidate_proposal_attempts, 1
    )
    candidate_proposal_operations = _integer(
        "candidate_proposal_operations", candidate_proposal_operations
    )
    common_validation_operations = _integer(
        "common_validation_operations", common_validation_operations
    )

    def evaluate(team):
        state = deepcopy(initial_state)
        results = []
        for index, stimulus in enumerate(bank.stimuli):
            result, next_state = runner(
                deepcopy(team), stimulus, deepcopy(state),
                bank.indexed_streams.rng_for,
            )
            if not isinstance(result, EpisodeResult):
                raise TypeError("runner must return an EpisodeResult for each episode")
            results.append(result)
            state = deepcopy(next_state)
        counts = _arm_summary(results, bank._targets)
        return counts, tuple(results)

    def score(results):
        value = objective(results, bank._targets)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("objective must return a finite scalar")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("objective must return a finite scalar") from exc
        if not math.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    incumbent_snapshot = deepcopy(incumbent_team)
    candidate_snapshot = deepcopy(candidate_team)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    incumbent_counts, incumbent_results = evaluate(incumbent_snapshot)
    candidate_counts, candidate_results = evaluate(candidate_snapshot)
    incumbent_score = score(incumbent_results)
    candidate_score = score(candidate_results)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    accepted = candidate_score > incumbent_score
    return PairedEvaluation(
        accepted_team=deepcopy(candidate_snapshot if accepted else incumbent_snapshot),
        accepted=accepted,
        objective_name=objective_name,
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        incumbent=incumbent_counts,
        candidate=candidate_counts,
        challenge_sha256=bank.sha256,
        candidate_proposal_attempts=candidate_proposal_attempts,
        candidate_proposal_operations=candidate_proposal_operations,
        common_validation_operations=common_validation_operations,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
    )


def worked_example():
    """Illustrative arithmetic only; values are not calibrated or optimal."""
    cooperative = normalized_cooperative_score(
        episodes=4,
        w=0.5,
        joint_true=1,
        true_counts=[2, 2, 1],
        false_counts=[1, 0, 1],
        agent=0,
        fcost=0.5,
    )
    utility = utility_score(
        10,
        tp=4,
        fp=1,
        fn=2,
        tn=3,
        b_tp=2.0,
        c_fp=1.0,
        c_fn=0.5,
        c_compute=0.01,
        c_attempt=0.2,
        compute_operations=100,
        proposal_attempts=2,
        raw_declarations=7,
        raw_true_declarations=4,
        raw_false_declarations=3,
        incorrect_direction_declarations=1,
    )
    return {
        "status": "illustrative_not_optimal",
        "normalized_cooperative": cooperative,
        "target_utility": utility,
    }
