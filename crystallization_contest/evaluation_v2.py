"""Opt-in scoring and paired-evaluation primitives (synthetic protocol v2 core).

This module does not change the legacy experiment or implement proposal
generation. Episode runners receive pre-materialized stimuli and indexed random
streams, never challenge labels. Scoring callbacks receive labels only after
the evaluation runs; Python callbacks and closures are not a security sandbox.
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
    joint_lower = max(0, sum(true_counts) - (len(true_counts) - 1) * episodes)
    joint_upper = min(true_counts)
    if not joint_lower <= joint_true <= joint_upper:
        raise ValueError(
            "joint_true must be consistent with the per-agent true counts "
            f"(expected {joint_lower} <= joint_true <= {joint_upper})"
        )
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


def utility_score(episodes, *, target_direction_tp, target_direction_fp,
                  target_direction_fn, target_direction_tn, b_tp, c_fp, c_fn,
                  c_compute, c_attempt, compute_operations, proposal_attempts,
                  incorrect_direction_declarations=0, raw_declarations=0,
                  raw_true_declarations=0, raw_false_declarations=0,
                  wall_seconds=None, cpu_seconds=None):
    """Evaluate target-detection utility with explicit costs and operation units.

    Counts are episode-level target-direction detection outcomes, not target-
    presence detection. A wrong-direction declaration on a target-present
    episode is an FN and is counted separately; no direction penalty is silently
    added. Raw membrane declarations are kept in separate fields.
    ``compute_operations`` must include all proposal,
    search, and evaluation work charged to this result; do not also price the
    same work through ``c_attempt`` unless intentional. Validation-only overhead
    should be reported separately and excluded from deployable-search cost.
    """
    episodes = _integer("episodes", episodes, 1)
    counts = {
        "TP": _integer("target_direction_tp", target_direction_tp),
        "FP": _integer("target_direction_fp", target_direction_fp),
        "FN": _integer("target_direction_fn", target_direction_fn),
        "TN": _integer("target_direction_tn", target_direction_tn),
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
        "endpoint": "episode_level_target_direction_detection",
        "raw_counts": {
            "target_direction_detection": counts,
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
            "target_direction_precision": _metric(
                counts["TP"], counts["TP"] + counts["FP"]
            ),
            "target_direction_recall": _metric(
                counts["TP"], counts["TP"] + counts["FN"]
            ),
            "target_direction_specificity": _metric(
                counts["TN"], counts["TN"] + counts["FP"]
            ),
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


@dataclass(frozen=True)
class ArmCostAttribution:
    """Deployable proposal/search cost and evaluation cost for one team arm."""

    search_attempts: int
    search_operations: int
    evaluation_operations: int

    def __post_init__(self):
        _integer("search_attempts", self.search_attempts)
        _integer("search_operations", self.search_operations)
        _integer("evaluation_operations", self.evaluation_operations)

    @property
    def compute_operations(self):
        return self.search_operations + self.evaluation_operations


@dataclass(frozen=True)
class ScoringContext:
    """Immutable per-comparison cost attribution passed to each objective call.

    Search attempts must include every failed proposal. Search operations
    include proposal generation/search work; evaluation operations come from
    EpisodeResult. Common validation overhead is reported separately and is
    excluded from the deployable arm costs.
    """

    arm: str
    incumbent: ArmCostAttribution | None
    candidate: ArmCostAttribution | None
    common_validation_operations: int
    frozen_team: ArmCostAttribution | None = None

    def __post_init__(self):
        if self.arm not in ("incumbent", "candidate", "frozen"):
            raise ValueError("arm must be incumbent, candidate, or frozen")
        for name in ("incumbent", "candidate", "frozen_team"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ArmCostAttribution):
                raise TypeError(f"{name} must be ArmCostAttribution or None")
        _integer("common_validation_operations", self.common_validation_operations)

    def costs_for_arm(self):
        if self.arm == "frozen":
            return self.frozen_team
        return getattr(self, self.arm)


def _cost_report(costs):
    return {
        "search_attempts_including_failed": costs.search_attempts,
        "search_operations": costs.search_operations,
        "evaluation_operations": costs.evaluation_operations,
        "deployable_compute_operations": costs.compute_operations,
    }


def make_cost_aware_utility_objective(*, b_tp, c_fp, c_fn, c_compute, c_attempt):
    """Build a scalar objective that charges each arm's attributed full cost."""
    coefficients = {
        "b_tp": _finite_nonnegative("b_tp", b_tp),
        "c_fp": _finite_nonnegative("c_fp", c_fp),
        "c_fn": _finite_nonnegative("c_fn", c_fn),
        "c_compute": _finite_nonnegative("c_compute", c_compute),
        "c_attempt": _finite_nonnegative("c_attempt", c_attempt),
    }

    def objective(results, labels, scoring_context):
        if not isinstance(scoring_context, ScoringContext):
            raise TypeError("objective requires a ScoringContext")
        costs = scoring_context.costs_for_arm()
        if costs is None:
            raise ValueError("scoring context has no cost attribution for this arm")
        counts = _arm_summary(results, labels)
        return utility_score(
            len(labels),
            target_direction_tp=counts["target_direction_detection"]["TP"],
            target_direction_fp=counts["target_direction_detection"]["FP"],
            target_direction_fn=counts["target_direction_detection"]["FN"],
            target_direction_tn=counts["target_direction_detection"]["TN"],
            b_tp=coefficients["b_tp"],
            c_fp=coefficients["c_fp"],
            c_fn=coefficients["c_fn"],
            c_compute=coefficients["c_compute"],
            c_attempt=coefficients["c_attempt"],
            compute_operations=costs.compute_operations,
            proposal_attempts=costs.search_attempts,
            incorrect_direction_declarations=counts[
                "incorrect_direction_declarations"
            ],
            raw_declarations=counts["raw_declarations"],
            raw_true_declarations=counts["raw_true_declarations"],
            raw_false_declarations=counts["raw_false_declarations"],
        )["total_utility"]

    return objective


@dataclass(frozen=True)
class EpisodeRandomStreams:
    """Random streams bound to one episode index, not inferred from stimuli."""

    indexed_streams: IndexedStreams = field(repr=False)
    episode_index: int

    def __post_init__(self):
        _integer("episode_index", self.episode_index)
        if self.episode_index >= self.indexed_streams.episodes:
            raise ValueError("episode_index is outside the stream bank")

    def rng_for(self, *, step=0, agent=0, channel="environment"):
        return self.indexed_streams.rng_for(
            self.episode_index, step=step, agent=agent, channel=channel
        )


def _arm_summary(results, targets):
    direction_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    raw_true = raw_false = incorrect_direction = operations = 0
    emitted = 0
    for result, target in zip(results, targets):
        emitted += result.declared_direction is not None
        if target is None:
            if result.declared_direction is None:
                direction_counts["TN"] += 1
            else:
                direction_counts["FP"] += 1
            raw_false += len(result.raw_directions)
        else:
            if result.declared_direction == target:
                direction_counts["TP"] += 1
            else:
                direction_counts["FN"] += 1
            raw_true += sum(direction == target for direction in result.raw_directions)
            wrong = sum(direction != target for direction in result.raw_directions)
            raw_false += wrong
            incorrect_direction += wrong
        operations += result.operation_count
    return {
        "episodes": len(results),
        "endpoint": "episode_level_target_direction_detection",
        "target_direction_detection": direction_counts,
        "raw_declarations": raw_true + raw_false,
        "raw_true_declarations": raw_true,
        "raw_false_declarations": raw_false,
        "incorrect_direction_declarations": incorrect_direction,
        "emitted_declarations": emitted,
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
    incumbent_costs: ArmCostAttribution
    candidate_costs: ArmCostAttribution
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
                "incumbent": _cost_report(self.incumbent_costs),
                "candidate": _cost_report(self.candidate_costs),
                "common_validation_operations": self.common_validation_operations,
                "wall_seconds": self.wall_seconds,
                "cpu_seconds": self.cpu_seconds,
            },
            "evaluation_state_policy": "independent deep-copy per arm; discard after comparison",
        }


@dataclass(frozen=True)
class FrozenTeamReport:
    """Held-out metrics for a copied frozen team; contains no promotion result."""

    objective_name: str
    score: float
    counts: dict
    challenge_sha256: str
    costs: ArmCostAttribution
    common_validation_operations: int
    wall_seconds: float
    cpu_seconds: float

    def report(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "report_type": "frozen_team_held_out",
            "objective_name": self.objective_name,
            "score": self.score,
            "counts": self.counts,
            "challenge_sha256": self.challenge_sha256,
            "costs": {
                "frozen_team": _cost_report(self.costs),
                "common_validation_operations": self.common_validation_operations,
                "wall_seconds": self.wall_seconds,
                "cpu_seconds": self.cpu_seconds,
            },
            "selection": "not_performed",
        }


def report_frozen_team(team, bank, runner, objective, initial_state, *,
                       objective_name, common_validation_operations=0):
    """Evaluate one frozen team on held-out data for reporting only.

    No team or promotion decision is returned. The original team and initial
    state are copied, and the runner receives no labels. Python callbacks and
    closures are not a security sandbox; use trusted runner/objective code and
    do not expose this report to proposal-generation or online-update code.
    """
    if not isinstance(bank, ChallengeBank):
        raise TypeError("bank must be a ChallengeBank")
    if bank.partition != "held-out":
        raise ValueError("report_frozen_team requires a held-out bank")
    if not callable(runner) or not callable(objective):
        raise TypeError("runner and objective must be callable")
    if not isinstance(objective_name, str) or not objective_name:
        raise ValueError("objective_name must be a nonempty string")
    common_validation_operations = _integer(
        "common_validation_operations", common_validation_operations
    )
    team_snapshot = deepcopy(team)
    state = deepcopy(initial_state)
    results = []
    streams = bank.indexed_streams
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for index, stimulus in enumerate(bank.stimuli):
        result, next_state = runner(
            deepcopy(team_snapshot), index, stimulus, deepcopy(state),
            EpisodeRandomStreams(streams, index),
        )
        if not isinstance(result, EpisodeResult):
            raise TypeError("runner must return an EpisodeResult for each episode")
        results.append(result)
        state = deepcopy(next_state)
    counts = _arm_summary(results, bank._targets)
    costs = ArmCostAttribution(0, 0, counts["compute_operations"])
    context = ScoringContext(
        arm="frozen",
        incumbent=None,
        candidate=None,
        common_validation_operations=common_validation_operations,
        frozen_team=costs,
    )
    value = objective(tuple(results), bank._targets, context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("objective must return a finite scalar")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("objective must return a finite scalar") from exc
    if not math.isfinite(value):
        raise ValueError("objective must return a finite scalar")
    return FrozenTeamReport(
        objective_name=objective_name,
        score=value,
        counts=counts,
        challenge_sha256=bank.sha256,
        costs=costs,
        common_validation_operations=common_validation_operations,
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
    )


def paired_team_evaluation(incumbent_team, candidate_team, bank, runner, objective,
                           initial_state, *, objective_name, candidate_proposal_attempts,
                           candidate_proposal_operations=0,
                           incumbent_search_attempts=0,
                           incumbent_search_operations=0,
                           common_validation_operations=0):
    """Select between complete teams on the same non-held-out immutable bank.

    ``runner(team_copy, episode_index, stimulus, state_copy, episode_streams)``
    returns ``(EpisodeResult, next_state)``. Callbacks are Python code, not a
    security sandbox: closures or globals may still access labels/state. Both
    arms start from independent copies of the same initial state; evaluation
    state is discarded. The objective receives explicit immutable per-arm
    costs. Held-out banks are rejected; use report_frozen_team for reporting
    without a winner or promotion.
    """
    if not isinstance(bank, ChallengeBank):
        raise TypeError("bank must be a ChallengeBank")
    if bank.partition == "held-out":
        raise ValueError("held-out banks cannot be used for team selection")
    if not callable(runner) or not callable(objective):
        raise TypeError("runner and objective must be callable")
    if not isinstance(objective_name, str) or not objective_name:
        raise ValueError("objective_name must be a nonempty string")
    candidate_proposal_attempts = _integer(
        "candidate_proposal_attempts", candidate_proposal_attempts
    )
    candidate_proposal_operations = _integer(
        "candidate_proposal_operations", candidate_proposal_operations
    )
    incumbent_search_attempts = _integer(
        "incumbent_search_attempts", incumbent_search_attempts
    )
    incumbent_search_operations = _integer(
        "incumbent_search_operations", incumbent_search_operations
    )
    common_validation_operations = _integer(
        "common_validation_operations", common_validation_operations
    )

    def evaluate(team):
        state = deepcopy(initial_state)
        results = []
        streams = bank.indexed_streams
        for index, stimulus in enumerate(bank.stimuli):
            result, next_state = runner(
                deepcopy(team), index, stimulus, deepcopy(state),
                EpisodeRandomStreams(streams, index),
            )
            if not isinstance(result, EpisodeResult):
                raise TypeError("runner must return an EpisodeResult for each episode")
            results.append(result)
            state = deepcopy(next_state)
        counts = _arm_summary(results, bank._targets)
        return counts, tuple(results)

    def score(results, arm, incumbent_costs, candidate_costs):
        context = ScoringContext(
            arm=arm,
            incumbent=incumbent_costs,
            candidate=candidate_costs,
            common_validation_operations=common_validation_operations,
        )
        value = objective(results, bank._targets, context)
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
    incumbent_costs = ArmCostAttribution(
        incumbent_search_attempts,
        incumbent_search_operations,
        incumbent_counts["compute_operations"],
    )
    candidate_costs = ArmCostAttribution(
        candidate_proposal_attempts,
        candidate_proposal_operations,
        candidate_counts["compute_operations"],
    )
    incumbent_score = score(
        incumbent_results, "incumbent", incumbent_costs, candidate_costs
    )
    candidate_score = score(
        candidate_results, "candidate", incumbent_costs, candidate_costs
    )
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
        incumbent_costs=incumbent_costs,
        candidate_costs=candidate_costs,
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
        target_direction_tp=4,
        target_direction_fp=1,
        target_direction_fn=2,
        target_direction_tn=3,
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
