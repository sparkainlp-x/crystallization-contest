import json
import math
from dataclasses import FrozenInstanceError

import pytest

from crystallization_contest.evaluation_v2 import (
    ArmCostAttribution,
    ChallengeBank,
    EpisodeResult,
    ScoringContext,
    make_cost_aware_utility_objective,
    legacy_cooperative_score,
    normalized_cooperative_score,
    paired_team_evaluation,
    protocol_manifest,
    report_frozen_team,
    strict_json_dumps,
    utility_score,
    worked_example,
)


def test_legacy_and_normalized_formula_breakdowns_and_ties():
    kwargs = dict(
        episodes=4,
        w=0.5,
        joint_true=1,
        true_counts=[2, 2, 1],
        false_counts=[1, 0, 1],
        agent=0,
        fcost=0.5,
    )
    legacy = legacy_cooperative_score(**kwargs)
    v2 = normalized_cooperative_score(**kwargs)
    assert legacy["version"] == "legacy-contest-v1"
    assert legacy["contributions"] == {
        "joint_true": 0.5,
        "rank_bonus": 0.25,
        "false_event_penalty": -0.125,
    }
    assert legacy["score"] == pytest.approx(0.625)
    assert v2["contributions"] == {
        "joint_true": 0.125,
        "rank_bonus": 0.25,
        "false_event_penalty": -0.125,
    }
    assert v2["score"] == pytest.approx(0.25)
    assert v2["parameters"]["rank_leaders"] == [0, 1]

    tied = normalized_cooperative_score(4, 0.5, 0, [0, 0, 0], [0, 0, 0], 0)
    assert tied["contributions"]["rank_bonus"] == 0
    assert tied["parameters"]["rank_bonus_zero_reason"] == "all_true_counts_zero"
    assert tied["parameters"]["rank_leaders"] == []


def test_normalized_score_is_episode_scaled_but_legacy_joint_term_is_not():
    small = dict(
        episodes=4,
        w=0.5,
        joint_true=1,
        true_counts=[2, 1, 1],
        false_counts=[1, 1, 0],
        agent=0,
        fcost=0.5,
    )
    large = dict(
        episodes=8,
        w=0.5,
        joint_true=2,
        true_counts=[4, 2, 2],
        false_counts=[2, 2, 0],
        agent=0,
        fcost=0.5,
    )
    assert normalized_cooperative_score(**small)["score"] == pytest.approx(
        normalized_cooperative_score(**large)["score"]
    )
    assert legacy_cooperative_score(**large)["contributions"]["joint_true"] == 2 * (
        legacy_cooperative_score(**small)["contributions"]["joint_true"]
    )


def test_joint_count_requires_feasible_intersection_bounds():
    valid_at_lower_bound = legacy_cooperative_score(
        4, 0.5, 1, [4, 4, 1], [0, 0, 3], 0
    )
    assert valid_at_lower_bound["contributions"]["joint_true"] == 0.5
    valid_at_upper_bound = normalized_cooperative_score(
        4, 0.5, 2, [4, 3, 2], [0, 1, 2], 0
    )
    assert valid_at_upper_bound["counts"]["joint_true_episodes"] == 2
    with pytest.raises(ValueError, match="consistent"):
        normalized_cooperative_score(4, 0.5, 0, [4, 4, 4], [0, 0, 0], 0)
    with pytest.raises(ValueError, match="consistent"):
        normalized_cooperative_score(4, 0.5, 2, [1, 1, 1], [0, 0, 0], 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": 0},
        {"episodes": True},
        {"w": float("nan")},
        {"w": 1.01},
        {"joint_true": 2},
        {"true_counts": [1.5, 1, 1]},
        {"true_counts": [True, 1, 1]},
        {"false_counts": [0, -1, 0]},
        {"false_counts": [0, 5, 0]},
        {"false_counts": [3, 0, 1]},
        {"true_counts": [1, 1]},
        {"agent": 3},
        {"fcost": float("inf")},
        {"fcost": -1},
    ],
)
def test_cooperative_scores_reject_malformed_inputs(kwargs):
    args = dict(
        episodes=4,
        w=0.5,
        joint_true=1,
        true_counts=[2, 2, 1],
        false_counts=[1, 0, 1],
        agent=0,
        fcost=0.5,
    )
    args.update(kwargs)
    with pytest.raises(ValueError):
        normalized_cooperative_score(**args)


def test_utility_arithmetic_units_zero_detection_and_strict_json():
    result = utility_score(
        10,
        target_direction_tp=4,
        target_direction_fp=1,
        target_direction_fn=2,
        target_direction_tn=3,
        b_tp=2,
        c_fp=1,
        c_fn=0.5,
        c_compute=0.01,
        c_attempt=0.2,
        compute_operations=100,
        proposal_attempts=2,
        incorrect_direction_declarations=1,
        raw_declarations=7,
        raw_true_declarations=4,
        raw_false_declarations=3,
        wall_seconds=0.25,
        cpu_seconds=0.1,
    )
    assert result["total_utility"] == pytest.approx(4.6)
    assert result["per_episode_utility"] == pytest.approx(0.46)
    assert result["component_totals"] == {
        "true_positive_reward": 8.0,
        "false_positive_penalty": -1.0,
        "false_negative_penalty": -1.0,
        "compute_penalty": -1.0,
        "attempt_penalty": -0.4,
    }
    assert result["raw_counts"]["incorrect_direction_declarations"] == 1
    assert result["units"]["c_compute"] == "utility per deterministic operation"
    assert result["endpoint"] == "episode_level_target_direction_detection"
    assert (
        json.loads(strict_json_dumps(result))["metrics"]
        ["target_direction_precision"]["value"] == 0.8
    )

    empty = utility_score(
        2, target_direction_tp=0, target_direction_fp=0,
        target_direction_fn=0, target_direction_tn=2, b_tp=1, c_fp=1, c_fn=1,
        c_compute=0, c_attempt=0, compute_operations=0, proposal_attempts=0,
    )
    assert empty["metrics"]["target_direction_precision"] == {
        "value": None,
        "status": "undefined_zero_denominator",
    }
    assert empty["metrics"]["target_direction_recall"]["value"] is None
    assert "null" in strict_json_dumps(empty)
    with pytest.raises(ValueError):
        strict_json_dumps({"not_json": math.nan})


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_direction_tp": 2},
        {"c_fp": -1},
        {"b_tp": math.inf},
        {"compute_operations": 1.5},
        {"proposal_attempts": True},
        {"raw_declarations": 2},
        {"incorrect_direction_declarations": 2},
    ],
)
def test_utility_rejects_bad_counts_and_coefficients(overrides):
    args = dict(
        episodes=2,
        target_direction_tp=1,
        target_direction_fp=0,
        target_direction_fn=0,
        target_direction_tn=1,
        b_tp=1,
        c_fp=1,
        c_fn=1,
        c_compute=0,
        c_attempt=0,
        compute_operations=0,
        proposal_attempts=1,
        raw_declarations=1,
        raw_true_declarations=1,
        raw_false_declarations=0,
    )
    args.update(overrides)
    with pytest.raises(ValueError):
        utility_score(**args)


def _paired_fixture(incumbent, candidate, initial_state=None, objective=None):
    bank = ChallengeBank(
        seed=17,
        partition="calibration",
        stimuli=[{"stimulus": [1, 2]}, {"stimulus": [1, 2]}],
        _targets=("target", None),
    )
    starts = []
    stream_values = []
    stimuli_seen = []

    def runner(team, episode_index, stimulus, state, episode_streams):
        assert not hasattr(episode_streams, "_targets")
        assert not hasattr(episode_streams, "stimuli")
        episode = episode_index
        stimuli_seen.append((team["name"], episode, stimulus))
        starts.append((team["name"], state["warm"], state["rate_ema"],
                       tuple(state["self_confirmation"])))
        # Candidate consumes extra random values; indexed streams keep all later
        # environment draws fixed for the same exogenous indices.
        first = episode_streams.rng_for(step=4, agent=1).random()
        if team["name"] == "candidate":
            episode_streams.rng_for(step=1, agent=0).random()
            episode_streams.rng_for(step=1, agent=0).random()
        later = episode_streams.rng_for(step=5, agent=1).random()
        stream_values.append((team["name"], episode, first, later))
        state["warm"] = team["name"]
        state["rate_ema"] += 0.1
        state["self_confirmation"].append(team["name"])
        direction = team["direction"] if episode == 0 else team.get("noise_direction")
        raw = (direction, direction) if direction and team["name"] == "candidate" else (
            (direction,) if direction else ()
        )
        return EpisodeResult(direction, raw, operation_count=10), state

    if objective is None:
        objective = lambda results, labels, context: sum(
            r.declared_direction == target
            for r, target in zip(results, labels)
            if target is not None
        )
    result = paired_team_evaluation(
        incumbent,
        candidate,
        bank,
        runner,
        objective,
        initial_state or {
            "warm": "initial",
            "rate_ema": 0.5,
            "self_confirmation": ["initial"],
        },
        objective_name="correct_target_detections",
        candidate_proposal_attempts=3,
        candidate_proposal_operations=12,
        incumbent_search_attempts=2,
        incumbent_search_operations=5,
        common_validation_operations=4,
    )
    return bank, result, starts, stream_values, stimuli_seen


def test_paired_evaluation_is_isolated_and_accepts_only_evaluated_full_team():
    incumbent = {"name": "incumbent", "direction": "wrong", "member_a": 1, "member_b": 2}
    candidate = {
        "name": "candidate", "direction": "target", "noise_direction": "noise",
        "member_a": 8, "member_b": 9,
    }
    initial = {
        "warm": "initial",
        "rate_ema": 0.5,
        "self_confirmation": ["initial"],
    }
    bank, result, starts, streams, stimuli = _paired_fixture(incumbent, candidate, initial)
    assert result.accepted
    assert result.accepted_team == candidate
    assert (result.accepted_team["member_a"], result.accepted_team["member_b"]) == (8, 9)
    assert initial == {
        "warm": "initial",
        "rate_ema": 0.5,
        "self_confirmation": ["initial"],
    }
    assert starts[0][1:] == starts[2][1:] == (
        "initial", 0.5, ("initial",)
    )
    assert not hasattr(bank, "targets")
    assert result.incumbent["target_direction_detection"]["FN"] == 1
    assert result.candidate["target_direction_detection"]["TP"] == 1
    assert result.incumbent["incorrect_direction_declarations"] == 1
    assert result.candidate["target_direction_detection"]["FP"] == 1
    assert result.candidate["raw_false_declarations"] == 2
    assert result.candidate["raw_declarations"] == 4
    assert result.candidate["emitted_declarations"] == 2
    assert result.challenge_sha256 == bank.sha256
    assert result.candidate_costs.search_attempts == 3
    assert result.incumbent_costs.search_attempts == 2
    assert result.report()["costs"]["candidate"]["search_operations"] == 12
    assert result.report()["costs"]["common_validation_operations"] == 4

    by_episode = {}
    for arm, episode, first, later in streams:
        by_episode.setdefault(episode, {})[arm] = (first, later)
    for arm_values in by_episode.values():
        assert arm_values["incumbent"] == arm_values["candidate"]
    stimuli_by_episode = {}
    for arm, episode, stimulus in stimuli:
        stimuli_by_episode.setdefault(episode, {})[arm] = stimulus
    for arm_values in stimuli_by_episode.values():
        assert arm_values["incumbent"] == arm_values["candidate"]


def test_paired_ties_keep_incumbent_and_nonfinite_objectives_reject():
    incumbent = {"name": "incumbent", "direction": None}
    candidate = {"name": "candidate", "direction": None}
    initial = {"warm": "rollback", "rate_ema": 0.25, "self_confirmation": []}
    _, tied, _, _, _ = _paired_fixture(incumbent, candidate, initial)
    assert not tied.accepted
    assert tied.accepted_team == incumbent
    assert initial == {"warm": "rollback", "rate_ema": 0.25, "self_confirmation": []}
    with pytest.raises(ValueError, match="finite scalar"):
        _paired_fixture(
            incumbent, candidate,
            objective=lambda results, labels, context: math.nan,
        )
    with pytest.raises(ValueError, match="finite scalar"):
        _paired_fixture(
            incumbent, candidate,
            objective=lambda results, labels, context: 10 ** 10000,
        )


def test_selection_rejects_held_out_bank_and_report_only_cannot_promote():
    bank = ChallengeBank(23, "held-out", ["stimulus"], ("secret-target",))
    actions = []

    def runner(team, episode_index, stimulus, state, episode_streams):
        actions.append(("run", team["id"]))
        return EpisodeResult(None, (), 1), state

    def objective(results, labels, context):
        assert actions == [("run", "untouched")]
        assert context.arm == "frozen"
        actions.append(("score", labels))
        return 0

    with pytest.raises(ValueError, match="held-out"):
        paired_team_evaluation(
            "incumbent", "candidate", bank, runner,
            lambda results, labels, context: 0, {},
            objective_name="fixed", candidate_proposal_attempts=1,
        )
    frozen_team = {"id": "untouched"}
    report = report_frozen_team(
        frozen_team, bank, runner, objective, {},
        objective_name="report-only",
    )
    assert report.report()["selection"] == "not_performed"
    assert not hasattr(report, "accepted_team")
    assert frozen_team == {"id": "untouched"}
    assert actions[-1] == ("score", ("secret-target",))
    with pytest.raises(ValueError, match="held-out"):
        report_frozen_team(
            frozen_team,
            ChallengeBank(24, "calibration", ["stimulus"], (None,)),
            runner,
            objective,
            {},
            objective_name="report-only",
        )


def test_accepted_snapshot_cannot_be_changed_during_scoring():
    incumbent = {"id": "incumbent"}
    candidate = {"id": "evaluated-candidate"}
    bank = ChallengeBank(29, "calibration", ["stimulus"], (None,))

    def runner(team, episode_index, stimulus, state, episode_streams):
        return EpisodeResult(None, (), 1), state

    scores = iter((0, 1))

    def objective(results, labels, context):
        candidate["id"] = "mutated-after-evaluation"
        return next(scores)

    result = paired_team_evaluation(
        incumbent,
        candidate,
        bank,
        runner,
        objective,
        {},
        objective_name="fixed",
        candidate_proposal_attempts=1,
    )
    assert result.accepted
    assert result.accepted_team == {"id": "evaluated-candidate"}


def test_cost_aware_objective_can_reject_more_detections_after_full_cost():
    bank = ChallengeBank(
        31,
        "calibration",
        ["present-0", "present-1"],
        ("target", "target"),
    )

    def runner(team, episode_index, stimulus, state, episode_streams):
        detected = episode_index < team["detections"]
        direction = "target" if detected else None
        raw = (direction,) if direction else ()
        return EpisodeResult(direction, raw, operation_count=1), state

    base_objective = make_cost_aware_utility_objective(
        b_tp=1, c_fp=0, c_fn=0, c_compute=0.01, c_attempt=0.1
    )
    contexts = []

    def objective(results, labels, context):
        contexts.append(context)
        return base_objective(results, labels, context)

    result = paired_team_evaluation(
        {"detections": 1},
        {"detections": 2},
        bank,
        runner,
        objective,
        {},
        objective_name="target_direction_utility_after_cost",
        incumbent_search_attempts=0,
        candidate_proposal_attempts=10,
        candidate_proposal_operations=1000,
        common_validation_operations=500,
    )
    assert result.candidate["target_direction_detection"]["TP"] == 2
    assert result.incumbent["target_direction_detection"]["TP"] == 1
    assert result.candidate_score < result.incumbent_score
    assert not result.accepted
    costs = result.report()["costs"]
    assert costs["candidate"]["search_attempts_including_failed"] == 10
    assert costs["candidate"]["deployable_compute_operations"] == 1002
    assert costs["common_validation_operations"] == 500
    assert contexts[0].arm == "incumbent" and contexts[1].arm == "candidate"
    assert contexts[0].candidate.search_attempts == 10
    with pytest.raises(FrozenInstanceError):
        contexts[0].arm = "candidate"


def test_bank_manifest_hash_and_indexed_streams_are_replayable():
    bank = ChallengeBank(8, "development", [(1, 2), (3, 4)], ("a", None))
    same = ChallengeBank(8, "development", [(1, 2), (3, 4)], ("a", None))
    other = ChallengeBank(9, "development", [(1, 2), (3, 4)], ("a", None))
    assert bank.sha256 == same.sha256
    assert bank.sha256 != other.sha256
    assert bank.rng_for(0, step=1).random() == bank.rng_for(0, step=1).random()
    assert bank.rng_for(0, step=1).random() != bank.rng_for(0, step=2).random()
    manifest = protocol_manifest(bank, {"w": 0.5, "mode": "frozen"})
    assert manifest["evidence_status"] == "exploratory_unregistered"
    assert manifest["seed_partition"] == "development"
    assert json.loads(strict_json_dumps(manifest))["challenge_sha256"] == bank.sha256
    with pytest.raises(ValueError):
        protocol_manifest(bank, {"config": "x" * 5000})


def test_paired_replay_is_deterministic_except_for_reported_timing():
    incumbent = {"name": "incumbent", "direction": None}
    candidate = {"name": "candidate", "direction": "target"}
    _, first, _, _, _ = _paired_fixture(incumbent, candidate)
    _, second, _, _, _ = _paired_fixture(incumbent, candidate)
    assert first.accepted == second.accepted
    assert first.incumbent_score == second.incumbent_score
    assert first.candidate_score == second.candidate_score
    assert first.incumbent == second.incumbent
    assert first.candidate == second.candidate
    assert first.challenge_sha256 == second.challenge_sha256
    duplicate_bank, _, _, _, stimuli = _paired_fixture(incumbent, candidate)
    assert stimuli[0][2] == stimuli[1][2]
    stream0 = duplicate_bank.indexed_streams
    assert stream0.rng_for(0, step=4, agent=1).random() != (
        stream0.rng_for(1, step=4, agent=1).random()
    )


def test_worked_example_is_machine_readable_and_matches_documented_values():
    example = worked_example()
    assert example["status"] == "illustrative_not_optimal"
    assert example["normalized_cooperative"]["score"] == pytest.approx(0.25)
    assert example["target_utility"]["total_utility"] == pytest.approx(4.6)
    assert example["target_utility"]["endpoint"] == (
        "episode_level_target_direction_detection"
    )
    assert json.loads(strict_json_dumps(example))["status"] == "illustrative_not_optimal"


def test_utility_costs_include_attempts_and_operations_without_merging_timing():
    _, paired, _, _, _ = _paired_fixture(
        {"name": "incumbent", "direction": None},
        {"name": "candidate", "direction": "target"},
    )
    costs = paired.report()["costs"]
    assert costs["candidate"]["search_attempts_including_failed"] == 3
    assert costs["candidate"]["search_operations"] == 12
    assert costs["incumbent"]["evaluation_operations"] == 20
    assert costs["candidate"]["evaluation_operations"] == 20
    assert costs["common_validation_operations"] == 4
    assert costs["wall_seconds"] >= 0
    assert costs["cpu_seconds"] >= 0
