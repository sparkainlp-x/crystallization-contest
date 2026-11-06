import json
import math

import pytest

from crystallization_contest.evaluation_v2 import (
    ChallengeBank,
    EpisodeResult,
    legacy_cooperative_score,
    normalized_cooperative_score,
    paired_team_evaluation,
    protocol_manifest,
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
        tp=4,
        fp=1,
        fn=2,
        tn=3,
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
    assert json.loads(strict_json_dumps(result))["metrics"]["precision"]["value"] == 0.8

    empty = utility_score(
        2, tp=0, fp=0, fn=0, tn=2, b_tp=1, c_fp=1, c_fn=1,
        c_compute=0, c_attempt=0, compute_operations=0, proposal_attempts=0,
    )
    assert empty["metrics"]["precision"] == {
        "value": None,
        "status": "undefined_zero_denominator",
    }
    assert empty["metrics"]["recall"]["value"] is None
    assert "null" in strict_json_dumps(empty)
    with pytest.raises(ValueError):
        strict_json_dumps({"not_json": math.nan})


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp": 2},
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
        tp=1,
        fp=0,
        fn=0,
        tn=1,
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
        partition="held-out",
        stimuli=[{"stimulus": [1, 2]}, {"stimulus": [3, 4]}],
        _targets=("target", None),
    )
    starts = []
    stream_values = []

    def runner(team, stimulus, state, rng_for):
        starts.append((team["name"], state["warm"], state["rate_ema"],
                       tuple(state["self_confirmation"])))
        episode = 0 if stimulus[0][1] == (1, 2) else 1
        # Candidate consumes extra random values; indexed streams keep all later
        # environment draws fixed for the same exogenous indices.
        first = rng_for(episode, step=4, agent=1).random()
        if team["name"] == "candidate":
            rng_for(episode, step=1, agent=0).random()
            rng_for(episode, step=1, agent=0).random()
        later = rng_for(episode, step=5, agent=1).random()
        stream_values.append((team["name"], episode, first, later))
        state["warm"] = team["name"]
        state["rate_ema"] += 0.1
        state["self_confirmation"].append(team["name"])
        direction = team["direction"] if episode == 0 else None
        raw = (direction,) if direction else ()
        return EpisodeResult(direction, raw, operation_count=10), state

    if objective is None:
        objective = lambda results, labels: sum(
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
        common_validation_operations=4,
    )
    return bank, result, starts, stream_values


def test_paired_evaluation_is_isolated_and_accepts_only_evaluated_full_team():
    incumbent = {"name": "incumbent", "direction": None, "member_a": 1, "member_b": 2}
    candidate = {"name": "candidate", "direction": "target", "member_a": 8, "member_b": 9}
    initial = {
        "warm": "initial",
        "rate_ema": 0.5,
        "self_confirmation": ["initial"],
    }
    bank, result, starts, streams = _paired_fixture(incumbent, candidate, initial)
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
    assert result.incumbent["FN"] == 1
    assert result.candidate["TP"] == 1
    assert result.challenge_sha256 == bank.sha256
    assert result.candidate_proposal_attempts == 3
    assert result.report()["costs"]["candidate_proposal_operations"] == 12

    by_episode = {}
    for arm, episode, first, later in streams:
        by_episode.setdefault(episode, {})[arm] = (first, later)
    for arm_values in by_episode.values():
        assert arm_values["incumbent"] == arm_values["candidate"]


def test_paired_ties_keep_incumbent_and_nonfinite_objectives_reject():
    incumbent = {"name": "incumbent", "direction": None}
    candidate = {"name": "candidate", "direction": None}
    _, tied, _, _ = _paired_fixture(incumbent, candidate)
    assert not tied.accepted
    assert tied.accepted_team == incumbent
    with pytest.raises(ValueError, match="finite scalar"):
        _paired_fixture(incumbent, candidate, objective=lambda results, labels: math.nan)


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


def test_worked_example_is_machine_readable_and_matches_documented_values():
    example = worked_example()
    assert example["status"] == "illustrative_not_optimal"
    assert example["normalized_cooperative"]["score"] == pytest.approx(0.25)
    assert example["target_utility"]["total_utility"] == pytest.approx(4.6)
    assert json.loads(strict_json_dumps(example))["status"] == "illustrative_not_optimal"


def test_utility_costs_include_attempts_and_operations_without_merging_timing():
    _, paired, _, _ = _paired_fixture(
        {"name": "incumbent", "direction": None},
        {"name": "candidate", "direction": "target"},
    )
    costs = paired.report()["costs"]
    assert costs["candidate_proposal_attempts"] == 3
    assert costs["candidate_proposal_operations"] == 12
    assert costs["incumbent_evaluation_operations"] == 20
    assert costs["candidate_evaluation_operations"] == 20
    assert costs["common_validation_operations"] == 4
    assert costs["wall_seconds"] >= 0
    assert costs["cpu_seconds"] >= 0
