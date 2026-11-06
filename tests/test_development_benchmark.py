import csv
import json

import pytest

from crystallization_contest import development_benchmark as benchmark
from crystallization_contest.evaluation_v2 import (
    EpisodeResult,
    EpisodeRandomStreams,
    IndexedStreams,
    ChallengeBank,
    make_cost_aware_utility_objective,
    paired_team_evaluation,
)
from crystallization_contest.helpers import fidelity


def _constant_stimulus(width, steps, coordinate):
    row = [
        [[1.0, 0.0] if index == coordinate else [0.0, 0.0]
         for index in range(width)]
        for _ in range(3)
    ]
    return [row for _ in range(steps)]


def _decode(encoded):
    return [complex(real, imag) for real, imag in json.loads(encoded)]


def test_real_membrane_dynamics_respond_to_numerical_stimuli():
    team = {"membranes": [{"th": 0.05, "c": 0.0, "g": 10.0} for _ in range(3)]}
    streams = EpisodeRandomStreams(IndexedStreams(22, "development", 1), 0)
    first, _ = benchmark.run_real_model_episode(
        team, 0, _constant_stimulus(16, 20, 0), {"sp": [0.5] * 3},
        streams, width=16, steps=20, shield=False,
    )
    second, _ = benchmark.run_real_model_episode(
        team, 0, _constant_stimulus(16, 20, 1), {"sp": [0.5] * 3},
        streams, width=16, steps=20, shield=False,
    )
    assert first.declared_direction is not None
    assert second.declared_direction is not None
    target_zero = [complex(index == 0) for index in range(16)]
    target_one = [complex(index == 1) for index in range(16)]
    assert fidelity(_decode(first.declared_direction), target_zero) > 0.99
    assert fidelity(_decode(second.declared_direction), target_one) > 0.99
    assert first.declared_direction != second.declared_direction


def test_real_model_dynamics_match_legacy_episode_reference(monkeypatch):
    from crystallization_contest import contest

    width = 16
    starts = [
        [complex(index == coordinate) for index in range(width)]
        for coordinate in range(3)
    ]
    stimulus = _constant_stimulus(width, 20, 0)
    params = [
        {"th": 0.05, "c": 0.0, "geff": 10.0, "sp": 0.5}
        for _ in range(3)
    ]

    def run_adapter():
        vectors = iter(starts)
        monkeypatch.setattr(benchmark, "rvec", lambda n, rng: next(vectors))
        return benchmark.run_real_model_episode(
            {"membranes": [
                {"th": p["th"], "c": p["c"], "g": p["geff"]}
                for p in params
            ]},
            0, stimulus, {"sp": [0.5] * 3},
            EpisodeRandomStreams(IndexedStreams(22, "development", 1), 0),
            width=width, steps=20, shield=False,
        )[0]

    adapted = run_adapter()
    vectors = iter(starts)
    monkeypatch.setattr(contest, "rvec", lambda n, rng: next(vectors))
    observations = iter(
        [[complex(real, imag) for real, imag in agent_row]
         for timestep in stimulus for agent_row in timestep]
    )
    monkeypatch.setattr(contest, "observe", lambda n, pattern, sigma, rng: next(observations))
    legacy = contest.episode(
        width, None, 1.5 / width ** 0.5, params, None, 0.0, None, steps=20,
    )
    assert adapted.raw_directions == tuple(
        benchmark._encode_vector(direction)
        for direction in legacy if direction is not None
    )


def test_banks_are_deterministic_disjoint_and_pre_materialized():
    development = benchmark.make_challenge_bank(
        730_001, "development", width=16, episodes=4, steps=2,
    )
    repeated = benchmark.make_challenge_bank(
        730_001, "development", width=16, episodes=4, steps=2,
    )
    validation = benchmark.make_challenge_bank(
        970_001, "calibration", width=16, episodes=4, steps=2,
    )
    assert development.sha256 == repeated.sha256
    assert development.sha256 != validation.sha256
    assert development.stimuli == repeated.stimuli
    assert development.stimuli != validation.stimuli
    with pytest.raises(ValueError, match="no held-out bank"):
        benchmark.make_challenge_bank(
            1, "held-out", width=16, episodes=2, steps=1,
        )


def test_paired_runner_gets_same_episode_stimuli_and_indexed_streams():
    duplicated_stimulus = _constant_stimulus(16, 1, 0)
    bank = ChallengeBank(
        73, "development",
        [duplicated_stimulus, duplicated_stimulus],
        ("target-a", "target-b"),
    )
    observations = {"incumbent": [], "candidate": []}

    def runner(team, episode_index, stimulus, state, streams):
        assert streams.episode_index == episode_index
        assert "labels" not in runner.__code__.co_varnames
        for _ in range(1 if team["name"] == "incumbent" else 7):
            streams.rng_for(channel="branch").random()
        observations[team["name"]].append((
            episode_index, stimulus,
            streams.rng_for(channel="paired-check").random(),
        ))
        return EpisodeResult(None, (), 0), state

    objective = make_cost_aware_utility_objective(
        b_tp=1, c_fp=2, c_fn=1, c_compute=0, c_attempt=0,
    )
    paired_team_evaluation(
        {"name": "incumbent"}, {"name": "candidate"}, bank, runner,
        objective, {"sp": [0.5] * 3}, objective_name="test",
        candidate_proposal_attempts=0,
    )
    assert observations["incumbent"] == observations["candidate"]
    assert observations["incumbent"][0][0] != observations["incumbent"][1][0]
    assert observations["incumbent"][0][1] == observations["incumbent"][1][1]
    assert observations["incumbent"][0][2] != observations["incumbent"][1][2]


def test_direction_metrics_keep_wrong_emissions_out_of_presence_tp():
    target = benchmark._encode_vector([complex(index == 0) for index in range(16)])
    wrong = benchmark._encode_vector([complex(index == 1) for index in range(16)])
    counts = benchmark.summarize_target_directions(
        (
            EpisodeResult(wrong, (target, wrong), 5),
            EpisodeResult(wrong, (wrong,), 4),
            EpisodeResult(None, (), 3),
        ),
        (target, None, None),
    )
    assert counts["target_direction_detection"] == {
        "TP": 0, "FP": 1, "FN": 1, "TN": 1,
    }
    assert counts["raw_declarations"] == 3
    assert counts["raw_correct_directions"] == 1
    assert counts["emitted_declarations_post_filter"] == 2
    assert counts["emitted_wrong_directions"] == 2
    assert counts["false_acceptance_target_absent_episodes"] == 1
    assert counts["correct_emitted_fraction"] == {
        "value": 0.0, "status": "defined",
    }


def test_benchmark_dry_run_writes_frozen_manifest_and_plan(tmp_path):
    plan = benchmark.run_development_benchmark(
        str(tmp_path), seeds=1, dry_run=True,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    written_plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["status"] == "planned"
    assert manifest["evidence_status"] == "exploratory_not_preregistered"
    assert manifest["frozen_config"]["policies"]["independent_candidate_search"] == (
        "NOT_IMPLEMENTED"
    )
    assert written_plan["notice"].startswith("No model evaluation")
    assert not (tmp_path / "results.json").exists()


def test_benchmark_smoke_replays_without_timing_in_hash(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = benchmark.run_development_benchmark(str(first_dir), seeds=1)
    second = benchmark.run_development_benchmark(str(second_dir), seeds=1)
    assert first["status"] == second["status"] == "completed"
    first_row, second_row = first["per_seed"][0], second["per_seed"][0]
    assert first_row["development_challenge_sha256"] == (
        second_row["development_challenge_sha256"]
    )
    assert first_row["validation_challenge_sha256"] == (
        second_row["validation_challenge_sha256"]
    )
    assert first_row["no_intervention"]["deterministic_replay_sha256"] == (
        second_row["no_intervention"]["deterministic_replay_sha256"]
    )
    assert first_row["random_bounded_team_proposals"]["costs"][
        "random_bounded_team_proposals"
    ]["search_attempts_including_failed"] == 4
    assert first_row["random_bounded_team_proposals"]["search_budget"][
        "proposal_operations"
    ] == 36
    assert first_row["random_bounded_team_proposals"]["search_budget"][
        "total_search_operations"
    ] <= first["source"]["frozen_config"]["budgets"][
        "random_search_operation_ceiling_per_seed"
    ]
    for path in (first_dir / "results.json", second_dir / "results.json"):
        result = json.loads(
            path.read_text(),
            parse_constant=lambda value: pytest.fail(f"invalid JSON number: {value}"),
        )
        assert result["evidence_note"].startswith("Exploratory")
        assert all(
            "held-out" not in seed["development_challenge_sha256"]
            for seed in result["per_seed"]
        )
    with (first_dir / "results.csv").open(newline="") as result_file:
        rows = list(csv.DictReader(result_file))
    assert {row["arm"] for row in rows} == set(benchmark.REGISTERED_ARMS)
    evidence = (first_dir / "EVIDENCE.md").read_text()
    assert "independent_candidate_search" in evidence
    assert "shared_candidate_search" in evidence


def test_benchmark_stops_at_traced_memory_budget(tmp_path):
    result = benchmark.run_development_benchmark(
        str(tmp_path), seeds=1, memory_bytes=1,
    )
    assert result["status"] == "memory_limit"
    assert result["per_seed"][0]["status"] == "memory_limit"
    assert result["per_seed"][0]["random_bounded_team_proposals"][
        "proposal_attempts"
    ] == 0
