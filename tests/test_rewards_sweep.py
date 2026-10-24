import csv
import math
import os
import random

import pytest

from crystallization_contest import rewards, sweep
from crystallization_contest.helpers import rvec

HERE = os.path.dirname(__file__)


def _vecs():
    rng = random.Random(0)
    return rvec(16, rng), rvec(16, rng)


def test_consensus_k_of_n_rule():
    assert [rewards.consensus_k(n) for n in (3, 5, 9)] == [2, 4, 6]
    a, b = _vecs()
    # 3 agents, k = 2: two agreeing agents are rewarded, the lone one is not
    d = [a, a, b]
    assert rewards.reward("consensus", 0, d, 3) == 1.0
    assert rewards.reward("consensus", 1, d, 3) == 1.0
    assert rewards.reward("consensus", 2, d, 3) == 0.0
    # 5 agents, k = 4: three agreeing agents are not enough
    d5 = [a, a, a, b, None]
    assert all(rewards.reward("consensus", i, d5, 5) == 0.0 for i in range(5))
    d5 = [a, a, a, a, None]
    assert rewards.reward("consensus", 0, d5, 5) == 1.0
    assert rewards.reward("consensus", 4, d5, 5) == 0.0


def test_individual_team_difference_random_values():
    a, b = _vecs()
    d = [a, a, None]
    assert [rewards.reward("individual", i, d, 3) for i in range(3)] == [1.0, 1.0, 0.0]
    assert [rewards.reward("team", i, d, 3) for i in range(3)] == [1.0, 1.0, 1.0]
    # removing agent 0 or 1 breaks the 2-of-3 event, removing agent 2 does not
    assert [rewards.reward("difference", i, d, 3) for i in range(3)] == [1.0, 1.0, 0.0]
    assert [rewards.reward("random", i, d, 3) for i in range(3)] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        rewards.reward("nope", 0, d, 3)


def test_reward_schemes_cover_all_names():
    a, _ = _vecs()
    for s in rewards.SCHEMES:
        assert isinstance(rewards.reward(s, 0, [a, a, a], 3), float)


def test_sweep_row_shape_and_ground_truth_bounds():
    row = sweep.one(("team", 16, 3, 0, 4, 40))
    assert row["scheme"] == "team" and row["width"] == 16 and row["N"] == 3
    assert 0 <= row["true_events"] <= 2          # 2 planted episodes of 4
    assert 0 <= row["false_events"] <= 4
    assert row["true_events"] + row["false_events"] <= 4
    if row["true_events"] + row["false_events"]:
        assert 0.0 <= row["precision"] <= 1.0
    else:
        assert math.isnan(row["precision"])


def test_sweep_is_deterministic_for_fixed_seed():
    a = sweep.one(("consensus", 16, 3, 7, 4, 40))
    b = sweep.one(("consensus", 16, 3, 7, 4, 40))
    assert repr(a) == repr(b)  # repr compares NaN precision as equal


def test_sweep_summary_shape():
    rows, agg = sweep.run_sweep([16], [3], 2, 1, episodes=4, steps=40)
    assert len(rows) == len(rewards.SCHEMES) * 2
    assert len(agg) == len(rewards.SCHEMES)
    assert {r["scheme"] for r in agg} == set(rewards.SCHEMES)
    assert all(r["seeds"] == 2 for r in agg)


def test_sweep_reproduces_a_stored_row():
    path = os.path.join(HERE, "..", "results", "sweep_raw.csv")
    with open(path) as f:
        stored = {(r["scheme"], int(r["width"]), int(r["N"]), int(r["seed"])): r
                  for r in csv.DictReader(f)}
    key = ("consensus", 32, 3, 0)
    row = sweep.one(key)
    assert row["true_events"] == int(stored[key]["true_events"])
    assert row["false_events"] == int(stored[key]["false_events"])
    assert row["agent_declarations"] == int(stored[key]["agent_declarations"])
