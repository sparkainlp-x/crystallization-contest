import csv
import os
import random

import pytest

from crystallization_contest import contest
from crystallization_contest.helpers import rvec, fidelity

HERE = os.path.dirname(__file__)


def _vecs(k=3, n=16, seed=0):
    rng = random.Random(seed)
    return [rvec(n, rng) for _ in range(k)]


def test_shield_requires_two_of_three():
    a, b, c = _vecs()
    assert contest.shield_select([a, a, None]) == 0
    assert contest.shield_select([None, a, a]) == 1
    assert contest.shield_select([a, b, c]) is None         # fail closed: no pair agrees
    assert contest.shield_select([a, None, None]) is None   # a single membrane is dropped
    assert contest.shield_select([None, None, None]) is None


def test_amplifier_weight_is_capped():
    v, warm = _vecs(2)
    blended = contest.warm_start(v, warm, 5.0)               # request far above the cap
    capped = contest.warm_start(v, warm, contest.AMP_CAP)
    assert blended == capped
    assert fidelity(capped, warm) > fidelity(v, warm)
    assert fidelity(capped, warm) < 1.0                      # never a pure copy
    assert contest.warm_start(v, None, 0.5) is v
    assert contest.warm_start(v, warm, 0.0) is v
    assert contest.warm_start(v, warm, -1.0) is v


def test_stabilizer_factor_and_clamp_bounds():
    for r in (0.0, 0.5, 1.0, 5.0, 100.0):
        f = contest.stabilizer_factor(r)
        assert contest.FACTOR_MIN <= f
    assert contest.stabilizer_factor(contest.TARGET) == 1.0
    assert contest.stabilizer_factor(1.0) < 1.0              # above target: damp
    assert contest.stabilizer_factor(0.0) > 1.0              # below target: boost
    for g in (-3.0, 0.0, 0.1, 1.0, 50.0, 1e9):
        assert contest.GAIN_MIN <= contest.clamp_gain(g) <= contest.GAIN_MAX


def test_leaky_integrator_stays_bounded_and_converges():
    r = 0.9
    for _ in range(200):
        r = contest.leaky_update(r, 0.0)
        assert 0.0 <= r <= 1.0
    assert r < 1e-6
    r = 0.0
    for _ in range(200):
        r = contest.leaky_update(r, 1.0)
        assert 0.0 <= r <= 1.0
    assert r == pytest.approx(1.0)
    assert contest.leaky_update(0.4, 0.4) == pytest.approx(0.4)


def test_config_tags_round_trip():
    assert contest.cfg_tag((1, 1, 1)) == "SAZ"
    assert contest.cfg_tag((0, 0, 0)) == "---"
    for tag, cfg in contest.CFG_BY_TAG.items():
        assert contest.cfg_tag(cfg) == tag


OPTS = dict(gens=3, ep=4, steps=40)


def test_contest_row_and_determinism():
    a = contest.run((16, 0.5, (1, 1, 1), 0, OPTS))
    b = contest.run((16, 0.5, (1, 1, 1), 0, OPTS))
    assert repr(a) == repr(b)  # repr compares NaN precision as equal
    assert a["cfg"] == "SAZ" and a["width"] == 16
    assert a["true_events"] >= 0 and a["false_events"] >= 0
    # shield: at most one output event per episode
    assert a["true_events"] + a["false_events"] <= OPTS["gens"] * OPTS["ep"]


def test_stabilizer_keeps_gain_bounded_under_stress():
    stress = dict(OPTS, gens=8, init_g=3.0, gsig=0.6)
    with_z = [contest.run((16, 0.5, (1, 1, 1), s, stress))["max_gain"] for s in range(4)]
    assert max(with_z) <= contest.GAIN_MAX + 1e-9


def test_contest_summary_shape():
    rows, agg = contest.run_contest([16], [0.5], [(0, 0, 0), (1, 1, 1)], 2, 1, OPTS)
    assert len(rows) == 4 and len(agg) == 2
    assert {r["cfg"] for r in agg} == {"---", "SAZ"}
    assert all("runaway_frac" in r for r in agg)


def test_contest_reproduces_a_stored_row():
    path = os.path.join(HERE, "..", "results", "contest_w32-128_raw.csv")
    with open(path) as f:
        stored = {(int(r["width"]), float(r["w"]), r["cfg"], int(r["seed"])): r
                  for r in csv.DictReader(f)}
    row = contest.run((32, 0.5, (0, 0, 0), 0))
    s = stored[(32, 0.5, "---", 0)]
    assert row["true_events"] == int(s["true_events"])
    assert row["false_events"] == int(s["false_events"])
    assert row["max_gain"] == pytest.approx(float(s["max_gain"]))
