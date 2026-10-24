import math
import random

import pytest

from crystallization_contest import events
from crystallization_contest.helpers import fidelity, inner, noisy, normalize, rvec


def test_threshold_scaling_formula():
    for n in (32, 64, 128, 256, 512):
        sigma = events.sigma_for(n)
        for q in events.Q_GRID:
            assert events.scaled_threshold(q, n, sigma) == pytest.approx(
                1.0 + q * n / (1.0 + 2.0 * sigma ** 2 * n))
    assert events.scaled_threshold(0.0, 64, 0.2) == 1.0


def test_threshold_grid_is_comparable_across_widths():
    # sigma = 1.5/sqrt(n): 2 sigma^2 n = 4.5, so threshold - 1 = q n / 5.5 grows ~ n
    # but as a fraction of the settled gap it is exactly q.
    for n in (32, 512):
        sigma = events.sigma_for(n)
        gap = n / (1.0 + 2.0 * sigma ** 2 * n)
        for q in events.Q_GRID:
            assert (events.scaled_threshold(q, n, sigma) - 1.0) / gap == pytest.approx(q)


def test_helpers_unit_norm_and_fidelity():
    rng = random.Random(1)
    v = rvec(16, rng)
    assert math.sqrt(sum(abs(c) ** 2 for c in v)) == pytest.approx(1.0)
    assert fidelity(v, v) == pytest.approx(1.0)
    assert 0.0 <= fidelity(v, rvec(16, rng)) <= 1.0
    assert inner(v, v).real == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize([0, 0])
    with pytest.raises(ValueError):
        noisy(v, -1.0, rng)


def test_oja_converges_to_planted_direction():
    n = 32
    rng = random.Random(3)
    pattern = rvec(n, rng)
    sigma = events.sigma_for(n)
    v = rvec(n, rng)
    start = fidelity(v, pattern)
    for _ in range(200):
        v = events.oja_step(v, noisy(pattern, sigma, rng), events.ETA)
    assert fidelity(v, pattern) > 0.6
    assert fidelity(v, pattern) > start


def test_run_agent_declares_on_pattern_and_not_on_noise_at_high_threshold():
    n = 32
    rng = random.Random(5)
    pattern = rvec(n, rng)
    sigma = events.sigma_for(n)
    got = events.run_agent(n, pattern, sigma, 0.1, rng)
    assert got is not None and len(got) == n
    # a high threshold on pure noise: the energy ratio of noise stays near 1
    declared = sum(events.run_agent(n, None, sigma, 0.9, random.Random(s)) is not None
                   for s in range(10))
    assert declared == 0


def test_event_labelling_true_and_false():
    rng = random.Random(2)
    pattern = rvec(16, rng)
    other = rvec(16, rng)
    assert events.is_true_event(pattern, pattern) is True
    assert not events.is_true_event(other, pattern)         # wrong direction
    assert not events.is_true_event(pattern, None)          # noise episode
    assert not events.is_true_event(None, pattern)          # no event


def test_system_event_needs_ceil_half_agreeing():
    rng = random.Random(4)
    a, b, c = rvec(16, rng), rvec(16, rng), rvec(16, rng)
    assert events.system_event([a, a, None], 3) is a        # 2 of 3 agree, need 2
    assert events.system_event([a, b, c], 3) is None        # nobody agrees
    assert events.system_event([a, None, None, None, None], 5) is None  # need 3
    assert events.system_event([a, a, a, None, None], 5) is a
    assert events.system_event([None, None, None], 3) is None


def test_determinism_of_run_agent():
    n = 32
    pattern = rvec(n, random.Random(9))
    sigma = events.sigma_for(n)
    a = events.run_agent(n, pattern, sigma, 0.5, random.Random(11))
    b = events.run_agent(n, pattern, sigma, 0.5, random.Random(11))
    assert a == b
