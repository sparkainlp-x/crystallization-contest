"""The crystallization event and its ground-truth labels (SYNTHETIC).

membrane-crystallization uses "crystallization" only as a descriptive label for
a streaming covariance estimate settling; it has no discrete event. This module
defines one for the harness:

* An agent tracks its top component v with Oja's rule (rank-1 form of the
  Hebbian running estimate) and declares an event at the first checkpoint where
  the explained-energy ratio  n * mean(last WINDOW observations |<x|v>|^2)
  reaches its threshold.
* A system event needs at least ceil(N/2) agents whose declared vectors agree
  (fidelity >= AGREE with a common anchor).
* Ground truth: in a planted episode the observations are noisy copies of one
  pattern; in a noise episode they are random vectors. A system event is TRUE
  when the episode is planted and its vector has fidelity >= TRUTH to the
  pattern. Every other system event is FALSE.
"""

import math

from .helpers import fidelity, inner, noisy, normalize, rvec

STEPS, CHECK, WINDOW, ETA = 100, 10, 5, 0.5
AGREE, TRUTH = 0.4, 0.5
Q_GRID = [0.1, 0.25, 0.5, 0.75, 0.9]
SIGMA_SCALE = 1.5


def sigma_for(n):
    """Per-component noise scale: fixed noise-norm to signal ratio across widths."""
    return SIGMA_SCALE / math.sqrt(n)


def scaled_threshold(q, n, sigma):
    """Threshold on the energy ratio as a fraction q of the gap noise -> settled.

    A noise-only stream has energy ratio about 1; a settled estimate of a
    planted pattern has about 1 + n / (1 + 2 sigma^2 n). The grid is therefore
    comparable across widths.
    """
    return 1.0 + q * n / (1.0 + 2.0 * sigma * sigma * n)


def oja_step(v, x, eta):
    """One Oja update of the unit vector v with observation x."""
    y = inner(x, v)
    return normalize([a + eta * b * y for a, b in zip(v, x)])


def energy_ratio(window, n):
    return n * sum(window) / len(window)


def observe(n, pattern, sigma, rng):
    """One observation: a noisy copy of the pattern, or a random vector."""
    return noisy(pattern, sigma, rng) if pattern is not None else rvec(n, rng)


def run_agent(n, pattern, sigma, q, rng, steps=STEPS, check=CHECK, eta=ETA):
    """Stream observations through one agent; return its declared vector or None."""
    th = scaled_threshold(q, n, sigma)
    v = rvec(n, rng)
    win = []
    for t in range(1, steps + 1):
        x = observe(n, pattern, sigma, rng)
        win = (win + [abs(inner(x, v)) ** 2])[-WINDOW:]
        v = oja_step(v, x, eta)
        if t % check == 0 and energy_ratio(win, n) >= th:
            return v[:]
    return None


def agreeing(vec, decls, thresh=AGREE):
    """Number of declared vectors (including vec itself) agreeing with vec."""
    return sum(1 for d in decls if d is not None and fidelity(vec, d) >= thresh)


def system_event(decls, n_agents, thresh=AGREE):
    """Declared vector of the first agent confirmed by >= ceil(N/2) agents, else None."""
    need = math.ceil(n_agents / 2)
    for a in decls:
        if a is not None and agreeing(a, decls, thresh) >= need:
            return a
    return None


def is_true_event(vec, pattern, truth=TRUTH):
    """True when the episode is planted and vec matches the pattern."""
    return vec is not None and pattern is not None and fidelity(vec, pattern) >= truth
