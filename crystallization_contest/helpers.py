"""Minimal vector helpers.

Vendored from membrane-crystallization v0.1.0 (Jean-François Brisson,
AGPL-3.0-only, DOI 10.5281/zenodo.23583050): normalize, inner, fidelity and
noisy keep the same definitions and are reduced to what this package needs,
so the package stays standard-library only.
"""

import cmath
import math
import random


def _check_vector(v, name="vector"):
    v = [complex(c) for c in v]
    if not v:
        raise ValueError(f"{name} is empty")
    if not all(cmath.isfinite(c) for c in v):
        raise ValueError(f"{name} has non-finite entries")
    return v


def norm(v):
    return math.sqrt(sum(abs(c) ** 2 for c in v))


def normalize(v):
    v = _check_vector(v)
    n = norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize a zero-norm vector")
    return [c / n for c in v]


def inner(a, b):
    """<a|b>."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x.conjugate() * y for x, y in zip(a, b))


def fidelity(a, b):
    """|<a|b>|^2 for unit vectors."""
    return abs(inner(a, b)) ** 2


def noisy(state, sigma, rng, scales=None):
    """state + complex Gaussian noise, renormalized."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if scales is None:
        scales = [1.0] * len(state)
    elif len(scales) != len(state):
        raise ValueError("scales length mismatch")
    return normalize([c + s * complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c, s in zip(state, scales)])


def rvec(n, rng):
    """Random unit vector in C^n (complex Gaussian, normalized)."""
    return normalize([complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)])


def make_rng(seed):
    return random.Random(seed)
