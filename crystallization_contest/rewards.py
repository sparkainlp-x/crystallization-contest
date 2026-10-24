"""Reward schemes between agents (SYNTHETIC).

Each function returns the reward for agent i given the list of declared vectors
(None = no declaration). Schemes: individual, team (shared), difference
(marginal contribution to the team), consensus (k-of-N agreement) and random
(no reward, the anti-gaming baseline).
"""

import math

from .events import AGREE, agreeing, system_event

SCHEMES = ["individual", "team", "difference", "consensus", "random"]


def consensus_k(n_agents):
    return math.ceil(2 * n_agents / 3)


def reward(scheme, i, decls, n_agents, thresh=AGREE):
    """Reward of agent i under `scheme`."""
    if scheme == "individual":
        return 1.0 if decls[i] is not None else 0.0
    if scheme in ("team", "difference"):
        team = 1.0 if system_event(decls, n_agents, thresh) is not None else 0.0
    if scheme == "team":
        return team
    if scheme == "difference":
        without = decls[:i] + [None] + decls[i + 1:]
        return team - (1.0 if system_event(without, n_agents, thresh) is not None else 0.0)
    if scheme == "consensus":
        d = decls[i]
        return 1.0 if d is not None and agreeing(d, decls, thresh) >= consensus_k(n_agents) else 0.0
    if scheme == "random":
        return 0.0
    raise ValueError(f"unknown scheme {scheme!r}")
