"""Experiment 1: reward schemes between N agents (SYNTHETIC).

Each agent picks a threshold fraction q from Q_GRID with an epsilon-greedy
bandit (optimistic start) and is rewarded under one scheme. Half of the
episodes plant a pattern, half are pure noise; every system event is labelled
true or false against that ground truth.
"""

from .events import Q_GRID, is_true_event, run_agent, sigma_for, system_event
from .helpers import make_rng, rvec
from .rewards import SCHEMES, reward
from .stats import map_jobs, mean_std

EPISODES, EPS = 30, 0.15


def one(args):
    """One (scheme, width, N, seed) run -> a result row."""
    scheme, n, n_agents, seed = args[:4]
    episodes = args[4] if len(args) > 4 else EPISODES
    steps = args[5] if len(args) > 5 else 100
    rng = make_rng(seed * 7919 + n * 31 + n_agents)  # same stream for every scheme
    pol = make_rng(seed * 104729 + SCHEMES.index(scheme))
    k = len(Q_GRID)
    Q = [[1.0] * k for _ in range(n_agents)]
    C = [[0] * k for _ in range(n_agents)]
    sigma = sigma_for(n)
    tp = fp = agent_decl = 0
    for ep in range(episodes):
        planted = ep % 2 == 0
        pattern = rvec(n, rng) if planted else None
        choice = []
        for i in range(n_agents):
            if scheme == "random" or pol.random() < EPS:
                choice.append(pol.randrange(k))
            else:
                choice.append(max(range(k), key=lambda j: (Q[i][j], pol.random())))
        decls = [run_agent(n, pattern, sigma, Q_GRID[c], rng, steps=steps) for c in choice]
        agent_decl += sum(d is not None for d in decls)
        ev = system_event(decls, n_agents)
        if ev is not None:
            if is_true_event(ev, pattern):
                tp += 1
            else:
                fp += 1
        for i in range(n_agents):
            r = reward(scheme, i, decls, n_agents)
            c = choice[i]
            C[i][c] += 1
            # optimistic start Q = 1 counts as one pseudo-sample
            Q[i][c] += (r - Q[i][c]) / (C[i][c] + 1)
    half = (episodes + 1) // 2
    return dict(scheme=scheme, width=n, N=n_agents, seed=seed, true_events=tp, false_events=fp,
                precision=(tp / (tp + fp)) if tp + fp else float("nan"),
                recall=tp / half, false_rate=fp / max(1, episodes - half),
                agent_declarations=agent_decl)


def summarize(rows, widths, agent_counts, schemes=SCHEMES):
    agg = []
    for n in widths:
        for na in agent_counts:
            for s in schemes:
                g = [r for r in rows if r["width"] == n and r["N"] == na and r["scheme"] == s]
                if not g:
                    continue
                row = dict(scheme=s, width=n, N=na, seeds=len(g))
                for k in ("true_events", "false_events", "precision", "agent_declarations"):
                    m, sd = mean_std([r[k] for r in g])
                    row[k + "_mean"] = round(m, 3)
                    row[k + "_std"] = round(sd, 3)
                agg.append(row)
    return agg


def format_row(r):
    return (f"{r['width']:>4} N={r['N']} {r['scheme']:<11} "
            f"TP {r['true_events_mean']:5.2f}±{r['true_events_std']:4.2f}  "
            f"FP {r['false_events_mean']:5.2f}±{r['false_events_std']:4.2f}  "
            f"P {r['precision_mean']:.3f}  decl {r['agent_declarations_mean']:.1f}")


def run_sweep(widths, agent_counts, seeds, n_jobs=1, episodes=EPISODES, steps=100, sink=None):
    jobs = [(s, n, na, sd, episodes, steps)
            for n in widths for na in agent_counts for s in SCHEMES for sd in range(seeds)]
    rows = map_jobs(one, jobs, n_jobs, sink)
    return rows, summarize(rows, widths, agent_counts)
