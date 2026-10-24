"""SYNTHETIC classical simulation: reward schemes between agents vs crystallization events.

Reuses membrane_crystallization (v0.1.0): noisy(), normalize(), inner(), fidelity().
That repo defines "crystallization" only as a descriptive label for a streaming
covariance estimate settling; it has no discrete event. The event here is a
harness-level definition (documented in README.md):
  agent tracks its top component v with Oja's rule (Oja 1982; rank-1 form of the
  repo's Hebbian EMA) and declares an event at the first checkpoint where the
  explained-energy ratio  n * mean_{last 5 obs} |<x|v>|^2  (top eigenvalue over
  the flat-spectrum level 1/n, the analog of the repo's entropy drop) reaches its
  chosen threshold. (Self-stability |<v_t,v_{t-10}>|^2 was tried first and dropped:
  pure-noise streams look MORE stable than patterned ones.) System event = >= ceil(N/2) agents declared with
  mutually aligned vectors (pairwise fidelity >= 0.4 with the majority anchor).
Ground truth: half of episodes contain a planted pattern, half are pure noise.
TRUE event = pattern episode and system vector fidelity to the pattern >= 0.5.
FALSE event = any other system event (noise episode, or wrong direction).
"""
import csv, math, random, statistics, sys
from multiprocessing import Pool
sys.path.insert(0, "/workspace/n8/membrane-crystallization")
from membrane_crystallization.stream import noisy
from membrane_crystallization.linalg import normalize, inner, fidelity

THRESH = [1.2, 1.5, 2.5, 4.0, 8.0]
STEPS, CHECK, EPISODES, ETA, EPS = 100, 10, 30, 0.5, 0.15
SCHEMES = ["individual", "team", "difference", "consensus", "random"]

def rvec(n, rng):
    return normalize([complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)])

def oja(v, x, eta):
    y = inner(x, v)  # <x|v>
    v = [a + eta * b * y for a, b in zip(v, x)]
    return normalize(v)

def run_agent(n, pattern, sigma, th, rng):
    v = rvec(n, rng); win = []
    for t in range(1, STEPS + 1):
        x = noisy(pattern, sigma, rng) if pattern else rvec(n, rng)
        win = (win + [abs(inner(x, v)) ** 2])[-5:]
        v = oja(v, x, ETA)
        if t % CHECK == 0 and n * sum(win) / len(win) >= th:
            return v[:]
    return None

def system_event(decls, N):
    need = math.ceil(N / 2)
    ds = [d for d in decls if d is not None]
    for a in ds:
        grp = [d for d in ds if fidelity(a, d) >= 0.4]
        if len(grp) >= need:
            return a
    return None

def one(args):
    scheme, n, N, seed = args
    rng = random.Random(seed * 7919 + n * 31 + N)  # deterministic, scheme-independent stream
    pol = random.Random(seed * 104729 + SCHEMES.index(scheme))
    Q = [[1.0] * len(THRESH) for _ in range(N)]; C = [[0] * len(THRESH) for _ in range(N)]
    sigma = 1.5 / math.sqrt(n)  # fixed noise-norm/signal ratio across widths
    tp = fp = agent_decl = 0
    for ep in range(EPISODES):
        planted = ep % 2 == 0
        pattern = rvec(n, rng) if planted else None
        choice = []
        for i in range(N):
            if scheme == "random" or pol.random() < EPS:
                choice.append(pol.randrange(len(THRESH)))
            else:
                choice.append(max(range(len(THRESH)), key=lambda j: (Q[i][j], pol.random())))
        decls = [run_agent(n, pattern, sigma, THRESH[c], rng) for c in choice]
        agent_decl += sum(d is not None for d in decls)
        ev = system_event(decls, N)
        if ev is not None:
            if planted and fidelity(ev, pattern) >= 0.5: tp += 1
            else: fp += 1
        G = 1.0 if ev is not None else 0.0
        for i in range(N):
            if scheme == "individual": r = 1.0 if decls[i] is not None else 0.0
            elif scheme == "team": r = G
            elif scheme == "difference":
                r = G - (1.0 if system_event(decls[:i] + [None] + decls[i+1:], N) is not None else 0.0)
            elif scheme == "consensus":
                k = math.ceil(2 * N / 3); d = decls[i]
                r = 1.0 if d is not None and sum(1 for e in decls if e is not None and fidelity(d, e) >= 0.4) >= k else 0.0
            else: r = 0.0
            c = choice[i]; C[i][c] += 1; Q[i][c] += (r - Q[i][c]) / (C[i][c] + 1)  # optimistic init Q=1 counts as one pseudo-sample
    half = EPISODES // 2
    return dict(scheme=scheme, width=n, N=N, seed=seed, true_events=tp, false_events=fp,
                precision=(tp / (tp + fp)) if tp + fp else float("nan"),
                recall=tp / half, false_rate=fp / half, agent_declarations=agent_decl)

if __name__ == "__main__":
    widths = [int(w) for w in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64, 128, 256, 512]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(s, n, N, sd) for n in widths for N in (3, 5, 9) for s in SCHEMES for sd in range(seeds)]
    with Pool() as p: rows = p.map(one, jobs, chunksize=1)
    with open("results_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = []
    for n in widths:
        for N in (3, 5, 9):
            for s in SCHEMES:
                g = [r for r in rows if r["width"] == n and r["N"] == N and r["scheme"] == s]
                def ms(k):
                    v = [r[k] for r in g if not math.isnan(r[k])]
                    return (statistics.mean(v), statistics.stdev(v)) if len(v) > 1 else (float("nan"),) * 2
                row = dict(scheme=s, width=n, N=N, seeds=len(g))
                for k in ("true_events", "false_events", "precision", "agent_declarations"):
                    m, sd = ms(k); row[k + "_mean"] = round(m, 3); row[k + "_std"] = round(sd, 3)
                agg.append(row)
    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    for r in agg:
        print(f"{r['width']:>4} N={r['N']} {r['scheme']:<11} TP {r['true_events_mean']:5.2f}±{r['true_events_std']:4.2f}  FP {r['false_events_mean']:5.2f}±{r['false_events_std']:4.2f}  P {r['precision_mean']:.3f}  decl {r['agent_declarations_mean']:.1f}")
