"""SYNTHETIC 'crystallization contest': 3 coupled self-knowing membranes, (1+1)-ES.

Reuses membrane_crystallization.noisy/normalize/inner/fidelity and the event
definition from sweep.py (explained-energy ratio crossing a threshold).
Each membrane i has params: th (threshold), c (coupling to peers), g (gain on
its perceived energy ratio). Self-knowledge: an EMA `selfprec` of how often its
own declarations were confirmed by >=1 peer (observable, no ground truth);
effective threshold = th * (2 - selfprec).
Score of a generation = w * joint_true (shared, all three) + (1-w) * rank bonus
(1 for the membrane with most true events, ties split). False events cost
nothing in the score -> room for gaming; that is measured, not assumed away.
Final stage (toggles): shield (output only events confirmed by >=2 of 3),
amplifier (warm-start next episode from last OUTPUT vector, weight <= cap),
stabilizer (leaky-integrator on output event rate damps gain; gain clamped).
"""
import csv, math, random, statistics, sys
from multiprocessing import Pool
sys.path.insert(0, "/workspace/n8/membrane-crystallization")
from membrane_crystallization.stream import noisy
from membrane_crystallization.linalg import normalize, inner, fidelity
from sweep import rvec

STEPS, CHECK, EP, GENS, ETA = 60, 10, 8, 12, 0.5
AMP_CAP, TARGET, LEAK, KSTAB = 0.5, 0.5, 0.3, 1.5

def episode(n, pattern, sigma, P, warm, amp, rng):
    vs = []
    for _ in range(3):
        v = rvec(n, rng)
        if warm is not None and amp > 0:
            v = normalize([(1 - amp) * a + amp * b for a, b in zip(v, warm)])
        vs.append(v)
    win = [[] for _ in range(3)]; decl = [None] * 3
    for t in range(1, STEPS + 1):
        for i in range(3):
            if decl[i] is not None: continue
            x = noisy(pattern, sigma, rng) if pattern else rvec(n, rng)
            win[i] = (win[i] + [abs(inner(x, vs[i])) ** 2])[-5:]
            v = [a + ETA * b * inner(x, vs[i]) for a, b in zip(vs[i], x)]
            c = P[i]["c"]
            for j in range(3):
                if j != i and c > 0:
                    ph = inner(vs[j], v); ph = ph / abs(ph) if abs(ph) > 0 else 1
                    v = [a + c * ph * b for a, b in zip(v, vs[j])]
            vs[i] = normalize(v)
        if t % CHECK == 0:
            for i in range(3):
                if decl[i] is None and P[i]["geff"] * n * sum(win[i]) / len(win[i]) >= P[i]["th"] * (2 - P[i]["sp"]):
                    decl[i] = vs[i][:]
    return decl

def run(args):
    n, w, cfg, seed = args
    shield, amp_on, stab = cfg
    rng = random.Random(seed * 7919 + n)
    mrng = random.Random(seed * 104729 + 1)
    sigma = 1.5 / math.sqrt(n)
    par = [dict(th=4.0, c=0.05, g=1.0, sp=0.5, geff=1.0) for _ in range(3)]
    best = [-1.0] * 3; rate_ema = TARGET
    hist = []
    warm = None
    for gen in range(GENS):
        cand = [dict(p) for p in par]
        if gen > 0:
            for p in cand:  # (1+1)-ES mutation
                p["th"] = max(1.0, p["th"] * math.exp(mrng.gauss(0, 0.2)))
                p["c"] = min(0.5, max(0.0, p["c"] + mrng.gauss(0, 0.05)))
                p["g"] = p["g"] * math.exp(mrng.gauss(0, 0.2))
                if stab: p["g"] = min(3.0, max(0.33, p["g"]))
        tp = [0] * 3; joint = 0; out_t = out_f = 0
        for ep in range(EP):
            planted = ep % 2 == 0
            pat = rvec(n, rng) if planted else None
            for p in cand:
                p["geff"] = p["g"] * (max(0.2, 1 - KSTAB * (rate_ema - TARGET)) if stab else 1.0)
            d = episode(n, pat, sigma, cand, warm, AMP_CAP if amp_on else 0.0, rng)
            truth = [di is not None and planted and fidelity(di, pat) >= 0.5 for di in d]
            for i in range(3):
                tp[i] += truth[i]
                if d[i] is not None:
                    conf = any(d[j] is not None and fidelity(d[i], d[j]) >= 0.4 for j in range(3) if j != i)
                    cand[i]["sp"] = 0.8 * cand[i]["sp"] + 0.2 * conf
            joint += all(truth)
            outs = []
            if shield:
                for i in range(3):
                    if d[i] is not None and sum(1 for j in range(3) if d[j] is not None and fidelity(d[i], d[j]) >= 0.4) >= 2:
                        outs.append(i); break  # one confirmed system event
            else:
                outs = [i for i in range(3) if d[i] is not None]
            for i in outs:
                if truth[i]: out_t += 1
                else: out_f += 1
            warm = d[outs[0]] if outs else None
            ev_rate = len(outs) / (1 if shield else 3)
            rate_ema = (1 - LEAK) * rate_ema + LEAK * ev_rate
        m = max(tp); top = [i for i in range(3) if tp[i] == m]
        for i in range(3):
            s = w * joint + (1 - w) * (1.0 / len(top) if i in top and m > 0 else 0.0)
            if s >= best[i]:
                best[i] = s; par[i] = cand[i]
        hist.append((out_t, out_f, max(p["g"] for p in par)))
    T = sum(h[0] for h in hist); F = sum(h[1] for h in hist)
    rates = [(h[0] + h[1]) / EP for h in hist]
    return dict(width=n, w=w, cfg="".join("SAZ"[k] if cfg[k] else "-" for k in range(3)), seed=seed,
                true_events=T, false_events=F, precision=T / (T + F) if T + F else float("nan"),
                rate_var=statistics.pvariance(rates), max_gain=max(h[2] for h in hist),
                final_gain=hist[-1][2])

CFGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]
if __name__ == "__main__":
    widths = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64, 128]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(n, w, c, s) for n in widths for w in (0.0, 0.25, 0.5, 0.75, 1.0) for c in CFGS for s in range(seeds)]
    with Pool() as p: rows = p.map(run, jobs, chunksize=2)
    with open("contest_raw.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    agg = []
    for n in widths:
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            for c in CFGS:
                tag = "".join("SAZ"[k] if c[k] else "-" for k in range(3))
                g = [r for r in rows if r["width"] == n and r["w"] == w and r["cfg"] == tag]
                row = dict(width=n, w=w, cfg=tag, seeds=len(g))
                for k in ("true_events", "false_events", "precision", "rate_var", "max_gain"):
                    v = [r[k] for r in g if not math.isnan(r[k])]
                    row[k + "_mean"] = round(statistics.mean(v), 3); row[k + "_std"] = round(statistics.stdev(v), 3)
                row["runaway_frac"] = round(sum(r["max_gain"] > 5 for r in g) / len(g), 2)
                agg.append(row)
    with open("contest_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(agg[0])); wr.writeheader(); wr.writerows(agg)
    for r in agg:
        print(f"{r['width']:>4} w={r['w']:.2f} {r['cfg']} TP {r['true_events_mean']:6.2f}±{r['true_events_std']:5.2f} FP {r['false_events_mean']:6.2f}±{r['false_events_std']:5.2f} P {r['precision_mean']:.3f} rateVar {r['rate_var_mean']:.3f} maxG {r['max_gain_mean']:.2f} runaway {r['runaway_frac']}")
