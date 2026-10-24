"""Experiment 2: the crystallization contest of 3 coupled membranes (SYNTHETIC).

Three membranes stream observations, each with parameters th (threshold
fraction), c (coupling to the other two) and g (gain on its perceived energy
ratio). Self-knowledge: each membrane keeps an exponential average `sp` of how
often its own declarations were confirmed by a peer (observable, no ground
truth); its effective threshold fraction is th * (2 - sp).

Score per generation: w * joint_true + (1 - w) * rank bonus - fcost * false
events per episode, where joint_true counts episodes in which all three
membranes were true, and the rank bonus goes to the membrane with the most true
events (ties split). Between generations each membrane mutates (th, c, g) and
keeps the mutation if its score does not fall ((1+1)-ES, fixed seeds).

Final stage, in this order, each switchable:
  S shield      keep only events confirmed by >= 2 of 3 membranes (fail closed)
  A amplifier   next episode warm-starts from the last output vector, weight capped
  Z stabilizer  leaky-integrator feedback damps the gain; gain is clamped
"""

import math
import statistics

from .events import (AGREE, CHECK, ETA, WINDOW, agreeing, energy_ratio, is_true_event,
                     observe, scaled_threshold, sigma_for)
from .helpers import fidelity, inner, make_rng, normalize, rvec
from .stats import map_jobs, mean_std

EP, GENS = 8, 12
AMP_CAP, TARGET, LEAK, KSTAB = 0.5, 0.5, 0.3, 1.5
GAIN_MIN, GAIN_MAX, FACTOR_MIN = 0.33, 3.0, 0.2
TH_MIN, TH_MAX, C_MAX = 0.02, 1.5, 0.5
CFGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]


def cfg_tag(cfg):
    return "".join("SAZ"[k] if cfg[k] else "-" for k in range(3))


CFG_BY_TAG = {cfg_tag(c): c for c in CFGS}


def shield_select(decls, thresh=AGREE, need=2):
    """Index of the first declared membrane confirmed by >= `need` of 3 (itself included).

    Fail closed: returns None (drop the event) when no membrane is confirmed.
    """
    for i, d in enumerate(decls):
        if d is not None and agreeing(d, decls, thresh) >= need:
            return i
    return None


def warm_start(v, warm, amp):
    """Blend a fresh random start v toward the last output vector, weight capped at AMP_CAP."""
    amp = min(max(amp, 0.0), AMP_CAP)
    if warm is None or amp == 0.0:
        return v
    return normalize([(1 - amp) * a + amp * b for a, b in zip(v, warm)])


def stabilizer_factor(rate_ema, target=TARGET, kstab=KSTAB):
    """Multiplier on the gain: shrinks when the output event rate runs above target."""
    return max(FACTOR_MIN, 1 - kstab * (rate_ema - target))


def leaky_update(rate_ema, ev_rate, leak=LEAK):
    """Leaky integrator of the output event rate."""
    return (1 - leak) * rate_ema + leak * ev_rate


def clamp_gain(g):
    return min(GAIN_MAX, max(GAIN_MIN, g))


def episode(n, pattern, sigma, P, warm, amp, rng, steps=60):
    """One episode for the 3 coupled membranes; returns their declared vectors."""
    vs = [warm_start(rvec(n, rng), warm, amp) for _ in range(3)]
    win = [[] for _ in range(3)]
    decl = [None] * 3
    for t in range(1, steps + 1):
        for i in range(3):
            if decl[i] is not None:
                continue
            x = observe(n, pattern, sigma, rng)
            win[i] = (win[i] + [abs(inner(x, vs[i])) ** 2])[-WINDOW:]
            v = [a + ETA * b * inner(x, vs[i]) for a, b in zip(vs[i], x)]
            c = P[i]["c"]
            for j in range(3):
                if j != i and c > 0:
                    ph = inner(vs[j], v)
                    ph = ph / abs(ph) if abs(ph) > 0 else 1
                    v = [a + c * ph * b for a, b in zip(v, vs[j])]
            vs[i] = normalize(v)
        if t % CHECK == 0:
            for i in range(3):
                if decl[i] is None:
                    q = min(TH_MAX, P[i]["th"] * (2 - P[i]["sp"]))
                    if P[i]["geff"] * energy_ratio(win[i], n) >= scaled_threshold(q, n, sigma):
                        decl[i] = vs[i][:]
    return decl


def run(args):
    """One (width, w, cfg, seed) contest -> a result row.

    args = (n, w, cfg, seed[, opts]) with opts a dict overriding gens, ep, steps,
    init_g, gsig, fcost.
    """
    n, w, cfg, seed = args[:4]
    o = args[4] if len(args) > 4 else {}
    gens, eps_ep, steps = o.get("gens", GENS), o.get("ep", EP), o.get("steps", 60)
    init_g, gsig, fcost = o.get("init_g", 1.0), o.get("gsig", 0.2), o.get("fcost", 0.0)
    shield, amp_on, stab = cfg
    rng = make_rng(seed * 7919 + n)
    mrng = make_rng(seed * 104729 + 1)
    sigma = sigma_for(n)
    par = [dict(th=0.5, c=0.05, g=init_g, sp=0.5, geff=1.0) for _ in range(3)]
    best = [-1.0] * 3
    rate_ema = TARGET
    hist = []
    warm = None
    for gen in range(gens):
        cand = [dict(p) for p in par]
        if gen > 0:
            for p in cand:  # (1+1)-ES mutation
                p["th"] = min(TH_MAX, max(TH_MIN, p["th"] * math.exp(mrng.gauss(0, 0.2))))
                p["c"] = min(C_MAX, max(0.0, p["c"] + mrng.gauss(0, 0.05)))
                p["g"] = p["g"] * math.exp(mrng.gauss(0, gsig))
                if stab:
                    p["g"] = clamp_gain(p["g"])
        fp = [0] * 3
        tp = [0] * 3
        joint = 0
        out_t = out_f = 0
        for e in range(eps_ep):
            planted = e % 2 == 0
            pat = rvec(n, rng) if planted else None
            for p in cand:
                p["geff"] = p["g"] * (stabilizer_factor(rate_ema) if stab else 1.0)
            d = episode(n, pat, sigma, cand, warm, AMP_CAP if amp_on else 0.0, rng, steps)
            truth = [is_true_event(di, pat) for di in d]
            for i in range(3):
                tp[i] += truth[i]
                fp[i] += (d[i] is not None and not truth[i])
                if d[i] is not None:
                    conf = any(d[j] is not None and fidelity(d[i], d[j]) >= AGREE
                               for j in range(3) if j != i)
                    cand[i]["sp"] = 0.8 * cand[i]["sp"] + 0.2 * conf
            joint += all(truth)
            if shield:
                k = shield_select(d)
                outs = [] if k is None else [k]
            else:
                outs = [i for i in range(3) if d[i] is not None]
            for i in outs:
                if truth[i]:
                    out_t += 1
                else:
                    out_f += 1
            warm = d[outs[0]] if outs else None
            ev_rate = len(outs) / (1 if shield else 3)
            rate_ema = leaky_update(rate_ema, ev_rate)
        m = max(tp)
        top = [i for i in range(3) if tp[i] == m]
        for i in range(3):
            s = (w * joint + (1 - w) * (1.0 / len(top) if i in top and m > 0 else 0.0)
                 - fcost * fp[i] / eps_ep)
            if s >= best[i]:
                best[i] = s
                par[i] = cand[i]
        hist.append((out_t, out_f, max(p["g"] for p in par)))
    T = sum(h[0] for h in hist)
    F = sum(h[1] for h in hist)
    rates = [(h[0] + h[1]) / eps_ep for h in hist]
    return dict(width=n, w=w, cfg=cfg_tag(cfg), seed=seed,
                true_events=T, false_events=F, precision=T / (T + F) if T + F else float("nan"),
                rate_var=statistics.pvariance(rates), max_gain=max(h[2] for h in hist),
                final_gain=hist[-1][2])


def summarize(rows, widths, ws, cfgs, runaway_gain=10.0):
    agg = []
    for n in widths:
        for w in ws:
            for c in cfgs:
                tag = cfg_tag(c)
                g = [r for r in rows if r["width"] == n and r["w"] == w and r["cfg"] == tag]
                if not g:
                    continue
                row = dict(width=n, w=w, cfg=tag, seeds=len(g))
                for k in ("true_events", "false_events", "precision", "rate_var", "max_gain"):
                    m, sd = mean_std([r[k] for r in g])
                    row[k + "_mean"] = round(m, 3)
                    row[k + "_std"] = round(sd, 3)
                row["runaway_frac"] = round(sum(r["max_gain"] > runaway_gain for r in g) / len(g), 2)
                agg.append(row)
    return agg


def format_row(r):
    return (f"{r['width']:>4} w={r['w']:.2f} {r['cfg']} "
            f"TP {r['true_events_mean']:6.2f}±{r['true_events_std']:5.2f} "
            f"FP {r['false_events_mean']:6.2f}±{r['false_events_std']:5.2f} "
            f"P {r['precision_mean']:.3f} rateVar {r['rate_var_mean']:.3f} "
            f"maxG {r['max_gain_mean']:.2f} runaway {r['runaway_frac']}")


def run_contest(widths, ws, cfgs, seeds, n_jobs=1, opts=None, sink=None):
    jobs = [(n, w, c, s, dict(opts or {})) for n in widths for w in ws for c in cfgs
            for s in range(seeds)]
    rows = map_jobs(run, jobs, n_jobs, sink)
    return rows, summarize(rows, widths, ws, cfgs)
