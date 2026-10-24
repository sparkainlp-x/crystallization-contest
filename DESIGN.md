# Design notes

SYNTHETIC. Classical numerical experiments on seeded synthetic data only.

## The event

membrane-crystallization v0.1.0 uses "crystallization" as a descriptive label for a streaming covariance estimate settling. It defines no discrete event, so this package defines one and keeps it small:

- Per agent: Oja's rule `v <- normalize(v + eta <x|v> x)` (the rank-1 form of the Hebbian running estimate) and the explained-energy ratio `n * mean(|<x|v>|^2)` over the last 5 observations. For a noise-only stream the ratio is about 1; for a settled estimate of a planted pattern it is about `1 + n / (1 + 2 sigma^2 n)`. An agent declares at the first checkpoint (every 10 steps) where the ratio reaches its threshold.
- Per system: at least `ceil(N/2)` declared vectors agreeing at fidelity >= 0.4 give a system event.
- Ground truth: planted episodes (observations are noisy copies of one pattern) alternate with pure-noise episodes. A system event is true when the episode is planted and the vector has fidelity >= 0.5 to the pattern; all other events are false. Counting false events is the guard against a reward scheme that only inflates the count.

An earlier candidate signal, the self-stability `|<v_t, v_{t-10}>|^2` of the estimate, was dropped: pure-noise streams look more stable than patterned ones.

## Width-scaled thresholds

The first run used a fixed grid on the energy ratio. The ratio of a settled estimate grows about linearly with n, so at n >= 128 every fixed threshold was crossed before the estimate had settled, and true events collapsed for every scheme. That was a harness artifact. The grid is now a fraction q of the gap between the noise level and the settled level, `1 + q n / (1 + 2 sigma^2 n)`, with sigma = 1.5 / sqrt(n) so the noise-to-signal norm ratio is the same at every width. The settled level is an analytic estimate that was not derived carefully.

## Reward schemes

All schemes use the same epsilon-greedy bandit (epsilon 0.15, optimistic start of 1 counted as one pseudo-sample, random tie-break) over the five threshold fractions, with the same observation stream for each scheme at a given seed, so schemes differ only in the reward:

- individual: 1 if my own declaration was made;
- team: 1 if a system event occurred;
- difference: team reward minus the team reward with my declaration removed;
- consensus: 1 if my vector is confirmed by at least `ceil(2N/3)` declared vectors;
- random: 0 always (no learning signal).

A first run had a tie-break that always chose the lowest threshold, so no scheme produced events; it is kept as `spec/results_*_v1_tiebias*`.

## The contest

Three membranes, parameters `th` (threshold fraction), `c` (coupling) and `g` (gain). Coupling: after each Oja step, membrane i adds `c` times a phase-aligned copy of each peer's current vector. Self-knowledge: `sp` is an exponential average of how often the membrane's declarations were confirmed by a peer; the effective threshold fraction is `th * (2 - sp)`. Self-evolution: (1+1)-ES, a log-normal mutation of `th` and `g`, an additive one of `c`, kept when the score does not fall.

Score per generation and membrane: `w * joint_true + (1 - w) * rank_bonus - fcost * false_events / episodes`, where `joint_true` counts episodes in which all three membranes were true and `rank_bonus` is 1 for the membrane with the most true events (ties split).

Final stage, in order:

1. **Shield**: output only an event confirmed by at least 2 of 3 membranes (itself included); otherwise output nothing (fail closed). At most one output event per episode.
2. **Amplifier**: the next episode's random starting vectors are blended toward the last output vector; the weight is capped at 0.5.
3. **Stabilizer**: a leaky integrator of the output event rate (leak 0.3, target 0.5) scales the gain by `max(0.2, 1 - 1.5 (rate - 0.5))`, and the gain parameter is clamped to [0.33, 3].

Design consequences worth knowing:

- The evolutionary score uses each membrane's raw events. The shield and amplifier act on the output stage only, so they do not change what the hill-climb sees, apart from the amplifier's effect on the next episode's starting vectors.
- Without the shield, up to 3 events per episode are counted (one per membrane); with it at most 1. Totals are not comparable between those rows; precision and rates are.
- The stabilizer has two parts, the leaky-integrator damping and the hard clamp. They are not ablated separately, and under stress the clamp most likely does most of the work.

## Reproducibility

Every job seeds its generators from `(seed, width, ...)`. The package reproduces rows stored in `results/` exactly; two tests check one sweep row and one contest row. Parallel runs return rows in job order, so output does not depend on scheduling. Raw rows are written as they arrive when `--out` is given.

## What is not covered

Other noise levels or pattern counts, other Oja step sizes, a threshold formula derived rather than estimated, charging false events at costs other than 2, and contest runs at widths 256 and 512 beyond w = 0.5.
