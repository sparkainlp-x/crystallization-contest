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

## Legacy protocol audit and v2 migration

The package's default `contest.run` is the historical seeded experiment. The
fixed-threshold code and its artifacts in `spec/v1_fixed/` are a separate older
archive. Do not silently substitute the scoring/evaluation core in
`crystallization_contest.evaluation_v2`; it is opt-in and does not change legacy
results.

The exact historical selection score for membrane `i` is
`w * joint_true + (1-w) * rank_bonus_i - fcost * raw_false_i / episodes`.
`joint_true` is an integer episode count (not divided by `episodes`);
`rank_bonus_i` splits one unit among positive true-event leaders, while an
all-zero true-event generation gives no rank bonus to anyone. `raw_false_i` is
the membrane's incorrect-declaration count before shielding. The `joint_true`
term can be as large as the episode count, but the rank bonus is at most one,
so `w=0.5` does not mean equal numerical weighting. The configured false-event
penalty has no fixed bound. This differs from
`normalized_cooperative_score`, which uses bounded `joint_true/episodes` and
bounded `B_i` base components but retains the unbounded configured
`fcost*(F_i/episodes)` penalty; no superiority or `[0,1]` score guarantee is
claimed.

The contest's generation-wide mutation proposes a changed parameter set for
all three membranes, then calculates true and false counts from the synthetic
ground truth and accepts each membrane independently. Acceptance compares with
`best[i]` accumulated from previous generations, not the current incumbent
scored on this generation's exact episodes. The stored memories can therefore
refer to different episode sets, and independent acceptance can assemble a
hybrid team that was never scored as a team. The `sp` EMA is updated when a
membrane declares based on peer agreement and does not itself consult ground
truth, but the true/false counts used for selection do. The shield filters
outputs after the declarations and cannot change the raw counts used for
selection; without it up to three declarations are counted per episode, and
with it at most one output is counted. `warm` and `rate_ema` are global state
and advance during candidate evaluation even when that candidate's parameter
is rejected. The historical protocol does not restore them on rejection.

Experiment 1's `rewards.reward` functions consume declaration presence,
consensus, or the system-event rule; they receive no ground-truth labels.
`sweep.one` uses ground truth to label measured system outcomes, not as input to
those reward functions.

The v2 core instead compares complete teams on one immutable, pre-materialized
challenge bank. Each episode's observation stimuli are provided as immutable
values. The runner receives the explicit episode index and an episode-bound
stream accessor; duplicate stimuli do not require inferring which episode is
running. Indexed random streams are keyed by seed partition, episode, step,
agent, and channel, so extra random calls in a branch cannot shift another
indexed stream. The runner gets stimuli and streams, not bank labels. Incumbent
and candidate each receive independent copies of the same initial state,
including caller-managed warm vectors, rate EMAs, and self-confirmation state.
Their evaluation state is discarded after scoring. An immutable `ScoringContext`
provides each objective call with both arms' attributed search/proposal attempts
(including failed attempts), search/proposal operation counts, and measured
evaluation operations. A cost-aware objective can charge search and evaluation
compute plus per-attempt costs; avoid double counting work in both channels.
Common validation overhead is reported separately and excluded from deployable
cost. One finite scalar objective is declared before the comparison; a strict
win accepts the whole evaluated candidate team, while a tie keeps the
incumbent. `paired_team_evaluation` rejects held-out banks for selection.
Do not use independently accepted membrane parameters as a hybrid team.
Initialize the next trial from the caller's documented reset policy rather than
carrying evaluation-derived state implicitly.

Held-out output is available only through `report_frozen_team`, which evaluates
one copy of a frozen team and returns metrics without returning a winner or
promoted team. This API structure does not create a security sandbox:
Python runners, objectives, closures, and globals can still access any state
their authors expose, so they must be trusted and must not feed labels into
action selection, proposal, calibration, or online updates.

The v2 normalized score is `S_i = w*(J/E) + (1-w)*B_i - fcost*(F_i/E)`.
The separate target-direction-detection utility is
`U = b_tp*TP - c_fp*FP - c_fn*FN - c_compute*C - c_attempt*A`, reported as
`U/E` per episode. Its explicitly named endpoint TP/FP/FN/TN are episode-level
target-direction detections, not ordinary target-presence detection; a wrong
direction when a target is present is an FN and is reported separately, not
mixed into classification or localization. No presence-detection or
localization metrics are derived from this matrix. Raw per-agent declarations
and emitted post-filter outputs remain separate; emitted outputs are shielded
counts only when the frozen configuration enables a shield. All utility
coefficients are explicit and require units;
examples are illustrative, not optimal. The operation count must include all
search and evaluation work, and the attempt count must include failed
proposals. Avoid charging the same work in both terms. Timing is recorded in
wall/CPU seconds separately, and common experimental validation overhead is
not deployable search cost.

`ChallengeBank` and its bounded manifest record a development, calibration, or
held-out partition and challenge/configuration hashes. The library boundary
withholds labels from the runner, but orchestration that trains only on
development/calibration labels, freezes the team/configuration before held-out
testing, disables label-dependent updates, and resets label-free adaptation is
not implemented here. Results must be called exploratory unless independent
preregistration exists; no preregistration is represented by this repository.
Scores and rates depend on target prevalence. Comparisons require matched,
declared prevalence and seed partitions. No seed-paired v2 comparisons or
uncertainty summaries have been run, and no held-out performance claim is made.
Proposal search and matched-budget no-intervention, random, independent, and
shared-proposal comparisons are also not implemented because no compatible
proposal interface exists. Do not claim full-linked-system injection or
clamp-only/damping-only ablations.

Migration: keep historical commands and archived CSVs unchanged; use
`python -m crystallization_contest v2` for the illustrative JSON breakdown, or
import `ChallengeBank`, `EpisodeResult`, the scoring functions, and
`paired_team_evaluation` for non-held-out selection. Use `report_frozen_team`
for held-out reporting only. Provide a trusted runner with the explicit episode
index and use `make_cost_aware_utility_objective` (or an equivalent objective)
with all failed proposal attempts and search/evaluation operation counts. Keep
common validation overhead separate and use the accepted complete team only.

## Reproducibility

Every job seeds its generators from `(seed, width, ...)`. The package reproduces rows stored in `results/` exactly; two tests check one sweep row and one contest row. Parallel runs return rows in job order, so output does not depend on scheduling. Raw rows are written as they arrive when `--out` is given.

## What is not covered

Other noise levels or pattern counts, other Oja step sizes, a threshold formula derived rather than estimated, charging false events at costs other than 2, and contest runs at widths 256 and 512 beyond w = 0.5.
