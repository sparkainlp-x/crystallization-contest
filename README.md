# Crystallization Contest

**SYNTHETIC classical simulation.** Every number below is measured in this repo on seeded synthetic data (random complex vectors in C^n with a planted pattern or pure noise). Nothing here is hardware or field data, and nothing is a claim about any physical system.

The package measures how reward schemes between agents change the number of "crystallization events" at widths n = 32, 64, 128, 256 and 512, and runs a contest of three coupled membranes with a shield, an amplifier and a stabilizer. Each event is labelled true or false against the generator's ground truth, because a raw event count can be inflated.

Résumé FR : simulation classique SYNTHÉTIQUE de schémas de récompense entre agents (individuel, équipe, différence, consensus, aléatoire) et d'un concours de trois membranes couplées avec bouclier, amplificateur et stabilisateur. Chaque événement est étiqueté vrai ou faux selon la vérité terrain du générateur. Les résultats et les limites suivent (en anglais).

## What an event is

"Crystallization" is a descriptive label in [membrane-crystallization](https://github.com/sparkainlp-x/membrane-crystallization) v0.1.0 (the estimate settling); that package has no discrete event. This repo defines one for the harness and says so:

- An agent tracks its top component with Oja's rule and declares an event at the first checkpoint (every 10 steps) where the explained-energy ratio `n * mean(|<x|v>|^2)` over the last 5 observations reaches its threshold.
- Thresholds are `1 + q * n / (1 + 2 sigma^2 n)` for q in {0.1, 0.25, 0.5, 0.75, 0.9}, with sigma = 1.5 / sqrt(n): a fraction q of the gap between the noise-only level (about 1) and the settled level, so the grid means the same thing at every width.
- A system event needs at least ceil(N/2) agents whose declared vectors agree (fidelity >= 0.4).
- Half of the episodes plant a pattern, half are pure noise. A system event is **true** if the episode is planted and its vector has fidelity >= 0.5 to the pattern. Every other event is **false**.

## Quick start

```bash
python -m crystallization_contest sweep --quick          # smoke run, under a second
python -m crystallization_contest contest --quick
python -m crystallization_contest stress --quick
python -m crystallization_contest fcost --quick
python -m crystallization_contest sweep --widths 32,64 --agents 3,5 --seeds 5 --jobs 4 --out out/
pip install -r requirements-dev.txt && python -W error -m pytest   # 29 tests
bash scripts/reproduce_full.sh                           # the full 20-seed runs; hours on 8 cores
```

`--quick` is a smoke run; its numbers are not comparable to the results below. Standard library only (Python 3.10 to 3.13).

## Experiment 1: reward schemes

N agents (3, 5 or 9) each choose a threshold fraction q with an epsilon-greedy bandit and receive a reward: **individual** (my own declaration), **team** (shared system event), **difference** (my marginal contribution to the team event), **consensus** (my vector is confirmed by at least ceil(2N/3) agents) or **random** (no reward, the baseline). 20 seeds, 30 episodes per seed (15 planted). True events per seed, mean ± std, best mean in bold:

| width | N | individual | team | difference | consensus | random |
|---|---|---|---|---|---|---|
| 32 | 3 | 6.75 ± 2.25 | 12.35 ± 1.78 | 12.75 ± 1.58 | **13.15 ± 1.14** | 11.50 ± 1.61 |
| 32 | 5 | 8.80 ± 2.33 | 13.50 ± 1.15 | 13.35 ± 0.93 | **14.25 ± 0.79** | 12.65 ± 1.50 |
| 32 | 9 | 9.45 ± 1.96 | 14.25 ± 0.85 | 14.25 ± 0.64 | **14.85 ± 0.37** | 14.25 ± 0.85 |
| 64 | 3 | 9.70 ± 3.47 | 12.50 ± 1.57 | 12.25 ± 1.52 | **13.85 ± 0.88** | 11.20 ± 1.82 |
| 64 | 5 | 11.80 ± 2.04 | 13.50 ± 1.24 | 13.35 ± 1.42 | **14.10 ± 0.79** | 12.50 ± 1.10 |
| 64 | 9 | 13.20 ± 1.64 | 14.15 ± 0.88 | 14.15 ± 0.81 | **14.70 ± 0.57** | 14.20 ± 0.83 |
| 128 | 3 | 10.20 ± 2.65 | 11.80 ± 1.54 | 12.15 ± 1.46 | **12.90 ± 1.48** | 10.35 ± 1.66 |
| 128 | 5 | 11.65 ± 2.66 | 13.10 ± 1.33 | 13.30 ± 1.30 | **13.80 ± 1.06** | 11.95 ± 1.47 |
| 128 | 9 | 13.00 ± 1.81 | 13.60 ± 0.82 | 13.15 ± 1.39 | **14.75 ± 0.55** | 13.50 ± 0.89 |
| 256 | 3 | 9.85 ± 3.01 | 11.75 ± 1.71 | 11.10 ± 2.45 | **12.80 ± 1.32** | 9.95 ± 1.99 |
| 256 | 5 | 11.35 ± 2.52 | 12.80 ± 1.06 | 12.75 ± 1.37 | **13.50 ± 1.32** | 10.90 ± 1.89 |
| 256 | 9 | 12.40 ± 1.79 | 12.95 ± 1.15 | 12.10 ± 1.71 | **14.40 ± 0.68** | 12.85 ± 1.14 |
| 512 | 3 | 9.45 ± 2.74 | 11.45 ± 1.40 | 11.50 ± 1.82 | **12.45 ± 1.91** | 8.40 ± 2.11 |
| 512 | 5 | 10.45 ± 2.74 | 11.20 ± 2.09 | 11.05 ± 1.73 | **12.45 ± 1.40** | 8.90 ± 2.71 |
| 512 | 9 | 11.60 ± 2.16 | 12.35 ± 1.14 | 11.85 ± 1.46 | **13.85 ± 1.27** | 11.05 ± 1.54 |

- Consensus has the highest mean true-event count in all 15 width-by-N cells. Its margin over team or difference is often within one standard deviation, so the ranking among the top three is not sharp.
- False events stay small: at most 0.5 per seed in any cell, and precision is at least 0.937 in every cell where an event occurred. No scheme here wins by inflating false events.
- Individual reward has the lowest mean in 10 of 15 cells (every width up to 128, and 256 with N = 3). In the other five (256 with N = 5 or 9, and all of 512) the random baseline or the difference reward is lowest, so "individual is worst" holds at the smaller widths only.
- At N = 9 the random baseline is close to the team reward (for example 14.25 vs 14.25 at width 32): with many agents, enough of them declare on their own that the reward adds little.

**What changed from the first run.** The first run used a fixed threshold grid (1.2 to 8 on the energy ratio). At widths >= 128 it produced about zero true events for every scheme, because the energy ratio grows with width and the thresholds were crossed before the estimate settled. That collapse was a harness artifact, not a property of the schemes. It is kept in [spec/v1_fixed/](spec/v1_fixed/); the width-scaled grid above replaces it. An even earlier run had a tie-break bug (always the lowest threshold, no events) and is kept as `spec/results_*_v1_tiebias*`.

## Experiment 2: the 3-membrane contest

Three coupled membranes, each with a threshold, a coupling to the other two and a gain. Each keeps a running estimate of how often its declarations are confirmed by a peer ("self-knowing", no ground truth used) and lowers or raises its threshold with it. Score per generation: `w * joint_true + (1 - w) * rank_bonus`, a mix of shared reward for joint true events and a bonus for the most true events; w is swept (0, 0.5, 1 in the main runs). Between generations each membrane mutates and keeps changes that do not lower its score ((1+1)-ES, 12 generations of 8 episodes, fixed seeds). The final stage is applied in this order and each part can be switched off: **S** shield (keep only events confirmed by at least 2 of 3 membranes), **A** amplifier (the next episode warm-starts from the last output vector, weight capped at 0.5), **Z** stabilizer (leaky-integrator feedback on the output event rate damps the gain; gain clamped to [0.33, 3]).

Widths 32 to 128 ran w = 0, 0.5 and 1 with configs `---`, `S--`, `SA-`, `SAZ`. **Widths 256 and 512 ran only w = 0.5 and a subset of configs** (512: `---` and `SAZ`). 20 seeds. At w = 0.5:

| width | config | true events | false events | precision | event-rate variance |
|---|---|---|---|---|---|
| 32 | --- | 119.9 ± 5.9 | 10.2 ± 6.0 | 0.922 | 0.041 |
| 32 | S-- | 42.1 ± 3.5 | 4.0 ± 3.9 | 0.915 | 0.004 |
| 32 | SA- | 42.1 ± 3.1 | 4.0 ± 3.3 | 0.913 | 0.004 |
| 32 | SAZ | 40.6 ± 4.5 | 6.3 ± 4.4 | 0.865 | 0.001 |
| 64 | --- | 124.6 ± 6.4 | 7.0 ± 3.8 | 0.947 | 0.030 |
| 64 | S-- | 44.3 ± 1.9 | 2.4 ± 1.5 | 0.950 | 0.002 |
| 64 | SA- | 44.4 ± 1.8 | 2.3 ± 1.5 | 0.951 | 0.002 |
| 64 | SAZ | 42.6 ± 3.4 | 4.4 ± 3.6 | 0.907 | 0.001 |
| 128 | --- | 124.5 ± 7.0 | 6.4 ± 4.5 | 0.952 | 0.034 |
| 128 | S-- | 44.5 ± 1.9 | 2.0 ± 2.2 | 0.958 | 0.002 |
| 128 | SA- | 44.5 ± 1.9 | 2.0 ± 2.2 | 0.958 | 0.002 |
| 128 | SAZ | 43.4 ± 3.7 | 3.6 ± 3.7 | 0.924 | 0.001 |
| 256 | --- | 120.9 ± 5.5 | 7.5 ± 4.8 | 0.943 | 0.048 |
| 256 | S-- | 42.0 ± 3.4 | 3.2 ± 3.5 | 0.929 | 0.006 |
| 256 | SA- | 42.0 ± 3.4 | 3.2 ± 3.5 | 0.929 | 0.006 |
| 256 | SAZ | 42.9 ± 3.7 | 3.5 ± 3.4 | 0.926 | 0.003 |
| 512 | --- | 117.7 ± 6.6 | 7.0 ± 5.0 | 0.946 | 0.054 |
| 512 | SAZ | 42.6 ± 4.2 | 4.0 ± 3.6 | 0.914 | 0.002 |

Counts are summed over 12 generations. Without the shield, each membrane's event counts (up to 3 per episode); with it, at most 1 per episode. So raw counts differ by construction between `---` and the shielded rows; compare precision and rates, not totals.

- **Shield:** false events fall by about 2.3 to 3.2 times, and event-rate variance by about 8 to 17 times (from the rounded values in the table). Part of both is mechanical (at most one output event per episode). Precision barely moves (0.922 to 0.915 at width 32, 0.947 to 0.950 at 64, 0.952 to 0.958 at 128, 0.943 to 0.929 at 256). Coupled membranes can agree on the same wrong direction, and the shield passes those.
- **Amplifier:** no measurable effect. `SA-` equals `S--` in the cells above within rounding (identical at widths 128 and 256). Without the shield it did not help either: in the fixed-threshold run in `spec/v1_fixed/`, `-A-` tracked `---` (not rerun with scaled thresholds).
- **Stabilizer:** the lowest event-rate variance, and it costs precision, 0.3 to 4.8 points at w = 0.5 (`SA-` to `SAZ`; up to 5.0 at other w). Under the default settings gain never runs away (the maximum gain in any seed of the main runs is 2.6, and the mean maximum is under 1.5 in every cell), so the benefit is not visible there.
- **Synergy vs competition weight:** w = 0.5 and w = 1 give identical or nearly identical rows. Pure competition (w = 0) is somewhat worse before the shield: precision 0.901 vs 0.922 at width 32, 0.927 vs 0.947 at 64, 0.945 vs 0.952 at 128, with more false events (13.0 vs 10.2, 9.6 vs 7.0, 7.3 vs 6.4; each difference is about one standard deviation or less). After the shield the difference is smaller and not consistent in sign (`S--` at width 64: 0.935 vs 0.950; at 128: 0.961 vs 0.958). The score does not charge false events, so gaming was possible; competition raised false events a little before the shield, and nothing larger was seen.

### Stabilizer stress test (width 64)

Initial gain 3 and large gain mutations (sigma 0.6), so runaway can occur. Runaway means max gain above 10. 20 seeds.

| w | config | true events | false events | precision | event-rate variance | max gain | runaway (gain > 10) |
|---|---|---|---|---|---|---|---|
| 0 | SA- | 17.6 ± 10.2 | 28.4 ± 11.5 | 0.388 | 0.025 | 21.5 | 0.35 |
| 0 | SAZ | 19.6 ± 7.5 | 24.6 ± 8.1 | 0.444 | 0.009 | 3.0 | 0.00 |
| 0.5 | SA- | 17.7 ± 8.8 | 29.4 ± 11.0 | 0.383 | 0.019 | 10.4 | 0.20 |
| 0.5 | SAZ | 20.1 ± 7.2 | 24.1 ± 8.8 | 0.462 | 0.010 | 3.0 | 0.00 |
| 1 | SA- | 14.8 ± 9.1 | 34.9 ± 16.5 | 0.318 | 0.031 | 15.5 | 0.35 |
| 1 | SAZ | 18.9 ± 7.1 | 26.4 ± 7.5 | 0.417 | 0.007 | 3.0 | 0.00 |

Without the stabilizer, runaway occurs in 20 to 35 percent of seeds; with it, none, and event-rate variance is 2 to 4 times lower. Precision is higher with the stabilizer under stress (0.42 to 0.46 against 0.32 to 0.39), though absolute precision is poor in both. **Most of this is the hard clamp of the gain at 3, not the leaky-integrator damping; there is no ablation separating the two.** In the default (non-stress) runs the stabilizer only costs precision, so it earns its place only where gain can run away.

### Cost of false events (cost 2 per false event per episode)

Does charging membranes for false events raise precision? Widths 64 and 128, 20 seeds:

| width | w | config | precision, no cost | precision, cost 2 | false events, no cost | false events, cost 2 |
|---|---|---|---|---|---|---|
| 64 | 0 | --- | 0.927 | 0.930 | 9.60 | 9.10 |
| 64 | 0 | S-- | 0.935 | 0.940 | 3.05 | 2.85 |
| 64 | 0 | SAZ | 0.905 | 0.922 | 4.50 | 3.70 |
| 64 | 0.5 | --- | 0.947 | 0.947 | 6.95 | 6.95 |
| 64 | 0.5 | S-- | 0.950 | 0.950 | 2.35 | 2.35 |
| 64 | 0.5 | SAZ | 0.907 | 0.895 | 4.40 | 4.95 |
| 64 | 1 | --- | 0.947 | 0.947 | 6.95 | 6.95 |
| 64 | 1 | S-- | 0.950 | 0.950 | 2.35 | 2.35 |
| 64 | 1 | SAZ | 0.901 | 0.899 | 4.65 | 4.75 |
| 128 | 0 | --- | 0.945 | 0.945 | 7.25 | 7.20 |
| 128 | 0 | S-- | 0.961 | 0.960 | 1.85 | 1.90 |
| 128 | 0 | SAZ | 0.926 | 0.921 | 3.55 | 3.75 |
| 128 | 0.5 | --- | 0.952 | 0.952 | 6.40 | 6.40 |
| 128 | 0.5 | S-- | 0.958 | 0.958 | 2.00 | 2.00 |
| 128 | 0.5 | SAZ | 0.924 | 0.924 | 3.60 | 3.60 |
| 128 | 1 | --- | 0.949 | 0.950 | 6.75 | 6.55 |
| 128 | 1 | S-- | 0.955 | 0.955 | 2.10 | 2.10 |
| 128 | 1 | SAZ | 0.927 | 0.927 | 3.45 | 3.45 |

No. Precision changes by at most 0.017 in either direction and is unchanged to three decimals in 10 of 18 cells. The selection score uses each membrane's own raw events, false events are already rare, and the shield and amplifier act only on the output stage, so the cost rarely changes which mutation is kept. A higher cost, or a regime with more false events, was not tried.

## Limits

- SYNTHETIC toy scale: random vectors, seeded noise, 12 generations of 8 episodes.
- The event definition and all parameters (steps, window, eta, thresholds, noise scale, agreement 0.4, truth 0.5) are hand-picked, with a few exploratory checks, not tuned or optimised.
- The settled energy ratio in the threshold formula is an analytic estimate that was not derived carefully; small mismatches at some widths are possible.
- Contest at widths 256 and 512 covers only w = 0.5 and a subset of configs.
- Individual-reward results at the largest widths are not lowest; see Experiment 1.
- The stabilizer's effect under stress is not separated into clamp and damping.
- Results describe this simulation only.

See [DESIGN.md](DESIGN.md) for the details and [CHANGELOG.md](CHANGELOG.md) for what changed.

## Origin

The exploration as received (fixed-threshold run, scaled-threshold run, scripts and CSVs) is kept unchanged in [spec/](spec/); the package, tests and `results/` follow from it. The package reproduces stored rows exactly (tests compare a sweep row and a contest row with `results/`). The helper functions `normalize`, `inner`, `fidelity` and `noisy` are vendored from membrane-crystallization v0.1.0 (same author, AGPL-3.0-only) so the package stays standard-library only.

## References

- Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology* 15(3), 267-273. https://doi.org/10.1007/BF00275687
- Brisson, J.-F. membrane-crystallization v0.1.0. https://doi.org/10.5281/zenodo.23583050 (all versions: https://doi.org/10.5281/zenodo.23583049)
- Brisson, J.-F. membrane-committee v0.2.0. https://doi.org/10.5281/zenodo.23583048

## How to cite

Cite the concept DOI [10.5281/zenodo.23619514](https://doi.org/10.5281/zenodo.23619514) for all versions, or the version DOI [10.5281/zenodo.23619515](https://doi.org/10.5281/zenodo.23619515) for v0.1.0. See also [CITATION.cff](CITATION.cff).

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
