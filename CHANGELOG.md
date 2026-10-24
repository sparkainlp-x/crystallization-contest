# Changelog

## 0.1.0 (2026-10-24)

- First commit: the exploration exactly as received (`spec/`): scripts, CSVs, tables, the fixed-threshold run in `spec/v1_fixed/` and the earlier tie-break run.
- Package `crystallization_contest`: event definition and ground-truth labels, five reward schemes, the reward sweep, the 3-membrane contest with shield, amplifier and stabilizer, and a command line (`sweep`, `contest`, `stress`, `fcost`, each with `--quick`). Standard library only; `normalize`, `inner`, `fidelity` and `noisy` are vendored from membrane-crystallization v0.1.0 (same author, AGPL-3.0-only).
- Threshold grid scaled with width (`1 + q n / (1 + 2 sigma^2 n)`), replacing the fixed grid whose collapse at widths >= 128 was a harness artifact.
- Results in `results/` (20 seeds): reward sweep at widths 32 to 512, contest at 32 to 128 for w in {0, 0.5, 1}, contest at 256 and 512 at w = 0.5 only, stabilizer stress test at width 64, false-event-cost variant at widths 64 and 128. `scripts/reproduce_full.sh` reruns them.
- 29 pytest tests with warnings as errors, including exact reproduction of one stored sweep row and one stored contest row.
- CI on Python 3.10 to 3.13, README, DESIGN, CITATION.cff and .zenodo.json.
- Fixed in the first run before this release: a tie-break that always chose the lowest threshold, so that no scheme produced events.
