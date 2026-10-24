#!/usr/bin/env bash
# Reproduces the full results in results/ (20 seeds). Takes a few hours on 8 cores;
# widths 256 and 512 dominate. Each experiment writes its CSVs as rows arrive.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results_repro}
J=${JOBS:-8}
python -m crystallization_contest sweep --widths 32,64,128,256,512 --agents 3,5,9 --seeds 20 --jobs "$J" --out "$OUT/sweep"
python -m crystallization_contest contest --widths 32,64,128 --weights 0,0.5,1 --configs ---,S--,SA-,SAZ --seeds 20 --jobs "$J" --out "$OUT/contest_w32-128"
python -m crystallization_contest contest --widths 256 --weights 0.5 --configs ---,S--,SA-,SAZ --seeds 20 --jobs "$J" --out "$OUT/contest_256"
python -m crystallization_contest contest --widths 512 --weights 0.5 --configs ---,SAZ --seeds 20 --jobs "$J" --out "$OUT/contest_512"
python -m crystallization_contest stress --widths 64 --weights 0,0.5,1 --configs SA-,SAZ --seeds 20 --jobs "$J" --out "$OUT/stress"
python -m crystallization_contest fcost --widths 64,128 --weights 0,0.5,1 --configs ---,S--,SAZ --seeds 20 --jobs "$J" --out "$OUT/fcost"
