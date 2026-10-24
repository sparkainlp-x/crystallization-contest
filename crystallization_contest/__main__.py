"""python -m crystallization_contest sweep|contest|stress|fcost [--quick] ...

All runs are SYNTHETIC classical simulations. Defaults are small; the full
published runs are listed in scripts/reproduce_full.sh.
"""

import argparse
import csv
import os
import sys

from . import contest, sweep
from .contest import CFG_BY_TAG
from .stats import write_rows


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def _floats(s):
    return [float(x) for x in s.split(",") if x]


class RawSink:
    """Appends each raw row to a CSV as it arrives, so progress survives interruption."""

    def __init__(self, path):
        self.f = open(path, "w", newline="") if path else None
        self.w = None

    def __call__(self, row):
        if self.f is None:
            return
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def build_parser():
    ap = argparse.ArgumentParser(prog="crystallization_contest",
                                 description="SYNTHETIC reward-scheme and contest simulations.")
    ap.add_argument("experiment", choices=["sweep", "contest", "stress", "fcost"])
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--widths", type=_ints, help="comma list, e.g. 32,64,128")
    ap.add_argument("--agents", type=_ints, help="sweep: comma list of N (default 3,5)")
    ap.add_argument("--weights", type=_floats, help="contest: synergy weights w in [0,1]")
    ap.add_argument("--configs", help="contest: comma list of ---,S--,-A-,SA-,SAZ,-AZ")
    ap.add_argument("--seeds", type=int, help="number of seeds (default 3, quick 2)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="directory for <experiment>_raw.csv and _summary.csv")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be >= 1")
    if a.seeds is not None and a.seeds < 2:
        ap.error("--seeds must be >= 2 (standard deviation needs two seeds)")
    seeds = a.seeds if a.seeds is not None else (2 if a.quick else 3)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, a.experiment + "_raw.csv") if a.out else None
    summ = os.path.join(a.out, a.experiment + "_summary.csv") if a.out else None
    sink = RawSink(raw)
    print("SYNTHETIC classical simulation; counts are labelled against the generator's ground truth.")
    if a.quick:
        print("--quick is a smoke run; its numbers are not comparable to the full results.")
    print()
    try:
        if a.experiment == "sweep":
            widths = a.widths or [32]
            agents = a.agents or ([3] if a.quick else [3, 5])
            episodes, steps = (6, 60) if a.quick else (sweep.EPISODES, 100)
            _, agg = sweep.run_sweep(widths, agents, seeds, a.jobs, episodes, steps, sink)
            fmt = sweep.format_row
        else:
            default_w = 64 if a.experiment == "stress" else 32
            widths = a.widths or [32 if a.quick else default_w]
            ws = a.weights or [0.5]
            default_cfg = {"contest": "---,S--,SAZ", "stress": "SA-,SAZ",
                           "fcost": "---,S--,SAZ"}[a.experiment]
            tags = (a.configs or ("---,SAZ" if a.quick and a.experiment != "stress" else default_cfg)).split(",")
            for t in tags:
                if t not in CFG_BY_TAG:
                    ap.error(f"unknown config {t!r}")
            cfgs = [CFG_BY_TAG[t] for t in tags]
            opts = {}
            if a.experiment == "stress":
                opts.update(init_g=3.0, gsig=0.6)
            if a.experiment == "fcost":
                opts.update(fcost=2.0)
            if a.quick:
                opts.update(gens=3, ep=4, steps=40)
            _, agg = contest.run_contest(widths, ws, cfgs, seeds, a.jobs, opts, sink)
            fmt = contest.format_row
    finally:
        sink.close()
    if summ:
        write_rows(summ, agg)
    for r in agg:
        print(fmt(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
