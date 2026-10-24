"""Small helpers for seed aggregation and CSV output (standard library)."""

import csv
import math
import statistics


def mean_std(values):
    """(mean, sample std) over non-NaN values; (nan, nan) if fewer than two."""
    v = [x for x in values if not math.isnan(x)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    return (statistics.mean(v), statistics.stdev(v))


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def map_jobs(fn, jobs, n_jobs, sink=None):
    """Run fn over jobs, serially or with a process pool; sink(row) after each row.

    Rows are returned sorted by job order so output does not depend on the
    scheduler.
    """
    rows = []
    if n_jobs <= 1:
        for j in jobs:
            r = fn(j)
            rows.append(r)
            if sink:
                sink(r)
        return rows
    from multiprocessing import Pool
    with Pool(n_jobs) as p:
        for r in p.imap(fn, jobs, chunksize=1):
            rows.append(r)
            if sink:
                sink(r)
    return rows
