"""Tiny synthetic temporal edge list for TESTS ONLY (never used for reported results)."""
import gzip
from pathlib import Path

import numpy as np


def make_synthetic(path: Path, n_nodes=60, n_rows=6000, days=120, seed=0, gap_days=None) -> Path:
    rng = np.random.default_rng(seed)
    weights = rng.pareto(1.5, n_nodes) + 1
    p = weights / weights.sum()
    s = rng.choice(n_nodes, n_rows, p=p) + 100        # IDs not starting at 0
    r = rng.choice(n_nodes, n_rows, p=p[::-1]) + 100
    t = rng.integers(0, days * 86400, n_rows)
    t[0] = 0
    t[1] = days * 86400 - 1
    if gap_days:                                       # force empty windows
        lo, hi = gap_days
        t = np.where((t >= lo * 86400) & (t < hi * 86400), t % (lo * 86400), t)
    order = np.argsort(t, kind="stable")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        for a, b, c in zip(s[order], r[order], t[order]):
            f.write(f"{a} {b} {c}\n")
    return Path(path)
