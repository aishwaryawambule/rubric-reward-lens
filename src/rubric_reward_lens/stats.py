"""Statistical utilities: bootstrap confidence intervals, rank correlation,
agreement (Cohen / quadratic-weighted kappa), and a significance helper.

Every headline metric in a report card carries a bootstrap CI from here, and
all randomness is seeded so report cards are reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns ``(point_estimate, ci_low, ci_high)``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    point = float(statistic(arr))
    if arr.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        boots[i] = statistic(sample)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return point, lo, hi


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of the mean paired difference ``a - b``."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("paired arrays must have equal length")
    return bootstrap_ci(aa - bb, np.mean, n_boot=n_boot, ci=ci, seed=seed)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``0.0`` if either side has no variance."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2:
        return 0.0
    xr = _rankdata(xa)
    yr = _rankdata(ya)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of values (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _confusion(a: np.ndarray, b: np.ndarray, n_bands: int) -> np.ndarray:
    m = np.zeros((n_bands, n_bands), dtype=float)
    for i, j in zip(a, b):
        m[int(i), int(j)] += 1
    return m


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for categorical agreement."""
    aa = np.asarray(a, dtype=int)
    bb = np.asarray(b, dtype=int)
    n_bands = int(max(aa.max(initial=0), bb.max(initial=0))) + 1
    return quadratic_weighted_kappa(aa, bb, n_bands, weighting="none")


def quadratic_weighted_kappa(
    a: Sequence[int], b: Sequence[int], n_bands: int, weighting: str = "quadratic"
) -> float:
    """Quadratic-weighted kappa (chance-corrected ordinal agreement)."""
    aa = np.asarray(a, dtype=int)
    bb = np.asarray(b, dtype=int)
    if aa.size == 0:
        return 0.0
    O = _confusion(aa, bb, n_bands)
    hist_a = O.sum(axis=1)
    hist_b = O.sum(axis=0)
    E = np.outer(hist_a, hist_b) / O.sum()
    idx = np.arange(n_bands)
    diff = idx[:, None] - idx[None, :]
    if weighting == "quadratic":
        W = (diff ** 2) / ((n_bands - 1) ** 2 if n_bands > 1 else 1)
    else:  # simple disagreement
        W = (diff != 0).astype(float)
    denom = (W * E).sum()
    if denom == 0:
        return 1.0
    return float(1 - (W * O).sum() / denom)


def significant(ci_low: float, ci_high: float) -> bool:
    """True when the confidence interval excludes zero."""
    return ci_low > 0 or ci_high < 0
