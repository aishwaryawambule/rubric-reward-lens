"""C1–C2 — human-alignment diagnostic (optional).

Runs only on responses that carry a ``human_score``. Measures how well the
reward tracks human judgment: rank correlation, quadratic-weighted kappa and
Cohen's kappa over binned scores, and a calibration error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grader import Grader
from ..models import Response, Rubric
from ..stats import cohen_kappa, quadratic_weighted_kappa, spearman


@dataclass
class AlignmentResult:
    correlation: float
    qwk: float
    kappa: float
    calibration_error: float
    ci: tuple[float, float]
    n: int = 0


def _normalize01(values: list[float]) -> list[float]:
    """Min-max scale to ``[0, 1]`` so human scores on any scale (0–1, 1–5,
    0–100) line up with the grader's ``[0, 1]`` reward. Degenerate (all-equal)
    input maps to ``0.5``."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _bin(values: list[float], n_bands: int) -> list[int]:
    edges = np.linspace(0.0, 1.0, n_bands + 1)
    # clip to last band; rightmost edge inclusive
    return [int(min(n_bands - 1, np.searchsorted(edges, v, side="right") - 1)) for v in values]


def run_alignment(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    n_bands: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
) -> AlignmentResult:
    labeled = [r for r in responses if r.human_score is not None]
    if not labeled:
        raise ValueError("run_alignment requires responses with human_score set")

    rewards = [grader.grade(rubric, r).reward for r in labeled]
    humans = _normalize01([float(r.human_score) for r in labeled])

    corr = spearman(rewards, humans)
    rb = _bin(rewards, n_bands)
    hb = _bin(humans, n_bands)
    qwk = quadratic_weighted_kappa(rb, hb, n_bands)
    kappa = cohen_kappa(rb, hb)
    cal_err = float(np.mean(np.abs(np.array(rewards) - np.array(humans))))

    # bootstrap CI on the correlation
    pairs = list(zip(rewards, humans))
    rng = np.random.default_rng(seed)

    if len(pairs) >= 2:
        boots = []
        n = len(pairs)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            xs = [pairs[i][0] for i in idx]
            ys = [pairs[i][1] for i in idx]
            boots.append(spearman(xs, ys))
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
    else:
        lo = hi = corr

    return AlignmentResult(
        correlation=corr,
        qwk=qwk,
        kappa=kappa,
        calibration_error=cal_err,
        ci=(lo, hi),
        n=len(labeled),
    )
