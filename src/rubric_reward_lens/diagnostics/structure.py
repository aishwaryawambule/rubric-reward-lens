"""B1–B3 — criterion-structure diagnostic.

Grades every response once to build a [responses × criteria] score matrix, then
reports: redundant criterion pairs (always co-fire), low-signal criteria
(always pass or always fail), per-criterion coverage (pass rate), and weight
sensitivity (how much the reward ranking shifts when a criterion is dropped).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grader import Grader
from ..models import Response, Rubric
from ..stats import spearman

REDUNDANCY_THRESHOLD = 0.9
LOW_SIGNAL_LOW = 0.05
LOW_SIGNAL_HIGH = 0.95


@dataclass
class StructureResult:
    redundant_pairs: list[tuple[str, str, float]]
    weight_sensitivity: dict[str, float]
    coverage: dict[str, float]
    low_signal_criteria: list[str]
    n: int = 0


def _reward_ranking(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rewards = matrix @ weights / weights.sum() if weights.sum() else matrix.mean(axis=1)
    return rewards


def run_structure(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    redundancy_thresh: float = REDUNDANCY_THRESHOLD,
    seed: int = 0,
) -> StructureResult:
    cids = rubric.criterion_ids()
    weights = np.array([c.weight for c in rubric.criteria], dtype=float)
    matrix = np.array(
        [[grader.grade(rubric, r).score_for(cid) for cid in cids] for r in responses],
        dtype=float,
    )
    if matrix.size == 0:
        return StructureResult([], {}, {}, [], 0)

    coverage = {cid: float(matrix[:, j].mean()) for j, cid in enumerate(cids)}
    low_signal = [
        cid for cid, cov in coverage.items() if cov <= LOW_SIGNAL_LOW or cov >= LOW_SIGNAL_HIGH
    ]

    redundant: list[tuple[str, str, float]] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = matrix[:, i], matrix[:, j]
            if ci.std() == 0 or cj.std() == 0:
                continue
            corr = float(np.corrcoef(ci, cj)[0, 1])
            if corr >= redundancy_thresh:
                redundant.append((cids[i], cids[j], corr))

    base_rank = _reward_ranking(matrix, weights)
    sensitivity: dict[str, float] = {}
    for j, cid in enumerate(cids):
        keep = [k for k in range(len(cids)) if k != j]
        if not keep:
            sensitivity[cid] = 0.0
            continue
        dropped_rank = _reward_ranking(matrix[:, keep], weights[keep])
        sensitivity[cid] = 1.0 - spearman(base_rank.tolist(), dropped_rank.tolist())

    return StructureResult(
        redundant_pairs=redundant,
        weight_sensitivity=sensitivity,
        coverage=coverage,
        low_signal_criteria=low_signal,
        n=len(responses),
    )
