"""A2 — discrimination / monotonicity diagnostic.

Builds a degradation ladder from each response (rung 0 = original, lower rungs
progressively worse) and checks that the reward falls as quality falls. A reward
that cannot rank an obviously-worse answer lower is unusable for RL.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..grader import Grader
from ..models import Response, Rubric
from ..probes import degradation_ladder
from ..stats import bootstrap_ci

MONOTONIC_SPEARMAN_THRESHOLD = -0.5


@dataclass
class MonotonicityResult:
    spearman: float
    ci: tuple[float, float]
    inversions: int
    separation: float
    monotonic: bool
    n: int = 0


def _ladder_spearman(rung_idx: list[int], rewards: list[float]) -> float:
    # corruption increases with rung index; reward should decrease.
    from ..stats import spearman

    return spearman(rung_idx, rewards)


def run_monotonicity(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    rungs: int = 4,
    n_boot: int = 1000,
    seed: int = 0,
) -> MonotonicityResult:
    per_response_rho: list[float] = []
    inversions = 0
    separations: list[float] = []

    for r in responses:
        ladder = degradation_ladder(r, rungs=rungs)
        rewards = [grader.grade(rubric, rung).reward for rung in ladder]
        idx = list(range(len(ladder)))
        per_response_rho.append(_ladder_spearman(idx, rewards))
        separations.append(rewards[0] - rewards[-1])
        for a, b in zip(rewards, rewards[1:]):
            if b > a + 1e-9:  # reward rose as corruption increased
                inversions += 1

    point, lo, hi = bootstrap_ci(per_response_rho, n_boot=n_boot, seed=seed)
    separation = sum(separations) / len(separations) if separations else 0.0
    monotonic = point <= MONOTONIC_SPEARMAN_THRESHOLD and separation > 0
    return MonotonicityResult(
        spearman=point,
        ci=(lo, hi),
        inversions=inversions,
        separation=separation,
        monotonic=monotonic,
        n=len(responses),
    )
