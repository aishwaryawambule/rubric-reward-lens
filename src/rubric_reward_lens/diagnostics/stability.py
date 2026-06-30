"""A3 — grader-stability diagnostic.

Re-grades each response several times. A deterministic grader yields zero
variance (stable); a noisy grader yields reward variance and per-criterion
pass/fail flips, which would inject noise into RL training.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..grader import Grader
from ..models import Response, Rubric
from ..stats import bootstrap_ci

REWARD_STD_THRESHOLD = 0.1
FLIP_RATE_THRESHOLD = 0.2


@dataclass
class StabilityResult:
    reward_std: float
    mean_ci_width: float
    per_criterion_flip_rate: dict[str, float]
    stable: bool
    n: int = 0


def run_stability(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    n_repeats: int = 5,
    seed: int = 0,
) -> StabilityResult:
    reward_stds: list[float] = []
    ci_widths: list[float] = []
    crit_flip: dict[str, list[float]] = {c.id: [] for c in rubric.criteria}

    for r in responses:
        results = [grader.grade(rubric, r) for _ in range(n_repeats)]
        rewards = [res.reward for res in results]
        reward_stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)
        _, lo, hi = bootstrap_ci(rewards, seed=seed)
        ci_widths.append(hi - lo)
        for c in rubric.criteria:
            passes = [1 if res.score_for(c.id) >= 0.5 else 0 for res in results]
            modal = 1 if sum(passes) * 2 >= len(passes) else 0
            flips = sum(1 for p in passes if p != modal) / len(passes)
            crit_flip[c.id].append(flips)

    reward_std = sum(reward_stds) / len(reward_stds) if reward_stds else 0.0
    mean_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 0.0
    flip_rate = {
        cid: (sum(v) / len(v) if v else 0.0) for cid, v in crit_flip.items()
    }
    stable = reward_std <= REWARD_STD_THRESHOLD and all(
        fr <= FLIP_RATE_THRESHOLD for fr in flip_rate.values()
    )
    return StabilityResult(
        reward_std=reward_std,
        mean_ci_width=mean_ci_width,
        per_criterion_flip_rate=flip_rate,
        stable=stable,
        n=len(responses),
    )
