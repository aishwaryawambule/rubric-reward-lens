"""Criterion-order invariance diagnostic.

A *pointwise* grader should score a response the same regardless of the order in
which its rubric criteria are listed. This re-grades each response with the
criteria shuffled (the reward's aggregation is id-keyed, so any change is pure
judge position bias) and measures the reward drift.

Motivated by "Am I More Pointwise or Pairwise?" (arXiv:2602.02219), which
isolates criterion-order bias for pointwise graders, and Rubrics-as-Rewards
(arXiv:2507.17746).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grader import Grader
from ..models import Response, Rubric

DRIFT_THRESHOLD = 0.05  # mean reward drift above this => order-sensitive


@dataclass
class OrderResult:
    mean_drift: float  # mean |reward(shuffled) - reward(original)| over responses
    max_drift: float  # worst single drift observed
    flip_rate: float  # fraction of responses whose pass/fail vector changed
    order_invariant: bool
    n: int = 0


def _pass_vector(result, rubric: Rubric) -> tuple[int, ...]:
    # id-keyed, so it is comparable across criterion orderings
    return tuple(1 if result.score_for(c.id) >= 0.5 else 0 for c in rubric.criteria)


def run_criterion_order(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    k: int = 3,
    seed: int = 0,
) -> OrderResult:
    crit = rubric.criteria
    if len(crit) < 2 or not responses:
        return OrderResult(0.0, 0.0, 0.0, True, len(responses))

    n = len(crit)
    rng = np.random.default_rng(seed)
    # Always probe the reverse order (the worst-case reordering), then fill with
    # seeded random permutations. Reverse guarantees an order-sensitive grader is
    # caught regardless of which random draws come up.
    orders: list[list[int]] = [list(range(n - 1, -1, -1))]
    while len(orders) < k:
        orders.append([int(i) for i in rng.permutation(n)])
    perm_rubrics = [Rubric(criteria=[crit[i] for i in order], name=rubric.name) for order in orders]

    per_response_mean: list[float] = []
    max_drift = 0.0
    flips = 0
    for r in responses:
        base = grader.grade(rubric, r)
        base_pass = _pass_vector(base, rubric)
        drifts: list[float] = []
        flipped = False
        for pr in perm_rubrics:
            g = grader.grade(pr, r)
            d = abs(g.reward - base.reward)
            drifts.append(d)
            max_drift = max(max_drift, d)
            if _pass_vector(g, rubric) != base_pass:
                flipped = True
        per_response_mean.append(sum(drifts) / len(drifts))
        if flipped:
            flips += 1

    mean_drift = float(np.mean(per_response_mean))
    return OrderResult(
        mean_drift=mean_drift,
        max_drift=float(max_drift),
        flip_rate=flips / len(responses),
        order_invariant=mean_drift <= DRIFT_THRESHOLD,
        n=len(responses),
    )
