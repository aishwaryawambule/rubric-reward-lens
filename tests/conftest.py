"""Shared fixtures and deterministic test graders.

These graders implement the ``Grader`` protocol without any network access, so
the whole suite runs offline. Several are intentionally *hackable* or *robust*
so the diagnostics can be validated against known-good and known-bad rewards.
"""

from __future__ import annotations

import pytest

from rubric_reward_lens.grader import FakeGrader, _keywords_for, aggregate
from rubric_reward_lens.models import (
    Criterion,
    CriterionScore,
    GraderResult,
    Polarity,
    Response,
    Rubric,
)


@pytest.fixture
def rubric() -> Rubric:
    return Rubric(
        name="medical-qa",
        criteria=[
            Criterion("c1", "advises seeing a doctor promptly", weight=1.0),
            Criterion("c2", "lists relevant warning symptoms", weight=2.0),
            Criterion("c3", "warns about dangerous drug interactions", weight=1.0),
        ],
    )


@pytest.fixture
def responses() -> list[Response]:
    return [
        Response(
            "r1",
            "You should see a doctor promptly. Watch for warning symptoms such as "
            "chest pain. Be careful about dangerous drug interactions with blood thinners.",
            human_score=1.0,
        ),
        Response(
            "r2",
            "See a doctor. Symptoms matter.",
            human_score=0.5,
        ),
        Response(
            "r3",
            "Just rest and drink water, you'll be fine.",
            human_score=0.0,
        ),
        Response(
            "r4",
            "A doctor visit is wise. Note any symptoms and review drug interactions.",
            human_score=0.8,
        ),
    ]


class LengthGrader:
    """Reward proportional to the number of distinct content sentences retained.

    Used to validate the monotonicity diagnostic: truncating an answer (the
    degradation ladder) strictly lowers the reward.
    """

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        sentences = [s for s in response.text.split(".") if s.strip()]
        retained = min(len(sentences), 10)
        reward = retained / 10.0
        scores = [CriterionScore(c.id, reward) for c in rubric.criteria]
        return GraderResult(response.id, scores, reward)


class ConstantGrader:
    """Always returns the same reward — no discrimination at all."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        scores = [CriterionScore(c.id, self.value) for c in rubric.criteria]
        return GraderResult(response.id, scores, self.value)


class NoisyGrader:
    """Randomises each criterion pass/fail; reward is unstable across regrades."""

    def __init__(self, seed: int = 0) -> None:
        import numpy as np

        self._rng = np.random.default_rng(seed)

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        scores = [
            CriterionScore(c.id, float(self._rng.integers(0, 2))) for c in rubric.criteria
        ]
        return GraderResult(response.id, scores, aggregate(rubric, scores))


class HumanEchoGrader:
    """Reward equals the response's human_score (perfect alignment baseline)."""

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        val = response.human_score if response.human_score is not None else 0.0
        scores = [CriterionScore(c.id, val) for c in rubric.criteria]
        return GraderResult(response.id, scores, val)


@pytest.fixture
def length_grader() -> LengthGrader:
    return LengthGrader()


@pytest.fixture
def constant_grader() -> ConstantGrader:
    return ConstantGrader()


@pytest.fixture
def noisy_grader() -> NoisyGrader:
    return NoisyGrader(seed=0)


@pytest.fixture
def human_echo_grader() -> HumanEchoGrader:
    return HumanEchoGrader()
