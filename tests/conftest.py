"""Shared fixtures and deterministic test graders.

These graders implement the ``Grader`` protocol without any network access, so
the whole suite runs offline. Several are intentionally *hackable* or *robust*
so the diagnostics can be validated against known-good and known-bad rewards.
"""

from __future__ import annotations

import re

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


_FUNCTION_WORDS = {
    "the", "a", "an", "to", "is", "are", "you", "your", "should", "of", "and",
    "for", "with", "that", "this", "it", "in", "on", "be", "will", "can",
    "about", "any", "see", "watch", "note", "review",
}
_NEGATIONS = {"never", "not", "no", "without", "none", "nothing"}


class RobustContentGrader:
    """A grader genuinely resistant to all four hack probes.

    It credits a criterion only when a keyword appears in a *content sentence*:
    >=4 words, containing at least one function word (so bare keyword lists from
    stuffing don't count), not a markdown line (defeats format-mimicry), and not
    negated (defeats confident-wrong claims). Verbosity filler carries no
    keywords, so it cannot help either. The score still falls as real content is
    removed, so the reward remains monotonic.
    """

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        prose = [
            ln for ln in response.text.splitlines()
            if not ln.lstrip().startswith(("#", "-", "*"))
        ]
        content: list[str] = []
        for s in ". ".join(prose).replace("\n", " ").split("."):
            words = s.lower().split()
            if (
                len(words) >= 4
                and any(w in _FUNCTION_WORDS for w in words)
                and not any(w in _NEGATIONS for w in words)
            ):
                content.append(s.lower())
        scores: list[CriterionScore] = []
        for c in rubric.criteria:
            kws = _keywords_for(c.text)
            ok = any(
                re.search(rf"\b{re.escape(kw)}\b", s) for s in content for kw in kws
            )
            scores.append(CriterionScore(c.id, 1.0 if ok else 0.0))
        return GraderResult(response.id, scores, aggregate(rubric, scores))


@pytest.fixture
def robust_grader() -> RobustContentGrader:
    return RobustContentGrader()


@pytest.fixture
def hackable_responses() -> list[Response]:
    """Responses spanning quality where most LACK rubric keywords, so a
    keyword-presence grader is reliably (and significantly) gameable."""
    return [
        Response("a", "Please rest and drink plenty of fluids.", human_score=0.1),
        Response("b", "You will likely feel better in a day or two.", human_score=0.15),
        Response("c", "Try to relax and avoid stress for now.", human_score=0.2),
        Response("d", "Get some sleep and take it easy at home.", human_score=0.1),
        Response("e", "It would be sensible to see a doctor about this.", human_score=0.5),
        Response("f", "Keep an eye on any worsening symptoms over time.", human_score=0.45),
        Response("g", "A good answer: see a doctor, watch for symptoms, check drug interactions.", human_score=0.95),
        Response("h", "Consider possible drug interactions before acting.", human_score=0.5),
    ]
