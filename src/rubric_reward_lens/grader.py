"""Grader interface, scalar aggregation, and a deterministic FakeGrader.

A ``Grader`` is anything with ``grade(rubric, response) -> GraderResult``. The
production grader is :class:`~rubric_reward_lens.openrouter.OpenRouterGrader`;
:class:`FakeGrader` is a deterministic, keyword-presence grader used in tests
and as an intentionally *hackable* baseline for validating the diagnostics.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from .models import Criterion, CriterionScore, GraderResult, Polarity, Response, Rubric

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "any", "you", "your",
    "should", "must", "about", "into", "from", "have", "has", "was", "were",
    "they", "their", "them", "a", "an", "of", "to", "in", "on", "or", "is",
    "it", "as", "at", "be", "by", "relevant", "such",
}


@runtime_checkable
class Grader(Protocol):
    """Anything that can score a response against a rubric."""

    def grade(self, rubric: Rubric, response: Response) -> GraderResult: ...


def aggregate(rubric: Rubric, per_criterion: list[CriterionScore]) -> float:
    """Weighted mean of per-criterion scores, in ``[0, 1]``.

    Empty input returns ``0.0``. Criteria with non-positive total weight fall
    back to an unweighted mean.
    """
    if not per_criterion:
        return 0.0
    weights = {c.id: c.weight for c in rubric.criteria}
    total_w = 0.0
    acc = 0.0
    for cs in per_criterion:
        w = weights.get(cs.criterion_id, 1.0)
        acc += w * cs.score
        total_w += w
    if total_w <= 0:
        return sum(cs.score for cs in per_criterion) / len(per_criterion)
    return acc / total_w


def _keywords_for(text: str) -> list[str]:
    """Content words from a criterion's text (lowercased, len>3, minus stopwords)."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    seen: list[str] = []
    for t in tokens:
        if len(t) > 3 and t not in _STOPWORDS and t not in seen:
            seen.append(t)
    return seen


class FakeGrader:
    """Deterministic grader: a criterion is satisfied if any of its keywords
    appears in the response.

    Intentionally hackable — keyword-stuffing fools it — so it doubles as the
    "known-bad" reward in the falsification test. Pass ``window`` to only look
    at the first N characters of the response, which makes it *robust* to
    keywords appended at the end (the "known-good" reward).
    """

    def __init__(
        self,
        keywords: dict[str, list[str]] | None = None,
        window: int | None = None,
    ) -> None:
        self.keywords = keywords
        self.window = window

    def _kw(self, criterion: Criterion) -> list[str]:
        if self.keywords and criterion.id in self.keywords:
            return [k.lower() for k in self.keywords[criterion.id]]
        return _keywords_for(criterion.text)

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        haystack = response.text.lower()
        if self.window is not None:
            haystack = haystack[: self.window]
        scores: list[CriterionScore] = []
        for c in rubric.criteria:
            present = any(
                re.search(rf"\b{re.escape(kw)}\b", haystack) for kw in self._kw(c)
            )
            if c.polarity is Polarity.AVOID:
                score = 0.0 if present else 1.0
            else:
                score = 1.0 if present else 0.0
            scores.append(CriterionScore(c.id, score))
        return GraderResult(response.id, scores, aggregate(rubric, scores))
