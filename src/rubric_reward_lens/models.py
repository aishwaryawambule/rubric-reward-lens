"""Core data models for rubric-reward-lens.

A ``Rubric`` is a weighted checklist of ``Criterion`` items. A ``Grader`` turns
a ``(Rubric, Response)`` pair into a ``GraderResult`` carrying a per-criterion
``CriterionScore`` list and an aggregated scalar ``reward``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import yaml


class Polarity(str, Enum):
    """Whether a criterion rewards including something or avoiding it."""

    INCLUDE = "include"
    AVOID = "avoid"


@dataclass(frozen=True)
class Criterion:
    """One rubric line item, with a relative ``weight`` and a ``polarity``."""

    id: str
    text: str
    weight: float = 1.0
    polarity: Polarity = Polarity.INCLUDE


@dataclass
class Rubric:
    """An ordered, weighted checklist of criteria."""

    criteria: list[Criterion]
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Rubric":
        criteria = [
            Criterion(
                id=str(c["id"]),
                text=str(c["text"]),
                weight=float(c.get("weight", 1.0)),
                polarity=Polarity(c.get("polarity", "include")),
            )
            for c in d.get("criteria", [])
        ]
        return cls(criteria=criteria, name=str(d.get("name", "")))

    @classmethod
    def from_json(cls, path: str) -> "Rubric":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_yaml(cls, path: str) -> "Rubric":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.criteria]

    def get(self, cid: str) -> Criterion:
        for c in self.criteria:
            if c.id == cid:
                return c
        raise KeyError(cid)


@dataclass
class Response:
    """A model answer to be graded. ``human_score`` is optional ground truth."""

    id: str
    text: str
    prompt_id: str = ""
    prompt: str = ""
    human_score: float | None = None


@dataclass(frozen=True)
class CriterionScore:
    """A grader's score (in ``[0, 1]``) for one criterion of one response."""

    criterion_id: str
    score: float
    justification: str = ""


@dataclass
class GraderResult:
    """The full grading of one response: per-criterion scores + scalar reward."""

    response_id: str
    per_criterion: list[CriterionScore] = field(default_factory=list)
    reward: float = 0.0

    def score_for(self, cid: str) -> float:
        for cs in self.per_criterion:
            if cs.criterion_id == cid:
                return cs.score
        raise KeyError(cid)
