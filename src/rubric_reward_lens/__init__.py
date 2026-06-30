"""rubric-reward-lens: a pre-flight auditor for rubric-based LLM rewards.

Point it at your rubric + grader + a few example answers and it tells you,
before you spend GPU on RL, whether the reward signal measures quality or can
be gamed.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .models import (
    Criterion,
    CriterionScore,
    GraderResult,
    Polarity,
    Response,
    Rubric,
)

__all__ = [
    "__version__",
    "Criterion",
    "CriterionScore",
    "GraderResult",
    "Polarity",
    "Response",
    "Rubric",
]
