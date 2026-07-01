"""rubric-reward-lens: a pre-flight auditor for rubric-based LLM rewards.

Point it at your rubric + grader + a few example answers and it tells you,
before you spend GPU on RL, whether the reward signal measures quality or can
be gamed.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .audit import DEFAULT_DIAGNOSTICS, audit
from .data import load_demo
from .grader import FakeGrader, aggregate
from .models import (
    Criterion,
    CriterionScore,
    DiagnosticResult,
    GraderResult,
    Polarity,
    Response,
    Rubric,
)
from .probes import Probe
from .ollama import OllamaGrader
from .openrouter import OpenRouterGrader
from .report import ReportCard

__all__ = [
    "__version__",
    "audit",
    "DEFAULT_DIAGNOSTICS",
    "Criterion",
    "CriterionScore",
    "DiagnosticResult",
    "GraderResult",
    "Polarity",
    "Response",
    "Rubric",
    "Probe",
    "FakeGrader",
    "aggregate",
    "OpenRouterGrader",
    "OllamaGrader",
    "ReportCard",
    "load_demo",
]
