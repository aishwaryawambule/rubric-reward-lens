"""Reward-quality diagnostics.

Each diagnostic takes ``(rubric, grader, responses)`` and returns a structured
result with the metrics and a boolean verdict used by the report card.
"""

from __future__ import annotations

from .alignment import AlignmentResult, run_alignment
from .hacking import HackingResult, run_hacking
from .monotonicity import MonotonicityResult, run_monotonicity
from .stability import StabilityResult, run_stability
from .structure import StructureResult, run_structure

__all__ = [
    "AlignmentResult",
    "run_alignment",
    "HackingResult",
    "run_hacking",
    "MonotonicityResult",
    "run_monotonicity",
    "StabilityResult",
    "run_stability",
    "StructureResult",
    "run_structure",
]
