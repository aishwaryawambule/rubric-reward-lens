"""The ``audit()`` orchestrator: run the requested diagnostics over a response
set and assemble a :class:`~rubric_reward_lens.report.ReportCard`.
"""

from __future__ import annotations

from .diagnostics.alignment import run_alignment
from .diagnostics.hacking import run_hacking
from .diagnostics.monotonicity import run_monotonicity
from .diagnostics.stability import run_stability
from .diagnostics.structure import run_structure
from .grader import Grader
from .models import Response, Rubric
from .report import ReportCard

DEFAULT_DIAGNOSTICS = ("hacking", "monotonicity", "stability", "structure")


def audit(
    rubric: Rubric,
    grader: Grader,
    responses: list[Response],
    diagnostics: tuple[str, ...] = DEFAULT_DIAGNOSTICS,
    human_labels: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> ReportCard:
    """Audit a rubric+grader reward signal and return a report card.

    ``diagnostics`` selects which label-free diagnostics to run. Set
    ``human_labels=True`` to additionally run the alignment diagnostic on
    responses that carry a ``human_score``.
    """
    card = ReportCard(rubric_name=rubric.name, n_responses=len(responses))

    if "hacking" in diagnostics:
        card.hacking = run_hacking(rubric, grader, responses, n_boot=n_boot, seed=seed)
    if "monotonicity" in diagnostics:
        card.monotonicity = run_monotonicity(rubric, grader, responses, n_boot=n_boot, seed=seed)
    if "stability" in diagnostics:
        card.stability = run_stability(rubric, grader, responses, seed=seed)
    if "structure" in diagnostics:
        card.structure = run_structure(rubric, grader, responses, seed=seed)

    if human_labels:
        labeled = [r for r in responses if r.human_score is not None]
        if labeled:
            card.alignment = run_alignment(rubric, grader, labeled, n_boot=n_boot, seed=seed)

    return card
