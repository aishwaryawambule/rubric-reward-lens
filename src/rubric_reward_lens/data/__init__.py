"""Bundled SYNTHETIC demo data.

This is a small, hand-authored medical-Q&A-style dataset used by ``rrl demo``
and the tests. It is **not** real HealthBench data — HealthBench
(https://openai.com/index/healthbench/, arXiv:2505.08775) inspired the *shape*
(physician-style rubric + graded responses + human scores), but to avoid any
licensing question we ship a synthetic illustrative sample instead. To audit a
real reward, bring your own rubric + responses.
"""

from __future__ import annotations

import json
from importlib import resources

from ..models import Response, Rubric


def load_demo() -> tuple[Rubric, list[Response]]:
    """Load the bundled synthetic rubric + responses (with human scores)."""
    pkg = resources.files(__package__) / "healthbench_demo"
    rubric = Rubric.from_dict(json.loads((pkg / "rubric.json").read_text(encoding="utf-8")))
    raw = json.loads((pkg / "responses.json").read_text(encoding="utf-8"))
    responses = [
        Response(
            id=str(r["id"]),
            text=str(r["text"]),
            prompt_id=str(r.get("prompt_id", "")),
            prompt=str(raw.get("prompt", "")),
            human_score=(None if r.get("human_score") is None else float(r["human_score"])),
        )
        for r in raw["responses"]
    ]
    return rubric, responses
