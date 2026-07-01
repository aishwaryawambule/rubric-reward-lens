"""Production grader backed by an OpenRouter (OpenAI-compatible) chat model.

The grader asks the model to score each rubric criterion in ``[0, 1]`` and
return strict JSON. ``parse_grader_json`` is separated out and tested directly
so the parsing logic is covered without any network access; ``OpenRouterGrader``
accepts an injectable ``transport`` so the HTTP path is also tested offline.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from .grader import aggregate
from .models import CriterionScore, GraderResult, Rubric, Response

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_PROMPT = """You are a strict grader. Score the RESPONSE against each rubric criterion.
For each criterion output a score in [0,1] where 1 means fully satisfied.
Return ONLY JSON: {{"per_criterion":[{{"criterion_id":"<id>","score":<0..1>,"justification":"<short>"}}]}}

RUBRIC:
{rubric}

RESPONSE:
{response}
"""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_grader_json(raw: str, rubric: Rubric, response_id: str) -> GraderResult:
    """Parse a grader model reply into a ``GraderResult``.

    Tolerates code fences and surrounding prose by extracting the outermost
    ``{...}``. Missing criteria default to ``0.0``; out-of-range scores are
    clamped to ``[0, 1]``.
    """
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        data: dict = {}
    else:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = {}

    by_id: dict[str, dict] = {}
    for item in data.get("per_criterion", []):
        cid = str(item.get("criterion_id", ""))
        if cid:
            by_id[cid] = item

    scores: list[CriterionScore] = []
    for c in rubric.criteria:
        item = by_id.get(c.id, {})
        try:
            score = _clamp01(float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        scores.append(CriterionScore(c.id, score, str(item.get("justification", ""))))
    return GraderResult(response_id, scores, aggregate(rubric, scores))


def _render_rubric(rubric: Rubric) -> str:
    lines = []
    for c in rubric.criteria:
        lines.append(f"- [{c.id}] ({c.polarity.value}, weight {c.weight}) {c.text}")
    return "\n".join(lines)


class OpenRouterGrader:
    """Grades responses with an OpenRouter chat model."""

    def __init__(
        self,
        model: str,
        prompt_template: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str = _API_URL,
    ) -> None:
        self.model = model
        self.prompt_template = prompt_template or _DEFAULT_PROMPT
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self._client = httpx.Client(transport=transport, timeout=120.0)

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        prompt = self.prompt_template.format(
            rubric=_render_rubric(rubric), response=response.text
        )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self._client.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_grader_json(content, rubric, response.id)
