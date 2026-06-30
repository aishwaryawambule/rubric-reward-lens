"""Command-line interface: ``rrl audit`` and ``rrl demo``.

``rrl demo`` audits the bundled synthetic dataset with a FakeGrader so it runs
with zero setup. ``rrl audit`` takes your rubric, grader config, and responses.
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from .audit import audit
from .data import load_demo
from .grader import FakeGrader
from .models import Response, Rubric
from .openrouter import OpenRouterGrader
from .report import ReportCard


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(fh)
        return json.load(fh)


def _make_grader(cfg: dict):
    gtype = cfg.get("type", "fake")
    if gtype == "fake":
        return FakeGrader(keywords=cfg.get("keywords"), window=cfg.get("window"))
    if gtype == "openrouter":
        return OpenRouterGrader(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.0),
            api_key=cfg.get("api_key"),
        )
    raise ValueError(f"unknown grader type: {gtype!r}")


def _load_responses(path: str) -> list[Response]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw["responses"] if isinstance(raw, dict) else raw
    return [
        Response(
            id=str(r["id"]),
            text=str(r["text"]),
            prompt_id=str(r.get("prompt_id", "")),
            prompt=str(r.get("prompt", "")),
            human_score=(None if r.get("human_score") is None else float(r["human_score"])),
        )
        for r in items
    ]


def _emit(card: ReportCard, out: str | None) -> None:
    if out is None:
        sys.stdout.write(card.to_markdown())
        return
    if out.endswith(".json"):
        card.to_json(out)
    elif out.endswith(".html"):
        card.to_html(out)
    else:
        card.to_markdown(out)


def _run_audit(rubric: Rubric, grader, responses, human_labels: bool, out: str | None) -> int:
    card = audit(rubric, grader, responses, human_labels=human_labels)
    _emit(card, out)
    sys.stderr.write(card.verdict + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rrl", description="Audit rubric-based LLM rewards.")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit your rubric + grader + responses")
    a.add_argument("--rubric", required=True)
    a.add_argument("--grader", required=True, help="grader config (yaml/json): {type: fake|openrouter, ...}")
    a.add_argument("--responses", required=True)
    a.add_argument("--out", default=None, help="output path (.json/.md/.html); default: markdown to stdout")
    a.add_argument("--human-labels", action="store_true")

    d = sub.add_parser("demo", help="audit the bundled synthetic dataset with a FakeGrader")
    d.add_argument("--out", default=None)

    args = parser.parse_args(argv)

    if args.command == "demo":
        rubric, responses = load_demo()
        return _run_audit(rubric, FakeGrader(), responses, human_labels=True, out=args.out)

    if args.command == "audit":
        rubric = (
            Rubric.from_yaml(args.rubric)
            if args.rubric.endswith((".yaml", ".yml"))
            else Rubric.from_json(args.rubric)
        )
        grader = _make_grader(_load_config(args.grader))
        responses = _load_responses(args.responses)
        return _run_audit(rubric, grader, responses, human_labels=args.human_labels, out=args.out)

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
