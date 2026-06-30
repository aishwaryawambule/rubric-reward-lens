"""The Reward Report Card: composite trust score, headline verdict, and
JSON / markdown / HTML renderers (stdlib only — no templating dependency).
"""

from __future__ import annotations

import dataclasses
import html
import json
from dataclasses import dataclass

from .diagnostics.alignment import AlignmentResult
from .diagnostics.hacking import HackingResult
from .diagnostics.monotonicity import MonotonicityResult
from .diagnostics.stability import StabilityResult
from .diagnostics.structure import StructureResult


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class ReportCard:
    rubric_name: str
    n_responses: int
    hacking: HackingResult | None = None
    monotonicity: MonotonicityResult | None = None
    stability: StabilityResult | None = None
    structure: StructureResult | None = None
    alignment: AlignmentResult | None = None

    # ---- composite scoring -------------------------------------------------
    def sub_scores(self) -> dict[str, float]:
        s: dict[str, float] = {}
        if self.hacking is not None:
            s["hacking"] = 1.0 - _clamp(self.hacking.overall_hack_gain)
        if self.monotonicity is not None:
            s["monotonicity"] = _clamp(-self.monotonicity.spearman)
        if self.stability is not None:
            s["stability"] = 1.0 - _clamp(self.stability.reward_std / 0.5)
        if self.structure is not None:
            n_crit = max(1, len(self.structure.coverage))
            frac_low = len(self.structure.low_signal_criteria) / n_crit
            s["structure"] = 1.0 - _clamp(frac_low)
        if self.alignment is not None:
            s["alignment"] = _clamp(self.alignment.qwk)
        return s

    @property
    def trust_score(self) -> float:
        sub = self.sub_scores()
        return sum(sub.values()) / len(sub) if sub else 0.0

    @property
    def verdict(self) -> str:
        if self.hacking is not None and self.hacking.hackable:
            worst = max(
                self.hacking.per_probe.items(), key=lambda kv: kv[1][0], default=None
            )
            probe = f" (worst: {worst[0]} +{worst[1][0]:.2f})" if worst else ""
            return (
                f"⚠️ Hackable — reward can be gamed for "
                f"+{self.hacking.overall_hack_gain:.2f}{probe}. Trust score "
                f"{self.trust_score:.2f}. Fix the rubric before training."
            )
        if self.trust_score >= 0.7:
            return (
                f"✅ Robust — no significant gaming detected. "
                f"Trust score {self.trust_score:.2f}."
            )
        return (
            f"⚠️ Caution — trust score {self.trust_score:.2f}; "
            f"review the diagnostics below before training."
        )

    # ---- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        def conv(x):
            if dataclasses.is_dataclass(x):
                return {k: conv(v) for k, v in dataclasses.asdict(x).items()}
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [conv(v) for v in x]
            return x

        return {
            "rubric_name": self.rubric_name,
            "n_responses": self.n_responses,
            "trust_score": self.trust_score,
            "verdict": self.verdict,
            "sub_scores": self.sub_scores(),
            "diagnostics": {
                "hacking": conv(self.hacking) if self.hacking else None,
                "monotonicity": conv(self.monotonicity) if self.monotonicity else None,
                "stability": conv(self.stability) if self.stability else None,
                "structure": conv(self.structure) if self.structure else None,
                "alignment": conv(self.alignment) if self.alignment else None,
            },
        }

    def to_json(self, path: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_markdown(self, path: str | None = None) -> str:
        lines = [
            f"# Reward Report Card — {self.rubric_name}",
            "",
            f"**{self.verdict}**",
            "",
            f"- Responses audited: {self.n_responses}",
            f"- Composite trust score: **{self.trust_score:.2f}**",
            "",
            "## Sub-scores",
            "",
            "| Diagnostic | Score |",
            "| --- | --- |",
        ]
        for k, v in self.sub_scores().items():
            lines.append(f"| {k} | {v:.2f} |")
        if self.hacking is not None:
            lines += [
                "",
                "## Reward hacking",
                f"- Overall hack gain: {self.hacking.overall_hack_gain:.3f} "
                f"(CI {self.hacking.ci[0]:.3f}–{self.hacking.ci[1]:.3f})",
                f"- Hackable: {self.hacking.hackable}",
            ]
            for name, (g, lo, hi) in self.hacking.per_probe.items():
                lines.append(f"  - {name}: +{g:.3f} (CI {lo:.3f}–{hi:.3f})")
        if self.monotonicity is not None:
            lines += [
                "",
                "## Discrimination / monotonicity",
                f"- Spearman(corruption, reward): {self.monotonicity.spearman:.3f}",
                f"- Inversions: {self.monotonicity.inversions}; "
                f"separation: {self.monotonicity.separation:.3f}; "
                f"monotonic: {self.monotonicity.monotonic}",
            ]
        if self.stability is not None:
            lines += [
                "",
                "## Grader stability",
                f"- Reward std: {self.stability.reward_std:.3f}; "
                f"stable: {self.stability.stable}",
            ]
        if self.structure is not None:
            lines += [
                "",
                "## Criterion structure",
                f"- Redundant pairs: {self.structure.redundant_pairs}",
                f"- Low-signal criteria: {self.structure.low_signal_criteria}",
            ]
        if self.alignment is not None:
            lines += [
                "",
                "## Human alignment",
                f"- Correlation: {self.alignment.correlation:.3f}; "
                f"QWK: {self.alignment.qwk:.3f}; "
                f"calibration error: {self.alignment.calibration_error:.3f}",
            ]
        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_html(self, path: str | None = None) -> str:
        body = html.escape(self.to_markdown())
        verdict = html.escape(self.verdict)
        doc = (
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>Reward Report Card — {html.escape(self.rubric_name)}</title>"
            "<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;"
            "margin:2rem auto;padding:0 1rem;line-height:1.5}"
            ".verdict{font-size:1.1rem;font-weight:600;padding:1rem;border-radius:8px;"
            "background:#f4f6f8;border:1px solid #dde}"
            "pre{white-space:pre-wrap;background:#fafbfc;padding:1rem;border-radius:8px;"
            "border:1px solid #eee}</style></head><body>"
            f"<div class=\"verdict\">{verdict}</div>"
            f"<pre>{body}</pre></body></html>\n"
        )
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(doc)
        return doc
