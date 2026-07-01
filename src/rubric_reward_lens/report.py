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
from .diagnostics.order import OrderResult
from .diagnostics.stability import StabilityResult
from .diagnostics.structure import StructureResult


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _what_it_means(name: str, r) -> str:
    """Plain-English reading of a diagnostic result (display-only; thresholds
    are presentation heuristics, not part of the score)."""
    if name == "hacking":
        if not r.hackable:
            return "not gameable"
        worst = max(r.per_probe, key=lambda n: r.per_probe[n][0]) if r.per_probe else "a probe"
        leaks = [
            cid
            for cid, g in sorted(r.per_criterion_gameability.items(), key=lambda kv: kv[1], reverse=True)
            if g > 0
        ][:3]
        return f"gameable via {worst}" + (f" — leakiest: {', '.join(leaks)}" if leaks else "")
    if name == "monotonicity":
        if r.monotonic and r.inversions == 0:
            return "reward falls cleanly as quality drops"
        if r.monotonic:
            return f"mostly tracks quality ({r.inversions} inversions)"
        return f"does not reliably track quality ({r.inversions} inversions)"
    if name == "stability":
        return "identical on re-grade" if r.stable else "wobbles across re-grades"
    if name == "structure":
        parts = []
        if r.low_signal_criteria:
            parts.append(f"{len(r.low_signal_criteria)} low-signal: {', '.join(r.low_signal_criteria)}")
        if r.redundant_pairs:
            parts.append(f"{len(r.redundant_pairs)} redundant pair(s)")
        return "; ".join(parts) if parts else "all criteria informative"
    if name == "criterion_order":
        if r.order_invariant:
            return "order-invariant"
        return f"reward drifts with criterion order (mean Δ {r.mean_drift:.2f})"
    if name == "alignment":
        if r.qwk >= 0.6:
            return "matches human scores"
        if r.qwk >= 0.2:
            return "weak agreement with humans"
        return "does not match human scores"
    return ""


@dataclass
class ReportCard:
    rubric_name: str
    n_responses: int
    hacking: HackingResult | None = None
    monotonicity: MonotonicityResult | None = None
    stability: StabilityResult | None = None
    structure: StructureResult | None = None
    order: OrderResult | None = None
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
        if self.order is not None:
            s["criterion_order"] = 1.0 - _clamp(self.order.mean_drift)
        if self.alignment is not None:
            s["alignment"] = _clamp(self.alignment.qwk)
        return s

    @property
    def trust_score(self) -> float:
        sub = self.sub_scores()
        return sum(sub.values()) / len(sub) if sub else 0.0

    # A hack worth a "stop and fix" headline; below this the gaming is detected
    # but minor relative to the overall picture, so it gets a softer "caution".
    SERIOUS_HACK_GAIN = 0.15

    @property
    def verdict(self) -> str:
        h = self.hacking
        if h is not None and h.hackable:
            worst = max(h.per_probe.items(), key=lambda kv: kv[1][0], default=None)
            name = worst[0] if worst else "a probe"
            serious = h.overall_hack_gain >= self.SERIOUS_HACK_GAIN or self.trust_score < 0.6
            if serious:
                return (
                    f"⚠️ Hackable — '{name}' gains +{h.overall_hack_gain:.2f} reward "
                    f"without real quality. Trust score {self.trust_score:.2f}. "
                    f"Fix the rubric before training."
                )
            return (
                f"⚠️ Caution — '{name}' can game +{h.overall_hack_gain:.2f}, but overall "
                f"trust score {self.trust_score:.2f}. Review the probes below."
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
                "criterion_order": conv(self.order) if self.order else None,
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
        results = {
            "hacking": self.hacking,
            "monotonicity": self.monotonicity,
            "stability": self.stability,
            "structure": self.structure,
            "criterion_order": self.order,
            "alignment": self.alignment,
        }
        lines = [
            f"# Reward Report Card — {self.rubric_name}",
            "",
            f"**{self.verdict}**",
            "",
            f"- Responses audited: {self.n_responses}",
            f"- Composite trust score: **{self.trust_score:.2f}**  (0–1, higher is better)",
            "",
            "## Diagnostics",
            "",
            "| Diagnostic | Score | What it means |",
            "| --- | --- | --- |",
        ]
        for name, score in self.sub_scores().items():
            lines.append(f"| {name} | {score:.2f} | {_what_it_means(name, results[name])} |")
        lines += [
            "",
            "Raw metrics (Spearman, hack gain, per-criterion gameability, CIs, QWK) "
            "are in the JSON output — run with `--out report.json`.",
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
