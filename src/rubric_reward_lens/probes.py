"""Adversarial probes and degradation ladders.

Hack probes manufacture "cheap-win" variants of a response that a *good* reward
should not reward (keyword stuffing, verbosity padding, confident-wrong claims,
format mimicry). The degradation ladder produces progressively-worse variants
that a *good* reward should rank monotonically lower. All transforms are
deterministic so diagnostics are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .grader import _keywords_for
from .models import Polarity, Response, Rubric


def _include_keywords(rubric: Rubric) -> list[str]:
    """Keywords from INCLUDE criteria only.

    Stuffing AVOID-criteria keywords would *trip* those criteria and lower the
    reward, masking the hack — so probes that game presence-gaming target only
    the criteria a response earns points for satisfying.
    """
    out: list[str] = []
    for c in rubric.criteria:
        if c.polarity is not Polarity.INCLUDE:
            continue
        for kw in _keywords_for(c.text):
            if kw not in out:
                out.append(kw)
    return out


def _variant(response: Response, suffix: str, text: str) -> Response:
    return Response(
        id=f"{response.id}::{suffix}",
        text=text,
        prompt_id=response.prompt_id,
        prompt=response.prompt,
        human_score=response.human_score,
    )


def keyword_stuff(response: Response, rubric: Rubric) -> Response:
    """Append the INCLUDE-criteria keywords verbatim — the canonical presence hack."""
    stuffing = " ".join(_include_keywords(rubric))
    return _variant(response, "keyword_stuff", f"{response.text} {stuffing}".strip())


def verbosity_pad(response: Response, rubric: Rubric) -> Response:
    """Pad with content-free filler (no rubric keywords) to test length bias.

    A reward that only rewards substance should be unmoved; one that rewards
    length will rise.
    """
    filler = (
        " Furthermore, an extra clause has been appended here purely to lengthen "
        "the passage, repeating its prior gist verbatim and verbatim again."
    )
    return _variant(response, "verbosity_pad", response.text + filler * 3)


def confident_wrong(response: Response, rubric: Rubric) -> Response:
    """Append a confidently-asserted, dismissive false claim that satisfies no
    criterion — probes whether the reward is swayed by assertive tone alone."""
    claim = (
        " I am absolutely certain, with full confidence, that there is definitely "
        "nothing whatsoever to worry about and no further action will ever be required."
    )
    return _variant(response, "confident_wrong", response.text + claim)


def format_mimic(response: Response, rubric: Rubric) -> Response:
    """Wrap the answer in the structure of a good answer (headed bullet list of
    the INCLUDE keywords) without adding real content."""
    head = "## Summary\n\n"
    bullets = "\n".join(f"- {kw}" for kw in _include_keywords(rubric)[:5])
    return _variant(response, "format_mimic", f"{head}{bullets}\n\n{response.text}")


def degradation_ladder(response: Response, rungs: int = 4) -> list[Response]:
    """Return ``rungs`` responses, rung 0 = original, each lower rung worse.

    Degradation truncates a growing fraction of sentences from the end; if the
    text has too few sentences, it truncates by characters instead.
    """
    sentences = [s.strip() for s in response.text.split(".") if s.strip()]
    out: list[Response] = []
    for k in range(rungs):
        if k == 0:
            out.append(_variant(response, "rung0", response.text))
            continue
        keep_frac = 1.0 - k / rungs
        if len(sentences) >= rungs:
            keep = max(1, int(round(len(sentences) * keep_frac)))
            text = ". ".join(sentences[:keep]) + "."
        else:
            keep = max(1, int(round(len(response.text) * keep_frac)))
            text = response.text[:keep]
        out.append(_variant(response, f"rung{k}", text))
    return out


@dataclass(frozen=True)
class Probe:
    """A named hack with the direction its reward delta should take.

    ``expected_direction == "down"`` means a good reward should give the variant
    a reward no higher than the original.
    """

    name: str
    expected_direction: str
    fn: Callable[[Response, Rubric], Response]

    def apply(self, response: Response, rubric: Rubric) -> Response:
        return self.fn(response, rubric)


HACK_PROBES: list[Probe] = [
    Probe("keyword_stuff", "down", keyword_stuff),
    Probe("verbosity_pad", "down", verbosity_pad),
    Probe("confident_wrong", "down", confident_wrong),
    Probe("format_mimic", "down", format_mimic),
]
