"""The tool's own falsification test.

Mirrors the qualitative finding of "Reproducing, Analyzing, and Detecting
Reward Hacking in Rubric-Based RL" (arXiv:2606.04923): a presence-only rubric
graded by a keyword-presence grader is gameable, while a grader that resists
appended keyword-stuffing is not. If the tool cannot separate the two, it is
not doing its job.
"""

from __future__ import annotations

from rubric_reward_lens.audit import audit
from rubric_reward_lens.grader import FakeGrader, _keywords_for, aggregate
from rubric_reward_lens.models import Criterion, CriterionScore, GraderResult, Response, Rubric

_NEGATIONS = {"never", "not", "no", "without", "none"}


class RobustContentGrader:
    """A grader genuinely resistant to the four hack probes.

    It only credits a criterion when one of its keywords appears in a *content
    sentence* (>=4 words, not a markdown bullet/header) that is not negated.
    This defeats keyword-stuffing (bare keyword lists aren't sentences),
    format-mimicry (markdown lines are ignored), verbosity padding (filler has
    no keywords) and confident-wrong claims (they contain negations), while
    still falling as real content is removed (monotonic).
    """

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        prose = [
            ln for ln in response.text.splitlines()
            if not ln.lstrip().startswith(("#", "-", "*"))
        ]
        sentences = [s for s in ". ".join(prose).replace("\n", " ").split(".")]
        content = [s for s in sentences if len(s.split()) >= 4]
        scores: list[CriterionScore] = []
        for c in rubric.criteria:
            kws = _keywords_for(c.text)
            ok = any(
                any(kw in s.lower() for kw in kws)
                and not any(neg in s.lower().split() for neg in _NEGATIONS)
                for s in content
            )
            scores.append(CriterionScore(c.id, 1.0 if ok else 0.0))
        return GraderResult(response.id, scores, aggregate(rubric, scores))


def _presence_rubric() -> Rubric:
    return Rubric(
        name="presence-only",
        criteria=[
            Criterion("c1", "mentions doctor", weight=1.0),
            Criterion("c2", "mentions symptoms", weight=1.0),
            Criterion("c3", "mentions interactions", weight=1.0),
        ],
    )


def _responses() -> list[Response]:
    return [
        Response(
            "r1",
            "You should see a doctor promptly. Watch for warning symptoms over time. "
            "Be careful about medication interactions with anything you take.",
            human_score=1.0,
        ),
        Response(
            "r2",
            "It is wise to see a doctor soon. Note any worsening symptoms carefully.",
            human_score=0.6,
        ),
        Response(
            "r3",
            "Just rest and relax quietly for now. You will probably feel fine soon.",
            human_score=0.2,
        ),
        Response(
            "r4",
            "A careful answer should mention symptoms clearly. It should flag drug "
            "interactions too. Seeing a doctor is sensible here.",
            human_score=0.8,
        ),
    ]


def test_tool_flags_hackable_reward():
    rubric, responses = _presence_rubric(), _responses()
    bad = audit(rubric, FakeGrader(), responses)  # keyword-presence: stuffable
    assert bad.hacking.hackable is True


def test_tool_clears_robust_reward():
    rubric, responses = _presence_rubric(), _responses()
    good = audit(rubric, RobustContentGrader(), responses)
    assert good.hacking.hackable is False


def test_robust_reward_scores_higher_trust():
    rubric, responses = _presence_rubric(), _responses()
    bad = audit(rubric, FakeGrader(), responses)
    good = audit(rubric, RobustContentGrader(), responses)
    assert good.trust_score > bad.trust_score
