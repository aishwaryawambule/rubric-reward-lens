from rubric_reward_lens.grader import FakeGrader, _keywords_for, aggregate
from rubric_reward_lens.models import (
    Criterion,
    CriterionScore,
    Polarity,
    Response,
    Rubric,
)


def test_aggregate_weighted_mean():
    r = Rubric([Criterion("c1", "x", weight=1.0), Criterion("c2", "y", weight=3.0)])
    scores = [CriterionScore("c1", 0.0), CriterionScore("c2", 1.0)]
    assert aggregate(r, scores) == 0.75


def test_aggregate_empty_is_zero():
    assert aggregate(Rubric([]), []) == 0.0


def test_keywords_drops_stopwords_and_short():
    kw = _keywords_for("advises seeing a doctor promptly")
    assert "doctor" in kw and "promptly" in kw
    assert "a" not in kw and "the" not in kw


def test_fakegrader_keyword_presence():
    r = Rubric([Criterion("c1", "mentions doctor"), Criterion("c2", "mentions symptoms")])
    g = FakeGrader()
    hit = g.grade(r, Response("a", "please see a doctor about your symptoms"))
    assert hit.score_for("c1") == 1.0 and hit.score_for("c2") == 1.0
    miss = g.grade(r, Response("b", "drink water and rest"))
    assert miss.score_for("c1") == 0.0
    assert 0.0 <= hit.reward <= 1.0


def test_fakegrader_avoid_polarity_inverts():
    r = Rubric([Criterion("c1", "jargon", polarity=Polarity.AVOID)])
    g = FakeGrader()
    assert g.grade(r, Response("a", "lots of jargon here")).score_for("c1") == 0.0
    assert g.grade(r, Response("b", "plain words")).score_for("c1") == 1.0


def test_fakegrader_window_ignores_late_keywords():
    r = Rubric([Criterion("c1", "doctor")])
    g = FakeGrader(window=20)
    late = Response("a", "x" * 40 + " doctor")
    assert g.grade(r, late).score_for("c1") == 0.0
    early = Response("b", "see a doctor now" + "y" * 40)
    assert g.grade(r, early).score_for("c1") == 1.0
