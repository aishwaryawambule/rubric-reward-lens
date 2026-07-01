from rubric_reward_lens.diagnostics.order import OrderResult, run_criterion_order
from rubric_reward_lens.grader import FakeGrader, aggregate
from rubric_reward_lens.models import Criterion, CriterionScore, GraderResult, Response, Rubric


def _rubric(weights=(1.0, 1.0, 1.0)):
    return Rubric(
        criteria=[
            Criterion("see_doctor", "advises seeing a doctor", weight=weights[0]),
            Criterion("symptoms", "lists warning symptoms", weight=weights[1]),
            Criterion("interactions", "warns about drug interactions", weight=weights[2]),
        ],
        name="t",
    )


def _responses():
    return [
        Response("r1", "see a doctor, watch symptoms, mind interactions"),
        Response("r2", "just rest"),
        Response("r3", "see a doctor about interactions"),
    ]


class FirstIdGrader:
    """Order-SENSITIVE: reward is 1.0 only when 'see_doctor' is listed first — a
    pure criterion-position bias. Reverse order always moves it, so it is always
    detected."""

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        reward = 1.0 if rubric.criteria[0].id == "see_doctor" else 0.0
        scores = [CriterionScore(c.id, reward) for c in rubric.criteria]
        return GraderResult(response.id, scores, reward)


def test_keyword_grader_is_order_invariant():
    res = run_criterion_order(_rubric(), FakeGrader(), _responses(), k=3, seed=0)
    assert isinstance(res, OrderResult)
    assert res.order_invariant is True
    assert res.mean_drift == 0.0


def test_order_sensitive_grader_is_flagged():
    res = run_criterion_order(_rubric(), FirstIdGrader(), _responses(), k=3, seed=0)
    assert res.order_invariant is False
    assert res.mean_drift > 0.05
    assert res.flip_rate > 0.0  # pass/fail vector also changes


def test_single_criterion_is_trivially_invariant():
    r = Rubric(criteria=[Criterion("c1", "x")], name="t")
    res = run_criterion_order(r, FirstIdGrader(), _responses(), k=3, seed=0)
    assert res.order_invariant is True
    assert res.mean_drift == 0.0


def test_empty_responses_is_invariant():
    res = run_criterion_order(_rubric(), FakeGrader(), [], k=3, seed=0)
    assert res.order_invariant is True and res.n == 0
