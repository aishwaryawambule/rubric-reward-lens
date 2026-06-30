from rubric_reward_lens.diagnostics.structure import run_structure
from rubric_reward_lens.models import Criterion, CriterionScore, GraderResult, Response, Rubric


class ScriptedGrader:
    """Returns fixed per-criterion scores keyed by response id."""

    def __init__(self, table: dict[str, dict[str, float]]):
        self.table = table

    def grade(self, rubric: Rubric, response: Response) -> GraderResult:
        row = self.table[response.id]
        scores = [CriterionScore(c.id, row[c.id]) for c in rubric.criteria]
        from rubric_reward_lens.grader import aggregate

        return GraderResult(response.id, scores, aggregate(rubric, scores))


def test_redundancy_low_signal_and_coverage():
    rubric = Rubric(
        [Criterion("c1", "a"), Criterion("c2", "b"), Criterion("c3", "c")]
    )
    # c1 and c2 always co-fire; c3 always passes (low signal).
    table = {
        "r1": {"c1": 1.0, "c2": 1.0, "c3": 1.0},
        "r2": {"c1": 0.0, "c2": 0.0, "c3": 1.0},
        "r3": {"c1": 1.0, "c2": 1.0, "c3": 1.0},
        "r4": {"c1": 0.0, "c2": 0.0, "c3": 1.0},
    }
    responses = [Response(rid, "x") for rid in table]
    res = run_structure(rubric, ScriptedGrader(table), responses)

    pairs = {(a, b) for a, b, _ in res.redundant_pairs}
    assert ("c1", "c2") in pairs
    assert "c3" in res.low_signal_criteria
    assert res.coverage["c3"] == 1.0
    assert res.coverage["c1"] == 0.5
    assert set(res.weight_sensitivity) == {"c1", "c2", "c3"}
