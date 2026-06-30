from rubric_reward_lens.diagnostics.monotonicity import run_monotonicity


def test_length_grader_is_monotonic(rubric, responses, length_grader):
    res = run_monotonicity(rubric, length_grader, responses, rungs=4, seed=0)
    assert res.spearman <= -0.5
    assert res.monotonic is True
    assert res.separation > 0
    assert res.inversions == 0


def test_constant_grader_not_monotonic(rubric, responses, constant_grader):
    res = run_monotonicity(rubric, constant_grader, responses, rungs=4, seed=0)
    assert res.monotonic is False
    assert abs(res.spearman) < 0.5
