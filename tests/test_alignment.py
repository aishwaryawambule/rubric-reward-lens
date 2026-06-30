import pytest

from rubric_reward_lens.diagnostics.alignment import run_alignment
from rubric_reward_lens.models import Response


def test_perfect_alignment(rubric, responses, human_echo_grader):
    res = run_alignment(rubric, human_echo_grader, responses, n_bands=5, seed=0)
    assert res.correlation > 0.8
    assert res.qwk > 0.8
    assert res.calibration_error < 1e-9
    assert res.n == len([r for r in responses if r.human_score is not None])


def test_unlabeled_responses_raise(rubric, human_echo_grader):
    unlabeled = [Response("r1", "x"), Response("r2", "y")]
    with pytest.raises(ValueError):
        run_alignment(rubric, human_echo_grader, unlabeled)


def test_ignores_unlabeled(rubric, human_echo_grader):
    mixed = [
        Response("r1", "a", human_score=1.0),
        Response("r2", "b"),
        Response("r3", "c", human_score=0.0),
    ]
    res = run_alignment(rubric, human_echo_grader, mixed)
    assert res.n == 2
