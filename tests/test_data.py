from rubric_reward_lens.data import load_demo
from rubric_reward_lens.models import Response, Rubric


def test_load_demo_shape():
    rubric, responses = load_demo()
    assert isinstance(rubric, Rubric)
    assert len(rubric.criteria) >= 4
    assert len(responses) >= 8
    assert all(isinstance(r, Response) for r in responses)


def test_load_demo_has_human_scores_and_spread():
    _, responses = load_demo()
    labeled = [r for r in responses if r.human_score is not None]
    assert len(labeled) >= 4
    scores = [r.human_score for r in labeled]
    assert max(scores) >= 0.9 and min(scores) <= 0.3  # strong and weak answers present
