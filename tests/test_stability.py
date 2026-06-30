from rubric_reward_lens.diagnostics.stability import run_stability
from rubric_reward_lens.grader import FakeGrader


def test_deterministic_grader_is_stable(rubric, responses):
    res = run_stability(rubric, FakeGrader(), responses, n_repeats=5, seed=0)
    assert res.reward_std == 0.0
    assert all(fr == 0.0 for fr in res.per_criterion_flip_rate.values())
    assert res.stable is True


def test_noisy_grader_is_unstable(rubric, responses, noisy_grader):
    res = run_stability(rubric, noisy_grader, responses, n_repeats=8, seed=0)
    assert res.reward_std > 0.0
    assert any(fr > 0.0 for fr in res.per_criterion_flip_rate.values())
    assert res.stable is False
