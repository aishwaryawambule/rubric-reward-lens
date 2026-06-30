from rubric_reward_lens.diagnostics.hacking import run_hacking
from rubric_reward_lens.grader import FakeGrader


def test_keyword_grader_is_flagged_hackable(rubric, responses):
    res = run_hacking(rubric, FakeGrader(), responses, seed=0)
    assert res.hackable is True
    assert res.overall_hack_gain > 0.05
    assert set(res.per_criterion_gameability) <= set(rubric.criterion_ids())


def test_windowed_grader_is_robust(rubric, responses):
    # window=40 ignores keywords appended at the end -> stuffing can't help.
    res = run_hacking(rubric, FakeGrader(window=40), responses, seed=0)
    assert res.hackable is False


def test_per_probe_entries_present(rubric, responses):
    res = run_hacking(rubric, FakeGrader(), responses, seed=0)
    assert set(res.per_probe) == {
        "keyword_stuff",
        "verbosity_pad",
        "confident_wrong",
        "format_mimic",
    }
    for gain, lo, hi in res.per_probe.values():
        assert lo <= gain <= hi or lo <= hi  # CI well-formed
