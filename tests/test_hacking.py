from rubric_reward_lens.diagnostics.hacking import run_hacking
from rubric_reward_lens.grader import FakeGrader
from rubric_reward_lens.probes import Probe, keyword_stuff


def test_keyword_grader_is_flagged_hackable(rubric, hackable_responses):
    res = run_hacking(rubric, FakeGrader(), hackable_responses, seed=0)
    assert res.hackable is True
    assert res.overall_hack_gain > 0.05
    assert set(res.per_criterion_gameability) <= set(rubric.criterion_ids())


def test_robust_grader_is_not_flagged(rubric, hackable_responses, robust_grader):
    # a grader that requires real prose (not keyword lists / markdown / tone)
    # cannot be gamed by any of the four probes.
    res = run_hacking(rubric, robust_grader, hackable_responses, seed=0)
    assert res.hackable is False


def test_per_probe_entries_present(rubric, hackable_responses):
    res = run_hacking(rubric, FakeGrader(), hackable_responses, seed=0)
    assert set(res.per_probe) == {
        "keyword_stuff",
        "verbosity_pad",
        "confident_wrong",
        "format_mimic",
    }
    for gain, lo, hi in res.per_probe.values():
        assert lo <= hi  # CI is well-formed
        assert -1.0 <= gain <= 1.0  # gain is a bounded reward delta


def test_keyword_stuff_drives_the_hack(rubric, hackable_responses):
    # the canonical presence hack must produce a positive gain on a
    # keyword-presence grader (it's what makes the grader "hackable").
    probe = [Probe("keyword_stuff", "down", keyword_stuff)]
    res = run_hacking(rubric, FakeGrader(), hackable_responses, probes=probe, seed=0)
    assert res.overall_hack_gain > 0.0


def test_length_only_probes_do_not_falsely_trigger(rubric, hackable_responses):
    # verbosity padding and confident-but-wrong tone carry no rubric keywords,
    # so a keyword grader must NOT reward them (no false hack signal).
    res = run_hacking(rubric, FakeGrader(), hackable_responses, seed=0)
    assert res.per_probe["verbosity_pad"][0] == 0.0
    assert res.per_probe["confident_wrong"][0] == 0.0
