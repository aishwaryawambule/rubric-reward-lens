from rubric_reward_lens.grader import _keywords_for
from rubric_reward_lens.models import Criterion, Response, Rubric
from rubric_reward_lens.probes import (
    HACK_PROBES,
    confident_wrong,
    degradation_ladder,
    format_mimic,
    keyword_stuff,
    verbosity_pad,
)


def _rubric():
    return Rubric(
        [
            Criterion("c1", "advises seeing a doctor"),
            Criterion("c2", "lists warning symptoms"),
        ]
    )


def _resp():
    return Response("r1", "Rest well. Stay hydrated. Take it easy for a few days.")


def test_keyword_stuff_inserts_all_keywords_and_grows():
    out = keyword_stuff(_resp(), _rubric())
    for kw in _keywords_for("advises seeing a doctor") + _keywords_for("lists warning symptoms"):
        assert kw in out.text.lower()
    assert len(out.text) > len(_resp().text)
    assert out.id == "r1::keyword_stuff"


def test_verbosity_pad_grows():
    out = verbosity_pad(_resp(), _rubric())
    assert len(out.text) > len(_resp().text)
    assert out.id == "r1::verbosity_pad"


def test_confident_wrong_adds_claim():
    out = confident_wrong(_resp(), _rubric())
    assert "certain" in out.text.lower()
    assert len(out.text) > len(_resp().text)


def test_format_mimic_adds_structure():
    out = format_mimic(_resp(), _rubric())
    assert "##" in out.text and "- " in out.text


def test_degradation_ladder_monotone_shrink():
    resp = Response("r1", "One. Two. Three. Four. Five. Six. Seven. Eight.")
    ladder = degradation_ladder(resp, rungs=4)
    assert len(ladder) == 4
    assert ladder[0].text == resp.text
    lengths = [len(r.text) for r in ladder]
    assert lengths == sorted(lengths, reverse=True)
    assert lengths[0] > lengths[-1]
    assert len({r.id for r in ladder}) == 4


def test_hack_probes_registry():
    names = {p.name for p in HACK_PROBES}
    assert names == {"keyword_stuff", "verbosity_pad", "confident_wrong", "format_mimic"}
    for p in HACK_PROBES:
        assert p.expected_direction == "down"
        assert p.apply(_resp(), _rubric()).id.startswith("r1::")
