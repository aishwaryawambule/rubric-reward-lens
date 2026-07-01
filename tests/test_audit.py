import pytest

import rubric_reward_lens as rrl
from rubric_reward_lens.audit import audit
from rubric_reward_lens.grader import FakeGrader
from rubric_reward_lens.models import Response
from rubric_reward_lens.probes import Probe


def _shout_probe(response, rubric):
    return Response(
        id=f"{response.id}::shout",
        text=f"{response.text} !!!",
        prompt_id=response.prompt_id,
        prompt=response.prompt,
        human_score=response.human_score,
    )


def _len_diagnostic(rubric, grader, responses):
    from rubric_reward_lens.models import DiagnosticResult

    return DiagnosticResult(name="answer_length", score=0.42, summary="a custom check", raw={"n": 7})


def test_audit_accepts_custom_diagnostic(rubric, responses):
    # A user-supplied diagnostic runs via audit(), contributes a sub-score to the
    # trust score, shows in the report table, and its raw data lands in the JSON.
    import json

    card = audit(rubric, FakeGrader(), responses, extra_diagnostics=[_len_diagnostic])

    assert abs(card.sub_scores()["answer_length"] - 0.42) < 1e-9
    md = card.to_markdown()
    assert "answer_length" in md and "a custom check" in md
    data = json.loads(card.to_json())
    assert data["diagnostics"]["answer_length"]["raw"]["n"] == 7


def test_audit_accepts_custom_probes(rubric, responses):
    # A user-supplied probe should be used by the hacking diagnostic via audit(),
    # replacing the built-in set — the extensibility that was previously only
    # reachable by calling run_hacking directly.
    card = audit(rubric, FakeGrader(), responses, probes=[Probe("shout", "down", _shout_probe)])
    assert "shout" in card.hacking.per_probe
    assert "keyword_stuff" not in card.hacking.per_probe


def test_audit_rejects_empty_responses(rubric):
    # Auditing zero responses must not silently report "Robust" — there is no
    # evidence to make any claim about, so it is an error.
    with pytest.raises(ValueError, match="empty"):
        audit(rubric, FakeGrader(), [])


def test_audit_runs_default_diagnostics(rubric, responses):
    card = audit(rubric, FakeGrader(), responses)
    assert card.hacking is not None
    assert card.monotonicity is not None
    assert card.stability is not None
    assert card.structure is not None
    assert card.order is not None
    assert card.alignment is None


def test_audit_subset(rubric, responses):
    card = audit(rubric, FakeGrader(), responses, diagnostics=("hacking",))
    assert card.hacking is not None
    assert card.monotonicity is None


def test_audit_human_labels(rubric, responses):
    card = audit(rubric, FakeGrader(), responses, human_labels=True)
    assert card.alignment is not None


def test_public_api_surface():
    for name in ("audit", "Rubric", "Criterion", "Response", "FakeGrader", "ReportCard"):
        assert hasattr(rrl, name)
