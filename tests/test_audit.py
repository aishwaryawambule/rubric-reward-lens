import rubric_reward_lens as rrl
from rubric_reward_lens.audit import audit
from rubric_reward_lens.grader import FakeGrader


def test_audit_runs_default_diagnostics(rubric, responses):
    card = audit(rubric, FakeGrader(), responses)
    assert card.hacking is not None
    assert card.monotonicity is not None
    assert card.stability is not None
    assert card.structure is not None
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
