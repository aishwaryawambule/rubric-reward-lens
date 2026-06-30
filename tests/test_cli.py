import json

import pytest

from rubric_reward_lens.cli import main


def test_demo_writes_html(tmp_path):
    out = tmp_path / "card.html"
    rc = main(["demo", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "trust score" in text.lower() or "robust" in text.lower() or "hackable" in text.lower()


def test_audit_writes_json(tmp_path):
    # write a rubric, a fake-grader config, and responses, then audit.
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps({"name": "t", "criteria": [{"id": "c1", "text": "mentions doctor"}]}))
    grader = tmp_path / "grader.json"
    grader.write_text(json.dumps({"type": "fake"}))
    responses = tmp_path / "resp.json"
    responses.write_text(
        json.dumps({"responses": [{"id": "r1", "text": "see a doctor"}, {"id": "r2", "text": "rest up"}]})
    )
    out = tmp_path / "card.json"
    rc = main(["audit", "--rubric", str(rubric), "--grader", str(grader), "--responses", str(responses), "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["rubric_name"] == "t"


def test_unknown_command_errors():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
