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


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return str(p)


def test_audit_malformed_responses_prints_clean_error(tmp_path, capsys):
    # A response missing the required "text" field must produce a clean
    # 'error: ...' message and a non-zero exit, not a raw Python traceback.
    rubric = _write(tmp_path / "rubric.json", {"name": "t", "criteria": [{"id": "c1", "text": "doctor"}]})
    grader = _write(tmp_path / "grader.json", {"type": "fake"})
    responses = _write(tmp_path / "resp.json", [{"id": "r1"}])  # no "text"

    rc = main(["audit", "--rubric", rubric, "--grader", grader, "--responses", responses])

    err = capsys.readouterr().err
    assert rc == 1
    assert err.lower().startswith("error:")
    assert "Traceback" not in err


def test_audit_unknown_grader_type_prints_clean_error(tmp_path, capsys):
    rubric = _write(tmp_path / "rubric.json", {"name": "t", "criteria": [{"id": "c1", "text": "doctor"}]})
    grader = _write(tmp_path / "grader.json", {"type": "nonsense"})
    responses = _write(tmp_path / "resp.json", [{"id": "r1", "text": "see a doctor"}])

    rc = main(["audit", "--rubric", rubric, "--grader", grader, "--responses", responses])

    err = capsys.readouterr().err
    assert rc == 1
    assert err.lower().startswith("error:")
    assert "Traceback" not in err
