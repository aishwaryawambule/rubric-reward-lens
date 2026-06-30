import json

import pytest
import yaml

from rubric_reward_lens.models import (
    Criterion,
    CriterionScore,
    GraderResult,
    Polarity,
    Response,
    Rubric,
)


def test_import_package():
    import rubric_reward_lens  # noqa: F401


def test_rubric_from_dict_coerces_types():
    r = Rubric.from_dict(
        {"name": "r", "criteria": [{"id": "c1", "text": "mentions a doctor", "weight": 2}]}
    )
    assert r.name == "r"
    assert len(r.criteria) == 1
    c = r.criteria[0]
    assert c.weight == 2.0 and isinstance(c.weight, float)
    assert c.polarity is Polarity.INCLUDE
    assert r.criterion_ids() == ["c1"]


def test_rubric_polarity_avoid():
    r = Rubric.from_dict(
        {"criteria": [{"id": "c1", "text": "avoid jargon", "polarity": "avoid"}]}
    )
    assert r.criteria[0].polarity is Polarity.AVOID


def test_rubric_get_and_missing():
    r = Rubric([Criterion("c1", "x")])
    assert r.get("c1").text == "x"
    with pytest.raises(KeyError):
        r.get("nope")


def test_rubric_from_json(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"criteria": [{"id": "a", "text": "t"}]}))
    r = Rubric.from_json(str(p))
    assert r.criterion_ids() == ["a"]


def test_rubric_from_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"criteria": [{"id": "a", "text": "t", "weight": 3}]}))
    r = Rubric.from_yaml(str(p))
    assert r.get("a").weight == 3.0


def test_grader_result_score_for():
    gr = GraderResult("r1", [CriterionScore("c1", 0.7), CriterionScore("c2", 0.2)], 0.45)
    assert gr.score_for("c1") == 0.7
    with pytest.raises(KeyError):
        gr.score_for("zzz")


def test_response_optional_human_score():
    assert Response("r", "txt").human_score is None
    assert Response("r", "txt", human_score=0.5).human_score == 0.5
