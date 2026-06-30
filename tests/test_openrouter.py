import json

import httpx

from rubric_reward_lens.models import Criterion, Response, Rubric
from rubric_reward_lens.openrouter import OpenRouterGrader, parse_grader_json


def _rubric():
    return Rubric([Criterion("c1", "x", weight=1.0), Criterion("c2", "y", weight=1.0)])


def test_parse_plain_json():
    raw = '{"per_criterion":[{"criterion_id":"c1","score":1,"justification":"ok"},{"criterion_id":"c2","score":0}]}'
    gr = parse_grader_json(raw, _rubric(), "r1")
    assert gr.score_for("c1") == 1.0 and gr.score_for("c2") == 0.0
    assert gr.reward == 0.5


def test_parse_code_fenced_with_prose():
    raw = "Here you go:\n```json\n{\"per_criterion\":[{\"criterion_id\":\"c1\",\"score\":0.5}]}\n```\nthanks"
    gr = parse_grader_json(raw, _rubric(), "r1")
    assert gr.score_for("c1") == 0.5


def test_parse_missing_criterion_defaults_zero():
    gr = parse_grader_json('{"per_criterion":[{"criterion_id":"c1","score":1}]}', _rubric(), "r1")
    assert gr.score_for("c2") == 0.0


def test_parse_out_of_range_clamped():
    raw = '{"per_criterion":[{"criterion_id":"c1","score":5},{"criterion_id":"c2","score":-3}]}'
    gr = parse_grader_json(raw, _rubric(), "r1")
    assert gr.score_for("c1") == 1.0 and gr.score_for("c2") == 0.0


def test_parse_garbage_is_all_zero():
    gr = parse_grader_json("not json at all", _rubric(), "r1")
    assert gr.reward == 0.0


def test_grade_with_mock_transport():
    body = {
        "choices": [
            {
                "message": {
                    "content": '{"per_criterion":[{"criterion_id":"c1","score":1},{"criterion_id":"c2","score":1}]}'
                }
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    grader = OpenRouterGrader(
        model="test/model", api_key="x", transport=httpx.MockTransport(handler)
    )
    gr = grader.grade(_rubric(), Response("r1", "anything"))
    assert gr.reward == 1.0
