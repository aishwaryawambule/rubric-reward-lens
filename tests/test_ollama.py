import httpx

from rubric_reward_lens.cli import _make_grader
from rubric_reward_lens.models import Criterion, Response, Rubric
from rubric_reward_lens.ollama import OllamaGrader


def _rubric():
    return Rubric([Criterion("c1", "x", weight=1.0), Criterion("c2", "y", weight=1.0)])


def test_ollama_defaults_to_local_server():
    g = OllamaGrader(model="llama3.1")
    assert "11434" in g.base_url  # the local Ollama port


def test_ollama_grade_posts_to_local_and_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"per_criterion":[{"criterion_id":"c1","score":1},{"criterion_id":"c2","score":0}]}'}}]},
        )

    g = OllamaGrader(model="llama3.1", transport=httpx.MockTransport(handler))
    gr = g.grade(_rubric(), Response("r1", "anything"))

    assert "localhost:11434" in seen["url"]
    assert gr.reward == 0.5


def test_cli_makes_ollama_grader():
    g = _make_grader({"type": "ollama", "model": "llama3.1"})
    assert isinstance(g, OllamaGrader)
