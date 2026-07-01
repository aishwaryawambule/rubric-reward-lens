"""Grade with a **local** model served by [Ollama](https://ollama.com).

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so this
is just :class:`~rubric_reward_lens.openrouter.OpenRouterGrader` pointed at the
local server with a throwaway key. No API key, no per-call cost, no data leaves
the machine — ideal for auditing a rubric+judge before committing to a paid API.

Prerequisite: ``ollama serve`` running and the model pulled (``ollama pull qwen2.5:14b``).

    from rubric_reward_lens import OllamaGrader, audit
    card = audit(rubric, OllamaGrader(model="qwen2.5:14b"), responses)
"""

from __future__ import annotations

import httpx

from .openrouter import OpenRouterGrader

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"


class OllamaGrader(OpenRouterGrader):
    """An OpenAI-compatible chat grader pointed at a local Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_URL,
        prompt_template: str | None = None,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            model=model,
            prompt_template=prompt_template,
            temperature=temperature,
            api_key="ollama",  # Ollama ignores the key; any non-empty value works
            transport=transport,
            base_url=base_url,
        )
