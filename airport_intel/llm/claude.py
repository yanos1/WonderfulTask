"""Claude provider (Anthropic key: ANTHROPIC_API_KEY)."""

from __future__ import annotations

import os

from .base import LLMProvider, Message


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, messages: list[Message], json_mode: bool = False) -> str:
        if json_mode:
            system = system + "\n\nRespond with ONLY a valid JSON object — no prose, no markdown fences."
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.2,
            system=system,
            messages=msgs,
        )
        u = getattr(resp, "usage", None)  # Anthropic returns exact token counts
        if u is not None:
            self._record_usage(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
