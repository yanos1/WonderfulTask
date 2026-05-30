"""Gemini provider (Google AI Studio free-tier key: GEMINI_API_KEY)."""

from __future__ import annotations

import os

from .base import LLMProvider, Message


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash"):
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._genai = genai
        self.model = model

    def complete(self, system: str, messages: list[Message], json_mode: bool = False) -> str:
        cfg = {"temperature": 0.2}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        model = self._genai.GenerativeModel(self.model, system_instruction=system, generation_config=cfg)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages
        ]
        resp = model.generate_content(contents)
        u = getattr(resp, "usage_metadata", None)  # Gemini returns token counts here
        if u is not None:
            self._record_usage(
                getattr(u, "prompt_token_count", 0),
                getattr(u, "candidates_token_count", 0),
            )
        return resp.text
