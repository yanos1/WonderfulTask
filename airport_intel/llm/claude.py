"""Claude provider (Anthropic key: ANTHROPIC_API_KEY)."""

from __future__ import annotations

import os

from .base import RESEARCH_ALLOWED_DOMAINS, LLMProvider, Message, ResearchResult

# Research mode runs on a heavier model than the light reviser/router. The light path stays
# on Haiku (cheap, fast); the grounded path steps up so the deep web reasoning is solid.
RESEARCH_MODEL = "claude-sonnet-4-6"
WEB_SEARCH_TOOL = "web_search_20250305"
MAX_WEB_SEARCHES = 5


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

    def research(self, system: str, messages: list[Message]) -> ResearchResult:
        """Grounded reply: Claude's server-side web_search tool, restricted to the allowlist.

        Runs on RESEARCH_MODEL (Sonnet) with extended thinking so the model can reason over
        what it finds. Citations are pulled from the `citations` attached to text blocks.
        """
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self._client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 2048},
            system=system,
            messages=msgs,
            tools=[{
                "type": WEB_SEARCH_TOOL,
                "name": "web_search",
                "max_uses": MAX_WEB_SEARCHES,
                "allowed_domains": RESEARCH_ALLOWED_DOMAINS,
            }],
        )
        u = getattr(resp, "usage", None)
        if u is not None:
            # bill research tokens against the heavier model so cost metering is honest
            try:
                self.metrics.record(RESEARCH_MODEL,
                                    getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
            except Exception:  # noqa: BLE001 - metering is non-critical
                pass

        text_parts, citations, seen = [], [], set()
        for block in resp.content:
            if getattr(block, "type", "") != "text":
                continue
            text_parts.append(block.text)
            for cit in (getattr(block, "citations", None) or []):
                url = getattr(cit, "url", None) or (cit.get("url") if isinstance(cit, dict) else None)
                if not url or url in seen:
                    continue
                seen.add(url)
                title = getattr(cit, "title", None) or (cit.get("title") if isinstance(cit, dict) else None)
                snippet = (getattr(cit, "cited_text", None)
                           or (cit.get("cited_text") if isinstance(cit, dict) else None))
                citations.append({"title": title or url, "url": url, "snippet": snippet or ""})
        return ResearchResult(text="".join(text_parts), citations=citations)
