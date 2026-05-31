"""Gemini provider (Google AI Studio free-tier key: GEMINI_API_KEY)."""

from __future__ import annotations

import os

from .base import RESEARCH_ALLOWED_DOMAINS, LLMProvider, Message, ResearchResult

# Research mode steps up from the light flash model to Pro, so the grounded deep web
# reasoning is solid. The light router/reviser stay on flash (cheap, fast).
RESEARCH_MODEL = "gemini-2.5-pro"


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

    def research(self, system: str, messages: list[Message]) -> ResearchResult:
        """Grounded reply: Gemini's Google-Search grounding on RESEARCH_MODEL (Pro).

        Unlike Claude's web_search, Google-Search grounding can't HARD-restrict to a domain
        allowlist, so we pass the allowlist into the system prompt as guidance and trust the
        cite-or-discard step in the agent to drop anything off-list. Citations come from the
        response's grounding_metadata.
        """
        allow = ", ".join(RESEARCH_ALLOWED_DOMAINS)
        system = (system + f"\n\nPrefer these authoritative domains when searching: {allow}. "
                  "Treat any other source with skepticism.")
        model = self._genai.GenerativeModel(
            RESEARCH_MODEL,
            system_instruction=system,
            generation_config={"temperature": 0.2},
            tools="google_search_retrieval",
        )
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages
        ]
        resp = model.generate_content(contents)
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            # bill research tokens against the heavier model so cost metering is honest
            try:
                self.metrics.record(RESEARCH_MODEL,
                                    getattr(u, "prompt_token_count", 0),
                                    getattr(u, "candidates_token_count", 0))
            except Exception:  # noqa: BLE001 - metering is non-critical
                pass
        return ResearchResult(text=_gemini_text(resp), citations=_gemini_citations(resp))


def _gemini_text(resp) -> str:
    """Pull plain text out of a Gemini response without raising on a blocked/empty reply."""
    try:
        return resp.text
    except Exception:  # noqa: BLE001 - .text raises if no candidates; degrade to ""
        parts = []
        for cand in getattr(resp, "candidates", None) or []:
            for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts)


def _gemini_citations(resp) -> list[dict]:
    """Extract {title, url, snippet} from Gemini's grounding_metadata, de-duplicated by URL."""
    citations, seen = [], set()
    for cand in getattr(resp, "candidates", None) or []:
        gm = getattr(cand, "grounding_metadata", None)
        for chunk in getattr(gm, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            url = getattr(web, "uri", None)
            if not url or url in seen:
                continue
            seen.add(url)
            title = getattr(web, "title", None)
            citations.append({"title": title or url, "url": url, "snippet": ""})
    return citations
