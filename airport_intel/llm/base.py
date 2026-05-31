"""LLM provider interface + factory.

A provider exposes two primitives:
  - `complete(system, messages, json_mode)` — the cheap path that drives tool routing
    (json_mode=True) and answer synthesis (json_mode=False).
  - `research(system, messages)` — an OPTIONAL grounded path that lets the model search the
    web and return its answer WITH citations. Providers that can't ground fall back to
    `complete()` and return no citations, so the surface stays uniform.

Keeping the surface tiny is what makes Gemini and Claude cleanly interchangeable.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# message format used across the app: {"role": "user"|"assistant", "content": str}
Message = dict

# Domain allowlist for the grounded reviser. Restricting to official / primary sources is
# what turns "the model cited a random blog" into "the model cited the FAA". Providers that
# support hard domain restriction (Claude) enforce it; others (Gemini) get it as guidance.
RESEARCH_ALLOWED_DOMAINS = [
    "faa.gov",
    "bts.gov",
    "transtats.bts.gov",
    "dot.gov",
    "gao.gov",
    "wikipedia.org",
    "aviationpros.com",
    "simpleflying.com",
    "airport-technology.com",
]


@dataclass
class ResearchResult:
    """A grounded model reply: free text plus the sources it was grounded on.

    `citations` is a list of {"title", "url", "snippet"} dicts. An empty list means the
    answer was NOT web-grounded (provider fell back to plain completion).
    """

    text: str
    citations: list[dict] = field(default_factory=list)


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, system: str, messages: list[Message], json_mode: bool = False) -> str:
        """Return the model's text reply. If json_mode, the reply must be a JSON object."""

    def research(self, system: str, messages: list[Message]) -> ResearchResult:
        """Grounded reply with citations. Default: no web access — fall back to `complete`.

        Providers with native web search (Claude, Gemini) override this. The default keeps
        the agent's research path working (uncited) on any provider, so the caller never has
        to branch on capability.
        """
        return ResearchResult(text=self.complete(system, messages, json_mode=True), citations=[])

    # -- usage metering (shared by all providers) -------------------------- #
    @property
    def metrics(self):
        """Lazily resolved usage tracker; defaults to the process-wide ledger.

        Lazy so subclasses need no super().__init__() call, and so tests can inject a
        custom tracker via ``provider._metrics = MetricsTracker(path)``.
        """
        m = getattr(self, "_metrics", None)
        if m is None:
            from ..metrics import global_tracker
            m = self._metrics = global_tracker()
        return m

    def _record_usage(self, input_tokens, output_tokens) -> None:
        """Record one call's token usage. Best-effort: never let metering break an answer."""
        try:
            self.metrics.record(self.model, input_tokens or 0, output_tokens or 0)
        except Exception:  # noqa: BLE001 - metering is non-critical
            pass


# Registry of (factory, env var) keyed by provider name.
def _make_gemini(model: Optional[str]):
    from .gemini import GeminiProvider
    return GeminiProvider(model=model or "gemini-2.5-flash")


def _make_claude(model: Optional[str]):
    from .claude import ClaudeProvider
    return ClaudeProvider(model=model or "claude-haiku-4-5-20251001")


_REGISTRY = {
    "gemini": ("GEMINI_API_KEY", _make_gemini),
    "claude": ("ANTHROPIC_API_KEY", _make_claude),
}


def available_providers() -> list[str]:
    """Names of providers whose API key is present in the environment."""
    return [name for name, (env, _) in _REGISTRY.items() if os.getenv(env)]


def get_provider(name: str, model: Optional[str] = None) -> Optional[LLMProvider]:
    """Construct a provider by name, or return None if its key/SDK is unavailable."""
    entry = _REGISTRY.get(name.lower())
    if not entry:
        return None
    env, factory = entry
    if not os.getenv(env):
        return None
    try:
        return factory(model)
    except Exception:  # noqa: BLE001 - missing SDK etc. -> caller falls back
        return None
