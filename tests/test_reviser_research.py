"""Tests for the agent's two-mode bounded reviser: light vs. grounded research.

These use fake providers (no network / no API key) so they exercise the agent's branching,
cite-or-discard discipline, citation plumbing, graceful fallback, and history threading —
all the seams research mode adds — without hitting a real model.
"""

import json

import pytest

from airport_intel.agent import Agent
from airport_intel.llm.base import LLMProvider, ResearchResult


class _FakeProvider(LLMProvider):
    """Records calls and returns canned payloads for both primitives."""

    name = "fake"
    model = "fake-model"

    def __init__(self, complete_payload=None, research_payload=None, citations=None,
                 research_error=None):
        self._complete_payload = complete_payload if complete_payload is not None else {"airports": []}
        self._research_payload = research_payload if research_payload is not None else {"airports": []}
        self._citations = citations or []
        self._research_error = research_error
        self.complete_calls = []
        self.research_calls = []

    def complete(self, system, messages, json_mode=False):
        self.complete_calls.append({"system": system, "messages": messages})
        return json.dumps(self._complete_payload)

    def research(self, system, messages):
        self.research_calls.append({"system": system, "messages": messages})
        if self._research_error is not None:
            raise self._research_error
        return ResearchResult(text=json.dumps(self._research_payload),
                              citations=list(self._citations))


def _rank_result(epi=70.0):
    return {"intent": "rank",
            "results": [{"iata": "SFO", "name": "San Francisco", "epi": epi, "subscores": {}}]}


# -- research path: cited, clamped, sources attached ------------------------ #
def test_research_mode_applies_cited_nudge_and_attaches_sources():
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "FAA awarded a new terminal concourse", "impact": 0.08,
         "source": "FAA ATP FY2024", "url": "https://faa.gov/x"}]}]}
    citations = [{"title": "FAA ATP", "url": "https://faa.gov/x", "snippet": "..."}]
    provider = _FakeProvider(research_payload=payload, citations=citations)
    agent = Agent(provider, lam=1.0, research_mode=True)

    out = agent._revise(_rank_result(70.0))
    row = out["results"][0]

    assert provider.research_calls and not provider.complete_calls  # grounded path used
    assert row["modifier"] == 1.08
    assert row["final_score"] == 75.6                  # 70 * 1.08
    assert row["factors"][0]["url"] == "https://faa.gov/x"
    assert out["reviser_mode"] == "research"
    assert out["sources"] == citations


def test_sources_fall_back_to_factor_urls_when_provider_cites_nothing():
    # JSON-only research replies often carry NO provider text-citations, yet every kept
    # factor is sourced. The Sources panel must still populate from the factors' own urls,
    # and the Why summary must name the source inline.
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "new concourse funded", "impact": 0.06,
         "source": "FAA ATP FY2024", "url": "https://faa.gov/atp"}]}]}
    provider = _FakeProvider(research_payload=payload, citations=[])  # provider cites nothing
    agent = Agent(provider, lam=1.0, research_mode=True)

    out = agent._revise(_rank_result(100.0))
    row = out["results"][0]

    assert out["sources"] == [{"title": "FAA ATP FY2024", "url": "https://faa.gov/atp",
                               "snippet": "new concourse funded"}]
    assert "[FAA ATP FY2024]" in row["modifier_reason"]   # source named inline in Why column


def test_why_column_falls_back_to_domain_when_no_source_name():
    # model gave a url but no "source" field -> inline label uses the bare domain
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "runway extension approved", "impact": 0.04,
         "url": "https://www.faa.gov/news/x"}]}]}
    provider = _FakeProvider(research_payload=payload, citations=[])
    agent = Agent(provider, lam=1.0, research_mode=True)

    row = agent._revise(_rank_result(100.0))["results"][0]
    assert "[faa.gov]" in row["modifier_reason"]          # www. stripped, path dropped


def test_research_named_source_without_url_is_kept_and_listed():
    # Gemini case: model names a source in prose but returns no structured URL, and the
    # provider attaches no text-citations. The nudge must still apply and the named source
    # must appear in the Sources panel (url-less).
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "DOT announced a $5M terminal grant", "impact": 0.05,
         "source": "U.S. DOT 2024"}]}]}
    provider = _FakeProvider(research_payload=payload, citations=[])
    agent = Agent(provider, lam=1.0, research_mode=True)

    out = agent._revise(_rank_result(100.0))
    row = out["results"][0]

    assert row["modifier"] == 1.05                       # named source counts
    assert out["sources"] == [{"title": "U.S. DOT 2024", "url": "",
                               "snippet": "DOT announced a $5M terminal grant"}]
    assert "[U.S. DOT 2024]" in row["modifier_reason"]


def test_research_mode_discards_uncited_factor():
    # one cited, one not — only the cited one may move the score
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "uncited rumor", "impact": 0.20},
        {"reason": "cited fact", "impact": 0.05, "source": "FAA", "url": "https://faa.gov/y"}]}]}
    provider = _FakeProvider(research_payload=payload,
                             citations=[{"title": "t", "url": "https://faa.gov/y"}])
    agent = Agent(provider, lam=1.0, research_mode=True)

    row = agent._revise(_rank_result(100.0))["results"][0]
    assert row["modifier"] == 1.05                      # uncited 0.20 dropped
    assert len(row["factors"]) == 1


# -- light path regression -------------------------------------------------- #
def test_light_mode_uses_complete_not_research():
    payload = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "model-knowledge nudge", "impact": 0.10}]}]}
    provider = _FakeProvider(complete_payload=payload)
    agent = Agent(provider, lam=1.0, research_mode=False)

    out = agent._revise(_rank_result(100.0))
    row = out["results"][0]

    assert provider.complete_calls and not provider.research_calls
    assert row["modifier"] == 1.10                      # unsourced nudge still counts (light)
    assert "sources" not in out
    assert out.get("reviser_mode") != "research"


# -- graceful fallback ------------------------------------------------------ #
def test_research_failure_falls_back_to_light_path():
    light = {"airports": [{"iata": "SFO", "factors": [
        {"reason": "fallback nudge", "impact": 0.05}]}]}
    provider = _FakeProvider(complete_payload=light,
                             research_error=RuntimeError("network down"))
    agent = Agent(provider, lam=1.0, research_mode=True)

    out = agent._revise(_rank_result(100.0))             # must not raise
    row = out["results"][0]

    assert provider.research_calls and provider.complete_calls  # tried research, fell back
    assert row["modifier"] == 1.05
    assert "sources" not in out


def test_default_research_without_web_support_degrades_to_no_nudge():
    # A provider that only implements complete() inherits base.research() -> complete() with
    # zero citations. In research mode every factor is uncited -> all discarded -> no nudge.
    class _LightOnly(LLMProvider):
        name = "light"
        model = "m"

        def complete(self, system, messages, json_mode=False):
            return json.dumps({"airports": [{"iata": "SFO", "factors": [
                {"reason": "no url here", "impact": 0.10}]}]})

    agent = Agent(_LightOnly(), lam=1.0, research_mode=True)
    out = agent._revise(_rank_result(100.0))
    row = out["results"][0]
    assert row["modifier"] == 1.0                        # uncited -> discarded
    assert "sources" not in out


# -- lambda gate still holds in research mode ------------------------------- #
def test_research_mode_noop_at_lambda_zero():
    provider = _FakeProvider(research_payload={"airports": [{"iata": "SFO", "factors": [
        {"reason": "x", "impact": 0.2, "source": "s", "url": "https://faa.gov/z"}]}]})
    agent = Agent(provider, lam=0.0, research_mode=True)
    out = agent._revise(_rank_result(100.0))
    assert not provider.research_calls                   # λ=0 short-circuits before any call
    assert "final_score" not in out["results"][0]


# -- history threading into narrate/explain --------------------------------- #
def test_narrate_includes_conversation_history():
    provider = _FakeProvider()
    agent = Agent(provider, lam=0.0)
    agent.history = [{"role": "user", "content": "earlier question"},
                     {"role": "assistant", "content": "earlier answer"}]
    agent._narrate("now this", {"intent": "rank", "results": []})

    sent = provider.complete_calls[-1]["messages"]
    contents = [m["content"] for m in sent]
    assert "earlier question" in contents                # prior turns carried into narration
    assert any("now this" in c for c in contents)


def test_explain_includes_conversation_history():
    provider = _FakeProvider()
    agent = Agent(provider, lam=0.0)
    agent.history = [{"role": "user", "content": "prior turn"}]
    agent.last_result = {"intent": "rank", "results": []}
    agent._explain("how were these computed?")

    sent = provider.complete_calls[-1]["messages"]
    assert any(m["content"] == "prior turn" for m in sent)
