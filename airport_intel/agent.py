"""Agent orchestration: route -> execute (deterministic) -> revise (bounded LLM) -> narrate.

The LLM appears only at the edges:
  1. ROUTE    - pick a tool + args from natural language (json_mode)
  2. REVISE   - optional bounded, justified score modifier on a ranking shortlist
  3. NARRATE  - turn the deterministic tool result into a professional answer

Everything in the middle (scoring, ranking) is deterministic Python. An LLM provider is
required -- the agent does not fall back to a keyword router or templated answers.
Conversation history is retained for follow-up questions.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from . import scoring
from . import tools
from .llm import LLMProvider

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM = """You route airport-investment questions to ONE deterministic tool.
Tools:
- rank_region(region, limit): rank airports by Expansion Profitability Index. region is a
  US region name like "New England" or null for nationwide. Use for "which airports are
  strong candidates / best / top".
- compare_airports(codes, metric): compare 2+ airports on a metric. metric is one of
  congestion, volume, long_haul, growth, runways. codes are airport names or IATA codes.
- airport_profile(code): full profile + unmet-demand explanation for one airport. Use for
  "unmet demand", "tell me about", "why".
- flight_breakdown(code): long-haul flight share for one airport. Use for "long haul %".
- explain: the user asks HOW a score was computed, the methodology, or "why these
  numbers/scores" about a PREVIOUS answer. No args.
- none: the message is a greeting, thanks, smalltalk, or NOT about US airport investment
  (e.g. "hi", "how are you", "what's the capital of France"). Use this whenever no tool
  clearly applies. NEVER force a ranking or any other tool onto a non-airport message.

Given the conversation, output JSON ONLY:
{"tool": "<tool name or 'none'>", "args": {...}, "assumptions": ["..."]}
Resolve cities to IATA codes when you know them (Los Angeles->LAX, Santa Ana->SNA,
Anchorage->ANC, San Francisco->SFO, Boston->BOS). For follow-ups, use prior context to
fill missing airports/regions."""

METHODOLOGY = """How scores are computed (deterministic Python, not the LLM):
- EPI = demand x feasibility x 100.
- demand = weighted blend of percentile-normalized sub-scores: load_factor (0.40),
  growth (0.25), volume (0.25), long_haul (0.10), delay (0.10) -- weights renormalized over
  whichever components have data (e.g. delay has no data yet, so its weight redistributes).
  Each sub-score is the airport's PERCENTILE RANK of that metric across ~900 US airports
  (so 0.93 means 93rd percentile).
- feasibility = runway count + longest-runway percentile, mapped into [0.3, 1] and used
  as a MULTIPLIER/gate: an airport with little physical room is downgraded even if demand
  is high (this is why a busy but constrained airport like SNA scores lower).
- FinalScore = EPI x clamp(LLM modifier, 1 +/- lambda*0.30). With lambda=0 the LLM has no
  effect and FinalScore == EPI; otherwise the LLM may nudge within the band WITH a stated
  reason, shown transparently.
Data vintage: volume + growth are FAA CY2024; load factor + long-haul are 2013 BTS T-100."""

EXPLAIN_SYSTEM = """You are an airport-investment analyst explaining HOW the deterministic
scores were computed. Use the methodology below and the per-airport sub-scores from the
previous result (provided as JSON). Walk through the math concretely for the airports in
question (demand blend -> x feasibility -> x100), using their actual sub-score numbers.
Do not invent numbers. Be clear and concise.

""" + METHODOLOGY

NARRATOR_SYSTEM = """You are an airport-investment analyst. Write a concise, professional
answer to the user's question grounded in the provided tool result JSON.

Lead with the conclusion -- what the numbers MEAN for an investment decision, not the
numbers themselves. Then support that judgment with the few figures that actually drive
it. Aim for interpretation first, evidence second:
- INSTEAD OF "load factor 0.93, growth 0.81, volume 0.77, EPI 71" -- say what that
  implies: this airport is running near capacity with rising demand, so expansion has a
  clear runway to pay off; then cite the one or two numbers that prove it.
- Translate percentiles into plain meaning (e.g. 0.93 = "busier than ~93% of US
  airports"). Explain WHY one airport ranks above another, don't just list both scores.
- Only quote numbers that support a point. Do not enumerate every sub-score.

Never invent figures; use only what's in the JSON. If the JSON has a "notes" list,
briefly surface the relevant caveat (data vintage / limitations) so the user understands
uncertainty. Keep it tight -- a short paragraph or a small list -- but every figure you
cite should earn its place by backing an insight."""

REVISER_SYSTEM = """You may apply a small qualitative adjustment to deterministic airport
scores, based on real-world knowledge the formula cannot see (e.g. recently announced
terminal funding, known geographic/runway constraints, regulatory limits).
For each airport you wish to adjust, return a modifier and a SPECIFIC factual reason.
A modifier of 1.0 means no change. You MUST justify any modifier != 1.0; unjustified
adjustments are discarded. Output JSON ONLY:
{"modifiers": [{"iata": "SFO", "modifier": 1.08, "reason": "..."}]}"""

HELP_SYSTEM = """You are the assistant for an Airport Investment Intelligence tool. The
user's latest message is NOT a data request — it may be a greeting, a thank-you, or an
off-topic question. Reply in 1-3 short, warm sentences: greet back if they greeted you,
briefly explain that you help identify US airports where renovation or expansion is most
likely to be profitable (based on passenger and flight capacity), and invite a relevant
question. If the message is off-topic, gently steer back to airport investment. Do NOT
invent any airport names, numbers, or scores. You may suggest one example, e.g. "Which
airports in New England are strong candidates for expansion?\""""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> Optional[dict]:
    """Lenient JSON extraction (handles stray prose / markdown fences)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


_DISPATCH = {
    "rank_region": lambda a: tools.rank_region(a.get("region"), int(a.get("limit", 5) or 5)),
    "compare_airports": lambda a: tools.compare_airports(a.get("codes", []), a.get("metric", "congestion")),
    "airport_profile": lambda a: tools.airport_profile(a.get("code", "")),
    "flight_breakdown": lambda a: tools.flight_breakdown(a.get("code", "")),
}


class Agent:
    def __init__(self, provider: LLMProvider, lam: float = 0.0):
        if provider is None:
            raise ValueError(
                "Agent requires an LLM provider; the deterministic fallback was removed."
            )
        self.provider = provider
        self.lam = lam
        self.history: list[dict] = []  # [{role, content}] for multi-turn follow-ups
        self.last_result: Optional[dict] = None  # last data result, for "explain these scores"

    # -- routing ---------------------------------------------------------- #
    def _route(self, user_message: str) -> dict:
        """The LLM picks one tool (+args) from the message and conversation history.

        A malformed/unparseable reply defaults to "none" (the LLM help path). There is no
        deterministic keyword router -- an LLM provider is always required.
        """
        msgs = self.history + [{"role": "user", "content": user_message}]
        raw = self.provider.complete(ROUTER_SYSTEM, msgs, json_mode=True)
        parsed = _parse_json(raw)
        if parsed and parsed.get("tool"):
            return parsed
        return {"tool": "none", "args": {}}

    # -- bounded reviser -------------------------------------------------- #
    def _revise(self, result: dict) -> dict:
        """Apply a bounded, justified LLM modifier to a ranking shortlist (if λ>0)."""
        if self.lam <= 0 or result.get("intent") != "rank":
            return result
        shortlist = result.get("results", [])
        if not shortlist:
            return result
        prompt = [{"role": "user", "content": json.dumps(
            {"airports": [{"iata": r["iata"], "name": r["name"], "epi": r["epi"],
                           "subscores": r["subscores"]} for r in shortlist]})}]
        try:
            raw = self.provider.complete(REVISER_SYSTEM, prompt, json_mode=True)
            mods = {m["iata"]: m for m in (_parse_json(raw) or {}).get("modifiers", [])}
        except Exception:
            mods = {}
        for r in shortlist:
            m = mods.get(r["iata"], {})
            adj = scoring.apply_modifier(r["epi"], m.get("modifier"), self.lam, m.get("reason"))
            r["final_score"] = adj["final_score"]
            r["modifier"] = adj["modifier"]
            r["modifier_reason"] = adj["reason"]
        shortlist.sort(key=lambda r: r["final_score"], reverse=True)
        result["reviser_applied"] = True
        return result

    # -- narration -------------------------------------------------------- #
    def _narrate(self, user_message: str, result: dict) -> str:
        msgs = [{"role": "user", "content":
                 f"Question: {user_message}\n\nTool result JSON:\n{json.dumps(result)}"}]
        return self.provider.complete(NARRATOR_SYSTEM, msgs).strip()

    # -- explain (methodology for the previous scores) -------------------- #
    def _explain(self, user_message: str) -> str:
        prev = self.last_result
        ctx = json.dumps(prev) if prev else "(no previous scored result this session)"
        msgs = [{"role": "user", "content":
                 f"Question: {user_message}\n\nPrevious result JSON:\n{ctx}"}]
        return self.provider.complete(EXPLAIN_SYSTEM, msgs).strip()

    # -- help / smalltalk (greeting, thanks, or off-topic) ---------------- #
    def _help(self, user_message: str) -> str:
        return self.provider.complete(
            HELP_SYSTEM, [{"role": "user", "content": user_message}]
        ).strip()

    # -- public ----------------------------------------------------------- #
    def ask(self, user_message: str) -> dict:
        route = self._route(user_message)
        tool = route.get("tool")
        if tool in _DISPATCH:
            result = _DISPATCH[tool](route.get("args", {}))
            result = self._revise(result)
            self.last_result = result  # remember for a later "explain these scores"
            answer = self._narrate(user_message, result)
        elif tool == "explain":
            result = {"intent": "explain"}
            answer = self._explain(user_message)
        else:  # tool == "none" / unrecognized -> greeting, thanks, or off-topic
            result = {"intent": "smalltalk"}
            answer = self._help(user_message)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "route": route, "result": result}
