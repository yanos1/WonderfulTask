"""Agent orchestration: route -> execute (deterministic) -> revise (bounded LLM) -> narrate.

The LLM appears only at the edges:
  1. ROUTE    - pick a tool + args from natural language (json_mode)
  2. REVISE   - optional bounded, justified score modifier on a ranking shortlist
  3. NARRATE  - turn the deterministic tool result into a professional answer

Everything in the middle (scoring, ranking) is deterministic Python. If no LLM provider
is configured, the agent degrades gracefully to a keyword router + templated answers, so
the app is always usable. Conversation history is retained for follow-up questions.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import scoring
import tools
from llm import LLMProvider
from regions import REGIONS, resolve_region

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

Given the conversation, output JSON ONLY:
{"tool": "<tool name or 'none'>", "args": {...}, "assumptions": ["..."]}
Resolve cities to IATA codes when you know them (Los Angeles->LAX, Santa Ana->SNA,
Anchorage->ANC, San Francisco->SFO, Boston->BOS). For follow-ups, use prior context to
fill missing airports/regions."""

METHODOLOGY = """How scores are computed (deterministic Python, not the LLM):
- EPI = demand x feasibility x 100.
- demand = weighted blend of percentile-normalized sub-scores: load_factor (0.40),
  growth (0.25), volume (0.25), delay (0.10) -- weights renormalized over whichever
  components have data. Each sub-score is the airport's PERCENTILE RANK of that metric
  across ~900 US airports (so 0.93 means 93rd percentile).
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
answer to the user's question using ONLY the numbers in the provided tool result JSON.
Never invent figures. Lead with the direct answer. If the JSON has a "notes" list,
briefly surface the relevant caveat (data vintage / limitations) so the user understands
uncertainty. Keep it tight (a short paragraph or a small list)."""

REVISER_SYSTEM = """You may apply a small qualitative adjustment to deterministic airport
scores, based on real-world knowledge the formula cannot see (e.g. recently announced
terminal funding, known geographic/runway constraints, regulatory limits).
For each airport you wish to adjust, return a modifier and a SPECIFIC factual reason.
A modifier of 1.0 means no change. You MUST justify any modifier != 1.0; unjustified
adjustments are discarded. Output JSON ONLY:
{"modifiers": [{"iata": "SFO", "modifier": 1.08, "reason": "..."}]}"""


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
    def __init__(self, provider: Optional[LLMProvider] = None, lam: float = 0.0):
        self.provider = provider
        self.lam = lam
        self.history: list[dict] = []  # [{role, content}] for multi-turn follow-ups
        self.last_result: Optional[dict] = None  # last data result, for "explain these scores"

    # -- routing ---------------------------------------------------------- #
    def _route(self, user_message: str) -> dict:
        if self.provider:
            msgs = self.history + [{"role": "user", "content": user_message}]
            try:
                raw = self.provider.complete(ROUTER_SYSTEM, msgs, json_mode=True)
                parsed = _parse_json(raw)
                if parsed and parsed.get("tool"):
                    return parsed
            except Exception:
                pass  # fall through to deterministic routing
        return self._fallback_route(user_message)

    def _fallback_route(self, msg: str) -> dict:
        low = msg.lower()
        if (("how" in low or "why" in low or "explain" in low)
                and ("score" in low or "epi" in low or "comput" in low or "reach" in low)):
            return {"tool": "explain", "args": {}}
        if "long haul" in low or "long-haul" in low:
            return {"tool": "flight_breakdown", "args": {"code": self._guess_airport(msg)}}
        if " vs " in low or "compare" in low:
            return {"tool": "compare_airports",
                    "args": {"codes": self._guess_airports(msg), "metric": self._guess_metric(low)}}
        if "unmet" in low or "profile" in low or "tell me about" in low or "why" in low:
            return {"tool": "airport_profile", "args": {"code": self._guess_airport(msg)}}
        return {"tool": "rank_region", "args": {"region": self._guess_region(low), "limit": 5}}

    # -- naive entity guessers (only used when no LLM is configured) ------- #
    @staticmethod
    def _guess_region(low: str) -> Optional[str]:
        for region in REGIONS:
            if region.replace("-", " ") in low:
                return region
        return None

    @staticmethod
    def _guess_airports(msg: str) -> list[str]:
        codes = re.findall(r"\b[A-Z]{3}\b", msg)
        if len(codes) >= 2:
            return codes[:2]
        # fall back to resolving notable tokens
        found = []
        for token in re.split(r"[,/]| vs | and ", msg, flags=re.IGNORECASE):
            c = tools.resolve_code(token.strip())
            if c and c not in found:
                found.append(c)
        return found[:2]

    @staticmethod
    def _guess_airport(msg: str) -> str:
        codes = re.findall(r"\b[A-Z]{3}\b", msg)
        if codes:
            return codes[0]
        for token in re.findall(r"[A-Za-z][A-Za-z ]+", msg):
            c = tools.resolve_code(token.strip())
            if c:
                return c
        return ""

    @staticmethod
    def _guess_metric(low: str) -> str:
        for m in ("congestion", "volume", "long_haul", "growth", "runways"):
            if m.replace("_", " ") in low or m in low:
                return m
        return "congestion"

    # -- bounded reviser -------------------------------------------------- #
    def _revise(self, result: dict) -> dict:
        """Apply a bounded, justified LLM modifier to a ranking shortlist (if λ>0)."""
        if not self.provider or self.lam <= 0 or result.get("intent") != "rank":
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
        if self.provider:
            msgs = [{"role": "user", "content":
                     f"Question: {user_message}\n\nTool result JSON:\n{json.dumps(result)}"}]
            try:
                return self.provider.complete(NARRATOR_SYSTEM, msgs).strip()
            except Exception:
                pass
        return self._fallback_narrate(result)

    @staticmethod
    def _fallback_narrate(result: dict) -> str:
        if result.get("error"):
            return result["error"]
        intent = result.get("intent")
        if intent == "rank":
            lines = [f"Top candidates{' in ' + result['region'] if result.get('region') else ''}:"]
            for r in result["results"]:
                score = r.get("final_score", r["epi"])
                lines.append(f"  - {r['iata']} ({r['city']}): EPI {score}")
            return "\n".join(lines)
        if intent == "compare":
            parts = [f"{r['iata']}: {r.get('display','n/a')}" for r in result["results"] if r.get("iata")]
            return f"Comparison on {result['metric']} - " + "; ".join(parts)
        if intent == "profile":
            u = result["unmet_demand"]
            return (f"{result['iata']} ({result['city']}): EPI {result['epi']}, "
                    f"unmet-demand {u['unmet_demand_score']}. " + "; ".join(u["reasons"]))
        if intent == "flight_breakdown":
            return f"{result['iata']}: long-haul share {result['long_haul_display']}."
        return "I couldn't determine how to answer that."

    # -- explain (methodology for the previous scores) -------------------- #
    def _explain(self, user_message: str) -> str:
        prev = self.last_result
        if self.provider:
            ctx = json.dumps(prev) if prev else "(no previous scored result this session)"
            msgs = [{"role": "user", "content":
                     f"Question: {user_message}\n\nPrevious result JSON:\n{ctx}"}]
            try:
                return self.provider.complete(EXPLAIN_SYSTEM, msgs).strip()
            except Exception:
                pass
        # deterministic fallback: spell out the math from the stored sub-scores
        lines = [METHODOLOGY]
        for r in (prev or {}).get("results", [])[:5] if prev else []:
            subs = ", ".join(f"{k} {v}" for k, v in r.get("subscores", {}).items())
            lines.append(f"\n{r['iata']}: demand={r.get('demand')} × feasibility="
                         f"{r.get('feasibility')} × 100 = EPI {r.get('epi')}  (sub-scores: {subs})")
        return "\n".join(lines)

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
        else:
            result = {"intent": "none", "note": "No matching tool."}
            answer = self._narrate(user_message, result)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "route": route, "result": result}
