# Design & Architecture

## 1. Problem & approach

The firm invests in US airport modernization and wants to find airports where expansion
yields the most return on **passenger/flight capacity**. The core thesis:

> The best renovation target is an airport with **high demand pressure** (full planes,
> growth) **and physical room to grow**. An airport that is bursting at the seams but
> cannot physically expand is a poor investment despite the demand.

The system is a **tool-calling agent**. The LLM understands the question and routes it to
**deterministic Python tools** that compute the ranking. An optional LLM step then revises
the score, within tight bounds, using real-world knowledge the data is blind to. Finally
the LLM explains the result in plain English.

This keeps the ranking deterministic and reproducible — the brief asked for *deterministic
ranking, not just LLM output* — while the conversation still feels natural. The LLM only
ever lives at the edges (read the question, nudge the score, explain it); it never computes
the score itself.

```
User → Streamlit chat
  1. ROUTE   (LLM)   history → {tool, args}
  2. EXECUTE (code)  deterministic tool computes EPI, ranking, breakdowns + a `steps` trace
  3. REVISE  (LLM)   optional, bounded, justified score nudge — from model knowledge
                     (light mode) or cited web research (research mode)
  4. NARRATE (LLM)   professional answer grounded strictly in the returned numbers
```


## 2. Scoring methodology — Expansion Profitability Index (EPI)

```
demand     = weighted, percentile-normalized blend of { load factor, growth, volume, long haul }
EPI        = demand × Feasibility × 100                       Feasibility ∈ [0.3, 1]
FinalScore = EPI × clamp(LLMModifier, 1 − λ·0.40, 1 + λ·0.40)
```

A weighted blend of percentile-normalized components —
load factor (0.40), growth (0.25), volume (0.25), long-haul mix (0.10), and delay (0.10) —
each **percentile-normalized across the full ~1,000-airport universe**, so the EPI is a
stable absolute index and any subset (e.g. one region) is directly comparable. Load factor
is the centerpiece — full planes mean demand is being turned away; long-haul mix is a small
route-quality nudge (long-haul skews to higher revenue/margin).

**Feasibility (the gate).** A multiplier in `[0.3, 1]` from runway count + longest runway
(proxy for physical room/infrastructure). Not accurate, ccouldnt find the data in time.

LLM Part - Inroducing "mutation" into scores based on llm knoleedge\actual data with citation.
The nudge is a list of small, separately-justified factors.
The user has complete control of this mutation factor - letting the llm decide on 0 up to 0.4 of the score.



## 4. Where & how AI is used

- **Writing the code / tests** — AI pair-programmed most of the implementation and the unit
  tests; I designed, reviewed, and corrected.
- **Design** — used AI as a sounding board for the architecture and the scoring formula.
- **Data sourcing & preprocessing** — AI helped locate the FAA/BTS/OurAirports datasets and
  write the ETL that normalizes them into `data/airports.json`.
- **At runtime** — the LLM does routing, the bounded score nudge, and the natural-language
  explanation. It never computes a score.


## 5. Key tradeoffs

**Baked JSON snapshot vs. live data APIs.** I ship a pre-built `data/airports.json` instead
of calling FAA/BTS at request time.
- Pro: fast, deterministic, no network failures, fully reproducible — the same question
  always gives the same score, which matters for a scoring tool.
- Con: the data is stale until I re-run the ETL. For an annual investment screen that's
  fine; the underlying datasets only update yearly anyway.

**Deterministic retrieval vs. RAG / semantic search. My data is small and structured —
~1,000 airports with numeric fields. I don't need semantic search; I need exact figures and
a formula. RAG would be overkill and would add a hallucination surface, the opposite of
what I want. So scoring is deterministic Python, and the LLM may only mutate it within
bounds.
Research mode is the one place I do go to the open web for grounding — that path is
genuinely retrieval-augmented, with cite-or-discard enforcement so an uncited claim can't
move a score.



## 6. Assumptions, uncertainty & scoping

- **Data vintage:** volume + YoY growth are **current** (FAA CY2024 enplanements, 2024 vs
  2023); load factor and long-haul % are from **2024** BTS T-100 Domestic Segment data;
  airport/runway metadata is from OurAirports.
- **Long-haul is domestic-only** — international segments aren't ingested yet — so it
  understates long-haul share at international gateways (ANC, SFO).
- **Feasibility is a proxy.** Runway count + length is a rough stand-in for "room to expand";
  it doesn't capture land availability, zoning, or environmental constraints. That's why it's
  floored and flagged as low-confidence, not used as a hard gate.
- **Deployment:** local, single-process. Deliberate choice for a 24h limit — no value in
  debugging RabbitMQ and DB queries when the goal is to prove the scoring approach works.


## 7. With more time

1. Improve agentic loop, allow multiple tool calling and maybe a rapch like agent structure where agents can talk to earlier
2. agents for validation and deeper insight.
1. **Better feasibility data** — land/zoning/airspace constraints, and live signals like
   flight delays. This is the weakest factor today, so it's the highest-leverage fix.
2. **RAG over unstructured sources** — exactly the feasibility data above (airport master
   plans, FAA NEPA/environmental filings, local zoning) lives in long documents, not tables.
   That's the case where semantic retrieval finally earns its place, unlike the structured
   data I have now.
3. **Make it a real product** — gateway + queue + workers + a database for multi-user
   concurrency and persistent history.
4. Focus a tiny bit more on user experience. It's already good but not perfect.
5. logging system so i can trace the exact path of what happens to each messege and analyze for better agentic loop and debugging.
6. trace when the keys are running low and add funding when they deplete.