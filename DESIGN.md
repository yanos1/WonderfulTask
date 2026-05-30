# Design & Architecture

## 1. Problem & approach

The firm invests in US airport modernization and wants to find airports where expansion
yields the most return on **passenger/flight capacity**. The core thesis:

> The best renovation target is an airport with **high demand pressure** (full planes,
> growth) **and physical room to grow**. An airport that is bursting at the seams but
> cannot physically expand is a poor investment despite the demand.

The system is a **tool-calling agent**: the LLM understands the question and routes it to
**deterministic Python tools** that compute the ranking; the LLM then explains the result.
This cleanly satisfies the brief's requirement for *deterministic ranking, not just LLM
output*, while keeping the conversation natural.

```
User → Streamlit chat
  1. ROUTE   (LLM)  NL + history → {tool, args}        ← or deterministic keyword fallback
  2. EXECUTE (code) deterministic tool computes EPI, ranking, breakdowns + a `steps` trace
  3. REVISE  (LLM)  optional bounded, justified score modifier on a ranking shortlist
  4. NARRATE (LLM)  professional answer grounded strictly in the returned numbers
```

Conversation history is retained (`st.session_state`) and passed to the router every turn,
so follow-ups like *"and what about SFO?"* resolve against prior context.

## 2. Scoring methodology — Expansion Profitability Index (EPI)

```
demand     = weighted, percentile-normalized blend of { load factor, growth, volume }
EPI        = demand × Feasibility × 100                       Feasibility ∈ [0.3, 1]
FinalScore = EPI × clamp(LLMModifier, 1 − λ·0.30, 1 + λ·0.30)
```

**Demand (the pressure signal).** Each component is **percentile-normalized across all ~900
airports**, so the EPI is a stable absolute index and any subset (e.g. one region) is
directly comparable. Load factor is the centerpiece — full planes mean demand is being
turned away. Components with no data anywhere (currently `delay`) are dropped and their
weight **redistributed**, so the formula stays correct as data sources are added. (This is
exactly how `growth` went live the moment FAA enplanements were wired in.)

**Feasibility (the gate).** A multiplier in `[0.3, 1]` from runway count + longest runway
(proxy for physical room/infrastructure). As a *multiplier*, it can veto: an airport with
high demand but low feasibility (e.g. **SNA**) is correctly downgraded — you can't expand
what has no room. It's floored at 0.3 because it's our **weakest-data** factor; flooring
lets it dampen without nuking a score, and we state its low confidence to the user.

**Why percentile (not min-max).** Robust to mega-hub outliers (one ATL won't crush the
scale) and reads naturally ("93rd-percentile load factor").

**Unmet demand (Q4).** A standalone indicator = blend of high load factor + growth, returned
with a plain-language "why", so *"unmet demand in SFO and why"* is answered with numbers,
not vibes.

## 3. Where & how AI is used

The LLM is confined to the **semantic edges**; all math is deterministic Python.

| Stage | LLM? | What it does | Deterministic guardrail |
|------|------|--------------|--------------------------|
| Route | yes | NL + history → tool + args (JSON) | falls back to keyword routing if absent |
| Execute | **no** | EPI, ranking, breakdowns | the graded "non-LLM" logic |
| Revise | yes | bounded score modifier + **justification** | clamp to `1 ± λ·0.30`; **blank reason ⇒ discarded**; λ=0 ⇒ no effect; failure ⇒ 1.0 |
| Narrate | yes | prose answer | instructed to use only returned numbers |

**The bounded reviser** is the interesting bit. The deterministic EPI owns the ranking and
is fully reproducible. The LLM may add a *small, justified* nudge for qualitative signal the
formula can't see (announced funding, geographic constraints) — but it is **clamped by a
user-controlled dial λ** (0 = off, 1 = ±30% max), **must cite a reason** (unjustified
adjustments are dropped), and is shown transparently in the UI as `EPI → modifier(+reason)
→ FinalScore`. So the LLM is a *governed* scoring contributor, never an override.

> This satisfies both halves of the brief: **"not only LLM"** (a reproducible deterministic
> spine) and **"AI-powered"** (the LLM genuinely participates in routing, scoring, and
> explanation).

## 4. Data storage

**Access pattern first.** The runtime is **read-only** over ~900 airport records (~340 KB),
with **no concurrent writes** and **no transactions** — every query is "load the universe,
filter, score." For that shape, the right store is a **single JSON artifact built offline by
the ETL** (`etl/build_airports.py` → `data/airports.json`), loaded once into memory. A
relational/vector DB here would be complexity the access pattern doesn't earn. The store is a
*materialized view* of public sources, rebuilt by re-running the ETL — not an authored DB.

**The seam is the point.** All runtime code depends only on the `Repository` ABC
(`repository.py`), never on JSON. `get`/`all`/`find` are the entire contract, so the backing
store is a **one-file swap** — `JsonRepository` → a SQLite/Postgres implementation — with
**zero changes to tools or scoring**. That's where indexing and a real query engine would
land if the dataset grew orders of magnitude.

**Typed, validated records.** Records are not loose dicts: each is parsed into an immutable
`Airport` (`models.py`) at the load boundary, where `Airport.from_raw` enforces the schema
(IATA shape, `load_factor`/`long_haul_pct ∈ [0,1]`, non-negative counts) and **fails loud,
naming the offending field**, instead of letting a bad value silently corrupt a score. The
ETL validates with the *same* model before publishing, so the schema has one source of truth.
`Airport` subclasses `Mapping`, so it stays a drop-in for the by-name field access the
deterministic layer uses while giving new code typed attribute access.

**Provenance.** The artifact is wrapped as `{"_meta": {...}, "airports": {...}}`; `_meta`
records `schema_version`, `generated_at`, `record_count`, the per-signal vintages (volume/
growth = 2024, structural ratios = 2013), and the source URLs — so any dataset is auditable
and reproducible (`repo.meta` exposes it).

**Owned tradeoffs.** (1) The derived `airports.json` is **committed to git** — a deliberate
choice so the app is demoable immediately without running the ETL or hitting the network;
the ETL remains the source of truth and can regenerate it. (2) `Repository.find` is a
**linear scan** — `O(n)` over 10³ rows is negligible, and the `Repository` seam is exactly
where an index would go if `n` grew.

## 5. Key tradeoffs

- **Tool-routing via structured JSON, not vendor function-calling APIs.** One tiny provider
  interface (`complete`) makes Gemini and Claude fully interchangeable. Tradeoff: we manage
  routing ourselves instead of using each SDK's native tools — worth it for portability and
  a clean swap.
- **Graceful degradation everywhere.** No API key → keyword routing + templated answers.
  Live/current data unavailable → fall back to baseline and *disclose it*. The app is always
  demoable; resilience doubles as honest uncertainty communication.
- **Derived data over a hand-curated seed.** Metrics are computed from bulk public data for
  ~900 airports, so regional ranking generalizes and "unmet demand" is *derived*, not
  authored. Tradeoff: more ETL plumbing up front.
- **Feasibility is a proxy.** True terminal/land capacity isn't in clean public data; we
  proxy with runways and flag it as low-confidence rather than overclaim.
- **Scope discipline (deliberately cut):** cargo analysis, multi-agent pipelines, a vector
  DB, and any distributed infrastructure — none earn their complexity in a prototype. The
  production path (gateway + queue + workers + DB behind the same `Repository`/`LLMProvider`
  seams) is described, not built.

## 6. Assumptions, uncertainty & scoping

- **Mixed vintage (deliberate):** volume + YoY growth are **current** (FAA CY2024
  enplanements, 2024 vs 2023); load factor and long-haul % are from **2013** BTS T-100
  segment data (the most recent reliably/programmatically downloadable segment-level source
  with seats + distance). Industry load factors have risen since, so absolute values are
  conservative, but the *relative* ranking the EPI uses remains informative. ETL is
  year-parameterized.
- **Long-haul is domestic-only** — international segments not yet ingested — so it understates
  long-haul share at international gateways (ANC, SFO).
- **Resilience:** the 2024 NTAD path we first tried is a restricted hosted view that rejects
  feature queries; the ETL falls back per-airport to the 2013 baseline and records the
  vintage, rather than failing. (Volume/growth now come from FAA; the fallback remains for
  airports absent from the FAA file.)
- **Cargo out of scope:** the thesis is passenger/terminal expansion; all-cargo service
  classes are filtered out, so cargo-dominant airports (ANC) are judged on passenger terms.
- **Feasibility ≠ true headroom:** runways proxy for physical room.

## 7. With more time
1. **International T-100 segments** → accurate long-haul at gateways.
2. **Live delays** (AeroDataBox) promoted from display-only to a real EPI signal.
3. **A current-year segment source** → bring load factor to 2024 (removes the mixed vintage).
4. **RAG over master plans / news** to ground the reviser's qualitative adjustments.
5. **Voice input**, trend charts, and per-component weight sliders in the UI.
