# Design & Architecture

## 1. Problem & approach

The firm invests in US airport modernization and wants to find airports where expansion
yields the most return on **passenger/flight capacity**. The core thesis:

> The best renovation target is an airport with **high demand pressure** (full planes,
> growth) **and physical room to grow**. An airport that is bursting at the seams but
> cannot physically expand is a poor investment despite the demand.

The system is a **tool-calling agent**: the LLM understands the question and routes it to
**deterministic Python tools** that compute the ranking; then llm call revises the score based on finding it has that our
data was blind to. Last but not least, the LLM explains the result .
This cleanly satisfies the brief's requirement for *deterministic ranking, not just LLM
output*, while keeping the conversation natural.

```
User → Streamlit chat
  1. ROUTE   (LLM)   history → {tool, args}
  2. EXECUTE (code) deterministic tool computes EPI, ranking, breakdowns + a `steps` trace
  3. REVISE  (LLM)  optional bounded, justified score modifier based on knowledge (baisc search) 
  or deep research and citation (research mode)
  4. NARRATE (LLM)  professional answer grounded strictly in the returned numbers
```


## 2. Scoring methodology — Expansion Profitability Index (EPI)

```
demand     = weighted, percentile-normalized blend of { load factor, growth, volume, long haul }
EPI        = demand × Feasibility × 100                       Feasibility ∈ [0.3, 1]
FinalScore = EPI × clamp(LLMModifier, 1 − λ·0.40, 1 + λ·0.40)
```

**Demand (the pressure signal).** A weighted blend of percentile-normalized components —
load factor (0.40), growth (0.25), volume (0.25), long-haul mix (0.10), and delay (0.10) —
each **percentile-normalized across all ~900 airports**, so the EPI is a stable absolute
index and any subset (e.g. one region) is directly comparable. Load factor is the centerpiece
— full planes mean demand is being turned away; long-haul mix is a small route-quality nudge
(long-haul skews to higher revenue/margin). Components with no data anywhere (currently
`delay`) are dropped and their
weight **redistributed**, so the formula stays correct as data sources are added. (This is
exactly how `growth` went live the moment FAA enplanements were wired in.)

**Feasibility (the gate).** A multiplier in `[0.3, 1]` from runway count + longest runway
(proxy for physical room/infrastructure). As a *multiplier*, it can veto: an airport with
high demand but low feasibility (e.g. **SNA**) is correctly downgraded — you can't expand
what has no room. It's floored at 0.3 because it's our **weakest-data** factor; flooring
lets it dampen without nuking a score, and we state its low confidence to the user.

**Why percentile (not min-max).** Robust to mega-hub outliers (one ATL won't crush the
scale) and reads naturally ("93rd-percentile load factor").


## 3. Where & how AI is used

The LLM is confined to the **semantic edges**; all math is deterministic Python.

| Stage | LLM? | What it does | Deterministic guardrail |
|------|------|--------------|--------------------------|
| Route | yes | NL + history → tool + args (JSON) | falls back to keyword routing if absent |
| Execute | **no** | EPI, ranking, breakdowns | the graded "non-LLM" logic |
| Revise | yes | bounded score modifier + **justification** | clamp to `1 ± λ·0.30`; **blank reason ⇒ discarded**; λ=0 ⇒ no effect; failure ⇒ 1.0 |
| Narrate | yes | prose answer | instructed to use only returned numbers |

**The bounded reviser** is the interesting bit. The deterministic EPI owns the ranking and
is fully reproducible. The LLM may mutate this score based on knowledge or citations


## 5. Key tradeoffs


- Json registry vs API:
  - Pros
  - saves run time, no failure, deterministic
  - Cons
  - Stale data until loaded new sources.
  
- RAG retrieval of data
    on a vendor's tool schema; the payoff is portability, no lock-in to one function-calling
    dialect, and a router that's just a `string → JSON` function we can unit-test directly.
- **A built JSON artifact, not a database.** (see §4) The runtime is read-only over ~900
  records (~340 KB) with no concurrent writes — a pure load → filter → score shape. A single
  ETL-built JSON loaded into memory beats standing up SQLite/Postgres: zero ops, instant cold
  start, git-diffable provenance. The cost is no indexing or ad-hoc query engine — but the
  `Repository` ABC is precisely the seam where a real DB drops in if the data grows orders of
  magnitude, with no change to tools or scoring.
- **Deterministic spine, LLM only at the edges.** The model never produces the ranking or any
  number — it picks a tool + args, then a separate pass narrates strictly from the values the
  Python tool returned. This is the brief's *"not only an LLM"* requirement made literal: the
  scoring is reproducible and testable, and the model is confined to intent→tool, numbers→prose,
  and a *bounded, justified* nudge (§3). The tradeoff is a routing layer to maintain versus
  letting the model free-form — worth it because outputs stay auditable and scores stay stable
  across runs and across providers.
- **No RAG / vector store.** Our knowledge base is ~900 *structured* rows with a fixed schema,
  not a document corpus. The correct retrieval primitive for structured data is a query
  (filter/sort by field), not semantic similarity — RAG would bolt on an embedding model, a
  vector DB, and chunking to answer what is really a `WHERE`/`ORDER BY`. Retrieval earns its
  place only once we ingest the *unstructured* sources in §7 (master plans, filings, news) to
  ground the reviser; until then it is pure complexity. Open-ended general-knowledge questions
  are already handled by the `aviation_qa` route from the model's parametric knowledge.

## 6. Assumptions, uncertainty & scoping

- **Current throughout:** volume + YoY growth are **current** (FAA CY2024 enplanements,
  2024 vs 2023); load factor and long-haul % are from **2024** BTS T-100 Domestic Segment
  data, OurAirports for metadata of airports and runways
- **Long-haul is domestic-only** — international segments not yet ingested — so it understates
  long-haul share at international gateways (ANC, SFO).
- **Local, single-process deployment.** The system runs as one Streamlit process — no queue,
  service mesh, or external DB — which is the right scope for a prototype that must be
  demoable in a single command. It maps cleanly onto the production path: the same
  `Repository` and `LLMProvider` seams sit behind a gateway + workers + managed DB without
  touching the scoring core. State is per-session (`st.session_state`); there is no
  multi-user persistence because the demo is single-user by design.
- **Cargo out of scope:** the thesis is passenger/terminal expansion; all-cargo service
  classes are filtered out, so cargo-dominant airports (ANC) are judged on passenger terms.
- **Feasibility ≠ true headroom:** true terminal/land/gate capacity isn't in clean public
  data, so we proxy with runway count + longest runway and flag it as our lowest-confidence
  factor (floored at 0.3) rather than overclaim — gathering real capacity data was out of
  reach in the timeframe.

## 7. With more time
1. **More signals, weights re-fit empirically.** Ingest international T-100 segments (accurate
   long-haul at gateways), real gate/terminal capacity, and on-time performance — then *fit*
   the demand weights against the ATP backtest instead of asserting them. The backtest harness
   already turns precision@N into an objective function; this closes the loop.
2. **Live delays** (AeroDataBox) promoted from display-only to a real EPI signal — the `delay`
   component already exists in the blend with auto-redistributed weight, so wiring a source
   activates it with no formula change.
3. **A real datastore behind the same seam.** Swap the committed JSON for SQLite/Postgres via a
   new `Repository` implementation — adds indexing, ad-hoc queries, and a write path for
   incremental ETL refreshes, with zero changes to tools or scoring.
4. **RAG over unstructured sources.** Index airport master plans, FAA NPIAS reports, and
   news/filings so the bounded reviser grounds its qualitative nudges in *citable* evidence
   (announced funding, construction moratoria) rather than parametric memory — this is the
   point where retrieval finally earns its complexity (see §5).
5. **Productionization & UX.** A gateway + queue + workers deployment for multi-user
   concurrency; an eval/regression suite pinning routing accuracy and backtest precision in CI;
   and UI niceties — trend charts, per-component weight sliders, exportable shortlists.
