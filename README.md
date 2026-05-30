# ✈️ Airport Investment Intelligence Agent

An AI agent that helps analysts identify US airports where **renovation/expansion will be
most profitable**, based on flight and passenger capacity. It ranks and compares airports
using a **deterministic Expansion Profitability Index (EPI)** and uses an LLM only at the
edges (understanding the question, explaining the answer, and an optional *bounded* score
nudge).

It answers questions like:
- *Which airports in New England are strong candidates for terminal expansion?*
- *Compare LA and Santa Ana airport congestion levels.*
- *What is the percentage of long-haul flights out of Anchorage?*
- *What is the unmet flight demand in SFO and why?*

## Quick start

```bash
pip install -r requirements.txt
python etl.py            # builds data/airports.json from public data (one-time; cached)
streamlit run app.py     # open the chat UI
```

The app runs **without any API key** (deterministic routing + templated answers). For
LLM-powered routing and natural explanations, add a key — a **free Google Gemini key**
([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) is the no-billing path:

```bash
cp .env.example .env     # then paste your key, OR paste it directly in the sidebar
```

In the sidebar you can switch model (Gemini / Claude / deterministic) and set **λ
(LLM influence)** — how much the LLM may adjust scores (0 = pure deterministic).

## How it works

```
You ─► Streamlit chat
        │
   LLM ROUTE  ──► picks a deterministic tool + args (or keyword fallback)
        │
   DETERMINISTIC TOOLS  ──► rank / compare / profile / flight_breakdown   (EPI math here)
        │
   LLM REVISE (optional, λ-capped, must justify) ──► FinalScore = EPI × clamp(modifier)
        │
   LLM NARRATE  ──► professional answer grounded in the numbers
```

- **Data** (`etl.py` → `data/airports.json`): OurAirports (metadata, runways) + BTS T-100
  segment data (load factor, long-haul %) + runway-based feasibility — derived for ~900
  US airports.
- **Scoring** (`scoring.py`): `EPI = demand × feasibility × 100`, percentile-normalized.
- **Tools** (`tools.py`): pure-Python, the graded non-LLM logic.
- **Agent** (`agent.py`): route → execute → revise → narrate, with conversation history
  for follow-ups.

See **[DESIGN.md](DESIGN.md)** for the scoring methodology, tradeoffs, and where AI is used.

## Tests

```bash
python -m pytest -q
```

## Key assumptions (full list in DESIGN.md)
- **Data vintage:** structural ratios (load factor, long-haul %) come from 2013 BTS T-100
  segment data — the most recent reliably/programmatically downloadable segment-level
  source. The ETL is year-parameterized; relative rankings are informative.
- **Long-haul is domestic-only** (international segments not yet ingested) → understates
  long-haul at international gateways.
- **Passenger growth** is currently unavailable (the 2024 NTAD source is unreachable); the
  pipeline degrades gracefully and redistributes that weight, disclosing the gap.
- **Cargo is out of scope** — the thesis is passenger/terminal expansion.

## With more time
- Wire a current-year volume source (FAA enplanements) → restores growth + 2024 volumes.
- Ingest T-100 international segments → accurate long-haul for gateways.
- Live AeroDataBox delays as a real EPI signal (currently display-only / graceful stub).
- RAG over airport master plans/news to ground the LLM reviser's qualitative adjustments.
- Voice input (browser `webkitSpeechRecognition`).
