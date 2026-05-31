# ✈️ Airport Investment Intelligence Agent

An AI agent that helps analysts identify US airports where **renovation/expansion will be
most profitable*.


## Quick start

```bash
pip install -r requirements.txt
python -m etl.build_airports   # builds data/airports.json from public data (one-time; cached)
streamlit run app.py           # open the chat UI
```

## How it works

```
You ─► Streamlit chat
        │
   LLM ROUTE  ──► picks a deterministic tool + args (or keyword fallback)
        │
   DETERMINISTIC TOOLS  ──► rank / compare / profile / flight_breakdown   (EPI math here)
        │
   LLM REVISE (optional, λ-capped, must justify) ──► FinalScore = EPI × clamp(modifier)
        │           • light (default): nudge from model knowledge — fast, cheap
        │           • research (opt-in): grounded web search, cite-or-discard, sources shown
        │
   LLM NARRATE  ──► professional answer grounded in the numbers
```

**Research mode.** A sidebar toggle (active when λ>0) upgrades the bounded nudge from the
model's own knowledge to a **grounded deep web lookup**: it searches official/primary
sources (FAA/BTS/DOT + aviation press) on a heavier model, keeps only factors that carry a
**cited source URL**, and shows those citations in a Sources panel. It costs more tokens
(visible in ⋮ → About) and trades byte-reproducibility for auditability — the deterministic
EPI is untouched in both modes. See [DESIGN.md](DESIGN.md) §3.


