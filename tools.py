"""Deterministic tools the LLM agent calls (function calling).

These contain ZERO LLM logic — they are the graded "deterministic ranking" layer.
Each returns structured JSON plus:
  - `steps`: a code-generated trace of what was actually computed (honest, auditable)
  - `notes`: assumptions/uncertainty to disclose (data vintage, missing growth, etc.)

The optional bounded LLM score-reviser is applied later, in the agent layer, so this
module stays purely deterministic.
"""

from __future__ import annotations

from typing import Optional

from regions import resolve_region
from repository import Repository, get_repository
from scoring import ScoringEngine

# Caveats surfaced to the user so the agent can communicate uncertainty honestly.
DATA_NOTES = [
    "Volume and YoY growth are current (FAA CY2024 enplanements, 2024 vs 2023).",
    "Structural ratios (load factor, long-haul %) are derived from 2013 BTS T-100 segment "
    "data — the most recent reliably-accessible segment-level source. Relative rankings are "
    "informative; absolute load factors are conservative vs. today.",
    "Long-haul % is domestic-only (international segments not yet ingested), so it understates "
    "long-haul share at international gateways.",
]

# Friendly metric names -> (record field, human label, formatter)
_METRICS = {
    "congestion": ("load_factor", "load factor", lambda v: f"{v:.1%}" if v is not None else "n/a"),
    "load_factor": ("load_factor", "load factor", lambda v: f"{v:.1%}" if v is not None else "n/a"),
    "volume": ("passengers", "passengers", lambda v: f"{v:,.0f}" if v is not None else "n/a"),
    "passengers": ("passengers", "passengers", lambda v: f"{v:,.0f}" if v is not None else "n/a"),
    "long_haul": ("long_haul_pct", "long-haul share", lambda v: f"{v:.1%}" if v is not None else "n/a"),
    "growth": ("pax_growth_cagr", "passenger growth", lambda v: f"{v:.1%}" if v is not None else "n/a"),
    "runways": ("runway_count", "runways", lambda v: f"{v:g}" if v is not None else "n/a"),
}

LONG_HAUL_MILES = 2500

_repo: Optional[Repository] = None
_engine: Optional[ScoringEngine] = None


def _engines():
    global _repo, _engine
    if _engine is None:
        _repo = get_repository()
        _engine = ScoringEngine(_repo.all())
    return _repo, _engine


# --------------------------------------------------------------------------- #
# Entity resolution: natural-language name/code -> validated IATA code
# --------------------------------------------------------------------------- #
def resolve_code(query: str) -> Optional[str]:
    repo, _ = _engines()
    if not query:
        return None
    q = query.strip().upper()
    if len(q) == 3 and repo.exists(q):
        return q
    ql = query.strip().lower()
    # match on city or name substring; prefer the highest-volume match
    matches = [
        a for a in repo.all()
        if ql in (a.get("city") or "").lower() or ql in (a.get("name") or "").lower()
    ]
    if matches:
        matches.sort(key=lambda a: a.get("passengers") or 0, reverse=True)
        return matches[0]["iata"]
    return None


def _slim(scored: dict, rec: dict) -> dict:
    return {
        "iata": rec["iata"],
        "name": rec.get("name"),
        "city": rec.get("city"),
        "state": rec.get("state"),
        "epi": scored["epi"],
        "demand": scored["demand"],
        "feasibility": scored["feasibility"],
        "subscores": scored["subscores"],
    }


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def rank_region(region: Optional[str] = None, limit: int = 5, weights: Optional[dict] = None) -> dict:
    """Rank airports by Expansion Profitability Index (Q1). region=None ranks nationally."""
    repo, eng = _engines()
    steps = []
    if region:
        states = resolve_region(region)
        if states is None:
            return {"error": f"Unknown region '{region}'.", "notes": DATA_NOTES}
        candidates = repo.find(states=states, min_passengers=1)
        steps.append(f"Resolved region '{region}' -> states {states}")
    else:
        states = None
        candidates = repo.find(min_passengers=1)
        steps.append("No region given -> ranking all US airports")

    steps.append(f"Scored {len(candidates)} airports on EPI = demand x feasibility x 100")
    scored = sorted((eng.epi(a, weights) for a in candidates), key=lambda x: x["epi"], reverse=True)
    by_code = {a["iata"]: a for a in candidates}
    results = [_slim(s, by_code[s["iata"]]) for s in scored[:limit]]
    steps.append(f"Sorted by EPI, selected top {len(results)}")
    return {
        "intent": "rank",
        "region": region,
        "states": states,
        "candidates_considered": len(candidates),
        "results": results,
        "weights": weights or "default",
        "steps": steps,
        "notes": DATA_NOTES,
    }


def compare_airports(codes: list[str], metric: str = "congestion") -> dict:
    """Compare airports on a chosen KPI (Q2, e.g. congestion=load factor)."""
    repo, eng = _engines()
    field, label, fmt = _METRICS.get(metric.lower(), _METRICS["congestion"])
    steps = [f"Comparing on '{label}' (field={field})"]
    rows = []
    for raw in codes:
        code = resolve_code(raw)
        if not code:
            rows.append({"input": raw, "error": "not found in dataset"})
            continue
        rec = repo.get(code)
        val = rec.get(field)
        rows.append({
            "iata": code, "name": rec.get("name"), "city": rec.get("city"),
            "value": val, "display": fmt(val), "epi": eng.epi(rec)["epi"],
        })
    valid = [r for r in rows if r.get("value") is not None]
    if len(valid) >= 2:
        hi = max(valid, key=lambda r: r["value"])
        steps.append(f"Highest {label}: {hi['iata']} ({hi['display']})")
    return {
        "intent": "compare", "metric": label, "results": rows,
        "steps": steps, "notes": DATA_NOTES,
    }


def airport_profile(code: str) -> dict:
    """Full profile incl. EPI + unmet-demand explanation (Q4: 'unmet demand in X and why')."""
    repo, eng = _engines()
    resolved = resolve_code(code)
    if not resolved:
        return {"error": f"Airport '{code}' not found.", "notes": DATA_NOTES}
    rec = repo.get(resolved)
    scored = eng.epi(rec)
    unmet = eng.unmet_demand(rec)
    return {
        "intent": "profile",
        "iata": resolved,
        "name": rec.get("name"),
        "city": rec.get("city"),
        "state": rec.get("state"),
        "metrics": {
            "passengers": rec.get("passengers"),
            "passengers_vintage": rec.get("passengers_vintage"),
            "load_factor": rec.get("load_factor"),
            "long_haul_pct": rec.get("long_haul_pct"),
            "growth_yoy": rec.get("pax_growth_yoy"),
            "runway_count": rec.get("runway_count"),
            "runway_length_ft": rec.get("runway_length_ft"),
        },
        "epi": scored["epi"],
        "demand": scored["demand"],
        "feasibility": scored["feasibility"],
        "subscores": scored["subscores"],
        "unmet_demand": unmet,
        "steps": [
            f"Loaded profile for {resolved}",
            f"EPI={scored['epi']} (demand={scored['demand']}, feasibility={scored['feasibility']})",
            f"Unmet-demand score={unmet['unmet_demand_score']}",
        ],
        "notes": DATA_NOTES,
    }


def flight_breakdown(code: str) -> dict:
    """Long-haul share and route mix for an airport (Q3: '% long-haul out of X')."""
    repo, _ = _engines()
    resolved = resolve_code(code)
    if not resolved:
        return {"error": f"Airport '{code}' not found.", "notes": DATA_NOTES}
    rec = repo.get(resolved)
    lh = rec.get("long_haul_pct")
    return {
        "intent": "flight_breakdown",
        "iata": resolved,
        "name": rec.get("name"),
        "long_haul_pct": lh,
        "long_haul_display": f"{lh:.1%}" if lh is not None else "n/a",
        "long_haul_threshold_miles": LONG_HAUL_MILES,
        "steps": [
            f"Loaded segment mix for {resolved}",
            f"Long-haul = flights with segment distance > {LONG_HAUL_MILES} mi",
            f"Long-haul share = {lh:.1%}" if lh is not None else "Long-haul share unavailable",
        ],
        "notes": DATA_NOTES,
    }


def live_status(code: str) -> dict:
    """Live operational status (AeroDataBox). Degrades gracefully without a key."""
    resolved = resolve_code(code)
    if not resolved:
        return {"error": f"Airport '{code}' not found."}
    # Wired in a later step; without a key we disclose unavailability rather than fail.
    return {
        "intent": "live_status",
        "iata": resolved,
        "available": False,
        "note": "Live feed not configured (no AeroDataBox key); using historical baseline.",
    }
