"""Deterministic tools the LLM agent calls (function calling).

These contain ZERO LLM logic — they are the graded "deterministic ranking" layer.
Each returns structured JSON plus:
  - `steps`: a code-generated trace of what was actually computed (honest, auditable)
  - `notes`: assumptions/uncertainty to disclose (data vintage, missing growth, etc.)

The optional bounded LLM score-reviser is applied later, in the agent layer, so this
module stays purely deterministic.
"""

from __future__ import annotations

import re
from typing import Optional

from .regions import resolve_region
from .repository import Repository, get_repository
from .scoring import ScoringEngine

# Caveats surfaced to the user so the agent can communicate uncertainty honestly.
DATA_NOTES = [
    "Volume and YoY growth are current (FAA CY2024 enplanements, 2024 vs 2023).",
    "Structural ratios (load factor, long-haul %) are derived from 2024 BTS T-100 Domestic "
    "Segment data (the segment-level source carrying seats and distance).",
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
    "growth": ("pax_growth_yoy", "passenger growth (YoY)", lambda v: f"{v:.1%}" if v is not None else "n/a"),
    "runways": ("runway_count", "runways", lambda v: f"{v:g}" if v is not None else "n/a"),
}

LONG_HAUL_MILES = 2500
MAX_RANK_LIMIT = 50  # hard cap so a runaway "list everything" stays a sane shortlist


def _clean_limit(limit, default: int = 5) -> int:
    """Coerce a router-supplied limit into [1, MAX_RANK_LIMIT].

    The LLM can hand us junk: None, 0, a negative number (which would make scored[:limit]
    silently drop from the END of the list), or a non-int string like "5.5" (which would
    raise). Normalize all of it instead of crashing or mis-slicing.
    """
    try:
        n = int(float(limit))
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, MAX_RANK_LIMIT)

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
def _resolve_matches(query: str) -> list[dict]:
    """All airport records a query could refer to, best (highest-volume) first.

    A 3-letter IATA code resolves to exactly that airport. Otherwise we match on a whole
    WORD in the city/name (or the full string), NOT an arbitrary substring -- otherwise
    "LA" matches "atLAnta". Multiple hits (e.g. "Portland" -> PDX/PWM) are all returned so
    callers can flag the ambiguity instead of silently picking one.
    """
    repo, _ = _engines()
    if not query:
        return []
    q = query.strip().upper()
    if len(q) == 3 and repo.exists(q):
        return [repo.get(q)]
    ql = query.strip().lower()

    def _tokens(text: Optional[str]) -> set[str]:
        # split on non-alphanumerics so "Los Angeles Intl" -> {los, angeles, intl}
        return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}

    matches = [
        a for a in repo.all()
        if ql == (a.get("city") or "").lower()
        or ql == (a.get("name") or "").lower()
        or ql in _tokens(a.get("city"))
        or ql in _tokens(a.get("name"))
    ]
    matches.sort(key=lambda a: a.get("passengers") or 0, reverse=True)
    return matches


def resolve_code(query: str) -> Optional[str]:
    """Best single IATA code for a query (highest-volume match), or None."""
    matches = _resolve_matches(query)
    return matches[0]["iata"] if matches else None


def _ambiguity_note(query: str, matches: list[dict]) -> Optional[str]:
    """Warn when a name matched airports in MORE THAN ONE state (the dangerous case:
    e.g. 'Portland' -> PDX Oregon vs PWM Maine). Same-metro multi-airport hits
    (e.g. Houston's IAH/HOU) are not flagged -- picking the busiest is reasonable there."""
    states = {(m.get("state") or "").upper() for m in matches if m.get("state")}
    if len(matches) < 2 or len(states) < 2:
        return None
    chosen = matches[0]
    others = ", ".join(
        f"{m['iata']} ({m.get('city')}, {m.get('state')})" for m in matches[1:4]
    )
    return (
        f"'{query}' is ambiguous — used {chosen['iata']} "
        f"({chosen.get('city')}, {chosen.get('state')}; highest passenger volume). "
        f"Other matches: {others}. Pass an IATA code to pick a specific airport."
    )


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
def rank_region(region: Optional[str] = None, limit: int = 5,
                metric: Optional[str] = None, weights: Optional[dict] = None) -> dict:
    """Rank airports for expansion (Q1). region=None ranks nationally.

    By default airports are ranked by the Expansion Profitability Index (EPI). Pass a
    single `metric` (e.g. "growth", "volume", "congestion") to rank by that raw KPI
    instead -- this answers "rank airports by growth" with real data rather than
    silently force-fitting an EPI ranking onto it. EPI is still reported per row for
    context. An unknown metric errors (like compare) instead of mislabeling.
    """
    repo, eng = _engines()
    limit = _clean_limit(limit)

    metric_info = None
    if metric:
        metric_info = _METRICS.get(metric.lower())
        if metric_info is None:
            supported = sorted({lbl for _, lbl, _ in _METRICS.values()})
            return {
                "intent": "rank",
                "error": f"Unsupported metric '{metric}'. Supported metrics: "
                         f"{', '.join(supported)}.",
                "notes": list(DATA_NOTES),
            }

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

    if metric_info is not None:
        field, label, fmt = metric_info
        # Rank by the raw KPI, high -> low. Airports missing the metric can't be ranked
        # on it, so drop them rather than sorting None to an arbitrary end.
        rated = [(a, a.get(field)) for a in candidates]
        rated = [(a, v) for a, v in rated if v is not None]
        rated.sort(key=lambda av: av[1], reverse=True)
        steps.append(f"Ranked {len(rated)} airports with data on {label} (field={field}), high to low")
        results = []
        for rec, val in rated[:limit]:
            row = _slim(eng.epi(rec, weights), rec)
            row["metric_value"] = val
            row["metric_display"] = fmt(val)
            results.append(row)
        steps.append(f"Selected top {len(results)} by {label}")
        return {
            "intent": "rank",
            "region": region,
            "states": states,
            "ranked_by": label,
            "candidates_considered": len(candidates),
            "results": results,
            "steps": steps,
            "notes": DATA_NOTES,
        }

    steps.append(f"Scored {len(candidates)} airports on EPI = demand x feasibility x 100")
    scored = sorted((eng.epi(a, weights) for a in candidates), key=lambda x: x["epi"], reverse=True)
    by_code = {a["iata"]: a for a in candidates}
    results = [_slim(s, by_code[s["iata"]]) for s in scored[:limit]]
    steps.append(f"Sorted by EPI, selected top {len(results)}")
    return {
        "intent": "rank",
        "region": region,
        "states": states,
        "ranked_by": "EPI",
        "candidates_considered": len(candidates),
        "results": results,
        "weights": weights or "default",
        "steps": steps,
        "notes": DATA_NOTES,
    }


def compare_airports(codes: list[str], metric: str = "congestion") -> dict:
    """Compare airports on a chosen KPI (Q2, e.g. congestion=load factor)."""
    repo, eng = _engines()
    metric_info = _METRICS.get((metric or "").lower())
    if metric_info is None:
        # Don't silently mislabel an unsupported metric (e.g. "delay") as load factor.
        supported = sorted({lbl for _, lbl, _ in _METRICS.values()})
        return {
            "intent": "compare",
            "error": f"Unsupported metric '{metric}'. Supported metrics: "
                     f"{', '.join(supported)}.",
            "notes": list(DATA_NOTES),
        }
    field, label, fmt = metric_info
    steps = [f"Comparing on '{label}' (field={field})"]
    notes = list(DATA_NOTES)
    rows, seen = [], set()
    for raw in codes:
        matches = _resolve_matches(raw)
        code = matches[0]["iata"] if matches else None
        if not code:
            rows.append({"input": raw, "error": "not found in dataset"})
            continue
        if code in seen:  # "compare LA and Los Angeles" -> don't compare LAX to itself
            steps.append(f"Skipped duplicate reference '{raw}' -> {code}")
            continue
        seen.add(code)
        note = _ambiguity_note(raw, matches)
        if note:
            notes.append(note)
        rec = repo.get(code)
        val = rec.get(field)
        rows.append({
            "iata": code, "name": rec.get("name"), "city": rec.get("city"),
            "value": val, "display": fmt(val), "epi": eng.epi(rec)["epi"],
        })
    if len(seen) < 2:
        steps.append("Need at least 2 distinct airports to compare.")
    valid = [r for r in rows if r.get("value") is not None]
    if len(valid) >= 2:
        hi = max(valid, key=lambda r: r["value"])
        steps.append(f"Highest {label}: {hi['iata']} ({hi['display']})")
    return {
        "intent": "compare", "metric": label, "results": rows,
        "steps": steps, "notes": notes,
    }


def airport_profile(code: str) -> dict:
    """Full profile incl. EPI + unmet-demand explanation (Q4: 'unmet demand in X and why')."""
    repo, eng = _engines()
    matches = _resolve_matches(code)
    resolved = matches[0]["iata"] if matches else None
    if not resolved:
        return {"error": f"Airport '{code}' not found.", "notes": list(DATA_NOTES)}
    notes = list(DATA_NOTES)
    note = _ambiguity_note(code, matches)
    if note:
        notes.append(note)
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
        "notes": notes,
    }


def flight_breakdown(code: str) -> dict:
    """Long-haul share and route mix for an airport (Q3: '% long-haul out of X')."""
    repo, _ = _engines()
    matches = _resolve_matches(code)
    resolved = matches[0]["iata"] if matches else None
    if not resolved:
        return {"error": f"Airport '{code}' not found.", "notes": list(DATA_NOTES)}
    notes = list(DATA_NOTES)
    note = _ambiguity_note(code, matches)
    if note:
        notes.append(note)
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
        "notes": notes,
    }
