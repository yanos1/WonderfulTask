"""Deterministic scoring engine — the graded "non-LLM" logic.

Expansion Profitability Index (EPI):

    demand    = weighted, percentile-normalized blend of load factor, growth, volume, long-haul mix
    EPI       = demand * Feasibility * 100            (Feasibility in [0.3, 1])
    FinalScore = EPI * clamp(LLMModifier, 1 - λ·MAX_SWING, 1 + λ·MAX_SWING)

The LLM never computes the base EPI. It may only return *bounded, justified* adjustments —
either a single modifier (scoring.apply_modifier) or a list of separately-justified factors
that sum into one (scoring.apply_factors). λ ("llm_influence", 0..1) caps how far it can move
a score (λ=1 ⇒ ±40%). λ=0 (or a missing/blank justification) ⇒ FinalScore == EPI.

All normalization is by percentile rank across the full airport universe, so the EPI is a
stable absolute index and any subset ranking (e.g. one region) is directly comparable.
"""

from __future__ import annotations

import bisect
from typing import Optional

MAX_SWING = 0.40          # λ=1 lets the LLM move a score at most ±40%
FEASIBILITY_FLOOR = 0.30  # weak proxy: dampen, never nuke

# Demand components and their default weights (renormalized over whatever data exists).
DEFAULT_WEIGHTS = {
    "load_factor": 0.40,   # centerpiece: full planes = demand pressure
    "growth": 0.25,        # rising demand
    "delay": 0.10,         # congestion symptom (added when delay data is wired)
    "volume": 0.25,        # absolute demand magnitude
    "long_haul": 0.10,     # route-mix quality: long-haul skews higher revenue/margin
}
# How to pull each demand component's raw value from an airport record.
_GETTERS = {
    "load_factor": lambda a: a.get("load_factor"),
    "growth": lambda a: a.get("pax_growth_yoy"),
    "delay": lambda a: a.get("avg_delay_min"),
    "volume": lambda a: a.get("passengers"),
    "long_haul": lambda a: a.get("long_haul_pct"),
}
_FEASIBILITY_FIELDS = ("runway_count", "runway_length_ft")


class _Percentiler:
    """Precomputes a sorted value list so percentile rank is O(log n) per lookup."""

    def __init__(self, values: list[float]):
        self._sorted = sorted(values)

    def rank(self, value: Optional[float]) -> Optional[float]:
        if value is None or not self._sorted:
            return None
        n = len(self._sorted)
        if n == 1:
            return 0.5
        below = bisect.bisect_left(self._sorted, value)
        return below / (n - 1)


class ScoringEngine:
    """Builds percentile baselines over a universe of airports, then scores any of them."""

    def __init__(self, airports: list[dict]):
        self._pct: dict[str, _Percentiler] = {}
        # demand components: only those with at least one value participate
        self.active_components: list[str] = []
        for comp, getter in _GETTERS.items():
            vals = [getter(a) for a in airports if getter(a) is not None]
            if vals:
                self._pct[comp] = _Percentiler(vals)
                self.active_components.append(comp)
        # feasibility inputs
        for field in _FEASIBILITY_FIELDS:
            vals = [a.get(field) for a in airports if a.get(field)]
            self._pct[field] = _Percentiler(vals) if vals else _Percentiler([0])

    # -- components ------------------------------------------------------- #
    def feasibility(self, airport: dict) -> float:
        ranks = [self._pct[f].rank(airport.get(f)) for f in _FEASIBILITY_FIELDS]
        ranks = [r for r in ranks if r is not None]
        raw = sum(ranks) / len(ranks) if ranks else 0.0
        return FEASIBILITY_FLOOR + (1 - FEASIBILITY_FLOOR) * raw

    def demand(self, airport: dict, weights: Optional[dict] = None) -> tuple[float, dict]:
        """Return (demand 0..1, per-component normalized subscores)."""
        weights = weights or DEFAULT_WEIGHTS
        subs, total_w, acc = {}, 0.0, 0.0
        for comp in self.active_components:
            rank = self._pct[comp].rank(_GETTERS[comp](airport))
            if rank is None:
                continue
            w = weights.get(comp, 0.0)
            subs[comp] = round(rank, 4)
            acc += w * rank
            total_w += w
        demand = acc / total_w if total_w > 0 else 0.0
        return demand, subs

    # -- scores ----------------------------------------------------------- #
    def epi(self, airport: dict, weights: Optional[dict] = None) -> dict:
        demand, subs = self.demand(airport, weights)
        feas = self.feasibility(airport)
        epi = round(demand * feas * 100, 2)
        return {
            "iata": airport["iata"],
            "epi": epi,
            "demand": round(demand, 4),
            "feasibility": round(feas, 4),
            "subscores": subs,
        }

    def unmet_demand(self, airport: dict) -> dict:
        """Standalone indicator for the 'unmet demand in X and why' question (Q4)."""
        lf = self._pct["load_factor"].rank(airport.get("load_factor"))
        gr = self._pct["growth"].rank(airport.get("pax_growth_yoy")) if "growth" in self._pct else None
        parts = [r for r in (lf, gr) if r is not None]
        score = round(100 * sum(parts) / len(parts), 1) if parts else None
        reasons = []
        if airport.get("load_factor") is not None:
            reasons.append(
                f"load factor {airport['load_factor']:.0%} "
                f"({lf:.0%} percentile) - fuller planes imply demand turned away"
            )
        if airport.get("pax_growth_yoy") is not None:
            reasons.append(f"passenger growth {airport['pax_growth_yoy']:.1%} YoY (2024 vs 2023)")
        else:
            reasons.append("growth unavailable for this airport")
        return {"iata": airport["iata"], "unmet_demand_score": score, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Bounded LLM modifier  (FinalScore = EPI × clamp(modifier, 1 ± λ·MAX_SWING))
# --------------------------------------------------------------------------- #
def clamp_modifier(modifier: float, lam: float) -> float:
    """Clamp the LLM modifier into the λ-controlled band. λ=0 ⇒ exactly 1.0."""
    lam = max(0.0, min(1.0, lam))
    band = lam * MAX_SWING
    lo, hi = 1 - band, 1 + band
    return max(lo, min(hi, float(modifier)))


def apply_modifier(epi: float, modifier: Optional[float], lam: float, reason: Optional[str]) -> dict:
    """Apply a bounded, justified modifier. A missing/blank reason is discarded (→ 1.0)."""
    effective = 1.0
    if modifier is not None and reason and reason.strip():
        effective = clamp_modifier(modifier, lam)
    return {
        "epi": round(epi, 2),
        "modifier": round(effective, 4),
        "final_score": round(epi * effective, 2),
        "reason": reason if effective != 1.0 else None,
    }


def apply_factors(epi: float, factors: Optional[list], lam: float) -> dict:
    """Combine a LIST of justified factors into one bounded modifier.

    Each factor is {"reason": str, "impact": float}, where impact is a signed *fractional*
    nudge (+0.08 ≈ +8%, -0.05 ≈ -5%). Only factors carrying a concrete reason count; the
    net modifier is 1 + Σ(impacts), then clamped into the λ band. λ=0 ⇒ no effect. This is
    the multi-factor analogue of apply_modifier: instead of one opaque scalar the LLM gives
    several small, separately-justified forces that add up and stay transparent.
    """
    kept: list[dict] = []
    net = 0.0
    for f in factors or []:
        if not isinstance(f, dict):
            continue
        reason = (f.get("reason") or "").strip()
        impact = f.get("impact")
        if not reason or impact is None:
            continue
        try:
            imp = float(impact)
        except (TypeError, ValueError):
            continue
        net += imp
        kept.append({"reason": reason, "impact": round(imp, 4)})
    effective = clamp_modifier(1.0 + net, lam) if kept else 1.0
    return {
        "epi": round(epi, 2),
        "modifier": round(effective, 4),
        "final_score": round(epi * effective, 2),
        "factors": kept if effective != 1.0 else [],
    }
