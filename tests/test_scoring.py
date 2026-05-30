"""Unit tests for the deterministic scoring engine and the bounded LLM modifier."""

from airport_intel import scoring
from airport_intel.scoring import ScoringEngine, apply_modifier, clamp_modifier, MAX_SWING

# synthetic universe with known ordering (no file/network dependency)
AIRPORTS = [
    {"iata": "AAA", "load_factor": 0.95, "passengers": 1_000_000, "pax_growth_yoy": 0.05,
     "runway_count": 4, "runway_length_ft": 12000},
    {"iata": "BBB", "load_factor": 0.70, "passengers": 500_000, "pax_growth_yoy": 0.01,
     "runway_count": 2, "runway_length_ft": 8000},
    {"iata": "CCC", "load_factor": 0.50, "passengers": 50_000, "pax_growth_yoy": None,
     "runway_count": 1, "runway_length_ft": 4000},
]


def _engine():
    return ScoringEngine(AIRPORTS)


def test_epi_is_deterministic_and_ordered():
    eng = _engine()
    a = eng.epi(AIRPORTS[0])["epi"]
    b = eng.epi(AIRPORTS[1])["epi"]
    c = eng.epi(AIRPORTS[2])["epi"]
    assert a > b > c                      # higher demand+feasibility -> higher EPI
    assert eng.epi(AIRPORTS[0])["epi"] == a  # stable across calls


def test_feasibility_is_floored():
    eng = _engine()
    # the weakest airport must still get at least the floor multiplier
    assert eng.feasibility(AIRPORTS[2]) >= scoring.FEASIBILITY_FLOOR
    assert eng.feasibility(AIRPORTS[0]) <= 1.0


def test_missing_component_is_handled():
    eng = _engine()
    # CCC has no growth; scoring must not crash and weights redistribute
    s = eng.epi(AIRPORTS[2])
    assert "growth" not in s["subscores"]
    assert 0 <= s["epi"] <= 100


def test_lambda_zero_means_pure_epi():
    out = apply_modifier(50.0, 1.15, lam=0.0, reason="big funding news")
    assert out["modifier"] == 1.0
    assert out["final_score"] == 50.0


def test_blank_reason_is_discarded():
    out = apply_modifier(50.0, 1.30, lam=1.0, reason="   ")
    assert out["modifier"] == 1.0
    assert out["final_score"] == 50.0
    assert out["reason"] is None


def test_modifier_is_clamped_to_band():
    # lam=1 -> max +-30%; an aggressive 2.0 request must be clamped
    assert clamp_modifier(2.0, 1.0) == 1 + MAX_SWING
    assert clamp_modifier(0.0, 1.0) == 1 - MAX_SWING
    # lam=0.5 -> +-15%
    assert clamp_modifier(2.0, 0.5) == 1 + 0.5 * MAX_SWING


def test_valid_modifier_applies_within_band():
    out = apply_modifier(100.0, 1.10, lam=1.0, reason="secured terminal funding")
    assert out["modifier"] == 1.10
    assert out["final_score"] == 110.0
    assert out["reason"]
