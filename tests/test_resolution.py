"""Tests for region resolution and natural-language airport resolution."""

import pytest

from airport_intel.regions import resolve_region


def test_region_exact():
    assert resolve_region("New England") == ["ME", "NH", "VT", "MA", "RI", "CT"]


def test_region_is_case_and_spacing_tolerant():
    assert resolve_region("  new   england ") == resolve_region("New England")
    assert resolve_region("NEW ENGLAND") == resolve_region("New England")


def test_region_phrase_contains():
    # "airports in the new england region" should still resolve
    assert resolve_region("the new england region") == resolve_region("New England")


def test_unknown_region_is_none():
    assert resolve_region("Atlantis") is None
    assert resolve_region(None) is None


# entity resolution touches the real dataset; skip cleanly if the ETL hasn't been run
tools = pytest.importorskip("airport_intel.tools")


def _has_data():
    try:
        return tools.resolve_code("LAX") == "LAX"
    except Exception:
        return False


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_resolve_code_by_iata_and_name():
    assert tools.resolve_code("LAX") == "LAX"
    assert tools.resolve_code("Santa Ana") == "SNA"
    assert tools.resolve_code("Anchorage") == "ANC"
    assert tools.resolve_code("definitely not an airport zzz") is None


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_resolve_code_does_not_substring_match():
    """'LA' must not resolve to 'atLAnta' (ATL) via a stray substring match.

    Whole-word/token matching only: 'LA' is not a token of any major city, so it
    resolves to nothing (the LLM router maps 'Los Angeles' -> LAX before we get here).
    """
    assert tools.resolve_code("LA") != "ATL"
    # a real whole-word token still resolves
    assert tools.resolve_code("Los Angeles") == "LAX"


# -- limit sanitization (pure, no dataset needed) --------------------------- #
def test_clean_limit_handles_junk():
    # negatives/zero/None fall back to the default instead of mis-slicing or crashing
    assert tools._clean_limit(-3) == 5
    assert tools._clean_limit(0) == 5
    assert tools._clean_limit(None) == 5
    assert tools._clean_limit("five") == 5
    # numeric strings and floats coerce; the cap holds
    assert tools._clean_limit("5.5") == 5
    assert tools._clean_limit(3) == 3
    assert tools._clean_limit(10**9) == tools.MAX_RANK_LIMIT


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_rank_region_negative_limit_does_not_overshoot():
    # a negative limit used to return all-but-the-last-N rows; now it's clamped
    out = tools.rank_region(None, -3)
    assert len(out["results"]) == 5


# -- ranking by a single KPI instead of EPI -------------------------------- #
@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_rank_region_by_metric_sorts_by_that_field():
    # "rank airports by growth" must order by pax_growth_yoy, not EPI
    out = tools.rank_region(None, 10, metric="growth")
    assert out["ranked_by"] == "passenger growth (YoY)"
    vals = [r["metric_value"] for r in out["results"]]
    assert vals == sorted(vals, reverse=True)        # descending by the metric
    # rows still carry EPI for context, and the metric column is populated
    assert all(r.get("metric_display") is not None and "epi" in r for r in out["results"])


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_rank_region_default_still_ranks_by_epi():
    out = tools.rank_region(None, 10)
    assert out["ranked_by"] == "EPI"
    epis = [r["epi"] for r in out["results"]]
    assert epis == sorted(epis, reverse=True)


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_rank_region_unsupported_metric_errors():
    # "delay" has no data -> error, don't silently fall back to EPI
    out = tools.rank_region(None, 5, metric="delay")
    assert "error" in out
    assert not out.get("results")


# -- unsupported compare metric -------------------------------------------- #
@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_compare_unsupported_metric_errors_not_mislabels():
    # "delay" has no data and isn't a compare metric: error, don't silently show load factor
    out = tools.compare_airports(["LAX", "SFO"], metric="delay")
    assert "error" in out
    assert "load factor" not in out.get("metric", "")


@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_compare_dedupes_same_airport():
    # "LA" + "Los Angeles" both -> LAX; we should not compare an airport to itself
    out = tools.compare_airports(["Los Angeles", "LAX"], metric="volume")
    iatas = [r["iata"] for r in out["results"] if r.get("iata")]
    assert iatas.count("LAX") == 1


# -- ambiguous city disambiguation ----------------------------------------- #
@pytest.mark.skipif(not _has_data(), reason="airports.json not built (run python -m etl.build_airports)")
def test_ambiguous_city_surfaces_note():
    # Portland matches PDX (OR) and PWM (ME) -> the profile must flag the ambiguity
    out = tools.airport_profile("Portland")
    assert any("ambiguous" in n.lower() for n in out["notes"])
