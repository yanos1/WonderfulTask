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
