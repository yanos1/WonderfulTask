"""Unit tests for the typed, validated Airport record schema."""

import pytest

from airport_intel.models import Airport, DataValidationError

GOOD = {
    "iata": "sfo", "name": "San Francisco Intl", "city": "San Francisco", "state": "ca",
    "type": "large_airport", "passengers": 25_000_000, "passengers_vintage": 2024,
    "departures": 200_000, "load_factor": 0.82, "long_haul_pct": 0.13,
    "pax_growth_yoy": 0.04, "runway_count": 4, "runway_length_ft": 11870.0,
}


def test_from_raw_normalizes_and_types():
    a = Airport.from_raw(GOOD)
    assert a.iata == "SFO" and a.state == "CA"      # codes upper-cased
    assert a.load_factor == 0.82
    assert isinstance(a.runway_count, int)


def test_airport_is_dict_compatible():
    a = Airport.from_raw(GOOD)
    assert a["iata"] == "SFO"                         # __getitem__
    assert a.get("load_factor") == 0.82              # Mapping.get
    assert a.get("avg_delay_min") is None            # absent field -> default, not KeyError
    assert "passengers" in a                          # __contains__
    assert dict(a)["name"] == "San Francisco Intl"   # iterable as a mapping


@pytest.mark.parametrize("bad", [
    {**GOOD, "iata": "SF"},                # not 3 letters
    {**GOOD, "iata": "12A"},               # not alpha
    {**GOOD, "name": ""},                  # required
    {**GOOD, "load_factor": 1.5},          # out of [0, 1]
    {**GOOD, "long_haul_pct": -0.1},       # out of [0, 1]
    {**GOOD, "passengers": -5},            # negative volume
])
def test_invalid_records_fail_loud(bad):
    with pytest.raises(DataValidationError):
        Airport.from_raw(bad)


def test_optional_fields_may_be_none():
    rec = {**GOOD, "pax_growth_yoy": None, "long_haul_pct": None, "departures": None}
    a = Airport.from_raw(rec)
    assert a.pax_growth_yoy is None and a.long_haul_pct is None and a.departures is None
