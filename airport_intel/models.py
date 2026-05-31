"""Typed, validated record schema for the airport dataset.

`Airport` is the single source of truth for what one record looks like. Two design choices
make it pull its weight without forcing a rewrite of the deterministic layer:

  * **Immutable + typed.** A frozen dataclass: fields are documented and type-annotated in
    one place, and records can't be mutated out from under the scoring engine.
  * **Dict-compatible (a `Mapping`).** The tools/scoring access fields by *dynamic* string
    name (e.g. ``a.get("load_factor")``, ``a["iata"]``). Subclassing ``Mapping`` means an
    ``Airport`` is a drop-in for the plain dicts those call sites (and the unit tests) used
    before — ``.get`` / ``[]`` / ``in`` all work — while new code can use typed attribute
    access (``a.load_factor``).

Validation lives in :meth:`Airport.from_raw`, which is the load-time boundary: a malformed
record fails loud *here* (with the offending field named) instead of silently propagating a
bad value into a score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Iterator, Optional


class DataValidationError(ValueError):
    """Raised when a raw record violates the :class:`Airport` schema."""


def _opt_number(
    raw: dict, key: str, *, lo: Optional[float] = None, hi: Optional[float] = None
) -> Optional[float]:
    """Validate an optional numeric field, enforcing an inclusive [lo, hi] range."""
    val = raw.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise DataValidationError(f"{raw.get('iata', '?')}.{key}: expected a number, got {val!r}")
    if lo is not None and val < lo:
        raise DataValidationError(f"{raw.get('iata', '?')}.{key}: {val} < {lo}")
    if hi is not None and val > hi:
        raise DataValidationError(f"{raw.get('iata', '?')}.{key}: {val} > {hi}")
    return val


@dataclass(frozen=True)
class Airport(Mapping):
    """One airport record: identity + demand signals + feasibility inputs."""

    # identity / metadata
    iata: str
    name: str
    city: Optional[str]
    state: Optional[str]
    type: Optional[str]
    # demand signals
    passengers: int
    passengers_vintage: int
    departures: Optional[int]
    load_factor: Optional[float]      # in [0, 1]
    long_haul_pct: Optional[float]    # in [0, 1]
    pax_growth_yoy: Optional[float]   # YoY fraction; may be negative
    # feasibility inputs
    runway_count: int
    runway_length_ft: float

    @classmethod
    def from_raw(cls, raw: dict) -> "Airport":
        """Validate and coerce one raw dict (e.g. a JSON record) into an `Airport`.

        Raises :class:`DataValidationError` with the offending field named.
        """
        iata = str(raw.get("iata", "")).strip().upper()
        if len(iata) != 3 or not iata.isalpha():
            raise DataValidationError(f"invalid IATA code {raw.get('iata')!r} (need 3 letters)")
        name = raw.get("name")
        if not name or not str(name).strip():
            raise DataValidationError(f"{iata}.name: required and non-empty")

        passengers = _opt_number(raw, "passengers", lo=0)
        runway_count = _opt_number(raw, "runway_count", lo=0)
        return cls(
            iata=iata,
            name=str(name),
            city=(str(raw["city"]) if raw.get("city") is not None else None),
            state=(str(raw["state"]).upper() if raw.get("state") is not None else None),
            type=(str(raw["type"]) if raw.get("type") is not None else None),
            passengers=int(passengers) if passengers is not None else 0,
            passengers_vintage=int(_opt_number(raw, "passengers_vintage") or 0),
            departures=(
                int(_opt_number(raw, "departures", lo=0))
                if raw.get("departures") is not None
                else None
            ),
            load_factor=_opt_number(raw, "load_factor", lo=0.0, hi=1.0),
            long_haul_pct=_opt_number(raw, "long_haul_pct", lo=0.0, hi=1.0),
            pax_growth_yoy=_opt_number(raw, "pax_growth_yoy"),
            runway_count=int(runway_count) if runway_count is not None else 0,
            runway_length_ft=float(_opt_number(raw, "runway_length_ft", lo=0.0) or 0.0),
        )

    # -- Mapping interface: lets an Airport stand in for the old plain dicts ---- #
    def __getitem__(self, key: str) -> Any:
        if key in _FIELD_NAMES:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(_FIELD_NAMES)

    def __len__(self) -> int:
        return len(_FIELD_NAMES)


_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Airport))