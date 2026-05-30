"""Data access layer.

`Repository` is the interface the rest of the app depends on, so the backing store
(JSON now -> SQLite/Postgres later) can be swapped without touching tools or scoring.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .models import Airport

# data/ lives at the repo root, one level above this package
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "airports.json"
)


class Repository(ABC):
    @abstractmethod
    def get(self, code: str) -> Optional[Airport]:
        """Return one airport record by IATA code, or None if absent."""

    @abstractmethod
    def all(self) -> list[Airport]:
        """Return every airport record."""

    def find(
        self,
        states: Optional[Iterable[str]] = None,
        codes: Optional[Iterable[str]] = None,
        min_passengers: int = 0,
    ) -> list[dict]:
        """Filter airports by state set and/or explicit code set, with a volume floor."""
        states = {s.upper() for s in states} if states else None
        codes = {c.upper() for c in codes} if codes else None
        out = []
        for a in self.all():
            if codes is not None and a["iata"] not in codes:
                continue
            if states is not None and (a.get("state") or "").upper() not in states:
                continue
            if (a.get("passengers") or 0) < min_passengers:
                continue
            out.append(a)
        return out

    def exists(self, code: str) -> bool:
        return self.get(code) is not None


class JsonRepository(Repository):
    """In-memory repository backed by the ETL output (data/airports.json).

    Records are validated into typed `Airport` objects at load time, so a malformed
    dataset fails loud here rather than corrupting a score downstream.
    """

    def __init__(self, path: str = DEFAULT_PATH):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Accept the provenance-wrapped form {"_meta": ..., "airports": ...} and, for
        # back-compat, the older flat form {CODE: {...}}.
        if isinstance(raw, dict) and isinstance(raw.get("airports"), dict):
            records, self.meta = raw["airports"], raw.get("_meta", {})
        else:
            records, self.meta = raw, {}
        self._data: dict[str, Airport] = {
            code.upper(): Airport.from_raw(rec) for code, rec in records.items()
        }

    def get(self, code: str) -> Optional[Airport]:
        return self._data.get(str(code).upper())

    def all(self) -> list[Airport]:
        return list(self._data.values())


# convenience singleton for the app (cheap: one small JSON load)
_default: Optional[Repository] = None


def get_repository() -> Repository:
    global _default
    if _default is None:
        _default = JsonRepository()
    return _default
