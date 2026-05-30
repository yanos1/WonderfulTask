"""LLM usage metrics: count calls, tokens, and estimated cost, persisted monotonically.

Every LLM call flows through ``LLMProvider.complete()``, so that is the single choke point
where we meter usage -- nothing can slip past uncounted. Totals accumulate in a small JSON
file and only ever grow: they survive restarts and are never reset to zero, so the menu's
"Metrics" panel reports *lifetime* usage of the tool.

Cost is an ESTIMATE: published per-model $/MTok rates (input, output) applied to the token
counts the SDKs return (Claude ``resp.usage``, Gemini ``resp.usage_metadata``). Rates live
in ``PRICING`` and are trivial to update as vendor pricing changes.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Optional

# Approximate public list prices in USD per 1,000,000 tokens, as (input, output).
# Estimates only -- keyed by a model-name prefix so dated suffixes still match. The first
# two are the app's actual defaults (see llm/base.py); the rest cover likely swaps.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
}
# Fallback for an unrecognized model so cost is never silently zero.
_DEFAULT_RATE: tuple[float, float] = (1.00, 3.00)


def price_for(model: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) in $/MTok for a model, by longest prefix match."""
    best: Optional[tuple[float, float]] = None
    best_len = -1
    for prefix, rate in PRICING.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best, best_len = rate, len(prefix)
    return best if best is not None else _DEFAULT_RATE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost of one call from its token counts and the model's rates."""
    rin, rout = price_for(model)
    return (input_tokens * rin + output_tokens * rout) / 1_000_000


# Where lifetime totals live. Overridable via env (tests / deployments).
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "usage_metrics.json")


@dataclass
class Totals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class MetricsTracker:
    """Thread-safe, monotonic usage accumulator with JSON persistence.

    ``record()`` is called once per LLM completion. Totals only ever increase: we load
    whatever is on disk and add to it, so a restart continues the lifetime tally and never
    resets to zero. A missing or corrupt file starts from zero rather than crashing.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("USAGE_METRICS_PATH") or DEFAULT_PATH
        self._lock = threading.Lock()
        self.totals = self._load()

    def _load(self) -> Totals:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return Totals(
                calls=int(d.get("calls", 0)),
                input_tokens=int(d.get("input_tokens", 0)),
                output_tokens=int(d.get("output_tokens", 0)),
                cost_usd=float(d.get("cost_usd", 0.0)),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return Totals()

    def _flush(self) -> None:
        # Write temp then atomically replace, so a crash mid-write can't truncate totals.
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.totals), f, indent=2)
        os.replace(tmp, self.path)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> dict:
        """Add one call's usage to lifetime totals, persist, and return this call's delta."""
        it, ot = int(input_tokens or 0), int(output_tokens or 0)
        cost = cost_usd(model, it, ot)
        with self._lock:
            self.totals.calls += 1
            self.totals.input_tokens += it
            self.totals.output_tokens += ot
            self.totals.cost_usd = round(self.totals.cost_usd + cost, 6)
            self._flush()
        return {"model": model, "input_tokens": it, "output_tokens": ot,
                "cost_usd": round(cost, 6)}

    def snapshot(self) -> Totals:
        with self._lock:
            return Totals(**asdict(self.totals))


# A process-wide default tracker so every provider shares one lifetime ledger.
_GLOBAL: Optional[MetricsTracker] = None
_GLOBAL_LOCK = threading.Lock()


def global_tracker() -> MetricsTracker:
    global _GLOBAL
    if _GLOBAL is None:
        with _GLOBAL_LOCK:
            if _GLOBAL is None:
                _GLOBAL = MetricsTracker()
    return _GLOBAL