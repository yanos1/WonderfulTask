"""Tests for the LLM usage metrics tracker (counts, cost estimate, monotonic persistence)."""

import json

from airport_intel import metrics
from airport_intel.metrics import MetricsTracker, cost_usd, price_for


def test_price_for_matches_by_longest_prefix():
    # Dated model suffixes still resolve to the right base rate.
    assert price_for("claude-3-5-sonnet-20241022") == (3.0, 15.0)
    assert price_for("gemini-2.0-flash") == (0.10, 0.40)


def test_unknown_model_falls_back_to_default_rate():
    assert price_for("some-future-model-v9") == metrics._DEFAULT_RATE


def test_cost_math_uses_per_model_rates():
    # claude-3-5-sonnet = $3/MTok in, $15/MTok out -> 1M+1M tokens = $18.
    assert round(cost_usd("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000), 6) == 18.0


def test_record_accumulates_and_persists(tmp_path):
    p = tmp_path / "usage.json"
    t = MetricsTracker(str(p))
    delta = t.record("gemini-2.0-flash", 1000, 500)
    t.record("gemini-2.0-flash", 2000, 1000)

    snap = t.snapshot()
    assert snap.calls == 2
    assert snap.input_tokens == 3000
    assert snap.output_tokens == 1500
    assert snap.total_tokens == 4500
    assert snap.cost_usd > 0
    assert delta["input_tokens"] == 1000  # per-call delta is returned

    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["calls"] == 2  # flushed to disk


def test_totals_are_monotonic_across_reload(tmp_path):
    p = tmp_path / "usage.json"
    MetricsTracker(str(p)).record("gemini-2.0-flash", 100, 100)

    # A fresh tracker over the same file continues the tally; it never resets to zero.
    t2 = MetricsTracker(str(p))
    assert t2.snapshot().calls == 1
    t2.record("gemini-2.0-flash", 100, 100)
    assert t2.snapshot().calls == 2


def test_corrupt_file_starts_from_zero_without_crashing(tmp_path):
    p = tmp_path / "usage.json"
    p.write_text("{ not valid json", encoding="utf-8")
    assert MetricsTracker(str(p)).snapshot().calls == 0


def test_missing_tokens_are_treated_as_zero(tmp_path):
    # A vendor response that omits usage passes None; it must not crash the count.
    t = MetricsTracker(str(tmp_path / "usage.json"))
    t.record("gemini-2.0-flash", None, None)
    assert t.snapshot().calls == 1
    assert t.snapshot().total_tokens == 0


def test_provider_record_usage_is_injectable_and_best_effort(tmp_path):
    # The base helper routes vendor token counts into an injected tracker without needing a
    # super().__init__ call, and swallows tracker errors so an answer is never broken.
    from airport_intel.llm.base import LLMProvider

    class _Dummy(LLMProvider):
        name = "dummy"
        model = "claude-3-5-sonnet-20241022"

        def complete(self, system, messages, json_mode=False):
            return ""

    d = _Dummy()
    d._metrics = MetricsTracker(str(tmp_path / "usage.json"))
    d._record_usage(10, 20)
    assert d.metrics.snapshot().calls == 1
    assert d.metrics.snapshot().total_tokens == 30