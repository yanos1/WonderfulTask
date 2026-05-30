"""Tests for the EPI backtest: metric primitives (toy data) + a synthetic end-to-end run.

No file or network dependency — a stub repository supplies a tiny airport universe and an
in-memory funding dict, so the validation logic is exercised deterministically.
"""

from airport_intel import backtest
from airport_intel.backtest import precision_at_n, run_backtest, spearman

# --- metric primitives ----------------------------------------------------- #


def test_precision_at_n_counts_positives_in_top_n():
    ranking = ["A", "B", "C", "D"]
    positives = {"A", "C"}
    assert precision_at_n(ranking, positives, 2) == 0.5   # A yes, B no
    assert precision_at_n(ranking, positives, 4) == 0.5   # A,C yes of 4
    assert precision_at_n(ranking, positives, 1) == 1.0   # A yes


def test_precision_at_n_handles_n_past_end():
    assert precision_at_n(["A"], {"A"}, 10) == 1.0        # n clamped to len
    assert precision_at_n([], {"A"}, 5) == 0.0


def test_spearman_perfect_monotonic():
    x = [1, 2, 3, 4, 5]
    assert spearman(x, [10, 20, 30, 40, 50]) == 1.0       # same order
    assert spearman(x, [50, 40, 30, 20, 10]) == -1.0      # reversed


def test_spearman_handles_ties_and_degenerate():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0          # no variance in x
    assert spearman([5], [5]) == 0.0                      # too few points


# --- synthetic end-to-end --------------------------------------------------- #

# Ordered so EPI(AAA) > EPI(BBB) > EPI(CCC) > EPI(DDD): demand + feasibility both descend.
_AIRPORTS = [
    {"iata": "AAA", "name": "Alpha", "city": "A", "state": "AA", "load_factor": 0.95,
     "passengers": 9_000_000, "pax_growth_yoy": 0.08, "runway_count": 4, "runway_length_ft": 12000},
    {"iata": "BBB", "name": "Bravo", "city": "B", "state": "BB", "load_factor": 0.80,
     "passengers": 3_000_000, "pax_growth_yoy": 0.04, "runway_count": 3, "runway_length_ft": 10000},
    {"iata": "CCC", "name": "Charlie", "city": "C", "state": "CC", "load_factor": 0.65,
     "passengers": 1_000_000, "pax_growth_yoy": 0.02, "runway_count": 2, "runway_length_ft": 8000},
    {"iata": "DDD", "name": "Delta", "city": "D", "state": "DD", "load_factor": 0.40,
     "passengers": 400_000, "pax_growth_yoy": -0.01, "runway_count": 1, "runway_length_ft": 5000},
]


class _StubRepo:
    def all(self):
        return list(_AIRPORTS)

    def get(self, code):
        return next((a for a in _AIRPORTS if a["iata"] == code.upper()), None)


def test_run_backtest_scores_and_labels():
    # Fund the two strongest airports -> the EPI top-2 should be perfectly precise.
    funding = {"AAA": {"funded": True, "announced_usd": 50_000_000},
               "BBB": {"funded": True, "announced_usd": 20_000_000}}
    res = run_backtest(repo=_StubRepo(), funding=funding, ns=(2,), min_passengers=0)

    assert res["universe"] == 4
    assert res["funded_in_universe"] == 2
    assert res["precision"][0]["n"] == 2
    assert res["precision"][0]["epi_precision"] == 1.0   # AAA, BBB are the top-2 by EPI

    # EPI aligns with grant dollars here -> positive rank correlation.
    assert res["spearman_epi_grant_usd"] > 0

    # Shortlist = high-EPI airports that were NOT funded; funded ones are excluded.
    codes = [r["iata"] for r in res["shortlist"]]
    assert "AAA" not in codes and "BBB" not in codes
    assert "CCC" in codes                                 # highest-EPI unfunded airport


def test_shortlist_size_floor_excludes_small_airports():
    res = run_backtest(repo=_StubRepo(), funding={}, ns=(2,), min_passengers=2_000_000)
    codes = [r["iata"] for r in res["shortlist"]]
    assert codes == ["AAA", "BBB"]                        # CCC (1M) and DDD (400k) filtered out


def test_load_funding_missing_file_raises(tmp_path):
    try:
        backtest.load_funding(str(tmp_path / "nope.json"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "build_funding" in str(e)                 # message points at the fix
