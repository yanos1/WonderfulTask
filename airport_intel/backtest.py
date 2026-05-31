"""Validate the EPI against external ground truth: FAA Airport Terminal Program (ATP) grants.

The EPI is a deterministic *prediction* of where terminal expansion is most worthwhile. The
ATP is $5B of *competitive* federal grants (FY2022-2026) that funded exactly that. So we can
treat ATP awards as labels and ask the question every reviewer will ask: **does the EPI
actually track reality, or is it a nice-looking number?**

Three uses, one artifact:
  1. PROVE   - precision@N of the EPI against ATP labels (does EPI beat the base rate?),
               plus rank correlation with grant dollars.
  2. TUNE    - the precision is an objective function for the scoring weights: they can be
               justified empirically instead of asserted (and pinned as a regression test).
  3. LEADS   - the off-diagonal (high EPI, NOT funded) is the investment shortlist: airports
               the competitive process has not captured yet.

Honesty: ATP also weighs geographic/equity distribution, not pure ROI, and "funded" airports
may already be addressed -- so this is *corroboration + lead generation*, not proof of profit
(no public profit ground truth exists). The unfunded high-EPI shortlist is the real signal.

This module is pure analysis: it imports the existing scoring/repository seams and the funding
ETL output, and touches neither. Run:  python -m airport_intel.backtest
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .repository import Repository, get_repository
from .scoring import ScoringEngine

DEFAULT_FUNDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "funding.json"
)
DEFAULT_NS = (10, 20, 50)       # precision@N cutoffs
HIGH_EPI_QUANTILE = 0.25        # "high EPI" = top quartile, for the 2x2 quadrant


# --------------------------------------------------------------------------- #
# Funding labels
# --------------------------------------------------------------------------- #
def load_funding(path: str = DEFAULT_FUNDING_PATH) -> dict:
    """Load {IATA: funding record}. Accepts the wrapped {_meta, funding} form."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it first:  python -m etl.build_funding"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("funding", raw) if isinstance(raw, dict) else {}


# --------------------------------------------------------------------------- #
# Metric primitives (pure, unit-tested on toy data)
# --------------------------------------------------------------------------- #
def precision_at_n(ranking: list[str], positives: set[str], n: int) -> float:
    """Fraction of the top-n ranked codes that are positives (funded)."""
    n = min(n, len(ranking))
    if n == 0:
        return 0.0
    return sum(1 for code in ranking[:n] if code in positives) / n


def _avg_ranks(values: list[float]) -> list[float]:
    """Rank values ascending, averaging ties (for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else 0.0


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks; no scipy dependency)."""
    if len(x) < 2:
        return 0.0
    return round(_pearson(_avg_ranks(x), _avg_ranks(y)), 4)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_backtest(
    repo: Optional[Repository] = None,
    funding: Optional[dict] = None,
    weights: Optional[dict] = None,
    ns: tuple[int, ...] = DEFAULT_NS,
    min_passengers: int = 250_000,
) -> dict:
    """Score the whole universe and measure EPI precision against ATP labels."""
    repo = repo or get_repository()
    funding = load_funding() if funding is None else funding
    airports = repo.all()
    engine = ScoringEngine(airports)

    rows = []
    for a in airports:
        code = a["iata"]
        fund = funding.get(code)
        rows.append({
            "iata": code,
            "name": a.get("name"),
            "city": a.get("city"),
            "state": a.get("state"),
            "passengers": a.get("passengers") or 0,
            "epi": engine.epi(a, weights)["epi"],
            "funded": bool(fund),
            "grant_usd": (fund or {}).get("announced_usd", 0) or 0,
        })

    funded_set = {r["iata"] for r in rows if r["funded"]}
    n_universe = len(rows)
    base_rate = round(len(funded_set) / n_universe, 4) if n_universe else 0.0

    epi_rank = [r["iata"] for r in sorted(rows, key=lambda r: r["epi"], reverse=True)]

    precision = []
    for n in ns:
        epi_p = precision_at_n(epi_rank, funded_set, n)
        precision.append({
            "n": n,
            "epi_precision": round(epi_p, 4),
        })

    # rank correlation of EPI with grant dollars across the whole universe (unfunded = $0)
    corr = spearman([r["epi"] for r in rows], [r["grant_usd"] for r in rows])

    # 2x2 quadrant: "high EPI" = top quartile by EPI
    epis_sorted = sorted((r["epi"] for r in rows))
    # EPI value at the top-quartile boundary (index clamped to a valid position)
    idx = max(0, min(len(epis_sorted) - 1, int(len(epis_sorted) * (1 - HIGH_EPI_QUANTILE))))
    high_cut = epis_sorted[idx]
    quad = {"high_funded": 0, "high_unfunded": 0, "low_funded": 0, "low_unfunded": 0}
    for r in rows:
        hi = r["epi"] >= high_cut
        key = ("high" if hi else "low") + ("_funded" if r["funded"] else "_unfunded")
        quad[key] += 1

    # the product: high-EPI airports that have NOT received an ATP grant (the opportunities).
    # A credible-size floor keeps these as real terminal-expansion candidates, not noise.
    shortlist = [r for r in sorted(rows, key=lambda r: r["epi"], reverse=True)
                 if not r["funded"] and r["passengers"] >= min_passengers][:15]

    return {
        "universe": n_universe,
        "funded_in_universe": len(funded_set),
        "base_rate": base_rate,
        "precision": precision,
        "spearman_epi_grant_usd": corr,
        "high_epi_cutoff": high_cut,
        "quadrant": quad,
        "shortlist": shortlist,
        "shortlist_min_passengers": min_passengers,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(result: dict) -> str:
    L = []
    L.append("=" * 70)
    L.append("EPI BACKTEST  vs  FAA Airport Terminal Program (ATP) grants")
    L.append("=" * 70)
    L.append("EPI (Expansion Profitability Index) is a 0-100 score of how worthwhile a")
    L.append("terminal expansion is at each airport: EPI = demand x feasibility, where")
    L.append("demand blends load factor, growth, passenger volume and long-haul mix, and")
    L.append("feasibility reflects runway capacity to build. This backtest checks whether")
    L.append("that score tracks where competitive FAA ATP grants actually went.")
    L.append("")
    L.append(f"Universe: {result['universe']} airports | "
             f"ATP-funded in universe: {result['funded_in_universe']} "
             f"(base rate {result['base_rate']:.1%})")
    L.append("")
    L.append("Precision@N - share of the top-N ranking that actually received ATP funding")
    L.append(f"  (vs base rate {result['base_rate']:.0%} = funded share of the whole universe)")
    L.append(f"  {'Top N':>6}  {'Funded':>8}")
    for p in result["precision"]:
        L.append(f"  {p['n']:>6}  {p['epi_precision']:>7.0%}")
    L.append("")
    L.append(f"Spearman(EPI, grant $) across universe: {result['spearman_epi_grant_usd']:+.3f}")
    q = result["quadrant"]
    L.append("")
    L.append("Confusion 2x2 (high EPI = top 25%):")
    L.append(f"                 funded     not-funded")
    L.append(f"  high EPI    {q['high_funded']:>7}   {q['high_unfunded']:>10}  <- shortlist pool")
    L.append(f"  low  EPI    {q['low_funded']:>7}   {q['low_unfunded']:>10}")
    L.append("")
    L.append(f"INVESTMENT SHORTLIST - high EPI, NOT yet ATP-funded "
             f"(>= {result['shortlist_min_passengers']:,} pax, top 15):")
    L.append(f"  {'IATA':>4}  {'EPI':>6}  {'pax':>12}  city")
    for r in result["shortlist"]:
        L.append(f"  {r['iata']:>4}  {r['epi']:>6.1f}  {r['passengers']:>12,}  {r.get('city')}")
    L.append("=" * 70)
    return "\n".join(L)


if __name__ == "__main__":
    print(format_report(run_backtest()))
