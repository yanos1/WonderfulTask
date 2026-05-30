"""Throwaway prototype: compare current (runway-size) feasibility vs a utilization
signal (passengers / runway-capacity, inverted) on the real dataset.

Run:  python -m etl.proto_feasibility

Nothing here is wired into the app — it only prints a before/after so we can decide
whether the utilization framing is worth adopting (and whether OSM land area is worth
chasing on top of it).
"""

import json
import os

from airport_intel.scoring import ScoringEngine, _Percentiler, FEASIBILITY_FLOOR

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "airports.json")


def utilization_feasibility(airports: list[dict]):
    """feasibility = FLOOR + (1-FLOOR) * (1 - percentile(passengers / runway_capacity)).

    capacity proxy = runway_count * runway_length_ft (total linear runway feet).
    Low utilization (lots of runway per passenger) -> lots of headroom -> high feasibility.
    Airports with no runway data can't be assessed -> floor (conservative).
    """
    def util(a):
        cap = (a.get("runway_count") or 0) * (a.get("runway_length_ft") or 0)
        pax = a.get("passengers")
        if cap <= 0 or not pax:
            return None
        return pax / cap

    pct = _Percentiler([u for a in airports if (u := util(a)) is not None])

    def score(a):
        u = util(a)
        if u is None:
            return FEASIBILITY_FLOOR
        raw = 1 - pct.rank(u)            # invert: crowded -> low feasibility
        return FEASIBILITY_FLOOR + (1 - FEASIBILITY_FLOOR) * raw

    return score


def main():
    with open(DATA) as f:
        blob = json.load(f)
    airports = list(blob["airports"].values())

    eng = ScoringEngine(airports)
    util_feas = utilization_feasibility(airports)

    rows = []
    for a in airports:
        demand, _ = eng.demand(a)
        f_old = eng.feasibility(a)
        f_new = util_feas(a)
        rows.append({
            "iata": a["iata"],
            "name": a.get("name", "")[:28],
            "pax": a.get("passengers") or 0,
            "rwy": a.get("runway_count") or 0,
            "demand": demand,
            "f_old": f_old,
            "f_new": f_new,
            "epi_old": demand * f_old * 100,
            "epi_new": demand * f_new * 100,
        })

    def show(title, key):
        print(f"\n=== Top 15 by {title} ===")
        print(f"  {'IATA':5}{'name':30}{'pax':>11}{'rwy':>4}{'demand':>8}{'f_old':>7}{'f_new':>7}{'EPI':>8}")
        for r in sorted(rows, key=lambda r: r[key], reverse=True)[:15]:
            print(f"  {r['iata']:5}{r['name']:30}{r['pax']:>11,}{r['rwy']:>4}"
                  f"{r['demand']:>8.3f}{r['f_old']:>7.2f}{r['f_new']:>7.2f}{r[key]:>8.1f}")

    show("OLD EPI (runway-size feasibility)", "epi_old")
    show("NEW EPI (utilization feasibility)", "epi_new")

    print("\n=== Named airports from the brief: feasibility old -> new ===")
    print(f"  {'IATA':5}{'pax':>11}{'rwy':>4}{'f_old':>8}{'f_new':>8}   delta")
    by_iata = {r["iata"]: r for r in rows}
    for code in ["SFO", "LAX", "SNA", "ANC", "BOS", "ATL", "JFK"]:
        r = by_iata.get(code)
        if not r:
            print(f"  {code:5} (missing)")
            continue
        print(f"  {code:5}{r['pax']:>11,}{r['rwy']:>4}{r['f_old']:>8.2f}{r['f_new']:>8.2f}"
              f"   {r['f_new'] - r['f_old']:+.2f}")


if __name__ == "__main__":
    main()