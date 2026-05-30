"""ETL: build airports.json from public aviation data (no API keys required).

Vintage strategy (documented as an explicit assumption in DESIGN.md):
  - Current volume + YoY growth come from FAA CY2024 enplanements (2024 vs 2023).
  - Structural ratios (load factor, long-haul share) come from the BTS T-100 Domestic
    Segment file for BASE_YEAR (2024), fetched live from TranStats by etl.fetch_t100_segment
    (it uniquely carries SEATS and DISTANCE). If that fetch fails the build fails loud rather
    than shipping a stale vintage; the committed data/airports.json is left untouched.

Sources (all public, keyless):
  - OurAirports airports.csv + runways.csv  -> metadata + runways (feasibility proxy)
  - BTS T-100 Domestic Segment (All Carriers), live from TranStats -> load factor, long-haul %
  - FAA CY2024 commercial-service enplanements -> current volume + YoY growth

Run:  python -m etl.build_airports        (full build -> data/airports.json)
"""

import json
import os
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone

import pandas as pd

from airport_intel.models import Airport
from etl import fetch_t100_segment

SCHEMA_VERSION = 1

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# data/ lives at the repo root, one level above this etl/ package
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_PATH = os.path.join(DATA_DIR, "airports.json")

OURAIRPORTS_AIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OURAIRPORTS_RUNWAYS = "https://davidmegginson.github.io/ourairports-data/runways.csv"
# Segment source: live BTS T-100, fetched per-year by etl.fetch_t100_segment.
T100_SOURCE = "BTS TranStats T-100 Domestic Segment (All Carriers), DL_SelectFields gnoyr_VQ=GEE"
FAA_ENPLANEMENTS = (
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/"
    "passenger/arp-cy2024-commercial-service-enplanements.xlsx"
)
FAA_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # FAA blocks default agents

BASE_YEAR = 2024          # vintage of the segment-level (seats/distance) data; was 2013
CURRENT_YEAR = 2024       # vintage of FAA enplanement volume + growth
LONG_HAUL_MILES = 2500    # a flight is "long haul" if its segment distance exceeds this
# Service classes to KEEP (passenger service). Excludes all-cargo classes G and P,
# honoring the "ignore cargo" scoping decision.
PASSENGER_CLASSES = {"F", "L"}

NAMED = ["SFO", "LAX", "SNA", "ANC", "BOS"]  # sanity-check airports from the brief


# --------------------------------------------------------------------------- #
# Download helper (cache-aside: fetch once, reuse the local copy)
# --------------------------------------------------------------------------- #
def download(url: str, filename: str, headers: dict | None = None) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"  downloading {filename} ...")
        req = urllib.request.Request(url, headers=headers or {})
        data = urllib.request.urlopen(req, timeout=120).read()
        with open(path, "wb") as f:
            f.write(data)
    else:
        print(f"  cached     {filename}")
    return path


# --------------------------------------------------------------------------- #
# Source loaders
# --------------------------------------------------------------------------- #
def load_ourairports():
    """Return (meta_by_iata, runways_by_iata) for US airports with an IATA code."""
    ap = pd.read_csv(download(OURAIRPORTS_AIRPORTS, "airports.csv"), low_memory=False)
    rw = pd.read_csv(download(OURAIRPORTS_RUNWAYS, "runways.csv"), low_memory=False)

    ap = ap[(ap["iso_country"] == "US") & ap["iata_code"].notna()].copy()
    ap["iata_code"] = ap["iata_code"].str.upper()

    # runways are keyed by airport_ident (== OurAirports `ident`); aggregate per ident
    rw_open = rw[rw["closed"] == 0]
    rw_agg = rw_open.groupby("airport_ident").agg(
        runway_count=("id", "count"),
        runway_length_ft=("length_ft", "max"),  # longest runway = practical capacity proxy
    )

    meta, runways = {}, {}
    for _, r in ap.iterrows():
        iata = r["iata_code"]
        meta[iata] = {
            "name": r["name"],
            "city": r.get("municipality"),
            "state": str(r["iso_region"]).split("-")[-1] if pd.notna(r["iso_region"]) else None,
            "type": r["type"],
        }
        agg = rw_agg.loc[r["ident"]] if r["ident"] in rw_agg.index else None
        runways[iata] = {
            "runway_count": int(agg["runway_count"]) if agg is not None else 0,
            "runway_length_ft": float(agg["runway_length_ft"]) if agg is not None and pd.notna(agg["runway_length_ft"]) else 0.0,
        }
    return meta, runways


def _read_t100_frame(year: int):
    """Return a DataFrame of raw T-100 segment rows for `year`.

    Source is the live BTS T-100 Domestic Segment (All Carriers) file, fetched on demand
    into data/raw and read from its zip. We deliberately do NOT fall back to an older
    vintage: shipping stale ratios silently would be worse than not rebuilding. The
    committed data/airports.json is the safety net -- a failed rebuild leaves it untouched
    rather than overwriting current data with a degraded one.
    """
    zip_path = os.path.join(RAW_DIR, f"t100_segment_{year}.zip")
    if not os.path.exists(zip_path):
        zip_path = fetch_t100_segment.fetch(year)  # raises (fails loud) if TranStats is down
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.upper().startswith("T_T100"))
        return pd.read_csv(zf.open(name), low_memory=False)


def load_t100(year: int = BASE_YEAR):
    """Aggregate T-100 segment data by origin: passengers, load factor, long-haul %."""
    df = _read_t100_frame(year)
    # keep passenger service classes only (drop all-cargo) and rows with real capacity
    df = df[df["CLASS"].isin(PASSENGER_CLASSES)].copy()
    df["ORIGIN"] = df["ORIGIN"].str.upper()

    df["lh_departures"] = df["DEPARTURES_PERFORMED"].where(df["DISTANCE"] > LONG_HAUL_MILES, 0.0)

    grp = df.groupby("ORIGIN").agg(
        passengers=("PASSENGERS", "sum"),
        seats=("SEATS", "sum"),
        departures=("DEPARTURES_PERFORMED", "sum"),
        lh_departures=("lh_departures", "sum"),
    )

    out = {}
    for iata, r in grp.iterrows():
        seats = r["seats"]
        deps = r["departures"]
        out[iata] = {
            "passengers_hist": float(r["passengers"]),
            "departures_hist": float(deps),
            "load_factor": round(float(r["passengers"] / seats), 4) if seats > 0 else None,
            "long_haul_pct": round(float(r["lh_departures"] / deps), 4) if deps > 0 else None,
        }
    return out


def load_faa_enplanements():
    """FAA CY2024 commercial-service enplanements by airport -> current volume + YoY growth.

    On any failure return {} so the build degrades gracefully to the 2013 baseline (vintage
    is recorded per airport and surfaced to the user) rather than failing the pipeline.
    """
    try:
        path = download(FAA_ENPLANEMENTS, "faa_cy2024_enplanements.xlsx", headers=FAA_UA)
        df = pd.read_excel(path, header=0)
    except Exception as e:  # noqa: BLE001 - resilience by design
        print(f"  FAA enplanements unavailable ({type(e).__name__}); falling back to 2013 baseline")
        return {}

    out = {}
    for _, r in df.iterrows():
        code = str(r.get("Locid", "")).strip().upper()
        enp = r.get("CY 24 Enplanements")
        chg = r.get("% Change")
        if not code or code == "NAN" or pd.isna(enp):
            continue
        out[code] = {
            "enplanements": float(enp),
            "growth": round(float(chg), 4) if pd.notna(chg) else None,
        }
    print(f"  FAA enplanements: {len(out)} airports (CY2024 vs CY2023 YoY)")
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build():
    print("Loading OurAirports ...")
    meta, runways = load_ourairports()
    print(f"Loading BTS T-100 {BASE_YEAR} segment ...")
    seg = load_t100(BASE_YEAR)
    print("Loading FAA CY2024 enplanements ...")
    faa = load_faa_enplanements()

    # union of airports with current (FAA) or segment (T-100) passenger activity
    codes = set(faa.keys()) | set(seg.keys())
    airports = {}
    for iata in sorted(codes):
        if iata not in meta:
            continue  # need metadata (name/city/state) to be useful
        cur = faa.get(iata)
        hist = seg.get(iata, {})

        # Volume + growth: prefer current FAA (2024); else fall back to the T-100 segment
        # passenger count (same BASE_YEAR vintage).
        if cur and cur.get("enplanements"):
            volume, vintage, growth = cur["enplanements"], CURRENT_YEAR, cur.get("growth")
        else:
            volume, vintage, growth = (hist.get("passengers_hist") or 0), BASE_YEAR, None

        deps = hist.get("departures_hist")
        airports[iata] = {
            "iata": iata,
            **meta[iata],
            # demand volume (with vintage so uncertainty is explicit downstream)
            "passengers": round(volume),
            "passengers_vintage": vintage,
            "departures": round(deps) if deps else None,
            # structural ratios (T-100 segment, BASE_YEAR)
            "load_factor": hist.get("load_factor"),
            "long_haul_pct": hist.get("long_haul_pct"),
            # growth: FAA CY2024 vs CY2023 YoY (None when current data is unavailable)
            "pax_growth_yoy": growth,
            # feasibility inputs
            **runways.get(iata, {"runway_count": 0, "runway_length_ft": 0.0}),
        }

    # Validate every record against the typed schema before publishing: the ETL fails
    # loud here if it produced a malformed value, instead of shipping a bad dataset.
    for rec in airports.values():
        Airport.from_raw(rec)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(airports),
        "vintages": {
            "volume_and_growth": CURRENT_YEAR,   # FAA enplanements (current)
            "structural_ratios": BASE_YEAR,       # load factor, long-haul % (T-100 segment)
        },
        "sources": {
            "metadata_runways": OURAIRPORTS_AIRPORTS,
            "segment_ratios": T100_SOURCE,
            "enplanements": FAA_ENPLANEMENTS,
        },
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"_meta": meta, "airports": airports}, f, indent=2)

    print(f"\nWrote {len(airports)} airports (schema v{SCHEMA_VERSION}) -> {OUT_PATH}\n")
    print("Sanity check (named airports from the brief):")
    print(f"  {'IATA':5} {'passengers':>12} {'vint':>5} {'loadF':>6} {'longHaul':>8} {'growth':>7} {'rwys':>4}")
    for code in NAMED:
        a = airports.get(code)
        if not a:
            print(f"  {code:5} (missing)")
            continue
        print(
            f"  {code:5} {a['passengers']:>12,} {a['passengers_vintage']:>5} {str(a['load_factor']):>6} "
            f"{str(a['long_haul_pct']):>8} {str(a['pax_growth_yoy']):>7} {a['runway_count']:>4}"
        )


if __name__ == "__main__":
    build()
