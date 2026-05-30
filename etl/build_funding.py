"""ETL: build funding.json from the FAA IIJA Airport Terminal Program (ATP) grant file.

This produces the EXTERNAL GROUND TRUTH the EPI is validated against (see
``airport_intel/backtest.py``). The ATP is a *competitive* program ($5B, FY2022-2026) that
funds passenger-terminal development -- the same thing the EPI tries to score. We treat the
FAA's published award list as labels, never as authored data: this is the opposite of
hand-curation, so it is consistent with the "derived/ground-truth over authored" principle
in DESIGN.md.

Freshness strategy (mirrors build_airports.py):
  1. fetch the live FAA workbook with a browser User-Agent (FAA 403s default agents),
     cache-aside into data/raw/;
  2. if the source is unreachable / blocked, fall back to the committed snapshot
     (data/atp_snapshot.xlsx) so the build is reproducible offline and the app always runs;
  3. stamp provenance (source_url, file vintage, fiscal years) into funding.json.

Re-run to pick up new award rounds. FY2026 is the program's final announcement, after which
the label set is complete -- so this ground truth stabilises rather than drifting forever.

Run:  python -m etl.build_funding        (-> data/funding.json)
"""

import json
import os
import shutil
import warnings
from datetime import datetime, timezone

import pandas as pd

# Reuse the existing ETL primitives instead of duplicating them (same cache-aside + UA).
from etl.build_airports import DATA_DIR, FAA_UA, download

SCHEMA_VERSION = 1
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Authoritative FAA ATP grant-status workbook. The dated filename is the FAA's; bump it when
# a newer status file is published (the URL pattern is stable: /iija/atp/<file>).
ATP_URL = "https://www.faa.gov/iija/atp/ATP_Grant_Status_1-27-2026.xlsx"
ATP_FILE = "ATP_Grant_Status_1-27-2026.xlsx"          # cache-aside name under data/raw/
SNAPSHOT_PATH = os.path.join(DATA_DIR, "atp_snapshot.xlsx")  # committed offline fallback
OUT_PATH = os.path.join(DATA_DIR, "funding.json")

PROGRAM = "FAA IIJA Airport Terminal Program (ATP)"
# Workbook layout: row 0 = title, row 1 = "as of" date, row 2 = column names, row 3+ = data.
HEADER_ROW = 2
SHEET = "ATP"


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _money(value) -> float | None:
    """Parse '$1,357,046' / 1357046 / '' / NaN -> float USD or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _resolve_source() -> str:
    """Live FAA file first (cache-aside); fall back to the committed snapshot if blocked."""
    try:
        return download(ATP_URL, ATP_FILE, headers=FAA_UA)
    except Exception as e:  # noqa: BLE001 - resilience by design (FAA 403 / offline)
        if os.path.exists(SNAPSHOT_PATH):
            print(f"  live ATP source unavailable ({type(e).__name__}); "
                  f"using committed snapshot {os.path.basename(SNAPSHOT_PATH)}")
            return SNAPSHOT_PATH
        raise SystemExit(
            f"ATP source unreachable ({e}) and no snapshot at {SNAPSHOT_PATH}. "
            "Download the workbook once in an unblocked environment to seed the snapshot."
        )


def load_atp() -> dict:
    """Return {LOCID: aggregated funding record} from the ATP workbook."""
    df = pd.read_excel(_resolve_source(), sheet_name=SHEET, header=HEADER_ROW)
    df = df[df["LOCID"].notna()
            & df["FY Announced"].astype(str).str.match(r"^\d{4}$", na=False)].copy()
    df["LOCID"] = df["LOCID"].str.strip().str.upper()

    funding: dict[str, dict] = {}
    for code, grp in df.groupby("LOCID"):
        years = sorted({int(y) for y in grp["FY Announced"]})
        announced = sum(m for m in (_money(v) for v in grp["DOT Secretary Announced Amount"]) if m)
        awarded = sum(m for m in (_money(v) for v in grp["Grant Award Amount"]) if m)
        first = grp.iloc[0]
        funding[code] = {
            "iata": code,
            "funded": True,
            "program": "ATP",
            "grant_count": int(len(grp)),
            "announced_usd": round(announced),
            "awarded_usd": round(awarded),
            "fiscal_years": years,
            "first_year": years[0],
            "state": str(first["State"]).upper() if pd.notna(first["State"]) else None,
            "city": str(first["City"]) if pd.notna(first["City"]) else None,
            "name": str(first["Airport"]) if pd.notna(first["Airport"]) else None,
        }
    return funding


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> None:
    print(f"Loading {PROGRAM} ...")
    funding = load_atp()

    # Seed/refresh the committed offline snapshot from a successful live fetch, so the
    # fallback path keeps working in future blocked/offline runs.
    cached = os.path.join(DATA_DIR, "raw", ATP_FILE)
    if os.path.exists(cached) and os.path.abspath(cached) != os.path.abspath(SNAPSHOT_PATH):
        shutil.copyfile(cached, SNAPSHOT_PATH)

    total = sum(r["announced_usd"] for r in funding.values())
    years = sorted({y for r in funding.values() for y in r["fiscal_years"]})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "program": PROGRAM,
        "source_url": ATP_URL,
        "source_file": ATP_FILE,
        "fiscal_years": years,
        "vintage": f"FY{years[0]}-FY{years[-1]} announced" if years else "n/a",
        "record_count": len(funding),
        "total_announced_usd": round(total),
        "note": ("Competitive terminal-development grants used as EXTERNAL GROUND TRUTH for "
                 "the EPI backtest (airport_intel/backtest.py). Published FAA labels, not "
                 "authored data."),
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"_meta": meta, "funding": funding}, f, indent=2)

    span = f"FY{years[0]}-{years[-1]}" if years else "n/a"
    print(f"\nWrote {len(funding)} ATP-funded airports "
          f"(${total:,.0f} announced, {span}) -> {OUT_PATH}")
    print("Top by announced amount:")
    for r in sorted(funding.values(), key=lambda x: x["announced_usd"], reverse=True)[:8]:
        print(f"  {r['iata']:4} {r['announced_usd']:>14,}  {r.get('city')}")


if __name__ == "__main__":
    build()
