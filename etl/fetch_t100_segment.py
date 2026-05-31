"""Fetch a year of BTS T-100 Domestic Segment (All Carriers) data straight from
TranStats, selecting only the columns the ETL needs.

The TranStats download form is ASP.NET WebForms: a GET yields per-request
__VIEWSTATE / __EVENTVALIDATION tokens that must be echoed back in the POST.
We request a single year, all months, all geography, zipped -- and only the
seven fields build_airports.py aggregates, which keeps the payload small.

Usage:  python -m etl.fetch_t100_segment 2024
Writes: data/raw/t100_segment_<year>.zip  (a CSV inside, BTS-named)
"""

import http.cookiejar
import os
import re
import sys
import urllib.parse
import urllib.request

# T-100 Domestic Segment (All Carriers); gnoyr_VQ is TranStats' table token.
FORM_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    "?gnoyr_VQ=GEE&QO_fu146_anzr=Nv4+Pn44vr45"
)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
# Only the fields build_airports.load_t100 reads (CLASS/ORIGIN/SEATS/DISTANCE/
# PASSENGERS/DEPARTURES_PERFORMED). DEST kept for traceability; tiny.
FIELDS = ["ORIGIN", "DEST", "PASSENGERS", "SEATS", "DEPARTURES_PERFORMED", "DISTANCE", "CLASS"]

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def _hidden(html: str, name: str) -> str:
    m = re.search(rf'id="{name}"[^>]*value="([^"]*)"', html) or re.search(
        rf'name="{name}"[^>]*value="([^"]*)"', html
    )
    if not m:
        raise RuntimeError(f"could not find ASP.NET token {name}")
    return m.group(1)


def fetch(year: int) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    opener.addheaders = [("User-Agent", UA)]

    print(f"  GET download form (tokens) ...")
    html = opener.open(FORM_URL, timeout=120).read().decode("utf-8", "replace")

    form = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__VIEWSTATE": _hidden(html, "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": _hidden(html, "__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION": _hidden(html, "__EVENTVALIDATION"),
        "cboGeography": "All",
        "cboYear": str(year),
        "cboPeriod": "All",
        "chkDownloadZip": "on",
        "btnDownload": "Download",
    }
    for f in FIELDS:
        form[f] = "on"

    print(f"  POST request for {year} (all months, all geography, {len(FIELDS)} fields) ...")
    req = urllib.request.Request(
        FORM_URL,
        data=urllib.parse.urlencode(form).encode(),
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = opener.open(req, timeout=600)
    ctype = resp.headers.get("Content-Type", "")
    body = resp.read()
    if b"PK\x03\x04" not in body[:4] and "zip" not in ctype.lower():
        snippet = body[:300].decode("utf-8", "replace")
        raise RuntimeError(f"expected a zip, got Content-Type={ctype!r}; body starts: {snippet!r}")

    out = os.path.join(RAW_DIR, f"t100_segment_{year}.zip")
    with open(out, "wb") as fh:
        fh.write(body)
    print(f"  wrote {out}  ({len(body) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    fetch(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)