"""US region -> state-code resolution (deterministic; used by the rank tool for Q1)."""

from __future__ import annotations

from typing import Optional

REGIONS: dict[str, list[str]] = {
    "new england": ["ME", "NH", "VT", "MA", "RI", "CT"],
    "mid-atlantic": ["NY", "NJ", "PA", "DE", "MD", "DC", "VA"],
    "northeast": ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"],
    "southeast": ["VA", "NC", "SC", "GA", "FL", "TN", "AL", "MS", "KY", "WV"],
    "south": ["TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL", "TN", "SC", "NC"],
    "southwest": ["AZ", "NM", "TX", "NV"],
    "west coast": ["CA", "OR", "WA"],
    "pacific northwest": ["WA", "OR", "ID"],
    "california": ["CA"],
    "southern california": ["CA"],
    "midwest": ["OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "great lakes": ["OH", "IN", "IL", "MI", "WI", "MN"],
    "mountain west": ["MT", "ID", "WY", "NV", "UT", "CO"],
    "rocky mountains": ["MT", "ID", "WY", "UT", "CO"],
    "new york area": ["NY", "NJ", "CT"],
    "bay area": ["CA"],
}


def resolve_region(name: Optional[str]) -> Optional[list[str]]:
    """Return the state codes for a region label (case/spacing tolerant), or None."""
    if not name:
        return None
    key = " ".join(name.lower().replace("-", " ").split())
    # exact match (normalizing hyphens to spaces)
    for region, states in REGIONS.items():
        if " ".join(region.replace("-", " ").split()) == key:
            return states
    # contains match ("airports in the new england region")
    for region, states in REGIONS.items():
        rk = " ".join(region.replace("-", " ").split())
        if rk in key or key in rk:
            return states
    return None
