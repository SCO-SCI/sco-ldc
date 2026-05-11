"""
NEA resolver - queries the NASA Exoplanet Archive Planetary Systems
Composite Parameters table (PSCompPars, DOI: 10.26133/NEA13) for
stellar parameters by planet name.

Phase 1: bare NEA lookup. No ExoFOP fallback, no fuzzy match, no
name normalization. Those come in later phases.

The NEA TAP (Table Access Protocol) endpoint accepts ADQL-like SQL
over HTTPS GET and returns JSON. Documentation:
  https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
"""

from __future__ import annotations

from typing import Optional

import httpx


# Public NEA TAP sync endpoint. No auth required.
NEA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Citation string we return alongside successful lookups so the
# frontend can display attribution to the user.
NEA_CITATION = "DOI: 10.26133/NEA13"

# Hard timeout (seconds) for the HTTPS call. NEA is usually fast
# (well under a second) but we don't want a hung request to tie up
# a worker for minutes. 10s is generous; tune later if needed.
NEA_TIMEOUT_SECONDS = 10.0


def query_nea(planet_name: str) -> dict:
    """
    Query NEA PSCompPars for a single planet by exact name match.

    Returns a dict in one of three shapes:

    Found:
        {
            "found": True,
            "planet": "WASP-23 b",
            "hostname": "WASP-23",
            "teff": 5150.0,
            "logg": 4.4,
            "feh": -0.05,
            "source": "NEA",
            "citation": "DOI: 10.26133/NEA13",
        }
        Any of teff / logg / feh may be None if the NEA value is null.

    Not found (no matching row in PSCompPars):
        {
            "found": False,
            "planet": "WASP-23 b",
            "reason": "not_in_nea",
        }

    Error (network failure, malformed response, NEA returned an error):
        {
            "found": False,
            "planet": "WASP-23 b",
            "reason": "error",
            "error": "<short human-readable description>",
        }

    The function never raises; all failure modes return a dict so the
    caller can render the response without exception handling.
    """
    # ADQL query selecting just the fields we need from PSCompPars,
    # filtered by exact pl_name match.
    #
    # PSCompPars has one row per confirmed planet, so an exact name
    # match returns at most one row. We use parameterized-style
    # quoting on the planet name to avoid breaking the URL if the
    # name contains spaces or special characters - httpx handles the
    # URL-encoding.
    #
    # Columns:
    #   pl_name   - planet name (e.g. "WASP-23 b")
    #   hostname  - host star name (e.g. "WASP-23")
    #   st_teff   - effective temperature in K
    #   st_logg   - surface gravity log g in cgs dex
    #   st_met    - metallicity [Fe/H] in dex
    adql = (
        "select pl_name, hostname, st_teff, st_logg, st_met "
        "from pscomppars "
        f"where pl_name = '{planet_name}'"
    )

    params = {
        "query": adql,
        "format": "json",
    }

    try:
        response = httpx.get(
            NEA_TAP_URL,
            params=params,
            timeout=NEA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return _error_response(planet_name, "NEA query timed out")
    except httpx.HTTPStatusError as exc:
        return _error_response(
            planet_name,
            f"NEA returned HTTP {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        return _error_response(planet_name, f"NEA request failed: {exc}")

    # NEA returns JSON as a list of dicts, one per row. Empty list
    # means no match.
    try:
        rows = response.json()
    except ValueError:
        return _error_response(planet_name, "NEA returned non-JSON response")

    if not isinstance(rows, list):
        return _error_response(planet_name, "NEA returned unexpected JSON shape")

    if len(rows) == 0:
        return {
            "found": False,
            "planet": planet_name,
            "reason": "not_in_nea",
        }

    # PSCompPars has one row per planet, so we take the first (and
    # only expected) row. If NEA ever returns more than one, we still
    # use the first - this would be a data anomaly worth flagging,
    # but it's not our problem to solve here.
    row = rows[0]

    return {
        "found": True,
        "planet": row.get("pl_name", planet_name),
        "hostname": row.get("hostname"),
        "teff": _coerce_float(row.get("st_teff")),
        "logg": _coerce_float(row.get("st_logg")),
        "feh":  _coerce_float(row.get("st_met")),
        "source": "NEA",
        "citation": NEA_CITATION,
    }


def _coerce_float(value) -> Optional[float]:
    """
    Convert an NEA-returned value to a float, or None if it's null
    or unparseable. NEA's JSON typically returns numbers as numbers
    and missing values as null (JSON null -> Python None), but we
    defend against string-numeric values too.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_response(planet_name: str, error_message: str) -> dict:
    return {
        "found": False,
        "planet": planet_name,
        "reason": "error",
        "error": error_message,
    }
