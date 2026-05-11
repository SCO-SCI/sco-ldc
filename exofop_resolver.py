"""
ExoFOP-TESS resolver - queries the Exoplanet Follow-up Observing
Program (ExoFOP-TESS) TOI table for stellar parameters by TOI
identifier.

Phase 2: TOI-only support. Pure-TIC lookups are not implemented
because the natural ExoFOP endpoint for that case would require
fetching the entire TOI table (hundreds of KB) and filtering
locally, which isn't worth the complexity for our use case.
Confirmed planet hosts go through NEA; TOI candidates go through
this resolver.

Accepted input formats (all case-insensitive, processed by
parse_toi_identifier):
    TOI-1234.01
    TOI 1234.01
    TOI1234.01

ExoFOP-TESS is maintained by NExScI at Caltech/IPAC.

Endpoint documentation:
    https://exofop.ipac.caltech.edu/tess/Introduction_to_ExoFOP_php_functions.php
"""

from __future__ import annotations

import re
from typing import Optional

import httpx


# ExoFOP-TESS download_toi.php endpoint. Accepts a `toi` integer
# parameter that returns all planet candidates for that host TOI
# number (e.g. ?toi=1234 returns 1234.01, 1234.02, ...).
# Output format is pipe-delimited with a header row.
EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"

# Citation string returned to the frontend when ExoFOP supplies
# the stellar parameters.
EXOFOP_CITATION = "ExoFOP-TESS, NExScI/Caltech-IPAC"

# Hard timeout (seconds) for the HTTPS call. ExoFOP is generally
# responsive but is a busier endpoint than NEA TAP.
EXOFOP_TIMEOUT_SECONDS = 15.0

# Regex matching the three accepted TOI input formats:
#     TOI-1234.01    TOI 1234.01    TOI1234.01
# The separator between "TOI" and the number is one of: '-', ' ', or nothing.
# The number is "host.candidate" with both parts required.
_TOI_PATTERN = re.compile(
    r"^\s*TOI[-\s]?(\d+)\.(\d+)\s*$",
    re.IGNORECASE,
)

# Column headers in the pipe-delimited response that we care about.
# These are the exact strings as they appear in the ExoFOP TOI table
# header row.
_COL_TOI = "TOI"
_COL_TEFF = "Stellar Eff Temp (K)"
_COL_LOGG = "Stellar log(g) (cm/s^2)"
_COL_FEH = "Stellar Metallicity"


def parse_toi_identifier(text: str) -> Optional[tuple[int, str]]:
    """
    Parse a user-supplied TOI identifier into (host_toi_int, canonical_toi_string).

    Returns a tuple like (1234, "1234.01") on a match, or None if the
    input doesn't look like a TOI identifier.

    The returned host_toi_int is what we pass to ExoFOP's ?toi= query
    parameter. The canonical_toi_string is what we match against the
    TOI column in the returned table to pick the right row.
    """
    if not isinstance(text, str):
        return None
    match = _TOI_PATTERN.match(text)
    if not match:
        return None
    host = int(match.group(1))
    candidate = match.group(2)
    canonical = f"{host}.{candidate}"
    return host, canonical


def query_exofop(toi_input: str) -> dict:
    """
    Query ExoFOP-TESS for stellar parameters by TOI identifier.

    The toi_input argument is the user's raw string (e.g. "TOI-1234.01").
    The function parses it, fetches the host TOI's row group from
    ExoFOP, locates the requested candidate row, and extracts the
    three stellar parameters.

    Returns a dict in one of three shapes, matching the contract used
    by nea_resolver.query_nea:

    Found:
        {
            "found": True,
            "planet": "TOI-1234.01",
            "hostname": "TOI-1234",
            "teff": <float or None>,
            "logg": <float or None>,
            "feh":  <float or None>,
            "source": "ExoFOP",
            "citation": "ExoFOP-TESS, NExScI/Caltech-IPAC",
        }

    Not found:
        {
            "found": False,
            "planet": <original input>,
            "reason": "not_in_exofop",
        }

    Error (network failure, malformed response, etc):
        {
            "found": False,
            "planet": <original input>,
            "reason": "error",
            "error": "<short human-readable description>",
        }

    The function never raises.
    """
    parsed = parse_toi_identifier(toi_input)
    if parsed is None:
        # Caller is expected to check looks_like_toi() before calling
        # us, but we defend anyway.
        return _error_response(toi_input, "Input is not a recognized TOI identifier")

    host_toi, canonical_toi = parsed

    params = {
        "toi": str(host_toi),
        "output": "pipe",
    }

    try:
        response = httpx.get(
            EXOFOP_TOI_URL,
            params=params,
            timeout=EXOFOP_TIMEOUT_SECONDS,
            # ExoFOP occasionally redirects; let httpx follow.
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return _error_response(toi_input, "ExoFOP query timed out")
    except httpx.HTTPStatusError as exc:
        return _error_response(
            toi_input,
            f"ExoFOP returned HTTP {exc.response.status_code}",
        )
    except httpx.RequestError as exc:
        return _error_response(toi_input, f"ExoFOP request failed: {exc}")

    text = response.text
    if not text.strip():
        # Empty body means ExoFOP found nothing for this TOI.
        return _not_found_response(toi_input)

    rows = _parse_pipe_table(text)
    if rows is None:
        return _error_response(toi_input, "ExoFOP returned unparseable response")

    if not rows:
        return _not_found_response(toi_input)

    # Find the row matching our canonical TOI. The TOI column comes
    # back as a string like "1234.01"; do an exact string match.
    matching_row = None
    for row in rows:
        toi_value = row.get(_COL_TOI, "").strip()
        if toi_value == canonical_toi:
            matching_row = row
            break

    if matching_row is None:
        # The host TOI exists but not this specific candidate.
        return _not_found_response(toi_input)

    return {
        "found": True,
        "planet": f"TOI-{canonical_toi}",
        "hostname": f"TOI-{host_toi}",
        "teff": _coerce_float(matching_row.get(_COL_TEFF)),
        "logg": _coerce_float(matching_row.get(_COL_LOGG)),
        "feh":  _coerce_float(matching_row.get(_COL_FEH)),
        "source": "ExoFOP",
        "citation": EXOFOP_CITATION,
    }


def looks_like_toi(text: str) -> bool:
    """
    Quick check: does the input look like a TOI identifier?

    Used by the resolver router (in app.py) to decide whether to
    bother calling ExoFOP at all after NEA returns "not found".
    A planet name like "WASP-23 b" returns False, so we skip the
    ExoFOP call and go straight to the final "not found" response.
    """
    return parse_toi_identifier(text) is not None


# --- Internal helpers ------------------------------------------------------


def _parse_pipe_table(text: str) -> Optional[list[dict]]:
    """
    Parse ExoFOP's pipe-delimited output into a list of dicts.

    The format is:
        Header1|Header2|Header3|...
        value1|value2|value3|...
        value1|value2|value3|...
        ...

    Returns a list of {header: value} dicts, or None if the response
    doesn't look like a pipe-delimited table.
    """
    lines = text.splitlines()
    if len(lines) < 1:
        return None

    header_line = lines[0]
    if "|" not in header_line:
        # Not a pipe-delimited table.
        return None

    headers = [h.strip() for h in header_line.split("|")]

    rows: list[dict] = []
    for raw_line in lines[1:]:
        if not raw_line.strip():
            continue
        values = raw_line.split("|")
        # Pad or truncate to header length so zip-up is well-defined.
        if len(values) < len(headers):
            values = values + [""] * (len(headers) - len(values))
        elif len(values) > len(headers):
            values = values[:len(headers)]
        row = {headers[i]: values[i].strip() for i in range(len(headers))}
        rows.append(row)
    return rows


def _coerce_float(value) -> Optional[float]:
    """
    Convert an ExoFOP table value (string) to a float, or None if
    it's empty, missing, or unparseable. ExoFOP returns numeric
    fields as plain numeric strings; blank cells appear as empty
    strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _not_found_response(toi_input: str) -> dict:
    return {
        "found": False,
        "planet": toi_input,
        "reason": "not_in_exofop",
    }


def _error_response(toi_input: str, error_message: str) -> dict:
    return {
        "found": False,
        "planet": toi_input,
        "reason": "error",
        "error": error_message,
    }
