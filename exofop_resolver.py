from __future__ import annotations

import re
from typing import Optional

import httpx



EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"


EXOFOP_CITATION = "ExoFOP-TESS, NExScI/Caltech-IPAC"


EXOFOP_TIMEOUT_SECONDS = 15.0


_TOI_PATTERN = re.compile(
    r"^\s*TOI[-\s]?(\d+)\.(\d+)\s*$",
    re.IGNORECASE,
)


_COL_TOI = "TOI"
_COL_TEFF = "Stellar Eff Temp (K)"
_COL_LOGG = "Stellar log(g) (cm/s^2)"
_COL_FEH = "Stellar Metallicity"


def parse_toi_identifier(text: str) -> Optional[tuple[int, str]]:
    
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
    
    parsed = parse_toi_identifier(toi_input)
    if parsed is None:
        
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
        
        return _not_found_response(toi_input)

    rows = _parse_pipe_table(text)
    if rows is None:
        return _error_response(toi_input, "ExoFOP returned unparseable response")

    if not rows:
        return _not_found_response(toi_input)

    
    matching_row = None
    for row in rows:
        toi_value = row.get(_COL_TOI, "").strip()
        if toi_value == canonical_toi:
            matching_row = row
            break

    if matching_row is None:
        
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
    
    return parse_toi_identifier(text) is not None





def _parse_pipe_table(text: str) -> Optional[list[dict]]:
    
    lines = text.splitlines()
    if len(lines) < 1:
        return None

    header_line = lines[0]
    if "|" not in header_line:
        
        return None

    headers = [h.strip() for h in header_line.split("|")]

    rows: list[dict] = []
    for raw_line in lines[1:]:
        if not raw_line.strip():
            continue
        values = raw_line.split("|")
        
        if len(values) < len(headers):
            values = values + [""] * (len(headers) - len(values))
        elif len(values) > len(headers):
            values = values[:len(headers)]
        row = {headers[i]: values[i].strip() for i in range(len(headers))}
        rows.append(row)
    return rows


def _coerce_float(value) -> Optional[float]:
    
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
