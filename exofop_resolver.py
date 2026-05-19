from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx



EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"


EXOFOP_CITATION = "NExScI/Caltech-IPAC"


def _full_citation() -> str:
    
    if _last_successful_live_refresh_utc is None:
        return EXOFOP_CITATION
    ts = _last_successful_live_refresh_utc.strftime("%Y-%m-%d %H:%M UTC")
    return f"{EXOFOP_CITATION}, {ts}"


EXOFOP_BULK_TIMEOUT_SECONDS = 60.0


EXOFOP_LIVE_TIMEOUT_SECONDS = 15.0


_TOI_PATTERN = re.compile(
    r"^\s*TOI[-\s]?(\d+)(?:(?:\.(\d+))|(?:\s+([a-z])))?\s*$",
    re.IGNORECASE,
)


_COL_TOI = "TOI"
_COL_TIC = "TIC ID"
_COL_TEFF = "Stellar Eff Temp (K)"
_COL_LOGG = "Stellar log(g) (cm/s^2)"
_COL_FEH = "Stellar Metallicity"




_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
_host_index: dict[int, list[str]] = {}
_last_successful_live_refresh_utc: Optional[datetime] = None





def parse_toi_identifier(text: str) -> Optional[tuple[int, Optional[str]]]:
    
    if not isinstance(text, str):
        return None
    match = _TOI_PATTERN.match(text)
    if not match:
        return None
    host = int(match.group(1))
    component_digits = match.group(2)
    if component_digits:
        canonical = f"{host}.{component_digits}"
        return host, canonical
    return host, None


def looks_like_toi(text: str) -> bool:
    
    return parse_toi_identifier(text) is not None


def query_exofop(toi_input: str) -> dict:
    
    parsed = parse_toi_identifier(toi_input)
    if parsed is None:
        return _error_response(toi_input, "Input is not a recognized TOI identifier")
    host_toi, canonical_toi = parsed

  
    if canonical_toi is not None:
        row = _cache.get(canonical_toi)
        if row is not None:
            return _row_to_response(toi_input, host_toi, canonical_toi, row)

    
    components = _host_index.get(host_toi)
    if components:
        first_canonical = components[0]
        row = _cache.get(first_canonical)
        if row is not None:
            return _row_to_response(toi_input, host_toi, first_canonical, row)

    
    if _cache:
        return {"found": False, "planet": toi_input, "reason": "not_in_exofop"}

    
    return _live_single_lookup(toi_input, host_toi, canonical_toi)


def _row_to_response(toi_input: str, host_toi: int, canonical_toi: str, row: dict) -> dict:
    
    return {
        "found": True,
        "planet": f"TOI-{canonical_toi}",
        "hostname": f"TOI-{host_toi}",
        "teff": _to_float_or_none(row.get("st_teff")),
        "logg": _to_float_or_none(row.get("st_logg")),
        "feh":  _to_float_or_none(row.get("st_met")),
        "source": "ExoFOP",
        "citation": _full_citation(),
    }


def _live_single_lookup(toi_input: str, host_toi: int, canonical_toi: Optional[str]) -> dict:
    
    params = {"toi": str(host_toi), "output": "pipe"}

    try:
        response = httpx.get(
            EXOFOP_TOI_URL,
            params=params,
            timeout=EXOFOP_LIVE_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return _error_response(toi_input, "ExoFOP query timed out")
    except httpx.HTTPStatusError as exc:
        return _error_response(toi_input, f"ExoFOP returned HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return _error_response(toi_input, f"ExoFOP request failed: {exc}")

    text = response.text
    if not text.strip():
        return {"found": False, "planet": toi_input, "reason": "not_in_exofop"}

    rows = _parse_pipe_table(text)
    if rows is None:
        return _error_response(toi_input, "ExoFOP returned unparseable response")

    
    if canonical_toi is not None:
        for row in rows:
            if row.get(_COL_TOI, "").strip() == canonical_toi:
                return _row_to_response(toi_input, host_toi, canonical_toi, {
                    "st_teff": row.get(_COL_TEFF),
                    "st_logg": row.get(_COL_LOGG),
                    "st_met":  row.get(_COL_FEH),
                })
        return {"found": False, "planet": toi_input, "reason": "not_in_exofop"}

   
    candidates = []
    for row in rows:
        toi_str = row.get(_COL_TOI, "").strip()
        if not toi_str:
            continue
        try:
            host_str, comp_str = toi_str.split(".", 1)
            comp_int = int(comp_str)
        except (ValueError, AttributeError):
            continue
        candidates.append((comp_int, toi_str, row))
    if not candidates:
        return {"found": False, "planet": toi_input, "reason": "not_in_exofop"}
    candidates.sort(key=lambda c: c[0])
    _, chosen_canonical, chosen_row = candidates[0]
    return _row_to_response(toi_input, host_toi, chosen_canonical, {
        "st_teff": chosen_row.get(_COL_TEFF),
        "st_logg": chosen_row.get(_COL_LOGG),
        "st_met":  chosen_row.get(_COL_FEH),
    })





def load_cache_at_startup(fallback_path: Optional[str] = None) -> dict:
    
    try:
        rows = _fetch_full_table_from_exofop()
        if rows:
            _set_cache(rows, refresh_time=datetime.now(timezone.utc))
            return {"source": "exofop", "count": len(rows)}
    except Exception:
        pass

    if fallback_path and os.path.exists(fallback_path):
        try:
            rows = _load_table_from_file(fallback_path)
            if rows:
                _set_cache(rows, refresh_time=None)
                return {"source": "fallback", "count": len(rows)}
        except Exception:
            pass

    return {"source": "empty", "count": 0}


def refresh_cache_from_live() -> bool:
    
    try:
        rows = _fetch_full_table_from_exofop()
    except Exception:
        return False
    if not rows:
        return False
    _set_cache(rows, refresh_time=datetime.now(timezone.utc))
    return True


def cache_status() -> dict:
    
    return {
        "count": len(_cache),
        "refreshed_utc": _last_successful_live_refresh_utc.strftime("%Y-%m-%d %H:%M UTC")
            if _last_successful_live_refresh_utc is not None else None,
    }





def _fetch_full_table_from_exofop() -> list[dict]:
    
    params = {"sort": "toi", "output": "pipe"}
    response = httpx.get(
        EXOFOP_TOI_URL,
        params=params,
        timeout=EXOFOP_BULK_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    parsed = _parse_pipe_table(response.text)
    if parsed is None:
        raise ValueError("Could not parse ExoFOP response")

    rows: list[dict] = []
    for row in parsed:
        toi = (row.get(_COL_TOI) or "").strip()
        if not toi:
            continue
        rows.append({
            "toi": toi,
            "tic": (row.get(_COL_TIC) or "").strip(),
            "st_teff": row.get(_COL_TEFF),
            "st_logg": row.get(_COL_LOGG),
            "st_met":  row.get(_COL_FEH),
        })
    return rows


def _parse_pipe_table(text: str) -> Optional[list[dict]]:
   
    lines = text.splitlines()
    if not lines:
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
        rows.append({headers[i]: values[i].strip() for i in range(len(headers))})
    return rows


def _load_table_from_file(path: str) -> list[dict]:
    
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            while len(parts) < 5:
                parts.append("")
            toi = parts[0].strip()
            if not toi:
                continue
            rows.append({
                "toi": toi,
                "tic": parts[1].strip(),
                "st_teff": _parse_cell(parts[2]),
                "st_logg": _parse_cell(parts[3]),
                "st_met":  _parse_cell(parts[4]),
            })
    return rows


def _parse_cell(cell: str):
    
    s = cell.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _set_cache(rows: list[dict], refresh_time: Optional[datetime]) -> None:
    
    with _cache_lock:
        _set_cache_locked(rows, refresh_time)


def _set_cache_locked(rows: list[dict], refresh_time: Optional[datetime]) -> None:
    
    global _cache, _host_index, _last_successful_live_refresh_utc
    new_cache: dict[str, dict] = {}
    new_host_index: dict[int, list[str]] = {}
    for row in rows:
        toi = row.get("toi")
        if not toi:
            continue
        if toi in new_cache:
            continue
        new_cache[toi] = row
        
        try:
            host_str, comp_str = toi.split(".", 1)
            host_int = int(host_str)
            new_host_index.setdefault(host_int, []).append(toi)
        except (ValueError, AttributeError):
            continue
   
    for host_int, components in new_host_index.items():
        components.sort(key=lambda s: _component_sort_key(s))

    _cache = new_cache
    _host_index = new_host_index
    if refresh_time is not None:
        _last_successful_live_refresh_utc = refresh_time


def _component_sort_key(canonical_toi: str) -> int:
    
    try:
        _, comp_str = canonical_toi.split(".", 1)
        return int(comp_str)
    except (ValueError, AttributeError):
        return 9999  


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_response(toi_input: str, error_message: str) -> dict:
    return {
        "found": False,
        "planet": toi_input,
        "reason": "error",
        "error": error_message,
    }
