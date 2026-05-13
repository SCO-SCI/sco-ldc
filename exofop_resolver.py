from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx



EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"


EXOFOP_CITATION = "ExoFOP-TESS, NExScI/Caltech-IPAC"


EXOFOP_BULK_TIMEOUT_SECONDS = 60.0


EXOFOP_LIVE_TIMEOUT_SECONDS = 15.0


_TOI_PATTERN = re.compile(
    r"^\s*TOI[-\s]?(\d+)\.(\d+)\s*$",
    re.IGNORECASE,
)


_COL_TOI = "TOI"
_COL_TIC = "TIC ID"
_COL_TEFF = "Stellar Eff Temp (K)"
_COL_LOGG = "Stellar log(g) (cm/s^2)"
_COL_FEH = "Stellar Metallicity"




_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
_loaded_utc_date: Optional[str] = None  # ISO YYYY-MM-DD or None





def parse_toi_identifier(text: str) -> Optional[tuple[int, str]]:
    
    if not isinstance(text, str):
        return None
    match = _TOI_PATTERN.match(text)
    if not match:
        return None
    host = int(match.group(1))
    canonical = f"{host}.{match.group(2)}"
    return host, canonical


def looks_like_toi(text: str) -> bool:
    
    return parse_toi_identifier(text) is not None


def query_exofop(toi_input: str) -> dict:
    
    parsed = parse_toi_identifier(toi_input)
    if parsed is None:
        return _error_response(toi_input, "Input is not a recognized TOI identifier")
    host_toi, canonical_toi = parsed

    
    row = _cache.get(canonical_toi)
    if row is not None:
        return _row_to_response(toi_input, host_toi, canonical_toi, row)

    
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
        "citation": EXOFOP_CITATION,
    }


def _live_single_lookup(toi_input: str, host_toi: int, canonical_toi: str) -> dict:
    
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

    for row in rows:
        if row.get(_COL_TOI, "").strip() == canonical_toi:
            return _row_to_response(toi_input, host_toi, canonical_toi, {
                "st_teff": row.get(_COL_TEFF),
                "st_logg": row.get(_COL_LOGG),
                "st_met":  row.get(_COL_FEH),
            })

    return {"found": False, "planet": toi_input, "reason": "not_in_exofop"}





def load_cache_at_startup(fallback_path: Optional[str] = None) -> dict:
    
    today = _utc_today_iso()

    try:
        rows = _fetch_full_table_from_exofop()
        if rows:
            _set_cache(rows, load_date=today)
            return {"source": "exofop", "count": len(rows)}
    except Exception:
        pass

    if fallback_path and os.path.exists(fallback_path):
        try:
            rows = _load_table_from_file(fallback_path)
            if rows:
                _set_cache(rows, load_date=None)
                return {"source": "fallback", "count": len(rows)}
        except Exception:
            pass

    return {"source": "empty", "count": 0}


def maybe_refresh_cache() -> None:
    
    today = _utc_today_iso()
    if _loaded_utc_date == today:
        return

    with _cache_lock:
        if _loaded_utc_date == today:
            return
        try:
            rows = _fetch_full_table_from_exofop()
            if rows:
                _set_cache_locked(rows, load_date=today)
        except Exception:
            pass


def cache_status() -> dict:
    
    return {
        "count": len(_cache),
        "loaded_utc_date": _loaded_utc_date,
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
    """TSV cell -> float, or None if empty/unparseable."""
    s = cell.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _set_cache(rows: list[dict], load_date: Optional[str]) -> None:
    
    with _cache_lock:
        _set_cache_locked(rows, load_date)


def _set_cache_locked(rows: list[dict], load_date: Optional[str]) -> None:
   
    global _cache, _loaded_utc_date
    new_cache: dict[str, dict] = {}
    for row in rows:
        toi = row.get("toi")
        if not toi:
            continue
        if toi not in new_cache:
            new_cache[toi] = row
    _cache = new_cache
    _loaded_utc_date = load_date


def _utc_today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
