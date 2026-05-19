from __future__ import annotations

import difflib
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx



NEA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


NEA_CITATION = "DOI: 10.26133/NEA13"


def _full_citation() -> str:

    if _last_successful_live_refresh_utc is None:
        return NEA_CITATION
    ts = _last_successful_live_refresh_utc.strftime("%Y-%m-%d %H:%M UTC")
    return f"{NEA_CITATION}, {ts}"


NEA_BULK_TIMEOUT_SECONDS = 60.0


NEA_LIVE_TIMEOUT_SECONDS = 10.0


SUGGESTION_CUTOFF = 0.7
SUGGESTION_LIMIT = 3


NEA_COLUMNS = ["pl_name", "hostname", "st_teff", "st_logg", "st_met"]




_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
_lower_keys: list[str] = []
_hostname_index: dict[str, list[dict]] = {}
_last_successful_live_refresh_utc: Optional[datetime] = None





def query_nea(planet_name: str) -> dict:
    
    if not planet_name:
        return {"found": False, "planet": planet_name, "reason": "not_in_nea"}

    
    row = _cache.get(planet_name.lower())
    if row is not None:
        return _row_to_response(row)

    
    if _cache:
        return {"found": False, "planet": planet_name, "reason": "not_in_nea"}

    
    return _live_single_lookup(planet_name)


def query_nea_by_host(hostname: str) -> Optional[dict]:
    
    if not hostname:
        return None
    rows = _hostname_index.get(hostname.lower())
    if not rows:
        return None
    
    return _row_to_response(rows[0])


def _row_to_response(row: dict) -> dict:
    
    return {
        "found": True,
        "planet": row.get("pl_name"),
        "hostname": row.get("hostname"),
        "teff": _to_float_or_none(row.get("st_teff")),
        "logg": _to_float_or_none(row.get("st_logg")),
        "feh":  _to_float_or_none(row.get("st_met")),
        "source": "NEA",
        "citation": _full_citation(),
    }


def _live_single_lookup(planet_name: str) -> dict:
    
    adql = (
        f"select {', '.join(NEA_COLUMNS)} from pscomppars "
        f"where pl_name = '{planet_name}'"
    )
    params = {"query": adql, "format": "json"}

    try:
        response = httpx.get(
            NEA_TAP_URL, params=params, timeout=NEA_LIVE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rows = response.json()
    except httpx.TimeoutException:
        return _error_response(planet_name, "NEA query timed out")
    except httpx.HTTPStatusError as exc:
        return _error_response(planet_name, f"NEA returned HTTP {exc.response.status_code}")
    except (httpx.RequestError, ValueError) as exc:
        return _error_response(planet_name, f"NEA request failed: {exc}")

    if not isinstance(rows, list) or len(rows) == 0:
        return {"found": False, "planet": planet_name, "reason": "not_in_nea"}

    return _row_to_response(rows[0])





def canonicalize_name(name: str) -> Optional[str]:
    
    if not name or not _cache:
        return None
    row = _cache.get(name.lower())
    return row.get("pl_name") if row else None


def get_suggestions(query: str) -> list[str]:
   
    if not _cache or not query:
        return []
    query_lower = query.lower()

   
    with _cache_lock:
        keys_snapshot = list(_lower_keys)
        cache_snapshot = dict(_cache)

    lower_matches = difflib.get_close_matches(
        query_lower, keys_snapshot, n=SUGGESTION_LIMIT, cutoff=SUGGESTION_CUTOFF,
    )
    return [cache_snapshot[k]["pl_name"] for k in lower_matches if k in cache_snapshot]





def load_cache_at_startup(fallback_path: Optional[str] = None) -> dict:
    
    try:
        rows = _fetch_full_table_from_nea()
        if rows:
            _set_cache(rows, refresh_time=datetime.now(timezone.utc))
            return {"source": "nea", "count": len(rows)}
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
        rows = _fetch_full_table_from_nea()
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





def _fetch_full_table_from_nea() -> list[dict]:
    
    adql = f"select {', '.join(NEA_COLUMNS)} from pscomppars"
    params = {"query": adql, "format": "json"}

    response = httpx.get(
        NEA_TAP_URL, params=params, timeout=NEA_BULK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json()

    if not isinstance(rows, list):
        raise ValueError("Unexpected JSON shape from NEA")

    
    cleaned: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("pl_name")
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned.append({
            "pl_name": name.strip(),
            "hostname": row.get("hostname"),
            "st_teff": row.get("st_teff"),
            "st_logg": row.get("st_logg"),
            "st_met":  row.get("st_met"),
        })
    return cleaned


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
            name = parts[0].strip()
            if not name:
                continue
            rows.append({
                "pl_name": name,
                "hostname": parts[1].strip() or None,
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
   
    global _cache, _lower_keys, _hostname_index, _last_successful_live_refresh_utc
    new_cache: dict[str, dict] = {}
    new_hostname_index: dict[str, list[dict]] = {}
    for row in rows:
        name = row.get("pl_name")
        if not name:
            continue
        key = name.lower()
        if key in new_cache:
            continue
        new_cache[key] = row
       
        hostname = row.get("hostname")
        if hostname:
            new_hostname_index.setdefault(hostname.lower(), []).append(row)
  
    for hostname_key, host_rows in new_hostname_index.items():
        host_rows.sort(key=lambda r: r.get("pl_name", "").lower())

    _cache = new_cache
    _lower_keys = list(new_cache.keys())
    _hostname_index = new_hostname_index
    if refresh_time is not None:
        _last_successful_live_refresh_utc = refresh_time


def _to_float_or_none(value):
    
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
