from __future__ import annotations

import difflib
import os
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx



NEA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


NEA_CITATION = "DOI: 10.26133/NEA13"


NEA_TIMEOUT_SECONDS = 10.0


NEA_NAMELIST_TIMEOUT_SECONDS = 60.0


SUGGESTION_CUTOFF = 0.7
SUGGESTION_LIMIT = 3





def query_nea(planet_name: str) -> dict:
    
    adql = (
        "select pl_name, hostname, st_teff, st_logg, st_met "
        "from pscomppars "
        f"where pl_name = '{planet_name}'"
    )

    params = {"query": adql, "format": "json"}

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




_namelist_lock = threading.Lock()
_namelist: list[str] = []
_lower_to_canonical: dict[str, str] = {}
_namelist_loaded_utc_date: Optional[str] = None  # ISO YYYY-MM-DD or None


def load_namelist_at_startup(fallback_path: Optional[str] = None) -> dict:
    
    today = _utc_today_iso()

    
    try:
        names = _fetch_namelist_from_nea()
        if names:
            _set_namelist(names, load_date=today)
            return {"source": "nea", "count": len(names)}
    except Exception:
        pass

    
    if fallback_path and os.path.exists(fallback_path):
        try:
            names = _load_namelist_from_file(fallback_path)
            if names:
                _set_namelist(names, load_date=None)
                return {"source": "fallback", "count": len(names)}
        except Exception:
            pass

    return {"source": "empty", "count": 0}


def maybe_refresh_namelist() -> None:
    
    today = _utc_today_iso()

    if _namelist_loaded_utc_date == today:
        return

    with _namelist_lock:
        if _namelist_loaded_utc_date == today:
            return

        try:
            names = _fetch_namelist_from_nea()
            if names:
                _set_namelist_locked(names, load_date=today)
        except Exception:
            pass


def canonicalize_name(name: str) -> Optional[str]:
    
    if not _lower_to_canonical:
        return None
    return _lower_to_canonical.get(name.lower())


def get_suggestions(query: str) -> list[str]:
    
    if not _lower_to_canonical:
        return []

    query_lower = query.lower()

    
    with _namelist_lock:
        lower_keys = list(_lower_to_canonical.keys())
        lower_to_canon = dict(_lower_to_canonical)

    lower_matches = difflib.get_close_matches(
        query_lower,
        lower_keys,
        n=SUGGESTION_LIMIT,
        cutoff=SUGGESTION_CUTOFF,
    )

    return [lower_to_canon[m] for m in lower_matches if m in lower_to_canon]


def namelist_status() -> dict:
    
    return {
        "count": len(_namelist),
        "loaded_utc_date": _namelist_loaded_utc_date,
    }





def _set_namelist(names: list[str], load_date: Optional[str]) -> None:
    
    with _namelist_lock:
        _set_namelist_locked(names, load_date)


def _set_namelist_locked(names: list[str], load_date: Optional[str]) -> None:
    
    global _namelist, _lower_to_canonical, _namelist_loaded_utc_date
    _namelist = list(names)
    new_dict: dict[str, str] = {}
    for n in names:
        lower = n.lower()
        if lower not in new_dict:
            new_dict[lower] = n
    _lower_to_canonical = new_dict
    _namelist_loaded_utc_date = load_date


def _fetch_namelist_from_nea() -> list[str]:
   
    adql = "select pl_name from pscomppars"
    params = {"query": adql, "format": "json"}

    response = httpx.get(
        NEA_TAP_URL,
        params=params,
        timeout=NEA_NAMELIST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json()

    if not isinstance(rows, list):
        raise ValueError("Unexpected JSON shape from NEA")

    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("pl_name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())

    return sorted(names)


def _load_namelist_from_file(path: str) -> list[str]:
   
    names: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            names.add(stripped)
    return sorted(names)


def _utc_today_iso() -> str:
    
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _coerce_float(value) -> Optional[float]:
    
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
