from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import ldc_core
import nea_resolver
import exofop_resolver

logger = logging.getLogger("scoldc")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
NEA_FALLBACK_PATH = os.path.join(DATA_DIR, "nea_parameters_fallback.tsv")
EXOFOP_FALLBACK_PATH = os.path.join(DATA_DIR, "exofop_toi_fallback.tsv")


REFRESH_HOUR_UTC = 17
REFRESH_MINUTE_UTC = 0


REFRESH_MAX_ATTEMPTS = 4
REFRESH_RETRY_INTERVAL_SECONDS = 300


FRESHNESS_THRESHOLD_HOURS = 48


REFRESH_HISTORY_MAX = 30


WORKER_STARTUP_UTC = datetime.now(timezone.utc)

app = FastAPI(
    title="scoldc v3",
    description="Quadratic limb-darkening coefficients by trilinear interpolation "
                "of Claret tables.",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


LOAD_COUNTS = ldc_core.load_tables(DATA_DIR)


NEA_STATUS = nea_resolver.load_cache_at_startup(fallback_path=NEA_FALLBACK_PATH)
logger.info(
    "Loaded NEA cache: source=%s count=%d",
    NEA_STATUS["source"], NEA_STATUS["count"],
)


EXOFOP_STATUS = exofop_resolver.load_cache_at_startup(fallback_path=EXOFOP_FALLBACK_PATH)
logger.info(
    "Loaded ExoFOP cache: source=%s count=%d",
    EXOFOP_STATUS["source"], EXOFOP_STATUS["count"],
)



_refresh_history: list[dict] = []
_refresh_history_lock = threading.Lock()


def _append_refresh_history(entry: dict) -> None:
   
    with _refresh_history_lock:
        _refresh_history.append(entry)
        
        while len(_refresh_history) > REFRESH_HISTORY_MAX:
            _refresh_history.pop(0)


def _get_refresh_history() -> list[dict]:
   
    with _refresh_history_lock:
        return list(reversed(_refresh_history))


def _is_cache_fresh() -> bool:
   
    nea_status = nea_resolver.cache_status()
    exofop_status = exofop_resolver.cache_status()
    nea_ts = nea_status.get("refreshed_utc")
    exofop_ts = exofop_status.get("refreshed_utc")
    if not nea_ts or not exofop_ts:
        return False

    now = datetime.now(timezone.utc)
    threshold = timedelta(hours=FRESHNESS_THRESHOLD_HOURS)
    for ts_str in (nea_ts, exofop_ts):
        try:
            
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M UTC")
            ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        if now - ts > threshold:
            return False
    return True


def _next_refresh_time_utc(now: datetime) -> datetime:
   
    candidate = now.replace(
        hour=REFRESH_HOUR_UTC, minute=REFRESH_MINUTE_UTC,
        second=0, microsecond=0,
    )
    if candidate <= now + timedelta(hours=1):
        candidate += timedelta(days=1)
    return candidate


def _attempt_refresh_both_caches() -> tuple[bool, bool]:
    
    nea_ok = nea_resolver.refresh_cache_from_live()
    exofop_ok = exofop_resolver.refresh_cache_from_live()
    return nea_ok, exofop_ok


def _refresh_scheduler() -> None:
    
    while True:
        now = datetime.now(timezone.utc)
        target = _next_refresh_time_utc(now)
        wait_seconds = (target - now).total_seconds()
        logger.info(
            "Refresh scheduler sleeping until %s (%.0f minutes)",
            target.strftime("%Y-%m-%d %H:%M UTC"), wait_seconds / 60.0,
        )
        time.sleep(wait_seconds)

       
        sequence_start = datetime.now(timezone.utc)
        nea_done = False
        exofop_done = False
        attempts_made = 0
        for attempt in range(1, REFRESH_MAX_ATTEMPTS + 1):
            attempts_made = attempt
            attempt_time = datetime.now(timezone.utc)
            logger.info(
                "Refresh attempt %d/%d at %s (nea_done=%s exofop_done=%s)",
                attempt, REFRESH_MAX_ATTEMPTS,
                attempt_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                nea_done, exofop_done,
            )

            
            if not nea_done:
                nea_done = nea_resolver.refresh_cache_from_live()
            if not exofop_done:
                exofop_done = exofop_resolver.refresh_cache_from_live()

            if nea_done and exofop_done:
                logger.info("Refresh succeeded (both caches updated)")
                break

            if attempt < REFRESH_MAX_ATTEMPTS:
                time.sleep(REFRESH_RETRY_INTERVAL_SECONDS)
            else:
                logger.warning(
                    "Refresh exhausted retries: nea_ok=%s exofop_ok=%s. "
                    "Next attempt tomorrow.",
                    nea_done, exofop_done,
                )

       
        _append_refresh_history({
            "started_utc": sequence_start,
            "finished_utc": datetime.now(timezone.utc),
            "attempts": attempts_made,
            "nea_ok": nea_done,
            "exofop_ok": exofop_done,
        })



_refresh_thread = threading.Thread(
    target=_refresh_scheduler, name="refresh-scheduler", daemon=True,
)
_refresh_thread.start()
logger.info("Refresh scheduler thread started")


@app.api_route("/api/health", methods=["GET", "HEAD"])
def health() -> dict:
   
    return {
        "status": "ok",
        "version": app.version,
        "freshness": "ok" if _is_cache_fresh() else "stale",
        "tables": LOAD_COUNTS,
        "filter_count": len(ldc_core.get_available_filters()),
        "nea_cache": nea_resolver.cache_status(),
        "exofop_cache": exofop_resolver.cache_status(),
    }


@app.get("/api/filters")
def filters() -> dict:
    
    return {"filters": ldc_core.get_available_filters()}


@app.get("/api/compute")
def compute(
    teff: float = Query(..., description="Effective temperature in K"),
    logg: float = Query(..., description="Surface gravity log g in cgs dex"),
    feh:  float = Query(0.0, description="Metallicity [Fe/H] in dex (solar=0.0)"),
    filter: str = Query(..., alias="filter", description="Filter code (e.g. 'V', 'Kp', 'TESS', 'CBB')"),
    model:  str = Query("ATLAS", description="Stellar atmosphere model: ATLAS, PHOENIX, or PHOENIX-COND"),
) -> dict:
    
    try:
        result = ldc_core.compute_ldcs(teff, logg, feh, filter, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/api/resolve")
def resolve(
    planet: str = Query(..., description="Exoplanet name (e.g. 'WASP-23 b') or TOI identifier (e.g. 'TOI-1234.01', 'TOI-1234 b', or 'TOI-1234')"),
) -> dict:
    
    canonical = nea_resolver.canonicalize_name(planet)
    query_name = canonical if canonical is not None else planet

    
    nea_result = nea_resolver.query_nea(query_name)
    if nea_result.get("found") is True:
        return nea_result
    if nea_result.get("reason") == "error":
        return nea_result

   
    if exofop_resolver.looks_like_toi(planet):
        parsed = exofop_resolver.parse_toi_identifier(planet)
        if parsed is not None:
            host_int, component = parsed
            if component is not None:
                # ".NN" form: ExoFOP first.
                exofop_result = exofop_resolver.query_exofop(planet)
                if exofop_result.get("found") is True:
                    return exofop_result
                host_nea_result = nea_resolver.query_nea_by_host(f"TOI-{host_int}")
                if host_nea_result is not None:
                    return host_nea_result
            else:
                # Bare-host or letter form: NEA first.
                host_nea_result = nea_resolver.query_nea_by_host(f"TOI-{host_int}")
                if host_nea_result is not None:
                    return host_nea_result
                exofop_result = exofop_resolver.query_exofop(planet)
                if exofop_result.get("found") is True:
                    return exofop_result

  
    nea_result["suggestions"] = nea_resolver.get_suggestions(planet)
    return nea_result





@app.get("/admin/cache-status", response_class=HTMLResponse)
def admin_cache_status() -> HTMLResponse:
   
    now = datetime.now(timezone.utc)
    uptime = now - WORKER_STARTUP_UTC
    uptime_days = uptime.days
    uptime_hours = uptime.seconds // 3600
    uptime_minutes = (uptime.seconds % 3600) // 60

    nea_status = nea_resolver.cache_status()
    exofop_status = exofop_resolver.cache_status()
    history = _get_refresh_history()
    fresh = _is_cache_fresh()

    
    next_refresh = _next_refresh_time_utc(now)

    history_rows = []
    if not history:
        history_rows.append(
            '<tr><td colspan="5" class="text-muted">'
            'No refresh attempts since this worker started. The first '
            'scheduled refresh will run at the time shown above.'
            '</td></tr>'
        )
    else:
        for entry in history:
            both_ok = entry["nea_ok"] and entry["exofop_ok"]
            row_class = "" if both_ok else 'class="table-warning"'
            duration_s = (entry["finished_utc"] - entry["started_utc"]).total_seconds()
            history_rows.append(
                f'<tr {row_class}>'
                f'<td>{entry["started_utc"].strftime("%Y-%m-%d %H:%M UTC")}</td>'
                f'<td>{entry["attempts"]} / {REFRESH_MAX_ATTEMPTS}</td>'
                f'<td>{"OK" if entry["nea_ok"] else "FAILED"}</td>'
                f'<td>{"OK" if entry["exofop_ok"] else "FAILED"}</td>'
                f'<td>{duration_s:.0f}s</td>'
                f'</tr>'
            )

    freshness_class = "text-success" if fresh else "text-danger"
    freshness_label = "OK" if fresh else "STALE"

    nea_ts = nea_status.get("refreshed_utc") or "(never)"
    exofop_ts = exofop_status.get("refreshed_utc") or "(never)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sco-ldc cache status</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {{ background: #fff; font-size: 0.875rem; }}
  h1 {{ font-size: 1.35rem; }}
  h2 {{ font-size: 1rem; font-weight: 600; margin-top: 1.25rem; }}
  .label {{ color: #4c555d; font-weight: 500; }}
  .value {{ font-family: 'Courier New', monospace; }}
  table {{ font-size: 0.825rem; }}
  table.detail td {{ padding: 0.2rem 0.5rem; }}
  table.detail td:first-child {{ color: #4c555d; width: 14rem; }}
  hr {{ margin: 0.75rem 0; }}
</style>
</head>
<body>
<div class="container py-3" style="max-width: 820px;">
  <h1>sco-ldc cache status</h1>
  <p class="text-muted small mb-0">
    Internal diagnostic page. Snapshot at page-load time;
    refresh the browser for updated values.
  </p>
  <hr>

  <h2>Overall</h2>
  <table class="detail">
    <tbody>
      <tr><td>Freshness</td>
          <td class="value"><span class="{freshness_class} fw-bold">{freshness_label}</span>
          (threshold: {FRESHNESS_THRESHOLD_HOURS} h)</td></tr>
      <tr><td>Current UTC time</td>
          <td class="value">{now.strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>
      <tr><td>Worker started</td>
          <td class="value">{WORKER_STARTUP_UTC.strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>
      <tr><td>Worker uptime</td>
          <td class="value">{uptime_days}d {uptime_hours}h {uptime_minutes}m</td></tr>
      <tr><td>Next scheduled refresh</td>
          <td class="value">{next_refresh.strftime("%Y-%m-%d %H:%M UTC")}</td></tr>
    </tbody>
  </table>

  <h2>NEA cache</h2>
  <table class="detail">
    <tbody>
      <tr><td>Row count</td>          <td class="value">{nea_status.get("count", 0)}</td></tr>
      <tr><td>Last refresh (UTC)</td> <td class="value">{nea_ts}</td></tr>
    </tbody>
  </table>

  <h2>ExoFOP cache</h2>
  <table class="detail">
    <tbody>
      <tr><td>Row count</td>          <td class="value">{exofop_status.get("count", 0)}</td></tr>
      <tr><td>Last refresh (UTC)</td> <td class="value">{exofop_ts}</td></tr>
    </tbody>
  </table>

  <h2>Recent refresh history</h2>
  <p class="text-muted small mb-1">
    Last {REFRESH_HISTORY_MAX} daily sequences since worker startup, newest first.
    Failed sequences are highlighted.
  </p>
  <table class="table table-sm table-bordered">
    <thead><tr>
      <th>Started</th>
      <th>Attempts</th>
      <th>NEA</th>
      <th>ExoFOP</th>
      <th>Duration</th>
    </tr></thead>
    <tbody>
      {''.join(history_rows)}
    </tbody>
  </table>
</div>
</body>
</html>
"""
    return HTMLResponse(content=html)




if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
   
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=500,
            content={"error": "static/index.html is missing"},
        )
    return FileResponse(index_path, media_type="text/html")
