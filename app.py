"""
scoldc v3 - FastAPI backend.

Endpoints:
  GET /api/health              - liveness + load summary
  GET /api/filters             - available filters with per-model grid ranges
  GET /api/compute             - compute (u1, u2) by trilinear interpolation
  GET /api/resolve             - resolve a planet name or TOI identifier to
                                 stellar parameters from NEA or ExoFOP-TESS
  GET /                        - serves static/index.html

Run locally:  uvicorn app:app --reload --port 8000
Render:       gunicorn -k uvicorn.workers.UvicornWorker app:app
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import ldc_core
import nea_resolver
import exofop_resolver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="scoldc v3",
    description="Quadratic limb-darkening coefficients by trilinear interpolation "
                "of Claret tables.",
    version="3.0.0",
)

# CORS is permissive because the app is a read-only public tool served from
# a single origin; the only traffic is same-origin fetches from index.html,
# but leaving CORS open makes it easy to hit /api/* from notebooks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load tables once at startup.
LOAD_COUNTS = ldc_core.load_tables(DATA_DIR)


@app.get("/api/health")
def health() -> dict:
    """Liveness probe with a summary of what was loaded at startup."""
    return {
        "status": "ok",
        "version": app.version,
        "tables": LOAD_COUNTS,
        "filter_count": len(ldc_core.get_available_filters()),
    }


@app.get("/api/filters")
def filters() -> dict:
    """Filter registry with per-model grid ranges for UI range hints."""
    return {"filters": ldc_core.get_available_filters()}


@app.get("/api/compute")
def compute(
    teff: float = Query(..., description="Effective temperature in K"),
    logg: float = Query(..., description="Surface gravity log g in cgs dex"),
    feh:  float = Query(0.0, description="Metallicity [Fe/H] in dex (solar=0.0)"),
    filter: str = Query(..., alias="filter", description="Filter code (e.g. 'V', 'Kp', 'TESS', 'CBB')"),
    model:  str = Query("ATLAS", description="Stellar atmosphere model: ATLAS, PHOENIX, or PHOENIX-COND"),
) -> dict:
    """
    Compute quadratic limb-darkening coefficients (u1, u2) by trilinear
    interpolation. Returns a JSON body with the coefficients, the filter/
    model metadata, the citation, and the interpolation bracket details.
    """
    try:
        result = ldc_core.compute_ldcs(teff, logg, feh, filter, model)
    except ValueError as e:
        # Validation / out-of-range / missing-grid-point: 400.
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/api/resolve")
def resolve(
    planet: str = Query(..., description="Exoplanet name (e.g. 'WASP-23 b') or TOI identifier (e.g. 'TOI-1234.01')"),
) -> dict:
    """
    Resolve an exoplanet name or TOI identifier to host-star stellar
    parameters.

    Routing logic:
      1. Query NEA PSCompPars. If found, return NEA result.
      2. If not found in NEA AND the input looks like a TOI identifier,
         fall through to ExoFOP-TESS.
      3. If neither source has it (or input is not a TOI and NEA had
         nothing), return the final "not found" result.

    The endpoint always returns 200 with a JSON body describing the
    outcome. Callers should branch on the "found" field rather than
    relying on HTTP status codes.
    """
    nea_result = nea_resolver.query_nea(planet)

    # NEA had it, or NEA errored. Either way, return what NEA gave us.
    # An NEA error is more informative than silently trying ExoFOP.
    if nea_result.get("found") is True:
        return nea_result
    if nea_result.get("reason") == "error":
        return nea_result

    # NEA returned "not_in_nea". Try ExoFOP only if the input looks
    # like a TOI identifier - for planet names like "WASP-23 b",
    # ExoFOP can't help and the call would just waste time.
    if exofop_resolver.looks_like_toi(planet):
        return exofop_resolver.query_exofop(planet)

    # Not in NEA and not a TOI. Final "not found".
    return nea_result


# --- Static frontend -------------------------------------------------------

# Serve /static/* as assets (not strictly needed for a single-file index,
# but keeps the door open for adding CSS/JS files without changing app.py).
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    """Serve the single-page frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=500,
            content={"error": "static/index.html is missing"},
        )
    return FileResponse(index_path, media_type="text/html")
