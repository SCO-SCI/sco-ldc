"""
scoldc v3 - FastAPI backend.

Endpoints:
  GET /api/health              - liveness + load summary
  GET /api/filters             - available filters with per-model grid ranges
  GET /api/compute             - compute (u1, u2) by trilinear interpolation
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
