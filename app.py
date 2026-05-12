from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import ldc_core
import nea_resolver
import exofop_resolver

logger = logging.getLogger("scoldc")
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
NEA_FALLBACK_PATH = os.path.join(DATA_DIR, "nea_name_list_fallback.txt")

app = FastAPI(
    title="scoldc v3",
    description="Quadratic limb-darkening coefficients by trilinear interpolation "
                "of Claret tables.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

LOAD_COUNTS = ldc_core.load_tables(DATA_DIR)

NAMELIST_STATUS = nea_resolver.load_namelist_at_startup(
    fallback_path=NEA_FALLBACK_PATH,
)
logger.info(
    "Loaded NEA name list: source=%s count=%d",
    NAMELIST_STATUS["source"],
    NAMELIST_STATUS["count"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "tables": LOAD_COUNTS,
        "filter_count": len(ldc_core.get_available_filters()),
        "namelist": nea_resolver.namelist_status(),
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
    planet: str = Query(..., description="Exoplanet name (e.g. 'WASP-23 b') or TOI identifier (e.g. 'TOI-1234.01')"),
) -> dict:
    nea_resolver.maybe_refresh_namelist()

    nea_result = nea_resolver.query_nea(planet)

    
    if nea_result.get("found") is True:
        return nea_result

    if nea_result.get("reason") == "error":
        return nea_result

    
    if exofop_resolver.looks_like_toi(planet):
        exofop_result = exofop_resolver.query_exofop(planet)
        if exofop_result.get("found") is True:
            return exofop_result
        
    nea_result["suggestions"] = nea_resolver.get_suggestions(planet)
    return nea_result




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
