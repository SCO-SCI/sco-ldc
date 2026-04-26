"""
scoldc v3 - Limb Darkening Coefficient core.

Loads Claret quadratic LDC tables and performs trilinear interpolation in
(T_eff, log g, [Fe/H]). Pure-Python, no numpy/scipy dependency.

Data sources:
  - tableab.dat     : Claret & Bloemen (2011, A&A 529, A75), 23 filters, ATLAS + PHOENIX
  - table5.dat      : Claret (2018, A&A 618, A20), TESS, PHOENIX-COND, solar Z only
  - CBBQUADRATIC.txt: Claret, Mullen & Gary (2022, RNAAS 6, 169), CBB, ATLAS only

LSM method and microturbulent velocity xi = 2.0 km/s are used throughout.
"""

from __future__ import annotations

import os
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

# Ordering here controls display order inside each category in the UI.
FILTER_REGISTRY: List[Dict] = [
    # Johnson-Cousins
    {"code": "U",  "name": "Johnson U",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "B",  "name": "Johnson B",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "V",  "name": "Johnson V",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "R",  "name": "Cousins R",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "I",  "name": "Cousins I",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "J",  "name": "Johnson J",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "H",  "name": "Johnson H",         "category": "Johnson-Cousins", "source": "CB2011"},
    {"code": "K",  "name": "Johnson K",         "category": "Johnson-Cousins", "source": "CB2011"},
    # Sloan / SDSS (codes end in '*' in tableab.dat)
    {"code": "u*", "name": "SDSS u'",           "category": "Sloan/SDSS",      "source": "CB2011"},
    {"code": "g*", "name": "SDSS g'",           "category": "Sloan/SDSS",      "source": "CB2011"},
    {"code": "r*", "name": "SDSS r'",           "category": "Sloan/SDSS",      "source": "CB2011"},
    {"code": "i*", "name": "SDSS i'",           "category": "Sloan/SDSS",      "source": "CB2011"},
    {"code": "z*", "name": "SDSS z'",           "category": "Sloan/SDSS",      "source": "CB2011"},
    # Stromgren
    {"code": "u",  "name": "Strömgren u",       "category": "Strömgren",       "source": "CB2011"},
    {"code": "v",  "name": "Strömgren v",       "category": "Strömgren",       "source": "CB2011"},
    {"code": "b",  "name": "Strömgren b",       "category": "Strömgren",       "source": "CB2011"},
    {"code": "y",  "name": "Strömgren y",       "category": "Strömgren",       "source": "CB2011"},
    # Space-based
    {"code": "Kp", "name": "Kepler",            "category": "Space-based",     "source": "CB2011"},
    {"code": "C",  "name": "CoRoT",             "category": "Space-based",     "source": "CB2011"},
    {"code": "S1", "name": "Spitzer 3.6 μm",    "category": "Space-based",     "source": "CB2011"},
    {"code": "S2", "name": "Spitzer 4.5 μm",    "category": "Space-based",     "source": "CB2011"},
    {"code": "S3", "name": "Spitzer 5.8 μm",    "category": "Space-based",     "source": "CB2011"},
    {"code": "S4", "name": "Spitzer 8.0 μm",    "category": "Space-based",     "source": "CB2011"},
    # TESS
    {"code": "TESS", "name": "TESS",            "category": "Space-based",     "source": "C2018"},
    # CBB (Blue Blocking Exoplanet) - no 'Astrodon' branding per spec.
    {"code": "CBB", "name": "CBB (Blue Blocking Exoplanet)",
                                                  "category": "Exoplanet",     "source": "CMG2022"},
]

# Citation strings per source tag.
SOURCE_CITATIONS: Dict[str, str] = {
    "CB2011":  "Claret & Bloemen (2011, A&A 529, A75)",
    "C2018":   "Claret (2018, A&A 618, A20)",
    "CMG2022": "Claret, Mullen & Gary (2022, RNAAS 6, 169)",
}

# Which models are populated for a given (filter_code, source).
# Resolved dynamically from the loaded tables but these are the expected sets.
EXPECTED_MODELS: Dict[str, List[str]] = {
    "CB2011":  ["ATLAS", "PHOENIX"],   # PHOENIX may be absent for some filters (e.g. Spitzer)
    "C2018":   ["PHOENIX"],            # stored internally; displayed as "PHOENIX-COND"
    "CMG2022": ["ATLAS"],
}

# Model display names (what the user sees) vs storage keys.
MODEL_DISPLAY_NAMES: Dict[Tuple[str, str], str] = {
    ("TESS", "PHOENIX"): "PHOENIX-COND",
}


def _display_model(filter_code: str, model: str) -> str:
    """Return the user-facing model name for a (filter, model) pair."""
    return MODEL_DISPLAY_NAMES.get((filter_code, model), model)


# ---------------------------------------------------------------------------
# Grid storage
#
# For every (filter_code, model) pair this stores:
#   grid[(filter_code, model)] = {
#       "teffs": sorted list of T_eff grid values,
#       "loggs": sorted list of log g grid values,
#       "fehs" : sorted list of [Fe/H] grid values,
#       "data" : dict[(teff, logg, feh)] -> (u1, u2),
#   }
# ---------------------------------------------------------------------------

Grid = Dict[str, object]
_TABLES: Dict[Tuple[str, str], Grid] = {}


def _add_point(table_key: Tuple[str, str],
               teff: float, logg: float, feh: float,
               u1: float, u2: float) -> None:
    """Insert a single grid point into the per-(filter, model) table."""
    grid = _TABLES.setdefault(table_key, {
        "teffs": set(), "loggs": set(), "fehs": set(), "data": {}
    })
    # Round grid coordinates to avoid floating-point key mismatches.
    t = round(float(teff), 2)
    g = round(float(logg), 3)
    z = round(float(feh), 3)
    grid["teffs"].add(t)         # type: ignore[union-attr]
    grid["loggs"].add(g)         # type: ignore[union-attr]
    grid["fehs"].add(z)          # type: ignore[union-attr]
    grid["data"][(t, g, z)] = (float(u1), float(u2))   # type: ignore[index]


def _finalize_tables() -> None:
    """Convert set-valued axis lists to sorted lists after loading."""
    for grid in _TABLES.values():
        grid["teffs"] = sorted(grid["teffs"])   # type: ignore[arg-type]
        grid["loggs"] = sorted(grid["loggs"])   # type: ignore[arg-type]
        grid["fehs"]  = sorted(grid["fehs"])    # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def _parse_tableab(path: str) -> int:
    """
    Parse Claret & Bloemen (2011) fixed-width table.
    Columns (1-indexed bytes):
       1-5  logg  F5.2
       7-12 Teff  F6.0
      14-17 Z     F4.1
      19-22 xi    F4.1
      24-30 a     F7.4   (= u1)
      32-38 b     F7.4   (= u2)
      40-41 Filt  A2
      43    Met   A1     'L' = LSM, 'F' = FCM
      45-51 Mod   A7     'ATLAS' or 'PHOENIX'

    Filter to rows where Met='L' and xi=2.0 km/s.
    """
    count = 0
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            # Strip trailing newline/\r but preserve leading whitespace alignment.
            line = raw.rstrip("\r\n")
            # Line is 49 chars for ATLAS (5-char model name) or 51 chars for
            # PHOENIX (7-char model name). Anything shorter is malformed.
            if len(line) < 49:
                continue
            # Byte positions: Python slice is 0-indexed [start:end) exclusive.
            try:
                logg = float(line[0:5])
                teff = float(line[6:12])
                feh  = float(line[13:17])
                xi   = float(line[18:22])
                u1   = float(line[23:30])
                u2   = float(line[31:38])
            except ValueError:
                continue
            # Preserve case ('u' vs 'U') and the '*' on SDSS codes.
            code = line[39:41].rstrip()
            met  = line[42:43].strip()
            # Model field runs from byte 45 to end-of-line; strip whitespace.
            mod  = line[44:].strip()

            if met != "L":
                continue
            if abs(xi - 2.0) > 1e-6:
                continue
            if mod not in ("ATLAS", "PHOENIX"):
                continue

            _add_point(("CB2011", code, mod), teff, logg, feh, u1, u2)
            count += 1
    return count


def _parse_table5(path: str) -> int:
    """
    Parse Claret (2018) TESS fixed-width table.
    Columns:
       1-5  logg   F5.2
       7-12 Teff   F6.0
      14-17 Z      F4.1   (solar only in practice)
      19-22 L/HP   F4.1   (mixing-length; fixed at 2.0, ignored)
      24-31 a      F8.4   (= u1)
      33-40 b      F8.4   (= u2)
      42-49 mu     F8.4   (unused)
      51-58 chi2   F8.4   (unused)
      60-61 Mod    A2     'PC' = PHOENIX-COND
      63-68 Sys    A6     'TESS'
    """
    count = 0
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if len(line) < 40:
                continue
            try:
                logg = float(line[0:5])
                teff = float(line[6:12])
                feh  = float(line[13:17])
                u1   = float(line[23:31])
                u2   = float(line[32:40])
            except ValueError:
                continue
            # Store under filter 'TESS', internal model key 'PHOENIX'
            # (displayed as 'PHOENIX-COND' in the UI).
            _add_point(("C2018", "TESS", "PHOENIX"), teff, logg, feh, u1, u2)
            count += 1
    return count


def _parse_cbbquadratic(path: str) -> int:
    """
    Parse CBB quadratic table (Claret, Mullen & Gary 2022).

    File format:
      Rows come in triplets for each (logg, Teff, Z, xi) grid point:
        row 1 = a(CBBED)  -> u1
        row 2 = b(CBBED)  -> u2
        row 3 = xi(CBBED) -> ignored (a fit-quality diagnostic, not stellar xi)

    Grid stellar microturbulent velocity appears in column 4 ('Vel').
    This will keep only rows where Vel == 2.0. Blank lines and header lines are
    skipped.

    The 'xi' value in row 3 is NOT the microturbulent velocity; it is a
    secondary coefficient from the fit. Not to be confused with col 4.
    """
    count = 0
    buf: List[Tuple[float, float, float, float]] = []   # (logg, teff, feh, coeff)
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            # Valid data rows have exactly 5 tokens, all numeric.
            if len(parts) != 5:
                continue
            try:
                logg = float(parts[0])
                teff = float(parts[1])
                feh  = float(parts[2])
                vel  = float(parts[3])
                coef = float(parts[4])
            except ValueError:
                # Header line like 'log g Teff  log Z Vel   a(CBBED)'.
                continue
            buf.append((logg, teff, feh, coef))
            # Once there are three rows, emit a grid point if they share (logg,teff,feh)
            # and the grid microturbulent velocity is 2.0.
            if len(buf) == 3:
                (lg1, te1, fe1, c_a) = buf[0]
                (lg2, te2, fe2, c_b) = buf[1]
                (lg3, te3, fe3, c_x) = buf[2]
                buf = []
                # Sanity check: all three rows should agree on (logg, teff, feh).
                if not (lg1 == lg2 == lg3 and te1 == te2 == te3 and fe1 == fe2 == fe3):
                    # Shouldn't happen with a well-formed file; skip defensively.
                    continue
                # The 'Vel' column is the same across the triplet by construction,
                # so this can gate on any of them. Use the first row's vel.
                # Then re-read vel from parts above; the last-seen 'vel' is row 3's,
                # but all three agree. Check it here:
                if abs(vel - 2.0) > 1e-6:
                    continue
                _add_point(("CMG2022", "CBB", "ATLAS"), te1, lg1, fe1, c_a, c_b)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Public loader with pickle cache
#
# The raw .dat files contain ~188k grid points across 25 filter/model
# combinations. Parsing them from text takes ~1 s on a fast server and
# 3-4 s on Render's starter tier, which shows up as cold-start latency
# every time the worker spins up after idle.
#
# This will cache the parsed _TABLES dict to a pickle file alongside the
# data files. The cache is regenerated automatically whenever:
#   - the pickle is missing, OR
#   - any source file is newer than the pickle, OR
#   - the cache format version doesn't match (CACHE_VERSION below)
#
# Bump CACHE_VERSION whenever the on-disk shape of _TABLES changes.
# ---------------------------------------------------------------------------

import pickle

CACHE_FILENAME = "tables.pkl"
CACHE_VERSION = 2        # bump if _TABLES layout changes
SOURCE_FILES = ("tableab.dat", "table5.dat", "CBBQUADRATIC.txt")


def _cache_is_fresh(cache_path: str, source_paths: List[str]) -> bool:
    """True iff the cache exists and is newer than every source file."""
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for p in source_paths:
        if not os.path.exists(p):
            # Source file missing -- fall through to full parse so the
            # real parser can raise a meaningful error.
            return False
        if os.path.getmtime(p) > cache_mtime:
            return False
    return True


def _parse_all(data_dir: str) -> Dict[str, int]:
    """Parse all three source files into _TABLES. Pure parse, no caching."""
    counts: Dict[str, int] = {}
    counts["tableab.dat"]      = _parse_tableab(os.path.join(data_dir, "tableab.dat"))
    counts["table5.dat"]       = _parse_table5(os.path.join(data_dir, "table5.dat"))
    counts["CBBQUADRATIC.txt"] = _parse_cbbquadratic(os.path.join(data_dir, "CBBQUADRATIC.txt"))
    _finalize_tables()
    return counts


def _save_cache(cache_path: str, counts: Dict[str, int]) -> None:
    """Write _TABLES + row counts to a pickle. Atomic via tmp+rename."""
    payload = {
        "version": CACHE_VERSION,
        "tables":  _TABLES,
        "counts":  counts,
    }
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def _load_cache(cache_path: str) -> Optional[Dict[str, int]]:
    """Load _TABLES from pickle. Returns counts on success, None on failure."""
    try:
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return None
    tables = payload.get("tables")
    counts = payload.get("counts")
    if not isinstance(tables, dict) or not isinstance(counts, dict):
        return None
    _TABLES.clear()
    _TABLES.update(tables)
    return counts


def load_tables(data_dir: str, use_cache: bool = True) -> Dict[str, int]:
    """
    Populate the in-memory grids. Uses a pickle cache when available and
    fresh; otherwise parses the source files and writes a new cache.

    Returns a dict of {filename: rows_loaded} for startup logging.

    Set use_cache=False to force a full parse (used when regenerating the
    cache via build_cache.py).
    """
    _TABLES.clear()

    cache_path = os.path.join(data_dir, CACHE_FILENAME)
    source_paths = [os.path.join(data_dir, f) for f in SOURCE_FILES]

    if use_cache and _cache_is_fresh(cache_path, source_paths):
        counts = _load_cache(cache_path)
        if counts is not None:
            return counts
        # Cache was stale or corrupt -- fall through to full parse.

    counts = _parse_all(data_dir)

    # Best-effort write. If the data directory is read-only (e.g. Render's
    # build filesystem after deploy), skip silently rather than crash.
    try:
        _save_cache(cache_path, counts)
    except OSError:
        pass

    return counts


# ---------------------------------------------------------------------------
# Filter / model introspection
# ---------------------------------------------------------------------------

def _resolve_table_key(filter_code: str, model: str) -> Tuple[str, str, str]:
    """
    Given a user-facing filter code and a model choice, return the internal
    storage key (source, code, model) used in _TABLES. The 'model' argument
    accepts both storage-form ('PHOENIX') and display-form ('PHOENIX-COND').
    """
    entry = None
    for f in FILTER_REGISTRY:
        if f["code"] == filter_code:
            entry = f
            break
    if entry is None:
        raise ValueError(f"Unknown filter: {filter_code!r}")
    source = entry["source"]

    # Normalise the model string.
    if model.upper() == "PHOENIX-COND":
        storage_model = "PHOENIX"
    else:
        storage_model = model.upper()

    return (source, filter_code, storage_model)


def get_available_filters() -> List[Dict]:
    """
    Return filter metadata for the UI, including which stellar models are
    actually populated and the min/max of each grid axis for range hints.
    """
    out: List[Dict] = []
    for f in FILTER_REGISTRY:
        code = f["code"]
        source = f["source"]
        models_present: List[Dict] = []
        for storage_model in EXPECTED_MODELS[source]:
            grid = _TABLES.get((source, code, storage_model))
            if grid is None:
                continue
            teffs = grid["teffs"]   # type: ignore[index]
            loggs = grid["loggs"]   # type: ignore[index]
            fehs  = grid["fehs"]    # type: ignore[index]
            models_present.append({
                "model": _display_model(code, storage_model),   # display name
                "model_key": storage_model,                      # storage key
                "teff_min": teffs[0],  "teff_max": teffs[-1],
                "logg_min": loggs[0],  "logg_max": loggs[-1],
                "feh_min":  fehs[0],   "feh_max":  fehs[-1],
                "feh_fixed": (len(fehs) == 1),
                "n_points": len(grid["data"]),                   # type: ignore[arg-type]
            })
        if not models_present:
            # Filter has no data loaded; skip it rather than offer a broken option.
            continue
        out.append({
            "code": code,
            "name": f["name"],
            "category": f["category"],
            "source": source,
            "citation": SOURCE_CITATIONS[source],
            "models": models_present,
        })
    return out


# ---------------------------------------------------------------------------
# Trilinear interpolation
# ---------------------------------------------------------------------------

def _bracket(axis: List[float], x: float) -> Tuple[int, int, float]:
    """
    Find indices (i_lo, i_hi) bracketing x on a sorted axis, plus the
    interpolation fraction t in [0, 1] such that
        x = axis[i_lo] + t * (axis[i_hi] - axis[i_lo]).

    If x exactly matches a grid point, i_lo == i_hi and t == 0.
    Raises ValueError if x is outside the axis range (no extrapolation).
    """
    lo = axis[0]
    hi = axis[-1]
    if x < lo - 1e-9 or x > hi + 1e-9:
        raise ValueError(f"value {x} outside grid [{lo}, {hi}]")
    # Clamp exact-edge cases that drifted by 1e-9.
    if x <= lo:
        return 0, 0, 0.0
    if x >= hi:
        n = len(axis) - 1
        return n, n, 0.0
    # Locate insertion point.
    idx = bisect_left(axis, x)
    if idx < len(axis) and axis[idx] == x:
        return idx, idx, 0.0
    i_hi = idx
    i_lo = idx - 1
    span = axis[i_hi] - axis[i_lo]
    t = (x - axis[i_lo]) / span if span > 0 else 0.0
    return i_lo, i_hi, t


def _nearest_available(data: Dict[Tuple[float, float, float], Tuple[float, float]],
                       teff_vals: Tuple[float, float],
                       logg_vals: Tuple[float, float],
                       feh_vals: Tuple[float, float]
                       ) -> Optional[Tuple[List[List[List[Tuple[float, float]]]],
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float]]]:
    """
    Try to assemble the 2x2x2 corner cube from `data`. If any corner is
    missing (which happens in the ATLAS grid because high-Teff cells drop
    rows at low log g / extreme Z), return None so the caller can report a
    clean error rather than crash.
    """
    cube: List[List[List[Tuple[float, float]]]] = [[[(0.0, 0.0)] * 2 for _ in range(2)] for _ in range(2)]
    for i, te in enumerate(teff_vals):
        for j, lg in enumerate(logg_vals):
            for k, fe in enumerate(feh_vals):
                key = (round(te, 2), round(lg, 3), round(fe, 3))
                if key not in data:
                    return None
                cube[i][j][k] = data[key]
    return cube, teff_vals, logg_vals, feh_vals


def compute_ldcs(teff: float, logg: float, feh: float,
                 filter_code: str, model: str
                 ) -> Dict[str, object]:
    """
    Compute quadratic limb-darkening coefficients (u1, u2) by trilinear
    interpolation.

    Degenerate axes (single grid value for TESS, or [Fe/H]=0 only for
    PHOENIX in CB2011) collapse to bilinear or linear automatically: the
    bracket returns i_lo == i_hi with t = 0, and that axis contributes unit
    weight.

    Raises:
        ValueError  - unknown filter, unknown model, or inputs outside grid.
    """
    source, code, storage_model = _resolve_table_key(filter_code, model)
    grid = _TABLES.get((source, code, storage_model))
    if grid is None:
        raise ValueError(
            f"no data for filter {filter_code!r} with model {model!r}")

    teffs: List[float] = grid["teffs"]   # type: ignore[assignment]
    loggs: List[float] = grid["loggs"]   # type: ignore[assignment]
    fehs:  List[float] = grid["fehs"]    # type: ignore[assignment]
    data = grid["data"]                  # type: ignore[assignment]

    # Bracket each axis.
    try:
        i0, i1, tT = _bracket(teffs, float(teff))
    except ValueError as e:
        raise ValueError(f"T_eff {teff} K outside grid [{teffs[0]}, {teffs[-1]}] for "
                         f"{filter_code}/{_display_model(filter_code, storage_model)}") from e
    try:
        j0, j1, tG = _bracket(loggs, float(logg))
    except ValueError as e:
        raise ValueError(f"log g {logg} outside grid [{loggs[0]}, {loggs[-1]}] for "
                         f"{filter_code}/{_display_model(filter_code, storage_model)}") from e
    # [Fe/H] is degenerate for TESS and for CB2011 PHOENIX. If the user
    # submits something other than the single available Z, this will snap to it
    # (with a 1e-6 tolerance) rather than error out, because the UI may
    # leave the input filled in from a prior selection.
    if len(fehs) == 1:
        if abs(float(feh) - fehs[0]) > 1e-6:
            # Treat as out-of-range only if the user's value is materially
            # different. In practice the UI disables the field, but be safe.
            raise ValueError(
                f"[Fe/H] {feh} not available for {filter_code}/"
                f"{_display_model(filter_code, storage_model)} "
                f"(solar metallicity only: {fehs[0]:+.1f})")
        k0, k1, tZ = 0, 0, 0.0
    else:
        try:
            k0, k1, tZ = _bracket(fehs, float(feh))
        except ValueError as e:
            raise ValueError(f"[Fe/H] {feh} outside grid [{fehs[0]}, {fehs[-1]}] for "
                             f"{filter_code}/{_display_model(filter_code, storage_model)}") from e

    teff_vals = (teffs[i0], teffs[i1])
    logg_vals = (loggs[j0], loggs[j1])
    feh_vals  = (fehs[k0],  fehs[k1])

    corners = _nearest_available(data, teff_vals, logg_vals, feh_vals)  # type: ignore[arg-type]
    if corners is None:
        # Inside the nominal axis ranges but the ATLAS grid has a gap here.
        raise ValueError(
            f"grid point missing for T_eff={teff}, log g={logg}, [Fe/H]={feh} "
            f"in {filter_code}/{_display_model(filter_code, storage_model)}; "
            f"Claret's table does not cover this corner of parameter space")

    cube, _, _, _ = corners

    # Trilinear weights (collapse to bilinear/linear automatically when an
    # axis is degenerate, because tT/tG/tZ = 0 on a degenerate axis).
    w = [
        [(1.0 - tT) * (1.0 - tG) * (1.0 - tZ),   # i=0 j=0 k=0
         (1.0 - tT) * (1.0 - tG) * tZ],          # i=0 j=0 k=1
        [(1.0 - tT) * tG * (1.0 - tZ),           # i=0 j=1 k=0
         (1.0 - tT) * tG * tZ],                  # i=0 j=1 k=1
    ], [
        [tT * (1.0 - tG) * (1.0 - tZ),           # i=1 j=0 k=0
         tT * (1.0 - tG) * tZ],                  # i=1 j=0 k=1
        [tT * tG * (1.0 - tZ),                   # i=1 j=1 k=0
         tT * tG * tZ],                          # i=1 j=1 k=1
    ]

    u1 = 0.0
    u2 = 0.0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                c1, c2 = cube[i][j][k]
                weight = w[i][j][k]
                u1 += weight * c1
                u2 += weight * c2

    return {
        "u1": u1,
        "u2": u2,
        "filter_code": filter_code,
        "filter_name": next(f["name"] for f in FILTER_REGISTRY if f["code"] == filter_code),
        "model": _display_model(filter_code, storage_model),
        "citation": SOURCE_CITATIONS[source],
        "grid": {
            "teff_bracket": [teff_vals[0], teff_vals[1]],
            "logg_bracket": [logg_vals[0], logg_vals[1]],
            "feh_bracket":  [feh_vals[0],  feh_vals[1]],
            "fractions":    {"teff": tT, "logg": tG, "feh": tZ},
            "on_grid":      (tT == 0.0 and tG == 0.0 and tZ == 0.0),
        },
    }
