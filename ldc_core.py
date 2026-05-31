from __future__ import annotations

import os
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple


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
    # CBB (Blue Blocking Exoplanet).
    {"code": "CBB", "name": "CBB (Blue Blocking Exoplanet)",
                                                  "category": "Exoplanet",     "source": "CMG2022"},
    # CHEOPS
    {"code": "CHEOPS", "name": "CHEOPS",        "category": "Space-based",     "source": "C2021"},
]


SOURCE_CITATIONS: Dict[str, str] = {
    "CB2011":  "Claret & Bloemen (2011, A&A 529, A75)",
    "C2018":   "Claret (2018, A&A 618, A20)",
    "CMG2022": "Claret, Mullen & Gary (2022, RNAAS 6, 169)",
    "C2021":   "Claret (2021, RNAAS 5, 13)",
}


EXPECTED_MODELS: Dict[str, List[str]] = {
    "CB2011":  ["ATLAS", "PHOENIX"],   
    "C2018":   ["PHOENIX"],            
    "CMG2022": ["ATLAS"],
    "C2021":   ["ATLAS", "PHOENIX"],
}


MODEL_DISPLAY_NAMES: Dict[Tuple[str, str], str] = {
    ("TESS", "PHOENIX"): "PHOENIX-COND",
    ("CHEOPS", "PHOENIX"): "PHOENIX-COND",
}


def _display_model(filter_code: str, model: str) -> str:
    
    return MODEL_DISPLAY_NAMES.get((filter_code, model), model)




Grid = Dict[str, object]
_TABLES: Dict[Tuple[str, str], Grid] = {}


def _add_point(table_key: Tuple[str, str],
               teff: float, logg: float, feh: float,
               u1: float, u2: float) -> None:
   
    grid = _TABLES.setdefault(table_key, {
        "teffs": set(), "loggs": set(), "fehs": set(), "data": {}
    })
    
    t = round(float(teff), 2)
    g = round(float(logg), 3)
    z = round(float(feh), 3)
    grid["teffs"].add(t)         # type: ignore[union-attr]
    grid["loggs"].add(g)         # type: ignore[union-attr]
    grid["fehs"].add(z)          # type: ignore[union-attr]
    grid["data"][(t, g, z)] = (float(u1), float(u2))   # type: ignore[index]


def _finalize_tables() -> None:
    
    for grid in _TABLES.values():
        grid["teffs"] = sorted(grid["teffs"])   # type: ignore[arg-type]
        grid["loggs"] = sorted(grid["loggs"])   # type: ignore[arg-type]
        grid["fehs"]  = sorted(grid["fehs"])    # type: ignore[arg-type]



def _parse_tableab(path: str) -> int:
    
    count = 0
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            
            line = raw.rstrip("\r\n")
            
            if len(line) < 49:
                continue
            
            try:
                logg = float(line[0:5])
                teff = float(line[6:12])
                feh  = float(line[13:17])
                xi   = float(line[18:22])
                u1   = float(line[23:30])
                u2   = float(line[31:38])
            except ValueError:
                continue
            
            code = line[39:41].rstrip()
            met  = line[42:43].strip()
            
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
            
            _add_point(("C2018", "TESS", "PHOENIX"), teff, logg, feh, u1, u2)
            count += 1
    return count


def _parse_cbbquadratic(path: str) -> int:
    
    count = 0
    buf: List[Tuple[float, float, float, float]] = []   # (logg, teff, feh, coeff)
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            
            if len(parts) != 5:
                continue
            try:
                logg = float(parts[0])
                teff = float(parts[1])
                feh  = float(parts[2])
                vel  = float(parts[3])
                coef = float(parts[4])
            except ValueError:
                
                continue
            buf.append((logg, teff, feh, coef))
            
            if len(buf) == 3:
                (lg1, te1, fe1, c_a) = buf[0]
                (lg2, te2, fe2, c_b) = buf[1]
                (lg3, te3, fe3, c_x) = buf[2]
                buf = []
                
                if not (lg1 == lg2 == lg3 and te1 == te2 == te3 and fe1 == fe2 == fe3):
                    
                    continue
                
                if abs(vel - 2.0) > 1e-6:
                    continue
                _add_point(("CMG2022", "CBB", "ATLAS"), te1, lg1, fe1, c_a, c_b)
                count += 1
    return count


def _parse_c2021(path: str, model: str) -> int:
    
    count = 0
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                if model == "ATLAS":
                    logg = float(parts[0])
                    teff = float(parts[1])
                    feh  = float(parts[2])
                    vel  = float(parts[3])
                    u1   = float(parts[4])
                    u2   = float(parts[5])
                    if abs(vel - 2.0) > 1e-6:
                        continue
                else:  # PHOENIX-COND
                    logg = float(parts[0])
                    teff = float(parts[1])
                    feh  = float(parts[2])
                    u1   = float(parts[3])
                    u2   = float(parts[4])
            except ValueError:
                continue

            _add_point(("C2021", "CHEOPS", model), teff, logg, feh, u1, u2)
            count += 1
    return count




import pickle

CACHE_FILENAME = "tables.pkl"
CACHE_VERSION = 3        
SOURCE_FILES = ("tableab.dat", "table5.dat", "CBBQUADRATIC.txt",
                "table2.dat", "table8.dat")


def _cache_is_fresh(cache_path: str, source_paths: List[str]) -> bool:
    
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for p in source_paths:
        if not os.path.exists(p):
            
            return False
        if os.path.getmtime(p) > cache_mtime:
            return False
    return True


def _parse_all(data_dir: str) -> Dict[str, int]:
    
    counts: Dict[str, int] = {}
    counts["tableab.dat"]      = _parse_tableab(os.path.join(data_dir, "tableab.dat"))
    counts["table5.dat"]       = _parse_table5(os.path.join(data_dir, "table5.dat"))
    counts["CBBQUADRATIC.txt"] = _parse_cbbquadratic(os.path.join(data_dir, "CBBQUADRATIC.txt"))
    counts["table2.dat"]       = _parse_c2021(os.path.join(data_dir, "table2.dat"), "PHOENIX")
    counts["table8.dat"]       = _parse_c2021(os.path.join(data_dir, "table8.dat"), "ATLAS")
    _finalize_tables()
    return counts


def _save_cache(cache_path: str, counts: Dict[str, int]) -> None:
    
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
    
    _TABLES.clear()

    cache_path = os.path.join(data_dir, CACHE_FILENAME)
    source_paths = [os.path.join(data_dir, f) for f in SOURCE_FILES]

    if use_cache and _cache_is_fresh(cache_path, source_paths):
        counts = _load_cache(cache_path)
        if counts is not None:
            return counts
        

    counts = _parse_all(data_dir)

    
    try:
        _save_cache(cache_path, counts)
    except OSError:
        pass

    return counts




def _resolve_table_key(filter_code: str, model: str) -> Tuple[str, str, str]:
    
    entry = None
    for f in FILTER_REGISTRY:
        if f["code"] == filter_code:
            entry = f
            break
    if entry is None:
        raise ValueError(f"Unknown filter: {filter_code!r}")
    source = entry["source"]

    
    if model.upper() == "PHOENIX-COND":
        storage_model = "PHOENIX"
    else:
        storage_model = model.upper()

    return (source, filter_code, storage_model)


def get_available_filters() -> List[Dict]:
    
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




def _bracket(axis: List[float], x: float) -> Tuple[int, int, float]:
    
    lo = axis[0]
    hi = axis[-1]
    if x < lo - 1e-9 or x > hi + 1e-9:
        raise ValueError(f"value {x} outside grid [{lo}, {hi}]")
    
    if x <= lo:
        return 0, 0, 0.0
    if x >= hi:
        n = len(axis) - 1
        return n, n, 0.0
    
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
    
    cube: List[List[List[Tuple[float, float]]]] = [[[(0.0, 0.0)] * 2 for _ in range(2)] for _ in range(2)]
    for i, te in enumerate(teff_vals):
        for j, lg in enumerate(logg_vals):
            for k, fe in enumerate(feh_vals):
                key = (round(te, 2), round(lg, 3), round(fe, 3))
                if key not in data:
                    return None
                cube[i][j][k] = data[key]
    return cube, teff_vals, logg_vals, feh_vals


def _filter_has_model(filter_code: str, storage_model: str) -> bool:
   
    for f in FILTER_REGISTRY:
        if f["code"] == filter_code:
            return storage_model in EXPECTED_MODELS.get(f["source"], [])
    return False


def compute_ldcs(teff: float, logg: float, feh: float,
                 filter_code: str, model: str
                 ) -> Dict[str, object]:
    
    source, code, storage_model = _resolve_table_key(filter_code, model)
    grid = _TABLES.get((source, code, storage_model))
    if grid is None:
        raise ValueError(
            f"no data for filter {filter_code!r} with model {model!r}")

    teffs: List[float] = grid["teffs"]   # type: ignore[assignment]
    loggs: List[float] = grid["loggs"]   # type: ignore[assignment]
    fehs:  List[float] = grid["fehs"]    # type: ignore[assignment]
    data = grid["data"]                  # type: ignore[assignment]

    
    try:
        i0, i1, tT = _bracket(teffs, float(teff))
    except ValueError as e:
        model_name = _display_model(filter_code, storage_model)
        if float(teff) < teffs[0]:
            suggestion = " Use the PHOENIX model instead." if storage_model == "ATLAS" and _filter_has_model(filter_code, "PHOENIX") else ""
            raise ValueError(
                f"Invalid Input (Teff = {teff} K): "
                f"The {model_name} model does not support values of Teff "
                f"below {teffs[0]:.0f} K.{suggestion}"
            ) from e
        else:
            suggestion = " Use the ATLAS model instead." if storage_model == "PHOENIX" and _filter_has_model(filter_code, "ATLAS") else ""
            raise ValueError(
                f"Invalid Input (Teff = {teff} K): "
                f"The {model_name} model does not support values of Teff "
                f"above {teffs[-1]:.0f} K.{suggestion}"
            ) from e
    try:
        j0, j1, tG = _bracket(loggs, float(logg))
    except ValueError as e:
        model_name = _display_model(filter_code, storage_model)
        if float(logg) < loggs[0]:
            suggestion = " Use the ATLAS model instead." if storage_model == "PHOENIX" and _filter_has_model(filter_code, "ATLAS") else ""
            raise ValueError(
                f"Invalid Input (log g = {logg}): "
                f"The {model_name} model does not support values of log g "
                f"below {loggs[0]:.1f}.{suggestion}"
            ) from e
        else:
            raise ValueError(
                f"Invalid Input (log g = {logg}): "
                f"The {model_name} model does not support values of log g "
                f"above {loggs[-1]:.1f}."
            ) from e            
    
    if len(fehs) == 1:
        if abs(float(feh) - fehs[0]) > 1e-6:
            
            raise ValueError(
                f"[Fe/H] {feh} not available for {filter_code}/"
                f"{_display_model(filter_code, storage_model)} "
                f"(solar metallicity only: {fehs[0]:+.1f})")
        k0, k1, tZ = 0, 0, 0.0
    else:
        try:
            k0, k1, tZ = _bracket(fehs, float(feh))
        except ValueError as e:
            model_name = _display_model(filter_code, storage_model)
            if float(feh) < fehs[0]:
                raise ValueError(
                    f"Invalid Input ([Fe/H] = {feh}): "
                    f"The {model_name} model does not support values of [Fe/H] "
                    f"below {fehs[0]:+.1f}."
                ) from e
            else:
                raise ValueError(
                    f"Invalid Input ([Fe/H] = {feh}): "
                    f"The {model_name} model does not support values of [Fe/H] "
                    f"above {fehs[-1]:+.1f}."
                ) from e

    teff_vals = (teffs[i0], teffs[i1])
    logg_vals = (loggs[j0], loggs[j1])
    feh_vals  = (fehs[k0],  fehs[k1])

    corners = _nearest_available(data, teff_vals, logg_vals, feh_vals)  # type: ignore[arg-type]
    if corners is None:
        
        model_name = _display_model(filter_code, storage_model)
        raise ValueError(
            f"Invalid Input: Tables do not include data for this combination of "
            f"Teff = {teff}, log g = {logg}, and [Fe/H] = {feh} "
            f"with the {model_name} model."
        )

    cube, _, _, _ = corners

    
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
