#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEM-only terrain burn for the Scenario-1 3D voxel model.

This script is intentionally separated from obstacle burning.
It does only one job:

    1. Read the base 3D voxel data-box model from script 02.
    2. Sample the DEM/topography to each XY voxel cell.
    3. Burn all voxel cells below/intersecting the DEM surface as no-fly.
    4. Plot:
         - DEM/topography on the XY cell grid.
         - DEM-burned cells at selected Z slices.
         - 3D filled DEM-burned voxel cells.

Coordinate convention
---------------------
The base voxel vertical coordinate is treated as MSL:

    z_center_msl_m = z_agl_m   # if the base file still uses z_agl_m

Local plotting coordinates are calculated from the southwest corner of the
XY cell data-box:

    x_from_sw_m = x_utm_m - x_sw_corner_m
    y_from_sw_m = y_utm_m - y_sw_corner_m
    distance_from_sw_m = sqrt(x_from_sw_m^2 + y_from_sw_m^2)

Run from make_model/
--------------------
    python 03a_burn_dem_cells_only.py

Outputs
-------
    output/03a_dem_terrain_burn_only_senario1/
    ├── data/
    │   ├── dem_only_voxel_model_50m.csv.gz
    │   ├── dem_only_voxel_model_50m.parquet
    │   ├── dem_only_voxel_model_50m.xyz
    │   ├── xy_grid_with_dem_terrain_msl_SW.csv.gz
    │   ├── xy_grid_with_dem_terrain_msl_SW.gpkg
    │   ├── dem_candidate_audit.csv
    │   └── dem_terrain_burn_summary.txt/json
    └── figures/
        ├── 00_dem_terrain_msl_cells_SW.png
        ├── 01_dem_terrain_burn_z_slices_SW.png
        └── 02_3d_dem_burned_cells_SW.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import box
from shapely.affinity import translate

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ======================================================================
# USER PARAMETERS
# ======================================================================

# Base voxel model from script 02.
BASE_MODEL_DIR = Path("output/02_base_3d_voxel_box_model_senario1/data")
BASE_MODEL_PARQUET = BASE_MODEL_DIR / "base_3d_voxel_data_box_50m.parquet"
BASE_MODEL_CSV_GZ = BASE_MODEL_DIR / "base_3d_voxel_data_box_50m.csv.gz"
BASE_SUMMARY_JSON = BASE_MODEL_DIR / "base_3d_voxel_data_box_summary.json"
AOI_UTM_FILE = BASE_MODEL_DIR / "aoi_polygon_utm.gpkg"
DATA_BOX_UTM_FILE = BASE_MODEL_DIR / "voxel_data_box_utm.gpkg"

# Input DEM data collected in script 01.
INPUT_DATA_DIR = Path("input/data_senario1")
OPENTOPOGRAPHY_DEM_DIR = INPUT_DATA_DIR / "opentopography"

# DEM source priority. Keep this strict so the script never grabs a wrong DEM/DSM file.
# Options: "tif_then_xyz", "xyz_then_tif", "tif_only", "xyz_only"
DEM_SOURCE_PRIORITY = "tif_then_xyz"

# Output folder for this DEM-only step.
OUTDIR = Path("output/03a_dem_terrain_burn_only_senario1")

# CRS fallback for Hoa Lac / Hanoi.
DEFAULT_UTM_CRS = "EPSG:32648"

# Set this to a specific DEM if you want to force the exact COP30 clipped file.
# Example:
# DEM_FILE_OVERRIDE = Path("input/data_senario1/opentopography/COP30_elevation_clipped.tif")
DEM_FILE_OVERRIDE: Path | None = None

# Hard no-fly convention.
NOFLY_SLOWNESS = 10.0
FLYABLE_SLOWNESS_OVERRIDE = None  # None = read median flyable slowness from base model.

# Voxel dimensions. Inferred from the base model if available.
DEFAULT_DX_M = 50.0
DEFAULT_DY_M = 50.0
DEFAULT_DZ_M = 5.0

# DEM terrain burn rule.
BURN_TERRAIN_UNDERGROUND = True
TERRAIN_CLEARANCE_BUFFER_M = 0.0
# Conservative cell rule: if the voxel vertical interval touches below DEM, burn it.
# This burns the cell containing the terrain surface, not only cells whose center is below DEM.
TERRAIN_BURN_VERTICAL_RULE = "z_bottom_below_dem"  # z_bottom_below_dem | z_center_below_dem | z_top_below_dem

# If no DEM is found, this fallback sets terrain to sea level.
DEFAULT_TERRAIN_MSL_M_IF_MISSING = 0.0

# DEM search patterns. Strictly search only inside:
#     input/data_senario1/opentopography/
# and only files that include "elevation" in the filename.
# This prevents accidentally selecting DSM/height/derivative rasters.
DEM_PATTERNS = [
    "**/*elevation*.tif",
    "**/*elevation*.tiff",
]
DEM_XYZ_PATTERNS = [
    "**/*elevation*.xyz",
    "**/*elevation*.txt",
    "**/*elevation*.csv",
]

# Plot settings.
FIG_DPI = 220
RANDOM_SEED = 42
SLICE_Z_LEVELS_MSL = [0, 5, 10, 15, 20, 25, 30, 40, 50]

# DEM plot display scale only.
# This does NOT change the DEM values used for terrain burning.
# Values above 40 m are clipped only for figure color mapping / display.
DEM_PLOT_VMIN_M = 5.0
DEM_PLOT_VMAX_M = 35.0
DEM_3D_ZMAX_PLOT_M = 40.0

# 3D figure uses sampled complete XY columns only, to keep the plot manageable.
# The saved model still burns every cell; this limit affects the figure only.
MAX_3D_DEM_XY_COLUMNS = 1200
MAX_3D_DEM_VOXELS_TO_RENDER = 22000

# Safety check: direct raster samples should not exceed the raw DEM range.
RASTER_RANGE_TOLERANCE_M = 0.75


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class Paths:
    data_dir: Path
    fig_dir: Path


def make_paths() -> Paths:
    data_dir = OUTDIR / "data"
    fig_dir = OUTDIR / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return Paths(data_dir=data_dir, fig_dir=fig_dir)


# ======================================================================
# GENERAL HELPERS
# ======================================================================


def unique_existing_files(patterns: Iterable[str], root: Path = INPUT_DATA_DIR) -> list[Path]:
    files: list[Path] = []
    seen = set()
    for pat in patterns:
        for p in root.glob(pat):
            if not p.is_file():
                continue
            name = str(p).lower()
            # Avoid wide/raw/temporary products and non-elevation derivatives.
            if any(bad in name for bad in [
                "raw_bbox", "bbox_layers", "_bbox", "dem_tiles",
                "slope", "aspect", "hillshade", "tri", "rugged", "roughness",
                "building", "height", "dsm", "chm",
            ]):
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                files.append(p)
    return files


def load_base_summary() -> dict:
    if BASE_SUMMARY_JSON.exists():
        try:
            return json.loads(BASE_SUMMARY_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def infer_spacing(values: pd.Series, default: float) -> float:
    vals = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    if vals.size < 2:
        return float(default)
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float(default)
    return float(np.nanmedian(diffs))


def get_utm_crs(summary: dict):
    if AOI_UTM_FILE.exists():
        try:
            aoi = gpd.read_file(AOI_UTM_FILE)
            if aoi.crs is not None:
                return aoi.crs
        except Exception:
            pass
    if "crs_utm" in summary:
        return summary["crs_utm"]
    return DEFAULT_UTM_CRS


def load_optional_outline(path: Path, target_crs, ref_x_sw_m: float, ref_y_sw_m: float) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        gdf = gdf.to_crs(target_crs)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)
        gdf["geometry"] = gdf.geometry.apply(lambda geom: translate(geom, xoff=-ref_x_sw_m, yoff=-ref_y_sw_m))
        return gdf
    except Exception as exc:
        print(f"[WARN] Could not read outline {path}: {exc}")
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)


# ======================================================================
# LOAD BASE VOXELS AND XY CELLS
# ======================================================================


def load_base_voxels() -> pd.DataFrame:
    if BASE_MODEL_PARQUET.exists():
        print(f"[OK] Reading base voxel parquet: {BASE_MODEL_PARQUET}")
        try:
            return pd.read_parquet(BASE_MODEL_PARQUET)
        except Exception as exc:
            print(f"[WARN] Could not read parquet, falling back to CSV: {exc}")

    if BASE_MODEL_CSV_GZ.exists():
        print(f"[OK] Reading base voxel CSV: {BASE_MODEL_CSV_GZ}")
        return pd.read_csv(BASE_MODEL_CSV_GZ)

    raise FileNotFoundError(
        "Base voxel model not found. Run 02_make_base_3d_voxel_box_model.py first.\n"
        f"Missing: {BASE_MODEL_PARQUET}\n"
        f"Missing: {BASE_MODEL_CSV_GZ}"
    )


def prepare_base_voxels(voxels: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, float, float, float, float]:
    required = ["ix", "iy", "iz", "x_utm_m", "y_utm_m"]
    for col in required:
        if col not in voxels.columns:
            raise ValueError(f"Base voxel model missing required column: {col}")

    voxels = voxels.copy()

    # Earlier base file may still call the vertical coordinate z_agl_m.
    # In this DEM-burn step, it is treated as absolute MSL height.
    if "z_center_msl_m" not in voxels.columns:
        if "z_agl_m" in voxels.columns:
            voxels["z_center_msl_m"] = pd.to_numeric(voxels["z_agl_m"], errors="coerce")
        elif "z_msl_m" in voxels.columns:
            voxels["z_center_msl_m"] = pd.to_numeric(voxels["z_msl_m"], errors="coerce")
        else:
            raise ValueError("Base voxel model must have z_agl_m, z_msl_m, or z_center_msl_m")

    for col in ["ix", "iy", "iz", "x_utm_m", "y_utm_m", "z_center_msl_m"]:
        voxels[col] = pd.to_numeric(voxels[col], errors="coerce")

    dx = float(summary.get("dx_m", infer_spacing(voxels["x_utm_m"], DEFAULT_DX_M)))
    dy = float(summary.get("dy_m", infer_spacing(voxels["y_utm_m"], DEFAULT_DY_M)))
    dz = float(summary.get("dz_m", infer_spacing(voxels["z_center_msl_m"], DEFAULT_DZ_M)))

    voxels["z_bottom_msl_m"] = voxels["z_center_msl_m"] - dz / 2.0
    voxels["z_top_msl_m"] = voxels["z_center_msl_m"] + dz / 2.0

    if "nofly" not in voxels.columns:
        if "flyable" in voxels.columns:
            voxels["nofly"] = 1 - pd.to_numeric(voxels["flyable"], errors="coerce").fillna(0).astype(int)
        else:
            voxels["nofly"] = 0
    if "flyable" not in voxels.columns:
        voxels["flyable"] = 1 - pd.to_numeric(voxels["nofly"], errors="coerce").fillna(0).astype(int)

    if "slowness" not in voxels.columns:
        voxels["slowness"] = np.where(voxels["flyable"].astype(int) == 1, 0.3, NOFLY_SLOWNESS)

    if FLYABLE_SLOWNESS_OVERRIDE is None:
        fly_slow = pd.to_numeric(voxels.loc[voxels["flyable"].astype(int) == 1, "slowness"], errors="coerce")
        flyable_slowness = float(fly_slow.median()) if fly_slow.notna().any() else 0.3
    else:
        flyable_slowness = float(FLYABLE_SLOWNESS_OVERRIDE)

    xy_pairs = voxels[["ix", "iy"]].drop_duplicates().sort_values(["ix", "iy"]).reset_index(drop=True)
    xy_pairs["xy_id"] = np.arange(len(xy_pairs), dtype=int)
    voxels = voxels.merge(xy_pairs, on=["ix", "iy"], how="left")

    return voxels, dx, dy, dz, flyable_slowness


def build_xy_cells(voxels: pd.DataFrame, dx: float, dy: float, utm_crs) -> gpd.GeoDataFrame:
    cols = ["xy_id", "ix", "iy", "x_utm_m", "y_utm_m"]
    optional = ["lon", "lat", "inside_polygon", "inside_buffer", "inside_data_box", "flyable", "nofly"]
    for col in optional:
        if col in voxels.columns:
            cols.append(col)

    xy = voxels[cols].drop_duplicates("xy_id").copy().reset_index(drop=True)
    geoms = [
        box(float(x) - dx / 2.0, float(y) - dy / 2.0, float(x) + dx / 2.0, float(y) + dy / 2.0)
        for x, y in zip(xy["x_utm_m"], xy["y_utm_m"])
    ]
    xy_gdf = gpd.GeoDataFrame(xy, geometry=geoms, crs=utm_crs)

    if "lon" not in xy_gdf.columns or "lat" not in xy_gdf.columns:
        centers_ll = xy_gdf.copy()
        centers_ll["geometry"] = centers_ll.geometry.centroid
        centers_ll = centers_ll.to_crs("EPSG:4326")
        xy_gdf["lon"] = centers_ll.geometry.x.to_numpy()
        xy_gdf["lat"] = centers_ll.geometry.y.to_numpy()

    return xy_gdf


def add_sw_reference_coordinates(
    xy_gdf: gpd.GeoDataFrame,
    voxels: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    """Add local distance coordinates measured from the southwest data-box corner."""
    xy_gdf = xy_gdf.copy()
    voxels = voxels.copy()

    minx, miny, maxx, maxy = xy_gdf.total_bounds
    ref_x_sw_m = float(minx)
    ref_y_sw_m = float(miny)

    xy_gdf["x_from_sw_m"] = xy_gdf["x_utm_m"].astype(float) - ref_x_sw_m
    xy_gdf["y_from_sw_m"] = xy_gdf["y_utm_m"].astype(float) - ref_y_sw_m
    xy_gdf["distance_from_sw_m"] = np.hypot(xy_gdf["x_from_sw_m"], xy_gdf["y_from_sw_m"])

    # Make a second local geometry for local SW-distance plotting.
    xy_gdf["geometry_utm"] = xy_gdf.geometry
    xy_gdf["geometry"] = xy_gdf.geometry.apply(lambda geom: translate(geom, xoff=-ref_x_sw_m, yoff=-ref_y_sw_m))

    maps = xy_gdf.set_index("xy_id")[["x_from_sw_m", "y_from_sw_m", "distance_from_sw_m"]]
    for col in maps.columns:
        voxels[col] = voxels["xy_id"].map(maps[col]).astype(float)

    ref = {
        "reference_name": "southwest_corner_of_xy_cell_data_box",
        "x_sw_corner_utm_m": ref_x_sw_m,
        "y_sw_corner_utm_m": ref_y_sw_m,
        "xmax_utm_m": float(maxx),
        "ymax_utm_m": float(maxy),
    }
    print(
        "[INFO] SW reference point: "
        f"X={ref_x_sw_m:.3f} m, Y={ref_y_sw_m:.3f} m. "
        "Local plot coordinates are distance east/north from this point."
    )
    return xy_gdf, voxels, ref


# ======================================================================
# DEM / TERRAIN SAMPLING
# ======================================================================


def raster_valid_values(src) -> np.ndarray:
    arr = src.read(1, masked=True).astype("float64")
    if src.nodata is not None:
        arr = np.ma.masked_where(np.isclose(arr, float(src.nodata)), arr)
    vals = arr.compressed()
    vals = vals[np.isfinite(vals)]
    return vals


def raster_stats(path: Path) -> dict:
    if rasterio is None:
        return {"path": str(path), "error": "rasterio_not_installed"}
    try:
        with rasterio.open(path) as src:
            vals = raster_valid_values(src)
            if vals.size == 0:
                return {
                    "path": str(path), "crs": str(src.crs), "width": src.width, "height": src.height,
                    "nodata": src.nodata, "valid_count": 0, "min": np.nan, "p1": np.nan,
                    "p50": np.nan, "p99": np.nan, "max": np.nan,
                }
            return {
                "path": str(path),
                "crs": str(src.crs),
                "width": int(src.width),
                "height": int(src.height),
                "nodata": src.nodata,
                "valid_count": int(vals.size),
                "min": float(np.nanmin(vals)),
                "p1": float(np.nanpercentile(vals, 1)),
                "p50": float(np.nanpercentile(vals, 50)),
                "p99": float(np.nanpercentile(vals, 99)),
                "max": float(np.nanmax(vals)),
            }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def dem_score(path: Path) -> tuple[int, str]:
    name = str(path).lower()
    s = 0
    # We already restrict to *elevation* files inside opentopography.
    # The score only decides between multiple valid elevation products.
    if "clipped" in name or "clip" in name:
        s -= 40
    if "hoalac" in name or "hoa_lac" in name or "study" in name:
        s -= 25
    if "cop30" in name:
        s -= 20
    if "elevation" in name:
        s -= 10
    for bad in ["slope", "aspect", "hillshade", "tri", "rugged", "roughness", "dsm", "chm", "height", "building"]:
        if bad in name:
            s += 500
    return s, str(path)


def choose_dem_raster(paths: Paths) -> Path | None:
    if DEM_FILE_OVERRIDE is not None:
        p = Path(DEM_FILE_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"DEM_FILE_OVERRIDE does not exist: {p}")
        print(f"[OK] DEM_FILE_OVERRIDE used: {p}")
        pd.DataFrame([raster_stats(p)]).to_csv(paths.data_dir / "dem_candidate_audit.csv", index=False)
        return p

    candidates = sorted(unique_existing_files(DEM_PATTERNS, root=OPENTOPOGRAPHY_DEM_DIR), key=dem_score)
    if not candidates:
        print(f"[WARN] No DEM raster candidates found in: {OPENTOPOGRAPHY_DEM_DIR}")
        return None

    audit_rows = []
    print("\n========== DEM RASTER CANDIDATES ==========")
    for i, p in enumerate(candidates):
        st = raster_stats(p)
        st["rank"] = i
        st["score"] = dem_score(p)[0]
        audit_rows.append(st)
        if "error" in st:
            print(f"[{i:02d}] score={st['score']:4d} ERROR | {p} | {st['error']}")
        else:
            print(
                f"[{i:02d}] score={st['score']:4d} "
                f"min={st['min']:.2f} p50={st['p50']:.2f} p99={st['p99']:.2f} max={st['max']:.2f} "
                f"| {p}"
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(paths.data_dir / "dem_candidate_audit.csv", index=False)

    # Choose the first readable candidate after scoring.
    for p, row in zip(candidates, audit_rows):
        if "error" not in row and int(row.get("valid_count", 0)) > 0:
            print(f"[OK] Selected DEM raster: {p}")
            return p

    print("[WARN] All DEM raster candidates failed.")
    return None


def sample_dem_raster_to_xy(xy_gdf: gpd.GeoDataFrame, raster_path: Path) -> tuple[pd.Series, dict]:
    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot sample DEM raster")

    print(f"\n[OK] Sampling DEM raster directly at XY cell centers: {raster_path}")
    with rasterio.open(raster_path) as src:
        raw_vals = raster_valid_values(src)
        if raw_vals.size == 0:
            raise RuntimeError(f"DEM raster has no valid values: {raster_path}")
        raw_min = float(np.nanmin(raw_vals))
        raw_max = float(np.nanmax(raw_vals))

        # Important: use UTM geometries for reprojection, because xy_gdf.geometry
        # may already be local SW geometry after add_sw_reference_coordinates().
        points = xy_gdf.copy()
        if "geometry_utm" in points.columns:
            points = points.set_geometry("geometry_utm", crs=xy_gdf.crs)
        if points.crs is None:
            points = points.set_crs(DEFAULT_UTM_CRS)
        points = points.to_crs(src.crs)

        centers = points.geometry.centroid
        coords = [(float(p.x), float(p.y)) for p in centers]

        vals = []
        nodata = src.nodata
        for val in src.sample(coords):
            v = float(val[0]) if len(val) else np.nan
            if nodata is not None and math.isclose(v, float(nodata), rel_tol=0, abs_tol=1e-8):
                v = np.nan
            vals.append(v)

    terrain = pd.Series(vals, index=xy_gdf.index, dtype="float64")
    finite = terrain[np.isfinite(terrain)]
    stats = {
        "source": str(raster_path),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "sample_count": int(len(terrain)),
        "sample_valid_count": int(finite.size),
        "sample_nan_count": int(terrain.isna().sum()),
        "sample_min_before_fill": float(finite.min()) if finite.size else np.nan,
        "sample_max_before_fill": float(finite.max()) if finite.size else np.nan,
    }

    if finite.size:
        bad = finite[(finite < raw_min - RASTER_RANGE_TOLERANCE_M) | (finite > raw_max + RASTER_RANGE_TOLERANCE_M)]
        if len(bad) > 0:
            raise RuntimeError(
                "DEM direct-sampling range check failed. "
                f"Raw raster range is {raw_min:.2f}..{raw_max:.2f} m, but sampled range is "
                f"{finite.min():.2f}..{finite.max():.2f} m. Do not continue until the DEM source/CRS is checked."
            )

    return terrain, stats


def xyz_stats(path: Path) -> dict:
    try:
        dem = pd.read_csv(path, sep=r"\s+|,", header=None, names=["lon", "lat", "value"], engine="python")
        dem = dem.dropna()
        vals = dem["value"].to_numpy(dtype=float) if not dem.empty else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"path": str(path), "valid_count": 0, "min": np.nan, "p50": np.nan, "p99": np.nan, "max": np.nan}
        return {
            "path": str(path),
            "valid_count": int(vals.size),
            "min": float(np.nanmin(vals)),
            "p1": float(np.nanpercentile(vals, 1)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p99": float(np.nanpercentile(vals, 99)),
            "max": float(np.nanmax(vals)),
        }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def choose_dem_xyz(paths: Paths) -> Path | None:
    candidates = sorted(unique_existing_files(DEM_XYZ_PATTERNS, root=OPENTOPOGRAPHY_DEM_DIR), key=dem_score)
    if not candidates:
        print(f"[WARN] No DEM XYZ candidates found in: {OPENTOPOGRAPHY_DEM_DIR}")
        return None

    audit_rows = []
    print("\n========== DEM XYZ CANDIDATES ==========")
    for i, p in enumerate(candidates):
        st = xyz_stats(p)
        st["rank"] = i
        st["score"] = dem_score(p)[0]
        audit_rows.append(st)
        if "error" in st:
            print(f"[{i:02d}] score={st['score']:4d} ERROR | {p} | {st['error']}")
        else:
            print(
                f"[{i:02d}] score={st['score']:4d} "
                f"min={st['min']:.2f} p50={st['p50']:.2f} p99={st['p99']:.2f} max={st['max']:.2f} "
                f"| {p}"
            )
    pd.DataFrame(audit_rows).to_csv(paths.data_dir / "dem_xyz_candidate_audit.csv", index=False)

    for p, row in zip(candidates, audit_rows):
        if "error" not in row and int(row.get("valid_count", 0)) > 0:
            print(f"[OK] Selected DEM XYZ: {p}")
            return p
    return None


def sample_dem_xyz_to_xy(xy_gdf: gpd.GeoDataFrame, xyz_path: Path) -> tuple[pd.Series, dict]:
    if cKDTree is None:
        raise RuntimeError("scipy is not installed, cannot use DEM XYZ fallback")

    print(f"\n[OK] Sampling DEM XYZ by nearest-neighbor: {xyz_path}")
    dem = pd.read_csv(xyz_path, sep=r"\s+|,", header=None, names=["lon", "lat", "value"], engine="python")
    dem = dem.dropna()
    if dem.empty:
        raise RuntimeError(f"DEM XYZ is empty: {xyz_path}")

    raw_vals = dem["value"].to_numpy(dtype=float)
    raw_min = float(np.nanmin(raw_vals))
    raw_max = float(np.nanmax(raw_vals))

    dem_gdf = gpd.GeoDataFrame(
        dem[["value"]].copy(),
        geometry=gpd.points_from_xy(dem["lon"], dem["lat"]),
        crs="EPSG:4326",
    ).to_crs(xy_gdf.crs)

    points = xy_gdf.copy()
    if "geometry_utm" in points.columns:
        points = points.set_geometry("geometry_utm", crs=xy_gdf.crs)
    centers = points.geometry.centroid

    tree = cKDTree(np.column_stack([dem_gdf.geometry.x.to_numpy(), dem_gdf.geometry.y.to_numpy()]))
    query = np.column_stack([centers.x.to_numpy(), centers.y.to_numpy()])
    _, idx = tree.query(query, k=1)
    vals = dem_gdf["value"].to_numpy(dtype=float)[idx]

    terrain = pd.Series(vals, index=xy_gdf.index, dtype="float64")
    stats = {
        "source": str(xyz_path),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "sample_count": int(len(terrain)),
        "sample_valid_count": int(np.isfinite(vals).sum()),
        "sample_nan_count": int(np.isnan(vals).sum()),
        "sample_min_before_fill": float(np.nanmin(vals)),
        "sample_max_before_fill": float(np.nanmax(vals)),
    }
    return terrain, stats


def add_terrain_to_xy(paths: Paths, xy_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str, dict]:
    xy_gdf = xy_gdf.copy()
    terrain = pd.Series(np.nan, index=xy_gdf.index, dtype="float64")
    source = "fallback_constant"
    stats: dict = {}

    if DEM_SOURCE_PRIORITY not in {"tif_then_xyz", "xyz_then_tif", "tif_only", "xyz_only"}:
        raise ValueError(
            'DEM_SOURCE_PRIORITY must be "tif_then_xyz", "xyz_then_tif", "tif_only", or "xyz_only"'
        )

    if DEM_SOURCE_PRIORITY == "tif_then_xyz":
        source_order = ["tif", "xyz"]
    elif DEM_SOURCE_PRIORITY == "xyz_then_tif":
        source_order = ["xyz", "tif"]
    elif DEM_SOURCE_PRIORITY == "tif_only":
        source_order = ["tif"]
    else:
        source_order = ["xyz"]

    print(f"[INFO] DEM search root: {OPENTOPOGRAPHY_DEM_DIR}")
    print(f"[INFO] DEM source priority: {DEM_SOURCE_PRIORITY}")

    for src_kind in source_order:
        if src_kind == "tif":
            raster_path = choose_dem_raster(paths)
            if raster_path is None:
                continue
            try:
                terrain, stats = sample_dem_raster_to_xy(xy_gdf, raster_path)
                source = str(raster_path)
                break
            except Exception as exc:
                print(f"[WARN] DEM raster sampling failed: {exc}")
                terrain = pd.Series(np.nan, index=xy_gdf.index, dtype="float64")
        elif src_kind == "xyz":
            xyz_path = choose_dem_xyz(paths)
            if xyz_path is None:
                continue
            try:
                terrain, stats = sample_dem_xyz_to_xy(xy_gdf, xyz_path)
                source = str(xyz_path)
                break
            except Exception as exc:
                print(f"[WARN] DEM XYZ sampling failed: {exc}")
                terrain = pd.Series(np.nan, index=xy_gdf.index, dtype="float64")

    if terrain.isna().all():
        print(f"[WARN] No usable DEM found. Terrain set to {DEFAULT_TERRAIN_MSL_M_IF_MISSING} m MSL.")
        terrain = pd.Series(DEFAULT_TERRAIN_MSL_M_IF_MISSING, index=xy_gdf.index, dtype="float64")
        source = f"constant_{DEFAULT_TERRAIN_MSL_M_IF_MISSING:g}_m_msl"
        stats = {"source": source}
    else:
        med = float(terrain.dropna().median())
        n_nan = int(terrain.isna().sum())
        if n_nan > 0:
            print(f"[WARN] DEM had {n_nan:,} NaN sampled cells; filling with median={med:.2f} m.")
        terrain = terrain.fillna(med)

    xy_gdf["terrain_msl_m"] = terrain.to_numpy(dtype=float)
    xy_gdf["terrain_source"] = source

    pct = np.nanpercentile(xy_gdf["terrain_msl_m"], [0, 1, 5, 50, 95, 99, 100])
    stats.update({
        "final_sample_min": float(pct[0]),
        "final_sample_p1": float(pct[1]),
        "final_sample_p5": float(pct[2]),
        "final_sample_p50": float(pct[3]),
        "final_sample_p95": float(pct[4]),
        "final_sample_p99": float(pct[5]),
        "final_sample_max": float(pct[6]),
    })

    print(
        "[CHECK] Final DEM terrain MSL sampled to XY cells: "
        f"min={pct[0]:.2f}, p50={pct[3]:.2f}, p99={pct[5]:.2f}, max={pct[6]:.2f} m | source={source}"
    )
    return xy_gdf, source, stats


# ======================================================================
# DEM BURN LOGIC
# ======================================================================


def add_dem_burn_columns(voxels: pd.DataFrame, xy_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    voxels = voxels.copy()

    terrain_map = xy_gdf.set_index("xy_id")["terrain_msl_m"]
    voxels["terrain_msl_m"] = voxels["xy_id"].map(terrain_map).astype(float)
    voxels["z_center_agl_m"] = voxels["z_center_msl_m"] - voxels["terrain_msl_m"]
    voxels["z_bottom_agl_m"] = voxels["z_bottom_msl_m"] - voxels["terrain_msl_m"]
    voxels["z_top_agl_m"] = voxels["z_top_msl_m"] - voxels["terrain_msl_m"]

    terrain_limit = voxels["terrain_msl_m"] + TERRAIN_CLEARANCE_BUFFER_M

    if not BURN_TERRAIN_UNDERGROUND:
        voxels["burn_dem_terrain"] = False
    elif TERRAIN_BURN_VERTICAL_RULE == "z_center_below_dem":
        voxels["burn_dem_terrain"] = voxels["z_center_msl_m"] < terrain_limit
    elif TERRAIN_BURN_VERTICAL_RULE == "z_top_below_dem":
        voxels["burn_dem_terrain"] = voxels["z_top_msl_m"] <= terrain_limit
    elif TERRAIN_BURN_VERTICAL_RULE == "z_bottom_below_dem":
        voxels["burn_dem_terrain"] = voxels["z_bottom_msl_m"] < terrain_limit
    else:
        raise ValueError(
            "TERRAIN_BURN_VERTICAL_RULE must be one of: "
            "z_bottom_below_dem, z_center_below_dem, z_top_below_dem"
        )

    voxels["burn_dem_terrain"] = voxels["burn_dem_terrain"].astype(bool)
    print(f"[CHECK] DEM terrain burned voxels: {int(voxels['burn_dem_terrain'].sum()):,}")
    return voxels


def finalize_dem_only_model(voxels: pd.DataFrame, flyable_slowness: float) -> pd.DataFrame:
    """
    Final DEM-only no-fly model.

    Important safety rule:
        final_nofly_dem_only = outside_polygon OR base_nofly OR burn_dem_terrain

    This means outside-AOI cells are treated as burned/no-fly in the output
    model and in all check figures.
    """
    voxels = voxels.copy()
    if "burn_dem_terrain" not in voxels.columns:
        voxels["burn_dem_terrain"] = False

    original_base_nofly = pd.to_numeric(
        voxels.get("nofly", 0),
        errors="coerce",
    ).fillna(0).astype(int) == 1

    if "inside_polygon" in voxels.columns:
        inside_polygon = pd.to_numeric(
            voxels["inside_polygon"],
            errors="coerce",
        ).fillna(0).astype(int) == 1
        outside_polygon = ~inside_polygon
    else:
        outside_polygon = pd.Series(False, index=voxels.index)

    voxels["burn_outside_polygon"] = outside_polygon.astype(bool)
    voxels["base_nofly_input"] = original_base_nofly.astype(bool)

    # Final safety/no-fly state used for pathfinding and for figures.
    voxels["final_nofly_dem_only"] = (
        voxels["base_nofly_input"]
        | voxels["burn_outside_polygon"]
        | voxels["burn_dem_terrain"].astype(bool)
    )
    voxels["final_flyable_dem_only"] = (~voxels["final_nofly_dem_only"]).astype(int)
    voxels["slowness_final_dem_only"] = np.where(
        voxels["final_nofly_dem_only"],
        NOFLY_SLOWNESS,
        flyable_slowness,
    )

    label = np.full(len(voxels), "flyable", dtype=object)
    label = np.where(voxels["base_nofly_input"], "nofly_base", label)
    label = np.where(voxels["burn_dem_terrain"], "nofly_dem_terrain", label)
    # Outside AOI wins as the displayed reason because it is a hard boundary.
    label = np.where(voxels["burn_outside_polygon"], "nofly_outside_polygon", label)
    voxels["label_final_dem_only"] = label

    print(f"[CHECK] Outside-polygon no-fly voxels: {int(voxels['burn_outside_polygon'].sum()):,}")
    print(f"[CHECK] Final DEM-only no-fly voxels: {int(voxels['final_nofly_dem_only'].sum()):,}")
    return voxels


# ======================================================================
# SAVE OUTPUTS
# ======================================================================


def save_outputs(
    paths: Paths,
    voxels: pd.DataFrame,
    xy_gdf: gpd.GeoDataFrame,
    terrain_source: str,
    terrain_stats: dict,
    sw_ref: dict,
    dx: float,
    dy: float,
    dz: float,
    flyable_slowness: float,
) -> None:
    data_dir = paths.data_dir

    main_csv = data_dir / "dem_only_voxel_model_50m.csv.gz"
    voxels.to_csv(main_csv, index=False, compression="gzip")
    print(f"[OK] Saved: {main_csv}")

    try:
        main_parquet = data_dir / "dem_only_voxel_model_50m.parquet"
        voxels.to_parquet(main_parquet, index=False)
        print(f"[OK] Saved: {main_parquet}")
    except Exception as exc:
        print(f"[WARN] Could not save parquet: {exc}")

    xyz_cols = ["lon", "lat", "z_center_msl_m", "slowness_final_dem_only", "label_final_dem_only"]
    if all(c in voxels.columns for c in xyz_cols):
        xyz = voxels[xyz_cols].copy()
        xyz.to_csv(
            data_dir / "dem_only_voxel_model_50m.xyz",
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )
        print(f"[OK] Saved: {data_dir / 'dem_only_voxel_model_50m.xyz'}")

    xy_out = pd.DataFrame(xy_gdf.drop(columns=["geometry", "geometry_utm"], errors="ignore"))
    xy_out.to_csv(data_dir / "xy_grid_with_dem_terrain_msl_SW.csv.gz", index=False, compression="gzip")

    # Save UTM geometry to GPKG, not local geometry, so GIS opens correctly.
    xy_save = xy_gdf.copy()
    if "geometry_utm" in xy_save.columns:
        xy_save = xy_save.set_geometry("geometry_utm", crs=xy_gdf.crs)
        xy_save = xy_save.drop(columns=["geometry"], errors="ignore")
        xy_save = xy_save.rename_geometry("geometry")
    xy_save.to_file(data_dir / "xy_grid_with_dem_terrain_msl_SW.gpkg", driver="GPKG")

    summary = {
        "terrain_source": terrain_source,
        "terrain_stats": terrain_stats,
        "sw_reference": sw_ref,
        "terrain_burn_vertical_rule": TERRAIN_BURN_VERTICAL_RULE,
        "voxel_vertical_reference": "MSL",
        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,
        "flyable_slowness": flyable_slowness,
        "nofly_slowness": NOFLY_SLOWNESS,
        "total_voxels": int(len(voxels)),
        "base_nofly_input_voxels": int(voxels.get("base_nofly_input", pd.Series(False, index=voxels.index)).sum()),
        "outside_polygon_nofly_voxels": int(voxels.get("burn_outside_polygon", pd.Series(False, index=voxels.index)).sum()),
        "dem_burned_voxels": int(voxels["burn_dem_terrain"].sum()),
        "final_flyable_dem_only_voxels": int(voxels["final_flyable_dem_only"].sum()),
        "final_nofly_dem_only_voxels": int(voxels["final_nofly_dem_only"].sum()),
    }
    (data_dir / "dem_terrain_burn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "DEM-ONLY TERRAIN BURN SUMMARY",
        "=" * 70,
        f"Base model dir             : {BASE_MODEL_DIR}",
        f"Input data dir             : {INPUT_DATA_DIR}",
        f"Terrain source             : {terrain_source}",
        f"Vertical reference         : MSL",
        f"SW reference UTM X         : {sw_ref['x_sw_corner_utm_m']:.3f} m",
        f"SW reference UTM Y         : {sw_ref['y_sw_corner_utm_m']:.3f} m",
        f"Voxel size                 : {dx:g} x {dy:g} x {dz:g} m",
        f"Terrain burn rule          : {TERRAIN_BURN_VERTICAL_RULE}",
        f"Terrain clearance buffer   : {TERRAIN_CLEARANCE_BUFFER_M:g} m",
        f"Flyable slowness           : {flyable_slowness:g} s/m",
        f"No-fly slowness            : {NOFLY_SLOWNESS:g} s/m",
        f"Total voxels               : {len(voxels):,}",
        f"Input/base no-fly voxels    : {int(voxels.get('base_nofly_input', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Outside-polygon no-fly     : {int(voxels.get('burn_outside_polygon', pd.Series(False, index=voxels.index)).sum()):,}",
        f"DEM-burned voxels          : {int(voxels['burn_dem_terrain'].sum()):,}",
        f"Final flyable voxels       : {int(voxels['final_flyable_dem_only'].sum()):,}",
        f"Final no-fly voxels        : {int(voxels['final_nofly_dem_only'].sum()):,}",
        "",
        "DEM sampled terrain statistics:",
    ]
    for key in ["raw_min", "raw_max", "sample_min_before_fill", "sample_max_before_fill", "final_sample_p50", "final_sample_p99", "final_sample_max"]:
        if key in terrain_stats:
            lines.append(f"  {key:28s}: {terrain_stats[key]}")
    lines += [
        "",
        "DEM burn equation:",
        "  terrain_msl = DEM(x, y)",
        "  burn_dem_terrain = z_bottom_msl < terrain_msl + clearance_buffer",
        "  burn_outside_polygon = inside_polygon != 1",
        "  final_nofly_dem_only = base_nofly OR burn_outside_polygon OR burn_dem_terrain",
        "  z_agl = z_msl - terrain_msl",
        "",
        "Plot coordinate equation:",
        "  x_from_sw_m = x_utm_m - x_sw_corner_utm_m",
        "  y_from_sw_m = y_utm_m - y_sw_corner_utm_m",
        "  distance_from_sw_m = sqrt(x_from_sw_m^2 + y_from_sw_m^2)",
    ]
    txt = "\n".join(lines)
    (data_dir / "dem_terrain_burn_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


# ======================================================================
# FIGURES
# ======================================================================


def plot_dem_terrain(paths: Paths, xy_gdf: gpd.GeoDataFrame, sw_ref: dict, utm_crs) -> None:
    out_png = paths.fig_dir / "00_dem_terrain_msl_cells_SW.png"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIG_DPI)
    xy_gdf.plot(
        ax=ax,
        column="terrain_msl_m",
        cmap="terrain",
        vmin=DEM_PLOT_VMIN_M,
        vmax=DEM_PLOT_VMAX_M,
        legend=True,
        linewidth=0.0,
        legend_kwds={"label": "DEM terrain elevation (m MSL)", "shrink": 0.78},
    )

    # Safety display: outside the operation polygon is hard no-fly.
    # Plot those XY cells in black on top of the DEM colors.
    outside = gpd.GeoDataFrame(geometry=[], crs=xy_gdf.crs)
    if "inside_polygon" in xy_gdf.columns:
        outside_mask = pd.to_numeric(
            xy_gdf["inside_polygon"],
            errors="coerce",
        ).fillna(0).astype(int) != 1
        outside = xy_gdf[outside_mask].copy()
    elif "nofly" in xy_gdf.columns:
        outside_mask = pd.to_numeric(
            xy_gdf["nofly"],
            errors="coerce",
        ).fillna(0).astype(int) == 1
        outside = xy_gdf[outside_mask].copy()

    if not outside.empty:
        outside.plot(ax=ax, color="black", linewidth=0.0, alpha=1.0)

    aoi_local = load_optional_outline(AOI_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    if not aoi_local.empty:
        aoi_local.boundary.plot(ax=ax, color="black", linewidth=1.3)

    data_box_local = load_optional_outline(DATA_BOX_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    if not data_box_local.empty:
        data_box_local.boundary.plot(ax=ax, color="gray", linewidth=0.7, linestyle="--")

    ax.scatter([0], [0], marker="*", s=90, color="black", zorder=5, label="SW reference")
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if not outside.empty:
        legend_handles.append(Patch(facecolor="black", edgecolor="black", label="Outside AOI / no-fly"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.text(
        0.02, 0.02,
        f"Display scale: {DEM_PLOT_VMIN_M:g}–{DEM_PLOT_VMAX_M:g} m MSL\n"
        "DEM values are not clipped for burning",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
    )
    ax.set_title("DEM terrain sampled to XY voxel cells", fontweight="bold")
    ax.set_xlabel("Distance east from SW reference (m)")
    ax.set_ylabel("Distance north from SW reference (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_dem_burn_z_slices(paths: Paths, voxels: pd.DataFrame) -> None:
    out_png = paths.fig_dir / "01_dem_terrain_burn_z_slices_SW.png"
    requested_levels = list(dict.fromkeys([float(z) for z in SLICE_Z_LEVELS_MSL]))
    if not requested_levels:
        return

    n = len(requested_levels)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), dpi=FIG_DPI)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, z_req in zip(axes, requested_levels):
        sub = voxels[(voxels["z_bottom_msl_m"] <= z_req) & (voxels["z_top_msl_m"] > z_req)].copy()
        if sub.empty:
            z_unique = np.sort(pd.to_numeric(voxels["z_center_msl_m"], errors="coerce").dropna().unique())
            if z_unique.size == 0:
                ax.axis("off")
                continue
            z_near = float(z_unique[np.argmin(np.abs(z_unique - z_req))])
            sub = voxels[np.isclose(voxels["z_center_msl_m"], z_near)].copy()
            title = f"requested z = {z_req:.1f} m\nnearest center = {z_near:.1f} m"
        else:
            title = f"z = {z_req:.1f} m MSL"

        piv = sub.pivot_table(
            index="y_from_sw_m",
            columns="x_from_sw_m",
            values="final_nofly_dem_only",
            aggfunc="max",
        )
        piv = piv.sort_index(ascending=True)
        arr = piv.to_numpy(dtype=float)
        extent = [piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()]
        im = ax.imshow(arr, extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("East from SW (m)")
        ax.set_ylabel("North from SW (m)")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[n:]:
        ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:n], shrink=0.72, pad=0.02)
        cbar.set_label("Final no-fly: 1 = no-fly (terrain or outside polygon), 0 = flyable")

    fig.suptitle("DEM terrain + outside-polygon no-fly: Z-slice check", fontweight="bold")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def select_complete_dem_columns_for_3d(voxels: pd.DataFrame) -> pd.DataFrame:
    burn_df = voxels[voxels["final_nofly_dem_only"].astype(bool)].copy()
    if burn_df.empty:
        return burn_df

    rng = np.random.default_rng(RANDOM_SEED)
    xy_ids = np.sort(burn_df["xy_id"].dropna().astype(int).unique())

    if xy_ids.size > MAX_3D_DEM_XY_COLUMNS:
        # Keep a reproducible spatially distributed subset by using random selection
        # of complete columns, never random individual voxels.
        xy_ids = np.sort(rng.choice(xy_ids, size=MAX_3D_DEM_XY_COLUMNS, replace=False))

    out = burn_df[burn_df["xy_id"].isin(xy_ids)].copy()

    while len(out) > MAX_3D_DEM_VOXELS_TO_RENDER and len(xy_ids) > 10:
        keep_count = max(10, int(len(xy_ids) * 0.85))
        xy_ids = np.sort(rng.choice(xy_ids, size=keep_count, replace=False))
        out = burn_df[burn_df["xy_id"].isin(xy_ids)].copy()

    print(
        f"[INFO] 3D DEM figure renders {len(out):,} burned voxels "
        f"from {len(xy_ids):,} complete XY columns. The output model contains all burned cells."
    )
    return out


def voxel_block_faces_from_df(
    df: pd.DataFrame,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float, float, float]]]:
    if df.empty:
        return [], []

    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    offsets = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=float)
    face_ids = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    rgba = (0.02, 0.02, 0.02, 0.48)
    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []
    xyz = df[["x_from_sw_m", "y_from_sw_m", "z_center_msl_m"]].to_numpy(dtype=float)

    for center in xyz:
        vertices = offsets + center.reshape(1, 3)
        for ids in face_ids:
            faces.append([(float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])) for i in ids])
            facecolors.append(rgba)
    return faces, facecolors


def plot_3d_dem_burn(paths: Paths, voxels: pd.DataFrame, xy_gdf: gpd.GeoDataFrame, dx: float, dy: float, dz: float) -> None:
    out_png = paths.fig_dir / "02_3d_dem_burned_cells_SW.png"

    plot_df = select_complete_dem_columns_for_3d(voxels)

    fig = plt.figure(figsize=(12, 9.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    if not plot_df.empty:
        faces, facecolors = voxel_block_faces_from_df(plot_df, dx, dy, dz)
        if faces:
            pc = Poly3DCollection(
                faces,
                facecolors=facecolors,
                edgecolors=(1.0, 1.0, 1.0, 0.12),
                linewidths=0.06,
                antialiased=True,
            )
            ax.add_collection3d(pc)

        # Overlay the DEM surface at every XY cell center as a topographic reference.
        # Inside AOI is colored by DEM elevation. Outside AOI is black because it is no-fly.
        if "inside_polygon" in xy_gdf.columns:
            inside_mask = pd.to_numeric(
                xy_gdf["inside_polygon"],
                errors="coerce",
            ).fillna(0).astype(int) == 1
        else:
            inside_mask = pd.Series(True, index=xy_gdf.index)

        inside_xy = xy_gdf[inside_mask].copy()
        outside_xy = xy_gdf[~inside_mask].copy()

        sc = ax.scatter(
            inside_xy["x_from_sw_m"],
            inside_xy["y_from_sw_m"],
            inside_xy["terrain_msl_m"],
            c=inside_xy["terrain_msl_m"],
            cmap="terrain",
            vmin=DEM_PLOT_VMIN_M,
            vmax=DEM_PLOT_VMAX_M,
            s=1.6,
            alpha=0.82,
            depthshade=False,
        )
        if not outside_xy.empty:
            ax.scatter(
                outside_xy["x_from_sw_m"],
                outside_xy["y_from_sw_m"],
                outside_xy["terrain_msl_m"],
                c="black",
                s=1.6,
                alpha=0.90,
                depthshade=False,
            )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08)
        cbar.set_label(f"DEM surface elevation inside AOI (m MSL), display {DEM_PLOT_VMIN_M:g}–{DEM_PLOT_VMAX_M:g}")

        ax.set_xlim(float(xy_gdf["x_from_sw_m"].min() - dx), float(xy_gdf["x_from_sw_m"].max() + dx))
        ax.set_ylim(float(xy_gdf["y_from_sw_m"].min() - dy), float(xy_gdf["y_from_sw_m"].max() + dy))
        zmin = min(0.0, float(voxels["z_bottom_msl_m"].min()))
        # Display only up to 40 m MSL so isolated high COP30/DSM pixels do not stretch the 3D view.
        # This is figure-only; the saved DEM burn model still uses the original sampled DEM.
        ax.set_zlim(zmin, DEM_3D_ZMAX_PLOT_M)

    ax.legend(handles=[Patch(facecolor=(0.45, 0.45, 0.45, 0.45), edgecolor="gray", label="Final no-fly voxel cells")], loc="upper left", fontsize=8)
    ax.set_title("3D DEM terrain + outside-polygon no-fly", fontweight="bold")
    ax.set_xlabel("Distance east from SW reference (m)")
    ax.set_ylabel("Distance north from SW reference (m)")
    ax.set_zlabel("Z MSL (m)")
    ax.view_init(elev=27, azim=-48)

    try:
        xr = ax.get_xlim3d()[1] - ax.get_xlim3d()[0]
        yr = ax.get_ylim3d()[1] - ax.get_ylim3d()[0]
        zr = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ax.set_box_aspect((xr, yr, zr * 16.0))
    except Exception:
        pass

    note = (
        "Gray filled blocks = final no-fly cells (terrain burn or outside polygon)\n"
        f"Colored DEM surface display is clipped to {DEM_PLOT_VMAX_M:g} m MSL\n"
        "3D figure samples complete XY columns; data output keeps all no-fly cells"
    )
    ax.text2D(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
              bbox=dict(facecolor="white", edgecolor="gray", alpha=0.88))

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def make_figures(paths: Paths, xy_gdf: gpd.GeoDataFrame, voxels: pd.DataFrame, sw_ref: dict, utm_crs, dx: float, dy: float, dz: float) -> None:
    plot_dem_terrain(paths, xy_gdf, sw_ref, utm_crs)
    plot_dem_burn_z_slices(paths, voxels)
    plot_3d_dem_burn(paths, voxels, xy_gdf, dx, dy, dz)


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:
    print("\n========== DEM-ONLY TERRAIN BURN INTO 3D VOXEL MODEL ==========")
    paths = make_paths()

    summary = load_base_summary()
    utm_crs = get_utm_crs(summary)
    print(f"[INFO] UTM CRS: {utm_crs}")

    voxels_raw = load_base_voxels()
    voxels, dx, dy, dz, flyable_slowness = prepare_base_voxels(voxels_raw, summary)
    xy_gdf = build_xy_cells(voxels, dx, dy, utm_crs)
    xy_gdf, voxels, sw_ref = add_sw_reference_coordinates(xy_gdf, voxels)

    print(f"[INFO] Voxel size inferred: dx={dx:g}, dy={dy:g}, dz={dz:g} m")
    print(f"[INFO] Base voxels: {len(voxels):,}; XY cells: {len(xy_gdf):,}")

    xy_gdf, terrain_source, terrain_stats = add_terrain_to_xy(paths, xy_gdf)
    voxels = add_dem_burn_columns(voxels, xy_gdf)
    voxels = finalize_dem_only_model(voxels, flyable_slowness)

    save_outputs(paths, voxels, xy_gdf, terrain_source, terrain_stats, sw_ref, dx, dy, dz, flyable_slowness)
    make_figures(paths, xy_gdf, voxels, sw_ref, utm_crs, dx, dy, dz)

    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")
    print(f"Use this DEM-only model for checking: {paths.data_dir / 'dem_only_voxel_model_50m.csv.gz'}")
    print("Use column: slowness_final_dem_only")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
