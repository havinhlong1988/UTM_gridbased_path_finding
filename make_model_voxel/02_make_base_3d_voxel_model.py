#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a BASE 3D voxel data box for LAE-UTM / Hoa Lac scenario 1.

Purpose
-------
Create a full rectangular 3D data volume that covers the study polygon and its
buffer, similar to a regular 3D data box / voxel volume:

    - The whole rectangular XY data box is filled by regular cells.
    - Cell centers inside the original polygon are marked flyable.
    - Cell centers outside the original polygon are marked no-fly.
    - A polygon buffer is still saved and reported.
    - Z is built from 0 to 3000 m AGL with 50 m voxel spacing by default.

This is only the BASE airspace domain. Later scripts can burn in DEM terrain,
GBA buildings, OSM/OIM powerlines, towers/poles, RA/no-fly zones, etc.

Recommended location:
    make_model/02_make_base_3d_voxel_box_model.py

Run from make_model/:
    python 02_make_base_3d_voxel_box_model.py
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate
from shapely.prepared import prep
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# ======================================================================
# USER PARAMETERS
# ======================================================================

OUTDIR = Path("output/02_base_3d_voxel_box_model_senario1")

# Optional AOI file. If none exists, the hardcoded Hoa Lac polygon is used.
AOI_FILE_CANDIDATES = [
    Path("input/data_senario1/metadata/study_area_aoi.gpkg"),
    Path("input/data_senario1/metadata/study_area_aoi.geojson"),
    Path("input/data_senario1/osm/hoalac_polygon.gpkg"),
    Path("input/data_senario1/opentopography/hoalac_polygon.gpkg"),
    Path("../downloaddata/output/01_HoaLac_studies_area/metadata/study_area_aoi.gpkg"),
    Path("../downloaddata/output/01_HoaLac_studies_area/osm/hoalac_polygon.gpkg"),
    Path("../downloaddata/output/01_HoaLac_studies_area/opentopography/hoalac_polygon.gpkg"),
]

# Fallback Hoa Lac polygon, lon/lat.
HOALAC_POLYGON = [
    (105.5035, 21.0145),
    (105.5125, 20.9935),
    (105.5310, 20.9815),
    (105.5565, 20.9845),
    (105.5735, 20.9985),
    (105.5705, 21.0190),
    (105.5480, 21.0285),
    (105.5205, 21.0270),
    (105.5035, 21.0145),
]

# Regular voxel size.
DX_M = 50.0
DY_M = 50.0
DZ_M = 5.0

# Vertical model extent in meters AGL.
Z_MIN_M = 0.0
Z_MAX_M = 130.0 # current plan for 120m

# Optional custom Z center array. If None, uniform 50 m centers are used.
# Example:
# CUSTOM_Z_LEVELS_M = [25, 75, 125, 175, 225, 275]
CUSTOM_Z_LEVELS_M = None

# Buffer distance around polygon. The data box is the rectangular bounding box
# of polygon.buffer(BUFFER_M), not the polygon shape itself.
BUFFER_M = 50.0

# Domain mode:
#   "bbox"          : full rectangular box covering buffered polygon. Recommended.
#   "buffer_shape"  : keep only cells inside polygon buffer shape.
# User requested the box-like volume, so keep bbox.
DOMAIN_MODE = "bbox"

# Pathfinding compatibility.
# Project convention: slowness >= 10.0 means no-fly; slowness < 10.0 means flyable.
FLYABLE_VALUE = 1
NOFLY_VALUE = 0
FLYABLE_SLOWNESS = 0.3 # s/m ~ 12 km/h
NOFLY_SLOWNESS = 10.0 # s/m

# Output controls.
SAVE_FULL_3D_CSV_GZ = True
SAVE_FULL_3D_PARQUET = True
SAVE_XYZ = True
SAVE_2D_GRID = True

# Figure controls.
FIG_DPI = 220
MAX_3D_PLOT_POINTS_PER_CLASS = 25000
RANDOM_SEED = 42

# Small cube/node explanatory plot controls.
BOX_DEMO_NX = 6
BOX_DEMO_NY = 4
BOX_DEMO_NZ = 4

# Actual 3D QC plot controls.
# The right panel is rendered as translucent voxel blocks, like a 3D data box.
# To keep the figure fast/readable, the full 50 m model is aggregated to a
# coarser plotting grid when necessary. The saved model itself is still 50 m.
MAX_ACTUAL_VOXEL_CUBES_TO_RENDER = 1200
PLOT_CUBE_ALPHA_FLYABLE = 0.34
PLOT_CUBE_ALPHA_NOFLY = 0.18
PLOT_CUBE_EDGE_ALPHA = 0.50
PLOT_CUBE_EDGE_LINEWIDTH = 0.20
PLOT_SHOW_CENTER_NODES = True
PLOT_CENTER_NODE_SIZE = 2.0

# Plot visibility for outside/no-fly voxels.
# The model still keeps outside cells as no-fly; these only control figures.
# Set these False to make the outside region uncolored/transparent so the
# flyable region inside the polygon is easier to see.
PLOT_NOFLY_2D_CELLS = False
PLOT_NOFLY_3D_BLOCKS = False
PLOT_NOFLY_CENTER_NODES = False

# Coordinate display for figures.
# UTM coordinates are absolute meter coordinates, so Hoa Lac appears around
# X ~ 552000 m and Y ~ 2320000 m in EPSG:32648. For readable figures, use
# local plot coordinates where the southwest corner of the voxel data box is
# defined as (0, 0). The model still stores both UTM and local coordinates.
USE_LOCAL_PLOT_COORDS = True
LOCAL_ORIGIN_MODE = "data_box_southwest_corner"

# Axis triad display for the actual 3D panel.
# "screen_inset" draws a simple publication-style X/Y/Z triad in 2D screen
# space so it does not get visually distorted by the Matplotlib 3D camera.
# "data_quiver" draws true data-coordinate arrows from the local origin.
# "none" disables the extra triad and keeps only normal axis labels.
PLOT_AXIS_TRIAD_MODE = "screen_inset"

# Visual-only vertical exaggeration for the actual 3D voxel panel.
# This does NOT change the saved model Z values. Increase if the model looks too flat.
RIGHT_PANEL_Z_EXAGGERATION = 14.0


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
# AOI / GEOMETRY HELPERS
# ======================================================================

def find_aoi_file() -> Path | None:
    for p in AOI_FILE_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def load_aoi_polygon() -> gpd.GeoDataFrame:
    aoi_file = find_aoi_file()
    if aoi_file is not None:
        print(f"[OK] Reading AOI polygon: {aoi_file}")
        gdf = gpd.read_file(aoi_file)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
        if gdf.empty:
            raise RuntimeError(f"AOI file has no valid geometry: {aoi_file}")
        gdf = gdf.to_crs("EPSG:4326")
        geom = gdf.geometry.unary_union
        if geom.geom_type not in ["Polygon", "MultiPolygon"]:
            geom = geom.convex_hull
        return gpd.GeoDataFrame({"name": ["Hoa_Lac_AOI"]}, geometry=[geom], crs="EPSG:4326")

    print("[WARN] AOI file not found. Using hardcoded HOALAC_POLYGON.")
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return gpd.GeoDataFrame({"name": ["Hoa_Lac_AOI_hardcoded"]}, geometry=[poly], crs="EPSG:4326")


def estimate_utm_crs(gdf: gpd.GeoDataFrame):
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        # Hoa Lac / Hanoi is UTM zone 48N.
        return "EPSG:32648"


def make_buffer_and_box(aoi_utm: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    aoi_geom = aoi_utm.geometry.unary_union
    buffer_geom = aoi_geom.buffer(BUFFER_M)

    minx, miny, maxx, maxy = buffer_geom.bounds
    data_box_geom = box(minx, miny, maxx, maxy)

    buffer_gdf = gpd.GeoDataFrame(
        {"name": [f"AOI_buffer_{BUFFER_M:g}m"]},
        geometry=[buffer_geom],
        crs=aoi_utm.crs,
    )
    data_box_gdf = gpd.GeoDataFrame(
        {"name": ["rectangular_voxel_data_box"]},
        geometry=[data_box_geom],
        crs=aoi_utm.crs,
    )
    return buffer_gdf, data_box_gdf


def make_z_levels() -> np.ndarray:
    if CUSTOM_Z_LEVELS_M is not None:
        z = np.asarray(CUSTOM_Z_LEVELS_M, dtype=float)
        z = z[np.isfinite(z)]
        z = z[(z >= Z_MIN_M) & (z <= Z_MAX_M)]
        z = np.unique(np.sort(z))
        if z.size == 0:
            raise ValueError("CUSTOM_Z_LEVELS_M has no valid levels inside Z range.")
        return z

    # Cell centers: 25, 75, ..., 2975 for 0..3000 and dz=50.
    return np.arange(Z_MIN_M + DZ_M / 2.0, Z_MAX_M, DZ_M, dtype=float)


# ======================================================================
# GRID BUILDERS
# ======================================================================

def aligned_centers(min_value: float, max_value: float, step: float) -> np.ndarray:
    """Create grid-cell centers aligned to step spacing."""
    start = math.floor(min_value / step) * step + step / 2.0
    stop = math.ceil(max_value / step) * step
    return np.arange(start, stop, step, dtype=float)


def make_xy_grid_box(
    aoi_utm: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    data_box_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Build a regular rectangular XY data box.

    Classification uses cell centers:
        inside original polygon -> flyable
        outside original polygon -> no-fly

    Additional flags:
        inside_buffer = inside polygon.buffer(BUFFER_M)
        inside_data_box = all generated cells are inside data box
    """
    aoi_geom = aoi_utm.geometry.unary_union
    buffer_geom = buffer_gdf.geometry.unary_union
    data_box_geom = data_box_gdf.geometry.unary_union

    minx, miny, maxx, maxy = data_box_geom.bounds
    xs = aligned_centers(minx, maxx, DX_M)
    ys = aligned_centers(miny, maxy, DY_M)

    prep_aoi = prep(aoi_geom)
    prep_buffer = prep(buffer_geom)

    records = []
    geoms = []

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            pt = Point(float(x), float(y))

            if DOMAIN_MODE == "buffer_shape" and not prep_buffer.contains(pt):
                continue

            inside_polygon = bool(prep_aoi.contains(pt) or prep_aoi.touches(pt))
            inside_buffer = bool(prep_buffer.contains(pt) or prep_buffer.touches(pt))
            flyable = inside_polygon

            if flyable:
                label = "flyable"
            elif inside_buffer:
                label = "nofly"
            else:
                label = "nofly_1"

            records.append({
                "ix": int(ix),
                "iy": int(iy),
                "x_utm_m": float(x),
                "y_utm_m": float(y),
                "x_local_m": float(x - minx),
                "y_local_m": float(y - miny),
                "inside_polygon": int(inside_polygon),
                "inside_buffer": int(inside_buffer),
                "inside_data_box": 1,
                "flyable_2d": int(flyable),
                "nofly_2d": int(not flyable),
                "slowness_2d": float(FLYABLE_SLOWNESS if flyable else NOFLY_SLOWNESS),
                "label_2d": label,
            })
            geoms.append(pt)

    grid = gpd.GeoDataFrame(records, geometry=geoms, crs=aoi_utm.crs)
    if grid.empty:
        raise RuntimeError("No XY grid cells were created. Check polygon / buffer / grid size.")

    grid_ll = grid.to_crs("EPSG:4326")
    grid["lon"] = grid_ll.geometry.x.to_numpy()
    grid["lat"] = grid_ll.geometry.y.to_numpy()
    return grid


def build_3d_voxel_table(xy_grid: gpd.GeoDataFrame, z_levels: np.ndarray) -> pd.DataFrame:
    base = xy_grid.drop(columns="geometry").copy().reset_index(drop=True)
    nxy = len(base)
    nz = len(z_levels)

    print(f"[INFO] XY cells: {nxy:,}")
    print(f"[INFO] Z levels: {nz:,}")
    print(f"[INFO] Total voxels: {nxy * nz:,}")

    repeated_arr = np.repeat(base.to_numpy(), nz, axis=0)
    repeated = pd.DataFrame(repeated_arr, columns=base.columns)

    numeric_cols = [
        "ix", "iy", "x_utm_m", "y_utm_m", "x_local_m", "y_local_m",
        "inside_polygon", "inside_buffer",
        "inside_data_box", "flyable_2d", "nofly_2d", "slowness_2d", "lon", "lat",
    ]
    for col in numeric_cols:
        if col in repeated.columns:
            repeated[col] = pd.to_numeric(repeated[col], errors="coerce")

    repeated["iz"] = np.tile(np.arange(nz, dtype=int), nxy)
    repeated["z_agl_m"] = np.tile(z_levels, nxy)

    repeated["flyable"] = repeated["flyable_2d"].astype(int)
    repeated["nofly"] = repeated["nofly_2d"].astype(int)
    repeated["voxel_value"] = np.where(repeated["flyable"] == 1, FLYABLE_VALUE, NOFLY_VALUE).astype(int)
    repeated["slowness"] = np.where(repeated["flyable"] == 1, FLYABLE_SLOWNESS, NOFLY_SLOWNESS).astype(float)
    repeated["label"] = repeated["label_2d"].astype(str)

    cols = [
        "ix", "iy", "iz",
        "lon", "lat", "x_utm_m", "y_utm_m", "x_local_m", "y_local_m", "z_agl_m",
        "inside_polygon", "inside_buffer", "inside_data_box",
        "flyable", "nofly", "voxel_value", "slowness", "label",
    ]
    return repeated[cols]


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

def save_outputs(
    paths: Paths,
    aoi_gdf: gpd.GeoDataFrame,
    aoi_utm: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    data_box_gdf: gpd.GeoDataFrame,
    xy_grid: gpd.GeoDataFrame,
    z_levels: np.ndarray,
    voxels: pd.DataFrame,
) -> None:
    data_dir = paths.data_dir

    # Geometry outputs.
    aoi_gdf.to_file(data_dir / "aoi_polygon_wgs84.gpkg", driver="GPKG")
    aoi_utm.to_file(data_dir / "aoi_polygon_utm.gpkg", driver="GPKG")
    buffer_gdf.to_file(data_dir / "aoi_buffer_utm.gpkg", driver="GPKG")
    buffer_gdf.to_crs("EPSG:4326").to_file(data_dir / "aoi_buffer_wgs84.gpkg", driver="GPKG")
    data_box_gdf.to_file(data_dir / "voxel_data_box_utm.gpkg", driver="GPKG")
    data_box_gdf.to_crs("EPSG:4326").to_file(data_dir / "voxel_data_box_wgs84.gpkg", driver="GPKG")

    # 2D grid outputs.
    if SAVE_2D_GRID:
        xy_grid.drop(columns="geometry").to_csv(
            data_dir / "base_xy_data_box_grid_50m.csv.gz",
            index=False,
            compression="gzip",
        )
        xy_grid.to_file(data_dir / "base_xy_data_box_grid_50m.gpkg", driver="GPKG")

    # 3D model outputs.
    if SAVE_FULL_3D_CSV_GZ:
        voxels.to_csv(
            data_dir / "base_3d_voxel_data_box_50m.csv.gz",
            index=False,
            compression="gzip",
        )

    if SAVE_FULL_3D_PARQUET:
        try:
            voxels.to_parquet(data_dir / "base_3d_voxel_data_box_50m.parquet", index=False)
        except Exception as exc:
            print(f"[WARN] Could not save parquet. Install pyarrow or fastparquet. Reason: {exc}")

    if SAVE_XYZ:
        xyz = voxels[["lon", "lat", "z_agl_m", "slowness", "label"]].copy()
        xyz.to_csv(
            data_dir / "base_3d_voxel_data_box_50m.xyz",
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )

    # Summary.
    fly_xy = int((xy_grid["flyable_2d"] == 1).sum())
    nofly_xy = int((xy_grid["nofly_2d"] == 1).sum())
    fly_3d = int((voxels["flyable"] == 1).sum())
    nofly_3d = int((voxels["nofly"] == 1).sum())

    summary = {
        "domain_mode": DOMAIN_MODE,
        "rule": "full rectangular voxel data box; inside polygon = flyable; outside polygon = no-fly",
        "dx_m": DX_M,
        "dy_m": DY_M,
        "dz_m": DZ_M if CUSTOM_Z_LEVELS_M is None else None,
        "z_min_m": Z_MIN_M,
        "z_max_m": Z_MAX_M,
        "z_level_count": int(len(z_levels)),
        "z_levels_m": [float(z) for z in z_levels],
        "buffer_m": BUFFER_M,
        "xy_cell_count": int(len(xy_grid)),
        "xy_flyable_count": fly_xy,
        "xy_nofly_count": nofly_xy,
        "voxel_count": int(len(voxels)),
        "voxel_flyable_count": fly_3d,
        "voxel_nofly_count": nofly_3d,
        "flyable_slowness": FLYABLE_SLOWNESS,
        "nofly_slowness": NOFLY_SLOWNESS,
        "crs_utm": str(aoi_utm.crs),
        "x_y_plot_coordinate_mode": "local_from_data_box_origin" if USE_LOCAL_PLOT_COORDS else "utm_absolute",
        "local_origin_note": "x_local_m = x_utm_m - data_box_minx; y_local_m = y_utm_m - data_box_miny",
    }
    (data_dir / "base_3d_voxel_data_box_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    txt = [
        "BASE 3D VOXEL DATA BOX SUMMARY",
        "=" * 70,
        f"Output dir              : {OUTDIR}",
        f"UTM CRS                 : {aoi_utm.crs}",
        f"Domain mode             : {DOMAIN_MODE}",
        f"Plot XY coordinate mode  : {'local origin at data-box SW corner' if USE_LOCAL_PLOT_COORDS else 'absolute UTM'}",
        f"Grid size               : {DX_M:g} x {DY_M:g} x {DZ_M:g} m",
        f"Z range                 : {Z_MIN_M:g} to {Z_MAX_M:g} m AGL",
        f"Buffer                  : {BUFFER_M:g} m",
        f"XY cell count            : {len(xy_grid):,}",
        f"XY flyable cells         : {fly_xy:,}",
        f"XY no-fly cells          : {nofly_xy:,}",
        f"Z level count            : {len(z_levels):,}",
        f"Total voxel count        : {len(voxels):,}",
        f"Flyable voxel count      : {fly_3d:,}",
        f"No-fly voxel count       : {nofly_3d:,}",
        "",
        "Rule:",
        "  Full rectangular data box covers polygon.buffer(BUFFER_M).bounds.",
        "  Voxel centers inside original polygon are flyable.",
        "  Voxel centers outside original polygon are no-fly.",
        "  Buffer polygon is saved for reference and later constraints.",
    ]
    (data_dir / "base_3d_voxel_data_box_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))


# ======================================================================
# PLOTTING HELPERS
# ======================================================================

def set_axes_equal_3d(ax) -> None:
    """Make 3D axes roughly equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])



def get_local_origin_from_xy_grid(xy_grid: gpd.GeoDataFrame) -> tuple[float, float]:
    """Return UTM origin used for local plotting coordinates."""
    if "x_local_m" in xy_grid.columns and "y_local_m" in xy_grid.columns:
        origin_x = float(pd.to_numeric(xy_grid["x_utm_m"], errors="coerce").min() - pd.to_numeric(xy_grid["x_local_m"], errors="coerce").min())
        origin_y = float(pd.to_numeric(xy_grid["y_utm_m"], errors="coerce").min() - pd.to_numeric(xy_grid["y_local_m"], errors="coerce").min())
        return origin_x, origin_y
    return float(xy_grid["x_utm_m"].min()), float(xy_grid["y_utm_m"].min())


def localize_geometry_gdf(gdf: gpd.GeoDataFrame, origin_x: float, origin_y: float) -> gpd.GeoDataFrame:
    """Translate UTM geometry so the local plot origin becomes (0, 0)."""
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: translate(geom, xoff=-origin_x, yoff=-origin_y) if geom is not None else geom)
    return out


def make_local_xy_plot_gdf(xy_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a point GeoDataFrame using local x/y coordinates for plotting."""
    out = xy_grid.drop(columns="geometry").copy()
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x_local_m"], out["y_local_m"]),
        crs=xy_grid.crs,
    )


def plot_xy_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Return x/y columns and labels for plotting."""
    if USE_LOCAL_PLOT_COORDS and {"x_local_m", "y_local_m"}.issubset(df.columns):
        return (
            "x_local_m",
            "y_local_m",
            "Local X from data-box origin (m)",
            "Local Y from data-box origin (m)",
        )
    return "x_utm_m", "y_utm_m", "UTM X / Easting (m)", "UTM Y / Northing (m)"


def plot_xy_data_box(
    paths: Paths,
    aoi_utm: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    data_box_gdf: gpd.GeoDataFrame,
    xy_grid: gpd.GeoDataFrame,
) -> None:
    out_png = paths.fig_dir / "00_xy_full_data_box_flyable_nofly.png"

    fig, ax = plt.subplots(figsize=(10, 9), dpi=FIG_DPI)

    if USE_LOCAL_PLOT_COORDS:
        origin_x, origin_y = get_local_origin_from_xy_grid(xy_grid)
        plot_data_box = localize_geometry_gdf(data_box_gdf, origin_x, origin_y)
        plot_buffer = localize_geometry_gdf(buffer_gdf, origin_x, origin_y)
        plot_aoi = localize_geometry_gdf(aoi_utm, origin_x, origin_y)
        plot_grid = make_local_xy_plot_gdf(xy_grid)
    else:
        plot_data_box = data_box_gdf
        plot_buffer = buffer_gdf
        plot_aoi = aoi_utm
        plot_grid = xy_grid

    plot_data_box.boundary.plot(ax=ax, color="black", linewidth=1.5, linestyle="-", label="Voxel data box")
    plot_buffer.boundary.plot(ax=ax, color="black", linewidth=1.0, linestyle="--", label="AOI buffer")
    plot_aoi.boundary.plot(ax=ax, color="black", linewidth=2.0, label="AOI polygon")

    nofly = plot_grid[plot_grid["flyable_2d"] == 0]
    fly = plot_grid[plot_grid["flyable_2d"] == 1]

    if PLOT_NOFLY_2D_CELLS and not nofly.empty:
        nofly.plot(ax=ax, color="red", markersize=4, alpha=0.20, label="No-fly outside polygon")
    if not fly.empty:
        fly.plot(ax=ax, color="limegreen", markersize=6, alpha=0.85, label="Flyable inside polygon")

    _, _, x_label, y_label = plot_xy_columns(plot_grid)
    ax.set_title("Full rectangular XY data box: polygon inside flyable, outside no-fly", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    # Force plot extent to the full data-box boundary. When local coordinates
    # are used, this makes the axes start cleanly at 0 m.
    bx0, by0, bx1, by1 = plot_data_box.total_bounds
    ax.set_xlim(float(bx0), float(bx1))
    ax.set_ylim(float(by0), float(by1))

    txt = (
        f"DX/DY = {DX_M:g} m\n"
        f"Buffer = {BUFFER_M:g} m\n"
        f"XY flyable = {len(fly):,}\n"
        f"XY no-fly = {len(nofly):,}\n"
        f"Outside/no-fly color = {'shown' if PLOT_NOFLY_2D_CELLS else 'hidden'}\n"
        f"Data box = rectangular"
    )
    ax.text(
        0.01, 0.01, txt,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def build_demo_grid_lines(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
    """Return 3D line segments and nodes for a small voxel/cube demo."""
    xs = np.arange(nx + 1) * dx
    ys = np.arange(ny + 1) * dy
    zs = np.arange(nz + 1) * dz

    segments = []

    # Lines parallel to X.
    for y in ys:
        for z in zs:
            segments.append([(xs[0], y, z), (xs[-1], y, z)])

    # Lines parallel to Y.
    for x in xs:
        for z in zs:
            segments.append([(x, ys[0], z), (x, ys[-1], z)])

    # Lines parallel to Z.
    for x in xs:
        for y in ys:
            segments.append([(x, y, zs[0]), (x, y, zs[-1])])

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return segments, nodes


def plot_voxel_cube_nodes(paths: Paths) -> None:
    """
    Plot a small explanatory cube with grid lines and nodes.
    Kept for optional debugging; not called by make_figures().
    """
    out_png = paths.fig_dir / "01_voxel_cube_nodes_resolution_cell.png"

    segments, nodes = build_demo_grid_lines(
        nx=BOX_DEMO_NX,
        ny=BOX_DEMO_NY,
        nz=BOX_DEMO_NZ,
        dx=DX_M,
        dy=DY_M,
        dz=DZ_M,
    )

    fig = plt.figure(figsize=(10, 7.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segments, colors="black", linewidths=0.6, alpha=0.85)
    ax.add_collection3d(lc)

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=12, c="red", depthshade=False, label="Voxel nodes")

    # Highlight one resolution cell near the front-right-top.
    x0 = (BOX_DEMO_NX - 2) * DX_M
    y0 = 0.0
    z0 = 0.0
    cell_segments, cell_nodes = build_demo_grid_lines(1, 1, 1, DX_M, DY_M, DZ_M)
    cell_segments = [[(px + x0, py + y0, pz + z0) for px, py, pz in seg] for seg in cell_segments]
    cell_lc = Line3DCollection(cell_segments, colors="black", linewidths=2.0, alpha=1.0)
    ax.add_collection3d(cell_lc)

    # Draw a translucent face-like marker by scatter at cell center.
    ax.scatter([x0 + DX_M / 2], [y0 + DY_M / 2], [z0 + DZ_M / 2], s=80, c="limegreen", marker="s", label="One voxel / resolution cell")

    # Dimension labels.
    ax.text(x0 + DX_M / 2, y0 - DY_M * 0.45, z0, f"Δx={DX_M:g} m", fontsize=9)
    ax.text(x0 + DX_M * 1.15, y0 + DY_M / 2, z0, f"Δy={DY_M:g} m", fontsize=9)
    ax.text(x0 + DX_M * 1.10, y0, z0 + DZ_M / 2, f"Δz={DZ_M:g} m", fontsize=9)

    ax.set_title("3D voxel data box with nodes and resolution cell", fontsize=12, fontweight="bold")
    ax.set_xlabel("X / UTM (m)")
    ax.set_ylabel("Y / UTM (m)")
    ax.set_zlabel("Z  (m)")
    ax.set_xlim(0, BOX_DEMO_NX * DX_M)
    ax.set_ylim(0, BOX_DEMO_NY * DY_M)
    ax.set_zlim(0, BOX_DEMO_NZ * DZ_M)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def sample_for_3d_plot(voxels: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    pieces = []
    for flyable_value in [1, 0]:
        sub = voxels[voxels["flyable"] == flyable_value]
        if len(sub) <= MAX_3D_PLOT_POINTS_PER_CLASS:
            pieces.append(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=MAX_3D_PLOT_POINTS_PER_CLASS, replace=False)
            pieces.append(sub.loc[idx])
    return pd.concat(pieces, ignore_index=True)


def plot_actual_voxel_sample(paths: Paths, voxels: pd.DataFrame) -> None:
    # Optional debugging figure; not called by make_figures().
    out_png = paths.fig_dir / "02_actual_3d_voxel_data_box_sample.png"
    sample = sample_for_3d_plot(voxels)

    fig = plt.figure(figsize=(11, 9), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    nofly = sample[sample["flyable"] == 0]
    fly = sample[sample["flyable"] == 1]
    x_col, y_col, x_label, y_label = plot_xy_columns(sample)

    if not nofly.empty:
        ax.scatter(
            nofly[x_col], nofly[y_col], nofly["z_agl_m"],
            s=1.3, c="red", alpha=0.10, label="No-fly outside polygon"
        )
    if not fly.empty:
        ax.scatter(
            fly[x_col], fly[y_col], fly["z_agl_m"],
            s=1.5, c="limegreen", alpha=0.18, label="Flyable inside polygon"
        )

    ax.set_title("Actual 3D voxel data box sample", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("Z  (m)")
    ax.set_zlim(Z_MIN_M, Z_MAX_M)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")



def choose_plot_strides(voxels: pd.DataFrame) -> tuple[int, int, int]:
    """
    Choose coarse plotting strides so the right-hand 3D QC panel can be drawn
    as translucent voxel blocks without trying to render every 50 m cell.

    The model output remains full resolution. These strides are only for the
    figure.
    """
    nx = int(voxels["ix"].nunique())
    ny = int(voxels["iy"].nunique())
    nz = int(voxels["iz"].nunique())

    sx = sy = sz = 1

    def displayed_count() -> int:
        return int(math.ceil(nx / sx) * math.ceil(ny / sy) * math.ceil(nz / sz))

    while displayed_count() > MAX_ACTUAL_VOXEL_CUBES_TO_RENDER:
        # Increase the stride of the dimension that still has the largest
        # displayed size. This keeps the plotted blocks roughly balanced.
        displayed_dims = np.array([nx / sx, ny / sy, nz / sz], dtype=float)
        which = int(np.argmax(displayed_dims))
        if which == 0:
            sx += 1
        elif which == 1:
            sy += 1
        else:
            sz += 1

    return sx, sy, sz


def make_coarse_voxels_for_block_plot(voxels: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """
    Aggregate the full 50 m voxel model into a coarser voxel table only for
    plotting. This creates connected cuboid blocks like a 3D data volume.

    Each plotted block stores flyable_fraction, so boundary blocks can be
    classified by majority.
    """
    sx, sy, sz = choose_plot_strides(voxels)
    df = voxels.copy()

    df["gx"] = (pd.to_numeric(df["ix"], errors="coerce").astype(int) // sx).astype(int)
    df["gy"] = (pd.to_numeric(df["iy"], errors="coerce").astype(int) // sy).astype(int)
    df["gz"] = (pd.to_numeric(df["iz"], errors="coerce").astype(int) // sz).astype(int)

    coarse = (
        df.groupby(["gx", "gy", "gz"], as_index=False)
        .agg(
            x_utm_m=("x_utm_m", "mean"),
            y_utm_m=("y_utm_m", "mean"),
            x_local_m=("x_local_m", "mean"),
            y_local_m=("y_local_m", "mean"),
            z_agl_m=("z_agl_m", "mean"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            flyable_fraction=("flyable", "mean"),
            voxel_count=("flyable", "size"),
        )
    )

    coarse["flyable"] = (coarse["flyable_fraction"] >= 0.5).astype(int)
    coarse["nofly"] = 1 - coarse["flyable"]
    coarse["label"] = np.where(
        coarse["flyable"] == 1,
        "flyable_majority_inside_polygon",
        "nofly_majority_outside_polygon",
    )

    return coarse, (sx, sy, sz)


def cube_faces_from_centers(
    centers_df: pd.DataFrame,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float, float, float]]]:
    """
    Convert voxel center points to cube faces for Poly3DCollection.

    Returns:
        faces      : list of 6 faces per voxel
        facecolors : one RGBA color per face
    """
    if centers_df is None or centers_df.empty:
        return [], []

    hx = dx / 2.0
    hy = dy / 2.0
    hz = dz / 2.0

    # 8 vertices around each voxel center.
    offsets = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=float)

    # Six cube faces, each defined by four vertex IDs.
    face_ids = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7],  # left
    ]

    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []

    x_col, y_col, _, _ = plot_xy_columns(centers_df)
    xyz = centers_df[[x_col, y_col, "z_agl_m"]].to_numpy(dtype=float)
    flyable = centers_df["flyable"].to_numpy(dtype=int)

    for center, f in zip(xyz, flyable):
        vertices = offsets + center.reshape(1, 3)
        if int(f) == 1:
            rgba = (0.25, 0.85, 0.35, PLOT_CUBE_ALPHA_FLYABLE)  # green flyable
        else:
            if not PLOT_NOFLY_3D_BLOCKS:
                continue
            rgba = (0.95, 0.20, 0.18, PLOT_CUBE_ALPHA_NOFLY)    # red no-fly

        for ids in face_ids:
            faces.append([
                (float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2]))
                for i in ids
            ])
            facecolors.append(rgba)

    return faces, facecolors


def draw_axis_arrows_3d(ax, xlim, ylim, zlim) -> None:
    """
    Draw a clean local X/Y/Z triad at the data-box origin.

    Earlier versions placed the arrows close to the subplot edge, and the
    3D camera projection made the X/Y arrows look misleading. This version
    anchors the triad at local (0, 0, 0), uses short arrows relative to the
    box size, and relies on the axis labels for the full-scale coordinate axes.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim

    xr = max(float(x1 - x0), 1.0)
    yr = max(float(y1 - y0), 1.0)
    zr = max(float(z1 - z0), 1.0)

    # For local-coordinate plots, keep the origin exactly at (0, 0, 0).
    # For absolute UTM plots, use the lower southwest data-box corner.
    base_x = 0.0 if USE_LOCAL_PLOT_COORDS else float(x0)
    base_y = 0.0 if USE_LOCAL_PLOT_COORDS else float(y0)
    base_z = float(z0)

    # Short, readable triad lengths. X/Y are capped so they do not cross the plot;
    # Z is larger because the model height is now only ~300 m.
    dx = min(xr * 0.14, 800.0)
    dy = min(yr * 0.14, 800.0)
    dz = min(zr * 0.60, 180.0)

    # Slight upward offset prevents the origin arrow heads from hiding in voxel faces.
    z_base = base_z + zr * 0.02

    ax.quiver(base_x, base_y, z_base, dx, 0, 0, color="black", linewidth=1.6, arrow_length_ratio=0.12)
    ax.quiver(base_x, base_y, z_base, 0, dy, 0, color="black", linewidth=1.6, arrow_length_ratio=0.12)
    ax.quiver(base_x, base_y, z_base, 0, 0, dz, color="black", linewidth=1.8, arrow_length_ratio=0.12)

    ax.text(base_x + dx * 1.10, base_y, z_base, "X", fontsize=11, fontweight="bold")
    ax.text(base_x, base_y + dy * 1.10, z_base, "Y", fontsize=11, fontweight="bold")
    ax.text(base_x, base_y, z_base + dz * 1.10, "Z", fontsize=11, fontweight="bold")


def draw_axis_triad_screen_inset(ax) -> None:
    """
    Draw a publication-style X/Y/Z orientation triad in 2D screen coordinates.

    This is only a visual orientation symbol. It is placed near the top-right
    of panel B so it does not interfere with the information box or the lower
    data region.
    """
    # Top-right placement inside the actual 3D panel.
    origin = (0.87, 0.82)
    x_tip = (0.95, 0.79)
    # Reverse the Y arrow direction relative to the previous version.
    y_tip = (0.94, 0.85)
    z_tip = (0.87, 0.93)

    arrow_kw = dict(
        arrowstyle="-|>",
        linewidth=1.9,
        color="black",
        mutation_scale=13,
        shrinkA=0,
        shrinkB=0,
    )

    ax.annotate("", xy=x_tip, xytext=origin, xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=y_tip, xytext=origin, xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=z_tip, xytext=origin, xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow_kw)

    ax.text2D(x_tip[0] + 0.008, x_tip[1] - 0.004, "X", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text2D(y_tip[0] + 0.006, y_tip[1] + 0.002, "Y", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.text2D(z_tip[0] - 0.004, z_tip[1] + 0.010, "Z", transform=ax.transAxes, fontsize=11, fontweight="bold")


def plot_voxel_cube_and_actual_data_box(paths: Paths, voxels: pd.DataFrame) -> None:
    """
    Combined QC figure:
      left  = conceptual voxel data box with vertices and one resolution cell
      right = actual model rendered as translucent voxel blocks, like a 3D
              data volume. The right panel is aggregated only for plotting;
              the saved model is still full-resolution 50 m voxels.
    """
    out_png = paths.fig_dir / "01_voxel_cube_and_actual_data_box.png"

    fig = plt.figure(figsize=(18, 10.2), dpi=FIG_DPI)

    # ------------------------------------------------------------
    # Left panel: conceptual voxel box / resolution cell
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(121, projection="3d")

    segments, nodes = build_demo_grid_lines(
        nx=BOX_DEMO_NX,
        ny=BOX_DEMO_NY,
        nz=BOX_DEMO_NZ,
        dx=DX_M,
        dy=DY_M,
        dz=DZ_M,
    )

    lc = Line3DCollection(segments, colors="black", linewidths=0.55, alpha=0.85)
    ax1.add_collection3d(lc)
    ax1.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=10, c="red", depthshade=False, label="Voxel vertices / nodes")

    # Highlight one resolution cell near the front-right-bottom.
    x0 = (BOX_DEMO_NX - 2) * DX_M
    y0 = 0.0
    z0 = 0.0
    cell_segments, _ = build_demo_grid_lines(1, 1, 1, DX_M, DY_M, DZ_M)
    cell_segments = [[(px + x0, py + y0, pz + z0) for px, py, pz in seg] for seg in cell_segments]
    cell_lc = Line3DCollection(cell_segments, colors="black", linewidths=2.0, alpha=1.0)
    ax1.add_collection3d(cell_lc)
    ax1.scatter(
        [x0 + DX_M / 2], [y0 + DY_M / 2], [z0 + DZ_M / 2],
        s=95, c="limegreen", marker="s", depthshade=False,
        label="One 50 m voxel center"
    )

    ax1.text(x0 + DX_M / 2, y0 - DY_M * 0.45, z0, f"Δx={DX_M:g} m", fontsize=8)
    ax1.text(x0 + DX_M * 1.12, y0 + DY_M / 2, z0, f"Δy={DY_M:g} m", fontsize=8)
    ax1.text(x0 + DX_M * 1.10, y0, z0 + DZ_M / 2, f"Δz={DZ_M:g} m", fontsize=8)

    ax1.set_title("A. Voxel data box: vertices and resolution cell", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Local X (m)")
    ax1.set_ylabel("Local Y (m)")
    ax1.set_zlabel("Z  (m)")
    ax1.set_xlim(0, BOX_DEMO_NX * DX_M)
    ax1.set_ylim(0, BOX_DEMO_NY * DY_M)
    ax1.set_zlim(0, BOX_DEMO_NZ * DZ_M)
    ax1.view_init(elev=22, azim=-58)
    ax1.legend(loc="upper left", fontsize=7)

    # ------------------------------------------------------------
    # Right panel: actual 3D data box rendered as translucent voxel blocks
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(122, projection="3d")

    coarse, strides = make_coarse_voxels_for_block_plot(voxels)
    sx, sy, sz = strides
    plot_dx = DX_M * sx
    plot_dy = DY_M * sy
    plot_dz = DZ_M * sz

    faces, facecolors = cube_faces_from_centers(coarse, plot_dx, plot_dy, plot_dz)
    if faces:
        pc = Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=(1.0, 1.0, 1.0, PLOT_CUBE_EDGE_ALPHA),
            linewidths=PLOT_CUBE_EDGE_LINEWIDTH,
            antialiased=True,
        )
        ax2.add_collection3d(pc)

    x_col, y_col, x_label, y_label = plot_xy_columns(coarse)

    # Add center nodes so the figure still clearly shows the voxel center points.
    if PLOT_SHOW_CENTER_NODES:
        fly = coarse[coarse["flyable"] == 1]
        nofly = coarse[coarse["flyable"] == 0]
        if PLOT_NOFLY_CENTER_NODES and not nofly.empty:
            ax2.scatter(
                nofly[x_col], nofly[y_col], nofly["z_agl_m"],
                s=PLOT_CENTER_NODE_SIZE, c="red", alpha=0.20, depthshade=False,
                label="No-fly voxel centers"
            )
        if not fly.empty:
            ax2.scatter(
                fly[x_col], fly[y_col], fly["z_agl_m"],
                s=PLOT_CENTER_NODE_SIZE, c="green", alpha=0.55, depthshade=False,
                label="Flyable voxel centers"
            )

    # Use the rendered coarse blocks to define the displayed extent.
    # For local coordinates this makes the plot start at 0 m, while the saved
    # model still keeps absolute UTM and WGS84 coordinates.
    x_min = float(coarse[x_col].min() - plot_dx / 2.0)
    x_max = float(coarse[x_col].max() + plot_dx / 2.0)
    y_min = float(coarse[y_col].min() - plot_dy / 2.0)
    y_max = float(coarse[y_col].max() + plot_dy / 2.0)
    if USE_LOCAL_PLOT_COORDS:
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
    xlim = (x_min, x_max)
    ylim = (y_min, y_max)
    zlim = (Z_MIN_M, Z_MAX_M)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_zlim(*zlim)

    # Keep the low-altitude vertical model readable without changing the
    # saved Z coordinates. This affects only the figure aspect.
    try:
        ax2.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], (zlim[1] - zlim[0]) * RIGHT_PANEL_Z_EXAGGERATION))
    except Exception:
        pass

    if PLOT_AXIS_TRIAD_MODE == "data_quiver":
        draw_axis_arrows_3d(ax2, xlim, ylim, zlim)
    elif PLOT_AXIS_TRIAD_MODE == "screen_inset":
        draw_axis_triad_screen_inset(ax2)

    ax2.set_title("B. Actual voxel data box: flyable voxels only", fontsize=11, fontweight="bold")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel("Z  (m)")
    ax2.view_init(elev=24, azim=-45)
    ax2.legend(loc="upper left", fontsize=7)

    txt = (
        f"Actual 3D voxel data box\n"
        f"Saved model voxel = {DX_M:g} × {DY_M:g} × {DZ_M:g} m\n"
        f"Rendered block = {plot_dx:g} × {plot_dy:g} × {plot_dz:g} m\n"
        f"Rendered blocks = {len(coarse):,}\n"
        f"Green = flyable inside polygon\n"
        f"Outside/no-fly voxels = {'shown' if PLOT_NOFLY_3D_BLOCKS else 'hidden in plot'}\n"
        f"Model still stores outside as no-fly\n"
        f"X/Y/Z axes = local coordinates; inset shows orientation\n"
        f"Vertical display exaggeration = {RIGHT_PANEL_Z_EXAGGERATION:g}×\n"
        f"Transparent faces + white edges = connected voxels"
    )
    ax2.text2D(
        0.02, 0.02, txt,
        transform=ax2.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
    )

    fig.suptitle("Base 3D voxel model: regular data volume and resolution cells", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved combined voxel figure: {out_png}")


def plot_vertical_section(paths: Paths, voxels: pd.DataFrame) -> None:
    # Optional debugging figure; not called by make_figures().
    out_png = paths.fig_dir / "03_vertical_xz_section_middle_y.png"
    y_mid = float(np.nanmedian(voxels["y_utm_m"]))
    unique_y = np.sort(voxels["y_utm_m"].unique())
    y_use = float(unique_y[np.argmin(np.abs(unique_y - y_mid))])

    sec = voxels[np.isclose(voxels["y_utm_m"], y_use)].copy()
    if sec.empty:
        print("[WARN] Empty vertical section. Skip plot.")
        return

    pivot = sec.pivot_table(index="z_agl_m", columns="x_utm_m", values="flyable", aggfunc="max")
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=FIG_DPI)
    arr = pivot.to_numpy()
    extent = [pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()]

    im = ax.imshow(
        arr,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="coolwarm_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Flyable = 1, no-fly = 0")

    ax.set_title(f"Vertical X-Z section through data box near Y = {y_use:.1f} m", fontsize=12, fontweight="bold")
    ax.set_xlabel("UTM X (m)")
    ax.set_ylabel("Z  (m)")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_z_levels(paths: Paths, z_levels: np.ndarray) -> None:
    # Optional debugging figure; not called by make_figures().
    out_png = paths.fig_dir / "04_z_levels.png"

    fig, ax = plt.subplots(figsize=(6, 8), dpi=FIG_DPI)
    ax.scatter(np.zeros_like(z_levels), z_levels, s=18)
    step = max(1, len(z_levels) // 20)
    for z in z_levels[::step]:
        ax.text(0.03, z, f"{z:.0f} m", va="center", fontsize=7)

    ax.set_xlim(-0.1, 0.45)
    ax.set_ylim(Z_MIN_M, Z_MAX_M)
    ax.set_xticks([])
    ax.set_ylabel("Z  (m)")
    ax.set_title("Voxel Z center levels", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def make_figures(
    paths: Paths,
    aoi_utm: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    data_box_gdf: gpd.GeoDataFrame,
    xy_grid: gpd.GeoDataFrame,
    z_levels: np.ndarray,
    voxels: pd.DataFrame,
) -> None:
    # Keep only two QC figures:
    #   00 = XY data-box classification
    #   01 = combined conceptual voxel cube + actual 3D sampled data box
    # No separate vertical-section or Z-level figures are generated.
    plot_xy_data_box(paths, aoi_utm, buffer_gdf, data_box_gdf, xy_grid)
    plot_voxel_cube_and_actual_data_box(paths, voxels)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    print("\n========== BUILD BASE 3D VOXEL DATA BOX ==========")
    paths = make_paths()

    aoi_gdf = load_aoi_polygon()
    utm_crs = estimate_utm_crs(aoi_gdf)
    aoi_utm = aoi_gdf.to_crs(utm_crs)

    buffer_gdf, data_box_gdf = make_buffer_and_box(aoi_utm)
    z_levels = make_z_levels()

    print(f"[INFO] UTM CRS: {utm_crs}")
    print(f"[INFO] Domain mode: {DOMAIN_MODE}")
    print(f"[INFO] XY voxel size: {DX_M:g} x {DY_M:g} m")
    print(f"[INFO] Z voxel size: {DZ_M:g} m")
    print(f"[INFO] Z max: {Z_MAX_M:g} m AGL")
    print(f"[INFO] Buffer: {BUFFER_M:g} m")

    xy_grid = make_xy_grid_box(aoi_utm, buffer_gdf, data_box_gdf)
    voxels = build_3d_voxel_table(xy_grid, z_levels)

    save_outputs(paths, aoi_gdf, aoi_utm, buffer_gdf, data_box_gdf, xy_grid, z_levels, voxels)
    make_figures(paths, aoi_utm, buffer_gdf, data_box_gdf, xy_grid, z_levels, voxels)

    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")
    print("Main outputs:")
    print(f"  {paths.data_dir / 'base_3d_voxel_data_box_50m.csv.gz'}")
    print(f"  {paths.data_dir / 'base_3d_voxel_data_box_50m.xyz'}")
    print(f"  {paths.fig_dir / '00_xy_full_data_box_flyable_nofly.png'}")
    print(f"  {paths.fig_dir / '01_voxel_cube_and_actual_data_box.png'}")


if __name__ == "__main__":
    main()
