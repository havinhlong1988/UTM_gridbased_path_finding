#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEM-only terrain burn for the Scenario-1 3D voxel model.

This script is intentionally separated from obstacle burning.
It does only one job:

    1. Read the base 3D voxel data-box model from script 02.
    2. Sample the DEM/topography to each XY voxel cell.
    3. Burn all voxel cells below/intersecting the DEM surface as no-fly.
    4. Burn all voxel cells colliding with GBA building volumes as no-fly.
    5. Burn OSM powerline poles/towers as ground-based columns and
       powerline strands/cables as airborne bands.
    6. Merge obstacle burn masks only after each source has its correct
       vertical MSL interval.
    7. Plot:
         - DEM/topography on the XY cell grid.
         - DEM/building-burned cells at selected Z slices.
         - 3D filled DEM/building-burned voxel cells.

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
    │   ├── final_output_voxel_model_50m.csv.gz      # next-step pathfinding table
    │   ├── final_output_voxel_model_50m.parquet     # next-step pathfinding table
    │   ├── final_output_voxel_model_50m.xyz         # next-step pathfinding XYZ
    │   ├── xy_grid_with_dem_terrain_msl_SW.csv.gz
    │   ├── xy_grid_with_dem_terrain_msl_SW.gpkg
    │   ├── dem_candidate_audit.csv
    │   └── dem_terrain_burn_summary.txt/json
    └── figures/
        ├── 00_dem_terrain_msl_cells_SW.png
        ├── 01_dem_terrain_burn_z_slices_SW.png
        ├── 02_3d_dem_burned_cells_SW.png
        └── 03_3d_topography_mesh_msl_SW.png
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
from matplotlib.colors import ListedColormap, BoundaryNorm
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

# Output folder for this DEM/building/powerline burn step.
OUTDIR = Path("output/03a_dem_terrain_burn_only_senario1")

# Final model files used by the next pathfinding/model step.
# Only these exported model files use the final_output_* prefix.
# Intermediate QC/debug files keep their original names so diagnostics are easy to track.
FINAL_OUTPUT_MODEL_BASENAME = "final_output_voxel_model_50m"

# Fixed vertical display range for every 3D plot.
# This is plotting-only; it does not change any voxel z values or burn logic.
PLOT_3D_Z_AXIS_MIN_M = 0.0
PLOT_3D_Z_AXIS_MAX_M = 120.0

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

# GBA / GlobalBuildingAtlas building-volume burn.
# Normal rule for GBA.Height / GBA.LoD1:
#     input height is AGL building height, not absolute MSL elevation.
# Therefore:
#     building_base_msl = terrain_msl_m
#     building_top_msl  = terrain_msl_m + building_height_agl_m
#
# If your building-height source is mixed or already stored as absolute building-top
# elevation, change BUILDING_HEIGHT_INPUT_REFERENCE below instead of changing the DEM.
BURN_BUILDING_VOLUME = True

# Building height reference in the source attribute/raster.
#   "AGL"      : source value = building height above local terrain.
#                building_top_msl = terrain_msl + source_height
#   "MSL_TOP"  : source value = absolute building-top elevation MSL.
#                building_height_agl = source_top_msl - terrain_msl
#   "AUTO"     : per-cell conservative check. Values that look like absolute
#                top elevation are converted to AGL; other values are kept as AGL.
# Default remains AGL because GBA height products are normally AGL.
BUILDING_HEIGHT_INPUT_REFERENCE = "AGL"  # AGL | MSL_TOP | AUTO
BUILDING_AUTO_MSL_TOP_MIN_VALUE_M = 35.0
BUILDING_AUTO_AGL_MAX_REASONABLE_M = 80.0
BUILDING_AUTO_MIN_ABOVE_TERRAIN_M = 1.0
GBA_BUILDING_DIR_CANDIDATES = [
    INPUT_DATA_DIR / "gba",
    INPUT_DATA_DIR / "GBA",
    INPUT_DATA_DIR / "globalbuildingatlas",
    INPUT_DATA_DIR / "GlobalBuildingAtlas",
    INPUT_DATA_DIR / "buildings",
    INPUT_DATA_DIR / "building",
]
GBA_BUILDING_VECTOR_PATTERNS = [
    "**/*gba*building*.gpkg", "**/*gba*polygon*.gpkg", "**/*gba*lod1*.gpkg",
    "**/*global*building*.gpkg", "**/*building*footprint*.gpkg", "**/*footprint*.gpkg",
    "**/*polygon*.gpkg", "**/*building*.gpkg",
    "**/*gba*building*.geojson", "**/*gba*polygon*.geojson", "**/*gba*lod1*.geojson",
    "**/*global*building*.geojson", "**/*building*footprint*.geojson", "**/*footprint*.geojson",
    "**/*polygon*.geojson", "**/*building*.geojson",
    "**/*gba*building*.shp", "**/*gba*polygon*.shp", "**/*gba*lod1*.shp",
    "**/*global*building*.shp", "**/*building*footprint*.shp", "**/*footprint*.shp",
    "**/*polygon*.shp", "**/*building*.shp",
    "**/*gba*building*.parquet", "**/*gba*polygon*.parquet", "**/*gba*lod1*.parquet",
    "**/*global*building*.parquet", "**/*building*footprint*.parquet", "**/*footprint*.parquet",
    "**/*polygon*.parquet", "**/*building*.parquet",
]
GBA_BUILDING_HEIGHT_RASTER_PATTERNS = [
    "**/*gba*height*.tif", "**/*building*height*.tif", "**/*height*.tif",
    "**/*gba*height*.tiff", "**/*building*height*.tiff", "**/*height*.tiff",
]
GBA_BUILDING_HEIGHT_COLUMN_CANDIDATES = [
    "height", "height_m", "building_height", "building_height_m", "bldg_height",
    "height_agl", "height_agl_m", "bh", "BH", "Height", "HEIGHT", "h", "H",
    "mean_height", "median_height", "pred_height", "pred_height_m",
]
# GBA polygons are documented as EPSG:3857 when CRS metadata is missing/ambiguous.
GBA_DEFAULT_VECTOR_CRS_IF_MISSING = "EPSG:3857"
GBA_HEIGHT_REFERENCE = "AGL"
BUILDING_MIN_HEIGHT_M = 1.0
BUILDING_HEIGHT_BUFFER_M = 0.0
SAVE_BUILDING_BURN_DEBUG_FILES = True

# OSM powerline / power-pole obstacle-geofence burn.
# Powerline system is treated as an obstacle/geofence volume similar to buildings:
#   powerline_base_msl = terrain_msl
#   powerline_top_msl  = terrain_msl + powerline_height_agl
# for the default conservative mode "from_ground_to_height".
# Line features are buffered horizontally; pole/tower point features are buffered as circular footprints.
BURN_POWERLINE_SYSTEM = True
OSM_POWERLINE_DIR_CANDIDATES = [
    INPUT_DATA_DIR / "osm",
    INPUT_DATA_DIR / "OSM",
    INPUT_DATA_DIR / "powerline",
    INPUT_DATA_DIR / "powerlines",
    INPUT_DATA_DIR / "power",
    INPUT_DATA_DIR / "geofence",
    INPUT_DATA_DIR,
]
OSM_POWERLINE_LINE_PATTERNS = [
    "**/*power*line*.gpkg", "**/*powerline*.gpkg", "**/*power_line*.gpkg",
    "**/*osm*power*.gpkg", "**/*transmission*.gpkg",
    "**/*power*line*.geojson", "**/*powerline*.geojson", "**/*power_line*.geojson",
    "**/*osm*power*.geojson", "**/*transmission*.geojson",
    "**/*power*line*.shp", "**/*powerline*.shp", "**/*power_line*.shp",
    "**/*osm*power*.shp", "**/*transmission*.shp",
    "**/*power*line*.parquet", "**/*powerline*.parquet", "**/*power_line*.parquet",
    "**/*osm*power*.parquet", "**/*transmission*.parquet",
]
OSM_POWERLINE_POLE_PATTERNS = [
    "**/*power*pole*.gpkg", "**/*powerpole*.gpkg", "**/*pole*.gpkg", "**/*tower*.gpkg", "**/*pylon*.gpkg",
    "**/*power*pole*.geojson", "**/*powerpole*.geojson", "**/*pole*.geojson", "**/*tower*.geojson", "**/*pylon*.geojson",
    "**/*power*pole*.shp", "**/*powerpole*.shp", "**/*pole*.shp", "**/*tower*.shp", "**/*pylon*.shp",
    "**/*power*pole*.parquet", "**/*powerpole*.parquet", "**/*pole*.parquet", "**/*tower*.parquet", "**/*pylon*.parquet",
]
OSM_POWERLINE_HEIGHT_COLUMN_CANDIDATES = [
    "height_agl_m", "height_agl", "agl_m", "powerline_height_agl_m", "powerline_height_m",
    "pole_height_agl_m", "pole_height_m", "tower_height_agl_m", "tower_height_m",
    "height", "height_m", "h", "H", "Height", "HEIGHT",
    "clearance_m", "clearance", "cable_height_m", "wire_height_m",
]
# If no height column is found, use conservative default AGL heights.
POWERLINE_DEFAULT_LINE_HEIGHT_AGL_M = 25.0
POWERLINE_DEFAULT_POLE_HEIGHT_AGL_M = 20.0
POWERLINE_MIN_HEIGHT_M = 1.0
POWERLINE_HEIGHT_REFERENCE = "AGL"  # AGL | MSL_TOP

# Horizontal obstacle/geofence buffers in UTM meters.
POWERLINE_HORIZONTAL_BUFFER_M = 15.0
POWERLINE_POLE_HORIZONTAL_BUFFER_M = 8.0

# The powerline system is separated into two physical burn types:
#   line/strand/cable: aerial band only, centered at terrain + line_height_agl
#   pole/tower/pylon : column from terrain to terrain + pole_height_agl
# This avoids incorrectly burning every voxel from the ground up to an overhead cable.
POWERLINE_LINE_BURN_VERTICAL_MODE = "cable_band"          # cable_band | from_ground_to_height
POWERLINE_POLE_BURN_VERTICAL_MODE = "from_ground_to_height"  # from_ground_to_height | cable_band
POWERLINE_LINE_VERTICAL_BUFFER_M = 2.5   # +/- around aerial cable height; 2.5 m ~= one 5 m voxel half-thickness
POWERLINE_POLE_VERTICAL_BUFFER_M = 0.0

# Topography-aware powerline source burn/display.
# Terrain burn uses whole voxel cells: if DEM terrain is at ~20 m and dz=5 m,
# the topography-burned voxel column may visually reach ~22.5 m.  The powerline
# source layer should not be classified/plotted inside that already-burned
# topographic voxel column.  When this is True, the powerline effective base is
# lifted to the top of the DEM-burned voxel column in the same XY cell:
#     effective_powerline_base = max(physical_powerline_base, topo_burn_top)
# The physical top stays unchanged (terrain + AGL height, or MSL_TOP input).
POWERLINE_RESPECT_TOPO_BURN_TOP = True
POWERLINE_TOPO_BURN_TOP_CLEARANCE_M = 0.0

# Figure-02 title/QC guard. The output title and QC file will explicitly show
# that the powerline base/top are calculated relative to local topography.
FIG02_SHOW_POWERLINE_TOPO_AWARE_FLAG_IN_TITLE = True
SAVE_POWERLINE_TOPO_AWARE_INTERVAL_QC = True

# Backward-compatible summary name. The actual burn now uses the kind-specific modes above.
POWERLINE_BURN_VERTICAL_MODE = "kind_specific"
SAVE_POWERLINE_BURN_DEBUG_FILES = True

# Plot settings.
FIG_DPI = 220
RANDOM_SEED = 42
SLICE_Z_LEVELS_MSL = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100]

# DEM plot display scale only.
# This does NOT change the DEM values used for terrain burning.
# Values above 40 m are clipped only for figure color mapping / display.
DEM_PLOT_VMIN_M = 5.0
DEM_PLOT_VMAX_M = 35.0
DEM_3D_ZMAX_PLOT_M = PLOT_3D_Z_AXIS_MAX_M

# Separate topography mesh figure.
# The mesh Z coordinate is the sampled terrain_msl_m value in meters MSL.
# Color scale is display-only and does not change DEM or burn values.
PLOT_3D_TOPOGRAPHY_MESH = True
TOPO_MESH_MAX_GRID_CELLS = 40000
TOPO_MESH_ZLIM_MAX_M = PLOT_3D_Z_AXIS_MAX_M  # fixed plot-only z-axis max
TOPO_MESH_EDGE_ALPHA = 0.22
TOPO_MESH_SURFACE_ALPHA = 0.92

# Separate building-volume QC figure.
# Left panel  = building height as AGL volume (base z = 0 m)
# Right panel = building volume in MSL (base z = terrain_msl_m, top z = building_top_msl_m)
# XY coordinates are always converted to distance from the SW reference point.
PLOT_3D_BUILDING_VOLUME_CHECK = True
MAX_3D_BUILDING_CELLS_TO_RENDER = 20000
BUILDING_VOLUME_PLOT_ALPHA = 0.48
BUILDING_VOLUME_PLOT_EDGE_ALPHA = 0.22
BUILDING_VOLUME_PLOT_EDGE_LINEWIDTH = 0.12
BUILDING_VOLUME_PLOT_Z_EXAGGERATION = 18.0
BUILDING_VOLUME_AGL_RGB = (0.98, 0.82, 0.32)
BUILDING_VOLUME_MSL_RGB = (0.95, 0.66, 0.20)
BUILDING_VOLUME_AGL_CMAP = "viridis"
BUILDING_VOLUME_MSL_CMAP = "plasma"
BUILDING_VOLUME_TERRAIN_CMAP = "terrain"
BUILDING_VOLUME_TERRAIN_MESH_ALPHA = 0.34
BUILDING_VOLUME_TERRAIN_GRAY_RGB = (0.62, 0.62, 0.62)
BUILDING_BASE_PROJECTION_RGB = (0.08, 0.08, 0.08)

# Keep Figure 04 clean. No long note boxes or extra debug markers by default.
BUILDING_VOLUME_QC_SHOW_TEXT_BOX = False
BUILDING_VOLUME_QC_SHOW_BASE_DOTS = False
BUILDING_VOLUME_QC_SHOW_LEGEND = False
BUILDING_VOLUME_TERRAIN_MAX_GRID_CELLS = 8000

# 3D figure uses sampled complete XY columns only, to keep the plot manageable.
# The saved model still burns every cell; this limit affects the figure only.
MAX_3D_DEM_XY_COLUMNS = 1200
MAX_3D_DEM_VOXELS_TO_RENDER = 22000
MAX_3D_DEM_VOXEL_CUBES_TO_RENDER = 1800
# Figure 02 voxel transparency.
# Smaller alpha = more transparent.
DEM_STATE_PLOT_GREEN_ALPHA = 0.24
DEM_STATE_PLOT_BURNED_GRAY_ALPHA = 0.46
DEM_STATE_PLOT_EDGE_ALPHA = 0.70
DEM_STATE_PLOT_EDGE_LINEWIDTH = 0.22
DEM_STATE_EDGE_BLACK_RGB = (0.0, 0.0, 0.0)
DEM_STATE_NODE_RED_RGB = (1.0, 0.0, 0.0)
DEM_STATE_BURNED_NODE_SIZE = 2.2
DEM_STATE_SHOW_NOTE_BOX = False

# Increase visual-only Z exaggeration for Figure 02.
# This does NOT change saved voxel Z values.
DEM_STATE_RIGHT_PANEL_Z_EXAGGERATION = 22.0

# Figure 02 colors.
DEM_STATE_UNBURNED_GREEN_RGB = (0.25, 0.85, 0.35)
DEM_STATE_BURNED_GRAY_RGB = (0.55, 0.55, 0.55)
DEM_STATE_BUILDING_YELLOW_RGB = (1.00, 0.96, 0.62)
DEM_STATE_PLOT_BUILDING_YELLOW_ALPHA = 0.42
DEM_STATE_SHOW_CENTER_NODES = True
# Figure 02 option:
#   True  = plot green non-burned inside-AOI cells + gray/yellow burned cells
#   False = plot only selected burned cells; hide green non-burned cells
PLOT_UNBURNED_3D_DEM_CELLS = False

# Burn-cell plotting toggles. These affect figures only; the saved model still
# follows the burn calculation order in main(): topography first, then building.
# Use these switches to debug each burn source independently.
PLOT_BURN_CELLS_BY_TOPO = False     # True/False: plot DEM/topography-burned cells
PLOT_BURN_CELLS_BY_BUILDING = True   # True/False: plot building-volume-burned cells
PLOT_BURN_CELLS_BY_POWERLINE = True  # True/False: plot OSM powerline/pole obstacle-geofence cells

DEM_STATE_POWERLINE_PURPLE_RGB = (0.55, 0.20, 0.85)
DEM_STATE_PLOT_POWERLINE_PURPLE_ALPHA = 0.36
FIG02_MAX_TRUE_XY_POWERLINE_VOXELS_TO_RENDER = 70000
# Figure-02 true-XY unified layer builder.
# When True, topo and building layers are built from the same burn-mask voxel
# logic used by the correct powerline QC layer: select burned voxel rows, clip
# each voxel to its physical MSL interval, then draw true 50 x 50 x dz cells.
# This avoids mixed paths such as topo from coarse blocks, building rebuilt from
# xy_gdf, and powerline from burn masks.
FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING = True
FIG02_MAX_TRUE_XY_BUILDING_VOXELS_TO_RENDER = 70000
FIG02_AXIS_LIMITS_FROM_VISIBLE_TRUE_XY_LAYERS = True

# Figure-02 height diagnostics.
# MSL elevation can make a low building on a high hill appear higher than a
# powerline on lower terrain.  Keep burn/plot in MSL for physics, but also save
# AGL and same-XY comparison tables so we can verify that the powerline is
# actually above nearby buildings where they overlap.
FIG02_SAVE_LAYER_HEIGHT_DIAGNOSTICS = True
FIG02_SAVE_POWERLINE_BUILDING_OVERLAP_QC = True

# Figure-02 powerline plotting guards. Figure 05 already uses the correct
# UTM -> local-SW coordinates, so Figure 02 should re-use the same true XY-cell
# coordinate source instead of trusting any copied x/y columns from intermediate
# tables.  This is plotting-only; collision/burn calculations are unchanged.
FIG02_POWERLINE_USE_XY_GRID_LOCAL_COORDS = True
# Figure 02 powerline fix:
# Use the exact same purple voxel-builder and visual style as Figure 05.
# This avoids any duplicated/alternative Figure-02 coordinate path.
FIG02_POWERLINE_USE_FIG05_QC_LOGIC = True
FIG02_POWERLINE_DRAW_LAST_WITH_QC_ALPHA = True
FIG02_SHOW_POWERLINE_CENTER_NODES = False

# Separate QC figure: gray topographic mesh + purple powerline-system burn blocks.
PLOT_3D_POWERLINE_TOPOGRAPHY_CHECK = True
POWERLINE_TOPO_MESH_MAX_GRID_CELLS = 12000
POWERLINE_TOPO_MESH_ALPHA = 0.32
POWERLINE_TOPO_MESH_EDGE_ALPHA = 0.12
POWERLINE_TOPO_MESH_GRAY_RGB = (0.62, 0.62, 0.62)
POWERLINE_VOLUME_PLOT_ALPHA = 0.42
POWERLINE_VOLUME_EDGE_ALPHA = 0.28
POWERLINE_VOLUME_EDGE_LINEWIDTH = 0.10
POWERLINE_VOLUME_Z_EXAGGERATION = 22.0
MAX_3D_POWERLINE_VOXELS_TO_RENDER = 70000

# Visual draw order for selected burn-cell layers.
#   "topo_then_building" = draw topo/DEM first, then building on top.
#   "building_then_topo" = draw building first, then topo/DEM on top.
# This is plotting-only and does not change final_nofly_dem_only.
# For a physically correct combined view, draw topo first and building later.
BURN_CELL_PLOT_DRAW_ORDER = "topo_then_building_then_powerline"

# Figure-02 clipping is plotting-only.
# Building-only QC must respect the physical MSL volume:
#     base = terrain_msl, top = terrain_msl + building_height_agl.
# Therefore building clipping is ON by default. Topo is kept as full burned
# voxels because the terrain-burn rule burns the voxel that touches/below DEM.
CLIP_3D_BUILDING_BURN_PLOT_TO_TRUE_VOLUME = True
CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE = False

# Plot-only stacking/filling fix for Figure 02.
# True means: when BOTH topo and building layers are visible, draw topo/DEM
# burned voxels first, then build a continuous building-class voxel stack
# immediately above the top of the topo voxel column.
# IMPORTANT: when PLOT_BURN_CELLS_BY_TOPO=False, the building-only plot uses
# the true MSL building volume instead of stacking on a hidden topo layer.
# This is only a display correction; it does not change burn_building_volume,
# building_top_msl_m, or the saved model.
PLOT_3D_BUILDING_STACK_ON_TOPO = True
PLOT_3D_BUILDING_FILL_GAP_ABOVE_TOPO = True
# "topo_voxel_top" matches the user's desired burn-cell QC logic:
# cells below topography are topo; cells above topo and below/inside building
# are building. "terrain_surface" uses the clipped DEM surface instead.
PLOT_3D_BUILDING_STACK_BASE_MODE = "topo_voxel_top"  # topo_voxel_top | terrain_surface

# Important Figure 02 correction:
#   True = draw building-burned cells from the original XY voxel columns,
#          not from the coarse topo plot blocks. This avoids false tall/
#          shifted building blocks caused by coarse aggregation using max height.
#          Topo may still be coarsened for readability, but building positions
#          and heights remain tied to the real XY cell and corrected AGL height.
PLOT_3D_BUILDING_FROM_TRUE_XY_CELLS = True

# When true-XY building cells are used, the topo layer must use the same
# true XY grid in the combined QC figure. Otherwise small 50 m building
# cells are visually compared with coarse topo blocks and look shifted/
# pushed outside the terrain blocks. This is plotting-only.
PLOT_3D_TOPO_FROM_TRUE_XY_WHEN_TRUE_BUILDING = True

# Final Figure-02 consistency switch.
# True means every visible Figure-02 layer uses the same original 50 x 50 x dz
# voxel grid. This fixes the bad switch cases where topo-only and unburned
# layers were still drawn from coarse blocks while building used true XY cells.
PLOT_3D_USE_TRUE_XY_FOR_ALL_SELECTED_LAYERS = True
FIG02_MAX_TRUE_XY_TOPO_VOXELS_TO_RENDER = 70000
FIG02_MAX_TRUE_XY_UNBURNED_VOXELS_TO_RENDER = 70000

FIG02_BUILDING_QC_WARN_HEIGHT_ABOVE_M = 35.0
FIG02_BUILDING_QC_SAVE_TOP_N = 50
DEM_STATE_CENTER_NODE_SIZE = 1.2
# For coarse figure blocks:
#   "any"      -> a coarse block becomes black if any full-resolution cell in it is DEM/building-burned.
#   "majority" -> a coarse block becomes black only if >=50% of its cells are DEM/building-burned.
DEM_STATE_COARSE_BURN_RULE = "any"

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


def fixed_plot_z_range() -> tuple[float, float]:
    """Return fixed plot-only Z limits for every 3D figure."""
    zmin = float(PLOT_3D_Z_AXIS_MIN_M)
    zmax = float(PLOT_3D_Z_AXIS_MAX_M)
    if not np.isfinite(zmin):
        zmin = 0.0
    if (not np.isfinite(zmax)) or zmax <= zmin:
        zmax = zmin + 1.0
    return zmin, zmax


def apply_fixed_3d_z_axis(ax) -> tuple[float, float]:
    """Apply the global fixed 3D Z-axis range and return it."""
    zmin, zmax = fixed_plot_z_range()
    ax.set_zlim(zmin, zmax)
    return zmin, zmax


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
# GBA BUILDING / VOLUME BURN
# ======================================================================


def unique_existing_building_files(patterns: Iterable[str], roots: Iterable[Path]) -> list[Path]:
    """Find building vector/raster candidates without applying DEM filters."""
    files: list[Path] = []
    seen = set()
    for root in roots:
        if not Path(root).exists():
            continue
        for pat in patterns:
            for p in Path(root).glob(pat):
                if not p.is_file():
                    continue
                name = str(p).lower()
                # Avoid non-building derivatives that may contain "height" in their name.
                if any(bad in name for bad in [
                    "dem", "elevation", "terrain", "slope", "aspect", "hillshade",
                    "population", "density", "road", "powerline", "traffic", "osm",
                ]):
                    continue
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def read_building_vector(path: Path) -> gpd.GeoDataFrame:
    """Read GBA/building vector data from common GIS formats."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq", ".geoparquet"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def infer_missing_building_crs(buildings: gpd.GeoDataFrame, utm_crs):
    """Infer CRS for building vectors when CRS metadata is missing.

    This prevents a common failure mode where lon/lat building footprints are
    accidentally treated as EPSG:3857 and then no longer intersect the Hoa Lac AOI.
    """
    try:
        minx, miny, maxx, maxy = buildings.total_bounds
    except Exception:
        return GBA_DEFAULT_VECTOR_CRS_IF_MISSING

    # lon/lat degrees
    if -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0:
        return "EPSG:4326"

    # UTM-like meter coordinates around Vietnam
    if 100000.0 <= minx <= 900000.0 and 100000.0 <= maxx <= 900000.0 and 1000000.0 <= miny <= 3000000.0 and 1000000.0 <= maxy <= 3000000.0:
        return utm_crs

    return GBA_DEFAULT_VECTOR_CRS_IF_MISSING



def find_building_height_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find a likely building-height column in a vector building table."""
    cols = list(gdf.columns)
    lower_to_real = {str(c).lower(): c for c in cols}
    for cand in GBA_BUILDING_HEIGHT_COLUMN_CANDIDATES:
        if cand in cols:
            return cand
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]

    # Fallback: any column containing height but not elevation/DEM/MSL.
    for c in cols:
        lc = str(c).lower()
        if "height" in lc and not any(bad in lc for bad in ["elev", "msl", "dem", "terrain"]):
            return c
    return None



def resolve_building_height_reference_for_xy(xy_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Convert source building-height values to a consistent AGL/MSL definition.

    The voxel burn always uses:
        building_base_msl_m = terrain_msl_m
        building_height_agl_m = corrected source height above terrain
        building_top_msl_m = terrain_msl_m + building_height_agl_m + buffer

    This fixes mixed-height products without allowing buildings to start below
    the DEM/topographic surface.
    """
    xy_gdf = xy_gdf.copy()
    source = pd.to_numeric(
        xy_gdf.get("building_height_input_m", xy_gdf.get("building_height_agl_m", 0.0)),
        errors="coerce",
    ).fillna(0.0).astype(float)
    terrain = pd.to_numeric(xy_gdf["terrain_msl_m"], errors="coerce").fillna(0.0).astype(float)

    mode = str(BUILDING_HEIGHT_INPUT_REFERENCE).strip().upper()
    if mode not in {"AGL", "MSL_TOP", "AUTO"}:
        raise ValueError('BUILDING_HEIGHT_INPUT_REFERENCE must be "AGL", "MSL_TOP", or "AUTO"')

    source_np = source.to_numpy(dtype=float)
    terrain_np = terrain.to_numpy(dtype=float)

    if mode == "AGL":
        use_as_msl_top = np.zeros(len(xy_gdf), dtype=bool)
        agl_np = source_np.copy()
    elif mode == "MSL_TOP":
        use_as_msl_top = np.ones(len(xy_gdf), dtype=bool)
        agl_np = source_np - terrain_np
    else:
        # AUTO is deliberately conservative. It converts only values that look
        # like an absolute top elevation above local terrain, while values that
        # are clearly ordinary building heights remain AGL.
        agl_if_msl_top = source_np - terrain_np
        use_as_msl_top = (
            (source_np >= BUILDING_AUTO_MSL_TOP_MIN_VALUE_M)
            & (agl_if_msl_top >= BUILDING_AUTO_MIN_ABOVE_TERRAIN_M)
            & (agl_if_msl_top <= BUILDING_AUTO_AGL_MAX_REASONABLE_M)
            & (source_np > terrain_np)
        )
        agl_np = np.where(use_as_msl_top, agl_if_msl_top, source_np)

    # Never allow a building volume below/inside the topographic surface.
    agl_np = np.where(np.isfinite(agl_np), agl_np, 0.0)
    agl_np = np.maximum(agl_np, 0.0)

    xy_gdf["building_height_input_m"] = source_np
    xy_gdf["building_height_reference_used"] = np.where(use_as_msl_top, "MSL_TOP", "AGL")
    xy_gdf["building_height_agl_m"] = agl_np
    xy_gdf["building_base_msl_m"] = terrain_np
    xy_gdf["building_top_msl_m"] = terrain_np + agl_np + float(BUILDING_HEIGHT_BUFFER_M)

    valid = agl_np >= float(BUILDING_MIN_HEIGHT_M)
    stats = {
        "building_height_input_reference_parameter": mode,
        "building_height_cells_treated_as_agl": int((valid & (~use_as_msl_top)).sum()),
        "building_height_cells_treated_as_msl_top": int((valid & use_as_msl_top).sum()),
        "building_height_agl_min_m": float(np.nanmin(agl_np[valid])) if np.any(valid) else 0.0,
        "building_height_agl_p50_m": float(np.nanpercentile(agl_np[valid], 50)) if np.any(valid) else 0.0,
        "building_height_agl_max_m": float(np.nanmax(agl_np[valid])) if np.any(valid) else 0.0,
        "building_top_msl_min_m": float(np.nanmin(xy_gdf.loc[valid, "building_top_msl_m"])) if np.any(valid) else 0.0,
        "building_top_msl_max_m": float(np.nanmax(xy_gdf.loc[valid, "building_top_msl_m"])) if np.any(valid) else 0.0,
    }

    print(
        "[CHECK] Building height reference conversion: "
        f"mode={mode}, AGL cells={stats['building_height_cells_treated_as_agl']:,}, "
        f"MSL_TOP cells={stats['building_height_cells_treated_as_msl_top']:,}, "
        f"corrected AGL={stats['building_height_agl_min_m']:.2f}..{stats['building_height_agl_max_m']:.2f} m, "
        f"top MSL={stats['building_top_msl_min_m']:.2f}..{stats['building_top_msl_max_m']:.2f} m"
    )
    return xy_gdf, stats


def choose_building_vector(paths: Paths) -> Path | None:
    candidates = unique_existing_building_files(
        GBA_BUILDING_VECTOR_PATTERNS,
        roots=GBA_BUILDING_DIR_CANDIDATES,
    )
    audit_rows = []
    if not candidates:
        print("[WARN] No GBA/building vector candidates found.")
        pd.DataFrame(audit_rows).to_csv(paths.data_dir / "building_vector_candidate_audit.csv", index=False)
        return None

    print("\n========== GBA / BUILDING VECTOR CANDIDATES ==========")
    for i, p in enumerate(candidates):
        row = {"rank": i, "path": str(p)}
        try:
            gdf = read_building_vector(p)
            hcol = find_building_height_column(gdf)
            row.update({
                "feature_count": int(len(gdf)),
                "crs": str(gdf.crs),
                "height_column": hcol,
                "columns": ",".join(map(str, gdf.columns[:30])),
            })
            print(f"[{i:02d}] n={len(gdf):,} height_col={hcol} crs={gdf.crs} | {p}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{i:02d}] ERROR | {p} | {exc}")
        audit_rows.append(row)

    pd.DataFrame(audit_rows).to_csv(paths.data_dir / "building_vector_candidate_audit.csv", index=False)

    # Prefer candidates with a height column, then any readable vector candidate.
    readable = [r for r in audit_rows if "error" not in r and int(r.get("feature_count", 0)) > 0]
    with_height = [r for r in readable if r.get("height_column") not in [None, "None", ""]]
    chosen = with_height[0] if with_height else (readable[0] if readable else None)
    if chosen is None:
        return None
    p = Path(chosen["path"])
    print(f"[OK] Selected building vector: {p}")
    return p


def choose_building_height_raster(paths: Paths) -> Path | None:
    if rasterio is None:
        return None
    candidates = unique_existing_building_files(
        GBA_BUILDING_HEIGHT_RASTER_PATTERNS,
        roots=GBA_BUILDING_DIR_CANDIDATES,
    )
    audit_rows = []
    if not candidates:
        pd.DataFrame(audit_rows).to_csv(paths.data_dir / "building_height_raster_candidate_audit.csv", index=False)
        return None

    print("\n========== GBA / BUILDING HEIGHT RASTER CANDIDATES ==========")
    for i, p in enumerate(candidates):
        st = raster_stats(p)
        st["rank"] = i
        audit_rows.append(st)
        if "error" in st:
            print(f"[{i:02d}] ERROR | {p} | {st['error']}")
        else:
            print(
                f"[{i:02d}] min={st['min']:.2f} p50={st['p50']:.2f} "
                f"p99={st['p99']:.2f} max={st['max']:.2f} | {p}"
            )

    pd.DataFrame(audit_rows).to_csv(paths.data_dir / "building_height_raster_candidate_audit.csv", index=False)
    for p, row in zip(candidates, audit_rows):
        if "error" not in row and int(row.get("valid_count", 0)) > 0:
            print(f"[OK] Selected building height raster: {p}")
            return p
    return None


def sample_building_height_raster_at_centroids(buildings_utm: gpd.GeoDataFrame, raster_path: Path) -> pd.Series:
    """Sample a GBA.Height raster at building centroids. Values are building height AGL in meters."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed, cannot sample building height raster")

    with rasterio.open(raster_path) as src:
        pts = buildings_utm.copy()
        if pts.crs is None:
            pts = pts.set_crs(DEFAULT_UTM_CRS)
        pts = pts.to_crs(src.crs)
        centroids = pts.geometry.centroid
        coords = [(float(p.x), float(p.y)) for p in centroids]
        vals = []
        nodata = src.nodata
        for val in src.sample(coords):
            v = float(val[0]) if len(val) else np.nan
            if nodata is not None and math.isclose(v, float(nodata), rel_tol=0, abs_tol=1e-8):
                v = np.nan
            vals.append(v)
    return pd.Series(vals, index=buildings_utm.index, dtype="float64")


def load_gba_buildings(paths: Paths, xy_gdf: gpd.GeoDataFrame, utm_crs) -> tuple[gpd.GeoDataFrame, dict]:
    """Load building footprints and assign an AGL building height in meters."""
    stats = {
        "building_source": None,
        "building_height_source": None,
        "height_reference": GBA_HEIGHT_REFERENCE,
        "building_count_raw": 0,
        "building_count_in_data_box": 0,
        "building_count_valid_height": 0,
    }

    vector_path = choose_building_vector(paths)
    if vector_path is None:
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    buildings = read_building_vector(vector_path)
    stats["building_source"] = str(vector_path)
    stats["building_count_raw"] = int(len(buildings))

    buildings = buildings[buildings.geometry.notna() & (~buildings.geometry.is_empty)].copy()
    if buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    if buildings.crs is None:
        inferred_crs = infer_missing_building_crs(buildings, utm_crs)
        print(f"[WARN] Building vector has no CRS. Inferred/assuming {inferred_crs}.")
        buildings = buildings.set_crs(inferred_crs)
    buildings = buildings.to_crs(utm_crs)
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    try:
        buildings["geometry"] = buildings.geometry.buffer(0)
    except Exception:
        pass
    buildings = buildings[buildings.geometry.notna() & (~buildings.geometry.is_empty)].copy()

    # Keep only buildings intersecting the XY data-box extent.
    xy_utm = xy_gdf.copy()
    if "geometry_utm" in xy_utm.columns:
        xy_utm = xy_utm.set_geometry("geometry_utm", crs=xy_gdf.crs)
    minx, miny, maxx, maxy = xy_utm.total_bounds
    data_box_geom = box(float(minx), float(miny), float(maxx), float(maxy))
    try:
        idx = buildings.sindex.query(data_box_geom, predicate="intersects")
        buildings = buildings.iloc[np.asarray(idx, dtype=int)].copy()
    except Exception:
        buildings = buildings[buildings.intersects(data_box_geom)].copy()

    if buildings.empty:
        print("[WARN] Building vector is readable, but no building footprints intersect the voxel data box.")
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats
    stats["building_count_in_data_box"] = int(len(buildings))
    try:
        bminx, bminy, bmaxx, bmaxy = buildings.total_bounds
        stats["building_bounds_utm"] = [float(bminx), float(bminy), float(bmaxx), float(bmaxy)]
    except Exception:
        pass

    hcol = find_building_height_column(buildings)
    if hcol is not None:
        buildings["building_height_agl_m"] = pd.to_numeric(buildings[hcol], errors="coerce")
        stats["building_height_source"] = f"vector_column:{hcol}"
    else:
        raster_path = choose_building_height_raster(paths)
        if raster_path is None:
            print("[WARN] Building vector found, but no height column/raster was found. Building burn skipped.")
            return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats
        buildings["building_height_agl_m"] = sample_building_height_raster_at_centroids(buildings, raster_path)
        stats["building_height_source"] = str(raster_path)

    buildings["building_height_agl_m"] = pd.to_numeric(buildings["building_height_agl_m"], errors="coerce")
    n_finite_height = int(np.isfinite(buildings["building_height_agl_m"]).sum())
    stats["building_count_finite_height_before_min_filter"] = n_finite_height
    buildings = buildings[np.isfinite(buildings["building_height_agl_m"])].copy()
    buildings = buildings[buildings["building_height_agl_m"] >= BUILDING_MIN_HEIGHT_M].copy()
    if buildings.empty:
        print(
            "[WARN] Building footprints were found, but no valid building height remained after filtering. "
            f"finite_height_before_filter={n_finite_height:,}, BUILDING_MIN_HEIGHT_M={BUILDING_MIN_HEIGHT_M:g}"
        )
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    buildings["building_id"] = np.arange(len(buildings), dtype=int)
    buildings["building_footprint_area_m2"] = buildings.geometry.area.astype(float)
    buildings["building_volume_m3"] = buildings["building_footprint_area_m2"] * buildings["building_height_agl_m"]
    stats["building_count_valid_height"] = int(len(buildings))
    stats["building_height_min_m"] = float(buildings["building_height_agl_m"].min())
    stats["building_height_p50_m"] = float(buildings["building_height_agl_m"].median())
    stats["building_height_max_m"] = float(buildings["building_height_agl_m"].max())
    stats["building_volume_total_m3"] = float(buildings["building_volume_m3"].sum())

    print(
        "[CHECK] Buildings for volume burn: "
        f"n={len(buildings):,}, height={stats['building_height_min_m']:.2f}.."
        f"{stats['building_height_max_m']:.2f} m AGL, volume={stats['building_volume_total_m3']:.1f} m3"
    )
    if SAVE_BUILDING_BURN_DEBUG_FILES:
        try:
            buildings.to_file(paths.data_dir / "building_volume_valid_footprints_debug.gpkg", driver="GPKG")
            print(f"[OK] Saved building debug footprints: {paths.data_dir / 'building_volume_valid_footprints_debug.gpkg'}")
        except Exception as exc:
            print(f"[WARN] Could not save building debug footprints: {exc}")
    return buildings, stats


def add_building_burn_columns(
    paths: Paths,
    voxels: pd.DataFrame,
    xy_gdf: gpd.GeoDataFrame,
    utm_crs,
    dx: float,
    dy: float,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict]:
    """
    Burn 3D building volumes into the voxel model.

    Building volume equation:
        building_volume = building_footprint_area * building_height_agl
        building_base_msl = DEM terrain_msl
        building_top_msl = terrain_msl + building_height_agl + BUILDING_HEIGHT_BUFFER_M

    A voxel burns if its XY cell intersects a building footprint and its vertical
    interval intersects the building base-to-top interval.
    """
    voxels = voxels.copy()
    xy_gdf = xy_gdf.copy()

    # Defaults, even when no building data are available.
    xy_gdf["building_height_input_m"] = 0.0
    xy_gdf["building_height_reference_used"] = "AGL"
    xy_gdf["building_height_agl_m"] = 0.0
    xy_gdf["building_base_msl_m"] = xy_gdf["terrain_msl_m"].astype(float)
    xy_gdf["building_top_msl_m"] = xy_gdf["terrain_msl_m"].astype(float)
    xy_gdf["building_volume_m3"] = 0.0
    xy_gdf["building_count_in_xy"] = 0
    voxels["building_height_input_m"] = 0.0
    voxels["building_height_reference_used"] = "AGL"
    voxels["building_height_agl_m"] = 0.0
    voxels["building_base_msl_m"] = voxels["terrain_msl_m"].astype(float)
    voxels["building_top_msl_m"] = voxels["terrain_msl_m"].astype(float)
    voxels["burn_building_volume"] = False
    voxels = rebuild_obstacle_union_columns(voxels, context="building_defaults")

    stats = {
        "building_burn_enabled": bool(BURN_BUILDING_VOLUME),
        "building_xy_cells": 0,
        "building_burned_voxels": 0,
    }

    if not BURN_BUILDING_VOLUME:
        print("[INFO] Building-volume burn is disabled.")
        return voxels, xy_gdf, stats

    buildings, load_stats = load_gba_buildings(paths, xy_gdf, utm_crs)
    stats.update(load_stats)
    if buildings.empty:
        print("[WARN] No valid building volumes were loaded. Building burn skipped.")
        return voxels, xy_gdf, stats

    # IMPORTANT:
    # xy_gdf.geometry is local SW geometry after add_sw_reference_coordinates().
    # xy_gdf.geometry_utm is the real UTM geometry used for collision.
    #
    # A previous version selected the column named "geometry" after activating
    # geometry_utm, which dropped the active geometry column and caused:
    #   "GeoDataFrame without an active geometry column"
    #
    # Here we explicitly build a clean GeoDataFrame whose active geometry is
    # the UTM XY-cell geometry. This is the geometry used for building/voxel
    # collision and for estimating the building burned cells.
    xy_utm = xy_gdf.copy()
    if "geometry_utm" in xy_utm.columns:
        xy_geom = gpd.GeoSeries(xy_utm["geometry_utm"], crs=xy_gdf.crs)
    else:
        xy_geom = gpd.GeoSeries(xy_utm.geometry, crs=xy_gdf.crs)

    xy_cells = gpd.GeoDataFrame(
        xy_utm[["xy_id", "terrain_msl_m"]].copy(),
        geometry=xy_geom,
        crs=xy_gdf.crs,
    )

    building_cols = ["building_id", "building_height_agl_m", "building_volume_m3", "geometry"]
    bld = buildings[building_cols].copy()
    bld = gpd.GeoDataFrame(bld, geometry="geometry", crs=buildings.crs).to_crs(xy_cells.crs)

    try:
        joined = gpd.sjoin(xy_cells, bld, how="inner", predicate="intersects")
    except Exception as exc:
        print(f"[WARN] Building/cell spatial join failed: {exc}")
        return voxels, xy_gdf, stats

    if joined.empty:
        print("[WARN] No XY cells intersect building footprints. Building burn skipped.")
        return voxels, xy_gdf, stats

    stats["building_xy_intersection_rows"] = int(len(joined))
    print(f"[CHECK] Building footprint / XY-cell intersection rows: {len(joined):,}")

    by_xy = (
        joined.groupby("xy_id", as_index=False)
        .agg(
            building_height_agl_m=("building_height_agl_m", "max"),
            building_volume_m3=("building_volume_m3", "sum"),
            building_count_in_xy=("building_id", "nunique"),
        )
    )

    xy_maps = by_xy.set_index("xy_id")
    # Preserve the raw height value from the selected source before converting
    # it to a consistent AGL/MSL definition. The selected source column is named
    # building_height_agl_m in load_gba_buildings() for backward compatibility,
    # but at this point it is still the input value.
    xy_gdf["building_height_input_m"] = xy_gdf["xy_id"].map(xy_maps["building_height_agl_m"]).fillna(0).astype(float)
    xy_gdf["building_volume_m3"] = xy_gdf["xy_id"].map(xy_maps["building_volume_m3"]).fillna(0).astype(float)
    xy_gdf["building_count_in_xy"] = xy_gdf["xy_id"].map(xy_maps["building_count_in_xy"]).fillna(0).astype(int)

    xy_gdf, height_ref_stats = resolve_building_height_reference_for_xy(xy_gdf)
    stats.update(height_ref_stats)

    height_input_map = xy_gdf.set_index("xy_id")["building_height_input_m"]
    height_ref_map = xy_gdf.set_index("xy_id")["building_height_reference_used"]
    height_map = xy_gdf.set_index("xy_id")["building_height_agl_m"]
    count_map = xy_gdf.set_index("xy_id")["building_count_in_xy"]
    volume_map = xy_gdf.set_index("xy_id")["building_volume_m3"]
    base_map = xy_gdf.set_index("xy_id")["building_base_msl_m"]
    top_map = xy_gdf.set_index("xy_id")["building_top_msl_m"]

    voxels["building_height_input_m"] = voxels["xy_id"].map(height_input_map).fillna(0).astype(float)
    voxels["building_height_reference_used"] = voxels["xy_id"].map(height_ref_map).fillna("AGL").astype(str)
    voxels["building_height_agl_m"] = voxels["xy_id"].map(height_map).fillna(0).astype(float)
    voxels["building_count_in_xy"] = voxels["xy_id"].map(count_map).fillna(0).astype(int)
    voxels["building_volume_m3"] = voxels["xy_id"].map(volume_map).fillna(0).astype(float)
    voxels["building_base_msl_m"] = voxels["xy_id"].map(base_map).fillna(voxels["terrain_msl_m"]).astype(float)
    voxels["building_top_msl_m"] = voxels["xy_id"].map(top_map).fillna(voxels["terrain_msl_m"]).astype(float)

    has_building_xy = voxels["building_height_agl_m"] >= BUILDING_MIN_HEIGHT_M
    vertical_collision = (
        (voxels["z_top_msl_m"].astype(float) > voxels["building_base_msl_m"].astype(float))
        & (voxels["z_bottom_msl_m"].astype(float) < voxels["building_top_msl_m"].astype(float))
    )
    voxels["burn_building_volume"] = (has_building_xy & vertical_collision).astype(bool)
    voxels = rebuild_obstacle_union_columns(voxels, context="after_building")

    stats["building_xy_cells"] = int((xy_gdf["building_height_agl_m"] >= BUILDING_MIN_HEIGHT_M).sum())
    stats["building_burned_voxels"] = int(voxels["burn_building_volume"].sum())
    stats["building_volume_intersected_xy_m3"] = float(xy_gdf["building_volume_m3"].sum())

    bxy = xy_gdf[xy_gdf["building_height_agl_m"] >= BUILDING_MIN_HEIGHT_M].copy()
    if not bxy.empty:
        stats["building_xy_height_min_m"] = float(bxy["building_height_agl_m"].min())
        stats["building_xy_height_max_m"] = float(bxy["building_height_agl_m"].max())
        stats["building_top_msl_min_m"] = float(bxy["building_top_msl_m"].min())
        stats["building_top_msl_max_m"] = float(bxy["building_top_msl_m"].max())

    print(f"[CHECK] Building-volume XY cells: {stats['building_xy_cells']:,}")
    print(f"[CHECK] Building-volume burned voxels: {stats['building_burned_voxels']:,}")
    if stats["building_burned_voxels"] == 0:
        print(
            "[WARN] Building footprints/heights may be loaded, but no voxel z-interval intersects the building volume. "
            "Check building_top_msl range, z model range, height units, and CRS."
        )

    if SAVE_BUILDING_BURN_DEBUG_FILES and not bxy.empty:
        try:
            bxy_save = bxy.copy()
            if "geometry_utm" in bxy_save.columns:
                bxy_save = gpd.GeoDataFrame(
                    bxy_save.drop(columns=["geometry"], errors="ignore"),
                    geometry=gpd.GeoSeries(bxy_save["geometry_utm"], crs=xy_gdf.crs),
                    crs=xy_gdf.crs,
                )
                bxy_save = bxy_save.drop(columns=["geometry_utm"], errors="ignore")
            bxy_save.to_file(paths.data_dir / "building_xy_intersections_debug.gpkg", driver="GPKG")
            print(f"[OK] Saved building XY debug cells: {paths.data_dir / 'building_xy_intersections_debug.gpkg'}")
        except Exception as exc:
            print(f"[WARN] Could not save building XY debug cells: {exc}")

    return voxels, xy_gdf, stats


# ======================================================================
# OSM POWERLINE / POWER-POLE GEOFENCE BURN
# ======================================================================


def unique_existing_powerline_files(patterns: Iterable[str], roots: Iterable[Path]) -> list[Path]:
    """Find OSM powerline/pole vector candidates."""
    files: list[Path] = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                if not p.is_file():
                    continue
                name = str(p).lower()
                if any(bad in name for bad in [
                    "dem", "elevation", "terrain", "slope", "aspect", "hillshade",
                    "building", "footprint", "population", "road", "traffic",
                ]):
                    continue
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def read_powerline_vector(path: Path) -> gpd.GeoDataFrame:
    """Read OSM powerline/pole vector data from common GIS formats."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq", ".geoparquet"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def infer_missing_powerline_crs(gdf: gpd.GeoDataFrame, target_crs):
    """Infer CRS for OSM powerline vectors when CRS metadata is missing.

    Powerline data copied from OSM is commonly EPSG:4326. Some local clipped
    files may already be UTM. This function prevents the failure mode where
    lon/lat powerline coordinates are accidentally treated as meters, or where
    UTM coordinates are accidentally treated as Web Mercator.
    """
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
    except Exception:
        return target_crs

    # OSM lon/lat degrees.
    if -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0:
        return "EPSG:4326"

    # UTM Zone 48N-like meter coordinates around northern Vietnam.
    if 100000.0 <= minx <= 900000.0 and 100000.0 <= maxx <= 900000.0 and 1000000.0 <= miny <= 3000000.0 and 1000000.0 <= maxy <= 3000000.0:
        return target_crs

    # Web Mercator-like meter coordinates in Vietnam.
    if 1.0e7 <= minx <= 1.3e7 and 1.0e7 <= maxx <= 1.3e7 and 1.0e6 <= miny <= 3.0e6 and 1.0e6 <= maxy <= 3.0e6:
        return "EPSG:3857"

    return target_crs


def find_powerline_height_column(gdf: gpd.GeoDataFrame) -> str | None:
    cols = list(gdf.columns)
    lower_to_real = {str(c).lower(): c for c in cols}
    for cand in OSM_POWERLINE_HEIGHT_COLUMN_CANDIDATES:
        if cand in cols:
            return cand
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    for c in cols:
        lc = str(c).lower()
        if any(key in lc for key in ["height", "agl", "clearance"]):
            if not any(bad in lc for bad in ["elev", "terrain", "dem", "msl"]):
                return c
    return None


def _load_powerline_kind_candidates(
    paths: Paths,
    patterns: Iterable[str],
    kind: str,
    target_crs,
) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Load all candidate vectors for one powerline-system kind.

    kind = "line" for LineString/MultiLineString cables.
    kind = "pole" for Point/MultiPoint poles/towers/pylons.
    """
    candidates = unique_existing_powerline_files(patterns, OSM_POWERLINE_DIR_CANDIDATES)
    rows: list[dict] = []
    parts: list[gpd.GeoDataFrame] = []
    wanted_geom = {"line": {"LineString", "MultiLineString"}, "pole": {"Point", "MultiPoint"}}[kind]

    if not candidates:
        print(f"[WARN] No OSM powerline {kind} vector candidates found.")
        return gpd.GeoDataFrame(geometry=[], crs=target_crs), rows

    print(f"\n========== OSM POWERLINE {kind.upper()} CANDIDATES ==========")
    for i, path in enumerate(candidates):
        row = {"rank": i, "kind": kind, "path": str(path)}
        try:
            gdf = read_powerline_vector(path)
            row["feature_count_raw"] = int(len(gdf))
            row["crs"] = str(gdf.crs)
            row["columns"] = ",".join(map(str, gdf.columns[:40]))
            if gdf.empty or "geometry" not in gdf.columns:
                row["feature_count_used"] = 0
                rows.append(row)
                print(f"[{i:02d}] empty/no geometry | {path}")
                continue

            gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
            if gdf.empty:
                row["feature_count_used"] = 0
                rows.append(row)
                print(f"[{i:02d}] no valid geometry | {path}")
                continue

            try:
                row["bounds_before_crs"] = list(map(float, gdf.total_bounds))
            except Exception:
                pass

            if gdf.crs is None:
                inferred = infer_missing_powerline_crs(gdf, target_crs)
                gdf = gdf.set_crs(inferred)
                row["crs_inferred"] = str(inferred)

            # Always convert the source data to the UTM voxel CRS before
            # buffering and spatial joining. Do not use local SW plot geometry
            # for any collision calculation.
            gdf = gdf.to_crs(target_crs)
            try:
                row["bounds_after_to_utm"] = list(map(float, gdf.total_bounds))
            except Exception:
                pass

            geom_mask = gdf.geometry.geom_type.isin(wanted_geom)
            gdf = gdf[geom_mask].copy()
            if gdf.empty:
                row["feature_count_used"] = 0
                rows.append(row)
                print(f"[{i:02d}] no {kind} geometries | {path}")
                continue

            hcol = find_powerline_height_column(gdf)
            row["height_column"] = hcol
            default_h = POWERLINE_DEFAULT_LINE_HEIGHT_AGL_M if kind == "line" else POWERLINE_DEFAULT_POLE_HEIGHT_AGL_M
            if hcol is None:
                gdf["powerline_height_input_m"] = float(default_h)
                row["height_source"] = f"default:{default_h:g}_m_AGL"
            else:
                gdf["powerline_height_input_m"] = pd.to_numeric(gdf[hcol], errors="coerce")
                gdf["powerline_height_input_m"] = gdf["powerline_height_input_m"].fillna(float(default_h))
                row["height_source"] = f"column:{hcol}; default_if_nan={default_h:g}_m_AGL"

            gdf = gdf[np.isfinite(gdf["powerline_height_input_m"])].copy()
            gdf = gdf[gdf["powerline_height_input_m"] >= float(POWERLINE_MIN_HEIGHT_M)].copy()
            if gdf.empty:
                row["feature_count_used"] = 0
                rows.append(row)
                print(f"[{i:02d}] no valid heights | {path}")
                continue

            gdf["powerline_kind"] = kind
            gdf["powerline_source_file"] = str(path)
            gdf["powerline_feature_id"] = [f"{kind}_{i}_{j}" for j in range(len(gdf))]
            row["feature_count_used"] = int(len(gdf))
            row["height_min_m"] = float(gdf["powerline_height_input_m"].min())
            row["height_p50_m"] = float(gdf["powerline_height_input_m"].median())
            row["height_max_m"] = float(gdf["powerline_height_input_m"].max())
            parts.append(gdf[["powerline_feature_id", "powerline_kind", "powerline_source_file", "powerline_height_input_m", "geometry"]].copy())
            print(
                f"[{i:02d}] n={len(gdf):,} height_col={hcol} "
                f"height={row['height_min_m']:.1f}..{row['height_max_m']:.1f} m | {path}"
            )
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{i:02d}] ERROR | {path} | {exc}")
        rows.append(row)

    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs), rows
    out = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=target_crs), rows


def load_osm_powerline_system(paths: Paths, xy_gdf: gpd.GeoDataFrame, utm_crs) -> tuple[gpd.GeoDataFrame, dict]:
    """Load OSM powerline lines and poles/towers, then create buffered footprints."""
    stats = {
        "powerline_source": "OSM",
        "powerline_line_features": 0,
        "powerline_pole_features": 0,
        "powerline_features_total": 0,
        "powerline_height_reference": POWERLINE_HEIGHT_REFERENCE,
        "powerline_vertical_mode": POWERLINE_BURN_VERTICAL_MODE,
        "powerline_line_vertical_mode": POWERLINE_LINE_BURN_VERTICAL_MODE,
        "powerline_pole_vertical_mode": POWERLINE_POLE_BURN_VERTICAL_MODE,
    }

    lines, line_audit = _load_powerline_kind_candidates(paths, OSM_POWERLINE_LINE_PATTERNS, "line", utm_crs)
    poles, pole_audit = _load_powerline_kind_candidates(paths, OSM_POWERLINE_POLE_PATTERNS, "pole", utm_crs)
    try:
        pd.DataFrame(line_audit).to_csv(paths.data_dir / "powerline_line_candidate_audit.csv", index=False)
        pd.DataFrame(pole_audit).to_csv(paths.data_dir / "powerline_pole_candidate_audit.csv", index=False)
    except Exception as exc:
        print(f"[WARN] Could not save powerline candidate audit CSVs: {exc}")

    parts = []
    if not lines.empty:
        stats["powerline_line_features"] = int(len(lines))
        parts.append(lines)
    if not poles.empty:
        stats["powerline_pole_features"] = int(len(poles))
        parts.append(poles)
    if not parts:
        print("[WARN] No valid OSM powerline/pole features loaded. Powerline burn skipped.")
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    power = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=utm_crs)

    # Keep only features intersecting the XY data box after applying footprint buffers.
    xy_utm = xy_gdf.copy()
    if "geometry_utm" in xy_utm.columns:
        xy_utm = xy_utm.set_geometry("geometry_utm", crs=xy_gdf.crs)
    minx, miny, maxx, maxy = xy_utm.total_bounds
    data_box_geom = box(float(minx), float(miny), float(maxx), float(maxy))

    line_mask = power["powerline_kind"].astype(str) == "line"
    buffer_dist = np.where(line_mask, float(POWERLINE_HORIZONTAL_BUFFER_M), float(POWERLINE_POLE_HORIZONTAL_BUFFER_M))
    power["powerline_horizontal_buffer_m"] = buffer_dist.astype(float)
    power["geometry_raw_utm"] = power.geometry
    power["geometry"] = [geom.buffer(float(buf)) for geom, buf in zip(power.geometry, buffer_dist)]
    power = power[power.geometry.notna() & (~power.geometry.is_empty)].copy()

    try:
        idx = power.sindex.query(data_box_geom, predicate="intersects")
        power = power.iloc[np.asarray(idx, dtype=int)].copy()
    except Exception:
        power = power[power.intersects(data_box_geom)].copy()

    if power.empty:
        print("[WARN] OSM powerline/pole features loaded, but none intersects the voxel data box.")
        return gpd.GeoDataFrame(geometry=[], crs=utm_crs), stats

    power["powerline_obstacle_area_m2"] = power.geometry.area.astype(float)
    stats["powerline_features_total"] = int(len(power))
    stats["powerline_height_input_min_m"] = float(power["powerline_height_input_m"].min())
    stats["powerline_height_input_p50_m"] = float(power["powerline_height_input_m"].median())
    stats["powerline_height_input_max_m"] = float(power["powerline_height_input_m"].max())
    stats["powerline_buffer_line_m"] = float(POWERLINE_HORIZONTAL_BUFFER_M)
    stats["powerline_buffer_pole_m"] = float(POWERLINE_POLE_HORIZONTAL_BUFFER_M)

    print(
        "[CHECK] OSM powerline system for obstacle/geofence burn: "
        f"features={len(power):,}, lines={stats['powerline_line_features']:,}, poles={stats['powerline_pole_features']:,}, "
        f"height={stats['powerline_height_input_min_m']:.1f}..{stats['powerline_height_input_max_m']:.1f} m AGL/input"
    )

    if SAVE_POWERLINE_BURN_DEBUG_FILES:
        try:
            # Save raw UTM line/point geometry before buffering. This is the
            # quickest way to verify CRS conversion and that OSM features land
            # on the same UTM grid as the voxel model.
            if "geometry_raw_utm" in power.columns:
                raw_save = power.copy()
                raw_save = gpd.GeoDataFrame(
                    raw_save.drop(columns=["geometry"], errors="ignore"),
                    geometry=gpd.GeoSeries(raw_save["geometry_raw_utm"], crs=utm_crs),
                    crs=utm_crs,
                )
                raw_save = raw_save.drop(columns=["geometry_raw_utm"], errors="ignore")
                raw_save.to_file(paths.data_dir / "powerline_system_raw_utm_debug.gpkg", driver="GPKG")
                print(f"[OK] Saved powerline raw UTM features: {paths.data_dir / 'powerline_system_raw_utm_debug.gpkg'}")

            save = power.drop(columns=["geometry_raw_utm"], errors="ignore").copy()
            save.to_file(paths.data_dir / "powerline_system_buffered_footprints_debug.gpkg", driver="GPKG")
            print(f"[OK] Saved powerline buffered footprints: {paths.data_dir / 'powerline_system_buffered_footprints_debug.gpkg'}")
        except Exception as exc:
            print(f"[WARN] Could not save powerline debug footprints: {exc}")
    return power, stats


def _convert_powerline_height_to_agl(height_input: pd.Series, terrain: pd.Series) -> tuple[pd.Series, str]:
    """Convert powerline input height to AGL according to POWERLINE_HEIGHT_REFERENCE."""
    h_in = pd.to_numeric(height_input, errors="coerce").fillna(0.0)
    terrain = pd.to_numeric(terrain, errors="coerce").fillna(0.0)
    mode = str(POWERLINE_HEIGHT_REFERENCE).strip().upper()
    if mode not in {"AGL", "MSL_TOP"}:
        raise ValueError('POWERLINE_HEIGHT_REFERENCE must be "AGL" or "MSL_TOP"')
    if mode == "MSL_TOP":
        return (h_in - terrain).clip(lower=0.0), "MSL_TOP"
    return h_in.clip(lower=0.0), "AGL"


def _powerline_vertical_interval_by_kind(
    terrain: pd.Series,
    height_agl: pd.Series,
    kind: str,
) -> tuple[pd.Series, pd.Series, str]:
    """Return base/top MSL for powerline line or pole cells.

    line = aerial cable/strand band only.
    pole = vertical column from ground to pole/tower top.
    """
    terrain = pd.to_numeric(terrain, errors="coerce").fillna(0.0)
    h_agl = pd.to_numeric(height_agl, errors="coerce").fillna(0.0).clip(lower=0.0)
    kind = str(kind).strip().lower()

    if kind == "line":
        mode = str(POWERLINE_LINE_BURN_VERTICAL_MODE).strip().lower()
        vbuf = float(POWERLINE_LINE_VERTICAL_BUFFER_M)
    elif kind == "pole":
        mode = str(POWERLINE_POLE_BURN_VERTICAL_MODE).strip().lower()
        vbuf = float(POWERLINE_POLE_VERTICAL_BUFFER_M)
    else:
        raise ValueError('powerline kind must be "line" or "pole"')

    if mode not in {"from_ground_to_height", "cable_band"}:
        raise ValueError('Powerline vertical mode must be "from_ground_to_height" or "cable_band"')

    if mode == "cable_band":
        base = terrain + h_agl - vbuf
        top = terrain + h_agl + vbuf
        # The cable is airborne but should never be plotted below terrain.
        base = np.maximum(base, terrain)
    else:
        base = terrain
        top = terrain + h_agl + vbuf

    return base.astype(float), top.astype(float), mode


def resolve_powerline_height_reference_for_xy(xy_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Convert OSM powerline input heights to separate line/pole AGL and MSL intervals."""
    out = xy_gdf.copy()
    terrain = pd.to_numeric(out.get("terrain_msl_m", 0.0), errors="coerce").fillna(0.0)

    # Backward-compatible combined input height, plus kind-specific fields.
    line_h_in = pd.to_numeric(out.get("powerline_line_height_input_m", 0.0), errors="coerce").fillna(0.0)
    pole_h_in = pd.to_numeric(out.get("powerline_pole_height_input_m", 0.0), errors="coerce").fillna(0.0)
    combined_h_in = pd.to_numeric(out.get("powerline_height_input_m", 0.0), errors="coerce").fillna(0.0)

    # If old/combined value exists but kind-specific values do not, keep it as line height.
    line_h_in = np.where((line_h_in <= 0.0) & (combined_h_in > 0.0), combined_h_in, line_h_in)
    line_h_in = pd.Series(line_h_in, index=out.index, dtype="float64")
    pole_h_in = pd.Series(pole_h_in, index=out.index, dtype="float64")

    line_h_agl, ref_used = _convert_powerline_height_to_agl(line_h_in, terrain)
    pole_h_agl, _ = _convert_powerline_height_to_agl(pole_h_in, terrain)

    line_base, line_top, line_mode = _powerline_vertical_interval_by_kind(terrain, line_h_agl, "line")
    pole_base, pole_top, pole_mode = _powerline_vertical_interval_by_kind(terrain, pole_h_agl, "pole")

    has_line = line_h_agl >= float(POWERLINE_MIN_HEIGHT_M)
    has_pole = pole_h_agl >= float(POWERLINE_MIN_HEIGHT_M)

    out["powerline_height_reference_used"] = ref_used
    out["powerline_line_height_input_m"] = line_h_in.astype(float)
    out["powerline_pole_height_input_m"] = pole_h_in.astype(float)
    out["powerline_line_height_agl_m"] = line_h_agl.astype(float)
    out["powerline_pole_height_agl_m"] = pole_h_agl.astype(float)
    out["powerline_line_base_msl_m"] = line_base.astype(float)
    out["powerline_line_top_msl_m"] = line_top.astype(float)
    out["powerline_pole_base_msl_m"] = pole_base.astype(float)
    out["powerline_pole_top_msl_m"] = pole_top.astype(float)
    out["has_powerline_line_xy"] = has_line.astype(bool)
    out["has_powerline_pole_xy"] = has_pole.astype(bool)

    # Combined compatibility fields represent the union interval only for summary/legacy plotting.
    out["powerline_height_agl_m"] = np.maximum(line_h_agl, pole_h_agl).astype(float)
    out["powerline_base_msl_m"] = np.where(
        has_line & has_pole,
        np.minimum(line_base, pole_base),
        np.where(has_line, line_base, np.where(has_pole, pole_base, terrain)),
    ).astype(float)
    out["powerline_top_msl_m"] = np.where(
        has_line & has_pole,
        np.maximum(line_top, pole_top),
        np.where(has_line, line_top, np.where(has_pole, pole_top, terrain)),
    ).astype(float)

    kind_label = np.full(len(out), "none", dtype=object)
    kind_label = np.where(has_line, "line", kind_label)
    kind_label = np.where(has_pole, "pole", kind_label)
    kind_label = np.where(has_line & has_pole, "line+pole", kind_label)
    out["powerline_kind_in_xy"] = kind_label

    valid_combined = (line_h_agl >= float(POWERLINE_MIN_HEIGHT_M)) | (pole_h_agl >= float(POWERLINE_MIN_HEIGHT_M))
    stats = {
        "powerline_height_reference_used": ref_used,
        "powerline_line_vertical_mode_used": line_mode,
        "powerline_pole_vertical_mode_used": pole_mode,
        "powerline_line_xy_height_min_m": float(line_h_agl[has_line].min()) if has_line.any() else 0.0,
        "powerline_line_xy_height_p50_m": float(line_h_agl[has_line].median()) if has_line.any() else 0.0,
        "powerline_line_xy_height_max_m": float(line_h_agl[has_line].max()) if has_line.any() else 0.0,
        "powerline_pole_xy_height_min_m": float(pole_h_agl[has_pole].min()) if has_pole.any() else 0.0,
        "powerline_pole_xy_height_p50_m": float(pole_h_agl[has_pole].median()) if has_pole.any() else 0.0,
        "powerline_pole_xy_height_max_m": float(pole_h_agl[has_pole].max()) if has_pole.any() else 0.0,
        "powerline_xy_height_min_m": float(out.loc[valid_combined, "powerline_height_agl_m"].min()) if valid_combined.any() else 0.0,
        "powerline_xy_height_p50_m": float(out.loc[valid_combined, "powerline_height_agl_m"].median()) if valid_combined.any() else 0.0,
        "powerline_xy_height_max_m": float(out.loc[valid_combined, "powerline_height_agl_m"].max()) if valid_combined.any() else 0.0,
    }
    return out, stats


def add_powerline_system_burn_columns(
    paths: Paths,
    voxels: pd.DataFrame,
    xy_gdf: gpd.GeoDataFrame,
    utm_crs,
    dx: float,
    dy: float,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict]:
    """Burn OSM powerline line strands and pole/tower volumes into the voxel model.

    The two OSM powerline system components use different vertical collision rules:
      - line/strand/cable: an airborne cable band only.
      - pole/tower/pylon: a vertical column from ground/topography to pole top.
    """
    voxels = voxels.copy()
    xy_gdf = xy_gdf.copy()

    # Defaults, even when no OSM powerline data are available.
    default_xy_cols = {
        "powerline_height_input_m": 0.0,
        "powerline_line_height_input_m": 0.0,
        "powerline_pole_height_input_m": 0.0,
        "powerline_height_agl_m": 0.0,
        "powerline_line_height_agl_m": 0.0,
        "powerline_pole_height_agl_m": 0.0,
        "powerline_count_in_xy": 0,
        "powerline_line_count_in_xy": 0,
        "powerline_pole_count_in_xy": 0,
        "powerline_obstacle_area_m2": 0.0,
        "powerline_line_obstacle_area_m2": 0.0,
        "powerline_pole_obstacle_area_m2": 0.0,
    }
    for col, val in default_xy_cols.items():
        xy_gdf[col] = val
    xy_gdf["powerline_height_reference_used"] = POWERLINE_HEIGHT_REFERENCE
    xy_gdf["powerline_kind_in_xy"] = "none"
    xy_gdf["has_powerline_line_xy"] = False
    xy_gdf["has_powerline_pole_xy"] = False
    for col in [
        "powerline_base_msl_m", "powerline_top_msl_m",
        "powerline_line_base_msl_m", "powerline_line_top_msl_m",
        "powerline_pole_base_msl_m", "powerline_pole_top_msl_m",
    ]:
        xy_gdf[col] = xy_gdf["terrain_msl_m"].astype(float)

    # Voxel defaults.
    for col, val in default_xy_cols.items():
        voxels[col] = val
    voxels["powerline_height_reference_used"] = POWERLINE_HEIGHT_REFERENCE
    voxels["powerline_kind_in_xy"] = "none"
    voxels["has_powerline_line_xy"] = False
    voxels["has_powerline_pole_xy"] = False
    for col in [
        "powerline_base_msl_m", "powerline_top_msl_m",
        "powerline_line_base_msl_m", "powerline_line_top_msl_m",
        "powerline_pole_base_msl_m", "powerline_pole_top_msl_m",
    ]:
        voxels[col] = voxels["terrain_msl_m"].astype(float)
    voxels["burn_powerline_line_air"] = False
    voxels["burn_powerline_pole_column"] = False
    voxels["burn_powerline_system"] = False

    stats = {
        "powerline_burn_enabled": bool(BURN_POWERLINE_SYSTEM),
        "powerline_xy_cells": 0,
        "powerline_line_xy_cells": 0,
        "powerline_pole_xy_cells": 0,
        "powerline_burned_voxels": 0,
        "powerline_line_burned_voxels": 0,
        "powerline_pole_burned_voxels": 0,
    }

    if not BURN_POWERLINE_SYSTEM:
        print("[INFO] OSM powerline-system burn is disabled.")
        voxels = rebuild_obstacle_union_columns(voxels, context="powerline_disabled")
        return voxels, xy_gdf, stats

    power, load_stats = load_osm_powerline_system(paths, xy_gdf, utm_crs)
    stats.update(load_stats)
    if power.empty:
        print("[WARN] No valid OSM powerline/pole geofence volumes loaded. Powerline burn skipped.")
        voxels = rebuild_obstacle_union_columns(voxels, context="powerline_disabled")
        return voxels, xy_gdf, stats

    # IMPORTANT: use the real UTM XY-cell geometry for collision.
    # xy_gdf.geometry may be local SW plot geometry after add_sw_reference_coordinates().
    xy_utm = xy_gdf.copy()
    if "geometry_utm" in xy_utm.columns:
        xy_geom = gpd.GeoSeries(xy_utm["geometry_utm"], crs=xy_gdf.crs)
    else:
        xy_geom = gpd.GeoSeries(xy_utm.geometry, crs=xy_gdf.crs)
    xy_cells = gpd.GeoDataFrame(
        xy_utm[["xy_id", "terrain_msl_m"]].copy(),
        geometry=xy_geom,
        crs=xy_gdf.crs,
    )

    power_cols = [
        "powerline_feature_id", "powerline_kind", "powerline_height_input_m",
        "powerline_obstacle_area_m2", "powerline_horizontal_buffer_m", "geometry",
    ]
    pwr = power[power_cols].copy()
    pwr = gpd.GeoDataFrame(pwr, geometry="geometry", crs=power.crs).to_crs(xy_cells.crs)

    try:
        joined = gpd.sjoin(xy_cells, pwr, how="inner", predicate="intersects")
    except Exception as exc:
        print(f"[WARN] Powerline/cell spatial join failed: {exc}")
        return voxels, xy_gdf, stats

    if joined.empty:
        print("[WARN] No XY cells intersect OSM powerline/pole buffered footprints. Powerline burn skipped.")
        return voxels, xy_gdf, stats

    stats["powerline_xy_intersection_rows"] = int(len(joined))
    print(f"[CHECK] Powerline/pole footprint / XY-cell intersection rows: {len(joined):,}")

    # Separate line strand and pole/tower attributes before collapsing to XY.
    joined["is_powerline_line"] = joined["powerline_kind"].astype(str).str.lower().eq("line")
    joined["is_powerline_pole"] = joined["powerline_kind"].astype(str).str.lower().eq("pole")
    joined["line_height_tmp"] = np.where(joined["is_powerline_line"], joined["powerline_height_input_m"], np.nan)
    joined["pole_height_tmp"] = np.where(joined["is_powerline_pole"], joined["powerline_height_input_m"], np.nan)
    joined["line_area_tmp"] = np.where(joined["is_powerline_line"], joined["powerline_obstacle_area_m2"], 0.0)
    joined["pole_area_tmp"] = np.where(joined["is_powerline_pole"], joined["powerline_obstacle_area_m2"], 0.0)

    by_xy = (
        joined.groupby("xy_id", as_index=False)
        .agg(
            powerline_height_input_m=("powerline_height_input_m", "max"),
            powerline_line_height_input_m=("line_height_tmp", "max"),
            powerline_pole_height_input_m=("pole_height_tmp", "max"),
            powerline_count_in_xy=("powerline_feature_id", "nunique"),
            powerline_line_count_in_xy=("is_powerline_line", "sum"),
            powerline_pole_count_in_xy=("is_powerline_pole", "sum"),
            powerline_obstacle_area_m2=("powerline_obstacle_area_m2", "sum"),
            powerline_line_obstacle_area_m2=("line_area_tmp", "sum"),
            powerline_pole_obstacle_area_m2=("pole_area_tmp", "sum"),
        )
    )
    xy_maps = by_xy.set_index("xy_id")
    for col in [
        "powerline_height_input_m", "powerline_line_height_input_m", "powerline_pole_height_input_m",
        "powerline_obstacle_area_m2", "powerline_line_obstacle_area_m2", "powerline_pole_obstacle_area_m2",
    ]:
        xy_gdf[col] = xy_gdf["xy_id"].map(xy_maps[col]).fillna(0).astype(float)
    for col in ["powerline_count_in_xy", "powerline_line_count_in_xy", "powerline_pole_count_in_xy"]:
        xy_gdf[col] = xy_gdf["xy_id"].map(xy_maps[col]).fillna(0).astype(int)

    xy_gdf, height_stats = resolve_powerline_height_reference_for_xy(xy_gdf)
    stats.update(height_stats)

    maps = xy_gdf.set_index("xy_id")
    map_cols = [
        "powerline_height_input_m", "powerline_line_height_input_m", "powerline_pole_height_input_m",
        "powerline_height_reference_used", "powerline_kind_in_xy",
        "powerline_height_agl_m", "powerline_line_height_agl_m", "powerline_pole_height_agl_m",
        "powerline_base_msl_m", "powerline_top_msl_m",
        "powerline_line_base_msl_m", "powerline_line_top_msl_m",
        "powerline_pole_base_msl_m", "powerline_pole_top_msl_m",
        "powerline_count_in_xy", "powerline_line_count_in_xy", "powerline_pole_count_in_xy",
        "powerline_obstacle_area_m2", "powerline_line_obstacle_area_m2", "powerline_pole_obstacle_area_m2",
        "has_powerline_line_xy", "has_powerline_pole_xy",
    ]
    for col in map_cols:
        if col in {"powerline_height_reference_used", "powerline_kind_in_xy"}:
            voxels[col] = voxels["xy_id"].map(maps[col]).fillna("none" if col == "powerline_kind_in_xy" else POWERLINE_HEIGHT_REFERENCE).astype(str)
        elif col in {"powerline_count_in_xy", "powerline_line_count_in_xy", "powerline_pole_count_in_xy"}:
            voxels[col] = voxels["xy_id"].map(maps[col]).fillna(0).astype(int)
        elif col in {"has_powerline_line_xy", "has_powerline_pole_xy"}:
            voxels[col] = voxels["xy_id"].map(maps[col]).fillna(False).astype(bool)
        else:
            voxels[col] = voxels["xy_id"].map(maps[col]).fillna(0).astype(float)

    # Fill missing base/top with terrain for non-powerline cells.
    no_line = ~voxels["has_powerline_line_xy"].astype(bool)
    no_pole = ~voxels["has_powerline_pole_xy"].astype(bool)
    for base_col, top_col, mask in [
        ("powerline_line_base_msl_m", "powerline_line_top_msl_m", no_line),
        ("powerline_pole_base_msl_m", "powerline_pole_top_msl_m", no_pole),
        ("powerline_base_msl_m", "powerline_top_msl_m", no_line & no_pole),
    ]:
        voxels.loc[mask, base_col] = voxels.loc[mask, "terrain_msl_m"]
        voxels.loc[mask, top_col] = voxels.loc[mask, "terrain_msl_m"]

    # Topography-aware powerline source interval.
    # burn_dem_terrain already marks the ground/topography voxel column as no-fly.
    # Therefore the powerline source should only add/plot cells above that local
    # topo-burned column.  This prevents the purple powerline layer from being
    # classified or displayed inside gray topography cells in Figure 02.
    if "burn_dem_terrain" in voxels.columns:
        topo_top_by_xy = (
            voxels.loc[voxels["burn_dem_terrain"].astype(bool)]
            .groupby("xy_id")["z_top_msl_m"]
            .max()
        )
    else:
        topo_top_by_xy = pd.Series(dtype="float64")

    voxels["topo_burn_top_msl_m"] = voxels["xy_id"].map(topo_top_by_xy)
    voxels["topo_burn_top_msl_m"] = pd.to_numeric(
        voxels["topo_burn_top_msl_m"], errors="coerce"
    ).fillna(pd.to_numeric(voxels["terrain_msl_m"], errors="coerce")).astype(float)

    xy_gdf["topo_burn_top_msl_m"] = xy_gdf["xy_id"].map(topo_top_by_xy)
    xy_gdf["topo_burn_top_msl_m"] = pd.to_numeric(
        xy_gdf["topo_burn_top_msl_m"], errors="coerce"
    ).fillna(pd.to_numeric(xy_gdf["terrain_msl_m"], errors="coerce")).astype(float)

    if bool(POWERLINE_RESPECT_TOPO_BURN_TOP):
        topo_floor_v = voxels["topo_burn_top_msl_m"].astype(float) + float(POWERLINE_TOPO_BURN_TOP_CLEARANCE_M)
        topo_floor_xy = xy_gdf["topo_burn_top_msl_m"].astype(float) + float(POWERLINE_TOPO_BURN_TOP_CLEARANCE_M)
    else:
        topo_floor_v = pd.to_numeric(voxels["terrain_msl_m"], errors="coerce").astype(float)
        topo_floor_xy = pd.to_numeric(xy_gdf["terrain_msl_m"], errors="coerce").astype(float)

    voxels["powerline_line_effective_base_msl_m"] = np.maximum(
        pd.to_numeric(voxels["powerline_line_base_msl_m"], errors="coerce").astype(float),
        topo_floor_v,
    )
    voxels["powerline_pole_effective_base_msl_m"] = np.maximum(
        pd.to_numeric(voxels["powerline_pole_base_msl_m"], errors="coerce").astype(float),
        topo_floor_v,
    )
    voxels["powerline_effective_base_msl_m"] = np.where(
        voxels["has_powerline_line_xy"].astype(bool) & voxels["has_powerline_pole_xy"].astype(bool),
        np.minimum(voxels["powerline_line_effective_base_msl_m"], voxels["powerline_pole_effective_base_msl_m"]),
        np.where(
            voxels["has_powerline_line_xy"].astype(bool),
            voxels["powerline_line_effective_base_msl_m"],
            np.where(voxels["has_powerline_pole_xy"].astype(bool), voxels["powerline_pole_effective_base_msl_m"], voxels["terrain_msl_m"]),
        ),
    ).astype(float)

    xy_gdf["powerline_line_effective_base_msl_m"] = np.maximum(
        pd.to_numeric(xy_gdf["powerline_line_base_msl_m"], errors="coerce").astype(float),
        topo_floor_xy,
    )
    xy_gdf["powerline_pole_effective_base_msl_m"] = np.maximum(
        pd.to_numeric(xy_gdf["powerline_pole_base_msl_m"], errors="coerce").astype(float),
        topo_floor_xy,
    )
    xy_gdf["powerline_effective_base_msl_m"] = np.where(
        xy_gdf["has_powerline_line_xy"].astype(bool) & xy_gdf["has_powerline_pole_xy"].astype(bool),
        np.minimum(xy_gdf["powerline_line_effective_base_msl_m"], xy_gdf["powerline_pole_effective_base_msl_m"]),
        np.where(
            xy_gdf["has_powerline_line_xy"].astype(bool),
            xy_gdf["powerline_line_effective_base_msl_m"],
            np.where(xy_gdf["has_powerline_pole_xy"].astype(bool), xy_gdf["powerline_pole_effective_base_msl_m"], xy_gdf["terrain_msl_m"]),
        ),
    ).astype(float)

    line_collision = (
        voxels["has_powerline_line_xy"].astype(bool)
        & (voxels["powerline_line_top_msl_m"].astype(float) > voxels["powerline_line_effective_base_msl_m"].astype(float))
        & (voxels["z_top_msl_m"].astype(float) > voxels["powerline_line_effective_base_msl_m"].astype(float))
        & (voxels["z_bottom_msl_m"].astype(float) < voxels["powerline_line_top_msl_m"].astype(float))
    )
    pole_collision = (
        voxels["has_powerline_pole_xy"].astype(bool)
        & (voxels["powerline_pole_top_msl_m"].astype(float) > voxels["powerline_pole_effective_base_msl_m"].astype(float))
        & (voxels["z_top_msl_m"].astype(float) > voxels["powerline_pole_effective_base_msl_m"].astype(float))
        & (voxels["z_bottom_msl_m"].astype(float) < voxels["powerline_pole_top_msl_m"].astype(float))
    )
    voxels["burn_powerline_line_air"] = line_collision.astype(bool)
    voxels["burn_powerline_pole_column"] = pole_collision.astype(bool)
    voxels["burn_powerline_system"] = (line_collision | pole_collision).astype(bool)

    voxels = rebuild_obstacle_union_columns(voxels, context="after_powerline")

    stats["powerline_line_xy_cells"] = int(xy_gdf["has_powerline_line_xy"].astype(bool).sum())
    stats["powerline_pole_xy_cells"] = int(xy_gdf["has_powerline_pole_xy"].astype(bool).sum())
    stats["powerline_xy_cells"] = int((xy_gdf["has_powerline_line_xy"].astype(bool) | xy_gdf["has_powerline_pole_xy"].astype(bool)).sum())
    stats["powerline_line_burned_voxels"] = int(voxels["burn_powerline_line_air"].sum())
    stats["powerline_pole_burned_voxels"] = int(voxels["burn_powerline_pole_column"].sum())
    stats["powerline_burned_voxels"] = int(voxels["burn_powerline_system"].sum())
    if stats["powerline_xy_cells"] > 0:
        pxy = xy_gdf[xy_gdf["has_powerline_line_xy"].astype(bool) | xy_gdf["has_powerline_pole_xy"].astype(bool)].copy()
        stats["powerline_top_msl_min_m"] = float(pxy["powerline_top_msl_m"].min())
        stats["powerline_top_msl_max_m"] = float(pxy["powerline_top_msl_m"].max())

    if stats["powerline_xy_cells"] > 0:
        pxy_eff = xy_gdf[xy_gdf["has_powerline_line_xy"].astype(bool) | xy_gdf["has_powerline_pole_xy"].astype(bool)].copy()
        try:
            print(
                "[CHECK] Powerline topo-aware effective base: "
                f"respect_topo={POWERLINE_RESPECT_TOPO_BURN_TOP}, "
                f"topo_burn_top_MSL p50/max={pxy_eff['topo_burn_top_msl_m'].median():.2f}/{pxy_eff['topo_burn_top_msl_m'].max():.2f} m, "
                f"line_effective_base_MSL max={pxy_eff['powerline_line_effective_base_msl_m'].max():.2f} m, "
                f"pole_effective_base_MSL max={pxy_eff['powerline_pole_effective_base_msl_m'].max():.2f} m"
            )
            if bool(SAVE_POWERLINE_TOPO_AWARE_INTERVAL_QC):
                try:
                    qc_cols = [
                        "xy_id", "x_from_sw_m", "y_from_sw_m", "terrain_msl_m", "topo_burn_top_msl_m",
                        "powerline_kind_in_xy",
                        "powerline_line_height_agl_m", "powerline_line_base_msl_m",
                        "powerline_line_effective_base_msl_m", "powerline_line_top_msl_m",
                        "powerline_pole_height_agl_m", "powerline_pole_base_msl_m",
                        "powerline_pole_effective_base_msl_m", "powerline_pole_top_msl_m",
                    ]
                    qc_cols = [c for c in qc_cols if c in pxy_eff.columns]
                    pxy_eff[qc_cols].to_csv(paths.data_dir / "powerline_topoaware_interval_qc.csv", index=False)
                    print(f"[OK] Saved powerline topo-aware interval QC: {paths.data_dir / 'powerline_topoaware_interval_qc.csv'}")
                except Exception as exc:
                    print(f"[WARN] Could not save powerline topo-aware interval QC: {exc}")
        except Exception:
            pass
    print(f"[CHECK] Powerline-system XY cells: {stats['powerline_xy_cells']:,} (line={stats['powerline_line_xy_cells']:,}, pole={stats['powerline_pole_xy_cells']:,})")
    print(f"[CHECK] Powerline-system burned voxels: {stats['powerline_burned_voxels']:,} (line_air={stats['powerline_line_burned_voxels']:,}, pole_column={stats['powerline_pole_burned_voxels']:,})")

    if SAVE_POWERLINE_BURN_DEBUG_FILES:
        try:
            pxy = xy_gdf[xy_gdf["has_powerline_line_xy"].astype(bool) | xy_gdf["has_powerline_pole_xy"].astype(bool)].copy()
            if not pxy.empty:
                pxy_save = pxy.copy()
                if "geometry_utm" in pxy_save.columns:
                    pxy_save = gpd.GeoDataFrame(
                        pxy_save.drop(columns=["geometry"], errors="ignore"),
                        geometry=gpd.GeoSeries(pxy_save["geometry_utm"], crs=xy_gdf.crs),
                        crs=xy_gdf.crs,
                    )
                    pxy_save = pxy_save.drop(columns=["geometry_utm"], errors="ignore")
                pxy_save.to_file(paths.data_dir / "powerline_xy_intersections_debug.gpkg", driver="GPKG")
                print(f"[OK] Saved powerline XY debug cells: {paths.data_dir / 'powerline_xy_intersections_debug.gpkg'}")
        except Exception as exc:
            print(f"[WARN] Could not save powerline XY debug cells: {exc}")

    return voxels, xy_gdf, stats


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



def rebuild_obstacle_union_columns(voxels: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Rebuild final obstacle-source burn unions from separated burn masks.

    New logic used in this script:
      - DEM/topography burn is NOT an obstacle-volume source. It is terrain.
      - Buildings and powerline poles/towers are solid ground-based columns:
            base_msl = terrain_msl
            top_msl  = terrain_msl + height_agl
      - Powerline strands/cables are airborne bands only:
            base_msl = terrain_msl + strand_height_agl - vertical_buffer
            top_msl  = terrain_msl + strand_height_agl + vertical_buffer
        so the air below the cable remains open unless another obstacle burns it.
      - Only after those three physical masks are computed do we merge them:
            burn_obstacle_volume = building OR pole_column OR strand_air
      - A voxel becomes no-fly if it collides with terrain OR any obstacle.

    This helper is safe to call many times; it creates missing masks as False.
    """
    out = voxels.copy()

    for col in [
        "burn_dem_terrain",
        "burn_building_volume",
        "burn_powerline_line_air",
        "burn_powerline_pole_column",
        "burn_powerline_system",
    ]:
        if col not in out.columns:
            out[col] = False
        out[col] = pd.Series(out[col], index=out.index).fillna(False).astype(bool)

    # Powerline system is the union of the two kind-specific powerline burns.
    out["burn_powerline_system"] = (
        out["burn_powerline_line_air"].astype(bool)
        | out["burn_powerline_pole_column"].astype(bool)
    )

    # Obstacle-only unions.  Topography is deliberately kept separate.
    out["burn_solid_obstacle_column"] = (
        out["burn_building_volume"].astype(bool)
        | out["burn_powerline_pole_column"].astype(bool)
    )
    out["burn_airborne_obstacle_band"] = out["burn_powerline_line_air"].astype(bool)
    out["burn_building_powerline_obstacle"] = (
        out["burn_solid_obstacle_column"].astype(bool)
        | out["burn_airborne_obstacle_band"].astype(bool)
    )

    # Backward-compatible name used by downstream code.  From this point on it
    # means obstacle volume only, not topography.
    out["burn_obstacle_volume"] = out["burn_building_powerline_obstacle"].astype(bool)

    # Terrain + obstacle union used for final no-fly logic.
    out["burn_topo_or_obstacle_volume"] = (
        out["burn_dem_terrain"].astype(bool)
        | out["burn_obstacle_volume"].astype(bool)
    )

    # QC source label.  Later labels win where masks overlap, so a real
    # collision with a cable/pole can be identified even if the same voxel was
    # also building-burned.
    source = np.full(len(out), "none", dtype=object)
    source = np.where(out["burn_dem_terrain"], "topo", source)
    source = np.where(out["burn_building_volume"], "building", source)
    source = np.where(out["burn_powerline_pole_column"], "powerline_pole_column", source)
    source = np.where(out["burn_powerline_line_air"], "powerline_line_air", source)
    out["burn_source_priority"] = source

    if context:
        print(
            f"[CHECK] Obstacle union rebuilt ({context}): "
            f"topo={int(out['burn_dem_terrain'].sum()):,}, "
            f"building={int(out['burn_building_volume'].sum()):,}, "
            f"pole_column={int(out['burn_powerline_pole_column'].sum()):,}, "
            f"line_air={int(out['burn_powerline_line_air'].sum()):,}, "
            f"obstacle_only={int(out['burn_obstacle_volume'].sum()):,}, "
            f"topo_or_obstacle={int(out['burn_topo_or_obstacle_volume'].sum()):,}"
        )
    return out


def finalize_dem_only_model(voxels: pd.DataFrame, flyable_slowness: float) -> pd.DataFrame:
    """
    Final DEM-only no-fly model.

    Important safety rule:
        final_nofly_dem_only = outside_polygon OR base_nofly OR burn_dem_terrain

    This means outside-AOI cells are treated as burned/no-fly in the output
    model and in all check figures.
    """
    voxels = voxels.copy()
    # Rebuild all source masks with the new separated obstacle logic.
    voxels = rebuild_obstacle_union_columns(voxels, context="finalize_start")

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

    # Rebuild once more after outside/base flags are known.  Obstacle volume
    # stays building/pole/strand only; topography remains a separate terrain mask.
    voxels = rebuild_obstacle_union_columns(voxels, context="finalize_after_boundary")

    # Final safety/no-fly state used for pathfinding and for figures.
    voxels["final_nofly_dem_only"] = (
        voxels["base_nofly_input"]
        | voxels["burn_outside_polygon"]
        | voxels["burn_dem_terrain"].astype(bool)
        | voxels["burn_obstacle_volume"].astype(bool)
    )
    voxels["final_flyable_dem_only"] = (~voxels["final_nofly_dem_only"]).astype(int)
    voxels["slowness_final_dem_only"] = np.where(
        voxels["final_nofly_dem_only"],
        NOFLY_SLOWNESS,
        flyable_slowness,
    )
    # Explicit alias for this updated DEM + building product.
    voxels["slowness_final_dem_building"] = voxels["slowness_final_dem_only"]

    label = np.full(len(voxels), "flyable", dtype=object)
    label = np.where(voxels["base_nofly_input"], "nofly_base", label)
    label = np.where(voxels["burn_dem_terrain"], "nofly_dem_terrain", label)
    label = np.where(voxels["burn_building_volume"], "nofly_building_volume", label)
    label = np.where(voxels["burn_powerline_pole_column"], "nofly_powerline_pole_column", label)
    label = np.where(voxels["burn_powerline_line_air"], "nofly_powerline_line_air", label)
    # Outside AOI wins as the displayed reason because it is a hard boundary.
    label = np.where(voxels["burn_outside_polygon"], "nofly_outside_polygon", label)
    voxels["label_final_dem_only"] = label

    print(f"[CHECK] Outside-polygon no-fly voxels: {int(voxels['burn_outside_polygon'].sum()):,}")
    print(f"[CHECK] Topo/DEM terrain no-fly voxels: {int(voxels['burn_dem_terrain'].sum()):,}")
    print(f"[CHECK] Building-volume obstacle voxels: {int(voxels['burn_building_volume'].sum()):,}")
    print(f"[CHECK] Powerline pole-column obstacle voxels: {int(voxels['burn_powerline_pole_column'].sum()):,}")
    print(f"[CHECK] Powerline line-air obstacle voxels: {int(voxels['burn_powerline_line_air'].sum()):,}")
    print(f"[CHECK] Obstacle-only voxels: {int(voxels['burn_obstacle_volume'].sum()):,}")
    print(f"[CHECK] Final DEM/building/powerline no-fly voxels: {int(voxels['final_nofly_dem_only'].sum()):,}")
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

    main_csv = data_dir / f"{FINAL_OUTPUT_MODEL_BASENAME}.csv.gz"
    voxels.to_csv(main_csv, index=False, compression="gzip")
    print(f"[OK] Saved: {main_csv}")

    try:
        main_parquet = data_dir / f"{FINAL_OUTPUT_MODEL_BASENAME}.parquet"
        voxels.to_parquet(main_parquet, index=False)
        print(f"[OK] Saved: {main_parquet}")
    except Exception as exc:
        print(f"[WARN] Could not save parquet: {exc}")

    xyz_cols = ["lon", "lat", "z_center_msl_m", "slowness_final_dem_only", "label_final_dem_only"]
    if all(c in voxels.columns for c in xyz_cols):
        xyz = voxels[xyz_cols].copy()
        xyz.to_csv(
            data_dir / f"{FINAL_OUTPUT_MODEL_BASENAME}.xyz",
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )
        print(f"[OK] Saved: {data_dir / f'{FINAL_OUTPUT_MODEL_BASENAME}.xyz'}")

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
        "building_burned_voxels": int(voxels.get("burn_building_volume", pd.Series(False, index=voxels.index)).sum()),
        "powerline_pole_column_burned_voxels": int(voxels.get("burn_powerline_pole_column", pd.Series(False, index=voxels.index)).sum()),
        "powerline_line_air_burned_voxels": int(voxels.get("burn_powerline_line_air", pd.Series(False, index=voxels.index)).sum()),
        "powerline_burned_voxels": int(voxels.get("burn_powerline_system", pd.Series(False, index=voxels.index)).sum()),
        "solid_obstacle_burned_voxels": int(voxels.get("burn_solid_obstacle_column", pd.Series(False, index=voxels.index)).sum()),
        "airborne_obstacle_burned_voxels": int(voxels.get("burn_airborne_obstacle_band", pd.Series(False, index=voxels.index)).sum()),
        "obstacle_only_burned_voxels": int(voxels.get("burn_obstacle_volume", pd.Series(False, index=voxels.index)).sum()),
        "topo_or_obstacle_burned_voxels": int(voxels.get("burn_topo_or_obstacle_volume", pd.Series(False, index=voxels.index)).sum()),
        "building_xy_cells": int((pd.to_numeric(xy_gdf.get("building_height_agl_m", pd.Series(0, index=xy_gdf.index)), errors="coerce").fillna(0) >= BUILDING_MIN_HEIGHT_M).sum()),
        "powerline_xy_cells": int((pd.to_numeric(xy_gdf.get("powerline_height_agl_m", pd.Series(0, index=xy_gdf.index)), errors="coerce").fillna(0) >= POWERLINE_MIN_HEIGHT_M).sum()),
        "building_height_input_reference": BUILDING_HEIGHT_INPUT_REFERENCE,
        "building_height_source_reference_note": GBA_HEIGHT_REFERENCE,
        "powerline_system_burn_enabled": bool(BURN_POWERLINE_SYSTEM),
        "powerline_height_reference": POWERLINE_HEIGHT_REFERENCE,
        "powerline_vertical_mode": POWERLINE_BURN_VERTICAL_MODE,
        "powerline_line_vertical_mode": POWERLINE_LINE_BURN_VERTICAL_MODE,
        "powerline_pole_vertical_mode": POWERLINE_POLE_BURN_VERTICAL_MODE,
        "powerline_line_buffer_m": POWERLINE_HORIZONTAL_BUFFER_M,
        "powerline_pole_buffer_m": POWERLINE_POLE_HORIZONTAL_BUFFER_M,
        "powerline_respect_topo_burn_top": bool(POWERLINE_RESPECT_TOPO_BURN_TOP),
        "powerline_topo_burn_top_clearance_m": float(POWERLINE_TOPO_BURN_TOP_CLEARANCE_M),
        "clip_3d_building_burn_plot_to_true_volume": bool(CLIP_3D_BUILDING_BURN_PLOT_TO_TRUE_VOLUME),
        "clip_3d_topo_burn_plot_to_terrain_surface": bool(CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE),
        "plot_3d_building_stack_on_topo": bool(PLOT_3D_BUILDING_STACK_ON_TOPO),
        "plot_3d_building_fill_gap_above_topo": bool(PLOT_3D_BUILDING_FILL_GAP_ABOVE_TOPO),
        "plot_3d_building_stack_base_mode": PLOT_3D_BUILDING_STACK_BASE_MODE,
        "final_flyable_dem_only_voxels": int(voxels["final_flyable_dem_only"].sum()),
        "final_nofly_dem_only_voxels": int(voxels["final_nofly_dem_only"].sum()),
        "plot_burn_cells_by_topo": bool(PLOT_BURN_CELLS_BY_TOPO),
        "plot_burn_cells_by_building": bool(PLOT_BURN_CELLS_BY_BUILDING),
        "plot_burn_cells_by_powerline": bool(PLOT_BURN_CELLS_BY_POWERLINE),
        "fig02_use_powerline_style_true_xy_for_topo_building": bool(FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING),
        "fig02_axis_limits_from_visible_true_xy_layers": bool(FIG02_AXIS_LIMITS_FROM_VISIBLE_TRUE_XY_LAYERS),
        "burn_cell_plot_draw_order": normalize_burn_cell_plot_draw_order(),
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
        f"Topo/DEM-burned voxels    : {int(voxels['burn_dem_terrain'].sum()):,}",
        f"Building-burned voxels     : {int(voxels.get('burn_building_volume', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Powerline pole-column voxels: {int(voxels.get('burn_powerline_pole_column', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Powerline line-air voxels  : {int(voxels.get('burn_powerline_line_air', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Obstacle-only voxels       : {int(voxels.get('burn_obstacle_volume', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Topo OR obstacle voxels    : {int(voxels.get('burn_topo_or_obstacle_volume', pd.Series(False, index=voxels.index)).sum()):,}",
        f"Building height input ref  : {BUILDING_HEIGHT_INPUT_REFERENCE} (base = DEM terrain MSL)",
        f"Powerline height ref       : {POWERLINE_HEIGHT_REFERENCE}; line mode={POWERLINE_LINE_BURN_VERTICAL_MODE}; pole mode={POWERLINE_POLE_BURN_VERTICAL_MODE}",
        f"Powerline line/pole buffer : {POWERLINE_HORIZONTAL_BUFFER_M:g} / {POWERLINE_POLE_HORIZONTAL_BUFFER_M:g} m",
        f"Powerline topo-aware base   : {POWERLINE_RESPECT_TOPO_BURN_TOP} (clearance={POWERLINE_TOPO_BURN_TOP_CLEARANCE_M:g} m)",
        f"Final flyable voxels       : {int(voxels['final_flyable_dem_only'].sum()):,}",
        f"Final no-fly voxels        : {int(voxels['final_nofly_dem_only'].sum()):,}",
        f"Plot topo-burn cells       : {PLOT_BURN_CELLS_BY_TOPO}",
        f"Plot building-burn cells   : {PLOT_BURN_CELLS_BY_BUILDING}",
        f"Plot powerline-burn cells  : {PLOT_BURN_CELLS_BY_POWERLINE}",
        f"Plot burn draw order       : {normalize_burn_cell_plot_draw_order()}",
        f"Fig02 true-XY unified path : {FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING} (topo/building use powerline-style burn-mask logic)",
        f"Fig02 visible-axis limits  : {FIG02_AXIS_LIMITS_FROM_VISIBLE_TRUE_XY_LAYERS}",
        f"Clip building plot volume  : {CLIP_3D_BUILDING_BURN_PLOT_TO_TRUE_VOLUME}",
        f"Clip topo plot volume      : {CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE}",
        f"Stack building on topo     : {PLOT_3D_BUILDING_STACK_ON_TOPO}",
        f"Fill building gap above topo: {PLOT_3D_BUILDING_FILL_GAP_ABOVE_TOPO}",
        f"Building stack base mode   : {PLOT_3D_BUILDING_STACK_BASE_MODE}",
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
        "  building_height_agl = source_height if BUILDING_HEIGHT_INPUT_REFERENCE='AGL'",
        "  building_height_agl = source_top_msl - terrain_msl if BUILDING_HEIGHT_INPUT_REFERENCE='MSL_TOP'",
        "  building_volume = building_footprint_area * building_height_agl",
        "  building_top_msl = terrain_msl + building_height_agl + building_buffer",
        "  burn_building_volume = XY_cell intersects footprint AND voxel_z intersects [terrain_msl, building_top_msl]",
        "  powerline line/strand: aerial band around terrain_msl + line_height_agl",
        "  powerline pole/tower : column from terrain_msl to terrain_msl + pole_height_agl",
        "  burn_powerline_system = burn_powerline_line_air OR burn_powerline_pole_column",
        "  burn_solid_obstacle_column = burn_building_volume OR burn_powerline_pole_column",
        "  burn_airborne_obstacle_band = burn_powerline_line_air",
        "  burn_obstacle_volume = burn_solid_obstacle_column OR burn_airborne_obstacle_band",
        "  burn_topo_or_obstacle_volume = burn_dem_terrain OR burn_obstacle_volume",
        "  burn_outside_polygon = inside_polygon != 1",
        "  final_nofly_dem_only = base_nofly OR burn_outside_polygon OR burn_dem_terrain OR burn_obstacle_volume",
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



def normalize_burn_cell_plot_draw_order() -> str:
    """Validate and normalize the requested burn-cell plotting order.

    Backward-compatible options without powerline still work; when powerline
    plotting is enabled and the order string does not mention powerline, it is
    drawn last by default.
    """
    order = str(BURN_CELL_PLOT_DRAW_ORDER).strip().lower()
    allowed = {
        "topo_then_building",
        "building_then_topo",
        "topo_then_building_then_powerline",
        "topo_then_powerline_then_building",
        "building_then_topo_then_powerline",
        "building_then_powerline_then_topo",
        "powerline_then_topo_then_building",
        "powerline_then_building_then_topo",
    }
    if order not in allowed:
        raise ValueError(
            "BURN_CELL_PLOT_DRAW_ORDER must be one of: "
            + ", ".join(sorted(allowed))
        )
    return order


def selected_burn_layer_names() -> list[str]:
    """Return enabled burn layers in the requested visual draw order."""
    order = normalize_burn_cell_plot_draw_order()
    if "powerline" in order:
        requested = order.split("_then_")
    else:
        requested = ["topo", "building"] if order == "topo_then_building" else ["building", "topo"]
        # Default for old two-layer settings: draw powerline/pole geofence on top.
        requested.append("powerline")

    enabled = []
    for name in requested:
        if name == "topo" and PLOT_BURN_CELLS_BY_TOPO:
            enabled.append(name)
        elif name == "building" and PLOT_BURN_CELLS_BY_BUILDING:
            enabled.append(name)
        elif name == "powerline" and PLOT_BURN_CELLS_BY_POWERLINE:
            enabled.append(name)
    return enabled


def add_burn_plot_display_columns(df: pd.DataFrame, keep_hard_nofly: bool = True) -> pd.DataFrame:
    """
    Add plotting-only burn-layer columns.

    The calculation/model columns are not changed. This helper only decides
    which burned cells are visible in QC figures and which layer wins where
    topo/building/powerline cells overlap in a 2D slice.

    display_code:
        0 = flyable / hidden
        1 = topo/DEM-burned cell
        2 = building-burned cell
        3 = powerline/pole geofence-burned cell
        4 = base/outside hard no-fly cell, used only when keep_hard_nofly=True
    """
    out = df.copy()

    topo_raw = out.get("burn_dem_terrain", False)
    building_raw = out.get("burn_building_volume", False)
    powerline_raw = out.get("burn_powerline_system", False)
    out["plot_burn_topo"] = pd.Series(topo_raw, index=out.index).astype(bool) & bool(PLOT_BURN_CELLS_BY_TOPO)
    out["plot_burn_building"] = pd.Series(building_raw, index=out.index).astype(bool) & bool(PLOT_BURN_CELLS_BY_BUILDING)
    out["plot_burn_powerline"] = pd.Series(powerline_raw, index=out.index).astype(bool) & bool(PLOT_BURN_CELLS_BY_POWERLINE)
    out["plot_burn_any"] = out["plot_burn_topo"] | out["plot_burn_building"] | out["plot_burn_powerline"]

    display_code = np.zeros(len(out), dtype=np.uint8)
    for layer in selected_burn_layer_names():
        if layer == "topo":
            display_code[out["plot_burn_topo"].to_numpy(dtype=bool)] = 1
        elif layer == "building":
            display_code[out["plot_burn_building"].to_numpy(dtype=bool)] = 2
        elif layer == "powerline":
            display_code[out["plot_burn_powerline"].to_numpy(dtype=bool)] = 3

    if keep_hard_nofly:
        hard_nofly = np.zeros(len(out), dtype=bool)
        if "base_nofly_input" in out.columns:
            hard_nofly |= out["base_nofly_input"].astype(bool).to_numpy()
        if "burn_outside_polygon" in out.columns:
            hard_nofly |= out["burn_outside_polygon"].astype(bool).to_numpy()
        display_code[hard_nofly] = 4

    out["burn_plot_display_code"] = display_code
    out["burn_display_class"] = np.select(
        [display_code == 1, display_code == 2, display_code == 3, display_code == 4],
        ["topo", "building", "powerline", "hard_nofly"],
        default="unburned",
    )
    return out


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

    # Plot display code:
    #   0 = flyable / hidden, 1 = topo/DEM burn, 2 = building burn,
    #   3 = OSM powerline/pole geofence, 4 = hard no-fly from base/outside polygon.
    cmap = ListedColormap([
        (1.0, 1.0, 1.0, 0.0),
        (*DEM_STATE_BURNED_GRAY_RGB, 0.92),
        (*DEM_STATE_BUILDING_YELLOW_RGB, 0.92),
        (*DEM_STATE_POWERLINE_PURPLE_RGB, 0.82),
        (0.0, 0.0, 0.0, 1.0),
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

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

        sub = add_burn_plot_display_columns(sub, keep_hard_nofly=True)
        piv = sub.pivot_table(
            index="y_from_sw_m",
            columns="x_from_sw_m",
            values="burn_plot_display_code",
            aggfunc="max",
        )
        piv = piv.sort_index(ascending=True)
        arr = piv.to_numpy(dtype=float)
        extent = [piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()]
        im = ax.imshow(arr, extent=extent, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("East from SW (m)")
        ax.set_ylabel("North from SW (m)")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[n:]:
        ax.axis("off")

    handles = []
    if PLOT_BURN_CELLS_BY_TOPO:
        handles.append(Patch(facecolor=(*DEM_STATE_BURNED_GRAY_RGB, 0.92), edgecolor="black", label="Topo/DEM-burned cells"))
    if PLOT_BURN_CELLS_BY_BUILDING:
        handles.append(Patch(facecolor=(*DEM_STATE_BUILDING_YELLOW_RGB, 0.92), edgecolor="black", label="Building-burned cells"))
    if PLOT_BURN_CELLS_BY_POWERLINE:
        handles.append(Patch(facecolor=(*DEM_STATE_POWERLINE_PURPLE_RGB, 0.82), edgecolor="black", label="Powerline/pole geofence cells"))
    handles.append(Patch(facecolor="black", edgecolor="black", label="Base/outside hard no-fly"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:n], shrink=0.72, pad=0.02, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(["hidden/flyable", "topo", "building", "powerline", "hard no-fly"])
        cbar.set_label("Burn-source display class")

    order_text = normalize_burn_cell_plot_draw_order().replace("_", " → ")
    fig.suptitle(
        "Topo/building/powerline burn-source Z-slice check "
        f"(topo={PLOT_BURN_CELLS_BY_TOPO}, building={PLOT_BURN_CELLS_BY_BUILDING}, "
        f"powerline={PLOT_BURN_CELLS_BY_POWERLINE}, order={order_text})",
        fontweight="bold",
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")

def draw_axis_triad_screen_inset(ax) -> None:
    """
    Draw a clean X/Y/Z orientation triad in screen coordinates.

    This is copied in spirit from the base voxel-box QC figure:
    it is visual-only and does not change data coordinates.
    """
    origin = (0.87, 0.82)
    x_tip = (0.95, 0.79)
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


def choose_dem_state_plot_strides(voxels_inside: pd.DataFrame) -> tuple[int, int, int]:
    """
    Choose coarse plotting strides for Figure 02.

    This follows the same idea as the base voxel-box script: the saved model
    remains full resolution, but the figure is aggregated to a readable number
    of translucent 3D blocks.

    For DEM-burn checking we prefer to keep z stride = 1 for as long as possible,
    so the vertical burn surface is not overly smeared.
    """
    nx = int(voxels_inside["ix"].nunique())
    ny = int(voxels_inside["iy"].nunique())
    nz = int(voxels_inside["iz"].nunique())

    sx = sy = 1
    sz = 1

    def displayed_count() -> int:
        return int(math.ceil(nx / sx) * math.ceil(ny / sy) * math.ceil(nz / sz))

    while displayed_count() > MAX_3D_DEM_VOXEL_CUBES_TO_RENDER:
        displayed_xy = np.array([nx / sx, ny / sy], dtype=float)
        if displayed_xy[0] >= displayed_xy[1]:
            sx += 1
        else:
            sy += 1

        # Only start thinning Z if XY thinning alone is not enough.
        if (sx > nx and sy > ny) and displayed_count() > MAX_3D_DEM_VOXEL_CUBES_TO_RENDER:
            sz += 1

        if sx > max(nx, 1) * 2 and sy > max(ny, 1) * 2:
            break

    return sx, sy, sz


def make_coarse_inside_voxels_for_dem_state_plot(voxels: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """
    Build the Figure-02 plotting model.

    Important display rules:
      - Outside-polygon cells are removed from this plot.
      - Non-burned inside-AOI cells are green.
      - DEM-burned cells are gray.
      - Building-burned cells are light yellow.
      - If both DEM and building burn a coarse block, building color wins for display.
      - Coarsening affects only the figure, never the saved model.
    """
    df = voxels.copy()

    if "inside_polygon" in df.columns:
        inside_mask = pd.to_numeric(df["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1
        df = df[inside_mask].copy()

    if df.empty:
        return df, (1, 1, 1)

    sx, sy, sz = choose_dem_state_plot_strides(df)

    df["gx"] = (pd.to_numeric(df["ix"], errors="coerce").astype(int) // sx).astype(int)
    df["gy"] = (pd.to_numeric(df["iy"], errors="coerce").astype(int) // sy).astype(int)
    df["gz"] = (pd.to_numeric(df["iz"], errors="coerce").astype(int) // sz).astype(int)

    if "burn_dem_terrain" not in df.columns:
        df["burn_dem_terrain"] = False
    if "burn_building_volume" not in df.columns:
        df["burn_building_volume"] = False
    if "burn_powerline_system" not in df.columns:
        df["burn_powerline_system"] = False
    if "building_base_msl_m" not in df.columns:
        df["building_base_msl_m"] = df["terrain_msl_m"]
    if "building_top_msl_m" not in df.columns:
        df["building_top_msl_m"] = df["terrain_msl_m"]
    if "powerline_height_agl_m" not in df.columns:
        df["powerline_height_agl_m"] = 0.0
    if "powerline_base_msl_m" not in df.columns:
        df["powerline_base_msl_m"] = df["terrain_msl_m"]
    if "powerline_top_msl_m" not in df.columns:
        df["powerline_top_msl_m"] = df["terrain_msl_m"]

    df["burn_dem_int"] = df["burn_dem_terrain"].astype(bool).astype(int)
    df["burn_building_int"] = df["burn_building_volume"].astype(bool).astype(int)
    df["burn_powerline_int"] = df["burn_powerline_system"].astype(bool).astype(int)

    coarse = (
        df.groupby(["gx", "gy", "gz"], as_index=False)
        .agg(
            x_from_sw_m=("x_from_sw_m", "mean"),
            y_from_sw_m=("y_from_sw_m", "mean"),
            z_center_msl_m=("z_center_msl_m", "mean"),
            z_bottom_msl_m=("z_bottom_msl_m", "min"),
            z_top_msl_m=("z_top_msl_m", "max"),
            ix=("ix", "mean"),
            iy=("iy", "mean"),
            iz=("iz", "mean"),
            terrain_msl_m=("terrain_msl_m", "mean"),
            building_base_msl_m=("building_base_msl_m", "min"),
            building_top_msl_m=("building_top_msl_m", "max"),
            powerline_height_agl_m=("powerline_height_agl_m", "max"),
            powerline_base_msl_m=("powerline_base_msl_m", "min"),
            powerline_top_msl_m=("powerline_top_msl_m", "max"),
            dem_burn_fraction=("burn_dem_int", "mean"),
            building_burn_fraction=("burn_building_int", "mean"),
            powerline_burn_fraction=("burn_powerline_int", "mean"),
            voxel_count=("burn_dem_int", "size"),
        )
    )

    if DEM_STATE_COARSE_BURN_RULE.lower() == "majority":
        coarse["burn_dem_terrain"] = coarse["dem_burn_fraction"] >= 0.5
        coarse["burn_building_volume"] = coarse["building_burn_fraction"] >= 0.5
        coarse["burn_powerline_system"] = coarse["powerline_burn_fraction"] >= 0.5
    else:
        coarse["burn_dem_terrain"] = coarse["dem_burn_fraction"] > 0.0
        coarse["burn_building_volume"] = coarse["building_burn_fraction"] > 0.0
        coarse["burn_powerline_system"] = coarse["powerline_burn_fraction"] > 0.0

    coarse["burn_any"] = coarse["burn_dem_terrain"] | coarse["burn_building_volume"] | coarse["burn_powerline_system"]
    coarse["flyable_after_dem_burn"] = (~coarse["burn_any"]).astype(int)

    # Plotting-only layer selection. Unlike the saved model, the figure can
    # independently hide/show topo-burn and building-burn cells and can draw
    # the selected layers in either visual order.
    coarse = add_burn_plot_display_columns(coarse, keep_hard_nofly=False)

    n_green = int((~coarse["plot_burn_any"]).sum())
    n_gray = int(coarse["plot_burn_topo"].sum())
    n_yellow = int(coarse["plot_burn_building"].sum())
    n_purple = int(coarse["plot_burn_powerline"].sum())

    raw_building = int(df["burn_building_int"].sum())
    raw_powerline = int(df["burn_powerline_int"].sum())
    raw_dem = int(df["burn_dem_int"].sum())
    print(
        f"[INFO] Figure 02 coarse blocks: {len(coarse):,}; "
        f"stride={sx} x {sy} x {sz}; "
        f"green={n_green:,}; gray_topo={n_gray:,}; yellow_building={n_yellow:,}; purple_powerline={n_purple:,}; "
        f"raw_dem_voxels={raw_dem:,}; raw_building_voxels={raw_building:,}; raw_powerline_voxels={raw_powerline:,}; "
        f"plot_topo={PLOT_BURN_CELLS_BY_TOPO}; plot_building={PLOT_BURN_CELLS_BY_BUILDING}; plot_powerline={PLOT_BURN_CELLS_BY_POWERLINE}; "
        f"draw_order={normalize_burn_cell_plot_draw_order()}; outside-polygon cells hidden."
    )
    return coarse, (sx, sy, sz)


def voxel_block_faces_from_df(
    df: pd.DataFrame,
    dx: float,
    dy: float,
    dz: float,
    color_mode: str,
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float, float, float]]]:
    """
    Convert voxel/prism center points to 3D faces.

    By default this draws full voxel cubes from z_center_msl_m and dz.
    If z_plot_bottom_msl_m and z_plot_top_msl_m are present, it draws a
    clipped prism instead. This is used only for Figure 02 to prevent the
    building display from extending below the DEM/topography surface.
    """
    if df is None or df.empty:
        return [], []

    hx, hy = dx / 2.0, dy / 2.0
    face_ids = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []

    x = pd.to_numeric(df["x_from_sw_m"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y_from_sw_m"], errors="coerce").to_numpy(dtype=float)

    if {"z_plot_bottom_msl_m", "z_plot_top_msl_m"}.issubset(df.columns):
        z0 = pd.to_numeric(df["z_plot_bottom_msl_m"], errors="coerce").to_numpy(dtype=float)
        z1 = pd.to_numeric(df["z_plot_top_msl_m"], errors="coerce").to_numpy(dtype=float)
    else:
        zc = pd.to_numeric(df["z_center_msl_m"], errors="coerce").to_numpy(dtype=float)
        z0 = zc - dz / 2.0
        z1 = zc + dz / 2.0

    classes = df["burn_display_class"].astype(str).to_numpy() if "burn_display_class" in df.columns else np.full(len(df), "unburned", dtype=object)

    for xi, yi, zlo, zhi, cls in zip(x, y, z0, z1, classes):
        if not (np.isfinite(xi) and np.isfinite(yi) and np.isfinite(zlo) and np.isfinite(zhi)):
            continue
        if zhi <= zlo:
            continue

        if color_mode == "dem_state":
            if cls == "building":
                rgba = (*DEM_STATE_BUILDING_YELLOW_RGB, DEM_STATE_PLOT_BUILDING_YELLOW_ALPHA)
            elif cls == "powerline":
                rgba = (*DEM_STATE_POWERLINE_PURPLE_RGB, DEM_STATE_PLOT_POWERLINE_PURPLE_ALPHA)
            elif cls in {"dem", "topo"}:
                rgba = (*DEM_STATE_BURNED_GRAY_RGB, DEM_STATE_PLOT_BURNED_GRAY_ALPHA)
            else:
                rgba = (*DEM_STATE_UNBURNED_GREEN_RGB, DEM_STATE_PLOT_GREEN_ALPHA)
        else:
            rgba = (0.45, 0.45, 0.45, 0.45)

        vertices = np.array([
            [xi - hx, yi - hy, zlo], [xi + hx, yi - hy, zlo], [xi + hx, yi + hy, zlo], [xi - hx, yi + hy, zlo],
            [xi - hx, yi - hy, zhi], [xi + hx, yi - hy, zhi], [xi + hx, yi + hy, zhi], [xi - hx, yi + hy, zhi],
        ], dtype=float)
        for ids in face_ids:
            faces.append([(float(vertices[i,0]), float(vertices[i,1]), float(vertices[i,2])) for i in ids])
            facecolors.append(rgba)

    return faces, facecolors



def stack_building_plot_voxels_on_topography(
    yellow: pd.DataFrame,
    gray: pd.DataFrame,
    plot_dz: float,
) -> pd.DataFrame:
    """
    Plot-only correction for Figure 02.

    User-requested display logic:
      1. Topo/DEM burned cells are drawn first.
      2. Building cells are drawn directly above the topo voxel column.
      3. If there is a vertical gap between the topo voxel stack and the raw
         building-burned cells, that gap is filled as building class.

    This function therefore does NOT trust the raw MSL position of the first
    building-burned voxel for the combined QC plot. Instead, it rebuilds a
    contiguous building display stack per coarse XY column.

    This is plotting-only. The saved voxel model still uses the physical MSL
    tests:
        building_base_msl = terrain_msl
        building_top_msl  = terrain_msl + building_height_agl
    """
    if yellow is None or yellow.empty:
        return yellow

    y = yellow.copy()

    # Decide where the building-class display should start.
    # topo_voxel_top = the top of the displayed topo-burned voxel column.
    # terrain_surface = exact terrain_msl_m, useful for clipped-volume plots.
    base_mode = str(PLOT_3D_BUILDING_STACK_BASE_MODE).strip().lower()
    if base_mode not in {"topo_voxel_top", "terrain_surface"}:
        raise ValueError('PLOT_3D_BUILDING_STACK_BASE_MODE must be "topo_voxel_top" or "terrain_surface"')

    topo_base_map = None
    if gray is not None and not gray.empty:
        if base_mode == "topo_voxel_top":
            # Use the true top of the topo voxel cells, not the clipped DEM
            # surface. This matches the requested voxel-class display: cells
            # below topo are gray topo cells; anything above that and below the
            # building stack becomes yellow building class.
            top_col = "z_top_msl_m"
        else:
            top_col = "z_plot_top_msl_m" if "z_plot_top_msl_m" in gray.columns else "terrain_msl_m"
        topo_base_map = (
            gray.groupby(["gx", "gy"])[top_col]
            .max()
            .rename("_stack_base_msl_m")
        )

    y = y.sort_values(["gx", "gy", "z_center_msl_m", "gz"]).copy()
    if topo_base_map is not None:
        y = y.merge(topo_base_map, on=["gx", "gy"], how="left")
    else:
        y["_stack_base_msl_m"] = np.nan

    # Fallback when topo display is hidden/missing in this column.
    fallback_base = pd.to_numeric(
        y.get("building_base_msl_m", y.get("terrain_msl_m", 0.0)),
        errors="coerce",
    )
    fallback_base = fallback_base.fillna(pd.to_numeric(y.get("terrain_msl_m", 0.0), errors="coerce"))
    y["_stack_base_msl_m"] = pd.to_numeric(y["_stack_base_msl_m"], errors="coerce").fillna(fallback_base)

    # If the raw building cells start above the topo stack, the rank-based stack
    # below fills that gap as building class. Keep the same number of building
    # layers as the selected raw building-burned cells in each coarse column.
    y["_building_stack_rank"] = y.groupby(["gx", "gy"]).cumcount().astype(float)
    y["z_plot_bottom_msl_m"] = y["_stack_base_msl_m"] + y["_building_stack_rank"] * float(plot_dz)
    y["z_plot_top_msl_m"] = y["z_plot_bottom_msl_m"] + float(plot_dz)
    y["z_center_msl_m"] = 0.5 * (y["z_plot_bottom_msl_m"] + y["z_plot_top_msl_m"])

    # Optional extra guard: make the column continuous up to at least the raw
    # building top. If coarsening removed intermediate building rows, synthesize
    # missing yellow rows so there is no white gap above topo.
    if PLOT_3D_BUILDING_FILL_GAP_ABOVE_TOPO:
        filled_rows = []
        for (gx, gy), col in y.groupby(["gx", "gy"], sort=False):
            col = col.sort_values("z_plot_bottom_msl_m").copy()
            base = float(pd.to_numeric(col["z_plot_bottom_msl_m"], errors="coerce").min())
            # Use the greater of the raw burned-cell top and the physical building top.
            raw_top_candidates = []
            for c in ["z_top_msl_m", "building_top_msl_m", "z_plot_top_msl_m"]:
                if c in col.columns:
                    raw_top_candidates.append(pd.to_numeric(col[c], errors="coerce").max())
            raw_top = float(np.nanmax(raw_top_candidates)) if raw_top_candidates else float(col["z_plot_top_msl_m"].max())
            current_top = float(pd.to_numeric(col["z_plot_top_msl_m"], errors="coerce").max())
            target_top = max(current_top, raw_top)
            n_layers = max(1, int(math.ceil((target_top - base) / float(plot_dz))))

            template = col.iloc[0].copy()
            for k in range(n_layers):
                row = template.copy()
                z0 = base + k * float(plot_dz)
                z1 = z0 + float(plot_dz)
                row["z_plot_bottom_msl_m"] = z0
                row["z_plot_top_msl_m"] = z1
                row["z_center_msl_m"] = 0.5 * (z0 + z1)
                row["z_bottom_msl_m"] = z0
                row["z_top_msl_m"] = z1
                row["gz"] = int(k)
                row["burn_display_class"] = "building"
                row["plot_burn_building"] = True
                row["plot_burn_topo"] = False
                row["plot_burn_any"] = True
                filled_rows.append(row)
        if filled_rows:
            y = pd.DataFrame(filled_rows).reset_index(drop=True)

    y = y.drop(
        columns=["_stack_base_msl_m", "_building_stack_rank"],
        errors="ignore",
    )
    return y



def make_true_xy_layer_plot_voxels_like_powerline_qc(
    voxels: pd.DataFrame,
    *,
    burn_col: str,
    display_class: str,
    base_col: str | None = None,
    top_col: str | None = None,
    height_col: str | None = None,
    max_voxels: int | None = None,
    clip_to_interval: bool = True,
    extra_keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a Figure-02 visible layer using the same logic as true-XY powerline.

    This is the unified plotting path for topo, building, and powerline-like
    voxel layers:
      1. start from the final voxel table;
      2. select rows where the requested burn mask is True;
      3. keep true voxel-grid coordinates x_from_sw_m/y_from_sw_m;
      4. optionally clip z_plot_bottom/top to a physical MSL interval;
      5. draw true 50 x 50 x dz cells.

    It does not use coarse blocks and it does not rebuild cells from xy_gdf.
    That is why it matches the powerline QC figure behavior.
    """
    if voxels is None or voxels.empty or burn_col not in voxels.columns:
        return pd.DataFrame()

    vv = voxels[voxels[burn_col].astype(bool)].copy()
    if "inside_polygon" in vv.columns:
        vv = vv[pd.to_numeric(vv["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if vv.empty:
        return pd.DataFrame()

    keep = [
        "xy_id", "ix", "iy", "iz", "x_from_sw_m", "y_from_sw_m",
        "z_center_msl_m", "z_bottom_msl_m", "z_top_msl_m", "terrain_msl_m",
        burn_col,
    ]
    for c in [base_col, top_col, height_col]:
        if c:
            keep.append(c)
    if extra_keep_cols:
        keep.extend(extra_keep_cols)
    keep = [c for c in dict.fromkeys(keep) if c in vv.columns]
    vv = vv[keep].copy()

    if clip_to_interval and base_col and top_col and base_col in vv.columns and top_col in vv.columns:
        vv["z_plot_bottom_msl_m"] = np.maximum(
            pd.to_numeric(vv["z_bottom_msl_m"], errors="coerce"),
            pd.to_numeric(vv[base_col], errors="coerce"),
        )
        vv["z_plot_top_msl_m"] = np.minimum(
            pd.to_numeric(vv["z_top_msl_m"], errors="coerce"),
            pd.to_numeric(vv[top_col], errors="coerce"),
        )
        vv = vv[vv["z_plot_top_msl_m"] > vv["z_plot_bottom_msl_m"]].copy()
    elif display_class == "topo" and CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE and "terrain_msl_m" in vv.columns:
        vv["z_plot_bottom_msl_m"] = pd.to_numeric(vv["z_bottom_msl_m"], errors="coerce")
        vv["z_plot_top_msl_m"] = np.minimum(
            pd.to_numeric(vv["z_top_msl_m"], errors="coerce"),
            pd.to_numeric(vv["terrain_msl_m"], errors="coerce"),
        )
        vv = vv[vv["z_plot_top_msl_m"] > vv["z_plot_bottom_msl_m"]].copy()

    vv = vv[
        np.isfinite(pd.to_numeric(vv["x_from_sw_m"], errors="coerce"))
        & np.isfinite(pd.to_numeric(vv["y_from_sw_m"], errors="coerce"))
    ].copy()
    if vv.empty:
        return pd.DataFrame()

    if max_voxels is not None and len(vv) > int(max_voxels):
        print(
            f"[WARN] Figure 02 TRUE-XY {display_class} cells = {len(vv):,}; "
            f"downsampling to {int(max_voxels):,} for plotting only."
        )
        vv = vv.sample(int(max_voxels), random_state=RANDOM_SEED).copy()

    vv["gx"] = pd.to_numeric(vv["ix"], errors="coerce").astype(int)
    vv["gy"] = pd.to_numeric(vv["iy"], errors="coerce").astype(int)
    vv["gz"] = pd.to_numeric(vv["iz"], errors="coerce").astype(int)
    vv["burn_display_class"] = str(display_class)
    vv["plot_burn_topo"] = display_class == "topo"
    vv["plot_burn_building"] = display_class == "building"
    vv["plot_burn_powerline"] = display_class == "powerline"
    vv["plot_burn_any"] = True
    vv["_use_true_xy_cell_size"] = True
    vv["_fig02_true_xy_builder"] = "powerline_style_burn_mask"

    msg = f"[CHECK] Figure 02 TRUE-XY {display_class} display voxels: {len(vv):,} using powerline-style burn-mask logic"
    if height_col and height_col in vv.columns:
        h = pd.to_numeric(vv[height_col], errors="coerce")
        if h.notna().any():
            msg += f", height_AGL min/p50/max={np.nanmin(h):.2f}/{np.nanpercentile(h, 50):.2f}/{np.nanmax(h):.2f} m"
    print(msg)
    return vv


def make_true_xy_topo_plot_voxels_for_figure02(
    voxels: pd.DataFrame,
    dz: float,
) -> pd.DataFrame:
    """Build Figure-02 topo cells from the same true-XY burn-mask logic as powerline."""
    if not bool(PLOT_BURN_CELLS_BY_TOPO):
        return pd.DataFrame()
    if bool(FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING):
        return make_true_xy_layer_plot_voxels_like_powerline_qc(
            voxels,
            burn_col="burn_dem_terrain",
            display_class="topo",
            base_col=None,
            top_col=None,
            height_col=None,
            max_voxels=int(FIG02_MAX_TRUE_XY_TOPO_VOXELS_TO_RENDER),
            clip_to_interval=False,
        )

    if "burn_dem_terrain" not in voxels.columns:
        return pd.DataFrame()

    vv = voxels[voxels["burn_dem_terrain"].astype(bool)].copy()
    if "inside_polygon" in vv.columns:
        vv = vv[pd.to_numeric(vv["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if vv.empty:
        return pd.DataFrame()

    keep = [
        "xy_id", "ix", "iy", "iz", "x_from_sw_m", "y_from_sw_m",
        "z_center_msl_m", "z_bottom_msl_m", "z_top_msl_m", "terrain_msl_m",
    ]
    keep = [c for c in keep if c in vv.columns]
    vv = vv[keep].copy()

    if CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE:
        vv["z_plot_bottom_msl_m"] = pd.to_numeric(vv["z_bottom_msl_m"], errors="coerce")
        vv["z_plot_top_msl_m"] = np.minimum(
            pd.to_numeric(vv["z_top_msl_m"], errors="coerce"),
            pd.to_numeric(vv["terrain_msl_m"], errors="coerce"),
        )
        vv = vv[vv["z_plot_top_msl_m"] > vv["z_plot_bottom_msl_m"]].copy()

    if len(vv) > int(FIG02_MAX_TRUE_XY_TOPO_VOXELS_TO_RENDER):
        print(
            f"[WARN] TRUE-XY topo cells for Figure 02 = {len(vv):,}, "
            f"downsampling to {FIG02_MAX_TRUE_XY_TOPO_VOXELS_TO_RENDER:,} for plotting only."
        )
        vv = vv.sample(int(FIG02_MAX_TRUE_XY_TOPO_VOXELS_TO_RENDER), random_state=RANDOM_SEED).copy()

    vv["gx"] = pd.to_numeric(vv["ix"], errors="coerce").astype(int)
    vv["gy"] = pd.to_numeric(vv["iy"], errors="coerce").astype(int)
    vv["gz"] = pd.to_numeric(vv["iz"], errors="coerce").astype(int)
    vv["burn_display_class"] = "topo"
    vv["plot_burn_topo"] = True
    vv["plot_burn_building"] = False
    vv["plot_burn_powerline"] = False
    vv["plot_burn_any"] = True
    vv["_use_true_xy_cell_size"] = True

    print(f"[CHECK] Figure 02 TRUE-XY topo display voxels: {len(vv):,}")
    return vv


def make_true_xy_unburned_plot_voxels_for_figure02(
    voxels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Figure-02 unburned cells from TRUE XY voxel columns.

    This is important when PLOT_UNBURNED_3D_DEM_CELLS=True.  The old plot used
    coarse blocks for green cells while topo/building could use true 50 m cells,
    so the switch looked wrong even when the burn data were correct.

    "Unburned" here means unburned by the currently selected visible burn
    sources. Therefore, when topo is hidden, topo-burned cells are not counted
    as selected burned cells in this green layer.
    """
    if not bool(PLOT_UNBURNED_3D_DEM_CELLS):
        return pd.DataFrame()
    if voxels is None or voxels.empty:
        return pd.DataFrame()

    vv = voxels.copy()
    if "inside_polygon" in vv.columns:
        vv = vv[pd.to_numeric(vv["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if vv.empty:
        return pd.DataFrame()

    # Apply the same selected-layer visibility rules used by the coarse plot.
    vv = add_burn_plot_display_columns(vv, keep_hard_nofly=False)
    vv = vv[~vv["plot_burn_any"].astype(bool)].copy()
    if vv.empty:
        return pd.DataFrame()

    keep = [
        "xy_id", "ix", "iy", "iz", "x_from_sw_m", "y_from_sw_m",
        "z_center_msl_m", "z_bottom_msl_m", "z_top_msl_m", "terrain_msl_m",
        "burn_display_class", "plot_burn_topo", "plot_burn_building", "plot_burn_powerline", "plot_burn_any",
    ]
    keep = [c for c in keep if c in vv.columns]
    vv = vv[keep].copy()

    if len(vv) > int(FIG02_MAX_TRUE_XY_UNBURNED_VOXELS_TO_RENDER):
        print(
            f"[WARN] TRUE-XY unburned cells for Figure 02 = {len(vv):,}, "
            f"downsampling to {FIG02_MAX_TRUE_XY_UNBURNED_VOXELS_TO_RENDER:,} for plotting only."
        )
        vv = vv.sample(int(FIG02_MAX_TRUE_XY_UNBURNED_VOXELS_TO_RENDER), random_state=RANDOM_SEED).copy()

    vv["gx"] = pd.to_numeric(vv["ix"], errors="coerce").astype(int)
    vv["gy"] = pd.to_numeric(vv["iy"], errors="coerce").astype(int)
    vv["gz"] = pd.to_numeric(vv["iz"], errors="coerce").astype(int)
    vv["burn_display_class"] = "unburned"
    vv["plot_burn_topo"] = False
    vv["plot_burn_building"] = False
    vv["plot_burn_powerline"] = False
    vv["plot_burn_any"] = False
    vv["_use_true_xy_cell_size"] = True

    print(f"[CHECK] Figure 02 TRUE-XY unburned display voxels: {len(vv):,}")
    return vv

def make_true_xy_building_plot_voxels_for_figure02(
    paths: Paths,
    voxels: pd.DataFrame,
    xy_gdf: gpd.GeoDataFrame,
    dz: float,
    combined_visible_view: bool,
) -> pd.DataFrame:
    """Build Figure-02 building cells from TRUE XY burn-mask rows.

    This now intentionally follows the same logic as the correct powerline QC
    layer.  It does not rebuild building display cells from xy_gdf and it does
    not stack/re-rank the building column for plotting.  The visible yellow
    voxels are exactly those that collide with the physical building MSL volume:

        building_base_msl = terrain_msl
        building_top_msl  = terrain_msl + building_height_agl

    Each displayed cube is clipped to [building_base_msl, building_top_msl],
    using the original 50 x 50 x dz voxel rows and local SW coordinates.
    """
    if not bool(PLOT_3D_BUILDING_FROM_TRUE_XY_CELLS):
        return pd.DataFrame()
    if not bool(PLOT_BURN_CELLS_BY_BUILDING):
        return pd.DataFrame()
    if voxels is None or voxels.empty or "burn_building_volume" not in voxels.columns:
        return pd.DataFrame()

    if bool(FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING):
        extra_keep = [
            "building_height_input_m", "building_height_reference_used",
            "building_height_agl_m", "building_base_msl_m", "building_top_msl_m",
            "building_count_in_xy", "building_volume_m3", "lon", "lat",
        ]
        vv = make_true_xy_layer_plot_voxels_like_powerline_qc(
            voxels,
            burn_col="burn_building_volume",
            display_class="building",
            base_col="building_base_msl_m",
            top_col="building_top_msl_m",
            height_col="building_height_agl_m",
            max_voxels=int(FIG02_MAX_TRUE_XY_BUILDING_VOXELS_TO_RENDER),
            clip_to_interval=bool(CLIP_3D_BUILDING_BURN_PLOT_TO_TRUE_VOLUME),
            extra_keep_cols=extra_keep,
        )
        if vv.empty:
            return vv

        # Save a small QC table similar to the old function, but directly from
        # the actual plotted voxel rows. This makes height/projection debugging
        # much more honest: no xy_gdf reconstruction and no stacked display base.
        try:
            qc_cols = [
                "xy_id", "x_from_sw_m", "y_from_sw_m", "lon", "lat",
                "terrain_msl_m", "building_height_input_m", "building_height_reference_used",
                "building_height_agl_m", "building_base_msl_m", "building_top_msl_m",
                "z_bottom_msl_m", "z_top_msl_m", "z_plot_bottom_msl_m", "z_plot_top_msl_m",
                "building_count_in_xy", "building_volume_m3", "_fig02_true_xy_builder",
            ]
            qc_cols = [c for c in qc_cols if c in vv.columns]
            qc = vv[qc_cols].copy()
            if "building_height_agl_m" in qc.columns:
                qc = qc.sort_values("building_height_agl_m", ascending=False)
            qc.to_csv(paths.data_dir / "fig02_true_xy_building_burnmask_qc.csv", index=False)
            qc.head(int(FIG02_BUILDING_QC_SAVE_TOP_N)).to_csv(
                paths.data_dir / "fig02_true_xy_building_burnmask_qc_top.csv",
                index=False,
            )
        except Exception as exc:
            print(f"[WARN] Could not save Figure 02 building burn-mask QC CSV: {exc}")

        h = pd.to_numeric(vv.get("building_height_agl_m", pd.Series([], dtype=float)), errors="coerce")
        top = pd.to_numeric(vv.get("building_top_msl_m", pd.Series([], dtype=float)), errors="coerce")
        if h.notna().any():
            print(
                "[CHECK] Figure 02 TRUE-XY building burn-mask QC: "
                f"voxels={len(vv):,}, xy_cells={vv['xy_id'].nunique():,}, "
                f"height_AGL min/p50/p95/max={np.nanmin(h):.2f}/{np.nanpercentile(h, 50):.2f}/"
                f"{np.nanpercentile(h, 95):.2f}/{np.nanmax(h):.2f} m, "
                f"top_MSL max={np.nanmax(top):.2f} m"
            )
            if np.nanmax(h) > float(FIG02_BUILDING_QC_WARN_HEIGHT_ABOVE_M):
                print(
                    f"[WARN] Figure 02 has building_height_agl_m above {FIG02_BUILDING_QC_WARN_HEIGHT_ABOVE_M:g} m. "
                    f"Check: {paths.data_dir / 'fig02_true_xy_building_burnmask_qc_top.csv'}"
                )
        return vv

    # Legacy branch retained for comparison if the unified true-XY builder is disabled.
    return pd.DataFrame()


def make_true_xy_powerline_plot_voxels_for_figure02(
    voxels: pd.DataFrame,
    xy_gdf: gpd.GeoDataFrame,
    dz: float,
) -> pd.DataFrame:
    """Build Figure-02 OSM powerline-system cells from TRUE XY voxel columns.

    Separate plotting intervals are used for the two burn classes:
      - line/strand/cable: purple airborne cable band only, clipped above the local topo-burn top.
      - pole/tower/pylon: purple vertical column from the local topo-burn top to pole top.
    """
    if not bool(PLOT_BURN_CELLS_BY_POWERLINE):
        return pd.DataFrame()
    if voxels is None or voxels.empty or "burn_powerline_system" not in voxels.columns:
        return pd.DataFrame()

    parts = []
    configs = [
        (
            "line",
            "burn_powerline_line_air",
            "powerline_line_effective_base_msl_m",
            "powerline_line_top_msl_m",
            "powerline_line_height_agl_m",
        ),
        (
            "pole",
            "burn_powerline_pole_column",
            "powerline_pole_effective_base_msl_m",
            "powerline_pole_top_msl_m",
            "powerline_pole_height_agl_m",
        ),
    ]

    for kind, burn_col, base_col, top_col, h_col in configs:
        if burn_col not in voxels.columns:
            continue
        vv = voxels[voxels[burn_col].astype(bool)].copy()
        if "inside_polygon" in vv.columns:
            vv = vv[pd.to_numeric(vv["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if vv.empty:
            continue

        keep = [
            "xy_id", "ix", "iy", "iz", "x_from_sw_m", "y_from_sw_m",
            "z_center_msl_m", "z_bottom_msl_m", "z_top_msl_m", "terrain_msl_m",
            "powerline_height_input_m", "powerline_height_reference_used", "powerline_kind_in_xy",
            "powerline_height_agl_m", "powerline_base_msl_m", "powerline_top_msl_m",
            "topo_burn_top_msl_m",
            "powerline_line_height_agl_m", "powerline_line_base_msl_m", "powerline_line_effective_base_msl_m", "powerline_line_top_msl_m",
            "powerline_pole_height_agl_m", "powerline_pole_base_msl_m", "powerline_pole_effective_base_msl_m", "powerline_pole_top_msl_m",
            "powerline_count_in_xy", "powerline_line_count_in_xy", "powerline_pole_count_in_xy",
        ]
        keep = [c for c in keep if c in vv.columns]
        vv = vv[keep].copy()
        if base_col not in vv.columns or top_col not in vv.columns:
            continue

        vv["z_plot_bottom_msl_m"] = np.maximum(
            pd.to_numeric(vv["z_bottom_msl_m"], errors="coerce"),
            pd.to_numeric(vv[base_col], errors="coerce"),
        )
        vv["z_plot_top_msl_m"] = np.minimum(
            pd.to_numeric(vv["z_top_msl_m"], errors="coerce"),
            pd.to_numeric(vv[top_col], errors="coerce"),
        )
        vv = vv[vv["z_plot_top_msl_m"] > vv["z_plot_bottom_msl_m"]].copy()
        if vv.empty:
            continue

        vv["powerline_plot_kind"] = kind
        vv["powerline_plot_height_agl_m"] = pd.to_numeric(vv.get(h_col, 0.0), errors="coerce").fillna(0.0)
        parts.append(vv)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)

    # Figure-02 coordinate fix:
    # Powerline source data may start in lon/lat degrees, but burn/collision is
    # performed on UTM XY cells.  Therefore the Figure-02 powerline layer must
    # plot using the voxel XY-grid local coordinates (distance from SW reference),
    # not any source-vector coordinates that may remain in intermediate tables.
    # This makes Figure 02 use the same coordinate basis as the dedicated
    # topography+powerline mesh QC figure.
    if bool(FIG02_POWERLINE_USE_XY_GRID_LOCAL_COORDS) and xy_gdf is not None and not xy_gdf.empty:
        try:
            xy_lookup = xy_gdf.drop_duplicates("xy_id").set_index("xy_id")[["x_from_sw_m", "y_from_sw_m"]]
            out["x_from_sw_m"] = out["xy_id"].map(xy_lookup["x_from_sw_m"]).astype(float)
            out["y_from_sw_m"] = out["xy_id"].map(xy_lookup["y_from_sw_m"]).astype(float)
        except Exception as exc:
            print(f"[WARN] Figure 02 powerline coordinate remap from XY grid failed; using voxel coordinates. Reason: {exc}")

    # Drop any rows that still do not have valid local meter coordinates.
    out = out[np.isfinite(pd.to_numeric(out["x_from_sw_m"], errors="coerce")) & np.isfinite(pd.to_numeric(out["y_from_sw_m"], errors="coerce"))].copy()
    if out.empty:
        return pd.DataFrame()

    if len(out) > int(FIG02_MAX_TRUE_XY_POWERLINE_VOXELS_TO_RENDER):
        print(
            f"[WARN] TRUE-XY powerline cells for Figure 02 = {len(out):,}, "
            f"downsampling to {FIG02_MAX_TRUE_XY_POWERLINE_VOXELS_TO_RENDER:,} for plotting only."
        )
        out = out.sample(int(FIG02_MAX_TRUE_XY_POWERLINE_VOXELS_TO_RENDER), random_state=RANDOM_SEED).copy()

    out["gx"] = pd.to_numeric(out["ix"], errors="coerce").astype(int)
    out["gy"] = pd.to_numeric(out["iy"], errors="coerce").astype(int)
    out["gz"] = pd.to_numeric(out["iz"], errors="coerce").astype(int)
    out["burn_display_class"] = "powerline"
    out["plot_burn_powerline"] = True
    out["plot_burn_building"] = False
    out["plot_burn_topo"] = False
    out["plot_burn_any"] = True
    out["_use_true_xy_cell_size"] = True

    try:
        n_line = int((out["powerline_plot_kind"] == "line").sum())
        n_pole = int((out["powerline_plot_kind"] == "pole").sum())
        h = pd.to_numeric(out.get("powerline_plot_height_agl_m", pd.Series([], dtype=float)), errors="coerce")
        print(
            "[CHECK] Figure 02 TRUE-XY powerline display voxels: "
            f"{len(out):,} (line_air={n_line:,}, pole_column={n_pole:,}), "
            f"height_AGL min/p50/max={np.nanmin(h):.2f}/{np.nanpercentile(h, 50):.2f}/{np.nanmax(h):.2f} m"
        )
    except Exception:
        print(f"[CHECK] Figure 02 TRUE-XY powerline display voxels: {len(out):,}")
    return out



def _fig02_z_bottom_top_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the plotted bottom/top MSL columns for a visible Figure-02 layer."""
    if df is None or df.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    if {"z_plot_bottom_msl_m", "z_plot_top_msl_m"}.issubset(df.columns):
        z0 = pd.to_numeric(df["z_plot_bottom_msl_m"], errors="coerce")
        z1 = pd.to_numeric(df["z_plot_top_msl_m"], errors="coerce")
    else:
        z0 = pd.to_numeric(df.get("z_bottom_msl_m", pd.Series(np.nan, index=df.index)), errors="coerce")
        z1 = pd.to_numeric(df.get("z_top_msl_m", pd.Series(np.nan, index=df.index)), errors="coerce")
    return z0, z1


def _fig02_layer_extent_row(layer_name: str, df: pd.DataFrame) -> dict:
    """Summarize one Figure-02 layer in both MSL and AGL coordinates."""
    row = {
        "layer": layer_name,
        "n_voxels_displayed": 0,
        "n_xy_cells_displayed": 0,
        "x_from_sw_min_m": np.nan,
        "x_from_sw_max_m": np.nan,
        "y_from_sw_min_m": np.nan,
        "y_from_sw_max_m": np.nan,
        "terrain_min_msl_m": np.nan,
        "terrain_p50_msl_m": np.nan,
        "terrain_max_msl_m": np.nan,
        "z_min_msl_m": np.nan,
        "z_max_msl_m": np.nan,
        "z_min_agl_m": np.nan,
        "z_max_agl_m": np.nan,
        "height_agl_min_m": np.nan,
        "height_agl_p50_m": np.nan,
        "height_agl_max_m": np.nan,
    }
    if df is None or df.empty:
        return row

    z0, z1 = _fig02_z_bottom_top_columns(df)
    terrain = pd.to_numeric(df.get("terrain_msl_m", pd.Series(np.nan, index=df.index)), errors="coerce")
    row.update({
        "n_voxels_displayed": int(len(df)),
        "n_xy_cells_displayed": int(df["xy_id"].nunique()) if "xy_id" in df.columns else 0,
        "x_from_sw_min_m": float(pd.to_numeric(df["x_from_sw_m"], errors="coerce").min()) if "x_from_sw_m" in df.columns else np.nan,
        "x_from_sw_max_m": float(pd.to_numeric(df["x_from_sw_m"], errors="coerce").max()) if "x_from_sw_m" in df.columns else np.nan,
        "y_from_sw_min_m": float(pd.to_numeric(df["y_from_sw_m"], errors="coerce").min()) if "y_from_sw_m" in df.columns else np.nan,
        "y_from_sw_max_m": float(pd.to_numeric(df["y_from_sw_m"], errors="coerce").max()) if "y_from_sw_m" in df.columns else np.nan,
        "terrain_min_msl_m": float(terrain.min()) if terrain.notna().any() else np.nan,
        "terrain_p50_msl_m": float(terrain.median()) if terrain.notna().any() else np.nan,
        "terrain_max_msl_m": float(terrain.max()) if terrain.notna().any() else np.nan,
        "z_min_msl_m": float(z0.min()) if z0.notna().any() else np.nan,
        "z_max_msl_m": float(z1.max()) if z1.notna().any() else np.nan,
        "z_min_agl_m": float((z0 - terrain).min()) if z0.notna().any() and terrain.notna().any() else np.nan,
        "z_max_agl_m": float((z1 - terrain).max()) if z1.notna().any() and terrain.notna().any() else np.nan,
    })

    # Source-specific height columns. These are easier to interpret than global MSL.
    height_candidates = []
    if layer_name == "building" and "building_height_agl_m" in df.columns:
        height_candidates.append(pd.to_numeric(df["building_height_agl_m"], errors="coerce"))
    if layer_name == "powerline":
        for c in ["powerline_plot_height_agl_m", "powerline_line_height_agl_m", "powerline_pole_height_agl_m", "powerline_height_agl_m"]:
            if c in df.columns:
                height_candidates.append(pd.to_numeric(df[c], errors="coerce"))
    if height_candidates:
        h = pd.concat(height_candidates, ignore_index=True)
        h = h[np.isfinite(h) & (h > 0.0)]
        if len(h):
            row["height_agl_min_m"] = float(h.min())
            row["height_agl_p50_m"] = float(h.median())
            row["height_agl_max_m"] = float(h.max())
    return row


def _fig02_xy_top_table(df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    """Collapse a plotted layer to one top value per true XY cell."""
    if df is None or df.empty or "xy_id" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    z0, z1 = _fig02_z_bottom_top_columns(d)
    terrain = pd.to_numeric(d.get("terrain_msl_m", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["_z_plot_bottom_msl_m"] = z0
    d["_z_plot_top_msl_m"] = z1
    d["_z_plot_top_agl_m"] = z1 - terrain
    agg = {
        "x_from_sw_m": ("x_from_sw_m", "first"),
        "y_from_sw_m": ("y_from_sw_m", "first"),
        "terrain_msl_m": ("terrain_msl_m", "first"),
        f"{layer_name}_z_top_msl_m": ("_z_plot_top_msl_m", "max"),
        f"{layer_name}_z_top_agl_m": ("_z_plot_top_agl_m", "max"),
        f"{layer_name}_z_bottom_msl_m": ("_z_plot_bottom_msl_m", "min"),
        f"{layer_name}_voxel_count": ("xy_id", "size"),
    }
    if layer_name == "building" and "building_height_agl_m" in d.columns:
        agg["building_height_agl_m"] = ("building_height_agl_m", "max")
    if layer_name == "powerline":
        for c in [
            "powerline_plot_kind",
            "powerline_kind_in_xy",
            "powerline_plot_height_agl_m",
            "powerline_line_height_agl_m",
            "powerline_pole_height_agl_m",
            "powerline_line_top_msl_m",
            "powerline_pole_top_msl_m",
        ]:
            if c in d.columns:
                if d[c].dtype == object:
                    agg[c] = (c, lambda s: "|".join(sorted(set(map(str, s.dropna()))))[:120])
                else:
                    agg[c] = (c, "max")
    return d.groupby("xy_id", as_index=False).agg(**agg)


def save_fig02_layer_height_diagnostics(
    paths: Paths,
    *,
    green: pd.DataFrame | None,
    gray: pd.DataFrame | None,
    yellow: pd.DataFrame | None,
    purple: pd.DataFrame | None,
    axis_limits: dict | None = None,
) -> None:
    """Save Figure-02 layer height QC tables.

    This explains the common confusion:
      * `z_max_msl_m` is absolute elevation, so terrain controls it.
      * `z_max_agl_m` and same-XY comparisons show obstacle height above ground.
    """
    if not bool(FIG02_SAVE_LAYER_HEIGHT_DIAGNOSTICS):
        return

    rows = []
    if PLOT_UNBURNED_3D_DEM_CELLS:
        rows.append(_fig02_layer_extent_row("unburned", green))
    rows.append(_fig02_layer_extent_row("topo", gray))
    rows.append(_fig02_layer_extent_row("building", yellow))
    rows.append(_fig02_layer_extent_row("powerline", purple))
    if axis_limits:
        rows.append({"layer": "figure_axis_limits", **axis_limits})
    extent = pd.DataFrame(rows)
    try:
        extent.to_csv(paths.data_dir / "fig02_visible_layer_extent_qc.csv", index=False)
        print(f"[OK] Saved Figure 02 visible layer height QC: {paths.data_dir / 'fig02_visible_layer_extent_qc.csv'}")
    except Exception as exc:
        print(f"[WARN] Could not save Figure 02 visible layer height QC: {exc}")

    # Save the exact plotted powerline voxel table with AGL columns as well as MSL.
    if purple is not None and not purple.empty:
        try:
            p = purple.copy()
            z0, z1 = _fig02_z_bottom_top_columns(p)
            terrain = pd.to_numeric(p.get("terrain_msl_m", pd.Series(np.nan, index=p.index)), errors="coerce")
            p["z_plot_bottom_agl_m"] = z0 - terrain
            p["z_plot_top_agl_m"] = z1 - terrain
            p.to_csv(paths.data_dir / "fig02_powerline_local_sw_xy_qc.csv", index=False)
            print(f"[OK] Saved Figure 02 powerline local SW XY QC: {paths.data_dir / 'fig02_powerline_local_sw_xy_qc.csv'}")
        except Exception as exc:
            print(f"[WARN] Could not save Figure 02 powerline QC table: {exc}")

    if not bool(FIG02_SAVE_POWERLINE_BUILDING_OVERLAP_QC):
        return
    bxy = _fig02_xy_top_table(yellow, "building")
    pxy = _fig02_xy_top_table(purple, "powerline")
    if bxy.empty or pxy.empty:
        return

    overlap = pxy.merge(
        bxy[[
            "xy_id", "building_z_top_msl_m", "building_z_top_agl_m",
            "building_z_bottom_msl_m", "building_voxel_count",
            "building_height_agl_m",
        ]],
        on="xy_id",
        how="inner",
    )
    if overlap.empty:
        print("[CHECK] No same-XY overlap between Figure 02 powerline and building layers.")
        return

    overlap["powerline_minus_building_top_msl_m"] = (
        pd.to_numeric(overlap["powerline_z_top_msl_m"], errors="coerce")
        - pd.to_numeric(overlap["building_z_top_msl_m"], errors="coerce")
    )
    overlap["powerline_minus_building_top_agl_m"] = (
        pd.to_numeric(overlap["powerline_z_top_agl_m"], errors="coerce")
        - pd.to_numeric(overlap["building_z_top_agl_m"], errors="coerce")
    )
    overlap = overlap.sort_values("powerline_minus_building_top_msl_m")
    try:
        overlap.to_csv(paths.data_dir / "fig02_powerline_vs_building_overlap_qc.csv", index=False)
        print(f"[OK] Saved Figure 02 powerline-vs-building overlap QC: {paths.data_dir / 'fig02_powerline_vs_building_overlap_qc.csv'}")
    except Exception as exc:
        print(f"[WARN] Could not save Figure 02 powerline-vs-building overlap QC: {exc}")

    diff = pd.to_numeric(overlap["powerline_minus_building_top_msl_m"], errors="coerce")
    diff_agl = pd.to_numeric(overlap["powerline_minus_building_top_agl_m"], errors="coerce")
    if diff.notna().any():
        print(
            "[CHECK] Same-XY powerline minus building top elevation: "
            f"MSL min/p50/max={diff.min():.2f}/{diff.median():.2f}/{diff.max():.2f} m; "
            f"AGL min/p50/max={diff_agl.min():.2f}/{diff_agl.median():.2f}/{diff_agl.max():.2f} m; "
            f"overlap_xy={overlap['xy_id'].nunique():,}"
        )
        if diff.min() > 0.0:
            print("[CHECK] In every overlapping XY cell, powerline top is higher than building top.")
        else:
            print("[WARN] Some overlapping XY cells have building top >= powerline top. Check fig02_powerline_vs_building_overlap_qc.csv.")


def plot_3d_dem_burn(paths: Paths, voxels: pd.DataFrame, xy_gdf: gpd.GeoDataFrame, dx: float, dy: float, dz: float) -> None:
    """
    Figure 02: 3D voxel burn state inside the AOI.

    Corrected display logic:
      - Topo/DEM cells can be coarsened for readability.
      - Building cells are drawn from true XY voxel columns by default, not
        from the coarsened topo table. This prevents false high/shifted towers
        from coarse aggregation.
      - In combined view, building cells may be stacked on the TRUE topo voxel
        top of the same xy_id so the class display has no gap.
      - In building-only view, the yellow layer shows the physical MSL interval:
        terrain_msl <= z <= terrain_msl + building_height_agl.
    """
    out_png = paths.fig_dir / "02_3d_dem_burned_cells_SW.png"

    coarse, strides = make_coarse_inside_voxels_for_dem_state_plot(voxels)
    sx, sy, sz = strides
    plot_dx = dx * sx
    plot_dy = dy * sy
    plot_dz = dz * sz

    fig = plt.figure(figsize=(12, 9.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    if not coarse.empty:
        combined_visible_view = bool(PLOT_BURN_CELLS_BY_TOPO and PLOT_BURN_CELLS_BY_BUILDING)

        # Use one consistent Figure-02 grid. The previous mixed mode used
        # coarse topo/unburned blocks in some switch cases and true 50 m
        # building blocks in others. That caused topo-only and unburned views
        # to look wrong even though the underlying burn data were correct.
        use_true_xy_layers = bool(
            PLOT_3D_USE_TRUE_XY_FOR_ALL_SELECTED_LAYERS
            or PLOT_UNBURNED_3D_DEM_CELLS
            or (PLOT_3D_BUILDING_FROM_TRUE_XY_CELLS and PLOT_BURN_CELLS_BY_BUILDING)
            or PLOT_BURN_CELLS_BY_POWERLINE
        )

        if use_true_xy_layers:
            green = make_true_xy_unburned_plot_voxels_for_figure02(voxels)
            gray = make_true_xy_topo_plot_voxels_for_figure02(voxels, dz)
            yellow = make_true_xy_building_plot_voxels_for_figure02(
                paths=paths,
                voxels=voxels,
                xy_gdf=xy_gdf,
                dz=dz,
                combined_visible_view=combined_visible_view,
            )
            if bool(FIG02_POWERLINE_USE_FIG05_QC_LOGIC):
                # IMPORTANT: Figure 05 is already the trusted powerline QC view.
                # Therefore Figure 02 uses the same powerline voxel table,
                # same x_from_sw_m/y_from_sw_m basis, and the same line-air / pole-column intervals.
                purple = make_true_xy_powerline_plot_voxels_for_qc(
                    voxels,
                    max_voxels=int(FIG02_MAX_TRUE_XY_POWERLINE_VOXELS_TO_RENDER),
                )
                if not purple.empty:
                    purple["plot_burn_powerline"] = True
                    purple["plot_burn_building"] = False
                    purple["plot_burn_topo"] = False
                    purple["plot_burn_any"] = True
                    purple["burn_display_class"] = "powerline"
                    purple["_use_true_xy_cell_size"] = True
            else:
                purple = make_true_xy_powerline_plot_voxels_for_figure02(voxels, xy_gdf, dz)
        else:
            green = coarse[~coarse["plot_burn_any"]].copy()
            gray = coarse[coarse["plot_burn_topo"]].copy()
            if not gray.empty:
                gray["burn_display_class"] = "topo"
                if CLIP_3D_TOPO_BURN_PLOT_TO_TERRAIN_SURFACE:
                    gray["z_plot_bottom_msl_m"] = pd.to_numeric(gray["z_bottom_msl_m"], errors="coerce")
                    gray["z_plot_top_msl_m"] = np.minimum(
                        pd.to_numeric(gray["z_top_msl_m"], errors="coerce"),
                        pd.to_numeric(gray["terrain_msl_m"], errors="coerce"),
                    )
                    gray = gray[gray["z_plot_top_msl_m"] > gray["z_plot_bottom_msl_m"]].copy()

            yellow = coarse[coarse["plot_burn_building"]].copy()
            if not yellow.empty:
                yellow["burn_display_class"] = "building"
                if CLIP_3D_BUILDING_BURN_PLOT_TO_TRUE_VOLUME:
                    yellow["z_plot_bottom_msl_m"] = np.maximum(
                        pd.to_numeric(yellow["z_bottom_msl_m"], errors="coerce"),
                        pd.to_numeric(yellow["building_base_msl_m"], errors="coerce"),
                    )
                    yellow["z_plot_top_msl_m"] = np.minimum(
                        pd.to_numeric(yellow["z_top_msl_m"], errors="coerce"),
                        pd.to_numeric(yellow["building_top_msl_m"], errors="coerce"),
                    )
                    yellow = yellow[yellow["z_plot_top_msl_m"] > yellow["z_plot_bottom_msl_m"]].copy()

            purple = coarse[coarse["plot_burn_powerline"]].copy()
            if not purple.empty:
                purple["burn_display_class"] = "powerline"
                if {"powerline_base_msl_m", "powerline_top_msl_m"}.issubset(purple.columns):
                    purple["z_plot_bottom_msl_m"] = np.maximum(
                        pd.to_numeric(purple["z_bottom_msl_m"], errors="coerce"),
                        pd.to_numeric(purple["powerline_base_msl_m"], errors="coerce"),
                    )
                    purple["z_plot_top_msl_m"] = np.minimum(
                        pd.to_numeric(purple["z_top_msl_m"], errors="coerce"),
                        pd.to_numeric(purple["powerline_top_msl_m"], errors="coerce"),
                    )
                    purple = purple[purple["z_plot_top_msl_m"] > purple["z_plot_bottom_msl_m"]].copy()

            stack_building_for_this_view = (
                bool(PLOT_3D_BUILDING_STACK_ON_TOPO)
                and bool(PLOT_BURN_CELLS_BY_TOPO)
                and bool(PLOT_BURN_CELLS_BY_BUILDING)
                and (not yellow.empty)
                and (not gray.empty)
            )
            if stack_building_for_this_view:
                yellow = stack_building_plot_voxels_on_topography(yellow, gray, plot_dz)

        def add_voxel_collection(
            df_part: pd.DataFrame,
            dx_use: float,
            dy_use: float,
            dz_use: float,
            alpha_override: float | None = None,
            zsort: str | None = None,
        ) -> None:
            if df_part is None or df_part.empty:
                return
            faces, facecolors = voxel_block_faces_from_df(df_part, dx_use, dy_use, dz_use, color_mode="dem_state")
            if not faces:
                return
            if alpha_override is not None:
                facecolors = [(float(fc[0]), float(fc[1]), float(fc[2]), float(alpha_override)) for fc in facecolors]
            pc = Poly3DCollection(
                faces,
                facecolors=facecolors,
                edgecolors=(*DEM_STATE_EDGE_BLACK_RGB, DEM_STATE_PLOT_EDGE_ALPHA),
                linewidths=DEM_STATE_PLOT_EDGE_LINEWIDTH,
                antialiased=True,
            )
            if zsort is not None:
                try:
                    pc.set_zsort(zsort)
                except Exception:
                    pass
            ax.add_collection3d(pc)

        def add_powerline_collection_from_fig05_logic(df_part: pd.DataFrame) -> None:
            """Draw Figure-02 purple powerline cells using Figure-05 visual logic."""
            if df_part is None or df_part.empty:
                return
            pdx = dx if ("_use_true_xy_cell_size" in df_part.columns) else plot_dx
            pdy = dy if ("_use_true_xy_cell_size" in df_part.columns) else plot_dy
            pdz = dz if ("_use_true_xy_cell_size" in df_part.columns) else plot_dz
            faces, facecolors = voxel_block_faces_from_df(df_part, pdx, pdy, pdz, color_mode="dem_state")
            if not faces:
                return
            # Same alpha/edge style as Figure 05.
            facecolors = [(float(fc[0]), float(fc[1]), float(fc[2]), float(POWERLINE_VOLUME_PLOT_ALPHA)) for fc in facecolors]
            pc = Poly3DCollection(
                faces,
                facecolors=facecolors,
                edgecolors=(0.10, 0.05, 0.16, POWERLINE_VOLUME_EDGE_ALPHA),
                linewidths=POWERLINE_VOLUME_EDGE_LINEWIDTH,
                antialiased=True,
            )
            try:
                pc.set_zsort("max")
            except Exception:
                pass
            ax.add_collection3d(pc)

        if PLOT_UNBURNED_3D_DEM_CELLS:
            gdx = dx if ("_use_true_xy_cell_size" in green.columns) else plot_dx
            gdy = dy if ("_use_true_xy_cell_size" in green.columns) else plot_dy
            gdz = dz if ("_use_true_xy_cell_size" in green.columns) else plot_dz
            add_voxel_collection(green, gdx, gdy, gdz, zsort="max" if ("_use_true_xy_cell_size" in green.columns) else None)

        # Draw topo and building in the selected visual order. Building uses
        # true voxel cell size when it was rebuilt from true XY cells.
        for layer_name in selected_burn_layer_names():
            if layer_name == "topo":
                tdx = dx if ("_use_true_xy_cell_size" in gray.columns) else plot_dx
                tdy = dy if ("_use_true_xy_cell_size" in gray.columns) else plot_dy
                tdz = dz if ("_use_true_xy_cell_size" in gray.columns) else plot_dz
                add_voxel_collection(gray, tdx, tdy, tdz, zsort="max" if ("_use_true_xy_cell_size" in gray.columns) else None)
            elif layer_name == "building":
                bdx = dx if ("_use_true_xy_cell_size" in yellow.columns) else plot_dx
                bdy = dy if ("_use_true_xy_cell_size" in yellow.columns) else plot_dy
                bdz = dz if ("_use_true_xy_cell_size" in yellow.columns) else plot_dz
                add_voxel_collection(yellow, bdx, bdy, bdz, zsort="max" if ("_use_true_xy_cell_size" in yellow.columns) else None)
            elif layer_name == "powerline":
                # Figure 02 always draws powerline LAST, with the same visual
                # logic as Figure 05, after topo and building have been drawn.
                # Do not draw it inside the normal loop, or matplotlib 3D
                # painter sorting can hide/shift the visual impression.
                continue

        # Hard drawing order for the final QC view:
        #   1) topo/DEM gray, 2) building yellow, 3) powerline purple.
        # The purple layer is drawn using exactly the same voxel table/style as Figure 05.
        if PLOT_BURN_CELLS_BY_POWERLINE:
            add_powerline_collection_from_fig05_logic(purple)

        node_layers = [gray, yellow]
        if bool(FIG02_SHOW_POWERLINE_CENTER_NODES):
            node_layers.append(purple)
        node_layers = [d for d in node_layers if d is not None and not d.empty]
        burned_nodes = pd.concat(node_layers, ignore_index=True) if node_layers else pd.DataFrame()
        if DEM_STATE_SHOW_CENTER_NODES and not burned_nodes.empty:
            if {"z_plot_bottom_msl_m", "z_plot_top_msl_m"}.issubset(burned_nodes.columns):
                z_nodes = 0.5 * (
                    pd.to_numeric(burned_nodes["z_plot_bottom_msl_m"], errors="coerce")
                    + pd.to_numeric(burned_nodes["z_plot_top_msl_m"], errors="coerce")
                )
                z_nodes = z_nodes.fillna(pd.to_numeric(burned_nodes["z_center_msl_m"], errors="coerce"))
            else:
                z_nodes = burned_nodes["z_center_msl_m"]
            ax.scatter(
                burned_nodes["x_from_sw_m"],
                burned_nodes["y_from_sw_m"],
                z_nodes,
                s=DEM_STATE_BURNED_NODE_SIZE,
                c=[DEM_STATE_NODE_RED_RGB],
                alpha=0.90,
                depthshade=False,
            )

        visible_extent_layers = [green if PLOT_UNBURNED_3D_DEM_CELLS else None, gray, yellow, purple]
        visible_extent_layers = [d for d in visible_extent_layers if d is not None and not d.empty]
        if bool(FIG02_AXIS_LIMITS_FROM_VISIBLE_TRUE_XY_LAYERS) and visible_extent_layers:
            all_xy = pd.concat(
                [d[["x_from_sw_m", "y_from_sw_m"]].copy() for d in visible_extent_layers],
                ignore_index=True,
            )
            # Use the true 50 m voxel half-size for true-XY layers.  This keeps
            # the axes consistent with the visible cells and avoids coarse-block
            # limits controlling a true-XY plot.
            x_min = float(pd.to_numeric(all_xy["x_from_sw_m"], errors="coerce").min() - dx / 2.0)
            x_max = float(pd.to_numeric(all_xy["x_from_sw_m"], errors="coerce").max() + dx / 2.0)
            y_min = float(pd.to_numeric(all_xy["y_from_sw_m"], errors="coerce").min() - dy / 2.0)
            y_max = float(pd.to_numeric(all_xy["y_from_sw_m"], errors="coerce").max() + dy / 2.0)
        else:
            x_min = float(coarse["x_from_sw_m"].min() - plot_dx / 2.0)
            x_max = float(coarse["x_from_sw_m"].max() + plot_dx / 2.0)
            y_min = float(coarse["y_from_sw_m"].min() - plot_dy / 2.0)
            y_max = float(coarse["y_from_sw_m"].max() + plot_dy / 2.0)

        displayed_z_bottoms = []
        displayed_z_tops = []
        for _df_z in [green if PLOT_UNBURNED_3D_DEM_CELLS else None, gray, yellow, purple]:
            if _df_z is None or _df_z.empty:
                continue
            if {"z_plot_bottom_msl_m", "z_plot_top_msl_m"}.issubset(_df_z.columns):
                displayed_z_bottoms.append(pd.to_numeric(_df_z["z_plot_bottom_msl_m"], errors="coerce").min())
                displayed_z_tops.append(pd.to_numeric(_df_z["z_plot_top_msl_m"], errors="coerce").max())
            else:
                displayed_z_bottoms.append(pd.to_numeric(_df_z["z_bottom_msl_m"], errors="coerce").min())
                displayed_z_tops.append(pd.to_numeric(_df_z["z_top_msl_m"], errors="coerce").max())
        if displayed_z_tops:
            z_min = float(max(0.0, np.nanmin(displayed_z_bottoms)))
            z_max = float(np.nanmax(displayed_z_tops))
        else:
            z_min = float(max(0.0, coarse["z_bottom_msl_m"].min()))
            z_max = float(pd.to_numeric(coarse["z_top_msl_m"], errors="coerce").max())
        if not np.isfinite(z_max) or z_max <= z_min:
            z_max = z_min + max(float(plot_dz), 1.0)
        z_pad = max(float(dz) * 0.25, 1.0)
        z_max = z_max + z_pad

        # Plot-only: all 3D figures use the same fixed vertical scale.
        # Keep the computed layer z extent in the QC rows above, but use the
        # fixed axis for the actual figure and axis-limit QC row.
        z_min, z_max = fixed_plot_z_range()

        if bool(FIG02_SAVE_LAYER_HEIGHT_DIAGNOSTICS):
            save_fig02_layer_height_diagnostics(
                paths,
                green=green if PLOT_UNBURNED_3D_DEM_CELLS else None,
                gray=gray,
                yellow=yellow,
                purple=purple,
                axis_limits={
                    "n_voxels_displayed": 0,
                    "n_xy_cells_displayed": 0,
                    "x_from_sw_min_m": x_min,
                    "x_from_sw_max_m": x_max,
                    "y_from_sw_min_m": y_min,
                    "y_from_sw_max_m": y_max,
                    "z_min_msl_m": z_min,
                    "z_max_msl_m": z_max,
                },
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, (z_max - z_min) * DEM_STATE_RIGHT_PANEL_Z_EXAGGERATION))
        except Exception:
            pass

        draw_axis_triad_screen_inset(ax)
        n_green = int(len(green)) if PLOT_UNBURNED_3D_DEM_CELLS else int((~coarse["plot_burn_any"]).sum())
        n_gray = int(len(gray))
        n_yellow = int(len(yellow))
        n_purple = int(len(purple))
    else:
        n_green = n_gray = n_yellow = n_purple = 0

    handles = []
    if PLOT_UNBURNED_3D_DEM_CELLS:
        handles.append(Patch(facecolor=(*DEM_STATE_UNBURNED_GREEN_RGB, DEM_STATE_PLOT_GREEN_ALPHA), edgecolor="black", label=f"Non-burned inside AOI ({n_green:,})"))
    if PLOT_BURN_CELLS_BY_TOPO:
        handles.append(Patch(facecolor=(*DEM_STATE_BURNED_GRAY_RGB, DEM_STATE_PLOT_BURNED_GRAY_ALPHA), edgecolor="black", label=f"Topo/DEM-burned inside AOI ({n_gray:,})"))
    if PLOT_BURN_CELLS_BY_BUILDING:
        handles.append(Patch(facecolor=(*DEM_STATE_BUILDING_YELLOW_RGB, DEM_STATE_PLOT_BUILDING_YELLOW_ALPHA), edgecolor="black", label=f"Building-burned inside AOI ({n_yellow:,})"))
    if PLOT_BURN_CELLS_BY_POWERLINE:
        handles.append(Patch(facecolor=(*DEM_STATE_POWERLINE_PURPLE_RGB, DEM_STATE_PLOT_POWERLINE_PURPLE_ALPHA), edgecolor="black", label=f"Powerline/pole geofence inside AOI ({n_purple:,})"))
    handles.append(Patch(facecolor=(*DEM_STATE_NODE_RED_RGB, 0.90), edgecolor="black", label="Selected burned voxel center nodes"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    order_text = normalize_burn_cell_plot_draw_order().replace("_", " → ")
    effective_stack_text = bool(
        (not FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING)
        and PLOT_3D_BUILDING_STACK_ON_TOPO
        and PLOT_BURN_CELLS_BY_TOPO
        and PLOT_BURN_CELLS_BY_BUILDING
        and PLOT_3D_BUILDING_FROM_TRUE_XY_CELLS
    )
    ax.set_title(
        "3D topo/building burn-source state: inside-AOI voxel model"
        f"topo={PLOT_BURN_CELLS_BY_TOPO}, building={PLOT_BURN_CELLS_BY_BUILDING}\n"
        f"order={order_text}"
        f"true_xy_all={PLOT_3D_USE_TRUE_XY_FOR_ALL_SELECTED_LAYERS}, "
        f"true_xy_like_powerline={FIG02_USE_POWERLINE_STYLE_TRUE_XY_FOR_TOPO_BUILDING},\n "
        f"true_xy_building={PLOT_3D_BUILDING_FROM_TRUE_XY_CELLS}, true_xy_topo={PLOT_3D_TOPO_FROM_TRUE_XY_WHEN_TRUE_BUILDING}, "
        f"powerline_xy_grid={FIG02_POWERLINE_USE_XY_GRID_LOCAL_COORDS}, "
        f"powerline_respect_topo={POWERLINE_RESPECT_TOPO_BURN_TOP}, "
        f"effective_stack={effective_stack_text}",
        fontweight="bold",
    )
    ax.set_xlabel("Distance east from SW reference (m)")
    ax.set_ylabel("Distance north from SW reference (m)")
    ax.set_zlabel("Z MSL (m)")
    ax.view_init(elev=24, azim=-45)

    if DEM_STATE_SHOW_NOTE_BOX:
        ax.text2D(
            0.02,
            0.02,
            "Black lines = voxel/block edges\nRed dots = burned voxel center nodes",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
        )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def _downsample_regular_grid_for_mesh(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_grid: np.ndarray,
    max_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a regular grid for faster 3D mesh plotting only."""
    ny, nx = z_grid.shape
    if nx * ny <= max_cells:
        return x_values, y_values, z_grid

    step = int(math.ceil(math.sqrt((nx * ny) / float(max_cells))))
    step = max(1, step)
    return x_values[::step], y_values[::step], z_grid[::step, ::step]


def _make_xy_grid_from_xy_gdf(
    xy_gdf: gpd.GeoDataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make X, Y, Z mesh arrays from the XY-cell table using local SW coordinates."""
    piv = xy_gdf.pivot_table(
        index="y_from_sw_m",
        columns="x_from_sw_m",
        values=value_col,
        aggfunc="mean",
    )
    piv = piv.sort_index(ascending=True)
    x_values = piv.columns.to_numpy(dtype=float)
    y_values = piv.index.to_numpy(dtype=float)
    z_grid = piv.to_numpy(dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, z_grid


def plot_3d_topography_mesh(paths: Paths, xy_gdf: gpd.GeoDataFrame, sw_ref: dict, utm_crs) -> None:
    """
    Plot a true 3D topographic mesh.

    Horizontal coordinates:
        X = distance east from SW reference (m)
        Y = distance north from SW reference (m)

    Vertical coordinate:
        Z = terrain_msl_m, sampled DEM elevation in meters MSL

    Outside-AOI cells are drawn black because they are treated as no-fly,
    but their Z coordinate is still the sampled terrain MSL elevation.
    """
    if not PLOT_3D_TOPOGRAPHY_MESH:
        return

    out_png = paths.fig_dir / "03_3d_topography_mesh_msl_SW.png"

    x_grid, y_grid, z_grid = _make_xy_grid_from_xy_gdf(xy_gdf, "terrain_msl_m")

    # Downsample plotting grid if needed. This affects only the figure, not the model.
    x1d = x_grid[0, :]
    y1d = y_grid[:, 0]
    x1d_ds, y1d_ds, z_ds = _downsample_regular_grid_for_mesh(
        x1d,
        y1d,
        z_grid,
        TOPO_MESH_MAX_GRID_CELLS,
    )
    x_ds, y_ds = np.meshgrid(x1d_ds, y1d_ds)

    finite = np.isfinite(z_ds)
    if not finite.any():
        print("[WARN] Topography mesh skipped because terrain_msl_m has no finite values.")
        return

    norm = plt.Normalize(vmin=DEM_PLOT_VMIN_M, vmax=DEM_PLOT_VMAX_M)
    cmap = plt.get_cmap("terrain")
    facecolors = cmap(norm(np.where(finite, z_ds, np.nan)))
    facecolors[..., -1] = TOPO_MESH_SURFACE_ALPHA

    # Outside polygon should be visually no-fly/black in the mesh too.
    outside_ds = None
    if "inside_polygon" in xy_gdf.columns:
        inside_numeric = xy_gdf.copy()
        inside_numeric["inside_polygon_int"] = pd.to_numeric(
            inside_numeric["inside_polygon"],
            errors="coerce",
        ).fillna(0).astype(int)
        _, _, inside_grid = _make_xy_grid_from_xy_gdf(inside_numeric, "inside_polygon_int")
        _, _, inside_ds = _downsample_regular_grid_for_mesh(
            x1d,
            y1d,
            inside_grid,
            TOPO_MESH_MAX_GRID_CELLS,
        )
        outside_ds = inside_ds < 1
        facecolors[outside_ds] = (0.0, 0.0, 0.0, TOPO_MESH_SURFACE_ALPHA)

    fig = plt.figure(figsize=(12, 9.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x_ds,
        y_ds,
        z_ds,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.18,
        edgecolor=(0.0, 0.0, 0.0, TOPO_MESH_EDGE_ALPHA),
        antialiased=True,
        shade=False,
    )

    # Add AOI/data-box outlines at the local base level for spatial reference.
    z_base = float(np.nanmin(z_ds[finite]))
    aoi_local = load_optional_outline(
        AOI_UTM_FILE,
        utm_crs,
        sw_ref["x_sw_corner_utm_m"],
        sw_ref["y_sw_corner_utm_m"],
    )
    if not aoi_local.empty:
        for geom in aoi_local.geometry:
            if geom.geom_type == "Polygon":
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, zs=z_base, color="black", linewidth=1.2)
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, zs=z_base, color="black", linewidth=1.2)

    data_box_local = load_optional_outline(
        DATA_BOX_UTM_FILE,
        utm_crs,
        sw_ref["x_sw_corner_utm_m"],
        sw_ref["y_sw_corner_utm_m"],
    )
    if not data_box_local.empty:
        for geom in data_box_local.geometry:
            if geom.geom_type == "Polygon":
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, zs=z_base, color="gray", linewidth=0.8, linestyle="--")
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, zs=z_base, color="gray", linewidth=0.8, linestyle="--")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.64, pad=0.08)
    cbar.set_label(f"DEM elevation color scale (m MSL), display {DEM_PLOT_VMIN_M:g}–{DEM_PLOT_VMAX_M:g}")

    handles = [
        Patch(facecolor=cmap(norm((DEM_PLOT_VMIN_M + DEM_PLOT_VMAX_M) / 2.0)), edgecolor="black", label="Topography mesh: Z = DEM MSL"),
    ]
    if outside_ds is not None and np.any(outside_ds):
        handles.append(Patch(facecolor="black", edgecolor="black", label="Outside AOI / no-fly"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_title("3D topography mesh from DEM MSL elevation", fontweight="bold")
    ax.set_xlabel("Distance east from SW reference (m)")
    ax.set_ylabel("Distance north from SW reference (m)")
    ax.set_zlabel("Terrain elevation, Z MSL (m)")
    ax.set_xlim(float(np.nanmin(x_ds)), float(np.nanmax(x_ds)))
    ax.set_ylim(float(np.nanmin(y_ds)), float(np.nanmax(y_ds)))

    zmin_real = float(np.nanmin(z_ds[finite]))
    zmax_real = float(np.nanmax(z_ds[finite]))
    zmin, zmax_plot = fixed_plot_z_range()
    ax.set_zlim(zmin, zmax_plot)

    ax.view_init(elev=32, azim=-48)
    try:
        xr = ax.get_xlim3d()[1] - ax.get_xlim3d()[0]
        yr = ax.get_ylim3d()[1] - ax.get_ylim3d()[0]
        zr = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ax.set_box_aspect((xr, yr, zr * 22.0))
    except Exception:
        pass

    note = (
        "Mesh Z coordinate = terrain_msl_m (DEM elevation, meters MSL)\n"
        f"Color display scale = {DEM_PLOT_VMIN_M:g}–{DEM_PLOT_VMAX_M:g} m; data values are not changed\n"
        f"Real mesh Z range shown: {zmin:.2f}–{zmax_real:.2f} m MSL"
    )
    ax.text2D(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.88),
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def _sample_building_xy_for_plot(xy_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select XY cells with valid building heights for 3D building-volume QC plotting."""
    if "building_height_agl_m" not in xy_gdf.columns:
        return xy_gdf.iloc[0:0].copy()

    out = xy_gdf[pd.to_numeric(xy_gdf["building_height_agl_m"], errors="coerce").fillna(0) >= BUILDING_MIN_HEIGHT_M].copy()
    if out.empty:
        return out

    # Render all if manageable; otherwise downsample randomly for figure speed only.
    if len(out) > MAX_3D_BUILDING_CELLS_TO_RENDER:
        out = out.sample(MAX_3D_BUILDING_CELLS_TO_RENDER, random_state=RANDOM_SEED).copy()
        print(
            f"[INFO] Building QC plot downsampled from {len(xy_gdf[xy_gdf['building_height_agl_m'] >= BUILDING_MIN_HEIGHT_M]):,} "
            f"to {len(out):,} XY building cells for plotting only."
        )
    return out


def _make_prism_faces_from_bottom_top(
    df: pd.DataFrame,
    dx: float,
    dy: float,
    bottom_col: str,
    top_col: str,
    rgba: tuple[float, float, float, float],
    value_col: str | None = None,
    cmap_name: str | None = None,
    norm=None,
) -> tuple[list[list[tuple[float, float, float]]], list[tuple[float, float, float, float]]]:
    """
    Build rectangular prism faces from XY-cell centers and vertical bottom/top values.

    If value_col + cmap_name + norm are provided, each prism is colored by that value.
    Otherwise a constant rgba is used.
    """
    if df.empty:
        return [], []

    hx, hy = dx / 2.0, dy / 2.0
    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []

    cmap = plt.get_cmap(cmap_name) if (value_col is not None and cmap_name is not None and norm is not None) else None

    for row in df.itertuples(index=False):
        x = float(getattr(row, "x_from_sw_m"))
        y = float(getattr(row, "y_from_sw_m"))
        z0 = float(getattr(row, bottom_col))
        z1 = float(getattr(row, top_col))
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z0) and np.isfinite(z1)):
            continue
        if z1 <= z0:
            continue

        if cmap is not None:
            value = float(getattr(row, value_col))
            r, g, b, _ = cmap(norm(value))
            rgba_local = (float(r), float(g), float(b), float(BUILDING_VOLUME_PLOT_ALPHA))
        else:
            rgba_local = rgba

        vertices = np.array([
            [x - hx, y - hy, z0], [x + hx, y - hy, z0], [x + hx, y + hy, z0], [x - hx, y + hy, z0],
            [x - hx, y - hy, z1], [x + hx, y - hy, z1], [x + hx, y + hy, z1], [x - hx, y + hy, z1],
        ], dtype=float)
        face_ids = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        for ids in face_ids:
            faces.append([(float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])) for i in ids])
            facecolors.append(rgba_local)

    return faces, facecolors


def _plot_local_outline_on_3d_axis(ax, gdf_local: gpd.GeoDataFrame, z_level: float, color: str, lw: float, ls: str = "-") -> None:
    """Plot polygon outlines on a constant z plane for spatial reference."""
    if gdf_local is None or gdf_local.empty:
        return
    for geom in gdf_local.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, zs=z_level, color=color, linewidth=lw, linestyle=ls)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot(xs, ys, zs=z_level, color=color, linewidth=lw, linestyle=ls)


def plot_3d_building_volume_check(paths: Paths, xy_gdf: gpd.GeoDataFrame, sw_ref: dict, utm_crs, dx: float, dy: float) -> None:
    """
    Plot 3D building-volume QC data in two panels.

    Left panel:
        Building height in AGL coordinates.
        Base z = 0 m; top z = building_height_agl_m.

    Right panel:
        Plot a gray DEM topography mesh first, then plot the building
        prisms on top of that mesh.
        Building base z = terrain_msl_m; top z = building_top_msl_m.

    Both subfigures use the same XY limits, the same Z-axis limits, and
    the same 3D box aspect, so the visual scale is comparable.

    Important:
        The building prism XY coordinates are recomputed from the UTM
        geometry / UTM center before plotting. This avoids confusing the
        local SW geometry with the real UTM geometry used for collision.
    """
    if not PLOT_3D_BUILDING_VOLUME_CHECK:
        return

    out_png = paths.fig_dir / "04_3d_building_volume_agl_vs_msl_SW.png"

    bxy = _sample_building_xy_for_plot(xy_gdf)
    if bxy.empty:
        print("[WARN] Building 3D QC figure skipped because no valid building XY cells were found. Check the building/cell spatial join log.")
        return

    bxy = bxy.copy()

    # Recompute local plotting coordinates from UTM coordinates / UTM geometry.
    # This is safer for checking whether the building projection is aligned
    # with the topography mesh and the collision grid.
    if {"x_utm_m", "y_utm_m"}.issubset(bxy.columns):
        bxy["x_from_sw_m"] = pd.to_numeric(bxy["x_utm_m"], errors="coerce").astype(float) - sw_ref["x_sw_corner_utm_m"]
        bxy["y_from_sw_m"] = pd.to_numeric(bxy["y_utm_m"], errors="coerce").astype(float) - sw_ref["y_sw_corner_utm_m"]
    elif "geometry_utm" in bxy.columns:
        bxy_utm_geom = gpd.GeoSeries(bxy["geometry_utm"], crs=utm_crs)
        centers = bxy_utm_geom.centroid
        bxy["x_from_sw_m"] = centers.x.to_numpy(dtype=float) - sw_ref["x_sw_corner_utm_m"]
        bxy["y_from_sw_m"] = centers.y.to_numpy(dtype=float) - sw_ref["y_sw_corner_utm_m"]

    bxy["building_height_agl_m"] = pd.to_numeric(bxy["building_height_agl_m"], errors="coerce").astype(float)
    bxy["terrain_msl_m"] = pd.to_numeric(bxy["terrain_msl_m"], errors="coerce").astype(float)
    bxy["building_top_msl_m"] = bxy["terrain_msl_m"] + bxy["building_height_agl_m"] + BUILDING_HEIGHT_BUFFER_M

    # Left panel: pure AGL prism.
    bxy["z_agl_bottom_m"] = 0.0
    bxy["z_agl_top_m"] = bxy["building_height_agl_m"]

    # Right panel: building prisms sit exactly on the DEM/topography.
    bxy["z_msl_bottom_m"] = bxy["terrain_msl_m"]
    bxy["z_msl_top_m"] = bxy["building_top_msl_m"]

    agl_min = float(np.nanmin(bxy["building_height_agl_m"]))
    agl_max = float(np.nanmax(bxy["building_height_agl_m"]))
    if math.isclose(agl_min, agl_max):
        agl_max = agl_min + 1.0
    msl_min = float(np.nanmin(bxy["building_top_msl_m"]))
    msl_max = float(np.nanmax(bxy["building_top_msl_m"]))
    if math.isclose(msl_min, msl_max):
        msl_max = msl_min + 1.0

    agl_norm = plt.Normalize(vmin=agl_min, vmax=agl_max)
    msl_norm = plt.Normalize(vmin=msl_min, vmax=msl_max)

    agl_faces, agl_colors = _make_prism_faces_from_bottom_top(
        bxy, dx, dy, "z_agl_bottom_m", "z_agl_top_m",
        (*BUILDING_VOLUME_AGL_RGB, BUILDING_VOLUME_PLOT_ALPHA),
        value_col="building_height_agl_m",
        cmap_name=BUILDING_VOLUME_AGL_CMAP,
        norm=agl_norm,
    )
    msl_faces, msl_colors = _make_prism_faces_from_bottom_top(
        bxy, dx, dy, "z_msl_bottom_m", "z_msl_top_m",
        (*BUILDING_VOLUME_MSL_RGB, BUILDING_VOLUME_PLOT_ALPHA),
        value_col="building_top_msl_m",
        cmap_name=BUILDING_VOLUME_MSL_CMAP,
        norm=msl_norm,
    )

    # Terrain mesh for the right-hand-side panel.
    x_grid, y_grid, z_grid = _make_xy_grid_from_xy_gdf(xy_gdf, "terrain_msl_m")
    x1d = x_grid[0, :]
    y1d = y_grid[:, 0]
    x1d_ds, y1d_ds, z_ds = _downsample_regular_grid_for_mesh(
        x1d,
        y1d,
        z_grid,
        BUILDING_VOLUME_TERRAIN_MAX_GRID_CELLS,
    )
    x_ds, y_ds = np.meshgrid(x1d_ds, y1d_ds)
    finite = np.isfinite(z_ds)

    # Gray topographic mesh: no height color fill, only geometry.
    terrain_facecolors = np.empty(z_ds.shape + (4,), dtype=float)
    terrain_facecolors[..., 0] = BUILDING_VOLUME_TERRAIN_GRAY_RGB[0]
    terrain_facecolors[..., 1] = BUILDING_VOLUME_TERRAIN_GRAY_RGB[1]
    terrain_facecolors[..., 2] = BUILDING_VOLUME_TERRAIN_GRAY_RGB[2]
    terrain_facecolors[..., 3] = BUILDING_VOLUME_TERRAIN_MESH_ALPHA
    terrain_facecolors[~finite, 3] = 0.0

    # Same XY limits for both subfigures.
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    y_min = float(np.nanmin(y_grid))
    y_max = float(np.nanmax(y_grid))

    # Same Z scale for both subfigures.
    common_zmin, common_zmax = fixed_plot_z_range()

    fig = plt.figure(figsize=(15.0, 8.2), dpi=FIG_DPI)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    edge_rgba = (0.25, 0.25, 0.25, BUILDING_VOLUME_PLOT_EDGE_ALPHA)
    if agl_faces:
        pc1 = Poly3DCollection(
            agl_faces, facecolors=agl_colors, edgecolors=edge_rgba,
            linewidths=BUILDING_VOLUME_PLOT_EDGE_LINEWIDTH, antialiased=True
        )
        ax1.add_collection3d(pc1)

    # Right panel: draw gray terrain mesh first, then building prisms above it.
    ax2.plot_surface(
        x_ds,
        y_ds,
        z_ds,
        facecolors=terrain_facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.10,
        edgecolor=(0.0, 0.0, 0.0, 0.10),
        antialiased=True,
        shade=False,
    )

    if BUILDING_VOLUME_QC_SHOW_BASE_DOTS:
        ax2.scatter(
            bxy["x_from_sw_m"],
            bxy["y_from_sw_m"],
            bxy["terrain_msl_m"],
            s=0.6,
            c=[BUILDING_BASE_PROJECTION_RGB],
            alpha=0.35,
            depthshade=False,
        )

    if msl_faces:
        pc2 = Poly3DCollection(
            msl_faces, facecolors=msl_colors, edgecolors=edge_rgba,
            linewidths=BUILDING_VOLUME_PLOT_EDGE_LINEWIDTH, antialiased=True
        )
        try:
            pc2.set_zsort("max")
        except Exception:
            pass
        ax2.add_collection3d(pc2)

    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(common_zmin, common_zmax)
        ax.view_init(elev=28, azim=-48)
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, max(common_zmax - common_zmin, 1.0) * BUILDING_VOLUME_PLOT_Z_EXAGGERATION))
        except Exception:
            pass
        draw_axis_triad_screen_inset(ax)
        ax.set_xlabel("Distance east from SW reference (m)")
        ax.set_ylabel("Distance north from SW reference (m)")

    ax1.set_title("Building volume QC (AGL height)", fontweight="bold")
    ax1.set_zlabel("Common Z scale (m)")
    ax2.set_title("Building volume QC (DEM mesh + building MSL)", fontweight="bold")
    ax2.set_zlabel("Common Z scale (m)")

    # AOI/data-box outlines.
    aoi_local = load_optional_outline(AOI_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    data_box_local = load_optional_outline(DATA_BOX_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    _plot_local_outline_on_3d_axis(ax1, aoi_local, 0.0, color="black", lw=1.0)
    _plot_local_outline_on_3d_axis(ax1, data_box_local, 0.0, color="gray", lw=0.8, ls="--")
    terrain_outline_z = float(np.nanmin(z_ds[finite])) if finite.any() else 0.0
    _plot_local_outline_on_3d_axis(ax2, aoi_local, terrain_outline_z, color="black", lw=1.0)
    _plot_local_outline_on_3d_axis(ax2, data_box_local, terrain_outline_z, color="gray", lw=0.8, ls="--")

    # Colorbars for building values only. Terrain is intentionally gray.
    sm_agl = plt.cm.ScalarMappable(norm=agl_norm, cmap=plt.get_cmap(BUILDING_VOLUME_AGL_CMAP))
    sm_agl.set_array([])
    cbar1 = fig.colorbar(
        sm_agl,
        ax=ax1,
        orientation="horizontal",
        location="bottom",
        shrink=0.78,
        pad=-0.08,
        fraction=0.05,
    )
    cbar1.set_label("Building height AGL (m)")

    sm_msl = plt.cm.ScalarMappable(norm=msl_norm, cmap=plt.get_cmap(BUILDING_VOLUME_MSL_CMAP))
    sm_msl.set_array([])
    cbar2 = fig.colorbar(
        sm_msl,
        ax=ax2,
        orientation="horizontal",
        location="bottom",
        shrink=0.78,
        pad=-0.08,
        fraction=0.05,
    )
    cbar2.set_label("Building top elevation MSL (m)")

    if BUILDING_VOLUME_QC_SHOW_LEGEND:
        handles_agl = [
            Patch(facecolor=(*BUILDING_VOLUME_AGL_RGB, BUILDING_VOLUME_PLOT_ALPHA), edgecolor="gray", label="Building AGL"),
        ]
        handles_msl = [
            Patch(facecolor=(*BUILDING_VOLUME_TERRAIN_GRAY_RGB, BUILDING_VOLUME_TERRAIN_MESH_ALPHA), edgecolor="gray", label="DEM mesh"),
            Patch(facecolor=(*BUILDING_VOLUME_MSL_RGB, BUILDING_VOLUME_PLOT_ALPHA), edgecolor="gray", label="Building MSL"),
        ]
        ax1.legend(handles=handles_agl, loc="upper left", fontsize=8)
        ax2.legend(handles=handles_msl, loc="upper left", fontsize=8)

    fig.suptitle(
        "3D building-volume QC",
        fontsize=13, fontweight="bold", y=0.985
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def make_true_xy_powerline_plot_voxels_for_qc(
    voxels: pd.DataFrame,
    max_voxels: int | None = None,
) -> pd.DataFrame:
    """Build purple powerline-system plot voxels for the separate topo-mesh QC figure.

    This helper ignores the Figure-02 layer toggle so the dedicated QC figure is
    generated whenever PLOT_3D_POWERLINE_TOPOGRAPHY_CHECK=True and valid powerline
    burn cells exist.
    """
    if voxels is None or voxels.empty:
        return pd.DataFrame()

    parts = []
    configs = [
        (
            "line-air strand/cable",
            "burn_powerline_line_air",
            "powerline_line_effective_base_msl_m",
            "powerline_line_top_msl_m",
            "powerline_line_height_agl_m",
        ),
        (
            "pole/tower column",
            "burn_powerline_pole_column",
            "powerline_pole_effective_base_msl_m",
            "powerline_pole_top_msl_m",
            "powerline_pole_height_agl_m",
        ),
    ]
    for kind_label, burn_col, base_col, top_col, h_col in configs:
        if burn_col not in voxels.columns:
            continue
        vv = voxels[voxels[burn_col].astype(bool)].copy()
        if "inside_polygon" in vv.columns:
            vv = vv[pd.to_numeric(vv["inside_polygon"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if vv.empty or base_col not in vv.columns or top_col not in vv.columns:
            continue

        keep = [
            "xy_id", "ix", "iy", "iz", "x_from_sw_m", "y_from_sw_m",
            "z_center_msl_m", "z_bottom_msl_m", "z_top_msl_m", "terrain_msl_m",
            "powerline_kind_in_xy", "powerline_height_reference_used",
            "topo_burn_top_msl_m",
            "powerline_line_height_agl_m", "powerline_line_base_msl_m", "powerline_line_effective_base_msl_m", "powerline_line_top_msl_m",
            "powerline_pole_height_agl_m", "powerline_pole_base_msl_m", "powerline_pole_effective_base_msl_m", "powerline_pole_top_msl_m",
        ]
        keep = [c for c in keep if c in vv.columns]
        vv = vv[keep].copy()
        vv["z_plot_bottom_msl_m"] = np.maximum(
            pd.to_numeric(vv["z_bottom_msl_m"], errors="coerce"),
            pd.to_numeric(vv[base_col], errors="coerce"),
        )
        vv["z_plot_top_msl_m"] = np.minimum(
            pd.to_numeric(vv["z_top_msl_m"], errors="coerce"),
            pd.to_numeric(vv[top_col], errors="coerce"),
        )
        vv = vv[vv["z_plot_top_msl_m"] > vv["z_plot_bottom_msl_m"]].copy()
        if vv.empty:
            continue
        vv["powerline_plot_kind"] = kind_label
        vv["powerline_plot_height_agl_m"] = pd.to_numeric(vv.get(h_col, 0.0), errors="coerce").fillna(0.0)
        parts.append(vv)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if max_voxels is None:
        max_voxels = int(MAX_3D_POWERLINE_VOXELS_TO_RENDER)
    if len(out) > int(max_voxels):
        print(f"[WARN] Powerline QC voxels={len(out):,}; downsampling to {int(max_voxels):,} for plotting only.")
        out = out.sample(int(max_voxels), random_state=RANDOM_SEED).copy()

    out["gx"] = pd.to_numeric(out["ix"], errors="coerce").astype(int)
    out["gy"] = pd.to_numeric(out["iy"], errors="coerce").astype(int)
    out["gz"] = pd.to_numeric(out["iz"], errors="coerce").astype(int)
    out["burn_display_class"] = "powerline"
    out["_use_true_xy_cell_size"] = True
    return out


def plot_3d_powerline_topography_check(
    paths: Paths,
    xy_gdf: gpd.GeoDataFrame,
    voxels: pd.DataFrame,
    sw_ref: dict,
    utm_crs,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """Plot gray topographic mesh plus purple powerline-system obstacle blocks.

    The purple blocks use the corrected kind-specific vertical burn types:
      - line/strand/cable blocks = airborne band only;
      - pole/tower/pylon blocks = column from terrain to pole top.
    """
    if not PLOT_3D_POWERLINE_TOPOGRAPHY_CHECK:
        return

    out_png = paths.fig_dir / "05_3d_topography_mesh_powerline_system_SW.png"
    pvv = make_true_xy_powerline_plot_voxels_for_qc(voxels)
    if pvv.empty:
        print("[WARN] Powerline/topography QC figure skipped because no powerline burn voxels were found.")
        return

    x_grid, y_grid, z_grid = _make_xy_grid_from_xy_gdf(xy_gdf, "terrain_msl_m")
    x1d = x_grid[0, :]
    y1d = y_grid[:, 0]
    x1d_ds, y1d_ds, z_ds = _downsample_regular_grid_for_mesh(
        x1d,
        y1d,
        z_grid,
        POWERLINE_TOPO_MESH_MAX_GRID_CELLS,
    )
    x_ds, y_ds = np.meshgrid(x1d_ds, y1d_ds)
    finite = np.isfinite(z_ds)

    terrain_facecolors = np.empty(z_ds.shape + (4,), dtype=float)
    terrain_facecolors[..., 0] = POWERLINE_TOPO_MESH_GRAY_RGB[0]
    terrain_facecolors[..., 1] = POWERLINE_TOPO_MESH_GRAY_RGB[1]
    terrain_facecolors[..., 2] = POWERLINE_TOPO_MESH_GRAY_RGB[2]
    terrain_facecolors[..., 3] = POWERLINE_TOPO_MESH_ALPHA
    terrain_facecolors[~finite, 3] = 0.0

    faces, facecolors = voxel_block_faces_from_df(pvv, dx, dy, dz, color_mode="dem_state")
    # Use the dedicated QC alpha for this figure.
    facecolors = [(*fc[:3], POWERLINE_VOLUME_PLOT_ALPHA) for fc in facecolors]

    fig = plt.figure(figsize=(12.6, 9.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_ds,
        y_ds,
        z_ds,
        facecolors=terrain_facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.10,
        edgecolor=(0.0, 0.0, 0.0, POWERLINE_TOPO_MESH_EDGE_ALPHA),
        antialiased=True,
        shade=False,
    )

    if faces:
        pc = Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=(0.10, 0.05, 0.16, POWERLINE_VOLUME_EDGE_ALPHA),
            linewidths=POWERLINE_VOLUME_EDGE_LINEWIDTH,
            antialiased=True,
        )
        try:
            pc.set_zsort("max")
        except Exception:
            pass
        ax.add_collection3d(pc)

    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    y_min = float(np.nanmin(y_grid))
    y_max = float(np.nanmax(y_grid))
    z_min_candidates = [float(np.nanmin(z_ds[finite])) if finite.any() else 0.0]
    z_max_candidates = [float(np.nanmax(z_ds[finite])) if finite.any() else 1.0]
    if {"z_plot_bottom_msl_m", "z_plot_top_msl_m"}.issubset(pvv.columns):
        z_min_candidates.append(float(pd.to_numeric(pvv["z_plot_bottom_msl_m"], errors="coerce").min()))
        z_max_candidates.append(float(pd.to_numeric(pvv["z_plot_top_msl_m"], errors="coerce").max()))
    z_min = max(0.0, float(np.nanmin(z_min_candidates)))
    z_max = float(np.nanmax(z_max_candidates)) + max(float(dz), 1.0)
    if z_max <= z_min:
        z_max = z_min + 1.0

    # Plot-only: fixed vertical scale for comparison with other 3D figures.
    z_min, z_max = fixed_plot_z_range()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=30, azim=-48)
    try:
        ax.set_box_aspect((x_max - x_min, y_max - y_min, max(z_max - z_min, 1.0) * POWERLINE_VOLUME_Z_EXAGGERATION))
    except Exception:
        pass
    draw_axis_triad_screen_inset(ax)

    aoi_local = load_optional_outline(AOI_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    data_box_local = load_optional_outline(DATA_BOX_UTM_FILE, utm_crs, sw_ref["x_sw_corner_utm_m"], sw_ref["y_sw_corner_utm_m"])
    terrain_outline_z = float(np.nanmin(z_ds[finite])) if finite.any() else 0.0
    _plot_local_outline_on_3d_axis(ax, aoi_local, terrain_outline_z, color="black", lw=1.0)
    _plot_local_outline_on_3d_axis(ax, data_box_local, terrain_outline_z, color="gray", lw=0.8, ls="--")

    n_line = int((pvv.get("powerline_plot_kind", pd.Series([], dtype=object)).astype(str).str.contains("line", na=False)).sum())
    n_pole = int((pvv.get("powerline_plot_kind", pd.Series([], dtype=object)).astype(str).str.contains("pole", na=False)).sum())
    handles = [
        Patch(facecolor=(*POWERLINE_TOPO_MESH_GRAY_RGB, POWERLINE_TOPO_MESH_ALPHA), edgecolor="gray", label="Topographic mesh (DEM MSL)"),
        Patch(facecolor=(*DEM_STATE_POWERLINE_PURPLE_RGB, POWERLINE_VOLUME_PLOT_ALPHA), edgecolor="black", label=f"Powerline system blocks ({len(pvv):,})"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title(
        "3D topography mesh + OSM powerline-system obstacle blocks\n"
        f"purple: line air-band={n_line:,}, pole/tower column={n_pole:,}",
        fontweight="bold",
    )
    ax.set_xlabel("Distance east from SW reference (m)")
    ax.set_ylabel("Distance north from SW reference (m)")
    ax.set_zlabel("Z MSL (m)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")

def make_figures(paths: Paths, xy_gdf: gpd.GeoDataFrame, voxels: pd.DataFrame, sw_ref: dict, utm_crs, dx: float, dy: float, dz: float) -> None:
    plot_dem_terrain(paths, xy_gdf, sw_ref, utm_crs)
    plot_dem_burn_z_slices(paths, voxels)
    plot_3d_dem_burn(paths, voxels, xy_gdf, dx, dy, dz)
    plot_3d_topography_mesh(paths, xy_gdf, sw_ref, utm_crs)
    plot_3d_building_volume_check(paths, xy_gdf, sw_ref, utm_crs, dx, dy)
    plot_3d_powerline_topography_check(paths, xy_gdf, voxels, sw_ref, utm_crs, dx, dy, dz)


# ======================================================================
# MAIN
# ======================================================================


def main() -> None:
    print("\n========== DEM + BUILDING + POWERLINE OBSTACLE BURN INTO 3D VOXEL MODEL ==========")
    paths = make_paths()

    summary = load_base_summary()
    utm_crs = get_utm_crs(summary)
    print(f"[INFO] UTM CRS: {utm_crs}")
    print(
        "[INFO] Burn-cell plot options: "
        f"topo={PLOT_BURN_CELLS_BY_TOPO}, "
        f"building={PLOT_BURN_CELLS_BY_BUILDING}, "
        f"powerline={PLOT_BURN_CELLS_BY_POWERLINE}, "
        f"draw_order={normalize_burn_cell_plot_draw_order()}"
    )

    voxels_raw = load_base_voxels()
    voxels, dx, dy, dz, flyable_slowness = prepare_base_voxels(voxels_raw, summary)
    xy_gdf = build_xy_cells(voxels, dx, dy, utm_crs)
    xy_gdf, voxels, sw_ref = add_sw_reference_coordinates(xy_gdf, voxels)

    print(f"[INFO] Voxel size inferred: dx={dx:g}, dy={dy:g}, dz={dz:g} m")
    print(f"[INFO] Base voxels: {len(voxels):,}; XY cells: {len(xy_gdf):,}")

    xy_gdf, terrain_source, terrain_stats = add_terrain_to_xy(paths, xy_gdf)
    voxels = add_dem_burn_columns(voxels, xy_gdf)
    voxels, xy_gdf, building_stats = add_building_burn_columns(paths, voxels, xy_gdf, utm_crs, dx, dy)
    voxels, xy_gdf, powerline_stats = add_powerline_system_burn_columns(paths, voxels, xy_gdf, utm_crs, dx, dy)
    voxels = finalize_dem_only_model(voxels, flyable_slowness)

    save_outputs(paths, voxels, xy_gdf, terrain_source, terrain_stats, sw_ref, dx, dy, dz, flyable_slowness)
    make_figures(paths, xy_gdf, voxels, sw_ref, utm_crs, dx, dy, dz)

    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")
    print(f"Use this final next-step model: {paths.data_dir / f'{FINAL_OUTPUT_MODEL_BASENAME}.csv.gz'}")
    print("Use column: slowness_final_dem_only")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
