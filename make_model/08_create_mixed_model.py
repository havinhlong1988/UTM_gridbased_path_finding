#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create final mixed model for Scenario 1 no velocity.

Main logic:
    - Read raw model:
          output/02_senario1_no_velocity/raw.xyz

      Expected columns:
          longitude latitude elevation_m slowness_s_per_m

    - Read building density grid from the population-density step:
          output/02_senario1_no_velocity/02_population_density/building_density_*.xyz

      Expected columns:
          longitude latitude normalized_density

    - Read building polygons from copied scenario input:
          input/02_data_senario1_no_velocity/buildings

      Building source can be:
          openbuildingmap, globalbuildingatlas, or both

    - Select high-rise buildings from the selected building polygons.
    - Create high-rise no-fly mask from:
          selected-source GPKG high-rise polygons + neighbor nodes
          AND
          normalized building density >= threshold

Final category:
    0 = Flyable
    1 = No-fly

Separate VTK flags:
    existing_nofly = old no-fly from raw.xyz
    highrise_nofly = new high-rise no-fly from GPKG + density threshold

Outputs:
    output/02_senario1_no_velocity/mixed_model.xyz
    output/02_senario1_no_velocity/mixed_model.vtk
    output/02_senario1_no_velocity/mixed_model_nodes.vtk
    output/02_senario1_no_velocity/mixed_model_cage.vtk

Figures:
    output/02_senario1_no_velocity/figures/mixed_model_2d_z0_categorical.png
    output/02_senario1_no_velocity/figures/mixed_model_2d_density_background_categorical_nodes.png
    output/02_senario1_no_velocity/figures/mixed_model_3d_categorical_0_100m.png

VTK coordinate rule:
    If x/y are lon/lat:
        x_vtk = longitude
        y_vtk = latitude
        z_vtk = elevation_m / 111320.0      # degree-equivalent

    If x/y are projected meters:
        x_vtk = x_m / 1000.0
        y_vtk = y_m / 1000.0
        z_vtk = z_m / 1000.0
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import xarray as xr

from scipy.spatial import cKDTree
from shapely.geometry import Polygon


# ============================================================
# USER SETTINGS
# ============================================================

RAW_MODEL_FILE = Path("output/02_senario1_no_velocity/01_raw_model/raw.xyz")

# Building data copied by:
#     01_copy_input_from_download_data.py
#
# Expected examples inside this folder:
#     OBM: obm_buildings_hoalac_clipped.gpkg
#     GBA: gba_lod1_buildings_hoalac_clipped.gpkg
BUILDING_INPUT_DIR = Path("input/02_data_senario1_no_velocity/buildings")

# Choose building polygon source for high-rise / footprint masking:
#   "openbuildingmap"
#   "globalbuildingatlas"
#   "both"
BUILDING_DATA_SOURCE = "globalbuildingatlas"

# To avoid duplicated input copies such as *_001.gpkg, use only the best
# polygon file for each source. Set False only if you intentionally want to
# merge multiple polygon files per source.
USE_ONLY_BEST_VECTOR_FILE_PER_SOURCE = True

# Density grid produced by the building-density step.
# Options:
#   "polygon"
#   "gaussian_points"
#   "combined"
BUILDING_DENSITY_GRID_MODE = "polygon"

BUILDING_DENSITY_GRID_FILES = {
    "polygon": Path(
        "output/02_senario1_no_velocity/02_population_density/building_density_polygon_grid.xyz"
    ),
    "gaussian_points": Path(
        "output/02_senario1_no_velocity/02_population_density/building_density_gaussian_points_grid.xyz"
    ),
    "combined": Path(
        "output/02_senario1_no_velocity/02_population_density/building_density_combined_grid.xyz"
    ),
}

# Fallback legacy OBM path, used only if BUILDING_INPUT_DIR has no OBM file.
LEGACY_OBM_BUILDINGS_GPKG = Path(
    "output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"
)

OUT_DIR = Path("output/02_senario1_no_velocity/03_mixed_model")

# Save all figures outside output folder.
FIG_DIR = Path("figures/02_senario1_no_velocity/03_mixed_model")

OUT_MIXED_XYZ = OUT_DIR / "mixed_model.xyz"
OUT_MIXED_VTK = OUT_DIR / "mixed_model.vtk"
OUT_MIXED_NODES_VTK = OUT_DIR / "mixed_model_nodes.vtk"
OUT_MIXED_CAGE_VTK = OUT_DIR / "mixed_model_cage.vtk"

# ------------------------------------------------------------
# Extract selected 2D model layers for next tests.
# Can be:
#   None          -> disable
#   0             -> single layer
#   (0, 5, 10)    -> multiple layers
# ------------------------------------------------------------
EXTRACT_2D_MODEL = (0, 5, 10)

OUT_EXTRACT_2D_DIR = OUT_DIR / "extracted_2d_models"

# Main mixed-model figures.
# Save directly in:
#     figures/02_senario1_no_velocity/02_mixed_model
OUT_2D_MODEL_FIG = FIG_DIR / "mixed_model_2d_z0_categorical.png"
OUT_2D_DENSITY_NODES_FIG = FIG_DIR / "mixed_model_2d_density_background_categorical_nodes.png"
OUT_3D_MODEL_FIG = FIG_DIR / "mixed_model_3d_categorical_0_100m.png"

PROJECTION = "M15c"
DPI = 300

# Hoa Lac polygon, lon/lat.
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

REGION_PADDING = 0.003

# Convert elevation meters to degree-equivalent when x/y are lon/lat.
METERS_PER_DEGREE = 111320.0

# Final slowness definition used by this mixed model.
# The output mixed_model.xyz will contain ONLY these two slowness values:
#     0.02 = Flyable
#     10   = No-fly
FLYABLE_SLOWNESS = 0.02
NO_FLY_SLOWNESS = 10.0

# Existing raw no-fly threshold.
# Raw files may use 10 or much larger values such as 100000 for no-fly.
# Anything >= 10 is treated as no-fly, then normalized to NO_FLY_SLOWNESS.
EXISTING_NO_FLY_THRESHOLD = 10.0

# New high-rise no-fly slowness.
# Kept for compatibility with the rest of the script.
HIGHRISE_NO_FLY_SLOWNESS = NO_FLY_SLOWNESS

# High-rise selection.
HEIGHT_COLUMN = "height_m"

# If None, use percentile.
HIGHRISE_HEIGHT_M = None
HIGHRISE_HEIGHT_PERCENTILE = 90

# ------------------------------------------------------------
# High-rise mask from GPKG geometry
# ------------------------------------------------------------
# Use model cell area + neighbor nodes.
# Total buffer:
#     (0.5 + HIGHRISE_NEIGHBOR_CELLS) * model_grid_spacing
#
# 0.5 means cell-area intersection.
# 1.0 means one additional neighbor ring.
HIGHRISE_NEIGHBOR_CELLS = 0.5

# Minimum buffer if grid spacing estimate fails.
HIGHRISE_MIN_BUFFER_M = 50.0

# Additional density criterion for high-rise no-fly mask.
# Only keep new no-fly nodes where normalized building density >= this value.
USE_DENSITY_THRESHOLD_FOR_HIGHRISE = True
HIGHRISE_DENSITY_THRESHOLD = 0.60

# If True, every z node under high-rise footprint is set no-fly.
# If False, only nodes from MIN_HIGHRISE_Z_M to MAX_HIGHRISE_Z_M are set no-fly.
SET_FULL_VERTICAL_COLUMN_NO_FLY = True
MIN_HIGHRISE_Z_M = 0.0
MAX_HIGHRISE_Z_M = 3000.0

# 2D categorical layer.
PLOT_2D_TARGET_Z_M = 0.0

# Fast plotting.
# False = plot categorical nodes directly; fast and recommended.
# True  = run pygmt.surface to create filled categorical background; slower.
USE_2D_SURFACE_FILL = True

# Density-background plot backend for:
#     mixed_model_2d_density_background_categorical_nodes.png
#
# Options:
#   "auto"       -> use matplotlib for GBA/both, PyGMT for OBM
#   "pygmt"      -> force PyGMT grdimage
#   "matplotlib" -> avoid PyGMT grdimage
#
# GBA/both can crash in PyGMT/GMT grdimage on some systems, so auto is safer.
DENSITY_BACKGROUND_PLOT_BACKEND = "auto"
AUTO_FORCE_MATPLOTLIB_DENSITY_BACKGROUND_FOR_GBA_OR_BOTH = True

# 3D categorical plot range.
PLOT_3D_Z_MIN_M = 0.0
PLOT_3D_Z_MAX_M = 100.0

# 3D plot settings.
FIG_3D_PROJECTION = "X15c/13c"
FIG_3D_ZSIZE = "7c"
FIG_3D_PERSPECTIVE = [135, 30]

# Downsample 3D points for speed.
MAX_3D_FLYABLE_POINTS = 30_000
MAX_3D_NOFLY_POINTS = 120_000

# Plot node sizes.
DOT_SIZE_2D = "s0.045c"
DOT_SIZE_3D = "c0.035c"

# Transparency.
CATEGORY_TRANSPARENCY_2D = 35
CATEGORY_TRANSPARENCY_3D = 45

# Fixed category colors.
# Important: use fixed fill colors for plotting each class.
# This avoids the PyGMT warning:
#   "Cannot use auto-legend -l for variable symbol color"
CATEGORY_COLORS = {
    0: "180/220/255",  # Flyable
    1: "220/40/40",    # No-fly
}

POLYGON_RGB = "purple"
POLYGON_PEN_2D = f"1.8p,{POLYGON_RGB}"
POLYGON_PEN_3D = f"1.4p,{POLYGON_RGB}"
VERTEX_CONNECT_PEN_3D = f"0.8p,{POLYGON_RGB},-"

# Surface spacing for optional 2D categorical fill.
# If None, estimate from model node spacing.
SURFACE_SPACING_DEG = 0.00025

CLEANUP_CPT_AND_TEMP_FILES = True


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs():
    """
    Create every output directory used by file writers and fig.savefig().

    The previous crash happened because OUT_EXTRACT_2D_DIR was created, but
    the extracted-figure directory under figures/.../04_extracted_2d_models
    was not created before PyGMT saved the PNG.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_EXTRACT_2D_DIR.mkdir(parents=True, exist_ok=True)

    # Figure paths are not all inside FIG_DIR, so create their parents too.
    for path in [
        OUT_2D_MODEL_FIG,
        OUT_2D_DENSITY_NODES_FIG,
        OUT_3D_MODEL_FIG,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_hoalac_polygon():
    geom = Polygon(HOALAC_POLYGON)

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def get_aoi_gdf():
    return gpd.GeoDataFrame(
        {"name": ["HoaLac_polygon"]},
        geometry=[get_hoalac_polygon()],
        crs="EPSG:4326",
    )


def polygon_to_dataframe():
    return pd.DataFrame(HOALAC_POLYGON, columns=["x", "y"])


def get_region_from_polygon(padding=REGION_PADDING):
    poly_df = polygon_to_dataframe()

    xmin = float(poly_df["x"].min()) - padding
    xmax = float(poly_df["x"].max()) + padding
    ymin = float(poly_df["y"].min()) - padding
    ymax = float(poly_df["y"].max()) + padding

    return [xmin, xmax, ymin, ymax]


def make_region_compatible_with_spacing(region, spacing):
    west, east, south, north = region

    nx = int(np.ceil((east - west) / spacing))
    ny = int(np.ceil((north - south) / spacing))

    east_new = west + nx * spacing
    north_new = south + ny * spacing

    return [west, east_new, south, north_new]


def coordinates_look_lonlat(df):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    return (
        np.nanmin(x) >= -180.0
        and np.nanmax(x) <= 180.0
        and np.nanmin(y) >= -90.0
        and np.nanmax(y) <= 90.0
    )


def get_local_utm_epsg_from_lonlat(lon, lat):
    """
    Estimate local UTM EPSG from lon/lat.
    Hoa Lac is normally EPSG:32648.
    """
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg


def estimate_horizontal_grid_spacing_m(raw_df):
    """
    Estimate horizontal model node spacing in meters.

    For lon/lat data:
        dx_degree -> meters using cos(latitude)
        dy_degree -> meters
    """
    xs = np.sort(raw_df["x"].unique())
    ys = np.sort(raw_df["y"].unique())

    dx = np.diff(xs)
    dy = np.diff(ys)

    dx = dx[dx > 0]
    dy = dy[dy > 0]

    if len(dx) == 0 or len(dy) == 0:
        print("[WARNING] Could not estimate grid spacing from nodes.")
        print(f"[WARNING] Use fallback spacing: {HIGHRISE_MIN_BUFFER_M:.2f} m")
        return HIGHRISE_MIN_BUFFER_M

    dx_med = float(np.median(dx))
    dy_med = float(np.median(dy))

    if coordinates_look_lonlat(raw_df):
        lat0 = float(raw_df["y"].median())
        dx_m = dx_med * METERS_PER_DEGREE * np.cos(np.deg2rad(lat0))
        dy_m = dy_med * METERS_PER_DEGREE
        spacing_m = min(abs(dx_m), abs(dy_m))
    else:
        spacing_m = min(abs(dx_med), abs(dy_med))

    if not np.isfinite(spacing_m) or spacing_m <= 0:
        spacing_m = HIGHRISE_MIN_BUFFER_M

    print("")
    print("========== MODEL GRID SPACING ==========")
    print(f"Median dx raw:          {dx_med}")
    print(f"Median dy raw:          {dy_med}")
    print(f"Estimated spacing:      {spacing_m:.2f} m")

    return spacing_m


def vtk_coordinate_arrays(df):
    """
    Convert coordinates for VTK.

    If x/y are lon/lat:
        x = lon
        y = lat
        z = elevation_m / 111320.0, degree-equivalent

    If x/y are projected meters:
        x = x_m / 1000
        y = y_m / 1000
        z = z_m / 1000
    """
    is_lonlat = coordinates_look_lonlat(df)

    if is_lonlat:
        xvtk = df["x"].to_numpy(dtype=float)
        yvtk = df["y"].to_numpy(dtype=float)
        zvtk = df["z"].to_numpy(dtype=float) / METERS_PER_DEGREE
        units = "x/y=lonlat_degree, z=degree_equivalent"
    else:
        xvtk = df["x"].to_numpy(dtype=float) / 1000.0
        yvtk = df["y"].to_numpy(dtype=float) / 1000.0
        zvtk = df["z"].to_numpy(dtype=float) / 1000.0
        units = "x/y/z=km"

    return xvtk, yvtk, zvtk, units


def get_surface_spacing_deg_from_nodes(df):
    if SURFACE_SPACING_DEG is not None:
        return SURFACE_SPACING_DEG

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())

    dxs = np.diff(xs)
    dys = np.diff(ys)

    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]

    if len(dxs) == 0 or len(dys) == 0:
        return 0.00045

    return float(min(np.median(dxs), np.median(dys)))


# ============================================================
# CPT
# ============================================================

def make_density_cpt(cpt_file):
    """
    Make continuous CPT for normalized building density.

    This CPT is only for the background density grid:
        0 = low building density
        1 = high building density
    """
    cpt_file = Path(cpt_file)
    cpt_file.parent.mkdir(parents=True, exist_ok=True)

    pygmt.makecpt(
        cmap="hot",
        series=[0, 1, 0.05],
        reverse=True,
        continuous=True,
        output=str(cpt_file),
    )

    return cpt_file


def make_2class_cpt(cpt_file):
    """
    Make categorical CPT for model node classes.

    Class definition:
        0 = Flyable
        1 = No-fly

    CPT interval definition:
        0 to 1 = Flyable
        1 to 2 = No-fly

    The same CATEGORY_COLORS are also used when plotting the dots,
    so the dot colors and colorbar colors always match.
    """
    cpt_file = Path(cpt_file)
    cpt_file.parent.mkdir(parents=True, exist_ok=True)

    pygmt.makecpt(
        cmap="cool",
        series=[0, 1, 1],
        reverse=False,
        color_model="+cFlyable,Nofly",
        output=str(cpt_file),
    )

    return cpt_file


def add_density_colorbar(fig, density_cpt):
    """
    Add continuous colorbar for normalized building density.
    """
    fig.colorbar(
        cmap=str(density_cpt),
        position="JBC+w8c/0.35c+o0.0c/1.0c+h",
        frame=[
            "xaf+lBuilding density",
            "y+lNormalized",
        ],
    )


def add_node_category_colorbar(fig, category_cpt, position="JMR+w5c/0.35c+o0.8c/0c"):
    """
    Add categorical colorbar for Flyable / No-fly model nodes.

    The CPT interval is 0-1 and 1-2. The labels are stored in the
    CPT using '; Flyable' and '; No-fly'.
    """
    fig.colorbar(
        cmap=str(category_cpt),
        position=position,
        frame=[
            "xa1f1+lNode class",
        ],
    )

# ============================================================
# BUILDING SOURCE HELPERS
# ============================================================

def normalized_building_source():
    source = BUILDING_DATA_SOURCE.lower().strip()

    allowed = {"openbuildingmap", "globalbuildingatlas", "both"}

    if source not in allowed:
        raise ValueError(
            "Invalid BUILDING_DATA_SOURCE. "
            "Use 'openbuildingmap', 'globalbuildingatlas', or 'both'."
        )

    return source


def get_density_grid_file():
    mode = BUILDING_DENSITY_GRID_MODE.lower().strip()

    if mode not in BUILDING_DENSITY_GRID_FILES:
        raise ValueError(
            "Invalid BUILDING_DENSITY_GRID_MODE. "
            "Use 'polygon', 'gaussian_points', or 'combined'."
        )

    path = BUILDING_DENSITY_GRID_FILES[mode]

    if path.exists():
        return path

    # Be conservative: if combined is requested but not available, try polygon.
    fallback = BUILDING_DENSITY_GRID_FILES.get("polygon")

    if mode != "polygon" and fallback is not None and fallback.exists():
        print(
            f"[WARNING] Requested density grid not found: {path}. "
            f"Use fallback polygon density grid: {fallback}"
        )
        return fallback

    raise FileNotFoundError(f"Building density grid not found: {path}")


def get_density_background_plot_backend():
    requested = DENSITY_BACKGROUND_PLOT_BACKEND.lower().strip()
    source = normalized_building_source()

    if requested not in {"auto", "pygmt", "matplotlib"}:
        raise ValueError(
            "Invalid DENSITY_BACKGROUND_PLOT_BACKEND. "
            "Use 'auto', 'pygmt', or 'matplotlib'."
        )

    if requested == "auto":
        if (
            AUTO_FORCE_MATPLOTLIB_DENSITY_BACKGROUND_FOR_GBA_OR_BOTH
            and source in {"globalbuildingatlas", "both"}
        ):
            return "matplotlib"
        return "pygmt"

    if (
        requested == "pygmt"
        and AUTO_FORCE_MATPLOTLIB_DENSITY_BACKGROUND_FOR_GBA_OR_BOTH
        and source in {"globalbuildingatlas", "both"}
    ):
        print(
            "[INFO] Density-background backend requested as PyGMT, "
            "but GBA/both source is active. Force matplotlib to avoid grdimage crash."
        )
        return "matplotlib"

    return requested


def is_vector_file(path: Path):
    return path.suffix.lower() in {".gpkg", ".geojson", ".shp"}


def is_bad_building_vector_candidate(path: Path):
    name = path.name.lower()

    bad_keys = (
        "selected_gba_5deg_tiles",
        "inventory",
        "tile",
        "tiles",
        "centroid",
        "vertices",
        "vertex",
    )

    return any(k in name for k in bad_keys)


def is_openbuildingmap_vector_file(path: Path):
    if not is_vector_file(path):
        return False

    if is_bad_building_vector_candidate(path):
        return False

    p = str(path).lower()
    name = path.name.lower()

    if "openbuildingmap" in p:
        return "building" in name or "buildings" in name or "obm" in name

    if "obm" in name:
        return True

    return False


def is_globalbuildingatlas_vector_file(path: Path):
    if not is_vector_file(path):
        return False

    if is_bad_building_vector_candidate(path):
        return False

    p = str(path).lower()
    name = path.name.lower()

    source_like = (
        "globalbuildingatlas" in p
        or "gba" in name
        or "lod1" in name
    )

    building_like = "building" in name or "buildings" in name

    return source_like and building_like


def score_building_vector_file(path: Path, source_label: str):
    name = path.name.lower()
    suffix = path.suffix.lower()

    score = 0

    if suffix == ".gpkg":
        score += 50
    elif suffix == ".geojson":
        score += 30
    elif suffix == ".shp":
        score += 20

    if "clipped" in name:
        score += 30

    if "hoalac" in name or "hoa_lac" in name:
        score += 20

    if "building" in name or "buildings" in name:
        score += 20

    if source_label == "openbuildingmap":
        if "obm" in name:
            score += 10
        if "openbuildingmap" in str(path).lower():
            score += 10

    if source_label == "globalbuildingatlas":
        if "gba" in name:
            score += 10
        if "lod1" in name:
            score += 10
        if "globalbuildingatlas" in str(path).lower():
            score += 10

    # Prefer original filename over repeated copy suffixes.
    if "_001" in name or "_002" in name or "_003" in name:
        score -= 2

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return score, size


def collect_building_vector_files_for_source(source_label: str):
    if not BUILDING_INPUT_DIR.exists():
        print(f"[WARNING] Building input folder not found: {BUILDING_INPUT_DIR}")
        candidates = []
    else:
        candidates = [f for f in BUILDING_INPUT_DIR.rglob("*") if f.is_file()]

    if source_label == "openbuildingmap":
        files = [f for f in candidates if is_openbuildingmap_vector_file(f)]

        # Legacy fallback for older workflow.
        if not files and LEGACY_OBM_BUILDINGS_GPKG.exists():
            files = [LEGACY_OBM_BUILDINGS_GPKG]

    elif source_label == "globalbuildingatlas":
        files = [f for f in candidates if is_globalbuildingatlas_vector_file(f)]

    else:
        raise ValueError(f"Unknown source label: {source_label}")

    files = sorted(
        files,
        key=lambda f: score_building_vector_file(f, source_label),
        reverse=True,
    )

    # Remove duplicate resolved paths while preserving sorted order.
    unique = []
    seen = set()

    for f in files:
        rp = f.resolve()
        if rp in seen:
            continue
        unique.append(f)
        seen.add(rp)

    if USE_ONLY_BEST_VECTOR_FILE_PER_SOURCE and unique:
        return unique[:1]

    return unique


def get_selected_building_vector_files():
    source = normalized_building_source()

    selected = []

    if source in {"openbuildingmap", "both"}:
        obm_files = collect_building_vector_files_for_source("openbuildingmap")
        selected.extend([("OpenBuildingMap", f) for f in obm_files])

    if source in {"globalbuildingatlas", "both"}:
        gba_files = collect_building_vector_files_for_source("globalbuildingatlas")
        selected.extend([("GlobalBuildingAtlas", f) for f in gba_files])

    if not selected:
        raise FileNotFoundError(
            "No building polygon file found for BUILDING_DATA_SOURCE="
            f"'{BUILDING_DATA_SOURCE}'. Search folder: {BUILDING_INPUT_DIR}"
        )

    return selected


def load_one_building_vector(path: Path, source_label: str):
    print(f"  - {source_label}: {path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"Building polygon file is empty: {path}")

    if gdf.crs is None:
        print(f"    [WARNING] File has no CRS. Assuming EPSG:4326: {path}")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        raise ValueError(f"No Polygon/MultiPolygon geometry found in: {path}")

    aoi = get_aoi_gdf()
    n_before = len(gdf)

    try:
        gdf = gpd.clip(gdf, aoi).copy()
    except Exception as exc:
        print(f"    [WARNING] gpd.clip failed, using intersects only: {exc}")
        gdf = gdf[gdf.intersects(aoi.geometry.iloc[0])].copy()

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if gdf.empty:
        raise ValueError(f"No building polygon left after AOI clip: {path}")

    if HEIGHT_COLUMN in gdf.columns:
        gdf[HEIGHT_COLUMN] = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")

    gdf["_building_source"] = source_label
    gdf["_building_file"] = str(path)

    print(f"    before clip: {n_before:,}")
    print(f"    after clip:  {len(gdf):,}")

    return gdf


def load_buildings_from_selected_source():
    selected_files = get_selected_building_vector_files()

    print("")
    print("========== LOAD BUILDING POLYGONS ==========")
    print(f"Building source option: {BUILDING_DATA_SOURCE}")
    print(f"Building input dir:     {BUILDING_INPUT_DIR}")
    print(f"Files selected:         {len(selected_files)}")

    parts = []

    for source_label, path in selected_files:
        try:
            part = load_one_building_vector(path, source_label)
            parts.append(part)
        except Exception as exc:
            print(f"[WARNING] Failed to load {path}: {exc}")

    if not parts:
        raise RuntimeError(
            "Building polygon files were found, but none could be loaded."
        )

    buildings = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    print("")
    print("========== BUILDING POLYGON SUMMARY ==========")
    print(f"Total selected building polygons: {len(buildings):,}")

    if "_building_source" in buildings.columns:
        counts = buildings["_building_source"].value_counts()
        for source_label, count in counts.items():
            print(f"  {source_label}: {int(count):,}")

    if HEIGHT_COLUMN in buildings.columns:
        vals = pd.to_numeric(buildings[HEIGHT_COLUMN], errors="coerce")
        print(f"Height column:              {HEIGHT_COLUMN}")
        print(f"Height valid count:         {vals.notna().sum():,}")
        print(f"Height max:                 {vals.max()}")
    else:
        print(f"[WARNING] Height column not found: {HEIGHT_COLUMN}")

    return buildings, selected_files

# ============================================================
# READ INPUTS
# ============================================================

def read_raw_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Raw model file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 4:
        raise ValueError(
            f"raw.xyz must have at least 4 columns: x y z slowness. File: {path}"
        )

    df = df.iloc[:, :4].copy()
    df.columns = ["x", "y", "z", "slowness"]

    for col in ["x", "y", "z", "slowness"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "slowness"]).copy()

    print("========== RAW MODEL ==========")
    print(f"Input raw model: {path}")
    print(f"Nodes:           {len(df):,}")
    print(f"x range:         {df['x'].min()} -> {df['x'].max()}")
    print(f"y range:         {df['y'].min()} -> {df['y'].max()}")
    print(f"z range (m):     {df['z'].min()} -> {df['z'].max()}")
    print(f"slowness range:  {df['slowness'].min()} -> {df['slowness'].max()}")
    print(f"Coordinates:     {'lon/lat' if coordinates_look_lonlat(df) else 'projected meter'}")

    return df


def read_density_grid_xyz(path: Path):
    """
    Read building density grid xyz.

    Expected columns:
        lon lat density
    """
    if not path.exists():
        raise FileNotFoundError(f"Building density grid not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 3:
        raise ValueError(
            f"Density grid must have 3 columns: lon lat density. File: {path}"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["x", "y", "density"]

    for col in ["x", "y", "density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "density"]).copy()

    print("")
    print("========== BUILDING DENSITY GRID ==========")
    print(f"Input density grid: {path}")
    print(f"Grid nodes:         {len(df):,}")
    print(f"density range:      {df['density'].min()} -> {df['density'].max()}")

    return df


def density_xyz_to_xarray(density_df):
    """
    Convert lon-lat-density xyz to xarray grid for PyGMT grdimage.
    """
    xs = np.sort(density_df["x"].unique())
    ys = np.sort(density_df["y"].unique())

    pivot = density_df.pivot_table(
        index="y",
        columns="x",
        values="density",
        aggfunc="mean",
    )

    pivot = pivot.reindex(index=ys, columns=xs)

    grid = xr.DataArray(
        pivot.to_numpy(),
        coords={
            "lat": ys,
            "lon": xs,
        },
        dims=("lat", "lon"),
        name="building_density",
    )

    return grid


def load_obm_buildings(path: Path):
    """
    Legacy loader kept for compatibility.
    New workflow uses load_buildings_from_selected_source().
    """
    if not path.exists():
        raise FileNotFoundError(f"OBM building polygon file not found: {path}")

    return load_one_building_vector(path, "OpenBuildingMap")


def select_highrise_buildings(buildings):
    if buildings is None or buildings.empty:
        raise ValueError("No building polygons available for high-rise selection.")

    gdf = buildings.copy()

    if HEIGHT_COLUMN in gdf.columns:
        vals = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")

        if vals.notna().sum() > 0 and vals.max() > 0:
            gdf["_height_for_model"] = vals.fillna(0.0)

            if HIGHRISE_HEIGHT_M is not None:
                threshold = float(HIGHRISE_HEIGHT_M)
            else:
                positive = gdf.loc[gdf["_height_for_model"] > 0, "_height_for_model"]
                threshold = float(np.nanpercentile(positive, HIGHRISE_HEIGHT_PERCENTILE))

            highrise = gdf[gdf["_height_for_model"] >= threshold].copy()
            method = f"{HEIGHT_COLUMN} >= {threshold:.2f} m"

            print("")
            print("========== HIGH-RISE SELECTION ==========")
            print(f"Method:          {method}")
            print(f"All buildings:   {len(gdf):,}")
            print(f"High-rise count: {len(highrise):,}")

            if highrise.empty:
                raise ValueError("High-rise selection is empty. Lower threshold.")

            return highrise, method

    try:
        centroid = gdf.geometry.union_all().centroid
    except Exception:
        centroid = gdf.geometry.unary_union.centroid

    zone = int(np.floor((centroid.x + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone

    gdf_m = gdf.to_crs(epsg)
    areas = gdf_m.geometry.area

    threshold = float(np.nanpercentile(areas, HIGHRISE_HEIGHT_PERCENTILE))
    highrise = gdf.loc[areas >= threshold].copy()

    method = f"footprint area >= P{HIGHRISE_HEIGHT_PERCENTILE}"

    print("")
    print("========== HIGH-RISE SELECTION ==========")
    print(f"Method:          {method}")
    print(f"All buildings:   {len(gdf):,}")
    print(f"High-rise count: {len(highrise):,}")

    if highrise.empty:
        raise ValueError("High-rise footprint selection is empty.")

    return highrise, method


# ============================================================
# MIXED MODEL
# ============================================================

def assign_nearest_density_to_xy_nodes(xy_df, density_df):
    """
    Assign nearest building-density value to each model xy node.

    density_df columns:
        x y density

    xy_df columns:
        x y
    """
    if density_df is None or density_df.empty:
        raise ValueError("density_df is empty. Cannot apply density threshold.")

    den = density_df[["x", "y", "density"]].dropna().copy()

    if den.empty:
        raise ValueError("Density dataframe has no valid x/y/density values.")

    tree = cKDTree(den[["x", "y"]].to_numpy(dtype=float))

    query_xy = xy_df[["x", "y"]].to_numpy(dtype=float)
    dist, idx = tree.query(query_xy, k=1)

    xy_df = xy_df.copy()
    xy_df["building_density"] = den["density"].to_numpy(dtype=float)[idx]
    xy_df["density_nearest_dist"] = dist

    print("")
    print("========== DENSITY THRESHOLD FOR HIGH-RISE MASK ==========")
    print(f"Density threshold:          {HIGHRISE_DENSITY_THRESHOLD}")
    print(
        "Node density min/max:       "
        f"{xy_df['building_density'].min()} -> {xy_df['building_density'].max()}"
    )
    print(
        "Nodes density >= threshold: "
        f"{int((xy_df['building_density'] >= HIGHRISE_DENSITY_THRESHOLD).sum()):,}"
    )

    return xy_df


def find_nodes_inside_highrise(raw_df, highrise_gdf, density_df=None):
    """
    Find model nodes affected by high-rise buildings.

    Method:
        1. Estimate model horizontal grid spacing.
        2. Convert high-rise GPKG polygons to local UTM.
        3. Buffer polygons by:
              (0.5 + HIGHRISE_NEIGHBOR_CELLS) * grid_spacing
           This includes cell-area intersection and neighbor nodes.
        4. Convert mask back to EPSG:4326.
        5. Mark model nodes that intersect this buffered high-rise mask.
        6. Additional criterion:
              keep only nodes where building density >= HIGHRISE_DENSITY_THRESHOLD.
    """
    xy_df = raw_df[["x", "y"]].drop_duplicates().reset_index(drop=True).copy()
    xy_df["xy_id"] = xy_df.index

    if xy_df.empty:
        raise ValueError("No unique x/y model nodes found.")

    if highrise_gdf is None or highrise_gdf.empty:
        raise ValueError("High-rise GPKG selection is empty.")

    grid_spacing_m = estimate_horizontal_grid_spacing_m(raw_df)

    buffer_m = (0.5 + HIGHRISE_NEIGHBOR_CELLS) * grid_spacing_m
    buffer_m = max(buffer_m, HIGHRISE_MIN_BUFFER_M)

    print("")
    print("========== HIGH-RISE AREA + NEIGHBOR MASK ==========")
    print(f"Grid spacing used:       {grid_spacing_m:.2f} m")
    print(f"Neighbor cells:          {HIGHRISE_NEIGHBOR_CELLS:.2f}")
    print(f"Total buffer:            {buffer_m:.2f} m")
    print(f"High-rise polygons raw:  {len(highrise_gdf):,}")

    points_gdf = gpd.GeoDataFrame(
        xy_df,
        geometry=gpd.points_from_xy(xy_df["x"], xy_df["y"]),
        crs="EPSG:4326",
    )

    lon0 = float(xy_df["x"].median())
    lat0 = float(xy_df["y"].median())
    epsg_utm = get_local_utm_epsg_from_lonlat(lon0, lat0)

    highrise = highrise_gdf.to_crs("EPSG:4326").copy()
    highrise = highrise[
        highrise.geometry.notna() & (~highrise.geometry.is_empty)
    ].copy()

    highrise_utm = highrise.to_crs(epsg=epsg_utm).copy()

    # Clean geometry and buffer in meters.
    highrise_utm["geometry"] = highrise_utm.geometry.buffer(0)
    highrise_utm["geometry"] = highrise_utm.geometry.buffer(buffer_m)

    highrise_utm = highrise_utm[
        highrise_utm.geometry.notna() & (~highrise_utm.geometry.is_empty)
    ].copy()

    try:
        mask_geom_utm = highrise_utm.geometry.union_all()
    except Exception:
        mask_geom_utm = highrise_utm.geometry.unary_union

    highrise_mask = gpd.GeoDataFrame(
        {"name": ["highrise_area_neighbor_mask"]},
        geometry=[mask_geom_utm],
        crs=f"EPSG:{epsg_utm}",
    ).to_crs("EPSG:4326")

    joined = gpd.sjoin(
        points_gdf,
        highrise_mask[["geometry"]],
        how="left",
        predicate="intersects",
    )

    inside_ids = joined.loc[joined["index_right"].notna(), "xy_id"].unique()

    xy_df["inside_highrise_geom"] = False
    xy_df.loc[xy_df["xy_id"].isin(inside_ids), "inside_highrise_geom"] = True

    # --------------------------------------------------------
    # Additional criterion: building density >= threshold.
    # --------------------------------------------------------
    if USE_DENSITY_THRESHOLD_FOR_HIGHRISE:
        if density_df is None:
            raise ValueError(
                "USE_DENSITY_THRESHOLD_FOR_HIGHRISE=True but density_df is None."
            )

        xy_df = assign_nearest_density_to_xy_nodes(
            xy_df=xy_df,
            density_df=density_df,
        )

        xy_df["inside_highrise"] = (
            xy_df["inside_highrise_geom"]
            & (xy_df["building_density"] >= HIGHRISE_DENSITY_THRESHOLD)
        )
    else:
        xy_df["building_density"] = np.nan
        xy_df["inside_highrise"] = xy_df["inside_highrise_geom"]

    raw_out = raw_df.merge(
        xy_df[
            [
                "x",
                "y",
                "inside_highrise",
                "inside_highrise_geom",
                "building_density",
            ]
        ],
        on=["x", "y"],
        how="left",
    )

    raw_out["inside_highrise"] = raw_out["inside_highrise"].fillna(False)
    raw_out["inside_highrise_geom"] = raw_out["inside_highrise_geom"].fillna(False)

    if SET_FULL_VERTICAL_COLUMN_NO_FLY:
        z_mask = np.ones(len(raw_out), dtype=bool)
    else:
        z_mask = (
            (raw_out["z"].to_numpy() >= MIN_HIGHRISE_Z_M)
            & (raw_out["z"].to_numpy() <= MAX_HIGHRISE_Z_M)
        )

    highrise_node_mask = raw_out["inside_highrise"].to_numpy() & z_mask

    print("")
    print("========== HIGH-RISE NODE OVERLAY ==========")
    print(f"Unique xy nodes:                         {len(xy_df):,}")
    print(
        "xy inside highrise geometry+neighbor:    "
        f"{int(xy_df['inside_highrise_geom'].sum()):,}"
    )
    print(
        "xy after density threshold:              "
        f"{int(xy_df['inside_highrise'].sum()):,}"
    )
    print(
        "3D nodes inside final highrise no-fly:   "
        f"{int(highrise_node_mask.sum()):,}"
    )
    print(f"Density threshold active:                {USE_DENSITY_THRESHOLD_FOR_HIGHRISE}")
    print(f"Density threshold value:                 {HIGHRISE_DENSITY_THRESHOLD}")
    print(f"Full vertical column no-fly:             {SET_FULL_VERTICAL_COLUMN_NO_FLY}")

    return raw_out, highrise_node_mask


def create_mixed_model(raw_df, highrise_gdf, density_df=None):
    raw_with_flags, highrise_node_mask = find_nodes_inside_highrise(
        raw_df=raw_df,
        highrise_gdf=highrise_gdf,
        density_df=density_df,
    )

    mixed = raw_with_flags.copy()

    # Keep the original input slowness for diagnostics and VTK export.
    mixed["original_slowness"] = mixed["slowness"].copy()

    # Existing no-fly from raw model.
    # New rule:
    #     raw slowness >= 10 is treated as no-fly.
    # This supports both old raw no-fly = 10 and old raw no-fly = 100000.
    mixed["existing_nofly"] = (
        mixed["original_slowness"] >= EXISTING_NO_FLY_THRESHOLD
    ).astype(int)

    # New high-rise no-fly flag.
    mixed["highrise_nofly"] = highrise_node_mask.astype(int)

    # Final combined category:
    #     0 = Flyable
    #     1 = No-fly
    mixed["slowness_class"] = 0
    mixed.loc[
        (mixed["existing_nofly"] == 1) | (mixed["highrise_nofly"] == 1),
        "slowness_class",
    ] = 1

    # --------------------------------------------------------
    # IMPORTANT FINAL NORMALIZATION
    # --------------------------------------------------------
    # Force output slowness to ONLY two values:
    #     0.02 = Flyable
    #     10   = No-fly
    # This removes any 100000 or other old no-fly values from the final model.
    mixed["slowness"] = FLYABLE_SLOWNESS
    mixed.loc[mixed["slowness_class"] == 1, "slowness"] = NO_FLY_SLOWNESS

    changed = mixed["slowness"] != mixed["original_slowness"]

    unique_slow = np.sort(mixed["slowness"].unique())

    print("")
    print("========== MIXED MODEL RESULT ==========")
    print(f"Total nodes:                   {len(mixed):,}")
    print(f"Nodes with changed slowness:   {int(changed.sum()):,}")
    print(f"Existing no-fly nodes kept:    {int((mixed['existing_nofly'] == 1).sum()):,}")
    print(f"High-rise no-fly nodes:        {int((mixed['highrise_nofly'] == 1).sum()):,}")
    print(f"Final no-fly nodes:            {int((mixed['slowness_class'] == 1).sum()):,}")
    print(f"Final flyable slowness:        {FLYABLE_SLOWNESS}")
    print(f"Final no-fly slowness:         {NO_FLY_SLOWNESS}")
    print(f"Final unique slowness values:  {unique_slow}")

    print("")
    print("========== CATEGORY COUNTS ==========")
    for cat, name in [(0, "Flyable"), (1, "No-fly")]:
        ncat = int((mixed["slowness_class"] == cat).sum())
        print(f"{cat}: {name}: {ncat:,}")

    return mixed

def save_mixed_xyz(mixed_df, out_file):
    out_df = mixed_df[["x", "y", "z", "slowness"]].copy()

    out_df.to_csv(
        out_file,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved mixed model XYZ: {out_file}")


# ============================================================
# VTK EXPORT
# ============================================================

def write_vtk_scalar_block(f, name, vtk_type, values):
    f.write(f"SCALARS {name} {vtk_type} 1\n")
    f.write("LOOKUP_TABLE default\n")

    if vtk_type == "int":
        for v in values:
            f.write(f"{int(v)}\n")
    else:
        for v in values:
            f.write(f"{float(v):.8f}\n")

    f.write("\n")


def write_legacy_structured_grid_vtk(df, out_file):
    """
    Write full mixed model as legacy ASCII STRUCTURED_GRID VTK.
    """
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())

    nx, ny, nz = len(xs), len(ys), len(zs)
    expected = nx * ny * nz

    if expected != len(df):
        print(
            "[WARNING] Model is not a complete structured grid. "
            "Writing mixed_model.vtk as POLYDATA instead."
        )
        write_legacy_polydata_nodes_vtk(
            df=df,
            out_file=out_file,
        )
        return

    work = df.copy()
    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(work)

    work["_xvtk"] = xvtk
    work["_yvtk"] = yvtk
    work["_zvtk"] = zvtk

    lookup = work.set_index(["x", "y", "z"])

    points = []
    slowness = []
    original_slowness = []
    existing_nofly = []
    highrise_nofly = []
    slowness_class = []

    for z in zs:
        for y in ys:
            for x in xs:
                row = lookup.loc[(x, y, z)]

                points.append(
                    (
                        float(row["_xvtk"]),
                        float(row["_yvtk"]),
                        float(row["_zvtk"]),
                    )
                )
                slowness.append(float(row["slowness"]))
                original_slowness.append(float(row["original_slowness"]))
                existing_nofly.append(int(row["existing_nofly"]))
                highrise_nofly.append(int(row["highrise_nofly"]))
                slowness_class.append(int(row["slowness_class"]))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_structured_grid coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

        f.write(f"\nPOINT_DATA {len(points)}\n")

        write_vtk_scalar_block(f, "slowness", "float", slowness)
        write_vtk_scalar_block(f, "original_slowness", "float", original_slowness)
        write_vtk_scalar_block(f, "existing_nofly", "int", existing_nofly)
        write_vtk_scalar_block(f, "highrise_nofly", "int", highrise_nofly)
        write_vtk_scalar_block(f, "category", "int", slowness_class)
        write_vtk_scalar_block(f, "slowness_class", "int", slowness_class)

    print(f"[OK] Saved structured-grid VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


def write_legacy_polydata_nodes_vtk(df, out_file):
    """
    Write model nodes as legacy ASCII POLYDATA VTK.
    """
    work = df.copy()
    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(work)

    n = len(work)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_nodes coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")

        for x, y, z in zip(xvtk, yvtk, zvtk):
            f.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")

        f.write(f"\nVERTICES {n} {n * 2}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        f.write(f"\nPOINT_DATA {n}\n")

        write_vtk_scalar_block(f, "slowness", "float", work["slowness"].to_numpy())
        write_vtk_scalar_block(f, "original_slowness", "float", work["original_slowness"].to_numpy())
        write_vtk_scalar_block(f, "existing_nofly", "int", work["existing_nofly"].to_numpy())
        write_vtk_scalar_block(f, "highrise_nofly", "int", work["highrise_nofly"].to_numpy())
        write_vtk_scalar_block(f, "category", "int", work["slowness_class"].to_numpy())
        write_vtk_scalar_block(f, "slowness_class", "int", work["slowness_class"].to_numpy())

    print(f"[OK] Saved node POLYDATA VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


def write_model_cage_vtk(df, out_file):
    """
    Write rectangular model cage as POLYDATA lines.
    """
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    zmin, zmax = float(df["z"].min()), float(df["z"].max())

    cage_df = pd.DataFrame(
        {
            "x": [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin],
            "y": [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax],
            "z": [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],
        }
    )

    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(cage_df)

    points = list(zip(xvtk, yvtk, zvtk))

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_cage coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")

        f.write(f"\nLINES {len(lines)} {len(lines) * 3}\n")

        for i, j in lines:
            f.write(f"2 {i} {j}\n")

    print(f"[OK] Saved model cage VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


# ============================================================
# PLOTTING
# ============================================================

def get_z_layer_model_nodes(mixed_df, target_z_m=0.0):
    """
    Extract only one horizontal model layer for 2D plotting.
    If exact target z does not exist, use nearest z layer.
    """
    zvals = np.sort(mixed_df["z"].unique())

    if len(zvals) == 0:
        raise ValueError("No z values found in mixed model.")

    nearest_z = float(zvals[np.argmin(np.abs(zvals - target_z_m))])

    z_layer = mixed_df[np.isclose(mixed_df["z"], nearest_z)].copy()

    print("")
    print("========== 2D Z-LAYER SELECTION ==========")
    print(f"Target z:       {target_z_m} m")
    print(f"Selected z:     {nearest_z} m")
    print(f"Layer nodes:    {len(z_layer):,}")

    for cat, name in [(0, "Flyable"), (1, "No-fly")]:
        ncat = int((z_layer["slowness_class"] == cat).sum())
        print(f"Layer class {cat} {name}: {ncat:,}")

    return z_layer, nearest_z


def make_2d_category_figure_base(region, title):
    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="DDD.xxx",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="9p",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            "xaf",
            "yaf",
            f'WSen+t"{title}"',
        ],
    )

    return fig


def add_aoi_polygon_2d(fig):
    poly_df = polygon_to_dataframe()
    fig.plot(
        x=poly_df["x"],
        y=poly_df["y"],
        pen=POLYGON_PEN_2D,
        label="AOI polygon",
    )


def add_category_nodes_2d(fig, z0, cpt_file=None):
    """
    Plot class nodes using colors directly from CPT.

    The node color is controlled by:
        fill = slowness_class
        cmap = categorical CPT

    Class:
        0 = Flyable
        1 = No-fly
    """

    for cat, label in [
        (0, "Flyable"),
        (1, "No-fly"),
    ]:
        sub = z0[z0["slowness_class"] == cat].copy()

        if sub.empty:
            continue

        if cpt_file is not None:
            fig.plot(
                x=sub["x"],
                y=sub["y"],
                fill=sub["slowness_class"].astype(int),
                cmap=str(cpt_file),
                style=DOT_SIZE_2D,
                pen=None,
                label=label,
            )
        else:
            fig.plot(
                x=sub["x"],
                y=sub["y"],
                style=DOT_SIZE_2D,
                pen="0.2p,black",
                label=label,
            )

        print(f"2D class {cat} {label}: {len(sub):,} nodes")


def plot_2d_z0_categorical(mixed_df, region0, out_png):
    """
    2D geographic map at z = 0 m.

    Fast default:
        Plot categorical model nodes directly.

    Optional:
        USE_2D_SURFACE_FILL = True will add filled categorical surface,
        but it can be slow for many nodes.
    """
    print("")
    print("========== PLOT 2D Z0 CATEGORICAL ==========")

    z0, selected_z = get_z_layer_model_nodes(
        mixed_df=mixed_df,
        target_z_m=PLOT_2D_TARGET_Z_M,
    )

    cpt_file = make_2class_cpt(FIG_DIR / "mixed_category_2class.cpt")

    region = region0

    if USE_2D_SURFACE_FILL:
        spacing = get_surface_spacing_deg_from_nodes(z0)
        region = make_region_compatible_with_spacing(region0, spacing)

    fig = make_2d_category_figure_base(
        region=region,
        title=f"Mixed model categorical z={selected_z:.0f} m",
    )

    if USE_2D_SURFACE_FILL:
        xyz_file = FIG_DIR / "_tmp_mixed_z0_category.xyz"
        grid_file = FIG_DIR / "_tmp_mixed_z0_category_surface.nc"

        np.savetxt(
            xyz_file,
            z0[["x", "y", "slowness_class"]].to_numpy(),
            fmt="%.8f %.8f %d",
        )

        pygmt.surface(
            data=str(xyz_file),
            region=region,
            spacing=spacing,
            outgrid=str(grid_file),
        )

        grid_cat = pygmt.grdclip(
            grid=str(grid_file),
            below=[0.5, 0],
            above=[0.5, 1],
        )

        fig.grdimage(
            grid=grid_cat,
            cmap=str(cpt_file),
            transparency=CATEGORY_TRANSPARENCY_2D,
        )

    add_category_nodes_2d(fig, z0, cpt_file)
    add_aoi_polygon_2d(fig)

    fig.legend(
        position="JBL+jBL+o0.3c/0.3c",
        box="+gwhite+p0.8p,black",
    )

    # Vertical categorical node-class colorbar on the right.
    add_node_category_colorbar(fig, cpt_file)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved 2D categorical figure: {out_png}")


def plot_2d_density_background_categorical_nodes_matplotlib(
    mixed_df,
    density_grid,
    region0,
    out_png,
):
    """
    Safe density-background plot using Matplotlib.

    This avoids PyGMT grdimage, which can crash with some GBA/both grids on
    some GMT installations.
    """
    import matplotlib.pyplot as plt

    print("")
    print("========== PLOT 2D DENSITY BACKGROUND + CATEGORICAL NODES ==========")
    print("Backend: matplotlib")

    z0, selected_z = get_z_layer_model_nodes(
        mixed_df=mixed_df,
        target_z_m=PLOT_2D_TARGET_Z_M,
    )

    lon = density_grid["lon"].to_numpy()
    lat = density_grid["lat"].to_numpy()
    den = density_grid.to_numpy()

    fig, ax = plt.subplots(figsize=(11, 8.5))

    im = ax.imshow(
        den,
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        origin="lower",
        cmap="hot_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )

    for cat, label in [(0, "Flyable"), (1, "No-fly")]:
        sub = z0[z0["slowness_class"] == cat]
        if sub.empty:
            continue

        # Matplotlib accepts RGB in 0-1 range.
        rgb = [int(v) / 255.0 for v in CATEGORY_COLORS[cat].split("/")]

        ax.scatter(
            sub["x"],
            sub["y"],
            s=7,
            c=[rgb],
            marker="s",
            linewidths=0.0,
            label=label,
            alpha=0.85,
        )

        print(f"2D class {cat} {label}: {len(sub):,} nodes")

    poly_df = polygon_to_dataframe()
    ax.plot(
        poly_df["x"],
        poly_df["y"],
        color=POLYGON_RGB,
        linewidth=1.8,
        label="AOI polygon",
    )

    ax.set_xlim(region0[0], region0[1])
    ax.set_ylim(region0[2], region0[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Building density + mixed model nodes z={selected_z:.0f} m")
    ax.legend(loc="lower left", frameon=True)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.75)
    cbar.set_label("Building density, normalized")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved density background + categorical nodes figure: {out_png}")


def plot_2d_density_background_categorical_nodes(
    mixed_df,
    density_grid,
    region0,
    out_png,
):
    """
    Plot building density gradient in background,
    then overlay z=0 model nodes with categorical color.

    Background:
        building density from building_density_polygon_grid.xyz

    Nodes:
        0 = Flyable
        1 = No-fly
    """
    backend = get_density_background_plot_backend()

    if backend == "matplotlib":
        return plot_2d_density_background_categorical_nodes_matplotlib(
            mixed_df=mixed_df,
            density_grid=density_grid,
            region0=region0,
            out_png=out_png,
        )

    print("")
    print("========== PLOT 2D DENSITY BACKGROUND + CATEGORICAL NODES ==========")
    print("Backend: PyGMT")

    z0, selected_z = get_z_layer_model_nodes(
        mixed_df=mixed_df,
        target_z_m=PLOT_2D_TARGET_Z_M,
    )

    # Two separated CPTs:
    #   1. density_cpt  = continuous background building density
    #   2. category_cpt = categorical node class dots / colorbar
    density_cpt = make_density_cpt(OUT_DIR / "mixed_density_background.cpt")
    category_cpt = make_2class_cpt(FIG_DIR / "mixed_category_2class.cpt")

    fig = make_2d_category_figure_base(
        region=region0,
        title=f"Building density + mixed model nodes z={selected_z:.0f} m",
    )

    fig.grdimage(
        grid=density_grid,
        cmap=str(density_cpt),
        transparency=0,
    )

    add_category_nodes_2d(fig, z0, category_cpt)
    add_aoi_polygon_2d(fig)

    fig.legend(
        position="JBL+jBL+o0.3c/0.3c",
        box="+gwhite+p0.8p,black",
    )

    # Colorbar 1: continuous building density.
    add_density_colorbar(fig, density_cpt)

    # Colorbar 2: categorical model nodes.
    add_node_category_colorbar(fig, category_cpt)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)

    print(f"[OK] Saved density background + categorical nodes figure: {out_png}")


def sample_for_3d_plot(df, max_points, random_state=12345):
    if df is None or df.empty:
        return df

    if max_points is None:
        return df

    if len(df) <= max_points:
        return df

    return df.sample(
        n=max_points,
        random_state=random_state,
    ).copy()


def plot_3d_nodes_categorical(mixed_df, region0, out_png):
    """
    3D perspective view of model nodes from 0 to 100 m.

    Uses categorical CPT:
        0 = Flyable
        1 = No-fly

    For PyGMT plotting:
        x/y are longitude/latitude
        z is converted to degree-equivalent.
    """
    print("")
    print("========== PLOT 3D CATEGORICAL NODES 0-100 M ==========")

    df = mixed_df[
        (mixed_df["z"] >= PLOT_3D_Z_MIN_M)
        & (mixed_df["z"] <= PLOT_3D_Z_MAX_M)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No model nodes found between z={PLOT_3D_Z_MIN_M} and z={PLOT_3D_Z_MAX_M} m."
        )

    df["z_plot"] = df["z"] / METERS_PER_DEGREE

    zmin_plot = PLOT_3D_Z_MIN_M / METERS_PER_DEGREE
    zmax_plot = PLOT_3D_Z_MAX_M / METERS_PER_DEGREE

    xmin, xmax, ymin, ymax = region0
    region3d = [xmin, xmax, ymin, ymax, zmin_plot, zmax_plot]

    cpt_file = make_2class_cpt(FIG_DIR / "mixed_category_2class.cpt")

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd:mmF",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="8p",
    )

    fig.basemap(
        region=region3d,
        projection=FIG_3D_PROJECTION,
        zsize=FIG_3D_ZSIZE,
        perspective=FIG_3D_PERSPECTIVE,
        frame=[
            "xaf+lLongitude",
            "yaf+lLatitude",
            "zaf+lElevation degree-equivalent",
            'WSenZ+b+t"Mixed model: categorical 3D nodes 0-100 m"',
        ],
    )

    max_points = {
        0: MAX_3D_FLYABLE_POINTS,
        1: MAX_3D_NOFLY_POINTS,
    }

    seeds = {
        0: 12345,
        1: 12346,
    }

    for cat in [0, 1]:
        sub = df[df["slowness_class"] == cat].copy()

        sub = sample_for_3d_plot(
            sub,
            max_points=max_points[cat],
            random_state=seeds[cat],
        )

        if sub is None or sub.empty:
            continue

        fig.plot3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["z_plot"],
            fill=sub["slowness_class"],
            cmap=str(cpt_file),
            style=DOT_SIZE_3D,
            pen=None,
            transparency=CATEGORY_TRANSPARENCY_3D,
            perspective=FIG_3D_PERSPECTIVE,
        )

        print(f"3D category {cat}: {len(sub):,} plotted nodes")

    # Bottom rectangular frame.
    bx = [xmin, xmax, xmax, xmin, xmin]
    by = [ymin, ymin, ymax, ymax, ymin]
    bz = [zmin_plot] * len(bx)

    fig.plot3d(
        x=bx,
        y=by,
        z=bz,
        pen="1.2p,black",
        perspective=FIG_3D_PERSPECTIVE,
    )

    # AOI polygon at bottom and top.
    poly_df = polygon_to_dataframe()
    px = poly_df["x"].to_numpy()
    py = poly_df["y"].to_numpy()

    fig.plot3d(
        x=px,
        y=py,
        z=np.full_like(px, zmin_plot, dtype=float),
        pen=POLYGON_PEN_3D,
        perspective=FIG_3D_PERSPECTIVE,
    )

    fig.plot3d(
        x=px,
        y=py,
        z=np.full_like(px, zmax_plot, dtype=float),
        pen=POLYGON_PEN_3D,
        perspective=FIG_3D_PERSPECTIVE,
    )

    # Vertical connectors.
    for vx, vy in zip(px[:-1], py[:-1]):
        fig.plot3d(
            x=[vx, vx],
            y=[vy, vy],
            z=[zmin_plot, zmax_plot],
            pen=VERTEX_CONNECT_PEN_3D,
            perspective=FIG_3D_PERSPECTIVE,
        )

    add_node_category_colorbar(
        fig,
        cpt_file,
        position="JBC+w8c/0.35c+h+o0c/1.0c",
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved 3D categorical figure: {out_png}")

def normalize_extract_2d_model_levels(value):
    """
    Normalize EXTRACT_2D_MODEL to a list of float z targets.
    """
    if value is None or value is False:
        return []

    if np.isscalar(value):
        return [float(value)]

    try:
        return [float(v) for v in value]
    except TypeError:
        return [float(value)]


def format_z_tag(z):
    """
    Convert z value to safe filename tag.
    Example:
        0    -> 0
        5    -> 5
        10.5 -> 10p5
        -5   -> neg5
    """
    s = f"{float(z):.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "neg").replace(".", "p")
    return s


def save_extracted_2d_model(z_layer_df, out_file):
    """
    Save extracted 2D model layer for next tests.

    Columns:
        x y z slowness slowness_class existing_nofly highrise_nofly
    """
    cols = [
        "x",
        "y",
        "z",
        "slowness",
        "slowness_class",
        "existing_nofly",
        "highrise_nofly",
    ]

    out = z_layer_df[cols].copy()

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(
            "# x y z slowness slowness_class existing_nofly highrise_nofly\n"
        )
        np.savetxt(
            f,
            out.to_numpy(),
            fmt="%.8f %.8f %.8f %.8e %d %d %d",
        )

    print(f"[OK] Saved extracted 2D model: {out_file}")


def plot_extracted_2d_model(z_layer_df, region0, selected_z, requested_z, out_png):
    """
    Plot extracted 2D categorical model layer.
    """
    print("")
    print("========== PLOT EXTRACTED 2D MODEL ==========")
    print(f"Requested z: {requested_z} m")
    print(f"Selected z:  {selected_z} m")
    print(f"Nodes:       {len(z_layer_df):,}")

    cpt_file = make_2class_cpt(FIG_DIR / "mixed_category_2class.cpt")

    fig = make_2d_category_figure_base(
        region=region0,
        title=f"Extracted 2D model req={requested_z:g} m, sel={selected_z:g} m",
    )

    add_category_nodes_2d(fig, z_layer_df, cpt_file)
    add_aoi_polygon_2d(fig)

    fig.legend(
        position="JBL+jBL+o0.3c/0.3c",
        box="+gwhite+p0.8p,black",
    )

    add_node_category_colorbar(fig, cpt_file)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved extracted 2D figure: {out_png}")


def extract_selected_2d_models(mixed_df, region0):
    """
    Extract selected 2D model layers based on EXTRACT_2D_MODEL.

    Example:
        EXTRACT_2D_MODEL = 0
        EXTRACT_2D_MODEL = (0, 5, 10)
    """
    z_targets = normalize_extract_2d_model_levels(EXTRACT_2D_MODEL)

    if len(z_targets) == 0:
        print("")
        print("========== EXTRACT 2D MODEL ==========")
        print("Disabled: EXTRACT_2D_MODEL is None/False")
        return []

    results = []

    print("")
    print("========== EXTRACT 2D MODEL ==========")
    print(f"Requested layers: {z_targets}")

    for requested_z in z_targets:
        z_layer_df, selected_z = get_z_layer_model_nodes(
            mixed_df=mixed_df,
            target_z_m=requested_z,
        )

        req_tag = format_z_tag(requested_z)
        sel_tag = format_z_tag(selected_z)

        out_xyz = (
            OUT_EXTRACT_2D_DIR
            / f"mixed_model_2d_req_{req_tag}m_sel_{sel_tag}m.xyz"
        )

        out_png = (
            FIG_DIR
            # / "03_extracted_2d_models"
            / f"mixed_model_2d_req_{req_tag}m_sel_{sel_tag}m.png"
        )

        save_extracted_2d_model(
            z_layer_df=z_layer_df,
            out_file=out_xyz,
        )

        plot_extracted_2d_model(
            z_layer_df=z_layer_df,
            region0=region0,
            selected_z=selected_z,
            requested_z=requested_z,
            out_png=out_png,
        )

        results.append(
            {
                "requested_z": requested_z,
                "selected_z": selected_z,
                "xyz": out_xyz,
                "png": out_png,
            }
        )

    return results

# ============================================================
# CLEANUP
# ============================================================

def cleanup_cpt_and_temp_files():
    if not CLEANUP_CPT_AND_TEMP_FILES:
        return

    print("")
    print("========== CLEANUP TEMP FILES ==========")

    removed = 0

    patterns = [
        OUT_DIR / "*.cpt",
        FIG_DIR / "*.cpt",
        FIG_DIR / "_tmp_*",
    ]

    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                    print(f"[CLEAN] Removed: {path}")
                except Exception as exc:
                    print(f"[WARN] Could not remove {path}: {exc}")

    print(f"[OK] Cleanup done. Removed files: {removed}")


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("\n========== CREATE MIXED MODEL ==========")

    region = get_region_from_polygon(padding=REGION_PADDING)

    raw_df = read_raw_model(RAW_MODEL_FILE)

    density_file = get_density_grid_file()
    density_df = read_density_grid_xyz(density_file)
    density_grid = density_xyz_to_xarray(density_df)

    buildings, building_files = load_buildings_from_selected_source()
    highrise, highrise_method = select_highrise_buildings(buildings)

    mixed_df = create_mixed_model(
        raw_df=raw_df,
        highrise_gdf=highrise,
        density_df=density_df,
    )

    save_mixed_xyz(
        mixed_df=mixed_df,
        out_file=OUT_MIXED_XYZ,
    )

    write_legacy_structured_grid_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_VTK,
    )

    write_legacy_polydata_nodes_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_NODES_VTK,
    )

    write_model_cage_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_CAGE_VTK,
    )

    print("\nPlotting three figures with PyGMT...")

    plot_2d_z0_categorical(
        mixed_df=mixed_df,
        region0=region,
        out_png=OUT_2D_MODEL_FIG,
    )

    plot_2d_density_background_categorical_nodes(
        mixed_df=mixed_df,
        density_grid=density_grid,
        region0=region,
        out_png=OUT_2D_DENSITY_NODES_FIG,
    )

    plot_3d_nodes_categorical(
        mixed_df=mixed_df,
        region0=region,
        out_png=OUT_3D_MODEL_FIG,
    )

    extracted_2d_results = extract_selected_2d_models(
        mixed_df=mixed_df,
        region0=region,
    )

    cleanup_cpt_and_temp_files()

    print("\n========== DONE ==========")
    print(f"Building data source:       {BUILDING_DATA_SOURCE}")
    print(f"Density grid mode:          {BUILDING_DENSITY_GRID_MODE}")
    print(f"Density grid file:          {density_file}")
    print(f"Density background backend: {get_density_background_plot_backend()}")
    print("Building polygon files:")
    for source_label, path in building_files:
        print(f"  {source_label}: {path}")
    print(f"Mixed model XYZ:            {OUT_MIXED_XYZ}")
    print(f"Mixed model VTK:            {OUT_MIXED_VTK}")
    print(f"Mixed model nodes VTK:      {OUT_MIXED_NODES_VTK}")
    print(f"Mixed model cage VTK:       {OUT_MIXED_CAGE_VTK}")
    print(f"2D model figure:            {OUT_2D_MODEL_FIG}")
    print(f"2D density+nodes figure:    {OUT_2D_DENSITY_NODES_FIG}")
    print(f"3D model figure:            {OUT_3D_MODEL_FIG}")
    print(f"Extracted 2D dir:          {OUT_EXTRACT_2D_DIR}")

    if len(extracted_2d_results) > 0:
        print("Extracted 2D layers:")
        for item in extracted_2d_results:
            print(
                f"  req={item['requested_z']} m | "
                f"sel={item['selected_z']} m | "
                f"xyz={item['xyz']} | "
                f"png={item['png']}"
            )    
    print(f"High-rise method:           {highrise_method}")
    print(f"Neighbor cells:             {HIGHRISE_NEIGHBOR_CELLS}")
    print(f"Density threshold active:   {USE_DENSITY_THRESHOLD_FOR_HIGHRISE}")
    print(f"Density threshold:          {HIGHRISE_DENSITY_THRESHOLD}")
    print(f"Flyable slowness:           {FLYABLE_SLOWNESS}")
    print(f"No-fly slowness:            {NO_FLY_SLOWNESS}")
    print(f"Existing no-fly threshold:  {EXISTING_NO_FLY_THRESHOLD}")
    print(f"High-rise no-fly slowness:  {HIGHRISE_NO_FLY_SLOWNESS}")
    print("VTK units if lon/lat:       x/y=degree, z=elevation_m/111320 degree-equivalent")
    print("VTK categories:             0=Flyable, 1=No-fly")
    print("VTK separate flags:         existing_nofly and highrise_nofly are still saved")


if __name__ == "__main__":
    main()