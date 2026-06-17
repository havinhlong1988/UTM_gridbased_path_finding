#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create building-density maps from copied Scenario-1 building inputs.

Building density can be created from:
  - OpenBuildingMap
  - GlobalBuildingAtlas
  - both

Input folder:
    input/02_data_senario1_no_velocity/buildings/

This script supports:
  1. Polygon footprint density from copied GPKG/GeoJSON/SHP building polygons.
  2. Gaussian point density from copied centroid/vertices XYZ files.
  3. Combined density from polygon + point density.

Low-memory mode:
  - reads point XYZ files in chunks
  - writes density XYZ files row-by-row
  - avoids keeping all point inputs in memory
  - limits the number of building polygons/points plotted
  - uses float32 grids where possible

Outputs:
    output/02_senario1_no_velocity/02_population_density/building_density_polygon_grid.xyz
    output/02_senario1_no_velocity/02_population_density/building_density_gaussian_points_grid.xyz
    output/02_senario1_no_velocity/02_population_density/building_density_combined_grid.xyz
    output/02_senario1_no_velocity/02_population_density/building_density_summary.csv

Figures:
    figures/02_senario1_no_velocity/02_population_density/buildings_by_height.png
    figures/02_senario1_no_velocity/02_population_density/building_density_polygon_map.png
    figures/02_senario1_no_velocity/02_population_density/building_density_gaussian_points_map.png
    figures/02_senario1_no_velocity/02_population_density/building_density_combined_map.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import xarray as xr

from scipy.ndimage import gaussian_filter
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

# Use Matplotlib only for the safest low-memory density raster plots.
# This avoids GMT/PyGMT grdimage crashes on some large grids.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


# ============================================================
# USER SETTINGS
# ============================================================

PROJECT_DIR = Path(".").resolve()

# New copied input folder.
BUILDING_INPUT_DIR = PROJECT_DIR / "input" / "02_data_senario1_no_velocity" / "buildings"

# Choose building source:
#   "openbuildingmap"
#   "globalbuildingatlas"
#   "both"
BUILDING_DATA_SOURCE = "globalbuildingatlas"

# Data outputs go here.
OUT_DIR = PROJECT_DIR / "output" / "02_senario1_no_velocity" / "02_population_density"

# Figure outputs go here.
FIG_DIR = PROJECT_DIR / "figures" / "02_senario1_no_velocity" / "02_population_density"

OUT_HEIGHT_FIG = FIG_DIR / "buildings_by_height.png"

OUT_POLYGON_DENSITY_FIG = FIG_DIR / "building_density_polygon_map.png"
OUT_GAUSSIAN_POINTS_DENSITY_FIG = FIG_DIR / "building_density_gaussian_points_map.png"
OUT_COMBINED_DENSITY_FIG = FIG_DIR / "building_density_combined_map.png"

OUT_POLYGON_DENSITY_XYZ = OUT_DIR / "building_density_polygon_grid.xyz"
OUT_GAUSSIAN_POINTS_DENSITY_XYZ = OUT_DIR / "building_density_gaussian_points_grid.xyz"
OUT_COMBINED_DENSITY_XYZ = OUT_DIR / "building_density_combined_grid.xyz"

OUT_SUMMARY = OUT_DIR / "building_density_summary.csv"

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


# ============================================================
# LOW-MEMORY OPTIONS
# ============================================================

LOW_MEMORY_MODE = False

# Read large centroid/vertices XYZ in chunks.
XYZ_CHUNKSIZE = 200_000 if LOW_MEMORY_MODE else None

# In low-memory mode, do not keep all points for plotting.
KEEP_POINT_DF_FOR_PLOTTING = False if LOW_MEMORY_MODE else True

# If True, skip exact duplicate xy removal in low-memory point mode.
# This avoids storing a huge set of coordinates in RAM.
LOW_MEMORY_SKIP_POINT_DEDUP = True

# Limit expensive plotting. None means plot all.
MAX_BUILDING_POLYGONS_TO_PLOT = 6000 if LOW_MEMORY_MODE else None
MAX_POINT_OVERLAY_TO_PLOT = 20000 if LOW_MEMORY_MODE else None

# In low-memory mode, height-colored polygon plotting can be slow.
# If False, buildings are plotted with constant fill instead of per-height colors.
PLOT_HEIGHT_COLOR_IN_LOW_MEMORY = False

# Use bbox read when loading vector files.
# This can reduce memory if the file contains data outside the AOI.
READ_VECTOR_WITH_BBOX = True

# Safer PyGMT plotting for large GBA/OBM polygon datasets.
# The error "free(): invalid next size" usually comes from GMT C-side memory
# when plotting many polygons or passing a large in-memory xarray grid.
SAFE_PYGMT_MODE = True

# In safe mode, write the density grid to a temporary NetCDF file first,
# then let GMT read the file instead of passing the in-memory xarray object.
SAFE_PYGMT_GRID_FILE_MODE = True

# In low-memory/safe mode, density maps show the density raster + AOI boundary only.
# This avoids many repeated fig.plot calls for building outlines/high-rise polygons.
DISABLE_DENSITY_POLYGON_OVERLAYS = True

# Keep point overlays off in low-memory mode unless explicitly needed.
DISABLE_DENSITY_POINT_OVERLAY = True

# Density plot backend.
# Options:
#   "pygmt"      -> original PyGMT grdimage style
#   "matplotlib" -> safer density raster plotting, avoids PyGMT grdimage
#
# Important:
#   GBA and both can crash PyGMT grdimage with:
#       free(): invalid next size
#   Therefore AUTO_FORCE_MATPLOTLIB_FOR_GBA_OR_BOTH keeps them safe.
DENSITY_PLOT_BACKEND = "pygmt"

# If True:
#   BUILDING_DATA_SOURCE = "globalbuildingatlas" -> force Matplotlib for density maps
#   BUILDING_DATA_SOURCE = "both"                -> force Matplotlib for density maps
#   BUILDING_DATA_SOURCE = "openbuildingmap"     -> use DENSITY_PLOT_BACKEND
AUTO_FORCE_MATPLOTLIB_FOR_GBA_OR_BOTH = True

# Matplotlib density plotting. Set to >1 only if the grid is extremely large.
DENSITY_MPL_DOWNSAMPLE = 1
DENSITY_MPL_INTERPOLATION = "nearest"


# ============================================================
# SWITCHES
# ============================================================

CREATE_POLYGON_DENSITY = True
CREATE_GAUSSIAN_POINT_DENSITY = True
CREATE_COMBINED_DENSITY = True

PLOT_HEIGHT_MAP = True
PLOT_POLYGON_DENSITY_MAP = True
PLOT_GAUSSIAN_POINT_DENSITY_MAP = True
PLOT_COMBINED_DENSITY_MAP = True

# If True, also plot height-colored building polygons on density map.
# Usually False because it hides density and can be slow.
PLOT_HEIGHT_POLYGONS_ON_DENSITY = False


# ============================================================
# DENSITY SETTINGS
# ============================================================

DENSITY_GRID_SPACING_DEG = 0.00010

# Polygon-footprint density smoothing.
POLYGON_GAUSSIAN_SIGMA_CELLS = 2.0
POLYGON_DENSITY_CUTOFF = 0.02
RASTERIZE_ALL_TOUCHED = True

# Centroid+vertices Gaussian point density smoothing.
POINT_GAUSSIAN_SIGMA_CELLS = 2.0
POINT_DENSITY_CUTOFF = 0.02

# Combined density.
# Options:
#   "weighted_mean"
#   "max"
#   "sum"
COMBINED_DENSITY_MODE = "weighted_mean"

POLYGON_DENSITY_WEIGHT = 0.40
POINT_DENSITY_WEIGHT = 0.60


# ============================================================
# BUILDING HEIGHT / HIGH-RISE SETTINGS
# ============================================================

HEIGHT_COLUMN = "height_m"

HEIGHT_COLUMN_CANDIDATES = [
    "height_m",
    "height",
    "Height",
    "HEIGHT",
    "building_height",
    "building:height",
    "mean_height",
    "median_height",
    "max_height",
]

HIGHRISE_HEIGHT_M = None
HIGHRISE_HEIGHT_PERCENTILE = 90

HIGHRISE_FILL = "green@15"
HIGHRISE_PEN = "0.45p,green"

BUILDING_FILL = "lightred@65"
BUILDING_PEN = "0.20p,black@35"
BUILDING_OUTLINE_PEN = "0.15p,black@45"
POLYGON_PEN = "1.4p,purple"

POINT_INPUT_STYLE = "c0.004c"

CLEANUP_TEMP_FILES = True


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_source_name():
    source = BUILDING_DATA_SOURCE.lower().strip()

    if source not in {"openbuildingmap", "globalbuildingatlas", "both"}:
        raise ValueError(
            "Invalid BUILDING_DATA_SOURCE. "
            "Use 'openbuildingmap', 'globalbuildingatlas', or 'both'."
        )

    return source


def get_effective_density_plot_backend():
    """
    Return the actual backend used for density maps.

    This keeps LOW_MEMORY_MODE independent from plotting backend choice.
    GBA and both are forced to Matplotlib when requested because PyGMT
    grdimage can crash on some large GBA density grids.
    """
    backend = DENSITY_PLOT_BACKEND.lower().strip()

    if backend not in {"pygmt", "matplotlib"}:
        raise ValueError(
            "Invalid DENSITY_PLOT_BACKEND. Use 'pygmt' or 'matplotlib'."
        )

    if AUTO_FORCE_MATPLOTLIB_FOR_GBA_OR_BOTH:
        if normalize_source_name() in {"globalbuildingatlas", "both"}:
            return "matplotlib"

    return backend


def source_label():
    source = normalize_source_name()

    if source == "openbuildingmap":
        return "OpenBuildingMap"

    if source == "globalbuildingatlas":
        return "GlobalBuildingAtlas"

    return "OpenBuildingMap + GlobalBuildingAtlas"


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def snap_region_to_spacing(region, spacing):
    xmin, xmax, ymin, ymax = region

    x0 = np.floor(xmin / spacing) * spacing
    x1 = np.ceil(xmax / spacing) * spacing
    y0 = np.floor(ymin / spacing) * spacing
    y1 = np.ceil(ymax / spacing) * spacing

    return [
        float(np.round(x0, 10)),
        float(np.round(x1, 10)),
        float(np.round(y0, 10)),
        float(np.round(y1, 10)),
    ]


def start_map(region, title):
    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd:mmF",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="9p",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            f'WSne+t"{title}"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    return fig


def plot_aoi_boundary(fig, pen=POLYGON_PEN):
    poly_df = polygon_to_dataframe()

    fig.plot(
        x=poly_df["x"],
        y=poly_df["y"],
        pen=pen,
        fill=None,
        label="Hoa Lac boundary",
    )


def save_fig(fig, out_png):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved: {out_png}")


def safe_polygons(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    return []


def make_polygon_mask_from_centers(lon_centers, lat_centers):
    poly_path = MplPath(HOALAC_POLYGON)

    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    xy = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    inside = poly_path.contains_points(xy, radius=1e-12)

    return inside.reshape(len(lat_centers), len(lon_centers))


def decimate_gdf_for_plot(gdf, max_count):
    if gdf is None or gdf.empty:
        return gdf

    if max_count is None:
        return gdf

    if len(gdf) <= max_count:
        return gdf

    step = int(np.ceil(len(gdf) / max_count))
    out = gdf.iloc[::step].copy()

    print(
        f"[LOW-MEMORY] Plot only {len(out):,}/{len(gdf):,} building polygons "
        f"(step={step})"
    )

    return out


def decimate_points_for_plot(df, max_count):
    if df is None or df.empty:
        return df

    if max_count is None:
        return df

    if len(df) <= max_count:
        return df

    step = int(np.ceil(len(df) / max_count))
    out = df.iloc[::step].copy()

    print(
        f"[LOW-MEMORY] Plot only {len(out):,}/{len(df):,} input points "
        f"(step={step})"
    )

    return out


# ============================================================
# INPUT FILE SELECTION
# ============================================================

def is_obm_name(path):
    p = str(path).lower()
    name = path.name.lower()

    return (
        "openbuildingmap" in p
        or "openbuildingmaps" in p
        or "obm" in name
    )


def is_gba_name(path):
    p = str(path).lower()
    name = path.name.lower()

    return (
        "globalbuildingatlas" in p
        or "gba" in name
        or "lod1" in name
    )


def is_metadata_name(path):
    name = path.name.lower()

    bad_keys = [
        "selected_gba_5deg_tiles",
        "tile",
        "tiles",
        "inventory",
        "metadata",
    ]

    # Allow a file under a metadata folder only if its filename clearly says building.
    if any(k in name for k in bad_keys) and "building" not in name:
        return True

    return False


def source_matches(path):
    source = normalize_source_name()

    if source == "openbuildingmap":
        return is_obm_name(path)

    if source == "globalbuildingatlas":
        return is_gba_name(path)

    return is_obm_name(path) or is_gba_name(path)


def unique_paths(paths):
    out = []
    seen = set()

    for path in paths:
        if not path.is_file():
            continue

        rp = path.resolve()

        if rp in seen:
            continue

        seen.add(rp)
        out.append(path)

    return out


def collect_building_polygon_files():
    """
    Find building polygon files from BUILDING_INPUT_DIR.

    For GBA, avoid selected_gba_5deg_tiles.gpkg and other metadata.
    """
    patterns = [
        "*.gpkg",
        "*.geojson",
        "*.shp",
        "**/*.gpkg",
        "**/*.geojson",
        "**/*.shp",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(BUILDING_INPUT_DIR.glob(pattern))

    candidates = unique_paths(candidates)

    selected = []

    for f in candidates:
        if is_metadata_name(f):
            continue

        if not source_matches(f):
            continue

        name = f.name.lower()

        # Keep only likely building polygon files.
        if "building" in name or "buildings" in name or "obm" in name:
            selected.append(f)

    selected = unique_paths(selected)

    return selected


def collect_gaussian_point_files():
    """
    Find centroid/vertices XYZ files from BUILDING_INPUT_DIR according to source.
    """
    patterns = [
        "*centroid*.xyz",
        "*vertices*.xyz",
        "*vertex*.xyz",
        "**/*centroid*.xyz",
        "**/*vertices*.xyz",
        "**/*vertex*.xyz",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(BUILDING_INPUT_DIR.glob(pattern))

    candidates = unique_paths(candidates)

    selected = []

    for f in candidates:
        if not source_matches(f):
            continue

        name = f.name.lower()

        if "centroid" in name or "vertices" in name or "vertex" in name:
            selected.append(f)

    selected = unique_paths(selected)

    return selected


# ============================================================
# LOAD BUILDING POLYGONS
# ============================================================

def find_height_column(gdf):
    for col in HEIGHT_COLUMN_CANDIDATES:
        if col in gdf.columns:
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().sum() > 0:
                return col

    return None


def read_vector_file_low_memory(path, bbox):
    """
    Read vector file.

    bbox is used when READ_VECTOR_WITH_BBOX = True.
    """
    if READ_VECTOR_WITH_BBOX:
        try:
            return gpd.read_file(path, bbox=bbox)
        except Exception as exc:
            print(f"[WARNING] bbox read failed for {path.name}, read full file: {exc}")

    return gpd.read_file(path)


def load_buildings():
    files = collect_building_polygon_files()

    print("")
    print("========== LOAD BUILDINGS ==========")
    print(f"Building source: {source_label()}")
    print(f"Input dir:        {BUILDING_INPUT_DIR}")
    print(f"Polygon files:    {len(files)}")

    for f in files:
        print(f"  - {f.name}")

    if not files:
        raise FileNotFoundError(
            "No building polygon file found in copied input folder:\n"
            f"  {BUILDING_INPUT_DIR}\n"
            f"Building data source: {BUILDING_DATA_SOURCE}"
        )

    aoi = get_aoi_gdf()
    aoi_geom = aoi.geometry.iloc[0]
    bbox = tuple(aoi.total_bounds)

    parts = []

    for path in files:
        try:
            gdf = read_vector_file_low_memory(path, bbox=bbox)

            if gdf.empty:
                print(f"[WARNING] Empty building file: {path.name}")
                continue

            if gdf.crs is None:
                print(f"[WARNING] {path.name} has no CRS. Assuming EPSG:4326.")
                gdf = gdf.set_crs("EPSG:4326")

            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf[gdf.geometry.notna()].copy()
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

            if gdf.empty:
                print(f"[WARNING] No polygon geometry in: {path.name}")
                continue

            n_before = len(gdf)

            try:
                gdf = gpd.clip(gdf, aoi).copy()
            except Exception as exc:
                print(f"[WARNING] gpd.clip failed for {path.name}, using intersects only: {exc}")
                gdf = gdf[gdf.intersects(aoi_geom)].copy()

            gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

            if gdf.empty:
                print(f"[WARNING] No buildings inside AOI after clip: {path.name}")
                continue

            height_col = find_height_column(gdf)

            if height_col is not None:
                gdf[HEIGHT_COLUMN] = pd.to_numeric(gdf[height_col], errors="coerce")
            elif HEIGHT_COLUMN not in gdf.columns:
                gdf[HEIGHT_COLUMN] = np.nan

            if is_obm_name(path):
                gdf["source_dataset"] = "OpenBuildingMap"
            elif is_gba_name(path):
                gdf["source_dataset"] = "GlobalBuildingAtlas"
            else:
                gdf["source_dataset"] = "unknown"

            gdf["source_file"] = path.name

            # Keep only useful columns to reduce memory.
            keep_cols = ["geometry", HEIGHT_COLUMN, "source_dataset", "source_file"]
            gdf = gdf[keep_cols].copy()

            print(f"  {path.name}: before clip={n_before:,}, after clip={len(gdf):,}")

            parts.append(gdf)

        except Exception as exc:
            print(f"[WARNING] Failed to read building file {path}: {exc}")

    if not parts:
        raise ValueError("No valid building polygons loaded.")

    buildings = pd.concat(parts, ignore_index=True)
    buildings = gpd.GeoDataFrame(buildings, geometry="geometry", crs="EPSG:4326")

    # Remove duplicated geometries when source = both or files overlap.
    n_before_unique = len(buildings)
    buildings["_wkb"] = buildings.geometry.to_wkb()
    buildings = buildings.drop_duplicates(subset=["_wkb"]).drop(columns=["_wkb"]).copy()

    print(f"Buildings before unique: {n_before_unique:,}")
    print(f"Buildings after unique:  {len(buildings):,}")

    if HEIGHT_COLUMN in buildings.columns:
        valid_h = buildings[HEIGHT_COLUMN].notna().sum()
        print(f"Height column:           {HEIGHT_COLUMN}")
        print(f"Height valid count:      {valid_h:,}")
        if valid_h > 0:
            print(f"Height max:              {buildings[HEIGHT_COLUMN].max()}")

    return buildings, files


# ============================================================
# POINT XYZ LOADING / STREAMING
# ============================================================

def read_xyz_auto(path: Path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")
    ncol = df.shape[1]

    if ncol < 2:
        raise ValueError(f"File must have at least lon lat columns: {path}")

    if ncol == 2:
        df = df.iloc[:, :2]
        df.columns = ["x", "y"]
        df["z"] = 0.0
        df["value"] = 0.0

    elif ncol == 3:
        df = df.iloc[:, :3]
        df.columns = ["x", "y", "z"]
        df["value"] = 0.0

    elif ncol == 4:
        df = df.iloc[:, :4]
        df.columns = ["x", "y", "z", "value"]

    else:
        df = df.iloc[:, :ncol]
        names = ["x", "y", "z", "value"]
        names += [f"extra_{i}" for i in range(ncol - 4)]
        df.columns = names

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y"]).copy()

    if "z" not in df.columns:
        df["z"] = 0.0

    if "value" not in df.columns:
        df["value"] = 0.0

    return df


def read_xyz_chunks(path: Path, chunksize: int):
    """
    Yield x/y/z/value DataFrames from an XYZ file.

    This avoids reading very large centroid/vertices files all at once.
    """
    if chunksize is None:
        yield read_xyz_auto(path)
        return

    reader = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk = chunk.dropna(axis=1, how="all")
        ncol = chunk.shape[1]

        if ncol < 2:
            continue

        if ncol == 2:
            df = chunk.iloc[:, :2].copy()
            df.columns = ["x", "y"]
            df["z"] = 0.0
            df["value"] = 0.0

        elif ncol == 3:
            df = chunk.iloc[:, :3].copy()
            df.columns = ["x", "y", "z"]
            df["value"] = 0.0

        else:
            df = chunk.iloc[:, :4].copy()
            df.columns = ["x", "y", "z", "value"]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["x", "y"]).copy()

        yield df


def infer_xy_are_lonlat(df):
    if df.empty:
        return False

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    return (
        np.nanmin(x) >= -180
        and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90
        and np.nanmax(y) <= 90
    )


def mask_points_inside_polygon(df):
    if df.empty:
        return df.copy()

    poly_path = MplPath(HOALAC_POLYGON)
    xy = df[["x", "y"]].to_numpy(dtype=float)

    inside = poly_path.contains_points(xy, radius=1e-12)

    return df.loc[inside].copy()


def load_combined_gaussian_points_for_plot():
    """
    Load points into one DataFrame.

    This is used only when KEEP_POINT_DF_FOR_PLOTTING = True.
    """
    files = collect_gaussian_point_files()

    if not files:
        print("[WARNING] No centroid/vertices XYZ files found. Skip Gaussian point density.")
        return None, [], "no files"

    print("")
    print("========== LOAD GAUSSIAN POINT INPUTS ==========")
    print(f"Building source: {source_label()}")
    print(f"Files used:      {len(files)}")

    parts = []

    for f in files:
        try:
            df = read_xyz_auto(f)

            if not infer_xy_are_lonlat(df):
                print(f"[WARNING] Skip non-lonlat file: {f}")
                continue

            df = df[["x", "y", "z", "value"]].copy()
            df["source_file"] = f.name

            name_lower = f.name.lower()

            if "centroid" in name_lower:
                df["source_type"] = "centroid"
            elif "vertices" in name_lower or "vertex" in name_lower:
                df["source_type"] = "vertices"
            else:
                df["source_type"] = "unknown"

            parts.append(df)

            print(f"  - {f.name}: {len(df):,} points")

        except Exception as exc:
            print(f"[WARNING] Failed to read {f}: {exc}")

    if not parts:
        print("[WARNING] No valid centroid/vertices points loaded. Skip Gaussian point density.")
        return None, files, "loaded but invalid"

    all_df = pd.concat(parts, ignore_index=True)

    all_before = len(all_df)

    all_df = all_df.drop_duplicates(subset=["x", "y"]).copy()
    all_inside = mask_points_inside_polygon(all_df)

    print(f"Combined points before unique: {all_before:,}")
    print(f"Combined unique xy points:     {len(all_df):,}")
    print(f"Points inside polygon:         {len(all_inside):,}")
    print(f"Points outside polygon masked: {len(all_df) - len(all_inside):,}")

    if all_inside.empty:
        print("[WARNING] No centroid/vertices points inside polygon. Skip Gaussian point density.")
        return None, files, "no inside points"

    return all_inside, files, "loaded all points"


# ============================================================
# PLOT BUILDING POLYGONS
# ============================================================

def plot_polygons_constant(fig, gdf, fill=None, pen=BUILDING_PEN, label=None):
    if gdf is None or gdf.empty:
        return

    plot_gdf = decimate_gdf_for_plot(gdf, MAX_BUILDING_POLYGONS_TO_PLOT)

    first = True

    for geom in plot_gdf.geometry:
        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy

            kwargs = {
                "x": list(x),
                "y": list(y),
                "fill": fill,
                "pen": pen,
            }

            if label is not None and first:
                kwargs["label"] = label
                first = False

            fig.plot(**kwargs)


def plot_buildings_by_height(fig, buildings, add_colorbar=True):
    if buildings is None or buildings.empty:
        return

    if LOW_MEMORY_MODE and not PLOT_HEIGHT_COLOR_IN_LOW_MEMORY:
        plot_polygons_constant(
            fig,
            buildings,
            fill=BUILDING_FILL,
            pen=BUILDING_PEN,
            label="Building polygon",
        )
        return

    if HEIGHT_COLUMN not in buildings.columns:
        plot_polygons_constant(
            fig,
            buildings,
            fill=BUILDING_FILL,
            pen=BUILDING_PEN,
            label="Building polygon",
        )
        return

    valid = buildings.dropna(subset=[HEIGHT_COLUMN]).copy()

    if valid.empty:
        plot_polygons_constant(
            fig,
            buildings,
            fill=BUILDING_FILL,
            pen=BUILDING_PEN,
            label="Building polygon",
        )
        return

    valid = decimate_gdf_for_plot(valid, MAX_BUILDING_POLYGONS_TO_PLOT)

    zmin = float(valid[HEIGHT_COLUMN].quantile(0.02))
    zmax = float(valid[HEIGHT_COLUMN].quantile(0.98))

    if np.isclose(zmin, zmax):
        zmin = float(valid[HEIGHT_COLUMN].min())
        zmax = float(valid[HEIGHT_COLUMN].max())

    if np.isclose(zmin, zmax):
        zmin, zmax = 0.0, 10.0

    step = (zmax - zmin) / 100.0

    pygmt.makecpt(
        cmap="turbo",
        series=[zmin, zmax, step],
        continuous=True,
    )

    for _, row in valid.iterrows():
        geom = row.geometry
        value = row[HEIGHT_COLUMN]

        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy

            fig.plot(
                x=list(x),
                y=list(y),
                fill=value,
                cmap=True,
                pen=BUILDING_PEN,
            )

    if add_colorbar:
        fig.colorbar(
            frame=f'af+l"Building height (m)"',
            position="JBC+w10c/0.35c+h+o0c/0.8c",
        )


def select_highrise_buildings(buildings):
    if buildings is None or buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), "none"

    gdf = buildings.copy()

    if HEIGHT_COLUMN in gdf.columns:
        vals = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")

        if vals.notna().sum() > 0 and vals.max() > 0:
            gdf["_height_for_plot"] = vals.fillna(0.0)

            if HIGHRISE_HEIGHT_M is not None:
                threshold = float(HIGHRISE_HEIGHT_M)
            else:
                positive = gdf.loc[gdf["_height_for_plot"] > 0, "_height_for_plot"]
                threshold = float(np.nanpercentile(positive, HIGHRISE_HEIGHT_PERCENTILE))

            highrise = gdf[gdf["_height_for_plot"] >= threshold].copy()
            method = f"{HEIGHT_COLUMN} >= {threshold:.2f} m"

            print("")
            print("========== HIGH-RISE SELECTION ==========")
            print(f"Method:          {method}")
            print(f"All buildings:   {len(gdf):,}")
            print(f"High-rise count: {len(highrise):,}")

            return highrise, method

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

    return highrise, method


# ============================================================
# DENSITY GRID CREATION
# ============================================================

def make_grid_geometry(region):
    xmin, xmax, ymin, ymax = region
    spacing = DENSITY_GRID_SPACING_DEG

    width = int(np.ceil((xmax - xmin) / spacing))
    height = int(np.ceil((ymax - ymin) / spacing))

    xmax2 = xmin + width * spacing
    ymax2 = ymin + height * spacing

    transform = from_origin(xmin, ymax2, spacing, spacing)

    lon_centers = xmin + (np.arange(width, dtype=np.float64) + 0.5) * spacing
    lat_desc = ymax2 - (np.arange(height, dtype=np.float64) + 0.5) * spacing
    lat_centers = lat_desc[::-1]

    exact_region = [xmin, xmax2, ymin, ymax2]

    return width, height, transform, lon_centers, lat_centers, exact_region


def create_polygon_density_grid(buildings, region):
    width, height, transform, lon_centers, lat_centers, exact_region = make_grid_geometry(region)

    print("")
    print("========== POLYGON DENSITY GRID ==========")
    print(f"Source:         {source_label()}")
    print(f"Spacing degree: {DENSITY_GRID_SPACING_DEG}")
    print(f"Width x height: {width} x {height}")
    print(f"Region used:    {exact_region}")
    print(f"Gaussian sigma: {POLYGON_GAUSSIAN_SIGMA_CELLS}")
    print(f"Density cutoff: {POLYGON_DENSITY_CUTOFF}")

    valid_geoms = [
        geom for geom in buildings.geometry
        if geom is not None and not geom.is_empty
    ]

    if not valid_geoms:
        raise ValueError("No valid building polygon shapes to rasterize.")

    # Generator avoids keeping a second large list of mapping dictionaries.
    shapes = ((mapping(geom), 1.0) for geom in valid_geoms)

    footprint = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=RASTERIZE_ALL_TOUCHED,
    )

    aoi_shape = [(mapping(get_hoalac_polygon()), 1)]

    aoi_mask = rasterize(
        aoi_shape,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    footprint[~aoi_mask] = 0.0

    if POLYGON_GAUSSIAN_SIGMA_CELLS > 0:
        density = gaussian_filter(
            footprint,
            sigma=POLYGON_GAUSSIAN_SIGMA_CELLS,
            mode="constant",
            cval=0.0,
        ).astype("float32")
    else:
        density = footprint.astype("float32")

    density[~aoi_mask] = 0.0

    max_val = float(np.nanmax(density))

    if max_val > 0:
        density = density / max_val

    density[density < POLYGON_DENSITY_CUTOFF] = 0.0
    density[~aoi_mask] = 0.0

    density_asc = density[::-1, :].astype("float32")

    density_grid = xr.DataArray(
        density_asc,
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        },
        dims=("lat", "lon"),
        name="polygon_building_density",
    )

    print(f"Footprint nonzero cells: {int(np.sum(footprint > 0)):,}")
    print(f"Density nonzero cells:   {int(np.sum(density_asc > 0)):,}")
    print(f"Max density:             {float(np.nanmax(density_asc)):.4f}")
    print(f"Mean density:            {float(np.nanmean(density_asc)):.6f}")

    return density_grid, exact_region


def add_points_to_count_grid(counts, df, exact_region):
    xmin, xmax, ymin, ymax = exact_region
    height, width = counts.shape
    spacing = DENSITY_GRID_SPACING_DEG

    if df.empty:
        return 0

    # Keep only lon/lat-looking rows.
    df = df[
        (df["x"] >= -180) & (df["x"] <= 180) &
        (df["y"] >= -90) & (df["y"] <= 90)
    ].copy()

    if df.empty:
        return 0

    df = mask_points_inside_polygon(df)

    if df.empty:
        return 0

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    ix = np.floor((x - xmin) / spacing).astype(np.int64)
    iy = np.floor((y - ymin) / spacing).astype(np.int64)

    valid = (
        (ix >= 0) & (ix < width) &
        (iy >= 0) & (iy < height)
    )

    ix = ix[valid]
    iy = iy[valid]

    if len(ix) == 0:
        return 0

    np.add.at(counts, (iy, ix), 1.0)

    return len(ix)


def create_gaussian_point_density_grid_streaming(files, region):
    """
    Create point-density grid from centroid + vertices files without
    loading all points into one DataFrame.
    """
    width, height, transform, lon_centers, lat_centers, exact_region = make_grid_geometry(region)

    print("")
    print("========== GAUSSIAN POINT DENSITY GRID ==========")
    print(f"Source:               {source_label()}")
    print(f"Spacing degree:       {DENSITY_GRID_SPACING_DEG}")
    print(f"Width x height:       {width} x {height}")
    print(f"Region used:          {exact_region}")
    print(f"Gaussian sigma cells: {POINT_GAUSSIAN_SIGMA_CELLS}")
    print(f"Density cutoff:       {POINT_DENSITY_CUTOFF}")
    print(f"Point files:          {len(files)}")
    print(f"Low-memory chunksize: {XYZ_CHUNKSIZE}")

    counts = np.zeros((height, width), dtype="float32")

    total_used = 0
    total_read = 0

    # Optional exact dedup is intentionally not used in low memory mode.
    seen_xy = set() if not LOW_MEMORY_SKIP_POINT_DEDUP else None

    for path in files:
        file_used = 0
        file_read = 0

        try:
            for chunk in read_xyz_chunks(path, XYZ_CHUNKSIZE):
                file_read += len(chunk)

                if chunk.empty:
                    continue

                if seen_xy is not None:
                    # This is memory-expensive; use only when LOW_MEMORY_SKIP_POINT_DEDUP = False.
                    before = len(chunk)
                    keep = []
                    for x, y in zip(chunk["x"].to_numpy(), chunk["y"].to_numpy()):
                        key = (round(float(x), 8), round(float(y), 8))
                        if key in seen_xy:
                            keep.append(False)
                        else:
                            seen_xy.add(key)
                            keep.append(True)

                    chunk = chunk.loc[keep].copy()
                    if before != len(chunk):
                        pass

                used = add_points_to_count_grid(counts, chunk, exact_region)
                file_used += used

        except Exception as exc:
            print(f"[WARNING] Failed to read point file {path}: {exc}")
            continue

        total_read += file_read
        total_used += file_used

        print(f"  - {path.name}: read={file_read:,}, used inside grid={file_used:,}")

    if total_used == 0:
        print("[WARNING] No valid centroid/vertices points inside polygon. Skip Gaussian point density.")
        return None, exact_region, total_read, total_used

    if POINT_GAUSSIAN_SIGMA_CELLS > 0:
        density = gaussian_filter(
            counts,
            sigma=POINT_GAUSSIAN_SIGMA_CELLS,
            mode="constant",
            cval=0.0,
        ).astype("float32")
    else:
        density = counts.astype("float32")

    mask = make_polygon_mask_from_centers(lon_centers, lat_centers)

    if mask.shape != density.shape:
        raise ValueError(
            f"Mask shape mismatch: mask={mask.shape}, density={density.shape}"
        )

    density[~mask] = 0.0

    max_val = float(np.nanmax(density))

    if max_val > 0:
        density = density / max_val

    density[density < POINT_DENSITY_CUTOFF] = 0.0
    density[~mask] = 0.0

    density_grid = xr.DataArray(
        density.astype("float32"),
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        },
        dims=("lat", "lon"),
        name="gaussian_point_building_density",
    )

    print(f"Input points read:        {total_read:,}")
    print(f"Input points used:        {total_used:,}")
    print(f"Max raw point count/cell: {float(np.nanmax(counts)):.3f}")
    print(f"Density nonzero cells:    {int(np.sum(density > 0)):,}")
    print(f"Max density:              {float(np.nanmax(density)):.4f}")
    print(f"Mean density:             {float(np.nanmean(density)):.6f}")

    return density_grid, exact_region, total_read, total_used


def create_gaussian_point_density_grid_from_df(points_df, region):
    width, height, transform, lon_centers, lat_centers, exact_region = make_grid_geometry(region)

    xmin, xmax, ymin, ymax = exact_region

    print("")
    print("========== GAUSSIAN POINT DENSITY GRID ==========")
    print(f"Source:               {source_label()}")
    print(f"Spacing degree:       {DENSITY_GRID_SPACING_DEG}")
    print(f"Width x height:       {width} x {height}")
    print(f"Region used:          {exact_region}")
    print(f"Gaussian sigma cells: {POINT_GAUSSIAN_SIGMA_CELLS}")
    print(f"Density cutoff:       {POINT_DENSITY_CUTOFF}")
    print(f"Input points:         {len(points_df):,}")

    lon_edges = np.linspace(xmin, xmax, width + 1)
    lat_edges = np.linspace(ymin, ymax, height + 1)

    counts_xy, _, _ = np.histogram2d(
        points_df["x"].to_numpy(),
        points_df["y"].to_numpy(),
        bins=[lon_edges, lat_edges],
    )

    counts = counts_xy.T.astype("float32")

    if counts.shape != (height, width):
        raise ValueError(
            f"Unexpected point-density count shape: {counts.shape}, "
            f"expected {(height, width)}"
        )

    if POINT_GAUSSIAN_SIGMA_CELLS > 0:
        density = gaussian_filter(
            counts,
            sigma=POINT_GAUSSIAN_SIGMA_CELLS,
            mode="constant",
            cval=0.0,
        ).astype("float32")
    else:
        density = counts.astype("float32")

    mask = make_polygon_mask_from_centers(lon_centers, lat_centers)

    if mask.shape != density.shape:
        raise ValueError(
            f"Mask shape mismatch: mask={mask.shape}, density={density.shape}"
        )

    density[~mask] = 0.0

    max_val = float(np.nanmax(density))

    if max_val > 0:
        density = density / max_val

    density[density < POINT_DENSITY_CUTOFF] = 0.0
    density[~mask] = 0.0

    density_grid = xr.DataArray(
        density.astype("float32"),
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        },
        dims=("lat", "lon"),
        name="gaussian_point_building_density",
    )

    print(f"Max raw point count/cell: {float(np.nanmax(counts)):.3f}")
    print(f"Density nonzero cells:    {int(np.sum(density > 0)):,}")
    print(f"Max density:              {float(np.nanmax(density)):.4f}")
    print(f"Mean density:             {float(np.nanmean(density)):.6f}")

    return density_grid, exact_region


def combine_density_grids(polygon_grid, point_grid):
    if polygon_grid is None and point_grid is None:
        return None

    if polygon_grid is None:
        return point_grid.copy()

    if point_grid is None:
        return polygon_grid.copy()

    p = polygon_grid.to_numpy().astype("float32")
    q = point_grid.to_numpy().astype("float32")

    if p.shape != q.shape:
        raise ValueError(
            f"Density grid shapes differ: polygon={p.shape}, point={q.shape}. "
            "Use same region and spacing."
        )

    pmax = float(np.nanmax(p))
    qmax = float(np.nanmax(q))

    if pmax > 0:
        p = p / pmax

    if qmax > 0:
        q = q / qmax

    if COMBINED_DENSITY_MODE == "max":
        combined = np.maximum(p, q)

    elif COMBINED_DENSITY_MODE == "weighted_mean":
        combined = (
            POLYGON_DENSITY_WEIGHT * p
            + POINT_DENSITY_WEIGHT * q
        ).astype("float32")

        max_val = float(np.nanmax(combined))
        if max_val > 0:
            combined = combined / max_val

    elif COMBINED_DENSITY_MODE == "sum":
        combined = p + q

        max_val = float(np.nanmax(combined))
        if max_val > 0:
            combined = combined / max_val

    else:
        raise ValueError(f"Unknown COMBINED_DENSITY_MODE: {COMBINED_DENSITY_MODE}")

    lon_centers = polygon_grid["lon"].to_numpy()
    lat_centers = polygon_grid["lat"].to_numpy()
    mask = make_polygon_mask_from_centers(lon_centers, lat_centers)

    combined[~mask] = 0.0

    combined_grid = xr.DataArray(
        combined.astype("float32"),
        coords=polygon_grid.coords,
        dims=polygon_grid.dims,
        name="combined_building_density",
    )

    print("")
    print("========== COMBINED DENSITY GRID ==========")
    print(f"Source:                {source_label()}")
    print(f"Mode:                  {COMBINED_DENSITY_MODE}")
    print(f"Polygon weight:        {POLYGON_DENSITY_WEIGHT}")
    print(f"Point weight:          {POINT_DENSITY_WEIGHT}")
    print(f"Nonzero density cells: {int(np.sum(combined > 0)):,}")
    print(f"Max density:           {float(np.nanmax(combined)):.4f}")
    print(f"Mean density:          {float(np.nanmean(combined)):.6f}")

    return combined_grid


def save_density_grid_xyz(density_grid, out_file):
    """
    Save density grid as lon lat density.

    Low-memory: write row-by-row instead of building a large DataFrame.
    """
    if density_grid is None:
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    lon = density_grid["lon"].to_numpy()
    lat = density_grid["lat"].to_numpy()
    den = density_grid.to_numpy()

    with open(out_file, "w", encoding="utf-8") as f:
        for iy, y in enumerate(lat):
            row_den = den[iy, :]
            for ix, x in enumerate(lon):
                f.write(f"{x:.8f} {y:.8f} {row_den[ix]:.8f}\n")

    print(f"[OK] Saved density grid xyz: {out_file}")


# ============================================================
# PLOTS
# ============================================================

def plot_buildings_height_map(buildings, highrise, region, out_png):
    print("")
    print("========== PLOT BUILDINGS HEIGHT MAP ==========")

    fig = start_map(region, f"{source_label()} buildings by height")

    plot_buildings_by_height(fig, buildings, add_colorbar=True)

    if highrise is not None and not highrise.empty:
        plot_polygons_constant(
            fig,
            highrise,
            fill=HIGHRISE_FILL,
            pen=HIGHRISE_PEN,
            label="High-rise building",
        )

    plot_aoi_boundary(fig)

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    save_fig(fig, out_png)


def save_grid_for_safe_pygmt(density_grid, out_nc):
    """
    Save density grid to a simple NetCDF file for GMT/PyGMT plotting.

    This avoids passing a large in-memory xarray object directly to GMT.
    Coordinates:
        x = longitude
        y = latitude
        z = density
    """
    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    da = density_grid.astype("float32").copy(deep=True)

    # GMT is happiest with x/y dimension names.
    rename_map = {}
    if "lon" in da.dims:
        rename_map["lon"] = "x"
    if "lat" in da.dims:
        rename_map["lat"] = "y"
    if rename_map:
        da = da.rename(rename_map)

    da.name = "z"
    da.to_netcdf(out_nc)
    return out_nc



def plot_density_map_matplotlib(
    density_grid,
    buildings,
    highrise,
    region,
    out_png,
    title,
    point_df=None,
    overlay_building_outlines=False,
    overlay_highrise=False,
    overlay_points=False,
):
    """
    Safest density-map plotter.

    Uses Matplotlib imshow instead of PyGMT grdimage. This avoids GMT C-side
    memory errors such as:
        free(): invalid next size (normal)

    In LOW_MEMORY_MODE, overlays should normally stay disabled.
    """
    if density_grid is None:
        return

    print("[INFO] Matplotlib density backend is active. PyGMT grdimage is not used.")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    lon = density_grid["lon"].to_numpy()
    lat = density_grid["lat"].to_numpy()
    den = density_grid.to_numpy().astype(np.float32, copy=False)

    ds = max(1, int(DENSITY_MPL_DOWNSAMPLE))
    if ds > 1:
        den = den[::ds, ::ds]
        lon = lon[::ds]
        lat = lat[::ds]
        print(f"[INFO] Matplotlib density grid downsampled by {ds}.")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    im = ax.imshow(
        den,
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        origin="lower",
        cmap="hot_r",
        vmin=0.0,
        vmax=1.0,
        interpolation=DENSITY_MPL_INTERPOLATION,
        aspect="auto",
    )

    # Optional overlays. These are disabled by default in low-memory mode.
    if overlay_building_outlines and buildings is not None and not buildings.empty:
        plot_gdf = decimate_buildings_for_plot(buildings, MAX_BUILDING_POLYGONS_TO_PLOT)
        for geom in plot_gdf.geometry:
            for poly in safe_polygons(geom):
                x, y = poly.exterior.xy
                ax.plot(x, y, linewidth=0.25, color="black", alpha=0.35)

    if overlay_highrise and highrise is not None and not highrise.empty:
        plot_gdf = decimate_buildings_for_plot(highrise, MAX_BUILDING_POLYGONS_TO_PLOT)
        for geom in plot_gdf.geometry:
            for poly in safe_polygons(geom):
                x, y = poly.exterior.xy
                ax.plot(x, y, linewidth=0.6, color="green", alpha=0.8)

    if overlay_points and point_df is not None and not point_df.empty:
        plot_df = decimate_points_for_plot(point_df, MAX_POINT_OVERLAY_TO_PLOT)
        ax.scatter(plot_df["x"], plot_df["y"], s=0.5, color="black", alpha=0.25)

    # AOI boundary.
    poly_df = polygon_to_dataframe()
    ax.plot(poly_df["x"], poly_df["y"], color="purple", linewidth=1.3, label="Hoa Lac boundary")

    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05)
    cbar.set_label("Building density, normalized")

    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved Matplotlib density figure: {out_png}")


def plot_density_map(
    density_grid,
    buildings,
    highrise,
    region,
    out_png,
    title,
    point_df=None,
    overlay_building_outlines=True,
    overlay_highrise=True,
    overlay_points=False,
):
    if density_grid is None:
        return

    print("")
    print(f"========== PLOT {title} ==========")

    # ------------------------------------------------------------------
    # Safe plotting decisions.
    # For large GBA files, repeated polygon overlay calls can crash GMT
    # with messages such as: free(): invalid next size.
    # ------------------------------------------------------------------
    do_overlay_building_outlines = overlay_building_outlines
    do_overlay_highrise = overlay_highrise
    do_overlay_points = overlay_points
    do_height_polygons = PLOT_HEIGHT_POLYGONS_ON_DENSITY

    if SAFE_PYGMT_MODE and DISABLE_DENSITY_POLYGON_OVERLAYS:
        do_overlay_building_outlines = False
        do_overlay_highrise = False
        do_height_polygons = False

    if SAFE_PYGMT_MODE and DISABLE_DENSITY_POINT_OVERLAY:
        do_overlay_points = False

    if SAFE_PYGMT_MODE:
        print("[INFO] Safe PyGMT mode is active for this density plot.")
        print(f"       building outlines overlay: {do_overlay_building_outlines}")
        print(f"       high-rise overlay:         {do_overlay_highrise}")
        print(f"       point overlay:             {do_overlay_points}")

    effective_backend = get_effective_density_plot_backend()

    print(f"[INFO] Requested density backend: {DENSITY_PLOT_BACKEND}")
    print(f"[INFO] Effective density backend: {effective_backend}")

    if effective_backend == "matplotlib":
        return plot_density_map_matplotlib(
            density_grid=density_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=out_png,
            title=title,
            point_df=point_df,
            overlay_building_outlines=do_overlay_building_outlines,
            overlay_highrise=do_overlay_highrise,
            overlay_points=do_overlay_points,
        )

    cpt_file = FIG_DIR / f"{out_png.stem}.cpt"

    pygmt.makecpt(
        cmap="hot",
        series=[0, 1, 0.05],
        reverse=True,
        continuous=True,
        output=str(cpt_file),
    )

    fig = start_map(region, title)

    grid_for_plot = density_grid
    tmp_grid_file = None

    if SAFE_PYGMT_MODE and SAFE_PYGMT_GRID_FILE_MODE:
        tmp_grid_file = FIG_DIR / f"_tmp_{out_png.stem}_density_grid.nc"
        grid_for_plot = str(save_grid_for_safe_pygmt(density_grid, tmp_grid_file))
        print(f"[INFO] Density grid passed to PyGMT as file: {tmp_grid_file}")

    fig.grdimage(
        grid=grid_for_plot,
        cmap=str(cpt_file),
        transparency=0,
    )

    if do_height_polygons:
        plot_buildings_by_height(fig, buildings, add_colorbar=False)

    if do_overlay_building_outlines:
        plot_polygons_constant(
            fig,
            buildings,
            fill=None,
            pen=BUILDING_OUTLINE_PEN,
            label="Building outline",
        )

    if do_overlay_highrise and highrise is not None and not highrise.empty:
        plot_polygons_constant(
            fig,
            highrise,
            fill=HIGHRISE_FILL,
            pen=HIGHRISE_PEN,
            label="High-rise building",
        )

    if do_overlay_points and point_df is not None and not point_df.empty:
        plot_df = decimate_points_for_plot(point_df, MAX_POINT_OVERLAY_TO_PLOT)

        fig.plot(
            x=plot_df["x"],
            y=plot_df["y"],
            style=POINT_INPUT_STYLE,
            fill="black",
            pen=None,
            transparency=60,
            label="Centroid + vertices",
        )

    plot_aoi_boundary(fig)

    fig.colorbar(
        cmap=str(cpt_file),
        position="JBC+w9c/0.35c+o0c/1.0c+h",
        frame=[
            "xaf+lBuilding density",
            "y+lNormalized",
        ],
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    save_fig(fig, out_png)


# ============================================================
# CLEANUP
# ============================================================

def cleanup_cpt_and_temp_files():
    if not CLEANUP_TEMP_FILES:
        print("")
        print("========== CLEANUP TEMP FILES ==========")
        print("Skipped because CLEANUP_TEMP_FILES = False")
        return

    print("")
    print("========== CLEANUP TEMP FILES ==========")

    cleanup_patterns = [
        OUT_DIR / "*.cpt",
        FIG_DIR / "*.cpt",
        FIG_DIR / "**" / "*.cpt",

        OUT_DIR / "gmt.history",
        FIG_DIR / "gmt.history",
        Path("gmt.history"),
        OUT_DIR / ".gmt*",
        FIG_DIR / ".gmt*",
        Path(".gmt*"),

        OUT_DIR / "_tmp*",
        FIG_DIR / "_tmp*",
        FIG_DIR / "**" / "_tmp*",
        OUT_DIR / "*.tmp",
        FIG_DIR / "*.tmp",
        OUT_DIR / "*.nc",
        FIG_DIR / "*.nc",
    ]

    protected_suffixes = {
        ".png", ".jpg", ".jpeg", ".pdf",
        ".xyz", ".csv", ".gpkg", ".tif", ".tiff",
        ".shp", ".shx", ".dbf", ".prj", ".cpg",
        ".geojson", ".json",
    }

    removed_files = 0

    for pattern in cleanup_patterns:
        for path in pattern.parent.glob(pattern.name):
            if not path.is_file():
                continue

            if path.suffix.lower() in protected_suffixes:
                continue

            try:
                path.unlink()
                removed_files += 1
                print(f"[CLEAN] Removed file: {path}")
            except Exception as exc:
                print(f"[WARN] Could not remove {path}: {exc}")

    temp_dirs = [
        FIG_DIR / "_tmp",
        FIG_DIR / "_tmp_reprojected_rasters",
        OUT_DIR / "_tmp",
        OUT_DIR / "_tmp_reprojected_rasters",
    ]

    removed_dirs = 0

    for d in temp_dirs:
        if d.exists() and d.is_dir():
            try:
                d.rmdir()
                removed_dirs += 1
                print(f"[CLEAN] Removed empty temp dir: {d}")
            except OSError:
                print(f"[INFO] Temp dir not empty, keep: {d}")

    print(
        f"[OK] Cleanup done. Removed files: {removed_files}, "
        f"removed empty dirs: {removed_dirs}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("")
    print("========== BUILDING DENSITY ==========")
    print(f"Building source:        {source_label()}")
    print(f"Building input dir:     {BUILDING_INPUT_DIR}")
    print(f"Low memory mode:        {LOW_MEMORY_MODE}")
    print(f"Density backend request:{DENSITY_PLOT_BACKEND}")
    print(f"Density backend actual: {get_effective_density_plot_backend()}")
    print(f"Auto force MPL for GBA: {AUTO_FORCE_MATPLOTLIB_FOR_GBA_OR_BOTH}")
    print(f"Disable density overlay:{DISABLE_DENSITY_POLYGON_OVERLAYS}")
    print(f"Keep point DF for plot: {KEEP_POINT_DF_FOR_PLOTTING}")
    print(f"XYZ chunksize:          {XYZ_CHUNKSIZE}")
    print(f"Output dir:             {OUT_DIR}")
    print(f"Figure dir:             {FIG_DIR}")

    if not BUILDING_INPUT_DIR.exists():
        raise FileNotFoundError(f"Building input directory does not exist: {BUILDING_INPUT_DIR}")

    region0 = get_region_from_polygon(padding=REGION_PADDING)
    region = snap_region_to_spacing(region0, DENSITY_GRID_SPACING_DEG)

    print(f"Plot region: {region}")

    buildings, building_files = load_buildings()
    highrise, highrise_method = select_highrise_buildings(buildings)

    gaussian_points_df = None
    gaussian_point_files = collect_gaussian_point_files()
    gaussian_point_mode = "streaming" if LOW_MEMORY_MODE else "dataframe"
    gaussian_points_read = 0
    gaussian_points_inside_polygon = 0

    print("")
    print("========== GAUSSIAN POINT FILES ==========")
    print(f"Files found: {len(gaussian_point_files)}")
    for f in gaussian_point_files:
        print(f"  - {f.name}")

    polygon_grid = None
    point_grid = None
    combined_grid = None

    if CREATE_POLYGON_DENSITY:
        polygon_grid, polygon_region = create_polygon_density_grid(
            buildings=buildings,
            region=region,
        )

        save_density_grid_xyz(
            density_grid=polygon_grid,
            out_file=OUT_POLYGON_DENSITY_XYZ,
        )

    if CREATE_GAUSSIAN_POINT_DENSITY and gaussian_point_files:
        if LOW_MEMORY_MODE:
            point_grid, point_region, gaussian_points_read, gaussian_points_inside_polygon = (
                create_gaussian_point_density_grid_streaming(
                    files=gaussian_point_files,
                    region=region,
                )
            )

            if KEEP_POINT_DF_FOR_PLOTTING:
                gaussian_points_df, _, _ = load_combined_gaussian_points_for_plot()

        else:
            gaussian_points_df, gaussian_point_files, gaussian_point_mode = (
                load_combined_gaussian_points_for_plot()
            )

            if gaussian_points_df is not None:
                gaussian_points_read = len(gaussian_points_df)
                gaussian_points_inside_polygon = len(gaussian_points_df)

                point_grid, point_region = create_gaussian_point_density_grid_from_df(
                    points_df=gaussian_points_df,
                    region=region,
                )

        save_density_grid_xyz(
            density_grid=point_grid,
            out_file=OUT_GAUSSIAN_POINTS_DENSITY_XYZ,
        )

    elif CREATE_GAUSSIAN_POINT_DENSITY:
        print("[WARNING] No Gaussian point files found. Skip point density.")

    if CREATE_COMBINED_DENSITY:
        combined_grid = combine_density_grids(
            polygon_grid=polygon_grid,
            point_grid=point_grid,
        )

        save_density_grid_xyz(
            density_grid=combined_grid,
            out_file=OUT_COMBINED_DENSITY_XYZ,
        )

    summary = pd.DataFrame(
        [
            {
                "building_data_source": BUILDING_DATA_SOURCE,
                "source_label": source_label(),
                "building_input_dir": str(BUILDING_INPUT_DIR),
                "building_files": "; ".join([f.name for f in building_files]),
                "buildings_used": len(buildings),
                "height_column": HEIGHT_COLUMN if HEIGHT_COLUMN in buildings.columns else "",
                "highrise_method": highrise_method,
                "highrise_count": len(highrise) if highrise is not None else 0,
                "gaussian_point_mode": gaussian_point_mode,
                "gaussian_point_files": "; ".join([f.name for f in gaussian_point_files]),
                "gaussian_points_read": gaussian_points_read,
                "gaussian_points_inside_polygon": gaussian_points_inside_polygon,
                "low_memory_mode": LOW_MEMORY_MODE,
                "density_plot_backend_requested": DENSITY_PLOT_BACKEND,
                "density_plot_backend_effective": get_effective_density_plot_backend(),
                "auto_force_matplotlib_for_gba_or_both": AUTO_FORCE_MATPLOTLIB_FOR_GBA_OR_BOTH,
                "xyz_chunksize": XYZ_CHUNKSIZE,
                "low_memory_skip_point_dedup": LOW_MEMORY_SKIP_POINT_DEDUP,
                "density_grid_spacing_degree": DENSITY_GRID_SPACING_DEG,
                "polygon_gaussian_sigma_cells": POLYGON_GAUSSIAN_SIGMA_CELLS,
                "polygon_density_cutoff": POLYGON_DENSITY_CUTOFF,
                "point_gaussian_sigma_cells": POINT_GAUSSIAN_SIGMA_CELLS,
                "point_density_cutoff": POINT_DENSITY_CUTOFF,
                "combined_density_mode": COMBINED_DENSITY_MODE,
                "polygon_density_weight": POLYGON_DENSITY_WEIGHT,
                "point_density_weight": POINT_DENSITY_WEIGHT,
                "rasterize_all_touched": RASTERIZE_ALL_TOUCHED,
                "outside_polygon_density": 0.0,
                "polygon_max_density": float(polygon_grid.max()) if polygon_grid is not None else np.nan,
                "point_max_density": float(point_grid.max()) if point_grid is not None else np.nan,
                "combined_max_density": float(combined_grid.max()) if combined_grid is not None else np.nan,
                "polygon_density_xyz": str(OUT_POLYGON_DENSITY_XYZ),
                "gaussian_points_density_xyz": str(OUT_GAUSSIAN_POINTS_DENSITY_XYZ),
                "combined_density_xyz": str(OUT_COMBINED_DENSITY_XYZ),
                "height_figure": str(OUT_HEIGHT_FIG),
                "polygon_density_figure": str(OUT_POLYGON_DENSITY_FIG),
                "gaussian_points_density_figure": str(OUT_GAUSSIAN_POINTS_DENSITY_FIG),
                "combined_density_figure": str(OUT_COMBINED_DENSITY_FIG),
            }
        ]
    )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] Saved summary CSV: {OUT_SUMMARY}")

    if PLOT_HEIGHT_MAP:
        plot_buildings_height_map(
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_HEIGHT_FIG,
        )

    if PLOT_POLYGON_DENSITY_MAP and polygon_grid is not None:
        plot_density_map(
            density_grid=polygon_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_POLYGON_DENSITY_FIG,
            title=f"{source_label()} polygon footprint density",
            point_df=None,
            overlay_building_outlines=True,
            overlay_highrise=True,
            overlay_points=False,
        )

    if PLOT_GAUSSIAN_POINT_DENSITY_MAP and point_grid is not None:
        plot_density_map(
            density_grid=point_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_GAUSSIAN_POINTS_DENSITY_FIG,
            title=f"Gaussian density from {source_label()} centroids + vertices",
            point_df=gaussian_points_df,
            overlay_building_outlines=False,
            overlay_highrise=False,
            overlay_points=(gaussian_points_df is not None),
        )

    if PLOT_COMBINED_DENSITY_MAP and combined_grid is not None:
        plot_density_map(
            density_grid=combined_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_COMBINED_DENSITY_FIG,
            title=f"Combined {source_label()} density: polygon 40% + points 60%",
            point_df=None,
            overlay_building_outlines=True,
            overlay_highrise=True,
            overlay_points=False,
        )

    cleanup_cpt_and_temp_files()

    print("")
    print("========== DONE ==========")
    print(f"Building source:               {source_label()}")
    print(f"Height figure:                 {OUT_HEIGHT_FIG}")
    print(f"Polygon density figure:        {OUT_POLYGON_DENSITY_FIG}")
    print(f"Gaussian point density figure: {OUT_GAUSSIAN_POINTS_DENSITY_FIG}")
    print(f"Combined density figure:       {OUT_COMBINED_DENSITY_FIG}")
    print(f"Polygon density xyz:           {OUT_POLYGON_DENSITY_XYZ}")
    print(f"Gaussian point density xyz:    {OUT_GAUSSIAN_POINTS_DENSITY_XYZ}")
    print(f"Combined density xyz:          {OUT_COMBINED_DENSITY_XYZ}")
    print(f"Summary CSV:                   {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
