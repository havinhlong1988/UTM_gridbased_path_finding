#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_collect_obstacle.py

Collect Scenario-1 obstacle input layers from downloaded Hoa Lac study-area data.

User requested source:
    ../downloaddata/output/01_HoaLac_studies_area/

Destination:
    input/data_senario1/

Diagnostic figures:
    input/data_senario1/figures/**

This script is intended to be placed in:
    {PROJECT_ROOT}/make_model/01_collect_obstacle.py

Run from make_model/:
    python 01_collect_obstacle.py

It collects and checks the minimum obstacle-input stack:

    (1) OSM / OpenInfraMap style urban layers
        - power lines / cables with AGL fields if available
        - power towers / poles with AGL or height fields if available
        - road center line / road vertices / road sample points
        - traffic lights with AGL / height fields if available
        - street lamps optional, if downloaded

    (2) GBA / GlobalBuildingAtlas LoD1
        - building footprint polygons
        - building height attributes
        - centroid XYZ / vertices XYZ / full attributes CSV if available

    (3) OTP / OpenTopography
        - DEM / elevation raster and XYZ products

Outputs:
    input/data_senario1/
    ├── metadata/
    ├── osm/
    ├── gba/
    ├── opentopography/
    ├── figures/
    │   ├── 00_summary/
    │   ├── 01_osm/
    │   ├── 02_gba/
    │   └── 03_opentopography/
    ├── scenario1_obstacle_input_manifest.csv
    ├── scenario1_obstacle_layer_summary.csv
    ├── scenario1_obstacle_file_inspection.csv
    ├── scenario1_building_height_summary.csv
    └── scenario1_obstacle_layer_summary.txt

Notes:
    - The script does not copy raw tiles, figure folders, cache parquet, or temporary files.
    - It is robust to several folder/file names produced by the previous download scripts.
    - Figure creation is skipped gracefully if optional packages are missing.
"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# USER PARAMETERS
# ============================================================

# Main input and output paths.
# If this script is in {PROJECT_ROOT}/make_model:
#   SOURCE_DIR -> {PROJECT_ROOT}/downloaddata/output/01_HoaLac_studies_area
#   DEST_DIR   -> {PROJECT_ROOT}/make_model/input/data_senario1
SOURCE_DIR = Path("../downloaddata/output/01_HoaLac_studies_area")
DEST_DIR = Path("input/data_senario1")
PATHS_RELATIVE_TO_SCRIPT = True

# Copy behavior.
OVERWRITE_EXISTING = True
CLEAN_DESTINATION_FIRST = True
DRY_RUN = False
STRICT_REQUIRED_LAYERS = False

# Plot behavior.
MAKE_FIGURES = True
FIGURES_SUBDIR = "figures"
MAX_VECTOR_PLOT_FEATURES = 50000
MAX_RASTER_PLOT_PIXELS = 3_000_000

# Building height assumptions.
DEFAULT_OBM_BUILDING_HEIGHT_M = 3.0
DEFAULT_GBA_BUILDING_HEIGHT_M = 6.0
OBM_LEVEL_HEIGHT_M = 3.0

# Optional: include AOI/project metadata.
COPY_METADATA = True

# Copy only files that are already clipped to the Hoa Lac/model area.
# This prevents bbox/raw/wide-region OSM files from entering the model input.
COPY_ONLY_CLIPPED_DATA = True

# Extra safety: after copying vector/XYZ files, clip them again to the hardcoded
# Hoa Lac polygon below when geopandas/shapely/pandas are available.
# If packages are missing, the script still copies filtered clipped files.
FORCE_RECLIP_COPIED_VECTORS_TO_AOI = True
FORCE_RECLIP_COPIED_XYZ_TO_AOI = True

# Hoa Lac model polygon used for strict re-clipping. Format: lon, lat.
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

# Do not copy these folders or suffixes.
EXCLUDE_DIR_NAMES = {
    "figures",
    "figure",
    "raw_tiles",
    "raw_bbox_tif",
    "dem_tiles",
    "api_errors",
    "__pycache__",
    ".ipynb_checkpoints",
}

EXCLUDE_SUFFIXES = {
    ".tmp",
    ".temp",
    ".lock",
}

# Explicitly avoid heavy raw/cache files.
EXCLUDE_NAME_CONTAINS = {
    "bbox_filtered_lowram.parquet",
    "raw.parquet",
    "raw_tiles",
    "cache",
    "bbox_merged",
    "_bbox",
    "bbox.",
}

# Used to avoid filename collisions inside one run.
USED_DESTINATIONS: set[Path] = set()


# ============================================================
# LAYER COPY SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class LayerSpec:
    layer_name: str
    destination_subdir: str
    voxel_role: str
    required: bool
    note: str
    patterns: tuple[str, ...]


LAYER_SPECS: list[LayerSpec] = [
    LayerSpec(
        layer_name="metadata_aoi",
        destination_subdir="metadata",
        voxel_role="AOI / clipping boundary",
        required=False,
        note="Study-area polygon and metadata used for checking extent and clipping.",
        patterns=(
            "metadata/study_area_aoi.gpkg",
            "metadata/study_area_aoi.geojson",
            "metadata/project_metadata.json",
            "metadata/*.csv",
            "**/hoalac_polygon.gpkg",
            "**/hoalac_polygon.geojson",
            "**/study_area_aoi.gpkg",
            "**/study_area_aoi.geojson",
        ),
    ),

    # ------------------------------
    # OSM / OpenInfraMap obstacle layers
    # ------------------------------
    LayerSpec(
        layer_name="osm_powerlines_agl",
        destination_subdir="osm/powerlines",
        voxel_role="hard linear obstacle; use z_min_agl_m / z_max_agl_m if present",
        required=True,
        note="Power line/cable geometry. Height is often assumed, so check AGL columns before using.",
        patterns=(
            "**/powerlines_hoalac_clipped.gpkg",
            "**/powerlines_hoalac_clipped.geojson",
            "**/powerlines_vertices_hoalac.xyz",
            "**/powerlines*.gpkg",
            "**/powerlines*.geojson",
            "**/powerlines*.xyz",
            "**/powerline*.gpkg",
            "**/powerline*.geojson",
            "**/powerline*.xyz",
            "**/power_lines*.gpkg",
            "**/power_lines*.geojson",
            "**/power_lines*.xyz",
            "**/*power*line*.gpkg",
            "**/*power*line*.geojson",
            "**/*power*line*.xyz",
        ),
    ),
    LayerSpec(
        layer_name="osm_power_poles_towers_agl",
        destination_subdir="osm/power_poles_towers",
        voxel_role="hard point/vertical obstacle; use height_m / z_max_agl_m if present",
        required=True,
        note="Power poles and towers. If no true height exists, use assumed height with low confidence.",
        patterns=(
            "**/power_towers_poles_hoalac_clipped.gpkg",
            "**/power_towers_poles_hoalac_clipped.geojson",
            "**/power_towers_poles_points_hoalac.xyz",
            "**/power_supports*.gpkg",
            "**/power_supports*.geojson",
            "**/power_supports*.xyz",
            "**/*power*tower*.gpkg",
            "**/*power*tower*.geojson",
            "**/*power*tower*.xyz",
            "**/*power*pole*.gpkg",
            "**/*power*pole*.geojson",
            "**/*power*pole*.xyz",
            "**/*tower*pole*.gpkg",
            "**/*tower*pole*.geojson",
            "**/*tower*pole*.xyz",
            "**/*towers_poles*.gpkg",
            "**/*towers_poles*.geojson",
            "**/*towers_poles*.xyz",
        ),
    ),
    LayerSpec(
        layer_name="osm_road_centerline",
        destination_subdir="osm/roads",
        voxel_role="road centerline / soft risk / road-crossing reference",
        required=True,
        note="Roads are usually not hard aerial obstacles, but useful for population/traffic/road-crossing risk.",
        patterns=(
            "osm/roads/osm_roads_edges.gpkg",
            "osm/roads/osm_roads_edges.geojson",
            "osm/roads/osm_roads_nodes.gpkg",
            "osm/roads/osm_road_class_summary.csv",
            "**/roads_hoalac_clipped.gpkg",
            "**/roads_hoalac_clipped.geojson",
            "**/roads_vertices_hoalac.xyz",
            "**/roads_points_hoalac.xyz",
            "**/roads_points_hoalac.csv",
            "**/osm_roads_edges.gpkg",
            "**/osm_roads_edges.geojson",
            "**/osm_roads_nodes.gpkg",
            "**/osm_road_class_summary.csv",
            "**/*road*center*.gpkg",
            "**/*road*center*.geojson",
            "**/*road*center*.xyz",
        ),
    ),
    LayerSpec(
        layer_name="osm_traffic_lights_agl",
        destination_subdir="osm/traffic_lights",
        voxel_role="low-altitude point obstacle near landing/takeoff; use height_m if present",
        required=True,
        note="Traffic lights are important mostly for low-altitude landing/takeoff corridors.",
        patterns=(
            "**/traffic_lights_hoalac_clipped.gpkg",
            "**/traffic_lights_hoalac_clipped.geojson",
            "**/traffic_lights_points_hoalac.xyz",
            "**/*traffic*light*.gpkg",
            "**/*traffic*light*.geojson",
            "**/*traffic*light*.xyz",
            "**/*traffic*signal*.gpkg",
            "**/*traffic*signal*.geojson",
            "**/*traffic*signal*.xyz",
            "**/*traffic*line*.gpkg",
            "**/*traffic*line*.geojson",
            "**/*traffic*line*.xyz",
        ),
    ),
    LayerSpec(
        layer_name="osm_street_lamps_agl_optional",
        destination_subdir="osm/street_lamps",
        voxel_role="optional low-altitude point obstacle; use height_m if present",
        required=False,
        note="Street lamps are optional but useful around takeoff/landing corridors if downloaded.",
        patterns=(
            "**/street_lamps_hoalac_clipped.gpkg",
            "**/street_lamps_hoalac_clipped.geojson",
            "**/street_lamps_points_hoalac.xyz",
            "**/*street*lamp*.gpkg",
            "**/*street*lamp*.geojson",
            "**/*street*lamp*.xyz",
        ),
    ),

    # ------------------------------
    # OBM optional, because the uploaded helper checks it too.
    # GBA remains the requested primary building source.
    # ------------------------------
    LayerSpec(
        layer_name="obm_building_optional",
        destination_subdir="obm",
        voxel_role="optional building footprint/height cross-check",
        required=False,
        note="Optional OpenBuildingMap source used only for building-height comparison if available.",
        patterns=(
            "openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg",
            "openbuildingmap/clipped/obm_buildings_hoalac_clipped.geojson",
            "openbuildingmap/clipped/obm_summary.csv",
            "**/obm_buildings_hoalac_clipped.gpkg",
            "**/obm_buildings_hoalac_clipped.geojson",
            "**/obm_summary.csv",
        ),
    ),
    LayerSpec(
        layer_name="gba_building_footprint_height",
        destination_subdir="gba",
        voxel_role="hard polygon obstacle; building footprint + building height",
        required=True,
        note="Primary building obstacle source. Use footprint polygons and height/LoD1 attributes.",
        patterns=(
            "globalbuildingatlas_lod1/metadata/hoalac_polygon.gpkg",
            "globalbuildingatlas_lod1/metadata/selected_gba_5deg_tiles.csv",
            "globalbuildingatlas_lod1/metadata/selected_gba_5deg_tiles.gpkg",
            "globalbuildingatlas_lod1/metadata/gba_lod1_summary.csv",
            "globalbuildingatlas_lod1/metadata/selected_tiles_download_status.csv",
            "globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg",
            "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac.xyz",
            "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac_with_info.xyz",
            "globalbuildingatlas_lod1/processed/gba_lod1_buildings_vertices_hoalac.xyz",
            "globalbuildingatlas_lod1/processed/gba_lod1_buildings_full_attributes.csv",
            "**/gba_lod1_buildings_hoalac_clipped.gpkg",
            "**/gba_lod1_buildings_centroid_hoalac*.xyz",
            "**/gba_lod1_buildings_vertices_hoalac.xyz",
            "**/gba_lod1_buildings_full_attributes.csv",
            "**/gba_lod1_summary.csv",
            "**/selected_gba_5deg_tiles.csv",
            "**/selected_gba_5deg_tiles.gpkg",
        ),
    ),

    # ------------------------------
    # OpenCelliD / communication-support soft-risk products
    # ------------------------------
    LayerSpec(
        layer_name="opencellid_communication_soft_risk",
        destination_subdir="opencellid",
        voxel_role="soft communication risk/support; not a hard obstacle",
        required=False,
        note="OpenCelliD cells/range/support/risk grids clipped to Hoa Lac. Use as soft cost, not hard obstacle.",
        patterns=(
            "opencellid/opencellid_cells_hoalac.gpkg",
            "opencellid/opencellid_cells_hoalac.csv",
            "opencellid/opencellid_range_circles_hoalac.gpkg",
            "opencellid/communication_support_grid_hoalac.gpkg",
            "opencellid/communication_support_grid_hoalac.csv",
            "opencellid/xyz/*hoalac.xyz",
            "opencellid/layers/*hoalac.gpkg",
            "**/opencellid/**/*hoalac*.gpkg",
            "**/opencellid/**/*hoalac*.csv",
            "**/opencellid/**/*hoalac*.xyz",
            "**/opencellid/**/*communication_support_grid*.gpkg",
            "**/opencellid/**/*communication_support_grid*.csv",
            "**/opencellid/**/*communication_support_grid*.xyz",
            "**/opencellid/**/*communication_risk_grid*.xyz",
        ),
    ),

    # ------------------------------
    # OpenTopography / OTP DEM and elevation products
    # ------------------------------
    LayerSpec(
        layer_name="otp_dem_elevation",
        destination_subdir="opentopography",
        voxel_role="terrain/elevation base for AGL calculation",
        required=True,
        note="DEM/elevation source. For AGL: height_agl = z_voxel - dem_elevation.",
        patterns=(
            "opentopography/*_dem_wgs84.tif",
            "opentopography/*_dem_utm.tif",
            "opentopography/*elevation*.tif",
            "opentopography/*dem*.tif",
            "opentopography/clipped_tif/*.tif",
            "opentopography/derived_tif/*elevation*.tif",
            "opentopography/derived_tif/*dem*.tif",
            "opentopography/xyz/*elevation*.xyz",
            "opentopography/xyz/*dem*.xyz",
            "opentopography/*elevation*.xyz",
            "opentopography/*dem*.xyz",
            "**/opentopography/**/*_dem_wgs84.tif",
            "**/opentopography/**/*_dem_utm.tif",
            "**/opentopography/**/*elevation*.tif",
            "**/opentopography/**/*dem*.tif",
            "**/opentopography/**/*elevation*.xyz",
            "**/opentopography/**/*dem*.xyz",
        ),
    ),
]


# ============================================================
# OPTIONAL IMPORTS
# ============================================================

def optional_imports() -> dict[str, Any | None]:
    """Import optional GIS/plotting libraries."""
    mods: dict[str, Any | None] = {}

    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        gpd = None
    mods["gpd"] = gpd

    try:
        import rasterio  # type: ignore
    except Exception:
        rasterio = None
    mods["rasterio"] = rasterio

    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
    mods["pd"] = pd

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    mods["np"] = np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None
    mods["plt"] = plt

    return mods


# ============================================================
# PATH AND COPY HELPERS
# ============================================================

def script_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def resolve_user_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    if PATHS_RELATIVE_TO_SCRIPT:
        return (script_base_dir() / path).expanduser().resolve()
    return (Path.cwd() / path).expanduser().resolve()


def human_size(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def is_excluded(path: Path) -> bool:
    lower_parts = {p.lower() for p in path.parts}
    if any(d.lower() in lower_parts for d in EXCLUDE_DIR_NAMES):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    lower_name = path.name.lower()
    if any(s.lower() in lower_name for s in EXCLUDE_NAME_CONTAINS):
        return True
    return False


def is_explicitly_clipped_model_area_file(path: Path, layer_name: str) -> bool:
    """
    Keep only Hoa-Lac/model-area products.

    This is intentionally strict because the download scripts often create both:
        - bbox / raw / wide-region files for downloading/checking
        - hoalac / clipped files for model input

    For model building we only want the second group.
    """
    if not COPY_ONLY_CLIPPED_DATA:
        return True

    rel = Path(path)
    rel_txt = str(rel).replace("\\", "/").lower()
    name = rel.name.lower()
    parts = {part.lower() for part in rel.parts}

    # Always reject wide/raw/bbox products.
    blocked_tokens = [
        "/bbox_layers/",
        "/raw_bbox_tif/",
        "/dem_tiles/",
        "/raw_tiles/",
        "/api_errors/",
        "_bbox.",
        "_bbox_",
        "bbox_",
        "bbox-",
        "bbox/",
        "raw_",
        "raw.",
        "merged",
    ]
    if any(tok in rel_txt for tok in blocked_tokens):
        return False

    # AOI metadata is model-area by definition if it is an AOI/Hoa Lac polygon.
    if layer_name == "metadata_aoi":
        return any(tok in rel_txt for tok in ["hoalac", "hoa_lac", "study_area", "aoi", "polygon"])

    # OSM/OpenInfraMap layers must be explicitly clipped to Hoa Lac.
    if layer_name.startswith("osm_"):
        # OpenInfraMap script stores the AOI copy here.
        if "hoalac_clipped_layers" in rel_txt:
            return True
        # Original OSM script uses these names.
        if "hoalac" in name or "clipped" in name:
            return True
        # Do not allow generic OSM road/power files such as osm_roads_edges.gpkg.
        return False

    # GBA/OBM buildings: only keep Hoa-Lac processed/clipped files.
    if layer_name.startswith("gba_") or layer_name.startswith("obm_"):
        if "hoalac" not in rel_txt and "clipped" not in rel_txt:
            return False
        # Tile metadata is not a model-area obstacle file.
        if "selected_gba_5deg_tiles" in rel_txt or "download_status" in rel_txt:
            return False
        return True

    # OpenTopography: keep clipped raster, derived raster, and hoalac XYZ only.
    if layer_name.startswith("otp_"):
        if "/clipped_tif/" in rel_txt or "/derived_tif/" in rel_txt:
            return True
        if "/xyz/" in rel_txt and "hoalac" in name:
            return True
        if "hoalac" in name and rel.suffix.lower() in {".tif", ".xyz"}:
            return True
        return False

    # OpenCelliD/communication products: keep only Hoa-Lac outputs.
    if "opencellid" in layer_name or "communication" in layer_name:
        return "hoalac" in rel_txt or "communication_support_grid" in rel_txt or "communication_risk_grid" in rel_txt

    # Default strict behavior: require clipped/HoaLac naming.
    return "hoalac" in rel_txt or "clipped" in rel_txt


def make_hardcoded_aoi_gdf(mods: dict[str, Any | None]) -> Any | None:
    """Create the Hoa Lac AOI GeoDataFrame for forced re-clipping."""
    gpd = mods.get("gpd")
    if gpd is None:
        return None
    try:
        from shapely.geometry import Polygon  # type: ignore
        poly = Polygon(HOALAC_POLYGON)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return gpd.GeoDataFrame({"name": ["Hoa_Lac_model_AOI"]}, geometry=[poly], crs="EPSG:4326")
    except Exception as exc:
        print(f"[WARN] Could not create hardcoded AOI polygon: {exc}")
        return None


def is_vector_suffix(path: Path) -> bool:
    return path.suffix.lower() in {".gpkg", ".geojson", ".shp"}


def is_tabular_xyz(path: Path) -> bool:
    return path.suffix.lower() == ".xyz"


def copy_vector_reclipped(src: Path, dst: Path, mods: dict[str, Any | None]) -> bool:
    """
    Try to read a vector file, clip to Hoa Lac polygon, then save it to dst.
    Return True when successful. If False, caller should fall back to copy2.
    """
    if not FORCE_RECLIP_COPIED_VECTORS_TO_AOI:
        return False

    gpd = mods.get("gpd")
    if gpd is None:
        return False

    aoi = make_hardcoded_aoi_gdf(mods)
    if aoi is None:
        return False

    try:
        gdf = gpd.read_file(src)
        if gdf.empty:
            dst.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(dst, driver="GPKG" if dst.suffix.lower() == ".gpkg" else None)
            return True
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
        clipped = gpd.clip(gdf, aoi).reset_index(drop=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix.lower() == ".geojson":
            clipped.to_file(dst, driver="GeoJSON")
        else:
            clipped.to_file(dst, driver="GPKG")
        print(f"      [RECLIP] vector clipped to Hoa Lac AOI: {dst.name} (n={len(clipped)})")
        return True
    except Exception as exc:
        print(f"      [WARN] Vector reclip failed for {src.name}: {exc}. Fallback copy.")
        return False


def copy_xyz_reclipped(src: Path, dst: Path, mods: dict[str, Any | None]) -> bool:
    """
    Clip XYZ rows by the first two columns lon/lat when possible.
    This prevents any remaining out-of-AOI vertices/points from being used.
    """
    if not FORCE_RECLIP_COPIED_XYZ_TO_AOI:
        return False

    pd = mods.get("pd")
    if pd is None:
        return False

    try:
        from shapely.geometry import Point, Polygon  # type: ignore
        poly = Polygon(HOALAC_POLYGON)
        if not poly.is_valid:
            poly = poly.buffer(0)
        df = pd.read_csv(src, sep=r"\s+", header=None, engine="python")
        if df.empty or df.shape[1] < 2:
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst, sep=" ", index=False, header=False)
            return True
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        keep = []
        for xi, yi in zip(x, y):
            if pd.isna(xi) or pd.isna(yi):
                # Drop separator rows for model-ready XYZ files.
                keep.append(False)
            else:
                keep.append(poly.contains(Point(float(xi), float(yi))) or poly.touches(Point(float(xi), float(yi))))
        out = df.loc[keep].copy()
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, sep=" ", index=False, header=False, float_format="%.8f")
        print(f"      [RECLIP] XYZ clipped to Hoa Lac AOI: {dst.name} (rows={len(out)})")
        return True
    except Exception as exc:
        print(f"      [WARN] XYZ reclip failed for {src.name}: {exc}. Fallback copy.")
        return False


def safe_unique_destination(dst_dir: Path, src: Path, src_root: Path) -> Path:
    """Create a stable destination filename, avoiding basename collisions."""
    candidate = dst_dir / src.name

    # Reuse the simple basename only if it is not already used in this run.
    # If it exists from an older run, OVERWRITE_EXISTING controls replacement.
    if candidate not in USED_DESTINATIONS and (OVERWRITE_EXISTING or not candidate.exists()):
        USED_DESTINATIONS.add(candidate)
        return candidate

    try:
        rel = src.relative_to(src_root)
        stem_prefix = "__".join(rel.parts[-4:-1])
        stem_prefix = stem_prefix.replace(" ", "_")
        candidate = dst_dir / f"{stem_prefix}__{src.name}"
        if candidate not in USED_DESTINATIONS and (OVERWRITE_EXISTING or not candidate.exists()):
            USED_DESTINATIONS.add(candidate)
            return candidate
    except Exception:
        pass

    i = 2
    while True:
        candidate = dst_dir / f"{src.stem}_{i}{src.suffix}"
        if candidate not in USED_DESTINATIONS and (OVERWRITE_EXISTING or not candidate.exists()):
            USED_DESTINATIONS.add(candidate)
            return candidate
        i += 1


def find_matches(src_root: Path, patterns: tuple[str, ...], layer_name: str = "") -> list[Path]:
    """Find matching files under source root, keeping only clipped/model-area files."""
    matches: list[Path] = []
    seen: set[Path] = set()
    skipped_wide = 0

    for pattern in patterns:
        for p in sorted(src_root.glob(pattern)):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(src_root)
            except Exception:
                rel = p
            if is_excluded(rel):
                continue
            if not is_explicitly_clipped_model_area_file(rel, layer_name):
                skipped_wide += 1
                continue
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(p)

    if skipped_wide:
        print(f"  [INFO] Skipped {skipped_wide} bbox/raw/out-region file(s) for {layer_name}")

    return matches


def copy_file(src: Path, src_root: Path, dst_root: Path, dst_subdir: str, layer_name: str, mods: dict[str, Any | None]) -> tuple[Path, str]:
    dst_dir = dst_root / dst_subdir
    dst_file = safe_unique_destination(dst_dir, src, src_root)

    if dst_file.exists() and not OVERWRITE_EXISTING:
        return dst_file, "exists_skip"

    if not DRY_RUN:
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Extra safety: vector and XYZ sources are clipped again to the hardcoded
        # Hoa Lac AOI. This fixes cases where a file name looked clipped but still
        # contained out-region geometries.
        reclipped = False
        if layer_name != "metadata_aoi" and is_vector_suffix(src):
            reclipped = copy_vector_reclipped(src, dst_file, mods)
        elif layer_name != "metadata_aoi" and is_tabular_xyz(src):
            reclipped = copy_xyz_reclipped(src, dst_file, mods)

        if not reclipped:
            shutil.copy2(src, dst_file)

    return dst_file, "dry_run_copy" if DRY_RUN else "copied"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_from_count(count: int, required: bool) -> str:
    if count > 0:
        return "available"
    if required:
        return "missing_required"
    return "missing_optional"


# ============================================================
# INSPECTION HELPERS
# ============================================================

def detect_height_columns(columns: list[str]) -> list[str]:
    """Detect likely height/AGL/elevation fields."""
    keywords = (
        "height", "h_m", "z_", "zmin", "zmax", "z_min", "z_max",
        "agl", "elev", "elevation", "alt", "range", "level", "floor"
    )
    found = []
    for col in columns:
        c = str(col).lower()
        if any(k in c for k in keywords):
            found.append(str(col))
    return found


def inspect_file(path: Path, mods: dict[str, Any | None]) -> dict[str, Any]:
    """Return lightweight metadata for copied file."""
    gpd = mods.get("gpd")
    rasterio = mods.get("rasterio")
    pd = mods.get("pd")
    np = mods.get("np")

    info: dict[str, Any] = {
        "file": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "size_human": human_size(path.stat().st_size) if path.exists() else "0 B",
        "feature_count": "",
        "raster_shape": "",
        "raster_crs": "",
        "columns": "",
        "height_agl_columns": "",
        "min_value": "",
        "max_value": "",
        "mean_value": "",
        "note": "",
    }

    suffix = path.suffix.lower()

    try:
        if suffix in {".gpkg", ".geojson", ".shp"} and gpd is not None:
            gdf = gpd.read_file(path)
            info["feature_count"] = len(gdf)
            columns = [str(c) for c in gdf.columns]
            info["columns"] = ";".join(columns[:120])
            info["height_agl_columns"] = ";".join(detect_height_columns(columns))
            if "geometry" in gdf:
                geom_types = sorted(set(gdf.geometry.geom_type.astype(str)))
                info["note"] = "geometry=" + ",".join(geom_types)

        elif suffix == ".tif" and rasterio is not None and np is not None:
            with rasterio.open(path) as src:
                info["raster_shape"] = f"bands={src.count}, height={src.height}, width={src.width}"
                info["raster_crs"] = str(src.crs)
                arr = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    info["min_value"] = float(np.nanmin(finite))
                    info["max_value"] = float(np.nanmax(finite))
                    info["mean_value"] = float(np.nanmean(finite))
                info["note"] = "raster_band_1_stats"

        elif suffix in {".csv", ".xyz"} and pd is not None:
            if suffix == ".csv":
                df = pd.read_csv(path, nrows=100000)
                try:
                    total_rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
                except Exception:
                    total_rows = len(df)
                columns = [str(c) for c in df.columns]
            else:
                df = pd.read_csv(path, sep=r"\s+", header=None, nrows=100000, engine="python")
                try:
                    total_rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))
                except Exception:
                    total_rows = len(df)
                columns = [f"col_{i}" for i in range(df.shape[1])]

            info["feature_count"] = max(total_rows, 0)
            info["columns"] = ";".join(columns[:120])
            info["height_agl_columns"] = ";".join(detect_height_columns(columns))
            info["note"] = f"tabular_rows={max(total_rows, 0)}"

    except Exception as e:
        info["note"] = f"inspection_failed: {e}"

    return info


# ============================================================
# BUILDING HEIGHT CHECK HELPERS
# Derived from the uploaded OBM/GBA height-check idea.
# ============================================================

def parse_obm_height_to_m(value: Any, default_m: float = DEFAULT_OBM_BUILDING_HEIGHT_M) -> float:
    """
    Convert OBM GEM taxonomy height strings to approximate meters.

    Examples:
        HHT:10.0   -> 10.0 m
        H:2        -> 6.0 m, assuming 3 m/story
        HBET:1-3   -> 6.0 m, average stories * 3 m
        UNK / NaN  -> default_m
    """
    try:
        import pandas as pd  # type: ignore
        if value is None or pd.isna(value):
            return default_m
    except Exception:
        if value is None:
            return default_m

    txt = str(value).strip()
    if not txt or txt.upper() in {"UNK", "NULL", "NAN", "NONE"}:
        return default_m

    parts = txt.split("+")

    for part in parts:
        part = part.strip()
        if part.startswith("HHT:"):
            try:
                return float(part.split(":", 1)[1])
            except Exception:
                pass

    for part in parts:
        part = part.strip()
        if part.startswith("H:"):
            try:
                return float(part.split(":", 1)[1]) * OBM_LEVEL_HEIGHT_M
            except Exception:
                pass

    for part in parts:
        part = part.strip()
        if part.startswith("HBET:"):
            try:
                rng = part.split(":", 1)[1]
                if "-" in rng:
                    a, b = rng.split("-", 1)
                    return ((float(a) + float(b)) / 2.0) * OBM_LEVEL_HEIGHT_M
                return float(rng) * OBM_LEVEL_HEIGHT_M
            except Exception:
                pass

    try:
        return float(txt)
    except Exception:
        return default_m


def get_numeric_height_series(gdf: Any, source_name: str, mods: dict[str, Any | None]) -> tuple[Any, str]:
    """Return best numeric height series for GBA/OBM."""
    pd = mods.get("pd")
    if pd is None:
        return [], "NO_PANDAS"

    source = source_name.upper()
    columns_lower = {str(c).lower(): c for c in gdf.columns}

    if source == "GBA":
        for key in ["height_m", "height", "building_height", "mean_height", "h_m", "z_max", "zmax"]:
            if key in columns_lower:
                col = columns_lower[key]
                vals = pd.to_numeric(gdf[col], errors="coerce")
                if vals.notna().any():
                    return vals, str(col)

    if source == "OBM":
        if "height_m" in columns_lower:
            col = columns_lower["height_m"]
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return vals, str(col)

        if "height" in columns_lower:
            col = columns_lower["height"]
            converted = gdf[col].apply(parse_obm_height_to_m)
            return pd.to_numeric(converted, errors="coerce"), f"{col} -> parsed_height_m"

    for col in gdf.columns:
        name = str(col).lower()
        if any(k in name for k in ["height", "level", "floor", "elev", "zmax", "z_max"]):
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return vals, str(col)

    return pd.Series([math.nan] * len(gdf)), "NOT_FOUND"


def calculate_area_if_missing(gdf: Any) -> Any:
    """Add footprint_area_m2 if not available."""
    gdf = gdf.copy()
    if "footprint_area_m2" in gdf.columns:
        try:
            import pandas as pd  # type: ignore
            gdf["footprint_area_m2"] = pd.to_numeric(gdf["footprint_area_m2"], errors="coerce")
        except Exception:
            pass
        return gdf

    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf["footprint_area_m2"] = gdf_utm.geometry.area.to_numpy()
    except Exception as e:
        print(f"[WARN] Could not calculate footprint_area_m2: {e}")
        gdf["footprint_area_m2"] = math.nan

    return gdf


def building_height_report_from_file(
    source_file: Path,
    source_name: str,
    dst_root: Path,
    mods: dict[str, Any | None],
) -> dict[str, Any] | None:
    """Read building GPKG/GeoJSON and save height/area/volume values + summary."""
    gpd = mods.get("gpd")
    pd = mods.get("pd")
    if gpd is None or pd is None:
        return None
    if source_file.suffix.lower() not in {".gpkg", ".geojson", ".shp"}:
        return None

    try:
        gdf = gpd.read_file(source_file)
    except Exception as e:
        print(f"[WARN] Cannot read building source {source_file}: {e}")
        return None

    if gdf.empty:
        return {
            "source": source_name,
            "file": str(source_file.relative_to(dst_root)) if source_file.is_relative_to(dst_root) else str(source_file),
            "n_buildings": 0,
            "height_column_used": "EMPTY",
        }

    height_m, height_col = get_numeric_height_series(gdf, source_name, mods)
    gdf = calculate_area_if_missing(gdf)

    height_m = pd.to_numeric(height_m, errors="coerce")
    area_m2 = pd.to_numeric(gdf["footprint_area_m2"], errors="coerce")
    volume_m3 = area_m2 * height_m

    valid_height = height_m.replace([float("inf"), float("-inf")], pd.NA).dropna()
    valid_area = area_m2.replace([float("inf"), float("-inf")], pd.NA).dropna()
    valid_volume = volume_m3.replace([float("inf"), float("-inf")], pd.NA).dropna()

    rel_file = str(source_file.relative_to(dst_root)) if str(source_file).startswith(str(dst_root)) else str(source_file)
    summary = {
        "source": source_name,
        "file": rel_file,
        "n_buildings": int(len(gdf)),
        "height_column_used": height_col,
        "height_valid_count": int(valid_height.count()),
        "height_missing_count": int(len(gdf) - valid_height.count()),
        "height_min_m": float(valid_height.min()) if len(valid_height) else "",
        "height_mean_m": float(valid_height.mean()) if len(valid_height) else "",
        "height_median_m": float(valid_height.median()) if len(valid_height) else "",
        "height_max_m": float(valid_height.max()) if len(valid_height) else "",
        "footprint_area_valid_count": int(valid_area.count()),
        "footprint_area_total_m2": float(valid_area.sum()) if len(valid_area) else "",
        "footprint_area_mean_m2": float(valid_area.mean()) if len(valid_area) else "",
        "footprint_area_median_m2": float(valid_area.median()) if len(valid_area) else "",
        "footprint_area_max_m2": float(valid_area.max()) if len(valid_area) else "",
        "volume_total_m3": float(valid_volume.sum()) if len(valid_volume) else "",
        "volume_mean_m3": float(valid_volume.mean()) if len(valid_volume) else "",
        "volume_median_m3": float(valid_volume.median()) if len(valid_volume) else "",
        "volume_max_m3": float(valid_volume.max()) if len(valid_volume) else "",
    }

    # Save per-building table.
    out_dir = dst_root / "gba" if source_name.upper() == "GBA" else dst_root / "obm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_table = pd.DataFrame({
        "source": source_name,
        "height_m": height_m,
        "footprint_area_m2": area_m2,
        "volume_m3": volume_m3,
    })

    # Add centroid for easy scatter checking.
    try:
        gdf_for_centroid = gdf.copy()
        if gdf_for_centroid.crs is None:
            gdf_for_centroid = gdf_for_centroid.set_crs("EPSG:4326")
        utm_crs = gdf_for_centroid.estimate_utm_crs()
        cent = gdf_for_centroid.to_crs(utm_crs).geometry.centroid
        cent = gpd.GeoSeries(cent, crs=utm_crs).to_crs("EPSG:4326")
        out_table["centroid_lon"] = cent.x.to_numpy()
        out_table["centroid_lat"] = cent.y.to_numpy()
    except Exception:
        pass

    out_csv = out_dir / f"{source_name.lower()}_height_area_volume_values.csv"
    out_table.to_csv(out_csv, index=False)
    print(f"[OK] Saved {source_name} height/area/volume values: {out_csv}")

    return summary


# ============================================================
# PLOTTING HELPERS
# ============================================================

def ensure_fig_dirs(dst_root: Path) -> dict[str, Path]:
    base = dst_root / FIGURES_SUBDIR
    dirs = {
        "base": base,
        "summary": base / "00_summary",
        "osm": base / "01_osm",
        "gba": base / "02_gba",
        "otp": base / "03_opentopography",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def find_aoi_gdf(dst_root: Path, mods: dict[str, Any | None]) -> Any | None:
    gpd = mods.get("gpd")
    if gpd is None:
        return None
    candidates = []
    for pattern in ["metadata/*aoi*.gpkg", "metadata/*aoi*.geojson", "metadata/*polygon*.gpkg", "**/hoalac_polygon.gpkg", "**/study_area_aoi.gpkg"]:
        candidates.extend(dst_root.glob(pattern))
    for p in candidates:
        try:
            gdf = gpd.read_file(p)
            if not gdf.empty:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                return gdf.to_crs("EPSG:4326")
        except Exception:
            pass
    return None


def choose_numeric_column(gdf: Any, preferred: list[str], mods: dict[str, Any | None]) -> str | None:
    pd = mods.get("pd")
    if pd is None or gdf.empty:
        return None

    colmap = {str(c).lower(): c for c in gdf.columns}
    for name in preferred:
        key = name.lower()
        if key in colmap:
            col = colmap[key]
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return str(col)

    # Fallback to any height-like numeric column.
    for col in gdf.columns:
        name = str(col).lower()
        if any(k in name for k in ["z_max", "zmax", "height", "agl", "elevation", "road_code", "range", "samples"]):
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return str(col)

    return None


def plot_empty_message(out_png: Path, title: str, message: str, mods: dict[str, Any | None]) -> None:
    plt = mods.get("plt")
    if plt is None:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_vector_file(
    path: Path,
    out_png: Path,
    title: str,
    value_fields: list[str],
    aoi_gdf: Any | None,
    mods: dict[str, Any | None],
) -> None:
    gpd = mods.get("gpd")
    plt = mods.get("plt")
    pd = mods.get("pd")
    if gpd is None or plt is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            plot_empty_message(out_png, title, "Layer exists but contains no feature.", mods)
            return
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    except Exception as e:
        plot_empty_message(out_png, title, f"Could not read vector layer:\n{e}", mods)
        return

    if len(gdf) > MAX_VECTOR_PLOT_FEATURES:
        gdf_plot = gdf.sample(MAX_VECTOR_PLOT_FEATURES, random_state=1)
        subtitle = f"Showing random {MAX_VECTOR_PLOT_FEATURES:,} of {len(gdf):,} features"
    else:
        gdf_plot = gdf
        subtitle = f"Features: {len(gdf):,}"

    value_col = choose_numeric_column(gdf_plot, value_fields, mods)

    fig, ax = plt.subplots(figsize=(10, 9))
    if aoi_gdf is not None:
        try:
            aoi_gdf.boundary.plot(ax=ax, linewidth=1.6)
        except Exception:
            pass

    try:
        if value_col is not None and pd is not None:
            gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors="coerce")
            gdf_plot.plot(ax=ax, column=value_col, legend=True, markersize=18, linewidth=1.2)
            cnote = f"Color: {value_col}"
        else:
            gdf_plot.plot(ax=ax, markersize=18, linewidth=1.2)
            cnote = "No numeric height/AGL field detected"
    except Exception:
        # Some mixed geometry collections fail with column plot. Fallback plain plot.
        gdf_plot.plot(ax=ax, markersize=18, linewidth=1.2)
        cnote = "Fallback plot"

    ax.set_title(f"{title}\n{subtitle}; {cnote}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_xyz_file(
    path: Path,
    out_png: Path,
    title: str,
    aoi_gdf: Any | None,
    mods: dict[str, Any | None],
) -> None:
    pd = mods.get("pd")
    plt = mods.get("plt")
    if pd is None or plt is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as e:
        plot_empty_message(out_png, title, f"Could not read XYZ:\n{e}", mods)
        return

    if df.empty or df.shape[1] < 2:
        plot_empty_message(out_png, title, "XYZ file is empty or has fewer than 2 columns.", mods)
        return

    if len(df) > MAX_VECTOR_PLOT_FEATURES:
        df_plot = df.sample(MAX_VECTOR_PLOT_FEATURES, random_state=1)
        subtitle = f"Showing random {MAX_VECTOR_PLOT_FEATURES:,} of {len(df):,} rows"
    else:
        df_plot = df
        subtitle = f"Rows: {len(df):,}"

    x = pd.to_numeric(df_plot.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df_plot.iloc[:, 1], errors="coerce")
    mask = x.notna() & y.notna()
    df_plot = df_plot[mask]
    x = x[mask]
    y = y[mask]

    value = None
    value_label = ""
    if df_plot.shape[1] >= 3:
        value = pd.to_numeric(df_plot.iloc[:, 2], errors="coerce")
        if value.notna().sum() == 0:
            value = None
        else:
            value_label = "col_2"

    fig, ax = plt.subplots(figsize=(10, 9))
    if aoi_gdf is not None:
        try:
            aoi_gdf.boundary.plot(ax=ax, linewidth=1.6)
        except Exception:
            pass

    if value is not None:
        sc = ax.scatter(x, y, c=value, s=8)
        fig.colorbar(sc, ax=ax, shrink=0.72, label=value_label)
        cnote = f"Color: {value_label}"
    else:
        ax.scatter(x, y, s=8)
        cnote = "No numeric value column"

    ax.set_title(f"{title}\n{subtitle}; {cnote}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_raster_file(path: Path, out_png: Path, title: str, aoi_gdf: Any | None, mods: dict[str, Any | None]) -> None:
    rasterio = mods.get("rasterio")
    np = mods.get("np")
    plt = mods.get("plt")
    if rasterio is None or np is None or plt is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan

            # Downsample for plotting if necessary.
            total_pixels = arr.shape[0] * arr.shape[1]
            if total_pixels > MAX_RASTER_PLOT_PIXELS:
                scale = math.ceil(math.sqrt(total_pixels / MAX_RASTER_PLOT_PIXELS))
                arr_plot = arr[::scale, ::scale]
                transform = src.transform * src.transform.scale(scale, scale)
            else:
                arr_plot = arr
                transform = src.transform

            left, top = transform * (0, 0)
            right, bottom = transform * (arr_plot.shape[1], arr_plot.shape[0])
            extent = [left, right, bottom, top]

            finite = arr_plot[np.isfinite(arr_plot)]
            if finite.size == 0:
                plot_empty_message(out_png, title, "Raster has no finite values.", mods)
                return
    except Exception as e:
        plot_empty_message(out_png, title, f"Could not read raster:\n{e}", mods)
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr_plot, extent=extent, origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.75, label="value")

    # AOI overlay only if raster is already geographic; skip otherwise to avoid CRS mismatch.
    if aoi_gdf is not None:
        try:
            # If values look lon/lat, overlay AOI.
            if -180 <= left <= 180 and -180 <= right <= 180 and -90 <= bottom <= 90 and -90 <= top <= 90:
                aoi_gdf.boundary.plot(ax=ax, linewidth=1.6)
        except Exception:
            pass

    ax.set_title(
        f"{title}\n"
        f"min={float(np.nanmin(finite)):.2f}, mean={float(np.nanmean(finite)):.2f}, max={float(np.nanmax(finite)):.2f}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_availability_summary(summary_rows: list[dict[str, Any]], fig_dirs: dict[str, Path], mods: dict[str, Any | None]) -> None:
    plt = mods.get("plt")
    if plt is None:
        return

    labels = [str(r["layer_name"]) for r in summary_rows]
    counts = [int(r.get("file_count", 0) or 0) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(labels))))
    y = list(range(len(labels)))
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("File count")
    ax.set_title("Scenario-1 obstacle input layer availability")
    for yi, val in zip(y, counts):
        ax.text(val + 0.05, yi, str(val), va="center")
    fig.tight_layout()
    out_png = fig_dirs["summary"] / "00_layer_availability_counts.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def find_first_file(paths: list[Path], keywords: list[str], suffixes: set[str]) -> Path | None:
    for p in paths:
        name = p.name.lower()
        if p.suffix.lower() in suffixes and all(k.lower() in name for k in keywords):
            return p
    for p in paths:
        if p.suffix.lower() in suffixes:
            return p
    return None


def make_layer_figures(
    dst_root: Path,
    copied_by_layer: dict[str, list[Path]],
    summary_rows: list[dict[str, Any]],
    mods: dict[str, Any | None],
) -> None:
    if not MAKE_FIGURES or DRY_RUN:
        return
    if mods.get("plt") is None:
        print("[WARN] matplotlib is not available. Skip diagnostic figures.")
        return

    fig_dirs = ensure_fig_dirs(dst_root)
    aoi_gdf = find_aoi_gdf(dst_root, mods)

    plot_availability_summary(summary_rows, fig_dirs, mods)

    # OSM / OpenInfraMap vector and XYZ plots.
    vector_specs = [
        ("osm_powerlines_agl", "Power lines / cables", ["z_max_agl_m", "height_m", "zmax", "z_max", "range"]),
        ("osm_power_poles_towers_agl", "Power towers / poles", ["height_m", "z_max_agl_m", "zmax", "z_max"]),
        ("osm_road_centerline", "Road center lines / road points", ["road_code", "highway_code", "class_code"]),
        ("osm_traffic_lights_agl", "Traffic lights / traffic lines", ["height_m", "z_max_agl_m", "zmax", "z_max"]),
        ("osm_street_lamps_agl_optional", "Street lamps", ["height_m", "z_max_agl_m", "zmax", "z_max"]),
    ]

    for layer_name, title, fields in vector_specs:
        files = copied_by_layer.get(layer_name, [])
        if not files:
            plot_empty_message(fig_dirs["osm"] / f"{layer_name}.png", title, "No copied file found for this layer.", mods)
            continue
        # Create one plot per useful file to check all available representations.
        for i, p in enumerate(files, 1):
            clean_name = p.stem.replace(" ", "_")[:80]
            out_png = fig_dirs["osm"] / f"{layer_name}_{i:02d}_{clean_name}.png"
            if p.suffix.lower() in {".gpkg", ".geojson", ".shp"}:
                plot_vector_file(p, out_png, f"{title}: {p.name}", fields, aoi_gdf, mods)
            elif p.suffix.lower() == ".xyz":
                plot_xyz_file(p, out_png, f"{title}: {p.name}", aoi_gdf, mods)

    # GBA/OBM building plots.
    for layer_name, source_name in [("gba_building_footprint_height", "GBA"), ("obm_building_optional", "OBM")]:
        files = copied_by_layer.get(layer_name, [])
        for i, p in enumerate(files, 1):
            if p.suffix.lower() in {".gpkg", ".geojson", ".shp"}:
                out_png = fig_dirs["gba"] / f"{source_name.lower()}_{i:02d}_{p.stem[:80]}_height_map.png"
                plot_vector_file(
                    p,
                    out_png,
                    f"{source_name} building footprint/height: {p.name}",
                    ["height_m", "height", "building_height", "mean_height", "z_max", "zmax"],
                    aoi_gdf,
                    mods,
                )
            elif p.suffix.lower() == ".xyz":
                out_png = fig_dirs["gba"] / f"{source_name.lower()}_{i:02d}_{p.stem[:80]}_xyz.png"
                plot_xyz_file(p, out_png, f"{source_name} building XYZ: {p.name}", aoi_gdf, mods)

    # OTP raster/XYZ plots.
    otp_files = copied_by_layer.get("otp_dem_elevation", [])
    for i, p in enumerate(otp_files, 1):
        clean_name = p.stem.replace(" ", "_")[:90]
        if p.suffix.lower() == ".tif":
            out_png = fig_dirs["otp"] / f"otp_{i:02d}_{clean_name}.png"
            plot_raster_file(p, out_png, f"OpenTopography DEM/elevation: {p.name}", aoi_gdf, mods)
        elif p.suffix.lower() == ".xyz":
            out_png = fig_dirs["otp"] / f"otp_{i:02d}_{clean_name}_xyz.png"
            plot_xyz_file(p, out_png, f"OpenTopography DEM/elevation XYZ: {p.name}", aoi_gdf, mods)

    print(f"[OK] Diagnostic figures saved under: {fig_dirs['base']}")


def plot_building_height_histograms(
    dst_root: Path,
    building_summary_rows: list[dict[str, Any]],
    mods: dict[str, Any | None],
) -> None:
    pd = mods.get("pd")
    plt = mods.get("plt")
    if pd is None or plt is None or DRY_RUN or not building_summary_rows:
        return

    fig_dirs = ensure_fig_dirs(dst_root)
    for source in ["GBA", "OBM"]:
        csv_path = (dst_root / "gba" / "gba_height_area_volume_values.csv") if source == "GBA" else (dst_root / "obm" / "obm_height_area_volume_values.csv")
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty or "height_m" not in df.columns:
            continue
        height = pd.to_numeric(df["height_m"], errors="coerce").dropna()
        area = pd.to_numeric(df["footprint_area_m2"], errors="coerce").dropna() if "footprint_area_m2" in df.columns else pd.Series(dtype=float)
        volume = pd.to_numeric(df["volume_m3"], errors="coerce").dropna() if "volume_m3" in df.columns else pd.Series(dtype=float)

        for values, label, fname in [
            (height, "Building height (m)", f"{source.lower()}_building_height_histogram.png"),
            (area, "Footprint area (m²)", f"{source.lower()}_building_area_histogram.png"),
            (volume, "Building volume (m³)", f"{source.lower()}_building_volume_histogram.png"),
        ]:
            if len(values) == 0:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.hist(values, bins=40)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{source} {label}\nN={len(values):,}, mean={values.mean():.2f}, median={values.median():.2f}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            out_png = fig_dirs["gba"] / fname
            fig.savefig(out_png, dpi=220)
            plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    src_root = resolve_user_path(SOURCE_DIR)
    dst_root = resolve_user_path(DEST_DIR)

    print("\n========== COLLECT SCENARIO-1 OBSTACLE INPUT DATA ==========")
    print(f"Script dir:    {script_base_dir()}")
    print(f"Source root:   {src_root}")
    print(f"Destination:   {dst_root}")
    print(f"Figures:       {dst_root / FIGURES_SUBDIR}")
    print(f"Dry run:       {DRY_RUN}")

    if not src_root.exists():
        raise FileNotFoundError(
            "\n[ERROR] Source folder does not exist:\n"
            f"  {src_root}\n\n"
            "Expected by user request:\n"
            "  ../downloaddata/output/01_HoaLac_studies_area/\n\n"
            "Put this script in make_model/ or edit SOURCE_DIR at the top of the script."
        )

    if CLEAN_DESTINATION_FIRST and dst_root.exists() and not DRY_RUN:
        print(f"\n[INFO] Removing old destination: {dst_root}")
        shutil.rmtree(dst_root)

    if not DRY_RUN:
        dst_root.mkdir(parents=True, exist_ok=True)
        ensure_fig_dirs(dst_root)

    mods = optional_imports()

    specs = LAYER_SPECS if COPY_METADATA else [s for s in LAYER_SPECS if s.layer_name != "metadata_aoi"]

    manifest_rows: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    copied_by_layer: dict[str, list[Path]] = {}

    for spec in specs:
        print(f"\n[INFO] Searching layer: {spec.layer_name}")
        matches = find_matches(src_root, spec.patterns, spec.layer_name)

        copied_destinations: list[Path] = []
        total_size = 0
        total_features = 0
        all_height_cols: set[str] = set()
        inspection_notes: list[str] = []

        if not matches:
            print(f"  [MISSING] {spec.layer_name}")
        else:
            print(f"  [FOUND] {len(matches)} file(s)")

        for src in matches:
            src_size = src.stat().st_size
            total_size += src_size
            dst, copy_status = copy_file(src, src_root, dst_root, spec.destination_subdir, spec.layer_name, mods)
            copied_destinations.append(dst)
            copied_by_layer.setdefault(spec.layer_name, []).append(dst)

            rel_src = src.relative_to(src_root)
            rel_dst = dst.relative_to(dst_root)

            manifest_rows.append({
                "layer_name": spec.layer_name,
                "status": copy_status,
                "source": str(rel_src),
                "destination": str(rel_dst),
                "size_bytes": src_size,
                "size_human": human_size(src_size),
                "voxel_role": spec.voxel_role,
            })

            print(f"    [{copy_status.upper()}] {rel_src} -> {rel_dst}")

            inspect_target = src if DRY_RUN else dst
            info = inspect_file(inspect_target, mods=mods)
            info.update({
                "layer_name": spec.layer_name,
                "relative_file": str(rel_dst),
                "voxel_role": spec.voxel_role,
            })
            inspection_rows.append(info)

            try:
                if info.get("feature_count") not in {"", None}:
                    total_features += int(float(info["feature_count"]))
            except Exception:
                pass

            if info.get("height_agl_columns"):
                for col in str(info["height_agl_columns"]).split(";"):
                    if col:
                        all_height_cols.add(col)

            if info.get("note"):
                inspection_notes.append(str(info["note"]))

        status = status_from_count(len(matches), spec.required)
        summary_rows.append({
            "layer_name": spec.layer_name,
            "status": status,
            "required": spec.required,
            "file_count": len(matches),
            "total_size_bytes": total_size,
            "total_size_human": human_size(total_size),
            "inspected_feature_or_row_count": total_features if total_features else "",
            "detected_height_agl_columns": ";".join(sorted(all_height_cols)),
            "destination_subdir": spec.destination_subdir,
            "voxel_role": spec.voxel_role,
            "note": spec.note,
            "copied_files": ";".join(str(p.relative_to(dst_root)) for p in copied_destinations),
            "inspection_notes": " | ".join(inspection_notes[:8]),
        })

    # --------------------------------------------------------
    # Building height/area/volume report for GBA and optional OBM
    # --------------------------------------------------------
    building_summary_rows: list[dict[str, Any]] = []
    for layer_name, source_name in [("gba_building_footprint_height", "GBA"), ("obm_building_optional", "OBM")]:
        for p in copied_by_layer.get(layer_name, []):
            if p.suffix.lower() in {".gpkg", ".geojson", ".shp"}:
                out = building_height_report_from_file(p, source_name, dst_root, mods)
                if out is not None:
                    building_summary_rows.append(out)

    building_summary_file = dst_root / "scenario1_building_height_summary.csv"
    if building_summary_rows:
        write_csv(
            building_summary_file,
            building_summary_rows,
            fieldnames=[
                "source", "file", "n_buildings", "height_column_used",
                "height_valid_count", "height_missing_count",
                "height_min_m", "height_mean_m", "height_median_m", "height_max_m",
                "footprint_area_valid_count", "footprint_area_total_m2", "footprint_area_mean_m2",
                "footprint_area_median_m2", "footprint_area_max_m2",
                "volume_total_m3", "volume_mean_m3", "volume_median_m3", "volume_max_m3",
            ],
        )
        print(f"[OK] Saved building height summary: {building_summary_file}")

    # --------------------------------------------------------
    # Write manifests and summaries
    # --------------------------------------------------------
    manifest_file = dst_root / "scenario1_obstacle_input_manifest.csv"
    summary_file = dst_root / "scenario1_obstacle_layer_summary.csv"
    inspection_file = dst_root / "scenario1_obstacle_file_inspection.csv"
    txt_summary_file = dst_root / "scenario1_obstacle_layer_summary.txt"
    info_file = dst_root / "copy_info.txt"

    write_csv(
        manifest_file,
        manifest_rows,
        fieldnames=[
            "layer_name", "status", "source", "destination",
            "size_bytes", "size_human", "voxel_role",
        ],
    )

    write_csv(
        summary_file,
        summary_rows,
        fieldnames=[
            "layer_name", "status", "required", "file_count",
            "total_size_bytes", "total_size_human",
            "inspected_feature_or_row_count", "detected_height_agl_columns",
            "destination_subdir", "voxel_role", "note",
            "copied_files", "inspection_notes",
        ],
    )

    write_csv(
        inspection_file,
        inspection_rows,
        fieldnames=[
            "layer_name", "relative_file", "file", "suffix",
            "size_bytes", "size_human", "feature_count",
            "raster_shape", "raster_crs", "columns",
            "height_agl_columns", "min_value", "max_value", "mean_value",
            "note", "voxel_role",
        ],
    )

    if not DRY_RUN:
        with open(info_file, "w", encoding="utf-8") as f:
            f.write("Scenario-1 obstacle input collection\n")
            f.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Source root: {src_root}\n")
            f.write(f"Destination: {dst_root}\n")
            f.write(f"Figures: {dst_root / FIGURES_SUBDIR}\n")
            f.write("Requested source path: ../downloaddata/output/01_HoaLac_studies_area/\n")
            f.write("Requested destination path: input/data_senario1/\n")
            f.write("\nCollected layer groups:\n")
            for row in summary_rows:
                f.write(f"  - {row['layer_name']}: {row['status']} ({row['file_count']} files)\n")

    with open(txt_summary_file, "w", encoding="utf-8") as f:
        f.write("SCENARIO-1 OBSTACLE INPUT LAYER SUMMARY\n")
        f.write("=======================================\n")
        f.write(f"Source: {src_root}\n")
        f.write(f"Destination: {dst_root}\n")
        f.write(f"Figures: {dst_root / FIGURES_SUBDIR}\n")
        f.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for row in summary_rows:
            f.write(f"Layer: {row['layer_name']}\n")
            f.write(f"  Status      : {row['status']}\n")
            f.write(f"  Required    : {row['required']}\n")
            f.write(f"  Files       : {row['file_count']}\n")
            f.write(f"  Size        : {row['total_size_human']}\n")
            f.write(f"  Role        : {row['voxel_role']}\n")
            f.write(f"  Height cols : {row['detected_height_agl_columns']}\n")
            f.write(f"  Destination : {row['destination_subdir']}\n")
            f.write(f"  Note        : {row['note']}\n")
            if row["copied_files"]:
                f.write("  Copied files:\n")
                for p in str(row["copied_files"]).split(";"):
                    if p:
                        f.write(f"    - {p}\n")
            f.write("\n")

    # --------------------------------------------------------
    # Create plots after all files are copied and summaries are known
    # --------------------------------------------------------
    make_layer_figures(dst_root, copied_by_layer, summary_rows, mods)
    plot_building_height_histograms(dst_root, building_summary_rows, mods)

    # Console summary
    print("\n========== SCENARIO-1 LAYER SUMMARY ==========")
    missing_required = []
    for row in summary_rows:
        print(
            f"{row['layer_name']:<35} "
            f"{row['status']:<18} "
            f"files={row['file_count']:<3} "
            f"height_cols={row['detected_height_agl_columns']}"
        )
        if row["status"] == "missing_required":
            missing_required.append(str(row["layer_name"]))

    print("\nOutput files:")
    print(f"  Manifest:        {manifest_file}")
    print(f"  Summary:         {summary_file}")
    print(f"  Inspection:      {inspection_file}")
    print(f"  Building summary:{building_summary_file}")
    print(f"  Text:            {txt_summary_file}")
    print(f"  Figures:         {dst_root / FIGURES_SUBDIR}")

    if missing_required:
        print("\n[WARN] Missing required layer groups:")
        for name in missing_required:
            print(f"  - {name}")
        print("\nThis may be okay if that data source was not downloaded yet or if filenames differ.")
        print("Check scenario1_obstacle_layer_summary.csv and update LAYER_SPECS patterns if needed.")
        if STRICT_REQUIRED_LAYERS:
            raise FileNotFoundError("Required Scenario-1 obstacle layers are missing.")

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
