#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download GIS layers for Hoa Lac mixed-model preparation:

1. OpenStreetMap:
   - road network
   - road classes
   - road nodes
   - landuse / natural / water / railway / building / amenity features

2. OpenTopography:
   - DEM GeoTIFF
   - projected DEM
   - slope
   - aspect
   - hillshade
   - roughness
   - TPI

3. OpenBuildingMap optional comparison data:
   - downloads OBM tiles intersecting AOI if requested
   - clips OBM buildings to AOI

Output:
    output/01_HoaLac_studies_area

Author: ChatGPT
Updated: OSM + OpenTopography + optional OpenBuildingMap
"""

from __future__ import annotations

import os
import sys
import bz2
import json
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio
import rasterio.mask
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from tqdm import tqdm


try:
    import osmnx as ox
except ImportError:
    raise ImportError(
        "Missing osmnx. Install with:\n"
        "conda install -c conda-forge osmnx"
    )


# =============================================================================
# USER SETTINGS
# =============================================================================

OUT_DIR = Path("output/01_HoaLac_studies_area")

# Use your exact Hoa Lac polygon, lon/lat
USE_HOALAC_POLYGON = True

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

# Fallback only, not used if USE_HOALAC_POLYGON = True
USE_PLACE_NAME = False
PLACE_NAME = "Hoa Lac Hi-Tech Park, Hanoi, Vietnam"

BBOX = {
    "west": 105.47,
    "south": 20.965,
    "east": 105.62,
    "north": 21.095,
}

# OpenTopography DEM type.
# Common options:
#   SRTMGL1  = SRTM 30 m
#   SRTMGL3  = SRTM 90 m
#   AW3D30   = ALOS World 3D 30 m
#   COP30    = Copernicus DEM 30 m
#   COP90    = Copernicus DEM 90 m
#   NASADEM  = NASADEM
OPENTOPO_DEM_TYPE = "SRTMGL1"
# OPENTOPO_API_KEY = os.environ.get("OPENTOPOGRAPHY_API_KEY", "").strip()
OPENTOPO_API_KEY = "9b13849a6bd3486c4ed72960d230a366" # Replace with your OpenTopography API key if required

# IMPORTANT:
# Your OSMnx version does not support "all_private".
# Valid safe value: "all"
OSM_NETWORK_TYPE = "all"

# NOTE:
# GlobalBuildingAtlas LoD1 is intentionally NOT processed here.
# It is prepared by the separate GBA script and saved to the same destination:
# output/01_HoaLac_studies_area/globalbuildingatlas_lod1/

# OpenBuildingMap optional comparison data
# This is NOT the main building layer for mixed model; GBA LoD1 is prepared separately.
# OBM is downloaded for comparison/checking if requested.
DOWNLOAD_OPENBUILDINGMAP = True
OBM_MANIFEST_URL = "https://umap.openstreetmap.de/de/datalayer/81684/ac9788a9-c215-417d-a580-cb545acd78b9/"
OBM_MAX_TILES = None  # None = all OBM tiles intersecting AOI

DOWNLOAD_EXTRA_OSM_FEATURES = True

# Clip roads exactly to Hoa Lac AOI polygon
# True  = roads outside AOI are removed / cut at boundary
# False = keep original OSMnx road geometry
MASK_ROADS_TO_AOI = False

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, out_path: Path, timeout: int = 120) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        logging.info(f"Already exists: {out_path}")
        return out_path

    logging.info(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return out_path


def decompress_bz2(in_path: Path, out_path: Optional[Path] = None) -> Path:
    if out_path is None:
        if in_path.name.endswith(".bz2"):
            out_path = in_path.with_name(in_path.name[:-4])
        else:
            out_path = in_path.with_suffix("")

    if out_path.exists() and out_path.stat().st_size > 0:
        logging.info(f"Already decompressed: {out_path}")
        return out_path

    logging.info(f"Decompressing: {in_path} -> {out_path}")
    with bz2.open(in_path, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return out_path


def get_utm_crs_for_gdf(gdf: gpd.GeoDataFrame):
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        # Fallback for northern Vietnam, UTM zone 48N
        return "EPSG:32648"


def write_gdf(gdf: gpd.GeoDataFrame, out_path: Path, layer: Optional[str] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        logging.warning(f"Empty GeoDataFrame, skip writing: {out_path}")
        return

    if out_path.suffix.lower() == ".gpkg":
        gdf.to_file(out_path, layer=layer or out_path.stem, driver="GPKG")
    elif out_path.suffix.lower() in [".geojson", ".json"]:
        gdf.to_file(out_path, driver="GeoJSON")
    else:
        gdf.to_file(out_path)


# =============================================================================
# AOI
# =============================================================================

def make_aoi() -> gpd.GeoDataFrame:
    """
    Build AOI from user-defined Hoa Lac polygon.
    Fallback to place name or bbox only if requested.
    """
    if USE_HOALAC_POLYGON:
        from shapely.geometry import Polygon

        geom = Polygon(HOALAC_POLYGON)

        if not geom.is_valid:
            geom = geom.buffer(0)

        aoi = gpd.GeoDataFrame(
            {"name": ["HoaLac_HiTech_Park_polygon"]},
            geometry=[geom],
            crs="EPSG:4326",
        )

        logging.info("Using user-defined Hoa Lac polygon AOI.")
        return aoi

    if USE_PLACE_NAME:
        try:
            logging.info(f"Trying to geocode AOI: {PLACE_NAME}")
            place_gdf = ox.geocode_to_gdf(PLACE_NAME)
            place_gdf = place_gdf.to_crs("EPSG:4326")
            if not place_gdf.empty:
                place_gdf["name"] = PLACE_NAME
                logging.info("Using geocoded AOI boundary.")
                return place_gdf[["name", "geometry"]]
        except Exception as e:
            logging.warning(f"Place geocoding failed, using fallback bbox. Reason: {e}")

    geom = box(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])
    aoi = gpd.GeoDataFrame(
        {"name": ["HoaLac_bbox_fallback"]},
        geometry=[geom],
        crs="EPSG:4326",
    )

    logging.info("Using fallback bbox AOI.")
    return aoi


# =============================================================================
# OPENSTREETMAP
# =============================================================================

def classify_road(highway_value) -> str:
    """
    Simplify OSM highway tags into useful road classes.
    """
    if isinstance(highway_value, list):
        highway_value = highway_value[0] if highway_value else None

    if highway_value is None or pd.isna(highway_value):
        return "unknown"

    h = str(highway_value).lower()

    if h in ["motorway", "motorway_link", "trunk", "trunk_link"]:
        return "expressway_or_trunk"
    if h in ["primary", "primary_link"]:
        return "primary"
    if h in ["secondary", "secondary_link"]:
        return "secondary"
    if h in ["tertiary", "tertiary_link"]:
        return "tertiary"
    if h in ["residential", "living_street"]:
        return "residential"
    if h in ["service", "track"]:
        return "service_or_track"
    if h in ["unclassified", "road"]:
        return "unclassified"
    if h in ["path", "footway", "cycleway", "pedestrian", "steps", "bridleway"]:
        return "non_motorized"
    return h


def osmnx_features_from_polygon(poly, tags):
    """
    Compatibility wrapper for osmnx 1.x / 2.x.
    """
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(poly, tags)
    return ox.geometries_from_polygon(poly, tags)

def clip_roads_to_aoi(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clip road edges to AOI polygon if MASK_ROADS_TO_AOI=True.

    This removes road geometry outside the study area and recalculates
    clipped road length in meters.
    """
    if not MASK_ROADS_TO_AOI:
        logging.info("MASK_ROADS_TO_AOI=False, keep original OSM road geometries.")
        return edges, nodes

    logging.info("MASK_ROADS_TO_AOI=True, clipping OSM roads to AOI polygon...")

    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    edges_wgs84 = edges.to_crs("EPSG:4326").copy()
    nodes_wgs84 = nodes.to_crs("EPSG:4326").copy()

    n_edges_before = len(edges_wgs84)
    n_nodes_before = len(nodes_wgs84)

    # Clip road line geometries exactly to AOI polygon
    clipped_edges = gpd.clip(edges_wgs84, aoi_wgs84)

    clipped_edges = clipped_edges[
        clipped_edges.geometry.notna() &
        (~clipped_edges.geometry.is_empty)
    ].copy()

    # Keep only line geometries
    clipped_edges = clipped_edges[
        clipped_edges.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()

    # Recalculate road length after clipping
    if not clipped_edges.empty:
        try:
            utm_crs = get_utm_crs_for_gdf(aoi_wgs84)
            clipped_edges_utm = clipped_edges.to_crs(utm_crs)
            clipped_edges["length_m"] = clipped_edges_utm.geometry.length
        except Exception as e:
            logging.warning(f"Could not recalculate clipped road length: {e}")

    # Clip nodes too, for consistency
    clipped_nodes = gpd.clip(nodes_wgs84, aoi_wgs84)

    clipped_nodes = clipped_nodes[
        clipped_nodes.geometry.notna() &
        (~clipped_nodes.geometry.is_empty)
    ].copy()

    logging.info(
        f"Road edges clipped: {n_edges_before} -> {len(clipped_edges)}"
    )
    logging.info(
        f"Road nodes clipped: {n_nodes_before} -> {len(clipped_nodes)}"
    )

    return clipped_edges, clipped_nodes


def download_osm(aoi: gpd.GeoDataFrame, out_dir: Path) -> None:
    osm_dir = safe_mkdir(out_dir / "osm")
    roads_dir = safe_mkdir(osm_dir / "roads")
    extra_dir = safe_mkdir(osm_dir / "extra_features")

    try:
        polygon = aoi.geometry.union_all()
    except AttributeError:
        polygon = aoi.geometry.unary_union

    logging.info("Downloading OSM road network...")
    G = ox.graph_from_polygon(
        polygon,
        network_type=OSM_NETWORK_TYPE,
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

    # Road class before clipping
    edges["road_class"] = edges.get("highway", None).apply(classify_road)

    # Initial OSMnx length
    edges["length_m"] = pd.to_numeric(edges.get("length", np.nan), errors="coerce")

    # Optional exact AOI clipping
    edges, nodes = clip_roads_to_aoi(
        edges=edges,
        nodes=nodes,
        aoi=aoi,
    )

    # Rebuild road class if needed after clipping
    if "road_class" not in edges.columns and "highway" in edges.columns:
        edges["road_class"] = edges["highway"].apply(classify_road)

    write_gdf(edges, roads_dir / "osm_roads_edges.gpkg", layer="roads")
    write_gdf(edges, roads_dir / "osm_roads_edges.geojson")
    write_gdf(nodes, roads_dir / "osm_roads_nodes.gpkg", layer="nodes")

    class_summary = (
        edges.groupby("road_class", dropna=False)
        .agg(
            n_segments=("road_class", "size"),
            total_length_m=("length_m", "sum"),
        )
        .reset_index()
        .sort_values("total_length_m", ascending=False)
    )
    class_summary["total_length_km"] = class_summary["total_length_m"] / 1000.0
    class_summary.to_csv(roads_dir / "osm_road_class_summary.csv", index=False)

    logging.info(f"OSM roads saved: {roads_dir}")

    if DOWNLOAD_EXTRA_OSM_FEATURES:
        logging.info("Downloading extra OSM features...")

        extra_tags = {
            "building": True,
            "landuse": True,
            "natural": True,
            "water": True,
            "waterway": True,
            "railway": True,
            "amenity": True,
            "aeroway": True,
            "man_made": True,
            "leisure": True,
            "barrier": True,
        }

        try:
            feats = osmnx_features_from_polygon(polygon, extra_tags)
            feats = feats.reset_index()
            feats = feats.set_geometry("geometry")
            feats = feats.to_crs("EPSG:4326")

            write_gdf(feats, extra_dir / "osm_extra_features.gpkg", layer="osm_features")
            write_gdf(feats, extra_dir / "osm_extra_features.geojson")

            cols = [c for c in extra_tags.keys() if c in feats.columns]
            summaries = []
            for c in cols:
                tmp = (
                    feats[c]
                    .dropna()
                    .astype(str)
                    .value_counts()
                    .rename_axis(c)
                    .reset_index(name="count")
                )
                tmp.insert(0, "tag", c)
                summaries.append(tmp.rename(columns={c: "value"}))

            if summaries:
                pd.concat(summaries, ignore_index=True).to_csv(
                    extra_dir / "osm_extra_tag_summary.csv",
                    index=False,
                )

            logging.info(f"Extra OSM features saved: {extra_dir}")

        except Exception as e:
            logging.warning(f"Extra OSM feature download failed: {e}")


# =============================================================================
# OPENTOPOGRAPHY + TERRAIN DERIVATIVES
# =============================================================================

def download_opentopography_dem(aoi: gpd.GeoDataFrame, out_dir: Path) -> Path:
    topo_dir = safe_mkdir(out_dir / "opentopography")

    bounds = aoi.to_crs("EPSG:4326").total_bounds
    west, south, east, north = bounds

    out_dem = topo_dir / f"opentopography_{OPENTOPO_DEM_TYPE}_dem_wgs84.tif"

    if out_dem.exists() and out_dem.stat().st_size > 0:
        logging.info(f"DEM already exists: {out_dem}")
        return out_dem

    params = {
        "demtype": OPENTOPO_DEM_TYPE,
        "south": f"{south:.8f}",
        "north": f"{north:.8f}",
        "west": f"{west:.8f}",
        "east": f"{east:.8f}",
        "outputFormat": "GTiff",
    }

    if OPENTOPO_API_KEY:
        params["API_Key"] = OPENTOPO_API_KEY
    else:
        logging.warning(
            "OPENTOPOGRAPHY_API_KEY is not set. "
            "The request may fail if OpenTopography requires authentication."
        )

    url = "https://portal.opentopography.org/API/globaldem"

    logging.info("Downloading DEM from OpenTopography...")
    r = requests.get(url, params=params, timeout=180)
    try:
        r.raise_for_status()
    except Exception:
        logging.error(f"OpenTopography response text:\n{r.text[:1000]}")
        raise

    content_type = r.headers.get("content-type", "")
    if "text" in content_type.lower() or r.content[:100].lower().startswith(b"<html"):
        logging.error(r.text[:2000])
        raise RuntimeError(
            "OpenTopography did not return a GeoTIFF. "
            "Check your API key, DEM type, and bbox."
        )

    with open(out_dem, "wb") as f:
        f.write(r.content)

    logging.info(f"DEM saved: {out_dem}")
    return out_dem


def reproject_raster_to_utm(in_tif: Path, out_tif: Path, target_crs) -> Path:
    if out_tif.exists() and out_tif.stat().st_size > 0:
        logging.info(f"Projected raster already exists: {out_tif}")
        return out_tif

    with rasterio.open(in_tif) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
                "nodata": src.nodata,
            }
        )

        with rasterio.open(out_tif, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    logging.info(f"Projected DEM saved: {out_tif}")
    return out_tif


def save_single_band_raster(
    out_path: Path,
    arr: np.ndarray,
    ref_profile: dict,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> None:
    profile = ref_profile.copy()

    # Avoid GDAL warnings:
    # BLOCKXSIZE/BLOCKYSIZE can only be used when TILED=YES.
    if not bool(profile.get("tiled", False)):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

    profile.update(
        {
            "count": 1,
            "dtype": dtype,
            "compress": "lzw",
            "nodata": nodata,
        }
    )

    arr2 = np.asarray(arr, dtype=dtype)
    arr2 = np.where(np.isfinite(arr2), arr2, nodata).astype(dtype)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr2, 1)


def compute_hillshade(dem: np.ndarray, transform: Affine, azimuth: float = 315, altitude: float = 45):
    xres = abs(transform.a)
    yres = abs(transform.e)

    dy, dx = np.gradient(dem, yres, xres)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)

    shaded = (
        np.sin(alt_rad) * np.sin(slope)
        + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    )

    return 255 * np.clip(shaded, 0, 1)


def moving_window_mean(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Simple 3x3 mean using numpy padding.
    Avoids scipy dependency.
    """
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr, dtype="float64")

    for i in range(size):
        for j in range(size):
            out += padded[i:i + arr.shape[0], j:j + arr.shape[1]]

    return out / float(size * size)


def moving_window_range(arr: np.ndarray, size: int = 3) -> np.ndarray:
    pad = size // 2
    padded = np.pad(arr, pad, mode="edge")
    stacks = []

    for i in range(size):
        for j in range(size):
            stacks.append(padded[i:i + arr.shape[0], j:j + arr.shape[1]])

    stack = np.stack(stacks, axis=0)
    return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)


def compute_terrain_products(dem_tif: Path, aoi: gpd.GeoDataFrame, out_dir: Path) -> None:
    topo_dir = safe_mkdir(out_dir / "opentopography")
    terrain_dir = safe_mkdir(topo_dir / "terrain_products")

    target_crs = get_utm_crs_for_gdf(aoi)
    dem_utm = topo_dir / f"opentopography_{OPENTOPO_DEM_TYPE}_dem_utm.tif"

    reproject_raster_to_utm(dem_tif, dem_utm, target_crs)

    with rasterio.open(dem_utm) as src:
        dem = src.read(1).astype("float64")
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    xres = abs(transform.a)
    yres = abs(transform.e)

    dy, dx = np.gradient(dem, yres, xres)

    slope_rad = np.arctan(np.sqrt(dx * dx + dy * dy))
    slope_deg = np.rad2deg(slope_rad)
    slope_pct = np.tan(slope_rad) * 100.0

    aspect = np.rad2deg(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, 360.0 + aspect, aspect)

    hillshade = compute_hillshade(dem, transform)
    roughness = moving_window_range(dem, size=3)
    tpi = dem - moving_window_mean(dem, size=3)

    save_single_band_raster(
        terrain_dir / "slope_degree.tif",
        slope_deg,
        profile,
    )
    save_single_band_raster(
        terrain_dir / "slope_percent.tif",
        slope_pct,
        profile,
    )
    save_single_band_raster(
        terrain_dir / "aspect_degree.tif",
        aspect,
        profile,
    )
    save_single_band_raster(
        terrain_dir / "hillshade.tif",
        hillshade,
        profile,
    )
    save_single_band_raster(
        terrain_dir / "roughness_3x3_m.tif",
        roughness,
        profile,
    )
    save_single_band_raster(
        terrain_dir / "tpi_3x3_m.tif",
        tpi,
        profile,
    )

    stats = pd.DataFrame(
        [
            ["dem_m", np.nanmin(dem), np.nanmax(dem), np.nanmean(dem), np.nanstd(dem)],
            ["slope_degree", np.nanmin(slope_deg), np.nanmax(slope_deg), np.nanmean(slope_deg), np.nanstd(slope_deg)],
            ["slope_percent", np.nanmin(slope_pct), np.nanmax(slope_pct), np.nanmean(slope_pct), np.nanstd(slope_pct)],
            ["roughness_3x3_m", np.nanmin(roughness), np.nanmax(roughness), np.nanmean(roughness), np.nanstd(roughness)],
            ["tpi_3x3_m", np.nanmin(tpi), np.nanmax(tpi), np.nanmean(tpi), np.nanstd(tpi)],
        ],
        columns=["layer", "min", "max", "mean", "std"],
    )
    stats.to_csv(terrain_dir / "terrain_statistics.csv", index=False)

    logging.info(f"Terrain products saved: {terrain_dir}")




# =============================================================================
# OPENBUILDINGMAP OPTIONAL COMPARISON DOWNLOAD
# =============================================================================

OBM_DIR = OUT_DIR / "openbuildingmap"
OBM_METADATA_DIR = OBM_DIR / "metadata"
OBM_RAW_DIR = OBM_DIR / "raw"
OBM_CLIPPED_DIR = OBM_DIR / "clipped"

for _obm_d in [OBM_METADATA_DIR, OBM_RAW_DIR, OBM_CLIPPED_DIR]:
    _obm_d.mkdir(parents=True, exist_ok=True)


def parse_obm_height_to_m(value, default_m=3.0):
    """
    Convert OBM GEM taxonomy height strings to approximate meters.

    Examples:
        HHT:10.0  -> 10.0 m
        H:2       -> 6.0 m, assuming 3 m/story
        HBET:1-3  -> 6.0 m, average stories * 3 m
        None/UNK  -> default_m
    """
    if value is None or pd.isna(value):
        return default_m

    txt = str(value).strip()
    if not txt or txt.upper() in ["UNK", "NULL", "NAN"]:
        return default_m

    # Some values can be combined, e.g. H:1+HHT:3.50.
    parts = txt.split("+")

    # Prefer explicit meter height.
    for part in parts:
        part = part.strip()
        if part.startswith("HHT:"):
            try:
                return float(part.split(":", 1)[1])
            except Exception:
                pass

    # Number of stories.
    for part in parts:
        part = part.strip()
        if part.startswith("H:"):
            try:
                return float(part.split(":", 1)[1]) * 3.0
            except Exception:
                pass

    # Range of stories.
    for part in parts:
        part = part.strip()
        if part.startswith("HBET:"):
            try:
                rng = part.split(":", 1)[1]
                if "-" in rng:
                    a, b = rng.split("-", 1)
                    return ((float(a) + float(b)) / 2.0) * 3.0
                return float(rng) * 3.0
            except Exception:
                pass

    return default_m


def download_obm_manifest() -> Path:
    """
    Download OBM GeoJSON manifest used by the official OBM download map.
    """
    out_manifest = OBM_METADATA_DIR / "obm_manifest.geojson"

    if out_manifest.exists() and out_manifest.stat().st_size > 1000:
        logging.info(f"OBM manifest already exists: {out_manifest}")
        return out_manifest

    logging.info(f"Downloading OBM manifest: {OBM_MANIFEST_URL}")
    downloaded = download_file(OBM_MANIFEST_URL, out_manifest)
    return downloaded


def select_obm_tiles_for_aoi(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Select OBM download tiles whose footprints intersect the AOI.
    """
    manifest_file = download_obm_manifest()

    manifest = gpd.read_file(manifest_file)
    if manifest.crs is None:
        manifest = manifest.set_crs("EPSG:4326")
    manifest = manifest.to_crs("EPSG:4326")

    aoi_wgs = aoi.to_crs("EPSG:4326")
    try:
        aoi_union = aoi_wgs.geometry.union_all()
    except AttributeError:
        aoi_union = aoi_wgs.geometry.unary_union

    selected = manifest[
        manifest.geometry.notna()
        & (~manifest.geometry.is_empty)
        & manifest.intersects(aoi_union)
    ].copy()

    if OBM_MAX_TILES is not None and len(selected) > int(OBM_MAX_TILES):
        logging.info(f"OBM_MAX_TILES={OBM_MAX_TILES}. Use first {OBM_MAX_TILES} intersecting OBM tiles.")
        selected = selected.head(int(OBM_MAX_TILES)).copy()

    selected.to_file(
        OBM_METADATA_DIR / "selected_obm_tiles.gpkg",
        layer="selected_obm_tiles",
        driver="GPKG",
    )

    selected.drop(columns="geometry").to_csv(
        OBM_METADATA_DIR / "selected_obm_tiles.csv",
        index=False,
    )

    logging.info(f"Selected OBM tiles intersecting AOI: {len(selected)}")
    return selected


def download_and_decompress_obm_tile(url: str) -> Path:
    """
    Download one OBM .gpkg.bz2 tile and return decompressed .gpkg path.
    """
    raw_name = Path(url.split("?")[0]).name
    if not raw_name:
        raw_name = "obm_tile.gpkg.bz2"

    raw_path = OBM_RAW_DIR / raw_name
    downloaded = download_file(url, raw_path)

    if downloaded.name.endswith(".bz2"):
        gpkg_path = downloaded.with_name(downloaded.name[:-4])
        if gpkg_path.exists() and gpkg_path.stat().st_size > 1000:
            logging.info(f"OBM tile already decompressed: {gpkg_path}")
            return gpkg_path

        logging.info(f"Decompressing OBM tile: {downloaded} -> {gpkg_path}")
        with bz2.open(downloaded, "rb") as src, open(gpkg_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return gpkg_path

    return downloaded


def read_obm_tile_clip(gpkg_path: Path, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Read and clip one OBM GPKG tile to AOI.
    """
    import fiona

    layers = fiona.listlayers(gpkg_path)
    preferred = ["buildings", "building", "OpenBuildingMap", "obm"]
    layer = None

    for cand in preferred:
        if cand in layers:
            layer = cand
            break

    if layer is None:
        layer = layers[0]

    bbox_wgs = tuple(aoi.to_crs("EPSG:4326").total_bounds)

    try:
        gdf = gpd.read_file(gpkg_path, layer=layer, bbox=bbox_wgs)
    except Exception:
        gdf = gpd.read_file(gpkg_path, layer=layer)

    if gdf.empty:
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(gdf, aoi.to_crs("EPSG:4326"))

    if clipped.empty:
        return clipped

    clipped["obm_source_file"] = gpkg_path.name

    # Add normalized useful columns for comparison with GBA.
    if "height" in clipped.columns:
        clipped["height_m"] = clipped["height"].apply(parse_obm_height_to_m)
    elif "height_m" not in clipped.columns:
        clipped["height_m"] = 3.0

    try:
        utm_crs = get_utm_crs_for_gdf(clipped)
        clipped_utm = clipped.to_crs(utm_crs)
        clipped["footprint_area_m2"] = clipped_utm.geometry.area.to_numpy()
        clipped["volume_m3"] = clipped["footprint_area_m2"] * pd.to_numeric(
            clipped["height_m"],
            errors="coerce",
        ).fillna(3.0)
        cent_wgs = gpd.GeoSeries(clipped_utm.geometry.centroid, crs=utm_crs).to_crs("EPSG:4326")
        clipped["centroid_lon"] = cent_wgs.x.to_numpy()
        clipped["centroid_lat"] = cent_wgs.y.to_numpy()
    except Exception as exc:
        logging.warning(f"Could not calculate OBM derived geometry columns: {exc}")

    return clipped


def process_openbuildingmap_optional(aoi: gpd.GeoDataFrame, out_dir: Path) -> None:
    """
    Download OpenBuildingMap tiles intersecting AOI and clip to study area.

    This is optional comparison data. The mixed-model building source remains
    GlobalBuildingAtlas LoD1 unless you explicitly change later scripts.
    """
    if not DOWNLOAD_OPENBUILDINGMAP:
        logging.info("DOWNLOAD_OPENBUILDINGMAP=False. Skip OBM optional download.")
        return

    logging.info("=" * 80)
    logging.info("START OPTIONAL OPENBUILDINGMAP DOWNLOAD")
    logging.info("=" * 80)

    out_gpkg = OBM_CLIPPED_DIR / "obm_buildings_hoalac_clipped.gpkg"

    if out_gpkg.exists() and out_gpkg.stat().st_size > 1000:
        logging.info(f"Existing clipped OBM file found. Skip OBM rebuild: {out_gpkg}")
        return

    selected = select_obm_tiles_for_aoi(aoi)

    if selected.empty:
        logging.warning("No OBM tile footprints intersect AOI.")
        return

    if "URL" not in selected.columns:
        raise RuntimeError("[ERROR] OBM manifest has no URL column.")

    parts = []

    for _, row in selected.iterrows():
        url = str(row["URL"])
        if not url or url.lower() == "nan":
            continue

        logging.info(f"OBM tile URL: {url}")

        try:
            gpkg_path = download_and_decompress_obm_tile(url)
            clipped = read_obm_tile_clip(gpkg_path, aoi)
            if not clipped.empty:
                parts.append(clipped)
        except Exception as exc:
            logging.warning(f"Could not process OBM tile {url}: {exc}")

    if not parts:
        logging.warning("No OBM buildings found after clipping selected tiles.")
        return

    obm = pd.concat(parts, ignore_index=True)
    obm = gpd.GeoDataFrame(obm, geometry="geometry", crs="EPSG:4326")

    if "id" in obm.columns:
        obm = obm.drop_duplicates(subset=["id", "obm_source_file"])
    else:
        obm = obm.drop_duplicates(subset=["geometry"])

    write_gdf(obm, out_gpkg, layer="buildings")
    write_gdf(obm, OBM_CLIPPED_DIR / "obm_buildings_hoalac_clipped.geojson")

    # Export simple centroid XYZ for quick PyGMT checks.
    if {"centroid_lon", "centroid_lat", "height_m"}.issubset(obm.columns):
        pd.DataFrame({
            "lon": obm["centroid_lon"],
            "lat": obm["centroid_lat"],
            "height_m": obm["height_m"],
        }).to_csv(
            OBM_CLIPPED_DIR / "obm_buildings_centroid_hoalac.xyz",
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )

    summary = {
        "n_buildings": int(len(obm)),
        "height_min_m": float(pd.to_numeric(obm.get("height_m", pd.Series(dtype=float)), errors="coerce").min()),
        "height_mean_m": float(pd.to_numeric(obm.get("height_m", pd.Series(dtype=float)), errors="coerce").mean()),
        "height_max_m": float(pd.to_numeric(obm.get("height_m", pd.Series(dtype=float)), errors="coerce").max()),
    }

    if "source_id" in obm.columns:
        for k, v in obm["source_id"].fillna("NULL").astype(str).value_counts().items():
            summary[f"source_id_{k}"] = int(v)

    pd.DataFrame([summary]).to_csv(OBM_CLIPPED_DIR / "obm_summary.csv", index=False)

    logging.info(f"OpenBuildingMap optional data saved: {OBM_CLIPPED_DIR}")


# =============================================================================
# PROJECT METADATA
# =============================================================================

def save_project_metadata(aoi: gpd.GeoDataFrame, out_dir: Path) -> None:
    meta_dir = safe_mkdir(out_dir / "metadata")

    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    bounds = aoi_wgs84.total_bounds

    write_gdf(aoi_wgs84, meta_dir / "study_area_aoi.gpkg", layer="aoi")
    write_gdf(aoi_wgs84, meta_dir / "study_area_aoi.geojson")

    metadata = {
        "project": "Hoa Lac study area GIS download",
        "output_dir": str(out_dir),
        "place_name": PLACE_NAME,
        "used_place_name": USE_PLACE_NAME,
        "bbox_fallback": BBOX,
        "aoi_bounds_wgs84": {
            "west": float(bounds[0]),
            "south": float(bounds[1]),
            "east": float(bounds[2]),
            "north": float(bounds[3]),
        },
        "osm_network_type": OSM_NETWORK_TYPE,
        "opentopography_dem_type": OPENTOPO_DEM_TYPE,
        "opentopography_api_key_set": bool(OPENTOPO_API_KEY),
        "download_openbuildingmap": bool(DOWNLOAD_OPENBUILDINGMAP),
        "obm_manifest_url": OBM_MANIFEST_URL,
    }

    with open(meta_dir / "project_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info(f"Metadata saved: {meta_dir}")

# =============================================================================
# Report file status
# =============================================================================

def report_file(label: str, path: Path) -> None:
    """
    Print output file status.
    """
    path = Path(path)

    if path.exists() and path.stat().st_size > 0:
        status = "[OK]"
    else:
        status = "[MISSING]"

    print(f"  {status:<10} {label:<16} {path}")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    setup_logging(OUT_DIR)

    logging.info("=" * 80)
    logging.info("START GIS DOWNLOAD PIPELINE: OSM + OPENTOPOGRAPHY + OPTIONAL OBM")
    logging.info("=" * 80)

    safe_mkdir(OUT_DIR)

    aoi = make_aoi()
    save_project_metadata(aoi, OUT_DIR)

    # 1. OSM
    download_osm(aoi, OUT_DIR)

    # 2. OpenTopography DEM + terrain
    try:
        dem_tif = download_opentopography_dem(aoi, OUT_DIR)
        compute_terrain_products(dem_tif, aoi, OUT_DIR)
    except Exception as e:
        logging.error(f"OpenTopography DEM / terrain step failed: {e}")

    # 3. Optional OpenBuildingMap comparison data
    try:
        process_openbuildingmap_optional(aoi, OUT_DIR)
    except Exception as e:
        logging.error(f"OpenBuildingMap optional step failed: {e}")

    logging.info("=" * 80)
    logging.info("DONE")
    logging.info(f"Output folder: {OUT_DIR.resolve()}")
    logging.info("=" * 80)


    print("\nImportant outputs:")

    report_file(
        "AOI:",
        OUT_DIR / "metadata/study_area_aoi.gpkg",
    )

    report_file(
        "OSM roads:",
        OUT_DIR / "osm/roads/osm_roads_edges.gpkg",
    )

    report_file(
        "Road summary:",
        OUT_DIR / "osm/roads/osm_road_class_summary.csv",
    )

    report_file(
        "OSM features:",
        OUT_DIR / "osm/extra_features/osm_extra_features.gpkg",
    )

    report_file(
        "DEM WGS84:",
        OUT_DIR / f"opentopography/opentopography_{OPENTOPO_DEM_TYPE}_dem_wgs84.tif",
    )

    report_file(
        "DEM UTM:",
        OUT_DIR / f"opentopography/opentopography_{OPENTOPO_DEM_TYPE}_dem_utm.tif",
    )

    report_file(
        "Slope degree:",
        OUT_DIR / "opentopography/terrain_products/slope_degree.tif",
    )

    report_file(
        "Hillshade:",
        OUT_DIR / "opentopography/terrain_products/hillshade.tif",
    )

    report_file(
        "OBM buildings:",
        OBM_CLIPPED_DIR / "obm_buildings_hoalac_clipped.gpkg",
    )

    report_file(
        "OBM summary:",
        OBM_CLIPPED_DIR / "obm_summary.csv",
    )
if __name__ == "__main__":
    main()