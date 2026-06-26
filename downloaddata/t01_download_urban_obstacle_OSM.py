#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download DEM/topography and OSM urban obstacle data inside Hoa Lac polygon.

This version does NOT use:
    - OpenTopography API
    - elevation package
    - make command

DEM source:
    AWS Terrain Tiles / Mapzen GeoTIFF tiles
    https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif

OSM source:
    OpenStreetMap through OSMnx

Extra urban layers added:
    - power lines / power cables
    - power towers / power poles
    - traffic lights
    - street lamps
    - road vertices and sampled road points
    - check plot for all downloaded vector layers

Outputs:
    output_hoalac_hitech_park/
    ├── hoalac_polygon.gpkg
    ├── dem_tiles/
    ├── dem_bbox_merged.tif
    ├── dem_hoalac_clipped.tif
    ├── terrain_dem_hoalac.xyz
    ├── terrain_slope_hoalac.xyz
    ├── terrain_ruggedness_TRI_hoalac.xyz
    ├── buildings_bbox.gpkg
    ├── buildings_hoalac_clipped.gpkg
    ├── buildings_centroid_hoalac.xyz
    ├── buildings_vertices_hoalac.xyz
    └── buildings_grid_hoalac.xyz

XYZ format:
    lon lat value
"""

from pathlib import Path
import math
import warnings
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import xy
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Hoa Lac polygon, format: lon, lat
# This polygon is used for final clipping.
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

# Output folder
OUTDIR = "output/01_HoaLac_studies_area/osm"

# DEM tile zoom level.
# Higher zoom = finer grid, more tiles.
# Recommended:
#   12 = coarser, faster
#   13 = good for small UAV/UTM area
#   14 = finer, more files
DEM_ZOOM = 13

# Extra padding around polygon bbox for downloading DEM/buildings.
# 0.002 degree is roughly 200 m.
BBOX_PADDING_DEG = 0.002

# OSM building height assumptions.
# OSM often has footprints but no height.
DEFAULT_BUILDING_HEIGHT = 6.0  # meter
LEVEL_HEIGHT = 3.0             # meter per floor

# Terrain ruggedness window
TRI_WINDOW_SIZE = 3

# Road data from OpenStreetMap
# Common highway classes:
# motorway, trunk, primary, secondary, tertiary,
# unclassified, residential, service, living_street, track, path
ROAD_HIGHWAY_TYPES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "track",
    "path",
]

# Save road vertices with road class as integer code
ROAD_CLASS_CODE = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "service": 8,
    "living_street": 9,
    "track": 10,
    "path": 11,
    "other": 99,
}

# Sampling interval for road points along road centerlines.
# These are additional points sampled every N meters, not only OSM vertices.
ROAD_POINT_INTERVAL_M = 25.0

# OSM urban obstacle tags to download.
# The power line height is often missing in OSM; height values below are only
# conservative assumptions for voxel pre-processing.
POWER_LINE_TYPES = ["line", "minor_line", "cable"]
POWER_POINT_TYPES = ["tower", "pole"]

POWER_FEATURE_CODE = {
    "line": 31,
    "minor_line": 32,
    "cable": 33,
    "tower": 34,
    "pole": 35,
    "other": 39,
}

URBAN_POINT_CODE = {
    "traffic_light": 41,
    "street_lamp": 42,
}

# Assumed heights when OSM has no explicit height tag.
# Adjust these after local checking / field survey / LiDAR / photogrammetry.
DEFAULT_POWER_LINE_Z_MIN_AGL = 8.0
DEFAULT_POWER_LINE_Z_MAX_AGL = 35.0
DEFAULT_POWER_TOWER_HEIGHT = 35.0
DEFAULT_POWER_POLE_HEIGHT = 12.0
DEFAULT_TRAFFIC_LIGHT_HEIGHT = 6.0
DEFAULT_STREET_LAMP_HEIGHT = 8.0

# If True, write PNG check maps for vector layers.
PLOT_URBAN_LAYER_CHECK = True

# Save all diagnostic figures in OUTDIR / FIGURES_SUBDIR.
FIGURES_SUBDIR = "figures"

# If True, write one separate figure per downloaded urban layer.
PLOT_EACH_URBAN_LAYER = True

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def make_hoalac_polygon_gdf():
    """
    Create Hoa Lac polygon GeoDataFrame.
    """
    poly = Polygon(HOALAC_POLYGON)

    if not poly.is_valid:
        poly = poly.buffer(0)

    gdf = gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_HiTech_Park_approx"]},
        geometry=[poly],
        crs="EPSG:4326",
    )

    return gdf


def get_bbox_from_polygon(poly_gdf, padding_deg=0.0):
    """
    Return bbox from polygon as:
        west, south, east, north
    """
    west, south, east, north = poly_gdf.total_bounds

    west -= padding_deg
    south -= padding_deg
    east += padding_deg
    north += padding_deg

    return west, south, east, north


# ============================================================
# AWS TERRAIN TILE HELPERS
# ============================================================

def lonlat_to_tile(lon, lat, zoom):
    """
    Convert lon/lat to Web Mercator tile x/y at zoom.
    Standard slippy-map tiling.
    """
    lat = max(min(lat, 85.05112878), -85.05112878)

    lat_rad = math.radians(lat)
    n = 2 ** zoom

    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    )

    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))

    return x, y


def get_tile_range_from_bbox(west, south, east, north, zoom):
    """
    Get all terrain tiles covering bbox.
    """
    x_min, y_north = lonlat_to_tile(west, north, zoom)
    x_max, y_south = lonlat_to_tile(east, south, zoom)

    x0 = min(x_min, x_max)
    x1 = max(x_min, x_max)
    y0 = min(y_north, y_south)
    y1 = max(y_north, y_south)

    return x0, x1, y0, y1


def download_aws_terrain_tiles(
    west,
    south,
    east,
    north,
    zoom,
    tile_dir,
):
    """
    Download AWS Terrain Tiles GeoTIFF tiles covering bbox.
    """
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    x0, x1, y0, y1 = get_tile_range_from_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        zoom=zoom,
    )

    print("\n[INFO] Downloading DEM tiles from AWS Terrain Tiles")
    print(f"[INFO] Zoom: {zoom}")
    print(f"[INFO] Tile x range: {x0} to {x1}")
    print(f"[INFO] Tile y range: {y0} to {y1}")

    downloaded_tiles = []

    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif"
            out_file = tile_dir / f"terrain_z{zoom}_x{x}_y{y}.tif"

            if out_file.exists() and out_file.stat().st_size > 0:
                print(f"[SKIP] Existing tile: {out_file.name}")
                downloaded_tiles.append(out_file)
                continue

            print(f"[INFO] Downloading tile z={zoom}, x={x}, y={y}")

            try:
                r = requests.get(url, timeout=120)
            except requests.RequestException as e:
                print(f"[WARN] Request failed for tile {x}/{y}: {e}")
                continue

            if r.status_code != 200:
                print(f"[WARN] Tile not available: {url}")
                print(f"[WARN] Status code: {r.status_code}")
                continue

            with open(out_file, "wb") as f:
                f.write(r.content)

            if out_file.stat().st_size == 0:
                print(f"[WARN] Empty tile: {out_file}")
                continue

            downloaded_tiles.append(out_file)
            print(f"[OK] Saved tile: {out_file}")

    if len(downloaded_tiles) == 0:
        raise RuntimeError(
            "No DEM tiles were downloaded. "
            "Check internet connection or try a lower DEM_ZOOM, e.g. DEM_ZOOM = 12."
        )

    return downloaded_tiles


def mosaic_dem_tiles(tile_files, out_tif):
    """
    Merge downloaded GeoTIFF DEM tiles into one raster.
    """
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Merging DEM tiles")

    datasets = []

    try:
        for f in tile_files:
            datasets.append(rasterio.open(f))

        mosaic_arr, mosaic_transform = merge(datasets)

        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            compress="lzw",
            nodata=datasets[0].nodata,
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(mosaic_arr)

    finally:
        for ds in datasets:
            ds.close()

    print(f"[OK] Saved merged DEM: {out_tif}")


def download_dem_aws_terrain(
    west,
    south,
    east,
    north,
    out_tif,
    tile_dir,
    zoom=13,
):
    """
    Main DEM downloader using AWS Terrain Tiles.
    """
    tile_files = download_aws_terrain_tiles(
        west=west,
        south=south,
        east=east,
        north=north,
        zoom=zoom,
        tile_dir=tile_dir,
    )

    mosaic_dem_tiles(
        tile_files=tile_files,
        out_tif=out_tif,
    )


# ============================================================
# RASTER CLIPPING
# ============================================================

def clip_raster_by_polygon(in_tif, poly_gdf, out_tif):
    """
    Clip raster by polygon boundary.
    """
    in_tif = Path(in_tif)
    out_tif = Path(out_tif)

    with rasterio.open(in_tif) as src:
        poly_for_raster = poly_gdf.to_crs(src.crs)
        geoms = [geom for geom in poly_for_raster.geometry]

        nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = -9999.0

        clipped, clipped_transform = mask(
            src,
            geoms,
            crop=True,
            nodata=nodata_value,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            nodata=nodata_value,
            compress="lzw",
        )

        out_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(clipped)

    print(f"[OK] Saved clipped raster: {out_tif}")


# ============================================================
# RASTER TO XYZ
# ============================================================

def raster_to_xyz(in_tif, out_xyz, band=1):
    """
    Convert raster to XYZ:
        lon lat value
    """
    in_tif = Path(in_tif)
    out_xyz = Path(out_xyz)

    with rasterio.open(in_tif) as src:
        arr = src.read(band).astype(float)
        nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan

        rows, cols = np.where(np.isfinite(arr))

        if len(rows) == 0:
            out_xyz.write_text("")
            print(f"[WARN] No valid raster cells. Empty XYZ saved: {out_xyz}")
            return

        xs, ys = xy(src.transform, rows, cols, offset="center")
        vals = arr[rows, cols]

        points = gpd.GeoDataFrame(
            {"value": vals},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": points.geometry.x,
        "lat": points.geometry.y,
        "value": points["value"].to_numpy(),
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved XYZ: {out_xyz}")


# ============================================================
# TERRAIN DERIVATIVES
# ============================================================

def calculate_slope_and_tri(dem_tif, out_slope_tif, out_tri_tif):
    """
    Calculate:
        slope in degrees
        TRI, terrain ruggedness index, in meters
    """
    dem_tif = Path(dem_tif)

    with rasterio.open(dem_tif) as src:
        dem = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

        if nodata is not None:
            dem[dem == nodata] = np.nan

        if src.crs and src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
            dy_m = abs(transform.e) * 111_320.0
            dx_m = abs(transform.a) * 111_320.0 * np.cos(np.deg2rad(center_lat))
        else:
            dx_m = abs(transform.a)
            dy_m = abs(transform.e)

        dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.rad2deg(slope_rad)

        def tri_func(window):
            center = window[len(window) // 2]

            if not np.isfinite(center):
                return np.nan

            diff = window - center
            return np.sqrt(np.nanmean(diff**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tri = generic_filter(
                dem,
                tri_func,
                size=TRI_WINDOW_SIZE,
                mode="nearest",
            )

        profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1,
            compress="lzw",
        )

        slope_write = np.where(
            np.isfinite(slope_deg),
            slope_deg,
            -9999.0,
        ).astype("float32")

        tri_write = np.where(
            np.isfinite(tri),
            tri,
            -9999.0,
        ).astype("float32")

        with rasterio.open(out_slope_tif, "w", **profile) as dst:
            dst.write(slope_write, 1)

        with rasterio.open(out_tri_tif, "w", **profile) as dst:
            dst.write(tri_write, 1)

    print(f"[OK] Saved slope raster: {out_slope_tif}")
    print(f"[OK] Saved TRI raster: {out_tri_tif}")



# ============================================================
# GENERIC OSM / VECTOR HELPERS
# ============================================================

def sanitize_gdf_for_file(gdf):
    """
    Make a GeoDataFrame safer for GeoPackage writing.

    OSMnx often returns list/dict/set values. GeoPackage drivers can fail on
    those object columns, so complex values are converted to strings.
    """
    if gdf is None or gdf.empty:
        return gdf

    clean = gdf.copy().reset_index(drop=True)

    for col in clean.columns:
        if col == clean.geometry.name:
            continue

        if clean[col].dtype == "object":
            clean[col] = clean[col].apply(
                lambda v: ";".join(map(str, v))
                if isinstance(v, (list, tuple, set))
                else (str(v) if isinstance(v, dict) else v)
            )

    return clean


def write_empty_gpkg_with_schema(out_gpkg, columns=None, crs="EPSG:4326"):
    """
    Write an empty GeoPackage with a simple schema.
    """
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if columns:
        for name, dtype in columns.items():
            data[name] = pd.Series(dtype=dtype)

    empty = gpd.GeoDataFrame(
        data,
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )
    empty.to_file(out_gpkg, driver="GPKG")
    return empty


def osm_features_from_bbox_compat(west, south, east, north, tags):
    """
    OSMnx compatibility wrapper for features_from_bbox.
    Works across common OSMnx 1.x and 2.x call signatures.
    """
    try:
        # OSMnx 2.x: bbox=(left, bottom, right, top)
        return ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            return ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            # Older OSMnx style: north, south, east, west, tags
            return ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )


def parse_osm_height_m(row, default_height):
    """
    Parse a height-like OSM field in meters.
    Falls back to building:levels or a provided default height.
    """
    for key in ["height", "building:height", "tower:height"]:
        if key in row and pd.notna(row[key]):
            raw = str(row[key]).lower()
            raw = raw.replace("meters", "")
            raw = raw.replace("meter", "")
            raw = raw.replace("m", "")
            raw = raw.strip()
            try:
                return float(raw)
            except ValueError:
                pass

    if "building:levels" in row and pd.notna(row["building:levels"]):
        try:
            return float(str(row["building:levels"]).strip()) * LEVEL_HEIGHT
        except ValueError:
            pass

    return float(default_height)


def first_osm_value(value, default="other"):
    """
    Return one representative value from an OSM tag that may be string/list/NaN.
    """
    if value is None:
        return default

    if isinstance(value, (list, tuple, set)):
        value = list(value)[0] if len(value) else default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return str(value)


def clip_vector_by_polygon(gdf, poly_gdf, out_gpkg, empty_columns=None, layer_name="features"):
    """
    Clip vector features to Hoa Lac polygon and save GeoPackage.
    """
    out_gpkg = Path(out_gpkg)

    if gdf is None or gdf.empty:
        print(f"[WARN] No {layer_name} to clip.")
        return write_empty_gpkg_with_schema(out_gpkg, columns=empty_columns)

    src = gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(src, polygon)

    if clipped.empty:
        print(f"[WARN] No {layer_name} inside Hoa Lac polygon.")
        return write_empty_gpkg_with_schema(out_gpkg, columns=empty_columns)

    clipped = clipped.copy().reset_index(drop=True)
    clipped = sanitize_gdf_for_file(clipped)
    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped {layer_name}: {out_gpkg}")
    print(f"[INFO] Number of clipped {layer_name}: {len(clipped)}")

    return clipped


def download_osm_feature_bbox(west, south, east, north, tags, keep_geom_types, out_gpkg, layer_name="features"):
    """
    Generic OSM feature downloader using a bbox and OSM tags.
    """
    print(f"\n[INFO] Downloading OSM {layer_name} from bbox")
    print(f"[INFO] OSM tags: {tags}")

    empty_columns = {
        "feature_class": "str",
        "feature_code": "int",
        "height_m": "float",
    }

    try:
        gdf = osm_features_from_bbox_compat(west, south, east, north, tags)
    except Exception as e:
        print(f"[WARN] OSM download failed for {layer_name}: {e}")
        return write_empty_gpkg_with_schema(out_gpkg, columns=empty_columns)

    if gdf.empty:
        print(f"[WARN] No OSM {layer_name} found in bbox.")
        return write_empty_gpkg_with_schema(out_gpkg, columns=empty_columns)

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    if keep_geom_types is not None:
        gdf = gdf[gdf.geometry.type.isin(keep_geom_types)].copy()

    if gdf.empty:
        print(f"[WARN] OSM data found, but no usable geometry for {layer_name}.")
        return write_empty_gpkg_with_schema(out_gpkg, columns=empty_columns)

    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    gdf = sanitize_gdf_for_file(gdf)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox {layer_name}: {out_gpkg}")
    print(f"[INFO] Number of bbox {layer_name}: {len(gdf)}")

    return gdf

# ============================================================
# OSM BUILDING DOWNLOAD
# ============================================================

def parse_building_height(row):
    """
    Estimate building height in meters.

    Priority:
        1. height
        2. building:height
        3. building:levels * LEVEL_HEIGHT
        4. DEFAULT_BUILDING_HEIGHT
    """
    for key in ["height", "building:height"]:
        if key in row and pd.notna(row[key]):
            raw = str(row[key]).lower()
            raw = raw.replace("meters", "")
            raw = raw.replace("meter", "")
            raw = raw.replace("m", "")
            raw = raw.strip()

            try:
                return float(raw)
            except ValueError:
                pass

    if "building:levels" in row and pd.notna(row["building:levels"]):
        raw = str(row["building:levels"]).strip()

        try:
            return float(raw) * LEVEL_HEIGHT
        except ValueError:
            pass

    return DEFAULT_BUILDING_HEIGHT


def write_empty_gpkg(out_gpkg, crs="EPSG:4326"):
    """
    Write empty GeoPackage with minimal schema.
    """
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    empty = gpd.GeoDataFrame(
        {"height_m": pd.Series(dtype="float")},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    empty.to_file(out_gpkg, driver="GPKG")
    return empty


def download_osm_buildings_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM building footprints inside bbox.
    """
    print("\n[INFO] Downloading OSM buildings from bbox")

    tags = {"building": True}

    try:
        # OSMnx 2.x
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            # Some versions accept positional bbox
            gdf = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            # Older OSMnx style
            gdf = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if gdf.empty:
        print("[WARN] No OSM buildings found in bbox.")
        return write_empty_gpkg(out_gpkg)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        print("[WARN] OSM data found, but no building polygons.")
        return write_empty_gpkg(out_gpkg)

    gdf = gdf.to_crs("EPSG:4326")
    gdf["height_m"] = gdf.apply(parse_building_height, axis=1)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    gdf = sanitize_gdf_for_file(gdf)
    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox buildings: {out_gpkg}")
    print(f"[INFO] Number of bbox buildings: {len(gdf)}")

    return gdf


def clip_buildings_by_polygon(buildings_gdf, poly_gdf, out_gpkg):
    """
    Clip OSM buildings by Hoa Lac polygon.
    """
    out_gpkg = Path(out_gpkg)

    if buildings_gdf.empty:
        print("[WARN] No buildings to clip.")
        return write_empty_gpkg(out_gpkg)

    buildings = buildings_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(buildings, polygon)

    if clipped.empty:
        print("[WARN] No buildings inside Hoa Lac polygon.")
        return write_empty_gpkg(out_gpkg)

    # Important fix for OSMnx MultiIndex problem
    clipped = clipped.copy().reset_index(drop=True)

    if "height_m" not in clipped.columns:
        clipped["height_m"] = DEFAULT_BUILDING_HEIGHT

    clipped = sanitize_gdf_for_file(clipped)
    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped buildings: {out_gpkg}")
    print(f"[INFO] Number of clipped buildings: {len(clipped)}")

    return clipped


# ============================================================
# BUILDING XYZ EXPORT
# ============================================================

def save_building_centroids_xyz(gdf, out_xyz):
    """
    Save one point per building:
        lon lat building_height_m

    Fixed for OSMnx MultiIndex / clipped GeoDataFrame index problem.
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty building centroid XYZ saved: {out_xyz}")
        return

    # Important fix:
    # OSMnx often returns MultiIndex. After clipping, index can be incompatible.
    # Reset index to simple 0,1,2,...
    gdf = gdf.copy().reset_index(drop=True)

    # Keep only valid geometries
    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy().reset_index(drop=True)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] No valid building polygons. Empty XYZ saved: {out_xyz}")
        return

    if "height_m" not in gdf.columns:
        gdf["height_m"] = DEFAULT_BUILDING_HEIGHT

    # Calculate centroid in projected CRS, then convert back to lon/lat
    projected_crs = gdf.estimate_utm_crs()
    gdf_projected = gdf.to_crs(projected_crs).reset_index(drop=True)

    centroid_geom = gdf_projected.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        {
            "height_m": gdf_projected["height_m"].to_numpy()
        },
        geometry=gpd.GeoSeries(
            centroid_geom.to_numpy(),
            crs=projected_crs,
        ),
        crs=projected_crs,
    ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": centroid_gdf.geometry.x.to_numpy(),
        "lat": centroid_gdf.geometry.y.to_numpy(),
        "height_m": centroid_gdf["height_m"].to_numpy(),
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved building centroid XYZ: {out_xyz}")


def save_building_vertices_xyz(gdf, out_xyz):
    """
    Save building polygon vertices:
        lon lat building_height_m
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty building vertices XYZ saved: {out_xyz}")
        return

    gdf = gdf.to_crs("EPSG:4326")

    records = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        height = row["height_m"]

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            for x, y in poly.exterior.coords:
                records.append((x, y, height))

    df = pd.DataFrame(records, columns=["lon", "lat", "height_m"])

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved building vertices XYZ: {out_xyz}")


def rasterize_buildings_to_dem_grid(gdf, dem_tif, out_xyz):
    """
    Rasterize building height to the same grid as clipped DEM.

    Output:
        lon lat building_height_m

    No-building cells are 0.
    """
    out_xyz = Path(out_xyz)

    with rasterio.open(dem_tif) as src:
        shape = (src.height, src.width)
        transform = src.transform

        if gdf.empty:
            building_grid = np.zeros(shape, dtype="float32")
        else:
            gdf_dem = gdf.to_crs(src.crs)

            shapes = [
                (geom, float(height))
                for geom, height in zip(gdf_dem.geometry, gdf_dem["height_m"])
                if geom is not None and not geom.is_empty
            ]

            building_grid = rasterize(
                shapes=shapes,
                out_shape=shape,
                transform=transform,
                fill=0.0,
                dtype="float32",
                all_touched=True,
            )

        rows, cols = np.where(np.isfinite(building_grid))
        xs, ys = xy(transform, rows, cols, offset="center")
        vals = building_grid[rows, cols]

        points = gpd.GeoDataFrame(
            {"height_m": vals},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": points.geometry.x,
        "lat": points.geometry.y,
        "height_m": points["height_m"].to_numpy(),
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved building grid XYZ: {out_xyz}")

# ============================================================
# OSM ROAD DOWNLOAD
# ============================================================

def normalize_highway_type(value):
    """
    Normalize OSM highway tag.

    OSM highway can be string, list/tuple/set, or missing.
    """
    value = first_osm_value(value, default="other")

    if value in ROAD_CLASS_CODE:
        return value

    return "other"


def download_osm_roads_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM road centerlines inside bbox.

    Output GeoPackage contains LineString/MultiLineString roads.
    """

    print("\n[INFO] Downloading OSM roads from bbox")

    tags = {
        "highway": ROAD_HIGHWAY_TYPES
    }

    try:
        # OSMnx 2.x
        roads = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            # Some OSMnx versions
            roads = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            # Older OSMnx style
            roads = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if roads.empty:
        print("[WARN] No OSM roads found in bbox.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads[
        roads.geometry.type.isin(
            ["LineString", "MultiLineString"]
        )
    ].copy()

    if roads.empty:
        print("[WARN] OSM data found, but no road lines.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads.to_crs("EPSG:4326").reset_index(drop=True)

    roads["highway_simple"] = roads["highway"].apply(normalize_highway_type)
    roads["road_code"] = roads["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    roads = sanitize_gdf_for_file(roads)
    roads.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox roads: {out_gpkg}")
    print(f"[INFO] Number of bbox road segments: {len(roads)}")

    return roads


def clip_roads_by_polygon(roads_gdf, poly_gdf, out_gpkg):
    """
    Clip OSM roads by Hoa Lac polygon.
    """

    out_gpkg = Path(out_gpkg)

    if roads_gdf.empty:
        print("[WARN] No roads to clip.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(roads, polygon)

    if clipped.empty:
        print("[WARN] No roads inside Hoa Lac polygon.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    clipped = clipped.copy().reset_index(drop=True)

    if "highway_simple" not in clipped.columns:
        clipped["highway_simple"] = clipped["highway"].apply(normalize_highway_type)

    if "road_code" not in clipped.columns:
        clipped["road_code"] = clipped["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    clipped = sanitize_gdf_for_file(clipped)
    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped roads: {out_gpkg}")
    print(f"[INFO] Number of clipped road segments: {len(clipped)}")

    return clipped


def save_road_vertices_xyz(roads_gdf, out_xyz):
    """
    Save road centerline vertices as XYZ.

    Format:
        lon lat road_code

    road_code:
        1  motorway
        2  trunk
        3  primary
        4  secondary
        5  tertiary
        6  unclassified
        7  residential
        8  service
        9  living_street
        10 track
        11 path
        99 other
    """

    out_xyz = Path(out_xyz)

    if roads_gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty road XYZ saved: {out_xyz}")
        return

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []

    for _, row in roads.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        road_code = int(row.get("road_code", 99))

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            for x, y in line.coords:
                records.append((x, y, road_code))

            # Add NaN separator between line segments.
            # Useful if later you want line plotting.
            records.append((np.nan, np.nan, np.nan))

    df = pd.DataFrame(records, columns=["lon", "lat", "road_code"])

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved road vertices XYZ: {out_xyz}")


def extract_road_vertices_points_gdf(roads_gdf):
    """
    Extract original OSM road vertices as a point GeoDataFrame for plotting.

    Output columns:
        road_id, vertex_id, road_code, geometry
    """
    if roads_gdf is None or roads_gdf.empty:
        return gpd.GeoDataFrame(
            {
                "road_id": pd.Series(dtype="int"),
                "vertex_id": pd.Series(dtype="int"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    records = []
    xs = []
    ys = []

    for road_id, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        road_code = int(row.get("road_code", 99))

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        vertex_id = 0
        for line in lines:
            for x, y in line.coords:
                xs.append(x)
                ys.append(y)
                records.append({
                    "road_id": int(road_id),
                    "vertex_id": int(vertex_id),
                    "road_code": road_code,
                })
                vertex_id += 1

    if not records:
        return gpd.GeoDataFrame(
            {
                "road_id": pd.Series(dtype="int"),
                "vertex_id": pd.Series(dtype="int"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:4326",
    )


def save_line_vertices_xyz(gdf, out_xyz, code_col="feature_code"):
    """
    Save line vertices to XYZ-like file:
        lon lat code
    """
    out_xyz = Path(out_xyz)

    if gdf is None or gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty line vertices XYZ saved: {out_xyz}")
        return

    lines_gdf = gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    records = []

    for _, row in lines_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        code = int(row.get(code_col, 99))

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            for x, y in line.coords:
                records.append((x, y, code))
            records.append((np.nan, np.nan, np.nan))

    df = pd.DataFrame(records, columns=["lon", "lat", "code"])
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved line vertices XYZ: {out_xyz}")


def save_point_features_xyz(gdf, out_xyz, code_col="feature_code", height_col="height_m"):
    """
    Save point/centroid features to XYZ-like file:
        lon lat height_m code
    """
    out_xyz = Path(out_xyz)

    if gdf is None or gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty point XYZ saved: {out_xyz}")
        return

    src = gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    src = src[src.geometry.notna() & (~src.geometry.is_empty)].copy().reset_index(drop=True)

    if src.empty:
        out_xyz.write_text("")
        print(f"[WARN] No valid point geometries. Empty XYZ saved: {out_xyz}")
        return

    # Convert polygons/lines to centroids so every object has one representative point.
    projected_crs = src.estimate_utm_crs()
    projected = src.to_crs(projected_crs).copy().reset_index(drop=True)
    centroid = projected.geometry.centroid

    pts = gpd.GeoDataFrame(
        {
            "height_m": projected.get(height_col, pd.Series([0.0] * len(projected))).to_numpy(),
            "code": projected.get(code_col, pd.Series([99] * len(projected))).to_numpy(),
        },
        geometry=gpd.GeoSeries(centroid.to_numpy(), crs=projected_crs),
        crs=projected_crs,
    ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": pts.geometry.x.to_numpy(),
        "lat": pts.geometry.y.to_numpy(),
        "height_m": pts["height_m"].to_numpy(),
        "code": pts["code"].to_numpy(),
    })

    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved point XYZ: {out_xyz}")


def save_road_sample_points_xyz(roads_gdf, out_xyz, interval_m=25.0):
    """
    Sample road centerlines every interval_m meters.

    Output XYZ-like file:
        lon lat road_code

    This is different from road vertices: vertices are the original OSM line
    breakpoints; sampled road points are regularly spaced points along roads.
    """
    out_xyz = Path(out_xyz)

    if roads_gdf is None or roads_gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty road sampled points XYZ saved: {out_xyz}")
        return gpd.GeoDataFrame(
            {"road_code": pd.Series(dtype="int")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    projected_crs = roads.estimate_utm_crs()
    roads_m = roads.to_crs(projected_crs).copy().reset_index(drop=True)

    point_records = []
    point_geoms = []

    for road_id, row in roads_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        road_code = int(row.get("road_code", 99))

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for part_id, line in enumerate(lines):
            length = float(line.length)
            if length <= 0:
                continue

            distances = list(np.arange(0.0, length, interval_m))
            if not distances or distances[-1] < length:
                distances.append(length)

            for d in distances:
                pt = line.interpolate(float(d))
                point_geoms.append(pt)
                point_records.append({
                    "road_id": int(road_id),
                    "part_id": int(part_id),
                    "distance_m": float(d),
                    "road_code": road_code,
                })

    if not point_geoms:
        out_xyz.write_text("")
        print(f"[WARN] No sampled road points. Empty XYZ saved: {out_xyz}")
        return gpd.GeoDataFrame(
            {"road_code": pd.Series(dtype="int")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    pts_m = gpd.GeoDataFrame(
        point_records,
        geometry=gpd.GeoSeries(point_geoms, crs=projected_crs),
        crs=projected_crs,
    )
    pts = pts_m.to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": pts.geometry.x.to_numpy(),
        "lat": pts.geometry.y.to_numpy(),
        "road_code": pts["road_code"].to_numpy(),
    })
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    csv_path = out_xyz.with_suffix(".csv")
    df2 = pd.DataFrame({
        "lon": pts.geometry.x.to_numpy(),
        "lat": pts.geometry.y.to_numpy(),
        "road_code": pts["road_code"].to_numpy(),
        "road_id": pts["road_id"].to_numpy(),
        "part_id": pts["part_id"].to_numpy(),
        "distance_m": pts["distance_m"].to_numpy(),
    })
    df2.to_csv(csv_path, index=False, float_format="%.8f")

    print(f"[OK] Saved sampled road points XYZ: {out_xyz}")
    print(f"[OK] Saved sampled road points CSV: {csv_path}")
    print(f"[INFO] Number of sampled road points: {len(pts)}")

    return pts


def download_powerlines_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM power line/cable features.
    """
    tags = {"power": POWER_LINE_TYPES}
    gdf = download_osm_feature_bbox(
        west, south, east, north,
        tags=tags,
        keep_geom_types=["LineString", "MultiLineString"],
        out_gpkg=out_gpkg,
        layer_name="power lines/cables",
    )

    if gdf is not None and not gdf.empty:
        gdf = gdf.copy().reset_index(drop=True)
        gdf["power_simple"] = gdf["power"].apply(first_osm_value) if "power" in gdf.columns else "other"
        gdf["feature_code"] = gdf["power_simple"].map(POWER_FEATURE_CODE).fillna(39).astype(int)
        gdf["z_min_agl_m"] = DEFAULT_POWER_LINE_Z_MIN_AGL
        gdf["z_max_agl_m"] = DEFAULT_POWER_LINE_Z_MAX_AGL
        gdf = sanitize_gdf_for_file(gdf)
        gdf.to_file(out_gpkg, driver="GPKG")

    return gdf


def download_power_towers_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM power tower/pole features.
    """
    tags = {"power": POWER_POINT_TYPES}
    gdf = download_osm_feature_bbox(
        west, south, east, north,
        tags=tags,
        keep_geom_types=["Point", "MultiPoint", "Polygon", "MultiPolygon"],
        out_gpkg=out_gpkg,
        layer_name="power towers/poles",
    )

    if gdf is not None and not gdf.empty:
        gdf = gdf.copy().reset_index(drop=True)
        gdf["power_simple"] = gdf["power"].apply(first_osm_value) if "power" in gdf.columns else "other"
        gdf["feature_code"] = gdf["power_simple"].map(POWER_FEATURE_CODE).fillna(39).astype(int)
        gdf["height_m"] = gdf.apply(
            lambda row: parse_osm_height_m(
                row,
                DEFAULT_POWER_TOWER_HEIGHT
                if row.get("power_simple", "other") == "tower"
                else DEFAULT_POWER_POLE_HEIGHT,
            ),
            axis=1,
        )
        gdf = sanitize_gdf_for_file(gdf)
        gdf.to_file(out_gpkg, driver="GPKG")

    return gdf


def download_traffic_lights_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM traffic light features.
    """
    tags = {"highway": "traffic_signals"}
    gdf = download_osm_feature_bbox(
        west, south, east, north,
        tags=tags,
        keep_geom_types=["Point", "MultiPoint", "Polygon", "MultiPolygon"],
        out_gpkg=out_gpkg,
        layer_name="traffic lights",
    )

    if gdf is not None and not gdf.empty:
        gdf = gdf.copy().reset_index(drop=True)
        gdf["feature_class"] = "traffic_light"
        gdf["feature_code"] = URBAN_POINT_CODE["traffic_light"]
        gdf["height_m"] = gdf.apply(lambda row: parse_osm_height_m(row, DEFAULT_TRAFFIC_LIGHT_HEIGHT), axis=1)
        gdf = sanitize_gdf_for_file(gdf)
        gdf.to_file(out_gpkg, driver="GPKG")

    return gdf


def download_street_lamps_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM street lamp features.
    """
    tags = {"highway": "street_lamp"}
    gdf = download_osm_feature_bbox(
        west, south, east, north,
        tags=tags,
        keep_geom_types=["Point", "MultiPoint", "Polygon", "MultiPolygon"],
        out_gpkg=out_gpkg,
        layer_name="street lamps",
    )

    if gdf is not None and not gdf.empty:
        gdf = gdf.copy().reset_index(drop=True)
        gdf["feature_class"] = "street_lamp"
        gdf["feature_code"] = URBAN_POINT_CODE["street_lamp"]
        gdf["height_m"] = gdf.apply(lambda row: parse_osm_height_m(row, DEFAULT_STREET_LAMP_HEIGHT), axis=1)
        gdf = sanitize_gdf_for_file(gdf)
        gdf.to_file(out_gpkg, driver="GPKG")

    return gdf


def plot_urban_layers_check(
    poly_gdf,
    buildings_gdf,
    roads_gdf,
    road_points_gdf,
    powerlines_gdf,
    powertowers_gdf,
    traffic_lights_gdf,
    street_lamps_gdf,
    out_png,
):
    """
    Plot downloaded OSM layers for quick visual checking.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    poly = poly_gdf.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)

    # Light background polygon.
    poly.plot(ax=ax, facecolor="#f7f7f7", edgecolor="black", linewidth=1.5, alpha=0.35)

    if buildings_gdf is not None and not buildings_gdf.empty:
        buildings_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=0.25,
            alpha=0.75,
            label="Buildings",
        )

    if roads_gdf is not None and not roads_gdf.empty:
        roads_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="dimgray",
            linewidth=0.8,
            alpha=0.85,
            label="Roads",
        )

    if road_points_gdf is not None and not road_points_gdf.empty:
        road_points_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="black",
            markersize=1.0,
            alpha=0.4,
            label=f"Road points {ROAD_POINT_INTERVAL_M:g} m",
        )

    if powerlines_gdf is not None and not powerlines_gdf.empty:
        powerlines_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="red",
            linewidth=1.4,
            alpha=0.9,
            label="Power lines/cables",
        )

    if powertowers_gdf is not None and not powertowers_gdf.empty:
        powertowers_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="purple",
            markersize=22,
            marker="^",
            alpha=0.9,
            label="Power towers/poles",
        )

    if traffic_lights_gdf is not None and not traffic_lights_gdf.empty:
        traffic_lights_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="limegreen",
            markersize=26,
            marker="o",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.95,
            label="Traffic lights",
        )

    if street_lamps_gdf is not None and not street_lamps_gdf.empty:
        street_lamps_gdf.to_crs("EPSG:4326").plot(
            ax=ax,
            color="orange",
            markersize=16,
            marker="*",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            label="Street lamps",
        )

    poly.boundary.plot(ax=ax, color="black", linewidth=1.8, label="Hoa Lac AOI")

    minx, miny, maxx, maxy = poly.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_title("Hoa Lac OSM urban obstacle layers", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.92,
        )

    summary = [
        f"Buildings: {0 if buildings_gdf is None else len(buildings_gdf)}",
        f"Roads: {0 if roads_gdf is None else len(roads_gdf)}",
        f"Road pts: {0 if road_points_gdf is None else len(road_points_gdf)}",
        f"Power lines: {0 if powerlines_gdf is None else len(powerlines_gdf)}",
        f"Towers/poles: {0 if powertowers_gdf is None else len(powertowers_gdf)}",
        f"Traffic lights: {0 if traffic_lights_gdf is None else len(traffic_lights_gdf)}",
        f"Street lamps: {0 if street_lamps_gdf is None else len(street_lamps_gdf)}",
    ]
    ax.text(
        0.01,
        0.01,
        "\n".join(summary),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved urban layer check plot: {out_png}")


def _set_common_map_format(ax, poly_gdf, title):
    """
    Apply common extent/title/grid formatting for all check maps.
    """
    poly = poly_gdf.to_crs("EPSG:4326")
    poly.boundary.plot(ax=ax, color="black", linewidth=1.6)

    minx, miny, maxx, maxy = poly.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")


def _plot_empty_layer(poly_gdf, out_png, title, message):
    """
    Save a map frame even when the requested OSM layer is empty.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    poly = poly_gdf.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    poly.plot(ax=ax, facecolor="#f7f7f7", edgecolor="black", linewidth=1.5, alpha=0.35)
    _set_common_map_format(ax, poly_gdf, title)

    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.90),
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved empty-layer figure: {out_png}")


def _numeric_column_or_default(gdf, column, default_value=0.0):
    """
    Return a GeoDataFrame copy with a numeric plotting column.
    """
    out = gdf.copy().reset_index(drop=True)
    if column not in out.columns:
        out[column] = default_value
    out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default_value)
    return out


def plot_layer_value_map(
    poly_gdf,
    layer_gdf,
    out_png,
    title,
    value_col,
    colorbar_label,
    default_value=0.0,
    linewidth=1.0,
    markersize=20,
    marker="o",
    polygon_edgecolor="gray",
    empty_message=None,
):
    """
    Plot one layer and encode value_col by colorbar.

    Works with polygon, line, and point GeoDataFrames. If the layer is empty,
    a blank diagnostic figure is still saved so the output folder is complete.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if layer_gdf is None or layer_gdf.empty:
        _plot_empty_layer(
            poly_gdf=poly_gdf,
            out_png=out_png,
            title=title,
            message=empty_message or "No OSM features found inside AOI",
        )
        return

    poly = poly_gdf.to_crs("EPSG:4326")
    layer = layer_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    layer = layer[layer.geometry.notna() & (~layer.geometry.is_empty)].copy().reset_index(drop=True)

    if layer.empty:
        _plot_empty_layer(
            poly_gdf=poly_gdf,
            out_png=out_png,
            title=title,
            message=empty_message or "No valid geometry inside AOI",
        )
        return

    layer = _numeric_column_or_default(layer, value_col, default_value)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    poly.plot(ax=ax, facecolor="#f7f7f7", edgecolor="black", linewidth=1.5, alpha=0.35)

    geom_types = set(layer.geometry.geom_type.unique())
    is_polygon = bool(geom_types & {"Polygon", "MultiPolygon"})
    is_line = bool(geom_types & {"LineString", "MultiLineString"})
    is_point = bool(geom_types & {"Point", "MultiPoint"})

    legend_kwds = {
        "label": colorbar_label,
        "shrink": 0.72,
        "pad": 0.02,
    }

    if is_polygon:
        layer.plot(
            ax=ax,
            column=value_col,
            cmap="viridis",
            legend=True,
            legend_kwds=legend_kwds,
            edgecolor=polygon_edgecolor,
            linewidth=0.25,
            alpha=0.85,
        )
    elif is_line:
        layer.plot(
            ax=ax,
            column=value_col,
            cmap="viridis",
            legend=True,
            legend_kwds=legend_kwds,
            linewidth=linewidth,
            alpha=0.95,
        )
    elif is_point:
        layer.plot(
            ax=ax,
            column=value_col,
            cmap="viridis",
            legend=True,
            legend_kwds=legend_kwds,
            markersize=markersize,
            marker=marker,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.95,
        )
    else:
        layer.plot(
            ax=ax,
            column=value_col,
            cmap="viridis",
            legend=True,
            legend_kwds=legend_kwds,
            markersize=markersize,
            alpha=0.95,
        )

    _set_common_map_format(ax, poly_gdf, title)

    summary = [
        f"Features: {len(layer)}",
        f"{value_col} min: {layer[value_col].min():.2f}",
        f"{value_col} max: {layer[value_col].max():.2f}",
    ]
    ax.text(
        0.01,
        0.01,
        "\n".join(summary),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.86),
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved layer value figure: {out_png}")


def plot_all_individual_urban_layer_maps(
    poly_gdf,
    buildings_gdf,
    roads_gdf,
    road_vertices_gdf,
    road_points_gdf,
    powerlines_gdf,
    powertowers_gdf,
    traffic_lights_gdf,
    street_lamps_gdf,
    figures_dir,
):
    """
    Write one diagnostic figure per urban layer.

    For physical obstacles, the colorbar shows height or assumed height.
    Roads are surface objects, so their maps use road class code instead of
    height. This avoids pretending that OSM road data contains true height.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=buildings_gdf,
        out_png=figures_dir / "01_buildings_height_m.png",
        title="Buildings — height map",
        value_col="height_m",
        colorbar_label="Building height (m)",
        default_value=DEFAULT_BUILDING_HEIGHT,
        polygon_edgecolor="gray",
        empty_message="No OSM building polygons found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=roads_gdf,
        out_png=figures_dir / "02_roads_class_code.png",
        title="Road centerlines — road class code",
        value_col="road_code",
        colorbar_label="Road class code",
        default_value=99,
        linewidth=1.2,
        empty_message="No OSM road centerlines found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=road_vertices_gdf,
        out_png=figures_dir / "03_road_vertices_class_code.png",
        title="Road original OSM vertices — road class code",
        value_col="road_code",
        colorbar_label="Road class code",
        default_value=99,
        markersize=9,
        marker="o",
        empty_message="No road vertices found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=road_points_gdf,
        out_png=figures_dir / "04_road_sample_points_class_code.png",
        title=f"Road sample points — every {ROAD_POINT_INTERVAL_M:g} m",
        value_col="road_code",
        colorbar_label="Road class code",
        default_value=99,
        markersize=7,
        marker="o",
        empty_message="No road sample points created inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=powerlines_gdf,
        out_png=figures_dir / "05_powerlines_assumed_zmax_agl_m.png",
        title="Power lines / cables — assumed upper obstacle height",
        value_col="z_max_agl_m",
        colorbar_label="Assumed z max AGL (m)",
        default_value=DEFAULT_POWER_LINE_Z_MAX_AGL,
        linewidth=1.8,
        empty_message="No OSM power lines/cables found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=powertowers_gdf,
        out_png=figures_dir / "06_power_towers_poles_height_m.png",
        title="Power towers / poles — height map",
        value_col="height_m",
        colorbar_label="Tower/pole height (m)",
        default_value=DEFAULT_POWER_POLE_HEIGHT,
        markersize=28,
        marker="^",
        empty_message="No OSM power towers/poles found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=traffic_lights_gdf,
        out_png=figures_dir / "07_traffic_lights_height_m.png",
        title="Traffic lights — assumed height map",
        value_col="height_m",
        colorbar_label="Traffic light height (m)",
        default_value=DEFAULT_TRAFFIC_LIGHT_HEIGHT,
        markersize=34,
        marker="o",
        empty_message="No OSM traffic lights found inside AOI",
    )

    plot_layer_value_map(
        poly_gdf=poly_gdf,
        layer_gdf=street_lamps_gdf,
        out_png=figures_dir / "08_street_lamps_height_m.png",
        title="Street lamps — assumed height map",
        value_col="height_m",
        colorbar_label="Street lamp height (m)",
        default_value=DEFAULT_STREET_LAMP_HEIGHT,
        markersize=34,
        marker="*",
        empty_message="No OSM street lamps found inside AOI",
    )



# ============================================================
# LAYER EXISTENCE / USABILITY CHECKS
# ============================================================

def _valid_feature_count(gdf):
    """
    Count valid non-empty geometries in a GeoDataFrame.
    """
    if gdf is None or gdf.empty:
        return 0
    try:
        valid = gdf.geometry.notna() & (~gdf.geometry.is_empty)
        return int(valid.sum())
    except Exception:
        return int(len(gdf))


def _geometry_type_summary(gdf):
    """
    Return a compact geometry type summary string.
    """
    if gdf is None or gdf.empty or "geometry" not in gdf:
        return "none"
    try:
        geom_types = (
            gdf.geometry[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
            .geom_type.value_counts()
            .to_dict()
        )
    except Exception:
        return "unknown"
    if not geom_types:
        return "none"
    return "; ".join(f"{k}:{v}" for k, v in geom_types.items())


def _numeric_min_max(gdf, column):
    """
    Return min/max for a numeric column, or NaN/NaN when unavailable.
    """
    if gdf is None or gdf.empty or column is None or column not in gdf.columns:
        return np.nan, np.nan
    vals = pd.to_numeric(gdf[column], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return np.nan, np.nan
    return float(vals.min()), float(vals.max())


def _file_exists(path):
    """
    File existence helper for report table.
    """
    if path is None:
        return False
    return Path(path).exists() and Path(path).stat().st_size >= 0


def _raster_valid_cell_count(tif_path):
    """
    Count valid raster cells. Used only for availability reporting.
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return 0
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            nodata = src.nodata
            if nodata is None:
                return int(np.isfinite(arr).sum())
            return int(((arr != nodata) & np.isfinite(arr)).sum())
    except Exception:
        return 0


def build_layer_availability_summary(
    outdir,
    dem_tif,
    slope_tif,
    tri_tif,
    buildings_gdf,
    roads_gdf,
    road_vertices_gdf,
    road_points_gdf,
    powerlines_gdf,
    powertowers_gdf,
    traffic_lights_gdf,
    street_lamps_gdf,
):
    """
    Build a table showing which layers exist and whether they are useful for
    the 3D voxel obstacle model.

    Important idea:
        - Roads are useful, but normally as surface/risk/reference data.
        - Buildings, power infrastructure, traffic lights, and lamps can become
          physical obstacle volumes, but many heights are assumed from defaults.
    """
    outdir = Path(outdir)

    vector_specs = [
        {
            "layer": "buildings",
            "source": "OSM building footprints",
            "gdf": buildings_gdf,
            "gpkg": outdir / "buildings_hoalac_clipped.gpkg",
            "xyz": outdir / "buildings_vertices_hoalac.xyz",
            "value_col": "height_m",
            "value_label": "height_m",
            "voxel_role": "hard_obstacle",
            "height_source": "OSM height/building:levels, otherwise assumed default",
            "use_now": "YES",
            "note": "Use as 3D blocked volume. Check heights because many OSM buildings may use default height.",
        },
        {
            "layer": "roads_centerlines",
            "source": "OSM highway=*",
            "gdf": roads_gdf,
            "gpkg": outdir / "roads_hoalac_clipped.gpkg",
            "xyz": outdir / "roads_vertices_hoalac.xyz",
            "value_col": "road_code",
            "value_label": "road_code",
            "voxel_role": "risk/reference",
            "height_source": "no height; surface network only",
            "use_now": "REFERENCE_ONLY",
            "note": "Use for road-crossing risk, access, or traffic exposure. Do not block airspace by default.",
        },
        {
            "layer": "road_vertices",
            "source": "extracted from road LineString vertices",
            "gdf": road_vertices_gdf,
            "gpkg": None,
            "xyz": outdir / "roads_vertices_hoalac.xyz",
            "value_col": "road_code",
            "value_label": "road_code",
            "voxel_role": "risk/reference points",
            "height_source": "no height; derived points only",
            "use_now": "REFERENCE_ONLY",
            "note": "Useful for checking OSM geometry vertices. Prefer sampled road points for gridded risk.",
        },
        {
            "layer": "road_sample_points",
            "source": f"sampled every {ROAD_POINT_INTERVAL_M:g} m along roads",
            "gdf": road_points_gdf,
            "gpkg": None,
            "xyz": outdir / "roads_points_hoalac.xyz",
            "value_col": "road_code",
            "value_label": "road_code",
            "voxel_role": "risk/reference points",
            "height_source": "no height; derived points only",
            "use_now": "YES_FOR_SOFT_RISK",
            "note": "Good for rasterizing road-proximity or road-crossing risk.",
        },
        {
            "layer": "powerlines_cables",
            "source": "OSM power=line/minor_line/cable",
            "gdf": powerlines_gdf,
            "gpkg": outdir / "powerlines_hoalac_clipped.gpkg",
            "xyz": outdir / "powerlines_vertices_hoalac.xyz",
            "value_col": "z_max_agl_m",
            "value_label": "assumed_zmax_agl_m",
            "voxel_role": "hard_obstacle",
            "height_source": "assumed z_min/z_max unless OSM/custom height added",
            "use_now": "YES_WITH_BUFFER",
            "note": "Use as conservative 3D blocked corridor. Need field/DSM check for true cable height and sag.",
        },
        {
            "layer": "power_towers_poles",
            "source": "OSM power=tower/pole",
            "gdf": powertowers_gdf,
            "gpkg": outdir / "power_towers_poles_hoalac_clipped.gpkg",
            "xyz": outdir / "power_towers_poles_points_hoalac.xyz",
            "value_col": "height_m",
            "value_label": "height_m",
            "voxel_role": "hard_obstacle",
            "height_source": "OSM height if available, otherwise tower/pole default",
            "use_now": "YES_WITH_BUFFER",
            "note": "Use as point/cylinder obstacle. Check whether tower/pole classification is complete.",
        },
        {
            "layer": "traffic_lights",
            "source": "OSM highway=traffic_signals",
            "gdf": traffic_lights_gdf,
            "gpkg": outdir / "traffic_lights_hoalac_clipped.gpkg",
            "xyz": outdir / "traffic_lights_points_hoalac.xyz",
            "value_col": "height_m",
            "value_label": "height_m",
            "voxel_role": "low_altitude_obstacle",
            "height_source": "usually assumed default height",
            "use_now": "YES_IF_LOW_ALTITUDE",
            "note": "Use near DB/DK/FLZ or flight below about 20 m AGL. Usually not important for normal cruise altitude.",
        },
        {
            "layer": "street_lamps",
            "source": "OSM highway=street_lamp",
            "gdf": street_lamps_gdf,
            "gpkg": outdir / "street_lamps_hoalac_clipped.gpkg",
            "xyz": outdir / "street_lamps_points_hoalac.xyz",
            "value_col": "height_m",
            "value_label": "height_m",
            "voxel_role": "low_altitude_obstacle",
            "height_source": "usually assumed default height",
            "use_now": "YES_IF_LOW_ALTITUDE",
            "note": "Use near takeoff/landing corridors. OSM completeness can be poor.",
        },
    ]

    records = []

    # Raster/terrain layers.
    raster_specs = [
        {
            "layer": "terrain_dem",
            "source": "AWS Terrain Tiles / Mapzen GeoTIFF",
            "path": dem_tif,
            "voxel_role": "terrain_lower_boundary",
            "use_now": "YES",
            "note": "Use as ground elevation / lower no-fly boundary.",
        },
        {
            "layer": "terrain_slope",
            "source": "derived from DEM",
            "path": slope_tif,
            "voxel_role": "terrain_risk/reference",
            "use_now": "OPTIONAL",
            "note": "Useful for terrain complexity, not a direct air obstacle.",
        },
        {
            "layer": "terrain_TRI",
            "source": "derived from DEM",
            "path": tri_tif,
            "voxel_role": "terrain_risk/reference",
            "use_now": "OPTIONAL",
            "note": "Useful for terrain roughness, not a direct air obstacle.",
        },
    ]

    for spec in raster_specs:
        path = Path(spec["path"])
        count = _raster_valid_cell_count(path)
        exists = _file_exists(path)
        status = "USABLE" if exists and count > 0 else "EMPTY_OR_MISSING"
        records.append({
            "layer": spec["layer"],
            "source": spec["source"],
            "feature_count": count,
            "geometry_types": "raster cells",
            "value_column": "elevation_or_derived",
            "value_min": np.nan,
            "value_max": np.nan,
            "gpkg_exists": False,
            "xyz_exists": _file_exists(path),
            "status": status,
            "voxel_role": spec["voxel_role"],
            "use_now": spec["use_now"],
            "height_source": "raster elevation",
            "note": spec["note"],
        })

    # Vector layers.
    for spec in vector_specs:
        gdf = spec["gdf"]
        count = _valid_feature_count(gdf)
        vmin, vmax = _numeric_min_max(gdf, spec["value_col"])
        status = "USABLE" if count > 0 else "EMPTY_IN_AOI"
        if count > 0 and spec["use_now"] in ["REFERENCE_ONLY", "OPTIONAL"]:
            status = "REFERENCE_AVAILABLE"
        elif count > 0 and "ASSUMED" in spec["height_source"].upper():
            status = "USABLE_BUT_HEIGHT_ASSUMED"

        records.append({
            "layer": spec["layer"],
            "source": spec["source"],
            "feature_count": count,
            "geometry_types": _geometry_type_summary(gdf),
            "value_column": spec["value_label"],
            "value_min": vmin,
            "value_max": vmax,
            "gpkg_exists": _file_exists(spec["gpkg"]),
            "xyz_exists": _file_exists(spec["xyz"]),
            "status": status,
            "voxel_role": spec["voxel_role"],
            "use_now": spec["use_now"] if count > 0 else "NO_DATA_IN_AOI",
            "height_source": spec["height_source"],
            "note": spec["note"],
        })

    return pd.DataFrame.from_records(records)


def save_layer_availability_report(summary_df, out_csv, out_txt):
    """
    Save CSV and readable TXT report showing which layers can be used.
    """
    out_csv = Path(out_csv)
    out_txt = Path(out_txt)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_csv, index=False)

    lines = []
    lines.append("HOA LAC URBAN / TERRAIN LAYER AVAILABILITY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  USABLE                  : layer has data and can be used now")
    lines.append("  USABLE_BUT_HEIGHT_ASSUMED: layer has data, but height should be checked")
    lines.append("  REFERENCE_AVAILABLE     : useful for risk/reference, not a hard obstacle")
    lines.append("  EMPTY_IN_AOI            : no OSM features found inside Hoa Lac polygon")
    lines.append("  EMPTY_OR_MISSING        : file missing or no valid raster cells")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(f"[{row['status']}] {row['layer']}")
        lines.append(f"  count       : {row['feature_count']}")
        lines.append(f"  role        : {row['voxel_role']}")
        lines.append(f"  use_now     : {row['use_now']}")
        lines.append(f"  value       : {row['value_column']} min={row['value_min']} max={row['value_max']}")
        lines.append(f"  geometry    : {row['geometry_types']}")
        lines.append(f"  height src  : {row['height_source']}")
        lines.append(f"  note        : {row['note']}")
        lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved layer availability CSV: {out_csv}")
    print(f"[OK] Saved layer availability TXT: {out_txt}")


def plot_layer_availability_counts(summary_df, out_png):
    """
    Plot feature/cell counts for all layers so the user can quickly see what
    exists in the downloaded data.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = summary_df.copy()
    df["feature_count"] = pd.to_numeric(df["feature_count"], errors="coerce").fillna(0)
    df = df.sort_values("feature_count", ascending=True)

    fig_h = max(6.0, 0.42 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=180)

    ax.barh(df["layer"], df["feature_count"])
    ax.set_xlabel("Feature count / valid raster cell count")
    ax.set_ylabel("Layer")
    ax.set_title("Hoa Lac data availability by layer", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

    for y, (_, row) in enumerate(df.iterrows()):
        count = int(row["feature_count"])
        label = f"{count} | {row['status']} | {row['use_now']}"
        ax.text(
            max(count, 1) * 1.01,
            y,
            label,
            va="center",
            fontsize=7,
        )

    xmax = max(float(df["feature_count"].max()), 1.0)
    ax.set_xlim(0, xmax * 1.45)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved layer count figure: {out_png}")


def _plot_layer_on_gallery_axis(ax, poly_gdf, layer_gdf, title, value_col=None, default_value=0.0):
    """
    Plot one layer into an existing subplot axis. This is intentionally compact;
    detailed colorbars are provided by the individual layer figures.
    """
    poly = poly_gdf.to_crs("EPSG:4326")
    poly.plot(ax=ax, facecolor="#f7f7f7", edgecolor="black", linewidth=0.8, alpha=0.35)

    count = _valid_feature_count(layer_gdf)
    if count > 0:
        layer = layer_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
        layer = layer[layer.geometry.notna() & (~layer.geometry.is_empty)].copy().reset_index(drop=True)
        geom_types = set(layer.geometry.geom_type.unique())

        if value_col is not None:
            layer = _numeric_column_or_default(layer, value_col, default_value)
            plot_kwargs = {"column": value_col, "cmap": "viridis", "alpha": 0.9}
        else:
            plot_kwargs = {"alpha": 0.9}

        if geom_types & {"Polygon", "MultiPolygon"}:
            layer.plot(ax=ax, edgecolor="gray", linewidth=0.15, **plot_kwargs)
        elif geom_types & {"LineString", "MultiLineString"}:
            layer.plot(ax=ax, linewidth=1.0, **plot_kwargs)
        else:
            layer.plot(ax=ax, markersize=8, edgecolor="black", linewidth=0.15, **plot_kwargs)
    else:
        ax.text(
            0.5, 0.5, "NO DATA",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
        )

    poly.boundary.plot(ax=ax, color="black", linewidth=0.9)
    minx, miny, maxx, maxy = poly.total_bounds
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_title(f"{title}\nN={count}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_all_layers_gallery(
    poly_gdf,
    buildings_gdf,
    roads_gdf,
    road_vertices_gdf,
    road_points_gdf,
    powerlines_gdf,
    powertowers_gdf,
    traffic_lights_gdf,
    street_lamps_gdf,
    out_png,
):
    """
    Plot all vector layers in one gallery image for rapid quality control.
    Detailed colorbar maps are saved separately by plot_all_individual_urban_layer_maps().
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    layer_specs = [
        ("Buildings", buildings_gdf, "height_m", DEFAULT_BUILDING_HEIGHT),
        ("Roads", roads_gdf, "road_code", 99),
        ("Road vertices", road_vertices_gdf, "road_code", 99),
        ("Road sample points", road_points_gdf, "road_code", 99),
        ("Power lines/cables", powerlines_gdf, "z_max_agl_m", DEFAULT_POWER_LINE_Z_MAX_AGL),
        ("Power towers/poles", powertowers_gdf, "height_m", DEFAULT_POWER_POLE_HEIGHT),
        ("Traffic lights", traffic_lights_gdf, "height_m", DEFAULT_TRAFFIC_LIGHT_HEIGHT),
        ("Street lamps", street_lamps_gdf, "height_m", DEFAULT_STREET_LAMP_HEIGHT),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), dpi=180)
    axes = axes.ravel()

    for ax, (title, gdf, value_col, default_value) in zip(axes, layer_specs):
        _plot_layer_on_gallery_axis(
            ax=ax,
            poly_gdf=poly_gdf,
            layer_gdf=gdf,
            title=title,
            value_col=value_col,
            default_value=default_value,
        )

    # Last panel: recommendation text.
    axes[-1].axis("off")
    recommendation = (
        "Use now for 3D obstacle:\n"
        "  buildings, power lines, towers/poles\n\n"
        "Use only at low altitude / landing:\n"
        "  traffic lights, street lamps\n\n"
        "Use as soft-risk/reference:\n"
        "  roads, road vertices, road sample points\n\n"
        "If a panel says NO DATA, OSM has no mapped\n"
        "features of that type inside this AOI."
    )
    axes[-1].text(
        0.02, 0.98, recommendation,
        ha="left", va="top",
        fontsize=10,
        transform=axes[-1].transAxes,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
    )

    fig.suptitle("Hoa Lac OSM layers — quick usability check", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved all-layer gallery figure: {out_png}")

# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    figures_dir = outdir / FIGURES_SUBDIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== HOA LAC DATA DOWNLOAD ==========")
    print("[INFO] DEM source: AWS Terrain Tiles / Mapzen GeoTIFF")
    print("[INFO] Building source: OpenStreetMap")

    # --------------------------------------------------------
    # 1. Prepare Hoa Lac polygon
    # --------------------------------------------------------
    poly_gdf = make_hoalac_polygon_gdf()

    poly_file = outdir / "hoalac_polygon.gpkg"
    poly_gdf.to_file(poly_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {poly_file}")

    # --------------------------------------------------------
    # 2. Get bbox from polygon
    # --------------------------------------------------------
    west, south, east, north = get_bbox_from_polygon(
        poly_gdf,
        padding_deg=BBOX_PADDING_DEG,
    )

    print("\n[INFO] Download bbox:")
    print(f"  WEST  = {west}")
    print(f"  SOUTH = {south}")
    print(f"  EAST  = {east}")
    print(f"  NORTH = {north}")

    # --------------------------------------------------------
    # 3. Download DEM from AWS terrain tiles and merge
    # --------------------------------------------------------
    dem_tile_dir = outdir / "dem_tiles"
    dem_bbox_tif = outdir / "dem_bbox_merged.tif"
    dem_clip_tif = outdir / "dem_hoalac_clipped.tif"

    download_dem_aws_terrain(
        west=west,
        south=south,
        east=east,
        north=north,
        out_tif=dem_bbox_tif,
        tile_dir=dem_tile_dir,
        zoom=DEM_ZOOM,
    )

    # --------------------------------------------------------
    # 4. Clip DEM to Hoa Lac polygon
    # --------------------------------------------------------
    clip_raster_by_polygon(
        in_tif=dem_bbox_tif,
        poly_gdf=poly_gdf,
        out_tif=dem_clip_tif,
    )

    # --------------------------------------------------------
    # 5. Export DEM to XYZ
    # --------------------------------------------------------
    raster_to_xyz(
        in_tif=dem_clip_tif,
        out_xyz=outdir / "terrain_dem_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 6. Calculate slope and TRI
    # --------------------------------------------------------
    slope_tif = outdir / "terrain_slope_hoalac.tif"
    tri_tif = outdir / "terrain_ruggedness_TRI_hoalac.tif"

    calculate_slope_and_tri(
        dem_tif=dem_clip_tif,
        out_slope_tif=slope_tif,
        out_tri_tif=tri_tif,
    )

    raster_to_xyz(
        in_tif=slope_tif,
        out_xyz=outdir / "terrain_slope_hoalac.xyz",
    )

    raster_to_xyz(
        in_tif=tri_tif,
        out_xyz=outdir / "terrain_ruggedness_TRI_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 7. Download OSM buildings using bbox
    # --------------------------------------------------------
    buildings_bbox_gpkg = outdir / "buildings_bbox.gpkg"

    buildings_bbox = download_osm_buildings_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=buildings_bbox_gpkg,
    )

    # --------------------------------------------------------
    # 8. Clip buildings to Hoa Lac polygon
    # --------------------------------------------------------
    buildings_clip_gpkg = outdir / "buildings_hoalac_clipped.gpkg"

    buildings_clip = clip_buildings_by_polygon(
        buildings_gdf=buildings_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=buildings_clip_gpkg,
    )

    # --------------------------------------------------------
    # 9. Export buildings to XYZ
    # --------------------------------------------------------
    save_building_centroids_xyz(
        buildings_clip,
        outdir / "buildings_centroid_hoalac.xyz",
    )

    save_building_vertices_xyz(
        buildings_clip,
        outdir / "buildings_vertices_hoalac.xyz",
    )

    rasterize_buildings_to_dem_grid(
        buildings_clip,
        dem_tif=dem_clip_tif,
        out_xyz=outdir / "buildings_grid_hoalac.xyz",
    )
        # --------------------------------------------------------
    # 10. Download OSM roads using bbox
    # --------------------------------------------------------
    roads_bbox_gpkg = outdir / "roads_bbox.gpkg"

    roads_bbox = download_osm_roads_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=roads_bbox_gpkg,
    )

    # --------------------------------------------------------
    # 11. Clip roads to Hoa Lac polygon
    # --------------------------------------------------------
    roads_clip_gpkg = outdir / "roads_hoalac_clipped.gpkg"

    roads_clip = clip_roads_by_polygon(
        roads_gdf=roads_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=roads_clip_gpkg,
    )

    # --------------------------------------------------------
    # 12. Export road vertices to XYZ
    # --------------------------------------------------------
    save_road_vertices_xyz(
        roads_clip,
        outdir / "roads_vertices_hoalac.xyz",
    )

    road_vertices_points = extract_road_vertices_points_gdf(roads_clip)

    road_points = save_road_sample_points_xyz(
        roads_clip,
        outdir / "roads_points_hoalac.xyz",
        interval_m=ROAD_POINT_INTERVAL_M,
    )

    # --------------------------------------------------------
    # 13. Download OSM power lines / cables
    # --------------------------------------------------------
    powerlines_bbox_gpkg = outdir / "powerlines_bbox.gpkg"
    powerlines_bbox = download_powerlines_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=powerlines_bbox_gpkg,
    )

    powerlines_clip_gpkg = outdir / "powerlines_hoalac_clipped.gpkg"
    powerlines_clip = clip_vector_by_polygon(
        powerlines_bbox,
        poly_gdf,
        powerlines_clip_gpkg,
        empty_columns={
            "power_simple": "str",
            "feature_code": "int",
            "z_min_agl_m": "float",
            "z_max_agl_m": "float",
        },
        layer_name="power lines/cables",
    )

    save_line_vertices_xyz(
        powerlines_clip,
        outdir / "powerlines_vertices_hoalac.xyz",
        code_col="feature_code",
    )

    # --------------------------------------------------------
    # 14. Download OSM power towers / poles
    # --------------------------------------------------------
    powertowers_bbox_gpkg = outdir / "power_towers_poles_bbox.gpkg"
    powertowers_bbox = download_power_towers_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=powertowers_bbox_gpkg,
    )

    powertowers_clip_gpkg = outdir / "power_towers_poles_hoalac_clipped.gpkg"
    powertowers_clip = clip_vector_by_polygon(
        powertowers_bbox,
        poly_gdf,
        powertowers_clip_gpkg,
        empty_columns={
            "power_simple": "str",
            "feature_code": "int",
            "height_m": "float",
        },
        layer_name="power towers/poles",
    )

    save_point_features_xyz(
        powertowers_clip,
        outdir / "power_towers_poles_points_hoalac.xyz",
        code_col="feature_code",
        height_col="height_m",
    )

    # --------------------------------------------------------
    # 15. Download OSM traffic lights
    # --------------------------------------------------------
    traffic_lights_bbox_gpkg = outdir / "traffic_lights_bbox.gpkg"
    traffic_lights_bbox = download_traffic_lights_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=traffic_lights_bbox_gpkg,
    )

    traffic_lights_clip_gpkg = outdir / "traffic_lights_hoalac_clipped.gpkg"
    traffic_lights_clip = clip_vector_by_polygon(
        traffic_lights_bbox,
        poly_gdf,
        traffic_lights_clip_gpkg,
        empty_columns={
            "feature_class": "str",
            "feature_code": "int",
            "height_m": "float",
        },
        layer_name="traffic lights",
    )

    save_point_features_xyz(
        traffic_lights_clip,
        outdir / "traffic_lights_points_hoalac.xyz",
        code_col="feature_code",
        height_col="height_m",
    )

    # --------------------------------------------------------
    # 16. Download OSM street lamps
    # --------------------------------------------------------
    street_lamps_bbox_gpkg = outdir / "street_lamps_bbox.gpkg"
    street_lamps_bbox = download_street_lamps_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=street_lamps_bbox_gpkg,
    )

    street_lamps_clip_gpkg = outdir / "street_lamps_hoalac_clipped.gpkg"
    street_lamps_clip = clip_vector_by_polygon(
        street_lamps_bbox,
        poly_gdf,
        street_lamps_clip_gpkg,
        empty_columns={
            "feature_class": "str",
            "feature_code": "int",
            "height_m": "float",
        },
        layer_name="street lamps",
    )

    save_point_features_xyz(
        street_lamps_clip,
        outdir / "street_lamps_points_hoalac.xyz",
        code_col="feature_code",
        height_col="height_m",
    )

    # --------------------------------------------------------
    # 17. Plot all urban obstacle layers for checking
    # --------------------------------------------------------
    if PLOT_URBAN_LAYER_CHECK:
        plot_urban_layers_check(
            poly_gdf=poly_gdf,
            buildings_gdf=buildings_clip,
            roads_gdf=roads_clip,
            road_points_gdf=road_points,
            powerlines_gdf=powerlines_clip,
            powertowers_gdf=powertowers_clip,
            traffic_lights_gdf=traffic_lights_clip,
            street_lamps_gdf=street_lamps_clip,
            out_png=figures_dir / "00_osm_urban_obstacle_layers_check.png",
        )

    if PLOT_EACH_URBAN_LAYER:
        plot_all_individual_urban_layer_maps(
            poly_gdf=poly_gdf,
            buildings_gdf=buildings_clip,
            roads_gdf=roads_clip,
            road_vertices_gdf=road_vertices_points,
            road_points_gdf=road_points,
            powerlines_gdf=powerlines_clip,
            powertowers_gdf=powertowers_clip,
            traffic_lights_gdf=traffic_lights_clip,
            street_lamps_gdf=street_lamps_clip,
            figures_dir=figures_dir,
        )

    # --------------------------------------------------------
    # 18. Check which layers exist and which ones can be used
    # --------------------------------------------------------
    layer_summary = build_layer_availability_summary(
        outdir=outdir,
        dem_tif=dem_clip_tif,
        slope_tif=slope_tif,
        tri_tif=tri_tif,
        buildings_gdf=buildings_clip,
        roads_gdf=roads_clip,
        road_vertices_gdf=road_vertices_points,
        road_points_gdf=road_points,
        powerlines_gdf=powerlines_clip,
        powertowers_gdf=powertowers_clip,
        traffic_lights_gdf=traffic_lights_clip,
        street_lamps_gdf=street_lamps_clip,
    )

    save_layer_availability_report(
        layer_summary,
        out_csv=outdir / "urban_layer_availability_summary.csv",
        out_txt=outdir / "urban_layer_availability_summary.txt",
    )

    plot_layer_availability_counts(
        layer_summary,
        out_png=figures_dir / "00a_layer_availability_counts.png",
    )

    plot_all_layers_gallery(
        poly_gdf=poly_gdf,
        buildings_gdf=buildings_clip,
        roads_gdf=roads_clip,
        road_vertices_gdf=road_vertices_points,
        road_points_gdf=road_points,
        powerlines_gdf=powerlines_clip,
        powertowers_gdf=powertowers_clip,
        traffic_lights_gdf=traffic_lights_clip,
        street_lamps_gdf=street_lamps_clip,
        out_png=figures_dir / "00b_all_layers_gallery.png",
    )

    print("\n========== LAYER USABILITY SUMMARY ==========")
    cols_to_print = ["layer", "feature_count", "status", "use_now", "voxel_role"]
    print(layer_summary[cols_to_print].to_string(index=False))
    
    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")

    print("\nImportant output files:")
    print(f"  Polygon:             {outdir / 'hoalac_polygon.gpkg'}")
    print(f"  DEM GeoTIFF:         {outdir / 'dem_hoalac_clipped.tif'}")
    print(f"  DEM XYZ:             {outdir / 'terrain_dem_hoalac.xyz'}")
    print(f"  Slope XYZ:           {outdir / 'terrain_slope_hoalac.xyz'}")
    print(f"  Ruggedness XYZ:      {outdir / 'terrain_ruggedness_TRI_hoalac.xyz'}")
    print(f"  Buildings GPKG:      {outdir / 'buildings_hoalac_clipped.gpkg'}")
    print(f"  Building centroid:   {outdir / 'buildings_centroid_hoalac.xyz'}")
    print(f"  Building vertices:   {outdir / 'buildings_vertices_hoalac.xyz'}")
    print(f"  Building grid:       {outdir / 'buildings_grid_hoalac.xyz'}")
    print(f"  Roads GPKG:          {outdir / 'roads_hoalac_clipped.gpkg'}")
    print(f"  Road vertices:       {outdir / 'roads_vertices_hoalac.xyz'}")
    print(f"  Road points:         {outdir / 'roads_points_hoalac.xyz'}")
    print(f"  Powerlines GPKG:     {outdir / 'powerlines_hoalac_clipped.gpkg'}")
    print(f"  Powerline vertices:  {outdir / 'powerlines_vertices_hoalac.xyz'}")
    print(f"  Power towers/poles:  {outdir / 'power_towers_poles_points_hoalac.xyz'}")
    print(f"  Traffic lights:      {outdir / 'traffic_lights_points_hoalac.xyz'}")
    print(f"  Street lamps:        {outdir / 'street_lamps_points_hoalac.xyz'}")
    print(f"  Layer summary CSV:   {outdir / 'urban_layer_availability_summary.csv'}")
    print(f"  Layer summary TXT:   {outdir / 'urban_layer_availability_summary.txt'}")
    print(f"  Figures dir:         {figures_dir}")
    print(f"  Check plot:          {figures_dir / '00_osm_urban_obstacle_layers_check.png'}")
    print(f"  Layer counts plot:   {figures_dir / '00a_layer_availability_counts.png'}")
    print(f"  All-layer gallery:   {figures_dir / '00b_all_layers_gallery.png'}")
    print(f"  Buildings figure:    {figures_dir / '01_buildings_height_m.png'}")
    print(f"  Road vertices fig:   {figures_dir / '03_road_vertices_class_code.png'}")
    print(f"  Road points fig:     {figures_dir / '04_road_sample_points_class_code.png'}")
    print(f"  Powerline figure:    {figures_dir / '05_powerlines_assumed_zmax_agl_m.png'}")


if __name__ == "__main__":
    main()
