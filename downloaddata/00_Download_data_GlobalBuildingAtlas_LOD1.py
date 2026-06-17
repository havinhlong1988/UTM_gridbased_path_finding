#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact low-RAM GlobalBuildingAtlas LoD1 processor for Hoa Lac.

What this script does:
    1. Define Hoa Lac polygon.
    2. Detect needed 5-degree GBA LoD1 tile.
    3. Use existing local tile by default, or download if allowed.
    4. Read Parquet by PyArrow batches to avoid RAM crash.
    5. Clip buildings to study polygon.
    6. Export GPKG/CSV/XYZ/OBJ.
    7. Plot 2D PyGMT maps.
    8. Plot 3D using either:
        - PyVista: recommended, better 3D LoD1 mesh.
        - PyGMT: simple fallback centroid-prism plot.

Install:
    conda activate utm
    conda install -c conda-forge geopandas pandas numpy shapely pyarrow requests tqdm pygmt gmt
    conda install -c conda-forge pyvista vtk trame trame-vtk trame-vuetify

Run:
    python t00_Download_data_LOD1_compact_pyvista.py
"""

from pathlib import Path
import math
import json
import gc
import tempfile
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from tqdm import tqdm

import pyarrow.parquet as pq
from shapely.geometry import Polygon, box
from shapely import wkb, wkt

import pygmt


# ============================================================
# ONE PARAMETER BLOCK ONLY
# ============================================================

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

OUTDIR = Path("output/01_HoaLac_studies_area/globalbuildingatlas_lod1")

# Existing Hoa Lac tile is usually:
# output/01_HoaLac_studies_area/globalbuildingatlas_lod1/raw_tiles/e105_n25_e110_n20.parquet
SKIP_DOWNLOAD_NEW = True
MANUAL_LOCAL_PARQUET_FILES = []
MANUAL_REMOTE_PARQUET_URLS = []

# Source Cooperative
SOURCECOOP_BUCKET_LIST_URL = "https://us-west-2.opendata.source.coop"
SOURCECOOP_PREFIX = "tge-labs/globalbuildingatlas-lod1"
SOURCECOOP_DIRECT_BASES = [
    "https://us-west-2.opendata.source.coop/tge-labs/globalbuildingatlas-lod1",
    "https://data.source.coop/tge-labs/globalbuildingatlas-lod1",
]

# Low RAM
BBOX_PADDING_DEG = 0.002
PARQUET_BATCH_SIZE = 25_000
READ_MINIMUM_COLUMNS_ONLY = True
SAVE_BBOX_FILTER_CACHE = True
FORCE_REBUILD_BBOX_CACHE = False
DEFAULT_BUILDING_HEIGHT_M = 6.0
WRITE_RAW_BBOX_GPKG = False

# Exports
EXPORT_OBJ = True
CLEAN_TEMP_FILES = True

# Plot
FIG_DPI = 300
REGION_PADDING = 0.003
PROJECTION = "M15c"
PYGMT_FRAME_MAP = ["xaf+lLongitude", "yaf+lLatitude", "WSen"]
MAX_POLYGON_PLOT_BUILDINGS = 20_000
MAX_3D_BUILDINGS = 3000

# 3D engine:
#   "pyvista" = recommended
#   "pygmt"   = simple fallback
PLOT_3D_ENGINE = "pyvista"

# PyVista options
PYVISTA_EXPORT_HTML = True
PYVISTA_EXPORT_VTP = True
PYVISTA_BACKGROUND = "white"
PYVISTA_WINDOW_SIZE = [2200, 1600]
PYVISTA_CMAP = "viridis"

# PyVista colorbar: same meaning/name as PyGMT height colorbar.
# Set to None for automatic range.
# Example fixed range:
#     PYVISTA_HEIGHT_CBAR_RANGE = [0.0, 45.0]
PYVISTA_COLORBAR_TITLE = "Building height (m)"
PYVISTA_HEIGHT_CBAR_RANGE = [0.0, 40.0]
PYVISTA_BUILDING_OPACITY = 0.90
PYVISTA_SHOW_EDGES = True
PYVISTA_EDGE_COLOR = "black"
PYVISTA_EDGE_WIDTH = 0.25
PYVISTA_Z_EXAGGERATION = 30.0
PYVISTA_MESH_METHOD = "extrude"  # "extrude" is safer than manual faces
PYVISTA_CAMERA_POSITION = "iso"
PYVISTA_CAMERA_AZIMUTH = 205
PYVISTA_CAMERA_ELEVATION = 5
PYVISTA_CAMERA_ZOOM = 1.20
PYVISTA_ADD_BOUNDARY = True
PYVISTA_BOUNDARY_COLOR = "purple"
PYVISTA_BOUNDARY_WIDTH = 5

# PyGMT 3D fallback options
PYGMT_3D_STYLE = "o0.06c"
PYGMT_3D_TRANSPARENCY = 25
PYGMT_3D_PERSPECTIVE = [225, 25]


# ============================================================
# PATHS
# ============================================================

METADATA_DIR = OUTDIR / "metadata"
RAW_TILE_DIR = OUTDIR / "raw_tiles"
PROCESSED_DIR = OUTDIR / "processed"
FIG_DIR = OUTDIR / "figures"

for _d in [METADATA_DIR, RAW_TILE_DIR, PROCESSED_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

BBOX_FILTER_CACHE_FILE = PROCESSED_DIR / "gba_lod1_buildings_bbox_filtered_lowram.parquet"


# ============================================================
# GEOMETRY AND TILE HELPERS
# ============================================================

def make_hoalac_polygon_gdf():
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_HiTech_Park_approx"]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def get_bbox_from_polygon(poly_gdf, padding_deg=0.0):
    west, south, east, north = poly_gdf.total_bounds
    return west - padding_deg, south - padding_deg, east + padding_deg, north + padding_deg


def get_plot_region(poly_gdf, padding=REGION_PADDING):
    west, south, east, north = poly_gdf.total_bounds
    return [west - padding, east + padding, south - padding, north + padding]


def get_polygon_area_stats(poly_gdf):
    poly_utm = poly_gdf.to_crs(poly_gdf.estimate_utm_crs())
    area_m2 = float(poly_utm.geometry.area.iloc[0])
    return area_m2, area_m2 / 1e6, area_m2 / 1e4


def floor_to_5(value):
    return math.floor(value / 5.0) * 5


def ceil_to_5(value):
    return math.ceil(value / 5.0) * 5


def fmt_lon(lon):
    lon = int(lon)
    return f"w{abs(lon)}" if lon < 0 else f"e{lon}"


def fmt_lat(lat):
    lat = int(lat)
    return f"s{abs(lat)}" if lat < 0 else f"n{lat}"


def get_gba_5deg_tiles_for_bbox(west, south, east, north):
    lon_mins = list(range(int(floor_to_5(west)), int(ceil_to_5(east)), 5))
    lat_mins = list(range(int(floor_to_5(south)), int(ceil_to_5(north)), 5))

    records = []
    for lon_min in lon_mins:
        lon_max = lon_min + 5
        for lat_min in lat_mins:
            lat_max = lat_min + 5
            tile_name = f"{fmt_lon(lon_min)}_{fmt_lat(lat_max)}_{fmt_lon(lon_max)}_{fmt_lat(lat_min)}"
            records.append({
                "tile_name": tile_name,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "geometry": box(lon_min, lat_min, lon_max, lat_max),
            })

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ============================================================
# DOWNLOAD / LOCAL TILE RESOLUTION
# ============================================================

def list_sourcecoop_keys(prefix=SOURCECOOP_PREFIX):
    keys = []
    continuation_token = None

    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation_token:
            params["continuation-token"] = continuation_token

        try:
            response = requests.get(SOURCECOOP_BUCKET_LIST_URL, params=params, timeout=60)
        except requests.RequestException as exc:
            print(f"[WARN] Source Cooperative listing failed: {exc}")
            return keys

        if response.status_code != 200:
            print(f"[WARN] Source Cooperative listing status code: {response.status_code}")
            return keys

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError:
            print("[WARN] Could not parse Source Cooperative XML listing.")
            return keys

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for content in root.findall(f".//{ns}Contents"):
            key_el = content.find(f"{ns}Key")
            if key_el is not None and key_el.text:
                keys.append(key_el.text)

        is_truncated_el = root.find(f"{ns}IsTruncated")
        is_truncated = (
            is_truncated_el is not None
            and is_truncated_el.text
            and is_truncated_el.text.lower() == "true"
        )

        if not is_truncated:
            break

        token_el = root.find(f"{ns}NextContinuationToken")
        if token_el is None or not token_el.text:
            break

        continuation_token = token_el.text

    return keys


def remote_file_exists(url):
    try:
        response = requests.head(url, timeout=30, allow_redirects=True)
        if response.status_code == 200:
            return True
        if response.status_code in [403, 405]:
            response = requests.get(
                url,
                timeout=30,
                stream=True,
                headers={"Range": "bytes=0-1023"},
            )
            return response.status_code in [200, 206]
    except requests.RequestException:
        return False

    return False


def candidate_remote_urls_for_tile(tile_name, all_keys=None):
    urls = []

    if all_keys:
        for key in all_keys:
            if key.lower().endswith((".parquet", ".pq")) and tile_name.lower() in key.lower():
                urls.append(f"{SOURCECOOP_BUCKET_LIST_URL.rstrip('/')}/{key}")

    candidate_relpaths = [
        f"{tile_name}.parquet",
        f"{tile_name}.pq",
        f"parquet/{tile_name}.parquet",
        f"data/{tile_name}.parquet",
        f"tiles/{tile_name}.parquet",
        f"{tile_name}/{tile_name}.parquet",
    ]

    for base in SOURCECOOP_DIRECT_BASES:
        for rel in candidate_relpaths:
            urls.append(f"{base.rstrip('/')}/{rel}")

    unique = []
    seen = set()
    for url in urls:
        if url not in seen:
            unique.append(url)
            seen.add(url)

    return unique


def download_file(url, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"[SKIP] Existing file: {out_file}")
        return out_file

    print(f"[INFO] Downloading: {url}")
    print(f"[INFO] Output: {out_file}")

    with requests.get(url, stream=True, timeout=600) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"[ERROR] Download failed: {url}\n"
                f"Status code: {response.status_code}\n"
                f"{response.text[:1000]}"
            )

        total = int(response.headers.get("content-length", 0))
        with open(out_file, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=out_file.name) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    if out_file.stat().st_size < 1000:
        raise RuntimeError(f"[ERROR] Downloaded file is too small: {out_file}")

    return out_file


def resolve_parquet_files(tile_gdf):
    local_files = []

    if MANUAL_LOCAL_PARQUET_FILES:
        print("[INFO] Use MANUAL_LOCAL_PARQUET_FILES.")
        for p in MANUAL_LOCAL_PARQUET_FILES:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Manual local Parquet not found: {p}")
            local_files.append(p)
        return local_files

    if SKIP_DOWNLOAD_NEW:
        print("[INFO] SKIP_DOWNLOAD_NEW=True. Use local raw tile only.")
        for _, row in tile_gdf.iterrows():
            tile_name = row["tile_name"]

            candidates = [
                RAW_TILE_DIR / f"{tile_name}.parquet",
                RAW_TILE_DIR / f"{tile_name}.pq",
            ]
            candidates.extend(sorted(RAW_TILE_DIR.glob(f"*{tile_name}*.parquet")))
            candidates.extend(sorted(RAW_TILE_DIR.glob(f"*{tile_name}*.pq")))

            found = None
            for c in candidates:
                if c.exists() and c.stat().st_size > 1000:
                    found = c
                    break

            if found is None:
                raise FileNotFoundError(
                    "\n[ERROR] SKIP_DOWNLOAD_NEW=True but local tile not found.\n"
                    f"Expected tile: {tile_name}\n"
                    f"Search folder: {RAW_TILE_DIR}\n"
                    "Set SKIP_DOWNLOAD_NEW=False or copy the tile to RAW_TILE_DIR.\n"
                )

            print(f"[OK] Use existing local tile: {found}")
            local_files.append(found)

        return local_files

    if MANUAL_REMOTE_PARQUET_URLS:
        print("[INFO] Download from MANUAL_REMOTE_PARQUET_URLS.")
        for url in MANUAL_REMOTE_PARQUET_URLS:
            out_file = RAW_TILE_DIR / Path(url.split("?")[0]).name
            local_files.append(download_file(url, out_file))
        return local_files

    print("[INFO] Searching Source Cooperative for tile files...")
    all_keys = list_sourcecoop_keys()

    selected_rows = []

    for _, row in tile_gdf.iterrows():
        tile_name = row["tile_name"]
        selected_url = None

        for url in candidate_remote_urls_for_tile(tile_name, all_keys):
            if remote_file_exists(url):
                selected_url = url
                break

        if selected_url is None:
            selected_rows.append({
                "tile_name": tile_name,
                "remote_url": "",
                "local_file": "",
                "status": "not_found",
            })
            continue

        suffix = Path(selected_url.split("?")[0]).suffix
        if suffix.lower() not in [".parquet", ".pq"]:
            suffix = ".parquet"

        out_file = RAW_TILE_DIR / f"{tile_name}{suffix}"
        downloaded = download_file(selected_url, out_file)
        local_files.append(downloaded)

        selected_rows.append({
            "tile_name": tile_name,
            "remote_url": selected_url,
            "local_file": str(downloaded),
            "status": "downloaded",
        })

    pd.DataFrame(selected_rows).to_csv(METADATA_DIR / "selected_tiles_download_status.csv", index=False)

    if not local_files:
        raise RuntimeError("[ERROR] No GBA tile file found or downloaded.")

    return local_files


# ============================================================
# LOW-RAM PARQUET READER
# ============================================================

def detect_geometry_column_from_schema(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    names = pf.schema.names

    for col in ["geometry", "geom", "wkb_geometry", "the_geom"]:
        if col in names:
            return col

    for col in names:
        if "geom" in col.lower():
            return col

    raise ValueError(f"No geometry column found. Columns: {names}")


def detect_height_column_from_schema(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    names = pf.schema.names

    candidates = [
        "height", "height_m", "building_height", "building_height_m",
        "pred_height", "height_mean", "mean_height", "h", "HEIGHT", "Height",
    ]

    for col in candidates:
        if col in names:
            return col

    for col in names:
        if "height" in col.lower():
            return col

    return None


def detect_height_column_in_gdf(gdf):
    candidates = [
        "height", "height_m", "building_height", "building_height_m",
        "pred_height", "height_mean", "mean_height", "h", "HEIGHT", "Height",
    ]

    for col in candidates:
        if col in gdf.columns:
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return col

    for col in gdf.columns:
        if "height" in col.lower():
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return col

    return None


def geometry_series_from_values(values, crs="EPSG:4326"):
    if len(values) == 0:
        return gpd.GeoSeries([], crs=crs)

    first_valid = next((v for v in values if v is not None), None)

    if first_valid is None:
        return gpd.GeoSeries([None] * len(values), crs=crs)

    if isinstance(first_valid, bytes):
        geoms = []
        for v in values:
            if v is None:
                geoms.append(None)
            else:
                try:
                    geoms.append(wkb.loads(v))
                except Exception:
                    geoms.append(None)
        return gpd.GeoSeries(geoms, crs=crs)

    if isinstance(first_valid, str):
        geoms = []
        for v in values:
            if not isinstance(v, str):
                geoms.append(None)
                continue

            try:
                geoms.append(wkt.loads(v))
            except Exception:
                geoms.append(None)

        return gpd.GeoSeries(geoms, crs=crs)

    return gpd.GeoSeries(values, crs=crs)


def normalize_gba_buildings(gdf, height_col=None):
    if gdf.empty:
        return gdf

    gdf = gdf.to_crs("EPSG:4326").copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        return gdf

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    if height_col is None:
        height_col = detect_height_column_in_gdf(gdf)

    if height_col is None:
        gdf["height_m"] = DEFAULT_BUILDING_HEIGHT_M
    else:
        gdf["height_m"] = pd.to_numeric(gdf[height_col], errors="coerce").fillna(DEFAULT_BUILDING_HEIGHT_M)

    gdf["height_m"] = gdf["height_m"].clip(lower=0.0)

    if "gba_id" not in gdf.columns:
        gdf["gba_id"] = [f"GBA{i + 1:08d}" for i in range(len(gdf))]

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)

    gdf["footprint_area_m2"] = gdf_utm.geometry.area.to_numpy()
    gdf["volume_m3"] = gdf["footprint_area_m2"] * gdf["height_m"]

    centroid_utm = gdf_utm.geometry.centroid
    centroid_wgs = gpd.GeoSeries(centroid_utm, crs=utm_crs).to_crs("EPSG:4326")

    gdf["centroid_lon"] = centroid_wgs.x.to_numpy()
    gdf["centroid_lat"] = centroid_wgs.y.to_numpy()

    return gdf


def read_gba_parquet_lowram_bbox(parquet_file, west, south, east, north):
    parquet_file = Path(parquet_file)

    if (
        SAVE_BBOX_FILTER_CACHE
        and BBOX_FILTER_CACHE_FILE.exists()
        and not FORCE_REBUILD_BBOX_CACHE
    ):
        print(f"[SKIP] Use cached bbox-filtered file: {BBOX_FILTER_CACHE_FILE}")
        cached = gpd.read_parquet(BBOX_FILTER_CACHE_FILE)
        if cached.crs is None:
            cached = cached.set_crs("EPSG:4326")
        return cached.to_crs("EPSG:4326")

    print(f"[INFO] Low-RAM read: {parquet_file}")

    pf = pq.ParquetFile(parquet_file)
    geom_col = detect_geometry_column_from_schema(parquet_file)
    height_col = detect_height_column_from_schema(parquet_file)

    print(f"[INFO] Geometry column: {geom_col}")
    print(f"[INFO] Height column: {height_col}")

    columns = [geom_col]

    if height_col is not None:
        columns.append(height_col)

    for col in ["id", "fid", "uid", "osm_id", "quadkey", "source", "confidence", "area", "area_m2"]:
        if col in pf.schema.names and col not in columns:
            columns.append(col)

    if not READ_MINIMUM_COLUMNS_ONLY:
        columns = pf.schema.names

    bbox_geom = box(west, south, east, north)

    selected_batches = []
    total_rows = 0
    kept_rows = 0

    for batch_id, record_batch in enumerate(
        pf.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns),
        start=1,
    ):
        batch_df = record_batch.to_pandas()
        total_rows += len(batch_df)

        if batch_df.empty:
            continue

        geometry = geometry_series_from_values(batch_df[geom_col].tolist(), crs="EPSG:4326")
        batch_gdf = gpd.GeoDataFrame(
            batch_df.drop(columns=[geom_col]),
            geometry=geometry,
            crs="EPSG:4326",
        )

        batch_gdf = batch_gdf[batch_gdf.geometry.notna() & (~batch_gdf.geometry.is_empty)]
        batch_gdf = batch_gdf[batch_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        if batch_gdf.empty:
            del batch_df, geometry, batch_gdf
            gc.collect()
            continue

        batch_gdf = batch_gdf[batch_gdf.intersects(bbox_geom)].copy()

        if batch_gdf.empty:
            del batch_df, geometry, batch_gdf
            gc.collect()
            continue

        batch_gdf = normalize_gba_buildings(batch_gdf, height_col=height_col)

        selected_batches.append(batch_gdf)
        kept_rows += len(batch_gdf)

        print(f"[INFO] Batch {batch_id:05d}: total={total_rows:,}, kept={kept_rows:,}")

        del batch_df, geometry, batch_gdf
        gc.collect()

    if not selected_batches:
        raise RuntimeError("[ERROR] No buildings found in bbox.")

    out_gdf = pd.concat(selected_batches, ignore_index=True)
    out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs="EPSG:4326")
    out_gdf = normalize_gba_buildings(out_gdf, height_col=height_col)

    print(f"[INFO] Total tile rows read: {total_rows:,}")
    print(f"[INFO] Bbox buildings kept: {len(out_gdf):,}")

    if SAVE_BBOX_FILTER_CACHE:
        out_gdf.to_parquet(BBOX_FILTER_CACHE_FILE)
        print(f"[OK] Saved cache: {BBOX_FILTER_CACHE_FILE}")

    return out_gdf


def clip_buildings_to_polygon(gdf, poly_gdf):
    print("[INFO] Clip buildings to Hoa Lac polygon...")
    clipped = gpd.clip(gdf.to_crs("EPSG:4326"), poly_gdf.to_crs("EPSG:4326"))

    if clipped.empty:
        return clipped

    clipped = normalize_gba_buildings(clipped)
    print(f"[INFO] Buildings inside polygon: {len(clipped):,}")

    return clipped


# ============================================================
# EXPORTS
# ============================================================

def clean_attributes_for_gpkg(gdf):
    gdf = gdf.copy()

    for col in gdf.columns:
        if col == "geometry":
            continue

        if gdf[col].dtype == "object":
            def convert_value(v):
                if isinstance(v, (list, tuple, dict, set)):
                    return json.dumps(v, ensure_ascii=False)
                return v

            gdf[col] = gdf[col].apply(convert_value)

    return gdf


def save_gpkg(gdf, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        out_file.unlink()

    clean_attributes_for_gpkg(gdf).to_file(out_file, driver="GPKG")
    print(f"[OK] Saved GPKG: {out_file}")


def save_full_attributes_csv(gdf, out_csv):
    df = gdf.copy()
    df["geometry_wkt"] = df.geometry.to_wkt()
    pd.DataFrame(df.drop(columns="geometry")).to_csv(out_csv, index=False)
    print(f"[OK] Saved CSV: {out_csv}")


def save_centroid_xyz(gdf, out_xyz):
    df = pd.DataFrame({
        "lon": gdf["centroid_lon"],
        "lat": gdf["centroid_lat"],
        "height_m": gdf["height_m"],
    })
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved centroid XYZ: {out_xyz}")


def save_centroid_with_info_xyz(gdf, out_xyz):
    df = pd.DataFrame({
        "gba_id": gdf["gba_id"],
        "lon": gdf["centroid_lon"],
        "lat": gdf["centroid_lat"],
        "height_m": gdf["height_m"],
        "footprint_area_m2": gdf["footprint_area_m2"],
        "volume_m3": gdf["volume_m3"],
    })
    df.to_csv(out_xyz, sep=" ", index=False, header=True, float_format="%.8f")
    print(f"[OK] Saved centroid info XYZ: {out_xyz}")


def save_vertices_xyz(gdf, out_xyz):
    records = []

    for _, row in gdf.to_crs("EPSG:4326").iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for part_id, poly in enumerate(polygons, start=1):
            for vertex_id, (x, y) in enumerate(poly.exterior.coords, start=1):
                records.append((
                    row["gba_id"],
                    x,
                    y,
                    float(row["height_m"]),
                    float(row["footprint_area_m2"]),
                    float(row["volume_m3"]),
                    part_id,
                    vertex_id,
                ))

    df = pd.DataFrame(
        records,
        columns=[
            "gba_id", "lon", "lat", "height_m", "footprint_area_m2",
            "volume_m3", "part_id", "vertex_id",
        ],
    )
    df.to_csv(out_xyz, sep=" ", index=False, header=True, float_format="%.8f")
    print(f"[OK] Saved vertices XYZ: {out_xyz}")


def save_summary(gdf, out_csv):
    if gdf.empty:
        summary = {"n_buildings": 0}
    else:
        summary = {
            "n_buildings": int(len(gdf)),
            "total_footprint_area_m2": float(gdf["footprint_area_m2"].sum()),
            "total_volume_m3": float(gdf["volume_m3"].sum()),
            "height_min_m": float(gdf["height_m"].min()),
            "height_mean_m": float(gdf["height_m"].mean()),
            "height_median_m": float(gdf["height_m"].median()),
            "height_max_m": float(gdf["height_m"].max()),
            "area_min_m2": float(gdf["footprint_area_m2"].min()),
            "area_mean_m2": float(gdf["footprint_area_m2"].mean()),
            "area_median_m2": float(gdf["footprint_area_m2"].median()),
            "area_max_m2": float(gdf["footprint_area_m2"].max()),
        }

    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f"[OK] Saved summary: {out_csv}")

    print("\n========== GBA LoD1 SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")


def write_lod1_obj(gdf, out_obj):
    if gdf.empty:
        print("[WARN] Empty GDF. Skip OBJ.")
        return

    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs()).copy()
    x0, y0 = gdf_utm.total_bounds[0], gdf_utm.total_bounds[1]

    vertices = []
    faces = []

    def add_vertex(x, y, z):
        vertices.append((x - x0, y - y0, z))
        return len(vertices)

    for _, row in gdf_utm.iterrows():
        geom = row.geometry
        height = float(row["height_m"])

        if geom is None or geom.is_empty or height <= 0:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            coords = list(poly.exterior.coords)
            if len(coords) < 4:
                continue

            coords_open = coords[:-1]
            bottom_ids = [add_vertex(x, y, 0.0) for x, y in coords_open]
            top_ids = [add_vertex(x, y, height) for x, y in coords_open]

            faces.append(top_ids)
            faces.append(list(reversed(bottom_ids)))

            n = len(coords_open)
            for i in range(n):
                j = (i + 1) % n
                faces.append([bottom_ids[i], bottom_ids[j], top_ids[j], top_ids[i]])

    with open(out_obj, "w", encoding="utf-8") as f:
        f.write("# GlobalBuildingAtlas LoD1 building model\n")
        f.write("# Local UTM meters, shifted to local origin\n")

        for x, y, z in vertices:
            f.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")

        for face in faces:
            f.write("f " + " ".join(str(idx) for idx in face) + "\n")

    print(f"[OK] Saved OBJ: {out_obj}")


# ============================================================
# PYGMT 2D PLOT HELPERS
# ============================================================

def make_temp_dir():
    return Path(tempfile.mkdtemp(prefix="gba_lod1_tmp_"))


def remove_temp_dir(tmp_dir):
    if CLEAN_TEMP_FILES:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def reduce_gdf_for_plot(gdf, max_features=MAX_POLYGON_PLOT_BUILDINGS):
    if max_features is None or len(gdf) <= max_features:
        return gdf.copy()

    print(f"[INFO] Use largest {max_features:,} buildings for plot only.")
    return gdf.sort_values("footprint_area_m2", ascending=False).head(max_features).copy()


def save_polygon_boundary_xy(poly_gdf, out_xy):
    poly = poly_gdf.to_crs("EPSG:4326").geometry.iloc[0]
    pd.DataFrame(list(poly.exterior.coords)).to_csv(
        out_xy,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )


def save_tile_boundary_segments(tile_gdf, out_xy):
    tile_gdf = tile_gdf.to_crs("EPSG:4326").copy()

    with open(out_xy, "w", encoding="utf-8") as f:
        for _, row in tile_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            f.write(">\n")
            for x, y in geom.exterior.coords:
                f.write(f"{x:.8f} {y:.8f}\n")


def save_building_polygons_for_pygmt(gdf, value_col, out_xy):
    gdf = gdf.to_crs("EPSG:4326").copy()

    with open(out_xy, "w", encoding="utf-8") as f:
        for _, row in gdf.iterrows():
            geom = row.geometry
            value = float(row.get(value_col, 0.0))

            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "Polygon":
                polygons = [geom]
            elif geom.geom_type == "MultiPolygon":
                polygons = list(geom.geoms)
            else:
                continue

            for poly in polygons:
                f.write(f"> -Z{value:.8f}\n")
                for x, y in poly.exterior.coords:
                    f.write(f"{x:.8f} {y:.8f}\n")


def save_building_centroid_for_pygmt(gdf, out_xyz):
    df = pd.DataFrame({
        "lon": gdf["centroid_lon"],
        "lat": gdf["centroid_lat"],
        "height_m": gdf["height_m"],
    })
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")


def make_height_cpt(gdf, out_cpt):
    hmin = float(gdf["height_m"].min())
    hmax = float(gdf["height_m"].max())

    if not np.isfinite(hmin):
        hmin = 0.0
    if not np.isfinite(hmax):
        hmax = 10.0
    if hmax <= hmin:
        hmax = hmin + 1.0

    pygmt.makecpt(cmap="viridis", series=[hmin, hmax], output=str(out_cpt))


def make_area_cpt(gdf, out_cpt):
    log_area = np.log10(gdf["footprint_area_m2"].clip(lower=1.0))
    amin = float(log_area.min())
    amax = float(log_area.max())

    if not np.isfinite(amin):
        amin = 0.0
    if not np.isfinite(amax):
        amax = 1.0
    if amax <= amin:
        amax = amin + 1.0

    pygmt.makecpt(cmap="plasma", series=[amin, amax], output=str(out_cpt))


# ============================================================
# PYGMT 2D PLOTS
# ============================================================

def plot_study_area_and_tiles(poly_gdf, tile_gdf, out_png):
    region = get_plot_region(poly_gdf)
    tmp_dir = make_temp_dir()

    poly_xy = tmp_dir / "hoalac_polygon.xy"
    tile_xy = tmp_dir / "gba_tiles.xy"

    save_polygon_boundary_xy(poly_gdf, poly_xy)
    save_tile_boundary_segments(tile_gdf, tile_xy)

    area_m2, area_km2, area_ha = get_polygon_area_stats(poly_gdf)

    xmin, xmax, ymin, ymax = region
    dx = xmax - xmin
    dy = ymax - ymin

    x_text = xmin + 0.015 * dx
    y_text = ymax - 0.04 * dy
    step = 0.045 * dy

    fig = pygmt.Figure()
    fig.basemap(region=region, projection=PROJECTION, frame=PYGMT_FRAME_MAP)
    fig.plot(data=str(tile_xy), pen="1.2p,black")
    fig.plot(data=str(poly_xy), pen="1.8p,purple")

    for _, row in tile_gdf.iterrows():
        fig.text(
            x=row.geometry.centroid.x,
            y=row.geometry.centroid.y,
            text=row["tile_name"],
            font="9p,Helvetica-Bold,black",
            justify="CM",
        )

    texts = [
        ("Estimated study area", "10p,Helvetica-Bold,black"),
        (f"Area = {area_m2:,.0f} m@+2@+", "9p,Helvetica,black"),
        (f"Area = {area_km2:.4f} km@+2@+", "9p,Helvetica,black"),
        (f"Area = {area_ha:.2f} ha", "9p,Helvetica,black"),
    ]

    for i, (txt, font) in enumerate(texts):
        fig.text(
            x=x_text,
            y=y_text - i * step,
            text=txt,
            font=font,
            justify="TL",
            fill="white@55",
            pen="0.25p,black",
            clearance="0.06c/0.06c",
        )

    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.savefig(str(out_png), dpi=FIG_DPI)
    remove_temp_dir(tmp_dir)
    print(f"[OK] Saved: {out_png}")


def plot_2d_height_map(gdf, poly_gdf, out_png):
    if gdf.empty:
        return

    plot_gdf = reduce_gdf_for_plot(gdf)
    region = get_plot_region(poly_gdf)
    tmp_dir = make_temp_dir()

    poly_xy = tmp_dir / "poly.xy"
    buildings_xy = tmp_dir / "buildings_height.xy"
    centroid_xyz = tmp_dir / "centroid.xyz"
    cpt = tmp_dir / "height.cpt"

    save_polygon_boundary_xy(poly_gdf, poly_xy)
    save_building_polygons_for_pygmt(plot_gdf, "height_m", buildings_xy)
    save_building_centroid_for_pygmt(plot_gdf, centroid_xyz)
    make_height_cpt(gdf, cpt)

    fig = pygmt.Figure()
    fig.basemap(region=region, projection=PROJECTION, frame=PYGMT_FRAME_MAP)
    fig.plot(data=str(buildings_xy), cmap=str(cpt), fill="+z", pen="0.03p,black", transparency=20)
    fig.plot(data=str(centroid_xyz), style="c0.025c", fill="black", pen="0.02p,black", transparency=45)
    fig.plot(data=str(poly_xy), pen="1.5p,purple")
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c/0c",
        frame=["xaf+lBuilding height", "y+l(m)"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.savefig(str(out_png), dpi=FIG_DPI)
    remove_temp_dir(tmp_dir)
    print(f"[OK] Saved: {out_png}")


def plot_2d_area_map(gdf, poly_gdf, out_png):
    if gdf.empty:
        return

    plot_gdf = reduce_gdf_for_plot(gdf)
    plot_gdf["log_area_m2"] = np.log10(plot_gdf["footprint_area_m2"].clip(lower=1.0))

    region = get_plot_region(poly_gdf)
    tmp_dir = make_temp_dir()

    poly_xy = tmp_dir / "poly.xy"
    buildings_xy = tmp_dir / "buildings_area.xy"
    cpt = tmp_dir / "area.cpt"

    save_polygon_boundary_xy(poly_gdf, poly_xy)
    save_building_polygons_for_pygmt(plot_gdf, "log_area_m2", buildings_xy)
    make_area_cpt(plot_gdf, cpt)

    fig = pygmt.Figure()
    fig.basemap(region=region, projection=PROJECTION, frame=PYGMT_FRAME_MAP)
    fig.plot(data=str(buildings_xy), cmap=str(cpt), fill="+z", pen="0.03p,black", transparency=20)
    fig.plot(data=str(poly_xy), pen="1.5p,purple")
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c/0c",
        frame=["xaf+llog10 footprint area", "y+l(m@+2@+)"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.savefig(str(out_png), dpi=FIG_DPI)
    remove_temp_dir(tmp_dir)
    print(f"[OK] Saved: {out_png}")


def plot_height_histogram(gdf, out_png):
    if gdf.empty:
        return

    tmp_dir = make_temp_dir()
    height_txt = tmp_dir / "height_values.txt"

    values = gdf["height_m"].replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        remove_temp_dir(tmp_dir)
        return

    values.to_csv(height_txt, sep=" ", index=False, header=False, float_format="%.8f")

    hmin = 0.0
    hmax = float(values.max())
    if hmax <= 0:
        hmax = 1.0

    bins = 30
    counts, _ = np.histogram(values, bins=bins, range=(hmin, hmax))
    ymax = max(1, int(counts.max() * 1.15))

    mean_h = float(values.mean())
    median_h = float(values.median())

    fig = pygmt.Figure()
    fig.histogram(
        data=str(height_txt),
        region=[hmin, hmax, 0, ymax],
        projection="X14c/8c",
        frame=["xaf+lBuilding height (m)", "yaf+lNumber of buildings", "WSen"],
        series=(hmax - hmin) / bins,
        fill="gray70",
        pen="0.5p,black",
    )
    fig.plot(x=[mean_h, mean_h], y=[0, ymax], pen="1.2p,red,--")
    fig.plot(x=[median_h, median_h], y=[0, ymax], pen="1.2p,blue,.")
    fig.text(x=mean_h, y=ymax * 0.92, text=f"Mean {mean_h:.1f} m", font="9p,Helvetica,red", justify="LM")
    fig.text(x=median_h, y=ymax * 0.70, text=f"Median {median_h:.1f} m", font="9p,Helvetica,blue", justify="LM")

    fig.savefig(str(out_png), dpi=FIG_DPI)
    remove_temp_dir(tmp_dir)
    print(f"[OK] Saved: {out_png}")


# ============================================================
# 3D PLOTS
# ============================================================

def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "\n[ERROR] PyVista is not installed.\n"
            "Install:\n"
            "  conda activate utm\n"
            "  conda install -c conda-forge pyvista vtk trame trame-vtk trame-vuetify\n"
            "Or set PLOT_3D_ENGINE = 'pygmt'.\n"
        ) from exc

    return pv


def select_gdf_for_3d(gdf, max_features=MAX_3D_BUILDINGS):
    if max_features is None or len(gdf) <= max_features:
        return gdf.copy()

    print(f"[INFO] Use largest {max_features:,} buildings for 3D only.")
    return gdf.sort_values("footprint_area_m2", ascending=False).head(max_features).copy()


def build_pyvista_lod1_mesh(gdf):
    """
    Build a real LoD1 3D block mesh using PyVista extrusion.

    This version is more robust than manually creating roof/wall faces.
    For each building footprint:
        1. convert polygon footprint to local UTM meters
        2. create a flat 2D polygon at z = 0
        3. extrude it upward by height_m * PYVISTA_Z_EXAGGERATION
        4. assign real height_m as the color scalar

    Result:
        buildings are real 3D blocks, not only 2D footprints.
    """
    pv = _import_pyvista()

    gdf = select_gdf_for_3d(gdf)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        raise ValueError("[ERROR] Empty GDF for PyVista mesh.")

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs).copy()

    x0, y0 = gdf_utm.total_bounds[0], gdf_utm.total_bounds[1]

    mesh_list = []
    n_buildings_used = 0
    n_parts_used = 0

    for _, row in gdf_utm.iterrows():
        geom = row.geometry
        height_real = float(row.get("height_m", 0.0))

        if geom is None or geom.is_empty:
            continue

        if not np.isfinite(height_real) or height_real <= 0:
            continue

        height_plot = height_real * PYVISTA_Z_EXAGGERATION

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        building_has_valid_part = False

        for poly in polygons:
            coords = list(poly.exterior.coords)

            if len(coords) < 4:
                continue

            coords_open = coords[:-1]

            # PyVista polygon points at z = 0.
            points = np.asarray(
                [[float(x - x0), float(y - y0), 0.0] for x, y in coords_open],
                dtype=float,
            )

            n = len(points)
            if n < 3:
                continue

            # One polygon face: [number_of_vertices, id0, id1, ...]
            faces = np.asarray([n] + list(range(n)), dtype=np.int64)

            footprint = pv.PolyData(points, faces)

            try:
                # capping=True makes roof and bottom closed.
                block = footprint.extrude(
                    [0.0, 0.0, float(height_plot)],
                    capping=True,
                )
            except TypeError:
                # Older PyVista compatibility.
                block = footprint.extrude(
                    [0.0, 0.0, float(height_plot)],
                )

            if block.n_cells <= 0:
                continue

            # Store true building height for colorbar, not exaggerated height.
            block.cell_data["height_m"] = np.full(
                block.n_cells,
                height_real,
                dtype=float,
            )

            mesh_list.append(block)
            n_parts_used += 1
            building_has_valid_part = True

        if building_has_valid_part:
            n_buildings_used += 1

    if not mesh_list:
        raise ValueError("[ERROR] No valid extruded PyVista building blocks created.")

    print(f"[INFO] PyVista extrusion buildings used: {n_buildings_used:,}")
    print(f"[INFO] PyVista extrusion polygon parts used: {n_parts_used:,}")
    print(f"[INFO] PYVISTA_Z_EXAGGERATION = {PYVISTA_Z_EXAGGERATION}")

    # Merge all blocks into one PolyData mesh.
    mesh = mesh_list[0]
    for block in mesh_list[1:]:
        mesh = mesh.merge(block)

    mesh.cell_data["height_m"] = np.asarray(mesh.cell_data["height_m"], dtype=float)

    info = {
        "mesh_method": "extrude",
        "utm_crs": str(utm_crs),
        "x_origin_m": float(x0),
        "y_origin_m": float(y0),
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "z_exaggeration": float(PYVISTA_Z_EXAGGERATION),
        "z_bounds_plot_m": tuple(float(v) for v in mesh.bounds[4:6]),
    }

    return mesh, info, gdf


def build_pyvista_boundary(poly_gdf, reference_gdf):
    pv = _import_pyvista()

    utm_crs = reference_gdf.estimate_utm_crs()
    ref = reference_gdf.to_crs(utm_crs)
    x0, y0 = ref.total_bounds[0], ref.total_bounds[1]

    poly = poly_gdf.to_crs(utm_crs).geometry.iloc[0]
    coords = list(poly.exterior.coords)

    points = [[float(x - x0), float(y - y0), 0.0] for x, y in coords]
    lines = [len(points)] + list(range(len(points)))

    return pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines, dtype=np.int64))


def plot_3d_pyvista(gdf, poly_gdf, out_png):
    pv = _import_pyvista()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    html_file = out_png.with_suffix(".html")
    vtp_file = PROCESSED_DIR / "gba_lod1_buildings_pyvista_mesh.vtp"

    print("[INFO] Build PyVista LoD1 mesh...")
    mesh, info, plot_gdf = build_pyvista_lod1_mesh(gdf)

    for k, v in info.items():
        print(f"[INFO] {k}: {v}")

    print(f"[INFO] PyVista colorbar title: {PYVISTA_COLORBAR_TITLE}")
    print(f"[INFO] PyVista colorbar range: {PYVISTA_HEIGHT_CBAR_RANGE}")

    if PYVISTA_EXPORT_VTP:
        mesh.save(str(vtp_file))
        print(f"[OK] Saved VTP mesh: {vtp_file}")

    plotter = pv.Plotter(off_screen=True, window_size=PYVISTA_WINDOW_SIZE)
    plotter.set_background(PYVISTA_BACKGROUND)

    scalar_bar_args = {
        "title": PYVISTA_COLORBAR_TITLE,
        "vertical": True,
        "position_x": 0.88,
        "position_y": 0.18,
        "width": 0.08,
        "height": 0.60,
        "title_font_size": 18,
        "label_font_size": 14,
    }

    plotter.add_mesh(
        mesh,
        scalars="height_m",
        cmap=PYVISTA_CMAP,
        clim=PYVISTA_HEIGHT_CBAR_RANGE,
        show_edges=PYVISTA_SHOW_EDGES,
        edge_color=PYVISTA_EDGE_COLOR,
        line_width=PYVISTA_EDGE_WIDTH,
        opacity=PYVISTA_BUILDING_OPACITY,
        smooth_shading=False,
        scalar_bar_args=scalar_bar_args,
    )

    if PYVISTA_ADD_BOUNDARY:
        try:
            boundary = build_pyvista_boundary(poly_gdf, plot_gdf)
            plotter.add_mesh(boundary, color=PYVISTA_BOUNDARY_COLOR, line_width=PYVISTA_BOUNDARY_WIDTH)
        except Exception as exc:
            print(f"[WARN] Could not add boundary: {exc}")

    plotter.show_axes()
    plotter.add_text("GlobalBuildingAtlas LoD1 - Hoa Lac", position="upper_left", font_size=14, color="black")

    plotter.add_light(pv.Light(light_type="headlight", intensity=0.9))
    plotter.add_light(pv.Light(position=(0, -1, 1), focal_point=(0, 0, 0), intensity=0.35))

    # Perspective projection and stable camera control.
    # This keeps XY orientation comparable but makes Z blocks visible.
    plotter.disable_parallel_projection()
    plotter.view_isometric()
    plotter.camera.Azimuth(PYVISTA_CAMERA_AZIMUTH)
    plotter.camera.Elevation(PYVISTA_CAMERA_ELEVATION)
    plotter.camera.Zoom(PYVISTA_CAMERA_ZOOM)

    plotter.screenshot(str(out_png))
    print(f"[OK] Saved PyVista PNG: {out_png}")

    if PYVISTA_EXPORT_HTML:
        try:
            plotter.export_html(str(html_file))
            print(f"[OK] Saved PyVista HTML: {html_file}")
        except Exception as exc:
            print(f"[WARN] Could not export HTML: {exc}")

    plotter.close()


def plot_3d_pygmt(gdf, poly_gdf, out_png):
    """
    Simple and fast PyGMT fallback:
        centroid prisms colored by height.
    """
    if gdf.empty:
        return

    plot_gdf = select_gdf_for_3d(gdf)

    region_2d = get_plot_region(poly_gdf)
    hmax = float(gdf["height_m"].max())
    if not np.isfinite(hmax) or hmax <= 0:
        hmax = 10.0

    region_3d = [region_2d[0], region_2d[1], region_2d[2], region_2d[3], 0, hmax * 1.25]

    tmp_dir = make_temp_dir()
    centroid_xyz = tmp_dir / "centroid_3d.xyz"
    poly_xyz = tmp_dir / "poly_3d.xyz"
    cpt = tmp_dir / "height_3d.cpt"

    # centroid file: lon lat z color_value
    pd.DataFrame({
        "lon": plot_gdf["centroid_lon"],
        "lat": plot_gdf["centroid_lat"],
        "z": plot_gdf["height_m"],
        "height_m": plot_gdf["height_m"],
    }).to_csv(centroid_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    poly = poly_gdf.to_crs("EPSG:4326").geometry.iloc[0]
    pd.DataFrame([(x, y, 0.0) for x, y in poly.exterior.coords]).to_csv(
        poly_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    make_height_cpt(gdf, cpt)

    fig = pygmt.Figure()
    fig.basemap(
        region=region_3d,
        projection=PROJECTION,
        perspective=PYGMT_3D_PERSPECTIVE,
        zsize="4c",
        frame=["xaf+lLongitude", "yaf+lLatitude", "zaf+lHeight (m)", "WSenZ"],
    )
    fig.plot3d(
        data=str(centroid_xyz),
        region=region_3d,
        projection=PROJECTION,
        perspective=True,
        style=PYGMT_3D_STYLE,
        cmap=str(cpt),
        fill="+z",
        pen="0.08p,black",
        transparency=PYGMT_3D_TRANSPARENCY,
    )
    fig.plot3d(
        data=str(poly_xyz),
        region=region_3d,
        projection=PROJECTION,
        perspective=True,
        pen="1.5p,purple",
    )
    fig.colorbar(
        cmap=str(cpt),
        perspective=True,
        position="JMR+w6c/0.35c+o0.8c/0c",
        frame=["xaf+lBuilding height", "y+l(m)"],
    )
    fig.savefig(str(out_png), dpi=FIG_DPI)
    remove_temp_dir(tmp_dir)
    print(f"[OK] Saved PyGMT 3D PNG: {out_png}")


def plot_3d(gdf, poly_gdf, out_png):
    engine = str(PLOT_3D_ENGINE).lower().strip()

    if engine == "pyvista":
        plot_3d_pyvista(gdf, poly_gdf, out_png)
        return

    if engine == "pygmt":
        plot_3d_pygmt(gdf, poly_gdf, out_png)
        return

    raise ValueError("PLOT_3D_ENGINE must be 'pyvista' or 'pygmt'.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== GBA LoD1 COMPACT PROCESSOR - HOA LAC ==========")

    poly_gdf = make_hoalac_polygon_gdf()
    save_gpkg(poly_gdf, METADATA_DIR / "hoalac_polygon.gpkg")

    west, south, east, north = get_bbox_from_polygon(poly_gdf, padding_deg=BBOX_PADDING_DEG)
    print(f"[INFO] BBox: W={west}, S={south}, E={east}, N={north}")

    tile_gdf = get_gba_5deg_tiles_for_bbox(west, south, east, north)
    tile_gdf.drop(columns="geometry").to_csv(METADATA_DIR / "selected_gba_5deg_tiles.csv", index=False)
    save_gpkg(tile_gdf, METADATA_DIR / "selected_gba_5deg_tiles.gpkg")

    print("[INFO] Candidate tiles:")
    print(tile_gdf[["tile_name", "lon_min", "lon_max", "lat_min", "lat_max"]])

    plot_study_area_and_tiles(
        poly_gdf,
        tile_gdf,
        FIG_DIR / "00_study_area_and_gba_tiles_pygmt.png",
    )

    parquet_files = resolve_parquet_files(tile_gdf)

    bbox_list = []
    for parquet_file in parquet_files:
        bbox_gdf = read_gba_parquet_lowram_bbox(parquet_file, west, south, east, north)
        if not bbox_gdf.empty:
            bbox_list.append(bbox_gdf)

    if not bbox_list:
        raise RuntimeError("[ERROR] No GBA LoD1 buildings found.")

    bbox_buildings = pd.concat(bbox_list, ignore_index=True)
    bbox_buildings = gpd.GeoDataFrame(bbox_buildings, geometry="geometry", crs="EPSG:4326")
    bbox_buildings = normalize_gba_buildings(bbox_buildings)

    del bbox_list
    gc.collect()

    if WRITE_RAW_BBOX_GPKG:
        save_gpkg(bbox_buildings, PROCESSED_DIR / "gba_lod1_buildings_bbox_raw.gpkg")
    else:
        print("[SKIP] Skip raw bbox GPKG.")

    clipped = clip_buildings_to_polygon(bbox_buildings, poly_gdf)

    del bbox_buildings
    gc.collect()

    if clipped.empty:
        raise RuntimeError("[ERROR] No GBA LoD1 buildings inside Hoa Lac polygon.")

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------
    save_gpkg(clipped, PROCESSED_DIR / "gba_lod1_buildings_hoalac_clipped.gpkg")
    save_full_attributes_csv(clipped, PROCESSED_DIR / "gba_lod1_buildings_full_attributes.csv")
    save_centroid_xyz(clipped, PROCESSED_DIR / "gba_lod1_buildings_centroid_hoalac.xyz")
    save_centroid_with_info_xyz(clipped, PROCESSED_DIR / "gba_lod1_buildings_centroid_hoalac_with_info.xyz")
    save_vertices_xyz(clipped, PROCESSED_DIR / "gba_lod1_buildings_vertices_hoalac.xyz")
    save_summary(clipped, METADATA_DIR / "gba_lod1_summary.csv")

    if EXPORT_OBJ:
        write_lod1_obj(clipped, PROCESSED_DIR / "gba_lod1_buildings_lod1.obj")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    print("\n========== PLOTTING ==========")

    plot_2d_height_map(
        clipped,
        poly_gdf,
        FIG_DIR / "01_gba_lod1_2d_height_overview_pygmt.png",
    )

    plot_2d_area_map(
        clipped,
        poly_gdf,
        FIG_DIR / "02_gba_lod1_2d_footprint_area_pygmt.png",
    )

    plot_height_histogram(
        clipped,
        FIG_DIR / "04_gba_lod1_height_histogram_pygmt.png",
    )

    plot_3d(
        clipped,
        poly_gdf,
        FIG_DIR / "05_gba_lod1_3d_height_overview.png",
    )

    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")

    print("\nImportant files:")
    print(f"  Clipped buildings:   {PROCESSED_DIR / 'gba_lod1_buildings_hoalac_clipped.gpkg'}")
    print(f"  Full CSV:            {PROCESSED_DIR / 'gba_lod1_buildings_full_attributes.csv'}")
    print(f"  Centroid XYZ:        {PROCESSED_DIR / 'gba_lod1_buildings_centroid_hoalac.xyz'}")
    print(f"  Vertices XYZ:        {PROCESSED_DIR / 'gba_lod1_buildings_vertices_hoalac.xyz'}")
    print(f"  OBJ:                 {PROCESSED_DIR / 'gba_lod1_buildings_lod1.obj'}")
    print(f"  Figures:             {FIG_DIR}")

    print("\nFigures:")
    print(f"  00 tile map:         {FIG_DIR / '00_study_area_and_gba_tiles_pygmt.png'}")
    print(f"  01 height map:       {FIG_DIR / '01_gba_lod1_2d_height_overview_pygmt.png'}")
    print(f"  02 area map:         {FIG_DIR / '02_gba_lod1_2d_footprint_area_pygmt.png'}")
    print(f"  04 histogram:        {FIG_DIR / '04_gba_lod1_height_histogram_pygmt.png'}")
    print(f"  05 3D PNG:           {FIG_DIR / '05_gba_lod1_3d_height_overview.png'}")

    if str(PLOT_3D_ENGINE).lower().strip() == "pyvista":
        print(f"  05 3D HTML:          {FIG_DIR / '05_gba_lod1_3d_height_overview.html'}")
        print(f"  05 3D VTP:           {PROCESSED_DIR / 'gba_lod1_buildings_pyvista_mesh.vtp'}")


if __name__ == "__main__":
    main()
