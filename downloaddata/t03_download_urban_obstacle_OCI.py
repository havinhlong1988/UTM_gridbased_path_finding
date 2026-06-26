#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download / load OpenCelliD cellular data for the Hoa Lac area, check which
radio layers exist, and plot all usable layers.

Why this script has two input modes
-----------------------------------
OpenCelliD provides:
  1) API area queries: /cell/getInArea and /cell/getInAreaSize
  2) downloadable country/global CSV files after entering an API Access Token

The area API has pagination and credit limits, while the CSV export can be very
large. This script therefore supports both:

  MODE A: API download by bbox
      - Paste your key into OPENCELLID_API_KEY_IN_SCRIPT near the top, or
        export it as OPENCELLID_API_KEY.
      - The script queries the Hoa Lac bbox for GSM/UMTS/LTE/NR/NBIOT/CDMA.

  MODE B: Local CSV clipping
      - Download Vietnam/global CSV manually from OpenCelliD.
      - Set LOCAL_OPENCELLID_CSV to the file path.
      - The script reads it in chunks and clips to the Hoa Lac polygon.

Main outputs
------------
    output_opencellid_hoalac/
    ├── hoalac_polygon.gpkg
    ├── opencellid_cells_hoalac.gpkg
    ├── opencellid_range_circles_hoalac.gpkg
    ├── opencellid_layer_availability_summary.csv
    ├── opencellid_layer_availability_summary.txt
    ├── xyz/
    │   ├── opencellid_cells_hoalac.xyz
    │   ├── opencellid_<RADIO>_cells_hoalac.xyz
    │   ├── communication_support_grid_hoalac.xyz
    │   └── communication_risk_grid_hoalac.xyz
    └── figures/
        ├── 00_opencellid_all_cells_by_radio.png
        ├── 00a_opencellid_layer_availability_counts.png
        ├── 00b_opencellid_radio_layers_gallery.png
        ├── 01_all_cells_average_signal_dbm.png
        ├── 02_all_cells_range_m.png
        ├── 03_all_cells_samples.png
        ├── 04_range_coverage_circles.png
        ├── 05_communication_support_grid.png
        ├── 06_communication_risk_grid.png
        └── 10_<RADIO>_cells_average_signal_dbm.png

Important modeling note
-----------------------
OpenCelliD is not a clean telecom coverage raster. It is a cell tower / cell
observation database. Use the output as a communication-support/risk layer, not
as a hard no-fly obstacle.

XYZ format
----------
Cell XYZ:
    lon lat value
where value is average_signal_dbm when available, otherwise samples.

Support/risk XYZ:
    lon lat value
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import gzip
import io
import json
import math
import os
import time
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from shapely.geometry import Polygon, Point


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Hoa Lac polygon, format: lon, lat. Used for final clipping.
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

OUTDIR = "output/01_HoaLac_studies_area/opencellid"

# Optional extra padding around the polygon bbox for API query / CSV prefilter.
# 0.01 degree is roughly 1.1 km in latitude.
BBOX_PADDING_DEG = 0.01

# Vietnam MCC.
MCC_FILTER = 452

# Download / load settings.
# If LOCAL_OPENCELLID_CSV is set and exists, the script uses local CSV mode.
# Otherwise, it tries API mode with the key below.
LOCAL_OPENCELLID_CSV = ""  # e.g. "/path/to/452.csv" or "/path/to/cell_towers.csv.gz"

# ----------------------------------------------------------------------
# OpenCelliD API key
# ----------------------------------------------------------------------
# Paste your OpenCelliD API key directly here.
# Example:
#     OPENCELLID_API_KEY_IN_SCRIPT = "1234567890abcdef"
#
# Keep this file private if you paste your real key. Do not commit it to GitHub.
OPENCELLID_API_KEY_IN_SCRIPT = "pk.86f460c15f64a5f16e8fc4a72f608e5d"

# The script prefers the key above. If it is still the placeholder, it falls
# back to the OPENCELLID_API_KEY environment variable.
if OPENCELLID_API_KEY_IN_SCRIPT.strip() and OPENCELLID_API_KEY_IN_SCRIPT.strip() != "PASTE_YOUR_OPENCELLID_API_KEY_HERE":
    OPENCELLID_API_KEY = OPENCELLID_API_KEY_IN_SCRIPT.strip()
else:
    OPENCELLID_API_KEY = os.environ.get("OPENCELLID_API_KEY", "").strip()

# Radio layers to query/check. NR and NBIOT may not exist in the downloaded CSV,
# but the OpenCelliD API supports them.
RADIO_LAYERS = ["GSM", "UMTS", "LTE", "NR", "NBIOT", "CDMA"]

# API settings.
OPENCELLID_BASE_URL = "https://www.opencellid.org"
API_TIMEOUT_SEC = 60
API_SLEEP_SEC = 0.35
API_PAGE_LIMIT = 50  # OpenCelliD max/default is 50 for getInArea.
MAX_CELLS_PER_RADIO = 3000  # safety cap to avoid excessive API credits.
USE_GET_IN_AREA_SIZE_FIRST = True

# CSV chunk settings. Global CSV can be huge, so keep chunks moderate.
CSV_CHUNKSIZE = 500_000

# Coverage / communication support settings.
MAKE_RANGE_CIRCLES = True
MAX_RANGE_FOR_COVERAGE_M = 5000.0  # cap extreme reported range for plotting/modeling
DEFAULT_RANGE_M_IF_MISSING = 1000.0

MAKE_COMMUNICATION_SUPPORT_GRID = True
COMM_GRID_SPACING_M = 250.0
COMM_DISTANCE_DECAY_POWER = 2.0

# Figure settings.
PLOT_FIGURES = True
FIG_DPI = 220
POINT_SIZE = 18
GALLERY_POINT_SIZE = 9

# Whether to show plots interactively. Usually False for batch scripts.
SHOW_PLOTS = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LayerSummary:
    layer: str
    feature_count: int
    status: str
    use_now: str
    voxel_role: str
    value_field: str
    note: str


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def make_hoalac_polygon_gdf() -> gpd.GeoDataFrame:
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_HiTech_Park_approx"]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def get_bbox_from_polygon(poly_gdf: gpd.GeoDataFrame, padding_deg: float = 0.0) -> Tuple[float, float, float, float]:
    west, south, east, north = poly_gdf.total_bounds
    return west - padding_deg, south - padding_deg, east + padding_deg, north + padding_deg


def clip_points_to_polygon(df: pd.DataFrame, poly_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return empty_cells_gdf()

    valid = df["lon"].notna() & df["lat"].notna()
    df = df.loc[valid].copy()
    if df.empty:
        return empty_cells_gdf()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    # Keep points within polygon only, not just bbox.
    poly = poly_gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(gdf, poly).reset_index(drop=True)
    if clipped.empty:
        return empty_cells_gdf()
    return clipped


def estimate_local_utm_crs(gdf: gpd.GeoDataFrame):
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        # Hoa Lac / Hanoi is UTM zone 48N.
        return "EPSG:32648"


# ============================================================
# EMPTY OUTPUT HELPERS
# ============================================================

STANDARD_COLUMNS = [
    "radio", "mcc", "net", "area", "cell", "unit",
    "lon", "lat", "range_m", "samples", "changeable",
    "created", "updated", "average_signal_dbm", "source_method",
]


def empty_cells_gdf() -> gpd.GeoDataFrame:
    data = {
        "radio": pd.Series(dtype="str"),
        "mcc": pd.Series(dtype="Int64"),
        "net": pd.Series(dtype="Int64"),
        "area": pd.Series(dtype="Int64"),
        "cell": pd.Series(dtype="Int64"),
        "unit": pd.Series(dtype="Int64"),
        "lon": pd.Series(dtype="float"),
        "lat": pd.Series(dtype="float"),
        "range_m": pd.Series(dtype="float"),
        "samples": pd.Series(dtype="float"),
        "changeable": pd.Series(dtype="object"),
        "created": pd.Series(dtype="float"),
        "updated": pd.Series(dtype="float"),
        "average_signal_dbm": pd.Series(dtype="float"),
        "source_method": pd.Series(dtype="str"),
    }
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")


def write_empty_gpkg(out_gpkg: Path, geom_type: str = "point") -> None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    empty = empty_cells_gdf()
    empty.to_file(out_gpkg, driver="GPKG")


# ============================================================
# OPENCELLID NORMALIZATION
# ============================================================

def to_numeric_safe(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_opencellid_dataframe(df: pd.DataFrame, source_method: str) -> pd.DataFrame:
    """
    Normalize OpenCelliD CSV/API fields to a consistent schema.
    Handles both CSV fields:
        radio,mcc,net,area,cell,unit,lon,lat,range,samples,changeable,created,updated,averageSignal
    and API JSON fields:
        radio,mcc,mnc,lac,cellid,lon,lat,range,samples,changeable,averageSignalStrength,...
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Rename common API variants.
    rename_map = {
        "mnc": "net",
        "lac": "area",
        "tac": "area",
        "cellid": "cell",
        "cid": "cell",
        "averageSignal": "average_signal_dbm",
        "averageSignalStrength": "average_signal_dbm",
        "range": "range_m",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # If both raw range and normalized range exist, prefer normalized value.
    if "range" in df.columns and "range_m" not in df.columns:
        df["range_m"] = df["range"]

    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df["radio"] = df["radio"].astype(str).str.upper().str.strip()
    df.loc[df["radio"].isin(["", "NAN", "NONE"]), "radio"] = "UNKNOWN"

    numeric_cols = [
        "mcc", "net", "area", "cell", "unit", "lon", "lat",
        "range_m", "samples", "created", "updated", "average_signal_dbm",
    ]
    for c in numeric_cols:
        df[c] = to_numeric_safe(df[c])

    df["source_method"] = source_method

    # Remove invalid coordinates.
    df = df[
        df["lon"].between(-180, 180)
        & df["lat"].between(-90, 90)
        & (df["lon"] != 0)
        & (df["lat"] != 0)
    ].copy()

    # Keep Vietnam MCC if present.
    if MCC_FILTER is not None and "mcc" in df.columns:
        df = df[(df["mcc"].isna()) | (df["mcc"].astype("float") == float(MCC_FILTER))].copy()

    return df[STANDARD_COLUMNS].reset_index(drop=True)


# ============================================================
# LOCAL CSV LOADER
# ============================================================

DEFAULT_OPENCELLID_COLUMNS = [
    "radio", "mcc", "net", "area", "cell", "unit", "lon", "lat",
    "range", "samples", "changeable", "created", "updated", "averageSignal",
]


def open_text_maybe_gzip(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def csv_has_header(path: Path) -> bool:
    with open_text_maybe_gzip(path) as f:
        first_line = f.readline().strip().lower()
    return ("radio" in first_line) and ("mcc" in first_line) and ("lon" in first_line) and ("lat" in first_line)


def load_opencellid_local_csv(local_csv: Path, west: float, south: float, east: float, north: float) -> pd.DataFrame:
    print("\n[INFO] Loading local OpenCelliD CSV in chunks")
    print(f"[INFO] CSV: {local_csv}")
    print(f"[INFO] Prefilter bbox: west={west}, south={south}, east={east}, north={north}")

    if not local_csv.exists():
        raise FileNotFoundError(f"LOCAL_OPENCELLID_CSV does not exist: {local_csv}")

    has_header = csv_has_header(local_csv)
    read_kwargs = dict(chunksize=CSV_CHUNKSIZE, low_memory=False)
    if has_header:
        read_kwargs.update(dict(header=0))
    else:
        read_kwargs.update(dict(header=None, names=DEFAULT_OPENCELLID_COLUMNS))

    collected = []
    total_rows = 0
    kept_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(local_csv, **read_kwargs), start=1):
        total_rows += len(chunk)
        norm = normalize_opencellid_dataframe(chunk, source_method="local_csv")

        if norm.empty:
            continue

        keep = (
            norm["lon"].between(west, east)
            & norm["lat"].between(south, north)
        )
        sub = norm.loc[keep].copy()
        kept_rows += len(sub)
        if not sub.empty:
            collected.append(sub)

        print(f"[INFO] CSV chunk {chunk_idx}: rows={len(chunk):,}, kept_in_bbox={len(sub):,}")

    print(f"[INFO] CSV total rows read: {total_rows:,}")
    print(f"[INFO] CSV rows kept in bbox: {kept_rows:,}")

    if len(collected) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(collected, ignore_index=True)


# ============================================================
# API DOWNLOADER
# ============================================================

def api_get_json(url: str, params: Dict) -> Dict:
    r = requests.get(url, params=params, timeout=API_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"OpenCelliD API HTTP {r.status_code}: {r.text[:500]}")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse OpenCelliD JSON response: {r.text[:500]}") from e
    return data


def get_area_count(api_key: str, bbox_api: str, radio: Optional[str]) -> Optional[int]:
    url = f"{OPENCELLID_BASE_URL}/cell/getInAreaSize"
    params = {
        "key": api_key,
        "BBOX": bbox_api,
        "mcc": MCC_FILTER,
        "format": "json",
    }
    if radio:
        params["radio"] = radio

    try:
        data = api_get_json(url, params)
        if "count" in data:
            return int(data["count"])
        if "cells" in data and isinstance(data["cells"], dict) and "count" in data["cells"]:
            return int(data["cells"]["count"])
    except Exception as e:
        print(f"[WARN] Could not get area count for radio={radio}: {e}")
    return None


def download_api_cells_for_radio(api_key: str, bbox_api: str, radio: str) -> pd.DataFrame:
    print(f"\n[INFO] Querying OpenCelliD API radio={radio}")

    if USE_GET_IN_AREA_SIZE_FIRST:
        count = get_area_count(api_key, bbox_api, radio)
        if count is not None:
            print(f"[INFO] API reports radio={radio} count={count:,}")
            if count == 0:
                return pd.DataFrame(columns=STANDARD_COLUMNS)
    else:
        count = None

    max_to_fetch = MAX_CELLS_PER_RADIO
    if count is not None:
        max_to_fetch = min(max_to_fetch, count)

    url = f"{OPENCELLID_BASE_URL}/cell/getInArea"
    collected = []
    offset = 0

    while offset < max_to_fetch:
        limit = min(API_PAGE_LIMIT, max_to_fetch - offset)
        params = {
            "key": api_key,
            "BBOX": bbox_api,
            "mcc": MCC_FILTER,
            "radio": radio,
            "limit": limit,
            "offset": offset,
            "format": "json",
        }

        try:
            data = api_get_json(url, params)
        except Exception as e:
            print(f"[WARN] API query failed for radio={radio}, offset={offset}: {e}")
            break

        cells = data.get("cells", [])
        if isinstance(cells, dict):
            # Defensive handling if API returns nested object.
            cells = cells.get("cell", [])
        if isinstance(cells, dict):
            cells = [cells]
        if not cells:
            break

        part = normalize_opencellid_dataframe(pd.DataFrame(cells), source_method="api_getInArea")
        if not part.empty:
            collected.append(part)

        print(f"[INFO] radio={radio}, offset={offset}, got={len(cells)}")
        offset += limit
        time.sleep(API_SLEEP_SEC)

        if len(cells) < limit:
            break

    if len(collected) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    out = pd.concat(collected, ignore_index=True)
    out = out.drop_duplicates(subset=["radio", "mcc", "net", "area", "cell", "lon", "lat"]).reset_index(drop=True)
    return out


def download_opencellid_api_bbox(api_key: str, west: float, south: float, east: float, north: float) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError(
            "No OpenCelliD API key found. Paste your key into "
            "OPENCELLID_API_KEY_IN_SCRIPT near the top of this script, "
            "export OPENCELLID_API_KEY, or provide LOCAL_OPENCELLID_CSV."
        )

    # API requires BBOX=<latmin>,<lonmin>,<latmax>,<lonmax>
    bbox_api = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    print("\n[INFO] Downloading OpenCelliD cells by API bbox")
    print(f"[INFO] API BBOX = {bbox_api}")
    print(f"[INFO] MCC filter = {MCC_FILTER}")

    collected = []
    for radio in RADIO_LAYERS:
        part = download_api_cells_for_radio(api_key, bbox_api, radio)
        if not part.empty:
            collected.append(part)

    if len(collected) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    out = pd.concat(collected, ignore_index=True)
    out = out.drop_duplicates(subset=["radio", "mcc", "net", "area", "cell", "lon", "lat"]).reset_index(drop=True)
    return out


# ============================================================
# OUTPUT EXPORTS
# ============================================================

def save_cells_outputs(cells_gdf: gpd.GeoDataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    xyz_dir = outdir / "xyz"
    layers_dir = outdir / "layers"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    cells_file = outdir / "opencellid_cells_hoalac.gpkg"
    if cells_gdf.empty:
        write_empty_gpkg(cells_file)
    else:
        cells_gdf.to_file(cells_file, driver="GPKG")
    print(f"[OK] Saved all cells GPKG: {cells_file}")

    # CSV copy is useful for quick inspection.
    csv_file = outdir / "opencellid_cells_hoalac.csv"
    cells_gdf.drop(columns="geometry", errors="ignore").to_csv(csv_file, index=False)
    print(f"[OK] Saved all cells CSV: {csv_file}")

    save_cells_xyz(cells_gdf, xyz_dir / "opencellid_cells_hoalac.xyz")

    # Per-radio GPKG and XYZ.
    for radio in RADIO_LAYERS + ["UNKNOWN"]:
        sub = cells_gdf[cells_gdf["radio"].astype(str).str.upper() == radio].copy() if not cells_gdf.empty else empty_cells_gdf()
        safe_radio = radio.lower()
        gpkg = layers_dir / f"opencellid_{safe_radio}_cells_hoalac.gpkg"
        if sub.empty:
            write_empty_gpkg(gpkg)
        else:
            sub.to_file(gpkg, driver="GPKG")
        save_cells_xyz(sub, xyz_dir / f"opencellid_{safe_radio}_cells_hoalac.xyz")


def save_cells_xyz(gdf: gpd.GeoDataFrame, out_xyz: Path) -> None:
    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty XYZ saved: {out_xyz}")
        return

    value = gdf["average_signal_dbm"].copy()
    # averageSignal often equals 0 when unavailable, so fall back to samples.
    missing_signal = value.isna() | (value == 0)
    value.loc[missing_signal] = gdf.loc[missing_signal, "samples"]

    df = pd.DataFrame({
        "lon": gdf.geometry.x.to_numpy(),
        "lat": gdf.geometry.y.to_numpy(),
        "value": pd.to_numeric(value, errors="coerce").to_numpy(),
    })
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved XYZ: {out_xyz}")


# ============================================================
# RANGE CIRCLES AND COMMUNICATION SUPPORT GRID
# ============================================================

def make_range_circles(cells_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, outdir: Path) -> gpd.GeoDataFrame:
    out_file = outdir / "opencellid_range_circles_hoalac.gpkg"
    if not MAKE_RANGE_CIRCLES or cells_gdf.empty:
        empty = gpd.GeoDataFrame(cells_gdf.drop(columns="geometry", errors="ignore"), geometry=[], crs="EPSG:4326")
        try:
            empty.to_file(out_file, driver="GPKG")
        except Exception:
            write_empty_gpkg(out_file)
        return empty

    local_crs = estimate_local_utm_crs(poly_gdf)
    cells_m = cells_gdf.to_crs(local_crs).copy()
    poly_m = poly_gdf.to_crs(local_crs)
    poly_union = poly_m.geometry.unary_union

    ranges = pd.to_numeric(cells_m["range_m"], errors="coerce").fillna(DEFAULT_RANGE_M_IF_MISSING)
    ranges = ranges.clip(lower=50.0, upper=MAX_RANGE_FOR_COVERAGE_M)
    cells_m["range_for_plot_m"] = ranges

    geoms = []
    for geom, r in zip(cells_m.geometry, ranges):
        if geom is None or geom.is_empty or not np.isfinite(r):
            geoms.append(None)
            continue
        # Clip each range circle to Hoa Lac polygon to keep plot tidy.
        try:
            geoms.append(geom.buffer(float(r)).intersection(poly_union))
        except Exception:
            geoms.append(None)

    circles_m = cells_m.copy()
    circles_m.geometry = geoms
    circles_m = circles_m[circles_m.geometry.notna() & (~circles_m.geometry.is_empty)].copy()
    circles = circles_m.to_crs("EPSG:4326") if not circles_m.empty else gpd.GeoDataFrame(cells_gdf.drop(columns="geometry", errors="ignore"), geometry=[], crs="EPSG:4326")

    if circles.empty:
        write_empty_gpkg(out_file, geom_type="polygon")
    else:
        circles.to_file(out_file, driver="GPKG")
    print(f"[OK] Saved range circles: {out_file}")
    return circles


def radio_weight(radio: str) -> float:
    r = str(radio).upper()
    if r == "NR":
        return 1.00
    if r == "LTE":
        return 0.90
    if r == "UMTS":
        return 0.60
    if r == "GSM":
        return 0.45
    if r == "NBIOT":
        return 0.35
    if r == "CDMA":
        return 0.40
    return 0.50


def make_communication_support_grid(cells_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, outdir: Path) -> gpd.GeoDataFrame:
    xyz_dir = outdir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    support_xyz = xyz_dir / "communication_support_grid_hoalac.xyz"
    risk_xyz = xyz_dir / "communication_risk_grid_hoalac.xyz"
    csv_file = outdir / "communication_support_grid_hoalac.csv"
    gpkg_file = outdir / "communication_support_grid_hoalac.gpkg"

    if not MAKE_COMMUNICATION_SUPPORT_GRID or cells_gdf.empty:
        for p in [support_xyz, risk_xyz]:
            p.write_text("")
        empty = gpd.GeoDataFrame(
            {"support": pd.Series(dtype="float"), "risk": pd.Series(dtype="float")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_csv(csv_file, index=False)
        empty.to_file(gpkg_file, driver="GPKG")
        return empty

    print("\n[INFO] Building communication support/risk grid")
    local_crs = estimate_local_utm_crs(poly_gdf)
    poly_m = poly_gdf.to_crs(local_crs)
    cells_m = cells_gdf.to_crs(local_crs).copy()

    minx, miny, maxx, maxy = poly_m.total_bounds
    xs = np.arange(minx, maxx + COMM_GRID_SPACING_M, COMM_GRID_SPACING_M)
    ys = np.arange(miny, maxy + COMM_GRID_SPACING_M, COMM_GRID_SPACING_M)
    poly_union = poly_m.geometry.unary_union

    grid_records = []
    for x in xs:
        for y in ys:
            p = Point(float(x), float(y))
            if not poly_union.contains(p):
                continue
            grid_records.append((x, y, p))

    if len(grid_records) == 0:
        print("[WARN] No grid points inside polygon.")
        return gpd.GeoDataFrame(
            {"support": pd.Series(dtype="float"), "risk": pd.Series(dtype="float")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    cell_x = cells_m.geometry.x.to_numpy(dtype=float)
    cell_y = cells_m.geometry.y.to_numpy(dtype=float)
    ranges = pd.to_numeric(cells_m["range_m"], errors="coerce").fillna(DEFAULT_RANGE_M_IF_MISSING).to_numpy(dtype=float)
    ranges = np.clip(ranges, 100.0, MAX_RANGE_FOR_COVERAGE_M)

    samples = pd.to_numeric(cells_m["samples"], errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
    sample_weight = np.log1p(samples)
    if np.nanmax(sample_weight) > 0:
        sample_weight = sample_weight / np.nanmax(sample_weight)
    sample_weight = np.clip(sample_weight, 0.20, 1.00)

    sig = pd.to_numeric(cells_m["average_signal_dbm"], errors="coerce").to_numpy(dtype=float)
    # averageSignal=0 usually means missing. Set neutral 0.5.
    signal_weight = np.where(np.isfinite(sig) & (sig != 0), (sig + 120.0) / 60.0, 0.50)
    signal_weight = np.clip(signal_weight, 0.05, 1.00)

    r_weight = np.array([radio_weight(r) for r in cells_m["radio"].astype(str)], dtype=float)
    base_weight = sample_weight * signal_weight * r_weight

    out_points = []
    for x, y, geom in grid_records:
        dx = cell_x - x
        dy = cell_y - y
        d = np.sqrt(dx * dx + dy * dy)
        # Ignore cells far beyond their capped range.
        active = d <= ranges
        if not np.any(active):
            support = 0.0
        else:
            decay = np.exp(-((d[active] / ranges[active]) ** COMM_DISTANCE_DECAY_POWER))
            vals = base_weight[active] * decay
            support = float(np.nanmax(vals)) if vals.size else 0.0
        support = float(np.clip(support, 0.0, 1.0))
        risk = 1.0 - support
        out_points.append((support, risk, geom))

    grid_m = gpd.GeoDataFrame(
        {
            "support": [r[0] for r in out_points],
            "risk": [r[1] for r in out_points],
        },
        geometry=[r[2] for r in out_points],
        crs=local_crs,
    )
    grid = grid_m.to_crs("EPSG:4326")

    grid.to_file(gpkg_file, driver="GPKG")
    grid.drop(columns="geometry").assign(lon=grid.geometry.x, lat=grid.geometry.y).to_csv(csv_file, index=False)

    pd.DataFrame({
        "lon": grid.geometry.x,
        "lat": grid.geometry.y,
        "support": grid["support"],
    }).to_csv(support_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    pd.DataFrame({
        "lon": grid.geometry.x,
        "lat": grid.geometry.y,
        "risk": grid["risk"],
    }).to_csv(risk_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    print(f"[OK] Saved support grid GPKG: {gpkg_file}")
    print(f"[OK] Saved support XYZ: {support_xyz}")
    print(f"[OK] Saved risk XYZ: {risk_xyz}")
    return grid


# ============================================================
# AVAILABILITY SUMMARY
# ============================================================

def layer_status(count: int) -> str:
    return "available" if int(count) > 0 else "missing_or_empty"


def use_now_from_count(count: int, caution: bool = False) -> str:
    if count <= 0:
        return "no"
    return "yes_with_caution" if caution else "yes"


def build_availability_summary(
    cells_gdf: gpd.GeoDataFrame,
    circles_gdf: gpd.GeoDataFrame,
    support_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: List[LayerSummary] = []

    total_count = 0 if cells_gdf.empty else len(cells_gdf)
    rows.append(LayerSummary(
        layer="all_cells",
        feature_count=total_count,
        status=layer_status(total_count),
        use_now=use_now_from_count(total_count, caution=True),
        voxel_role="communication support / communication risk input",
        value_field="radio, range_m, samples, average_signal_dbm",
        note="Cell observations/estimated tower positions; not a direct coverage raster.",
    ))

    for radio in RADIO_LAYERS:
        count = 0 if cells_gdf.empty else int((cells_gdf["radio"].astype(str).str.upper() == radio).sum())
        rows.append(LayerSummary(
            layer=f"radio_{radio}",
            feature_count=count,
            status=layer_status(count),
            use_now=use_now_from_count(count, caution=True),
            voxel_role="radio-specific communication support layer",
            value_field="average_signal_dbm or samples",
            note=f"{radio} cells available inside Hoa Lac polygon." if count else f"No {radio} cells found inside Hoa Lac polygon.",
        ))

    if cells_gdf.empty:
        signal_count = 0
        range_count = 0
        sample_count = 0
    else:
        sig = pd.to_numeric(cells_gdf["average_signal_dbm"], errors="coerce")
        signal_count = int((sig.notna() & (sig != 0)).sum())
        range_count = int(pd.to_numeric(cells_gdf["range_m"], errors="coerce").notna().sum())
        sample_count = int(pd.to_numeric(cells_gdf["samples"], errors="coerce").notna().sum())

    rows.extend([
        LayerSummary(
            layer="average_signal_dbm",
            feature_count=signal_count,
            status=layer_status(signal_count),
            use_now=use_now_from_count(signal_count, caution=True),
            voxel_role="communication quality weighting if values are real dBm",
            value_field="average_signal_dbm",
            note="OpenCelliD averageSignal can be missing or 0; verify before treating as measured signal strength.",
        ),
        LayerSummary(
            layer="range_m",
            feature_count=range_count,
            status=layer_status(range_count),
            use_now=use_now_from_count(range_count, caution=True),
            voxel_role="approximate coverage radius / search buffer",
            value_field="range_m",
            note="Range is an estimate, not guaranteed cellular service radius; script caps it for modeling.",
        ),
        LayerSummary(
            layer="samples",
            feature_count=sample_count,
            status=layer_status(sample_count),
            use_now=use_now_from_count(sample_count, caution=False),
            voxel_role="confidence weight",
            value_field="samples",
            note="Higher sample count means stronger confidence in the cell position estimate.",
        ),
        LayerSummary(
            layer="range_coverage_circles",
            feature_count=0 if circles_gdf is None or circles_gdf.empty else len(circles_gdf),
            status=layer_status(0 if circles_gdf is None or circles_gdf.empty else len(circles_gdf)),
            use_now=use_now_from_count(0 if circles_gdf is None or circles_gdf.empty else len(circles_gdf), caution=True),
            voxel_role="rough communication coverage support, not hard obstacle",
            value_field="range_for_plot_m",
            note="Generated from OpenCelliD range_m; clipped to Hoa Lac polygon.",
        ),
        LayerSummary(
            layer="communication_support_grid",
            feature_count=0 if support_grid is None or support_grid.empty else len(support_grid),
            status=layer_status(0 if support_grid is None or support_grid.empty else len(support_grid)),
            use_now=use_now_from_count(0 if support_grid is None or support_grid.empty else len(support_grid), caution=True),
            voxel_role="soft cost/risk input for UAV pathfinding",
            value_field="support, risk",
            note="Estimated from distance, radio type, samples, range, and average signal where available.",
        ),
    ])

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def save_summary(summary_df: pd.DataFrame, outdir: Path) -> None:
    csv_file = outdir / "opencellid_layer_availability_summary.csv"
    txt_file = outdir / "opencellid_layer_availability_summary.txt"
    summary_df.to_csv(csv_file, index=False)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("OPENCELLID LAYER AVAILABILITY SUMMARY\n")
        f.write("======================================\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"Layer        : {row['layer']}\n")
            f.write(f"Count        : {row['feature_count']}\n")
            f.write(f"Status       : {row['status']}\n")
            f.write(f"Use now      : {row['use_now']}\n")
            f.write(f"Voxel role   : {row['voxel_role']}\n")
            f.write(f"Value field  : {row['value_field']}\n")
            f.write(f"Note         : {row['note']}\n")
            f.write("\n")

    print(f"[OK] Saved summary CSV: {csv_file}")
    print(f"[OK] Saved summary TXT: {txt_file}")


# ============================================================
# PLOTTING
# ============================================================

def setup_ax(ax, title: str, poly_gdf: gpd.GeoDataFrame):
    poly_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)


def save_or_show(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[OK] Saved figure: {out_png}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_all_cells_by_radio(cells_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, "OpenCelliD cells by radio layer", poly_gdf)

    if cells_gdf.empty:
        ax.text(0.5, 0.5, "No OpenCelliD cells found", transform=ax.transAxes, ha="center", va="center")
    else:
        radios = sorted(cells_gdf["radio"].astype(str).unique())
        for radio in radios:
            sub = cells_gdf[cells_gdf["radio"].astype(str) == radio]
            sub.plot(ax=ax, markersize=POINT_SIZE, label=f"{radio} ({len(sub)})", alpha=0.85)
        ax.legend(loc="best", fontsize=8)

    save_or_show(fig, out_png)


def plot_availability_counts(summary_df: pd.DataFrame, out_png: Path):
    plot_df = summary_df[summary_df["layer"].str.startswith("radio_")].copy()
    if plot_df.empty:
        plot_df = summary_df.copy()
    plot_df["feature_count"] = pd.to_numeric(plot_df["feature_count"], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_df["layer"], plot_df["feature_count"])
    ax.set_title("OpenCelliD layer availability counts")
    ax.set_ylabel("Feature count")
    ax.set_xlabel("Layer")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    save_or_show(fig, out_png)


def plot_radio_gallery(cells_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, out_png: Path):
    radios = RADIO_LAYERS
    n = len(radios)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, radio in zip(axes, radios):
        setup_ax(ax, f"{radio}", poly_gdf)
        if not cells_gdf.empty:
            sub = cells_gdf[cells_gdf["radio"].astype(str).str.upper() == radio]
        else:
            sub = empty_cells_gdf()

        if sub.empty:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center")
        else:
            sub.plot(ax=ax, markersize=GALLERY_POINT_SIZE, alpha=0.8)
            ax.text(0.02, 0.98, f"n={len(sub)}", transform=ax.transAxes, ha="left", va="top", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    save_or_show(fig, out_png)


def plot_points_by_value(
    gdf: gpd.GeoDataFrame,
    poly_gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    out_png: Path,
    cmap: str = "viridis",
):
    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, title, poly_gdf)

    if gdf.empty or value_col not in gdf.columns:
        ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center")
        save_or_show(fig, out_png)
        return

    vals = pd.to_numeric(gdf[value_col], errors="coerce")
    valid = vals.notna()
    # For signal, 0 often means missing; remove from colorbar plot.
    if value_col == "average_signal_dbm":
        valid = valid & (vals != 0)

    if valid.sum() == 0:
        gdf.plot(ax=ax, markersize=POINT_SIZE, alpha=0.6)
        ax.text(0.5, 0.08, f"No valid {value_col} values; plotted locations only", transform=ax.transAxes, ha="center")
        save_or_show(fig, out_png)
        return

    sub = gdf.loc[valid].copy()
    sub[value_col] = vals.loc[valid]
    sub.plot(
        ax=ax,
        column=value_col,
        markersize=POINT_SIZE,
        legend=True,
        cmap=cmap,
        alpha=0.9,
        legend_kwds={"label": colorbar_label, "shrink": 0.72},
    )
    ax.text(0.02, 0.98, f"n={len(sub)}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    save_or_show(fig, out_png)


def plot_range_circles(circles_gdf: gpd.GeoDataFrame, cells_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, "OpenCelliD approximate range circles", poly_gdf)

    if circles_gdf is None or circles_gdf.empty:
        ax.text(0.5, 0.5, "No range circles", transform=ax.transAxes, ha="center", va="center")
    else:
        if "range_for_plot_m" in circles_gdf.columns:
            circles_gdf.plot(
                ax=ax,
                column="range_for_plot_m",
                cmap="viridis",
                alpha=0.28,
                legend=True,
                legend_kwds={"label": "Range used for plot/model (m)", "shrink": 0.72},
            )
        else:
            circles_gdf.plot(ax=ax, alpha=0.28)
        if not cells_gdf.empty:
            cells_gdf.plot(ax=ax, color="black", markersize=5, alpha=0.55)

    save_or_show(fig, out_png)


def plot_grid_value(grid_gdf: gpd.GeoDataFrame, poly_gdf: gpd.GeoDataFrame, value_col: str, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, title, poly_gdf)

    if grid_gdf is None or grid_gdf.empty or value_col not in grid_gdf.columns:
        ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center", va="center")
    else:
        grid_gdf.plot(
            ax=ax,
            column=value_col,
            markersize=28,
            cmap="viridis",
            legend=True,
            legend_kwds={"label": value_col, "shrink": 0.72},
        )

    save_or_show(fig, out_png)


def plot_all_figures(
    cells_gdf: gpd.GeoDataFrame,
    circles_gdf: gpd.GeoDataFrame,
    support_grid: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    poly_gdf: gpd.GeoDataFrame,
    outdir: Path,
) -> None:
    if not PLOT_FIGURES:
        return

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_all_cells_by_radio(cells_gdf, poly_gdf, figdir / "00_opencellid_all_cells_by_radio.png")
    plot_availability_counts(summary_df, figdir / "00a_opencellid_layer_availability_counts.png")
    plot_radio_gallery(cells_gdf, poly_gdf, figdir / "00b_opencellid_radio_layers_gallery.png")

    plot_points_by_value(
        cells_gdf, poly_gdf,
        value_col="average_signal_dbm",
        title="OpenCelliD average signal strength",
        colorbar_label="Average signal (dBm or source-defined)",
        out_png=figdir / "01_all_cells_average_signal_dbm.png",
        cmap="viridis",
    )
    plot_points_by_value(
        cells_gdf, poly_gdf,
        value_col="range_m",
        title="OpenCelliD estimated cell range",
        colorbar_label="Range (m)",
        out_png=figdir / "02_all_cells_range_m.png",
        cmap="viridis",
    )
    plot_points_by_value(
        cells_gdf, poly_gdf,
        value_col="samples",
        title="OpenCelliD sample count",
        colorbar_label="Samples", 
        out_png=figdir / "03_all_cells_samples.png",
        cmap="viridis",
    )
    plot_range_circles(circles_gdf, cells_gdf, poly_gdf, figdir / "04_range_coverage_circles.png")
    plot_grid_value(support_grid, poly_gdf, "support", "Estimated communication support grid", figdir / "05_communication_support_grid.png")
    plot_grid_value(support_grid, poly_gdf, "risk", "Estimated communication risk grid", figdir / "06_communication_risk_grid.png")

    for idx, radio in enumerate(RADIO_LAYERS, start=10):
        sub = cells_gdf[cells_gdf["radio"].astype(str).str.upper() == radio].copy() if not cells_gdf.empty else empty_cells_gdf()
        plot_points_by_value(
            sub,
            poly_gdf,
            value_col="average_signal_dbm",
            title=f"OpenCelliD {radio} cells: average signal",
            colorbar_label="Average signal (dBm or source-defined)",
            out_png=figdir / f"{idx:02d}_{radio.lower()}_cells_average_signal_dbm.png",
            cmap="viridis",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== OPENCELLID HOA LAC DOWNLOAD / LAYER CHECK ==========")
    print("[INFO] OpenCelliD input modes: local CSV or API bbox")
    print(f"[INFO] Output folder: {outdir.resolve()}")

    poly_gdf = make_hoalac_polygon_gdf()
    poly_file = outdir / "hoalac_polygon.gpkg"
    poly_gdf.to_file(poly_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {poly_file}")

    west, south, east, north = get_bbox_from_polygon(poly_gdf, BBOX_PADDING_DEG)
    print("\n[INFO] Download/load bbox:")
    print(f"  WEST  = {west:.8f}")
    print(f"  SOUTH = {south:.8f}")
    print(f"  EAST  = {east:.8f}")
    print(f"  NORTH = {north:.8f}")

    local_csv = Path(LOCAL_OPENCELLID_CSV).expanduser() if LOCAL_OPENCELLID_CSV else None

    if local_csv and local_csv.exists():
        raw_df = load_opencellid_local_csv(local_csv, west, south, east, north)
    else:
        raw_df = download_opencellid_api_bbox(OPENCELLID_API_KEY, west, south, east, north)

    print(f"\n[INFO] Raw cells from source/bbox: {len(raw_df):,}")
    cells_gdf = clip_points_to_polygon(raw_df, poly_gdf)
    print(f"[INFO] Cells inside Hoa Lac polygon: {len(cells_gdf):,}")

    if not cells_gdf.empty:
        # Clean and sort.
        cells_gdf["radio"] = cells_gdf["radio"].astype(str).str.upper()
        cells_gdf = cells_gdf.drop_duplicates(subset=["radio", "mcc", "net", "area", "cell", "lon", "lat"]).reset_index(drop=True)

    save_cells_outputs(cells_gdf, outdir)

    circles_gdf = make_range_circles(cells_gdf, poly_gdf, outdir)
    support_grid = make_communication_support_grid(cells_gdf, poly_gdf, outdir)

    summary_df = build_availability_summary(cells_gdf, circles_gdf, support_grid)
    save_summary(summary_df, outdir)

    plot_all_figures(cells_gdf, circles_gdf, support_grid, summary_df, poly_gdf, outdir)

    print("\n========== LAYER USABILITY SUMMARY ==========")
    print(summary_df[["layer", "feature_count", "status", "use_now", "voxel_role"]].to_string(index=False))

    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")
    print("\nImportant files:")
    print(f"  Cells GPKG:          {outdir / 'opencellid_cells_hoalac.gpkg'}")
    print(f"  Cells CSV:           {outdir / 'opencellid_cells_hoalac.csv'}")
    print(f"  Range circles GPKG:  {outdir / 'opencellid_range_circles_hoalac.gpkg'}")
    print(f"  Summary CSV:         {outdir / 'opencellid_layer_availability_summary.csv'}")
    print(f"  Figures folder:      {outdir / 'figures'}")

    if not OPENCELLID_API_KEY and not (local_csv and local_csv.exists()):
        print("\n[NOTE] For API mode, export your OpenCelliD token first:")
        print("       Paste your token into OPENCELLID_API_KEY_IN_SCRIPT near the top of this script")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
