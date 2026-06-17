#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot and check downloaded / copied Hoa Lac input data only.

IMPORTANT
---------
This script does NOT read or use any path-finding model files such as:
    output/02_senario1_no_velocity/raw.xyz
    output/**/mixed_model*.xyz
    output/**/*slowness*.xyz

It only scans and plots downloaded input data under:
    output/01_HoaLac_studies_area/globalbuildingatlas_lod1
    output/01_HoaLac_studies_area/openbuildingmap
    output/01_HoaLac_studies_area/opentopography
    output/01_HoaLac_studies_area/osm

Figures and reports are written to:
    figures/01_HoaLac_studies_area_input

Main outputs
------------
    00_input_data_inventory.csv
    00_input_data_report.txt
    01_input_data_overview_map.png

Dataset figures:
    globalbuildingatlas_lod1/
    openbuildingmap/
    opentopography/
    osm/

Recommended run:
    conda activate utm
    python 01_plot_hoalac_downloaded_input_data_with_gba.py
"""

from __future__ import annotations
import os
from pathlib import Path
import math
import shutil
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt

from shapely.geometry import Polygon, box


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_ROOT = Path("output/01_HoaLac_studies_area")
FIG_ROOT = Path("figures/01_HoaLac_studies_area_input")

DATASET_DIRS = {
    "globalbuildingatlas_lod1": INPUT_ROOT / "globalbuildingatlas_lod1",
    "openbuildingmap": INPUT_ROOT / "openbuildingmap",
    "opentopography": INPUT_ROOT / "opentopography",
    "osm": INPUT_ROOT / "osm",
}

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
PROJECTION = "M15c"
DPI = 300

# Data inventory / quicklook settings.
TOUCH_ALL_FILES = True
MAX_QUICKLOOK_POINTS = 250_000
MAX_POLYGON_PLOT_FEATURES = 30_000
MAX_OVERVIEW_POLYGONS = 10_000
MAX_OVERVIEW_POINTS = 80_000
MAX_RASTER_QUICKLOOK_POINTS = 250_000
MAX_RASTER_OVERVIEW_POINTS = 80_000

# Dataset overview maps.
DATASET_OVERVIEW_BUILDING_ALPHA = 65
DATASET_OVERVIEW_POINT_ALPHA = 25

# Low-memory OpenTopography surface settings.
# True  = smooth surface: blockmean -> surface -> grdimage
# False = fallback dots: plot sampled points directly
PLOT_TOPO_SURFACE = True
MAX_TOPO_SURFACE_POINTS = 120_000
TOPO_SURFACE_SPACING = "0.0002"
TOPO_SURFACE_TENSION = 0.35
TOPO_SURFACE_CMAP = "geo"
TOPO_HILLSHADE_CMAP = "gray"

# OpenTopography spatial plotting settings.
# Summary/statistic files are still inventoried, but are not plotted as maps.
OPENTOPOGRAPHY_SKIP_SPATIAL_TABLE_NAME_KEYWORDS = [
    "summary", "statistics", "statistic", "metadata", "readme", "tile",
]
OPENTOPOGRAPHY_MIN_SPATIAL_ROWS = 20
OPENTOPOGRAPHY_ASSUMED_PROJECTED_CRS = "EPSG:32648"  # Hoa Lac / Hanoi is UTM zone 48N.

# Avoid accidental use of path-finding model outputs.
FORBIDDEN_MODEL_NAME_PATTERNS = [
    "raw.xyz",
    "mixed_model",
    "slowness",
    "pathfinding",
    "for_pathfinding",
    "collision",
]

# GlobalBuildingAtlas plotting.
PLOT_GLOBALBUILDINGATLAS_LOD1 = True
GBA_PLOT_3D = True
GBA_PLOT_3D_ENGINE = "pygmt"       # "pygmt" or "pyvista"
GBA_FALLBACK_TO_PYGMT = True
GBA_DEFAULT_BUILDING_HEIGHT_M = 6.0
GBA_MAX_3D_BUILDINGS = 3_000

# PyVista options for GBA 3D.
GBA_PYVISTA_EXPORT_HTML = True
GBA_PYVISTA_EXPORT_VTP = True
GBA_PYVISTA_BACKGROUND = "white"
GBA_PYVISTA_WINDOW_SIZE = [2200, 1600]
GBA_PYVISTA_CMAP = "viridis"
GBA_PYVISTA_HEIGHT_CBAR_RANGE = [0.0, 40.0]  # None for auto
GBA_PYVISTA_Z_EXAGGERATION = 30.0
GBA_PYVISTA_BUILDING_OPACITY = 0.90
GBA_PYVISTA_SHOW_EDGES = True
GBA_PYVISTA_EDGE_COLOR = "black"
GBA_PYVISTA_EDGE_WIDTH = 0.25
GBA_PYVISTA_CAMERA_AZIMUTH = 205
GBA_PYVISTA_CAMERA_ELEVATION = 5
GBA_PYVISTA_CAMERA_ZOOM = 1.20

REMOVE_OUTSIDE_OSM = True

# Style.
AOI_PEN = "1.6p,purple"
AOI_FILL = None
OSM_LINE_PEN = "0.7p,blue"
OBM_PEN = "0.05p,black"
GBA_PEN = "0.05p,black"

OSM_EXTRA_FEATURE_STYLES = {
    "water": {
        "poly_fill": "lightblue@35",
        "poly_pen": "0.5p,cornflowerblue",
        "line_pen": "0.9p,cornflowerblue",
        "point_fill": "lightblue",
        "point_pen": "0.1p,cornflowerblue",
    },
    "waterway": {
        "poly_fill": None,
        "poly_pen": "0.5p,blue",
        "line_pen": "1.0p,blue",
        "point_fill": "blue",
        "point_pen": "0.1p,blue",
    },
    "natural": {
        "poly_fill": "darkseagreen@35",
        "poly_pen": "0.4p,darkseagreen4",
        "line_pen": "0.7p,darkseagreen4",
        "point_fill": "darkseagreen4",
        "point_pen": "0.1p,darkseagreen4",
    },
    "landuse": {
        "poly_fill": "palegreen@30",
        "poly_pen": "0.4p,green4",
        "line_pen": "0.7p,green4",
        "point_fill": "green4",
        "point_pen": "0.1p,green4",
    },
    "railway": {
        "poly_fill": None,
        "poly_pen": "0.5p,black",
        "line_pen": "1.0p,black",
        "point_fill": "black",
        "point_pen": "0.1p,black",
    },
    "amenity": {
        "poly_fill": "khaki@45",
        "poly_pen": "0.4p,goldenrod3",
        "line_pen": "0.7p,goldenrod3",
        "point_fill": "gold",
        "point_pen": "0.1p,black",
    },
    "man_made": {
        "poly_fill": "gray80@50",
        "poly_pen": "0.4p,gray40",
        "line_pen": "0.7p,gray40",
        "point_fill": "gray40",
        "point_pen": "0.1p,gray20",
    },
    "leisure": {
        "poly_fill": "cyan@25",
        "poly_pen": "0.4p,cyan3",
        "line_pen": "0.7p,cyan3",
        "point_fill": "cyan3",
        "point_pen": "0.1p,cyan4",
    },
    "barrier": {
        "poly_fill": None,
        "poly_pen": "0.5p,red3",
        "line_pen": "0.9p,red3",
        "point_fill": "red3",
        "point_pen": "0.1p,red4",
    },
    "building": {
        "poly_fill": "mistyrose@40",
        "poly_pen": "0.35p,gray40",
        "line_pen": "0.6p,gray40",
        "point_fill": "gray30",
        "point_pen": "0.1p,black",
    },
    "other": {
        "poly_fill": "gray90@35",
        "poly_pen": "0.3p,gray50",
        "line_pen": "0.5p,gray50",
        "point_fill": "gray50",
        "point_pen": "0.1p,gray30",
    },
}

OSM_EXTRA_FEATURE_ORDER = [
    "water",
    "waterway",
    "natural",
    "landuse",
    "railway",
    "amenity",
    "man_made",
    "leisure",
    "barrier",
    "building",
    "other",
]

# Marking style for data outside the Hoa Lac polygon.
# The data are NOT dropped. Objects inside/intersecting the polygon are plotted normally;
# objects outside the polygon but still inside the map region are plotted in grey/faint style.
# Raster/topography outside the polygon is plotted as pure white, then the inside polygon is overlaid in color.
MARK_OUTSIDE_AOI = True
AOI_INSIDE_TEST = "intersects"   # "intersects" or "within"
OUTSIDE_AOI_FILL = "gray85@60"
OUTSIDE_AOI_PEN = "0.12p,gray55"
OUTSIDE_AOI_LINE_PEN = "0.45p,gray60"
OUTSIDE_AOI_POINT_FILL = "gray65"
OUTSIDE_AOI_POINT_TRANSPARENCY = 65
OUTSIDE_RASTER_TRANSPARENCY = 100
OUTSIDE_RASTER_CMAP = "white"
STATS_USE_INSIDE_AOI_ONLY = True

# Temporary files.
CLEAN_TEMP_FILES = True


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for key in DATASET_DIRS:
        (FIG_ROOT / key).mkdir(parents=True, exist_ok=True)


def make_temp_dir(prefix: str = "hoalac_input_plot_tmp_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def remove_temp_dir(tmp_dir: Path) -> None:
    if CLEAN_TEMP_FILES:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def get_aoi_polygon() -> Polygon:
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def get_aoi_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_study_area"]},
        geometry=[get_aoi_polygon()],
        crs="EPSG:4326",
    )


def get_aoi_area_stats() -> tuple[float, float, float]:
    """Return Hoa Lac polygon area in m2, km2, and ha."""
    aoi = get_aoi_gdf()
    try:
        aoi_utm = aoi.to_crs(aoi.estimate_utm_crs())
        area_m2 = float(aoi_utm.geometry.area.iloc[0])
    except Exception as exc:
        print(f"[WARN] Could not estimate AOI area in UTM: {exc}")
        area_m2 = float("nan")
    return area_m2, area_m2 / 1.0e6, area_m2 / 1.0e4


def add_aoi_area_text_box(fig: pygmt.Figure, region: list[float]) -> None:
    """Add AOI area information to a map, matching the old GBA tile-plot style."""
    area_m2, area_km2, area_ha = get_aoi_area_stats()
    xmin, xmax, ymin, ymax = region
    dx = xmax - xmin
    dy = ymax - ymin

    x_text = xmin + 0.015 * dx
    y_text = ymax - 0.040 * dy
    step = 0.045 * dy

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
            fill="white@72",
            pen="0.55p,black",
            clearance="0.08c/0.08c",
        )


def get_region_from_aoi(padding: float = REGION_PADDING) -> list[float]:
    aoi = get_aoi_gdf()
    west, south, east, north = aoi.total_bounds
    return [west - padding, east + padding, south - padding, north + padding]


def save_aoi_xy(out_xy: Path) -> None:
    pd.DataFrame(HOALAC_POLYGON, columns=["lon", "lat"]).to_csv(
        out_xy,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )


def start_map(region: list[float], title: str) -> pygmt.Figure:
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
        frame=[f'WSne+t"{title}"', "xaf+lLongitude", "yaf+lLatitude"],
    )
    return fig


def plot_aoi_boundary(fig: pygmt.Figure, label: str | None = "Hoa Lac boundary") -> None:
    xs = [p[0] for p in HOALAC_POLYGON]
    ys = [p[1] for p in HOALAC_POLYGON]
    kwargs = {"x": xs, "y": ys, "pen": AOI_PEN, "fill": AOI_FILL}
    if label:
        kwargs["label"] = label
    fig.plot(**kwargs)


def safe_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def is_forbidden_model_file(path: Path) -> bool:
    name = path.name.lower()
    full = str(path).lower()
    for pattern in FORBIDDEN_MODEL_NAME_PATTERNS:
        p = pattern.lower()
        if p == name or p in full:
            return True
    return False


def safe_to_crs_4326(gdf: gpd.GeoDataFrame, source_name: str = "file") -> gpd.GeoDataFrame:
    if gdf.crs is None:
        print(f"[WARN] {source_name} has no CRS. Assume EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")



def get_map_region_polygon(padding: float = REGION_PADDING) -> Polygon:
    """Return the rectangular plotting region as a shapely polygon."""
    west, east, south, north = get_region_from_aoi(padding=padding)
    return box(west, south, east, north)


def keep_data_in_map_region(
    gdf: gpd.GeoDataFrame,
    source_name: str = "file",
    padding: float = REGION_PADDING,
) -> gpd.GeoDataFrame:
    """
    Keep data within the map rectangle only.

    This is NOT an AOI clip. It only prevents huge off-map datasets from being plotted.
    Inside/outside the Hoa Lac polygon is handled separately by add_aoi_flag_to_gdf().
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = safe_to_crs_4326(gdf, source_name).copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return gdf

    region_poly = get_map_region_polygon(padding=padding)
    try:
        out = gdf[gdf.geometry.intersects(region_poly)].copy()
    except Exception as exc:
        print(f"[WARN] Region filter failed for {source_name}; use total-bounds fallback. Reason: {exc}")
        west, east, south, north = get_region_from_aoi(padding=padding)
        b = gdf.bounds
        out = gdf[
            (b["maxx"] >= west) & (b["minx"] <= east)
            & (b["maxy"] >= south) & (b["miny"] <= north)
        ].copy()

    out = out[out.geometry.notna() & (~out.geometry.is_empty)].copy()
    return out

def clip_gdf_to_aoi_geometry(
    gdf: gpd.GeoDataFrame,
    source_name: str = "file",
) -> gpd.GeoDataFrame:
    """
    Clip geometries to Hoa Lac polygon.

    This really removes outside-AOI geometry.
    Use this for OSM only.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = safe_to_crs_4326(gdf, source_name).copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return gdf

    aoi_geom = get_aoi_polygon()

    try:
        gdf = gdf[gdf.geometry.intersects(aoi_geom)].copy()
        if gdf.empty:
            return gdf

        def _clip_one(geom):
            if geom is None or geom.is_empty:
                return None
            try:
                if not geom.is_valid:
                    geom = geom.buffer(0)
                return geom.intersection(aoi_geom)
            except Exception:
                return None

        gdf["geometry"] = gdf.geometry.apply(_clip_one)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
        if gdf.empty:
            return gdf

        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    except Exception as exc:
        print(f"[WARN] OSM AOI clip failed for {source_name}: {exc}")
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs="EPSG:4326")

    gdf["inside_aoi"] = True
    gdf["aoi_status"] = "inside"
    return gdf


def clip_points_df_to_aoi(df: pd.DataFrame) -> pd.DataFrame:
    """Remove OSM table/point records outside Hoa Lac polygon."""
    if df is None or df.empty:
        return df
    if "lon" not in df.columns or "lat" not in df.columns:
        return df

    out = add_aoi_flag_to_points_df(df)
    out = out[out["inside_aoi"] == True].copy()
    out["inside_aoi"] = True
    out["aoi_status"] = "inside"
    return out

def add_aoi_flag_to_gdf(gdf: gpd.GeoDataFrame, source_name: str = "file") -> gpd.GeoDataFrame:
    """Add inside_aoi/aoi_status columns without clipping or deleting outside data."""
    if gdf is None or gdf.empty:
        return gdf

    gdf = safe_to_crs_4326(gdf, source_name).copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return gdf

    aoi_geom = get_aoi_polygon()

    if AOI_INSIDE_TEST.lower().strip() == "within":
        inside = gdf.geometry.apply(lambda geom: aoi_geom.covers(geom))
    else:
        # Default: a polygon/line is treated as inside if it intersects the AOI.
        # This keeps buildings/roads touching the boundary in the normal layer.
        inside = gdf.geometry.intersects(aoi_geom)

    gdf["inside_aoi"] = inside.astype(bool).to_numpy()
    gdf["aoi_status"] = np.where(gdf["inside_aoi"], "inside", "outside")
    return gdf


def prepare_vector_for_map_region(
    gdf: gpd.GeoDataFrame,
    source_name: str = "file",
    padding: float = REGION_PADDING,
) -> gpd.GeoDataFrame:
    """Keep features in map rectangle and mark whether they are inside/outside Hoa Lac polygon."""
    gdf = keep_data_in_map_region(gdf, source_name=source_name, padding=padding)
    if gdf is None or gdf.empty:
        return gdf
    return add_aoi_flag_to_gdf(gdf, source_name=source_name)


def split_inside_outside_gdf(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return inside and outside GeoDataFrames. Adds AOI flag if missing."""
    if gdf is None or gdf.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty, empty
    if "inside_aoi" not in gdf.columns:
        gdf = add_aoi_flag_to_gdf(gdf)
    inside = gdf[gdf["inside_aoi"] == True].copy()
    outside = gdf[gdf["inside_aoi"] == False].copy()
    return inside, outside


def get_stats_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Use inside-AOI features for statistics by default, while maps can still show outside data."""
    if gdf is None or gdf.empty:
        return gdf
    if STATS_USE_INSIDE_AOI_ONLY and "inside_aoi" in gdf.columns:
        inside = gdf[gdf["inside_aoi"] == True].copy()
        if not inside.empty:
            return inside
    return gdf


def add_aoi_flag_to_points_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add inside_aoi/aoi_status columns to a lon/lat/value DataFrame."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if "lon" not in out.columns or "lat" not in out.columns:
        return out

    try:
        from shapely.prepared import prep
        aoi_prepared = prep(get_aoi_polygon())
        inside = [aoi_prepared.covers(pt) for pt in gpd.points_from_xy(out["lon"], out["lat"], crs="EPSG:4326")]
    except Exception:
        pts = gpd.GeoSeries(gpd.points_from_xy(out["lon"], out["lat"]), crs="EPSG:4326")
        inside = pts.apply(lambda pt: get_aoi_polygon().covers(pt)).to_list()

    out["inside_aoi"] = np.asarray(inside, dtype=bool)
    out["aoi_status"] = np.where(out["inside_aoi"], "inside", "outside")
    return out


def split_inside_outside_points(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inside and outside point tables. Adds AOI flag if missing."""
    if df is None or df.empty:
        empty = pd.DataFrame(columns=[])
        return empty, empty
    if "inside_aoi" not in df.columns:
        df = add_aoi_flag_to_points_df(df)
    inside = df[df["inside_aoi"] == True].copy()
    outside = df[df["inside_aoi"] == False].copy()
    return inside, outside


def get_stats_points_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use inside-AOI points for color ranges/statistics by default."""
    if df is None or df.empty:
        return df
    if STATS_USE_INSIDE_AOI_ONLY and "inside_aoi" in df.columns:
        inside = df[df["inside_aoi"] == True].copy()
        if not inside.empty:
            return inside
    return df


def clip_to_aoi(gdf: gpd.GeoDataFrame, source_name: str = "file") -> gpd.GeoDataFrame:
    """
    Backward-compatible wrapper.

    Older versions clipped/deleted outside-AOI data here. This version keeps data in the
    map region and marks outside-AOI features using inside_aoi/aoi_status.
    """
    return prepare_vector_for_map_region(gdf, source_name=source_name)

def estimate_area_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add footprint_area_m2, centroid_lon, centroid_lat where possible."""
    if gdf is None or gdf.empty:
        return gdf

    gdf = safe_to_crs_4326(gdf).copy()

    try:
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf["footprint_area_m2"] = gdf_utm.geometry.area.to_numpy()
        cent_utm = gdf_utm.geometry.centroid
        cent_wgs = gpd.GeoSeries(cent_utm, crs=utm_crs).to_crs("EPSG:4326")
        gdf["centroid_lon"] = cent_wgs.x.to_numpy()
        gdf["centroid_lat"] = cent_wgs.y.to_numpy()
    except Exception as exc:
        print(f"[WARN] Could not estimate UTM area/centroid: {exc}")
        gdf["footprint_area_m2"] = np.nan
        cent = gdf.geometry.centroid
        gdf["centroid_lon"] = cent.x.to_numpy()
        gdf["centroid_lat"] = cent.y.to_numpy()

    return gdf

def save_osm_lines_by_aoi_for_overview(
    osm_vectors: list[tuple[Path, gpd.GeoDataFrame]],
    out_inside_xy: Path,
    out_outside_xy: Path,
) -> tuple[int, int]:
    n_inside = 0
    n_outside = 0

    def write_only_lines(geom, f) -> int:
        n = 0
        for part in iter_flat_geometries(geom):
            if part is None or part.is_empty:
                continue
            if part.geom_type in ["LineString", "LinearRing"]:
                coords = list(part.coords)
                if len(coords) >= 2:
                    f.write(">\n")
                    for x, y in coords:
                        f.write(f"{x:.8f} {y:.8f}\n")
                    n += 1
        return n

    with open(out_inside_xy, "w", encoding="utf-8") as f_in, \
         open(out_outside_xy, "w", encoding="utf-8") as f_out:

        for path, gdf in osm_vectors:
            gdf = safe_to_crs_4326(gdf, str(path)).copy()
            gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type not in ["LineString", "LinearRing", "MultiLineString"]:
                    continue

                inside_geom, outside_geom = split_geometry_by_aoi_for_plot(geom)

                if inside_geom is not None and not inside_geom.is_empty:
                    n_inside += write_only_lines(inside_geom, f_in)

                if outside_geom is not None and not outside_geom.is_empty:
                    n_outside += write_only_lines(outside_geom, f_out)

    return n_inside, n_outside


# ============================================================
# INVENTORY / TOUCH ALL FILES
# ============================================================

def collect_inventory() -> pd.DataFrame:
    records = []

    for dataset, folder in DATASET_DIRS.items():
        if not folder.exists():
            records.append({
                "dataset": dataset,
                "relative_path": str(folder),
                "suffix": "",
                "size_mb": np.nan,
                "modified_time": "",
                "status": "missing_dataset_folder",
                "forbidden_model_file": False,
            })
            continue

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue

            forbidden = is_forbidden_model_file(path)
            status = "ok"
            size_mb = np.nan
            mtime = ""

            try:
                stat = path.stat()
                size_mb = stat.st_size / 1024.0 / 1024.0
                mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            except Exception as exc:
                status = f"stat_failed: {exc}"

            records.append({
                "dataset": dataset,
                "relative_path": str(path.relative_to(INPUT_ROOT)),
                "suffix": path.suffix.lower(),
                "size_mb": size_mb,
                "modified_time": mtime,
                "status": status,
                "forbidden_model_file": forbidden,
            })

    inv = pd.DataFrame(records)
    if inv.empty:
        inv = pd.DataFrame(columns=[
            "dataset", "relative_path", "suffix", "size_mb", "modified_time",
            "status", "forbidden_model_file",
        ])

    out_csv = FIG_ROOT / "00_input_data_inventory.csv"
    inv.to_csv(out_csv, index=False)
    print(f"[OK] Saved inventory: {out_csv}")
    return inv


def write_report(inventory: pd.DataFrame, summary_records: list[dict]) -> None:
    out_txt = FIG_ROOT / "00_input_data_report.txt"
    out_csv = FIG_ROOT / "00_input_data_plot_summary.csv"

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Hoa Lac downloaded input data report\n")
        f.write("====================================\n\n")
        f.write(f"Input root: {INPUT_ROOT}\n")
        f.write(f"Figure root: {FIG_ROOT}\n")

        area_m2, area_km2, area_ha = get_aoi_area_stats()
        f.write("\nStudy area:\n")
        f.write(f"  Area = {area_m2:,.0f} m2\n")
        f.write(f"  Area = {area_km2:.4f} km2\n")
        f.write(f"  Area = {area_ha:.2f} ha\n")

        f.write("\nScanned dataset folders:\n")
        for dataset, folder in DATASET_DIRS.items():
            f.write(f"  - {dataset}: {folder}\n")

        f.write("\nFile inventory summary:\n")
        if not inventory.empty:
            group = inventory.groupby("dataset", dropna=False).agg(
                n_files=("relative_path", "count"),
                total_size_mb=("size_mb", "sum"),
                n_forbidden_model_files=("forbidden_model_file", "sum"),
            )
            f.write(group.to_string())
            f.write("\n")

        f.write("\nPlot summary:\n")
        if summary_df.empty:
            f.write("No plots generated.\n")
        else:
            f.write(summary_df.to_string(index=False))
            f.write("\n")

        forbidden = inventory[inventory["forbidden_model_file"] == True]
        if not forbidden.empty:
            f.write("\nSkipped files matching forbidden model-file patterns:\n")
            for p in forbidden["relative_path"].tolist():
                f.write(f"\n  - {p}")
            f.write("\n")

    print(f"[OK] Saved report TXT: {out_txt}")
    print(f"[OK] Saved plot summary CSV: {out_csv}")


# ============================================================
# GENERIC GEO FILE LOADING / PLOTTING
# ============================================================

def find_vector_files(folder: Path) -> list[Path]:
    suffixes = {".gpkg", ".geojson", ".json", ".shp"}
    files = []
    if not folder.exists():
        return files
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes and not is_forbidden_model_file(p):
            files.append(p)
    return files


def load_vector_file(path: Path, source_name: str) -> gpd.GeoDataFrame | None:
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        print(f"[WARN] Could not read vector file {path}: {exc}")
        return None

    if gdf is None or gdf.empty:
        print(f"[WARN] Empty vector file: {path}")
        return None

    gdf = safe_to_crs_4326(gdf, source_name)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return None
    return gdf


def find_table_files(folder: Path) -> list[Path]:
    suffixes = {".xyz", ".csv", ".txt", ".dat"}
    files = []
    if not folder.exists():
        return files
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes and not is_forbidden_model_file(p):
            files.append(p)
    return files

def plot_osm_extra_features_map(
    path: Path,
    out_png: Path,
    gdf: gpd.GeoDataFrame | None = None,
) -> dict | None:
    """
    Plot OSM extra features with legend labels by feature group.

    Example legend groups:
      water, waterway, natural, landuse, railway,
      amenity, man_made, leisure, barrier, building
    """
    if gdf is None:
        gdf = load_vector_file(path, str(path))

    if gdf is None or gdf.empty:
        return None

    # Keep only AOI geometry for OSM.
    if REMOVE_OUTSIDE_OSM:
        gdf = clip_gdf_to_aoi_geometry(gdf, str(path))
    else:
        gdf = prepare_vector_for_map_region(gdf, str(path))

    if gdf is None or gdf.empty:
        print(f"[SKIP] No OSM extra features inside AOI: {path}")
        return None

    gdf = add_osm_group_column(gdf)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir(prefix="osm_extra_")
    aoi_xy = tmp_dir / "aoi.xy"
    save_aoi_xy(aoi_xy)

    fig = start_map(region, "OSM extra features")

    n_total_parts = 0

    present_groups = [
        g for g in OSM_EXTRA_FEATURE_ORDER
        if g != "other" and g in set(gdf["osm_group"].unique())
    ]

    if "other" in set(gdf["osm_group"].unique()):
        present_groups.append("other")

    for group in present_groups:
        sub = gdf[gdf["osm_group"] == group].copy()
        if sub.empty:
            continue

        style = OSM_EXTRA_FEATURE_STYLES.get(group, OSM_EXTRA_FEATURE_STYLES["other"])

        line_xy = tmp_dir / f"{group}_line.xy"
        poly_xy = tmp_dir / f"{group}_poly.xy"
        pts_xy = tmp_dir / f"{group}_pts.xy"

        n_lines = 0
        n_polys = 0
        point_records = []

        with open(line_xy, "w", encoding="utf-8") as f_line, \
             open(poly_xy, "w", encoding="utf-8") as f_poly:

            for geom in sub.geometry:
                if geom is None or geom.is_empty:
                    continue
                li, pi, _ = write_geom_parts_to_xy(
                    geom,
                    f_line,
                    f_poly,
                    point_records,
                )
                n_lines += li
                n_polys += pi

        # Plot polygon first so line/point stay visible.
        if n_polys > 0:
            fig.plot(
                data=str(poly_xy),
                fill=style["poly_fill"],
                pen=style["poly_pen"],
                label=group,
            )
            n_total_parts += n_polys
        elif n_lines > 0:
            fig.plot(
                data=str(line_xy),
                pen=style["line_pen"],
                label=group,
            )
            n_total_parts += n_lines
        elif len(point_records) > 0:
            pd.DataFrame(point_records, columns=["lon", "lat"]).to_csv(
                pts_xy,
                sep=" ",
                index=False,
                header=False,
                float_format="%.8f",
            )
            fig.plot(
                data=str(pts_xy),
                style="c0.05c",
                fill=style["point_fill"],
                pen=style["point_pen"],
                label=group,
            )
            n_total_parts += len(point_records)

        # If a group has both polygon and line/point, plot them too without repeating label.
        if n_polys > 0 and n_lines > 0:
            fig.plot(data=str(line_xy), pen=style["line_pen"])
        if len(point_records) > 0:
            pd.DataFrame(point_records, columns=["lon", "lat"]).to_csv(
                pts_xy,
                sep=" ",
                index=False,
                header=False,
                float_format="%.8f",
            )
            fig.plot(
                data=str(pts_xy),
                style="c0.04c",
                fill=style["point_fill"],
                pen=style["point_pen"],
            )

    fig.plot(data=str(aoi_xy), pen=AOI_PEN, label="Hoa Lac boundary")
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved OSM extra-features map: {out_png}")
    return {
        "dataset": "osm",
        "source_file": str(path),
        "plot_file": str(out_png),
        "plot_type": "osm_extra_features_by_group",
        "n_records_plotted": int(len(gdf)),
        "n_inside_aoi": int(len(gdf)),
        "n_outside_aoi": 0 if REMOVE_OUTSIDE_OSM else None,
    }


def infer_coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Return lon_col, lat_col, value_col from a table."""
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}

    lon_candidates = ["lon", "long", "longitude", "x", "centroid_lon"]
    lat_candidates = ["lat", "latitude", "y", "centroid_lat"]
    value_candidates = [
        "elevation", "elevation_m", "z", "height", "height_m", "dem", "slope",
        "hillshade", "value", "area_m2", "footprint_area_m2",
    ]

    lon_col = next((lower[c] for c in lon_candidates if c in lower), None)
    lat_col = next((lower[c] for c in lat_candidates if c in lower), None)
    value_col = next((lower[c] for c in value_candidates if c in lower), None)

    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(c)

    # Headerless or unknown columns.
    if lon_col is None or lat_col is None:
        if len(numeric_cols) >= 2:
            # Prefer a pair that looks like lon/lat.
            for i in range(len(numeric_cols) - 1):
                c1, c2 = numeric_cols[i], numeric_cols[i + 1]
                x = pd.to_numeric(df[c1], errors="coerce")
                y = pd.to_numeric(df[c2], errors="coerce")
                ok = (
                    x.between(-180, 180).mean() > 0.8
                    and y.between(-90, 90).mean() > 0.8
                )
                if ok:
                    lon_col, lat_col = c1, c2
                    break

    if value_col is None:
        for c in numeric_cols:
            if c not in [lon_col, lat_col]:
                value_col = c
                break

    return lon_col, lat_col, value_col


def try_projected_xy_to_lonlat(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series, str] | tuple[None, None, None]:
    """
    Try converting projected x/y to lon/lat.

    This is mainly for DEM/terrain tables that may be exported in UTM meters.
    Hoa Lac is in UTM zone 48N, so EPSG:32648 is tried first. EPSG:3857 is a
    fallback for web-mercator-like coordinates.
    """
    try:
        from pyproj import Transformer
    except Exception:
        return None, None, None

    region = get_region_from_aoi(padding=REGION_PADDING * 4.0)
    candidate_crs = [OPENTOPOGRAPHY_ASSUMED_PROJECTED_CRS, "EPSG:3857"]

    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()
    if valid.sum() < OPENTOPOGRAPHY_MIN_SPATIAL_ROWS:
        return None, None, None

    for crs in candidate_crs:
        try:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_arr, lat_arr = transformer.transform(x_num[valid].to_numpy(), y_num[valid].to_numpy())
            lon = pd.Series(np.nan, index=x_num.index, dtype=float)
            lat = pd.Series(np.nan, index=y_num.index, dtype=float)
            lon.loc[valid] = lon_arr
            lat.loc[valid] = lat_arr

            inside_ratio = (
                lon.between(region[0], region[1])
                & lat.between(region[2], region[3])
            ).mean()
            if inside_ratio > 0.20:
                return lon, lat, crs
        except Exception:
            continue

    return None, None, None


def table_name_is_summary_or_metadata(path: Path) -> bool:
    name = path.name.lower()
    return any(k in name for k in OPENTOPOGRAPHY_SKIP_SPATIAL_TABLE_NAME_KEYWORDS)


def read_table_coordinate_file(path: Path) -> pd.DataFrame | None:
    """
    Read xyz/csv/txt-like file and return lon, lat, value table if possible.

    Improvements for OpenTopography:
      - skips summary/statistics tables as spatial maps;
      - accepts lon/lat tables;
      - accepts projected UTM/web-mercator x/y tables and converts to lon/lat;
      - requires enough coordinate rows so metadata/statistics are not mistaken for maps.
    """
    if is_forbidden_model_file(path):
        return None

    if "opentopography" in str(path).lower() and table_name_is_summary_or_metadata(path):
        return None

    def finalize_from_df(df: pd.DataFrame, lon_col, lat_col, value_col=None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None

        x = pd.to_numeric(df[lon_col], errors="coerce")
        y = pd.to_numeric(df[lat_col], errors="coerce")

        # Case 1: already lon/lat.
        lon = x.copy()
        lat = y.copy()
        coord_crs = "EPSG:4326"
        lonlat_ratio = (lon.between(-180, 180) & lat.between(-90, 90)).mean()

        # Case 2: likely projected coordinates; try conversion.
        if lonlat_ratio < 0.80:
            lon2, lat2, used_crs = try_projected_xy_to_lonlat(x, y)
            if lon2 is None or lat2 is None:
                return None
            lon, lat = lon2, lat2
            coord_crs = used_crs

        if value_col is not None:
            value = pd.to_numeric(df[value_col], errors="coerce")
        else:
            value = pd.Series(1.0, index=df.index, dtype=float)

        out = pd.DataFrame({"lon": lon, "lat": lat, "value": value})
        out = out.dropna(subset=["lon", "lat"]).copy()
        out = out[out["lon"].between(-180, 180) & out["lat"].between(-90, 90)].copy()

        # Crop to AOI with padding. This also removes metadata/statistics false positives.
        region = get_region_from_aoi(padding=REGION_PADDING * 4.0)
        out = out[
            out["lon"].between(region[0], region[1])
            & out["lat"].between(region[2], region[3])
        ].copy()

        if len(out) < OPENTOPOGRAPHY_MIN_SPATIAL_ROWS and "opentopography" in str(path).lower():
            return None

        if out.empty:
            return None

        out = add_aoi_flag_to_points_df(out)
        out.attrs["coordinate_crs"] = coord_crs
        return out

    # First try header-aware reading.
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#")
    except Exception:
        df = None

    if df is not None and not df.empty:
        lon_col, lat_col, value_col = infer_coordinate_columns(df)
        if lon_col is not None and lat_col is not None:
            out = finalize_from_df(df, lon_col, lat_col, value_col)
            if out is not None and not out.empty:
                return out

    # Fallback headerless read. Useful for:
    # lon lat z
    # name lon lat z
    try:
        raw = pd.read_csv(
            path,
            sep=r"\s+|,",
            engine="python",
            comment="#",
            header=None,
            on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"[WARN] Could not read table file {path}: {exc}")
        return None

    if raw.empty or raw.shape[1] < 2:
        return None

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric.columns if numeric[c].notna().sum() > 0]

    if len(numeric_cols) < 2:
        return None

    # Try consecutive numeric columns first, then all pairs.
    candidate_pairs = []
    for i in range(len(numeric_cols) - 1):
        candidate_pairs.append((numeric_cols[i], numeric_cols[i + 1]))
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            pair = (numeric_cols[i], numeric_cols[j])
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

    for lon_col, lat_col in candidate_pairs:
        value_col = None
        for c in numeric_cols:
            if c not in [lon_col, lat_col]:
                value_col = c
                break

        test_df = pd.DataFrame({
            "x": numeric[lon_col],
            "y": numeric[lat_col],
            "value": numeric[value_col] if value_col is not None else 1.0,
        })
        out = finalize_from_df(test_df, "x", "y", "value")
        if out is not None and not out.empty:
            return out

    return None


def sample_points(df: pd.DataFrame, max_points: int = MAX_QUICKLOOK_POINTS) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if max_points is None or len(df) <= max_points:
        return df.copy()
    return df.sample(n=max_points, random_state=12345).copy()


def make_value_cpt(values: pd.Series, out_cpt: Path, cmap: str = "viridis") -> None:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(values.quantile(0.02))
        vmax = float(values.quantile(0.98))
        if not np.isfinite(vmin):
            vmin = float(values.min())
        if not np.isfinite(vmax):
            vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    pygmt.makecpt(cmap=cmap, series=[vmin, vmax], output=str(out_cpt))


def make_outside_raster_cpt(values: pd.Series, out_cpt: Path, fill: str = OUTSIDE_RASTER_CMAP) -> None:
    """Create a constant-color CPT for raster cells outside the Hoa Lac polygon.

    This keeps the topography data available in the script, but visually marks the
    outside-AOI raster as a plain background color instead of a grey/color surface.
    """
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = vmin + 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

    with open(out_cpt, "w", encoding="utf-8") as f:
        f.write(f"{vmin:.8f} {fill} {vmax:.8f} {fill}\n")
        f.write(f"B {fill}\n")
        f.write(f"F {fill}\n")
        f.write(f"N {fill}\n")



def plot_table_quicklook(path: Path, dataset: str, out_png: Path) -> dict | None:
    df = read_table_coordinate_file(path)
    if df is None or df.empty:
        return None
    if dataset.lower() == "osm" and REMOVE_OUTSIDE_OSM:
        df = clip_points_df_to_aoi(df)
        if df is None or df.empty:
            print(f"[SKIP] No OSM table points inside AOI: {path}")
            return None
    
    df = sample_points(df, MAX_QUICKLOOK_POINTS)
    df = add_aoi_flag_to_points_df(df)
    inside_df, outside_df = split_inside_outside_points(df)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()
    cpt = tmp_dir / "value.cpt"
    inside_xyz = tmp_dir / "points_inside.xyz"
    outside_xy = tmp_dir / "points_outside.xy"

    stats_df = get_stats_points_df(df)
    make_value_cpt(stats_df["value"], cpt)

    fig = start_map(region, f"{dataset}: {path.name}")

    if MARK_OUTSIDE_AOI and not outside_df.empty:
        outside_df[["lon", "lat"]].to_csv(
            outside_xy,
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )
        fig.plot(
            data=str(outside_xy),
            style="c0.030c",
            fill=OUTSIDE_AOI_POINT_FILL,
            pen=None,
            transparency=OUTSIDE_AOI_POINT_TRANSPARENCY,
            label="Outside Hoa Lac polygon",
        )

    if not inside_df.empty:
        inside_df[["lon", "lat", "value"]].to_csv(
            inside_xyz,
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )
        fig.plot(
            data=str(inside_xyz),
            style="c0.035c",
            cmap=str(cpt),
            fill="+z",
            pen="0.02p,black",
            transparency=30,
            label="Inside Hoa Lac polygon",
        )

    plot_aoi_boundary(fig)
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c/0c",
        frame=["xaf+lValue"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    if MARK_OUTSIDE_AOI and not outside_df.empty:
        fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved quicklook: {out_png}")
    return {
        "dataset": dataset,
        "source_file": str(path),
        "plot_file": str(out_png),
        "plot_type": "table_coordinate_quicklook_mark_outside_aoi",
        "n_records_plotted": len(df),
        "n_inside_aoi": int(len(inside_df)),
        "n_outside_aoi": int(len(outside_df)),
    }

def save_geometries_segments(gdf: gpd.GeoDataFrame, out_xy: Path, value_col: str | None = None) -> None:
    gdf = safe_to_crs_4326(gdf).copy()
    with open(out_xy, "w", encoding="utf-8") as f:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            value = None
            if value_col is not None and value_col in gdf.columns:
                try:
                    value = float(row[value_col])
                except Exception:
                    value = None

            def write_header():
                if value is None or not np.isfinite(value):
                    f.write(">\n")
                else:
                    f.write(f"> -Z{value:.8f}\n")

            if geom.geom_type in ["Polygon", "MultiPolygon"]:
                for poly in safe_polygons(geom):
                    write_header()
                    for x, y in poly.exterior.coords:
                        f.write(f"{x:.8f} {y:.8f}\n")
            elif geom.geom_type in ["LineString", "LinearRing"]:
                write_header()
                for x, y in geom.coords:
                    f.write(f"{x:.8f} {y:.8f}\n")
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    write_header()
                    for x, y in line.coords:
                        f.write(f"{x:.8f} {y:.8f}\n")



def plot_vector_quicklook(path: Path, dataset: str, out_png: Path) -> dict | None:
    gdf = load_vector_file(path, str(path))
    if gdf is None or gdf.empty:
        return None

    if dataset.lower() == "osm" and REMOVE_OUTSIDE_OSM:
        gdf = clip_gdf_to_aoi_geometry(gdf, str(path))
    else:
        gdf = prepare_vector_for_map_region(gdf, str(path))

    if gdf is None or gdf.empty:
        print(f"[SKIP] No features inside map region: {path}")
        return None

    if len(gdf) > MAX_POLYGON_PLOT_FEATURES:
        print(f"[INFO] Use first {MAX_POLYGON_PLOT_FEATURES:,} features for plot only: {path.name}")
        gdf = gdf.head(MAX_POLYGON_PLOT_FEATURES).copy()

    inside_gdf, outside_gdf = split_inside_outside_gdf(gdf)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()
    inside_seg_xy = tmp_dir / "geom_inside.xy"
    outside_seg_xy = tmp_dir / "geom_outside.xy"
    inside_pts_xy = tmp_dir / "pts_inside.xy"
    outside_pts_xy = tmp_dir / "pts_outside.xy"

    fig = start_map(region, f"{dataset}: {path.name}")

    # Outside features first: muted grey layer, not clipped/deleted.
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        outside_geom_types = set(outside_gdf.geometry.geom_type.unique())
        if any(t in outside_geom_types for t in ["Polygon", "MultiPolygon", "LineString", "MultiLineString", "LinearRing"]):
            save_geometries_segments(outside_gdf, outside_seg_xy)
            if any(t in outside_geom_types for t in ["Polygon", "MultiPolygon"]):
                fig.plot(
                    data=str(outside_seg_xy),
                    pen=OUTSIDE_AOI_PEN,
                    fill=OUTSIDE_AOI_FILL,
                    label="Outside Hoa Lac polygon",
                )
            else:
                fig.plot(
                    data=str(outside_seg_xy),
                    pen=OUTSIDE_AOI_LINE_PEN,
                    label="Outside Hoa Lac polygon",
                )

        outside_point_mask = outside_gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
        if outside_point_mask.any():
            pts = outside_gdf[outside_point_mask].explode(index_parts=False).copy()
            coords = pd.DataFrame({"lon": pts.geometry.x, "lat": pts.geometry.y})
            coords.to_csv(outside_pts_xy, sep=" ", index=False, header=False, float_format="%.8f")
            fig.plot(
                data=str(outside_pts_xy),
                style="c0.05c",
                fill=OUTSIDE_AOI_POINT_FILL,
                pen="0.05p,gray40",
                transparency=OUTSIDE_AOI_POINT_TRANSPARENCY,
            )

    # Inside/intersecting-AOI features: original style.
    if not inside_gdf.empty:
        inside_geom_types = set(inside_gdf.geometry.geom_type.unique())
        if any(t in inside_geom_types for t in ["Polygon", "MultiPolygon", "LineString", "MultiLineString", "LinearRing"]):
            save_geometries_segments(inside_gdf, inside_seg_xy)
            if any(t in inside_geom_types for t in ["Polygon", "MultiPolygon"]):
                fig.plot(data=str(inside_seg_xy), pen="0.2p,black", fill="gray70@55", label="Inside Hoa Lac polygon")
            else:
                fig.plot(data=str(inside_seg_xy), pen=OSM_LINE_PEN, label="Inside Hoa Lac polygon")

        inside_point_mask = inside_gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
        if inside_point_mask.any():
            pts = inside_gdf[inside_point_mask].explode(index_parts=False).copy()
            coords = pd.DataFrame({"lon": pts.geometry.x, "lat": pts.geometry.y})
            coords.to_csv(inside_pts_xy, sep=" ", index=False, header=False, float_format="%.8f")
            fig.plot(data=str(inside_pts_xy), style="c0.06c", fill="red", pen="0.1p,black", transparency=20)

    plot_aoi_boundary(fig)
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved vector quicklook: {out_png}")
    return {
        "dataset": dataset,
        "source_file": str(path),
        "plot_file": str(out_png),
        "plot_type": "vector_quicklook_mark_outside_aoi",
        "n_records_plotted": len(gdf),
        "n_inside_aoi": int(len(inside_gdf)),
        "n_outside_aoi": int(len(outside_gdf)),
    }


def plot_building_dataset_overview(
    gdf: gpd.GeoDataFrame,
    title: str,
    out_png: Path,
    fill: str = "gray70@55",
    pen: str = "0.08p,black",
) -> dict | None:
    """Simple overview map for one building dataset subdirectory, with outside-AOI data marked."""
    if gdf is None or gdf.empty:
        return None

    plot_gdf = reduce_building_gdf_for_plot(gdf, max_features=MAX_OVERVIEW_POLYGONS)
    inside_gdf, outside_gdf = split_inside_outside_gdf(plot_gdf)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()
    inside_xy = tmp_dir / "dataset_buildings_inside.xy"
    outside_xy = tmp_dir / "dataset_buildings_outside.xy"
    aoi_xy = tmp_dir / "aoi.xy"

    save_aoi_xy(aoi_xy)
    if not inside_gdf.empty:
        save_building_polygons_value(inside_gdf, "height_m", inside_xy)
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        save_building_polygons_value(outside_gdf, "height_m", outside_xy)

    fig = start_map(region, title)
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.plot(data=str(outside_xy), fill=OUTSIDE_AOI_FILL, pen=OUTSIDE_AOI_PEN, label="Outside Hoa Lac polygon")
    if not inside_gdf.empty:
        fig.plot(data=str(inside_xy), fill=fill, pen=pen, transparency=10, label="Inside Hoa Lac polygon")
    fig.plot(data=str(aoi_xy), pen=AOI_PEN, label="Hoa Lac boundary")
    add_aoi_area_text_box(fig, region)
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved dataset overview map: {out_png}")
    return {
        "dataset": out_png.parent.name,
        "source_file": "building_polygon_gdf",
        "plot_file": str(out_png),
        "plot_type": "dataset_overview_map_mark_outside_aoi",
        "n_records_plotted": len(plot_gdf),
        "n_inside_aoi": int(len(inside_gdf)),
        "n_outside_aoi": int(len(outside_gdf)),
    }

def iter_flat_geometries(geom):
    """Yield simple geometries from Multi* or GeometryCollection."""
    if geom is None or geom.is_empty:
        return

    if geom.geom_type in [
        "GeometryCollection",
        "MultiPolygon",
        "MultiLineString",
        "MultiPoint",
    ]:
        for part in geom.geoms:
            yield from iter_flat_geometries(part)
    else:
        yield geom

def normalize_osm_group_name(value) -> str:
    """Normalize OSM category names to a compact legend/group name."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "other"

    s = str(value).strip().lower()

    alias = {
        "water": "water",
        "waterway": "waterway",
        "natural": "natural",
        "landuse": "landuse",
        "railway": "railway",
        "amenity": "amenity",
        "man_made": "man_made",
        "manmade": "man_made",
        "man-made": "man_made",
        "leisure": "leisure",
        "barrier": "barrier",
        "building": "building",
    }
    return alias.get(s, s if s else "other")


def infer_osm_group_from_row(row: pd.Series) -> str:
    """
    Infer one OSM feature group from a row.

    Do NOT use generic columns like:
      layer = -1 / 0 / 1
      type = multipolygon
      kind = ...
    because they are not feature categories for this legend.
    """

    # 1) If you already created an explicit category column, use it.
    preferred_cols = [
        "feature",
        "feature_type",
        "feature_group",
        "group",
        "category",
        "class",
        "fclass",
        "osm_feature",
    ]

    valid_groups = set(OSM_EXTRA_FEATURE_ORDER)

    for col in preferred_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                group = normalize_osm_group_name(val)
                if group in valid_groups:
                    return group

    # 2) Use real OSM tag columns only.
    tag_cols = [
        "water",
        "waterway",
        "natural",
        "landuse",
        "railway",
        "amenity",
        "man_made",
        "leisure",
        "barrier",
        "building",
    ]

    for col in tag_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                s = str(val).strip().lower()
                if s not in ["", "nan", "none", "0", "false"]:
                    return normalize_osm_group_name(col)

    return "other"

def add_osm_group_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add normalized osm_group column for legend-aware OSM plotting."""
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    out["osm_group"] = out.apply(infer_osm_group_from_row, axis=1)
    out["osm_group"] = out["osm_group"].fillna("other")
    return out


def write_geom_parts_to_xy(
    geom,
    line_handle,
    poly_handle,
    point_records,
) -> tuple[int, int, int]:
    """Write one geometry into line/polygon/point containers."""
    n_lines = 0
    n_polys = 0
    n_points = 0

    for part in iter_flat_geometries(geom):
        if part is None or part.is_empty:
            continue

        gtype = part.geom_type

        if gtype in ["LineString", "LinearRing"]:
            coords = list(part.coords)
            if len(coords) >= 2:
                line_handle.write(">\n")
                for x, y in coords:
                    line_handle.write(f"{x:.8f} {y:.8f}\n")
                n_lines += 1

        elif gtype == "Polygon":
            coords = list(part.exterior.coords)
            if len(coords) >= 4:
                poly_handle.write(">\n")
                for x, y in coords:
                    poly_handle.write(f"{x:.8f} {y:.8f}\n")
                n_polys += 1

        elif gtype == "Point":
            point_records.append((part.x, part.y))
            n_points += 1

    return n_lines, n_polys, n_points


def write_osm_geom_parts(geom, f_line, f_poly, pts_records) -> tuple[int, int, int]:
    """
    Write geometry parts to OSM line/polygon/point containers.

    Returns:
        n_lines, n_polys, n_points
    """
    n_lines = 0
    n_polys = 0
    n_points = 0

    for part in iter_flat_geometries(geom):
        if part is None or part.is_empty:
            continue

        gtype = part.geom_type

        if gtype in ["LineString", "LinearRing"]:
            coords = list(part.coords)
            if len(coords) >= 2:
                f_line.write(">\n")
                for x, y in coords:
                    f_line.write(f"{x:.8f} {y:.8f}\n")
                n_lines += 1

        elif gtype == "Polygon":
            coords = list(part.exterior.coords)
            if len(coords) >= 4:
                f_poly.write(">\n")
                for x, y in coords:
                    f_poly.write(f"{x:.8f} {y:.8f}\n")
                n_polys += 1

        elif gtype == "Point":
            pts_records.append((part.x, part.y))
            n_points += 1

    return n_lines, n_polys, n_points


def split_geometry_by_aoi_for_plot(geom):
    """
    Split one geometry into inside-AOI and outside-AOI pieces.

    This is the key fix for OSM. It does not only classify the whole feature.
    It cuts the geometry using the Hoa Lac polygon.
    """
    if geom is None or geom.is_empty:
        return None, None

    aoi_geom = get_aoi_polygon()

    try:
        if not geom.is_valid:
            geom = geom.buffer(0)

        inside_geom = geom.intersection(aoi_geom)
        outside_geom = geom.difference(aoi_geom)

        return inside_geom, outside_geom

    except Exception:
        # Safe fallback: classify whole feature only.
        if geom.intersects(aoi_geom):
            return geom, None
        return None, geom
    

def plot_osm_dataset_overview(
    osm_vectors: list[tuple[Path, gpd.GeoDataFrame]],
    out_png: Path,
) -> dict | None:
    """
    OSM overview map.

    OSM data outside Hoa Lac are removed before plotting.
    """
    if not osm_vectors:
        return None

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()

    line_xy = tmp_dir / "osm_lines_inside_only.xy"
    poly_xy = tmp_dir / "osm_polygons_inside_only.xy"
    pts_xy = tmp_dir / "osm_points_inside_only.xy"
    aoi_xy = tmp_dir / "aoi.xy"

    n_lines = 0
    n_polys = 0
    pts_records: list[tuple[float, float]] = []

    with open(line_xy, "w", encoding="utf-8") as f_line, \
         open(poly_xy, "w", encoding="utf-8") as f_poly:

        for path, gdf in osm_vectors:
            if REMOVE_OUTSIDE_OSM:
                gdf = clip_gdf_to_aoi_geometry(gdf, str(path))
            else:
                gdf = safe_to_crs_4326(gdf, str(path))

            if gdf is None or gdf.empty:
                continue

            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                for part in iter_flat_geometries(geom):
                    if part is None or part.is_empty:
                        continue

                    gtype = part.geom_type

                    if gtype in ["LineString", "LinearRing"]:
                        coords = list(part.coords)
                        if len(coords) >= 2:
                            f_line.write(">\n")
                            for x, y in coords:
                                f_line.write(f"{x:.8f} {y:.8f}\n")
                            n_lines += 1

                    elif gtype == "Polygon":
                        coords = list(part.exterior.coords)
                        if len(coords) >= 4:
                            f_poly.write(">\n")
                            for x, y in coords:
                                f_poly.write(f"{x:.8f} {y:.8f}\n")
                            n_polys += 1

                    elif gtype == "Point":
                        pts_records.append((part.x, part.y))

    if pts_records:
        pd.DataFrame(pts_records, columns=["lon", "lat"]).to_csv(
            pts_xy,
            sep=" ",
            index=False,
            header=False,
            float_format="%.8f",
        )

    save_aoi_xy(aoi_xy)

    fig = start_map(region, "OSM overview map")

    if n_polys > 0:
        fig.plot(
            data=str(poly_xy),
            fill="lightgray@55",
            pen="0.10p,gray40",
            label="OSM polygons inside",
        )

    if n_lines > 0:
        fig.plot(
            data=str(line_xy),
            pen="0.70p,gray35",
            label="Road / line inside",
        )

    if pts_records:
        fig.plot(
            data=str(pts_xy),
            style="c0.05c",
            fill="orange",
            pen="0.1p,black",
            transparency=20,
            label="Point feature inside",
        )

    fig.plot(data=str(aoi_xy), pen=AOI_PEN, label="Hoa Lac boundary")
    add_aoi_area_text_box(fig, region)
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    n_inside = int(n_lines + n_polys + len(pts_records))

    print(f"[OK] Saved OSM overview map: {out_png}")
    print(f"[INFO] OSM inside-only parts: {n_inside:,}")

    return {
        "dataset": "osm",
        "source_file": "OSM vectors",
        "plot_file": str(out_png),
        "plot_type": "dataset_overview_map_osm_inside_only",
        "n_records_plotted": n_inside,
        "n_inside_aoi": n_inside,
        "n_outside_aoi": 0,
    }

# ============================================================
# HEIGHT / AREA PLOT HELPERS FOR BUILDING POLYGONS
# ============================================================

def detect_height_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "height_m", "height", "building_height", "building_height_m", "pred_height",
        "height_mean", "mean_height", "h", "HEIGHT", "Height",
    ]
    for col in candidates:
        if col in gdf.columns:
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return col
    for col in gdf.columns:
        if "height" in str(col).lower():
            vals = pd.to_numeric(gdf[col], errors="coerce")
            if vals.notna().any():
                return col
    return None


def normalize_building_gdf(gdf: gpd.GeoDataFrame, default_height_m: float = 6.0) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = safe_to_crs_4326(gdf).copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        return gdf

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    height_col = detect_height_column(gdf)
    if height_col is None:
        gdf["height_m"] = default_height_m
        print(f"[WARN] No height column found. Use default height = {default_height_m} m")
    else:
        gdf["height_m"] = (
            pd.to_numeric(gdf[height_col], errors="coerce")
            .fillna(default_height_m)
            .clip(lower=0.0)
        )
        print(f"[INFO] Height column: {height_col}")

    gdf = estimate_area_centroids(gdf)
    gdf["volume_m3"] = gdf["footprint_area_m2"].fillna(0.0) * gdf["height_m"]
    return gdf


def save_building_polygons_value(gdf: gpd.GeoDataFrame, value_col: str, out_xy: Path) -> None:
    save_geometries_segments(gdf, out_xy, value_col=value_col)


def make_building_height_cpt(gdf: gpd.GeoDataFrame, out_cpt: Path) -> None:
    make_value_cpt(gdf["height_m"], out_cpt, cmap="viridis")


def make_log_area_cpt(gdf: gpd.GeoDataFrame, out_cpt: Path) -> None:
    vals = np.log10(pd.to_numeric(gdf["footprint_area_m2"], errors="coerce").clip(lower=1.0))
    make_value_cpt(vals, out_cpt, cmap="plasma")


def reduce_building_gdf_for_plot(gdf: gpd.GeoDataFrame, max_features: int = MAX_POLYGON_PLOT_FEATURES) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if max_features is None or len(gdf) <= max_features:
        return gdf.copy()
    print(f"[INFO] Use largest {max_features:,} buildings for plot only.")
    return gdf.sort_values("footprint_area_m2", ascending=False).head(max_features).copy()



def plot_building_height_map(gdf: gpd.GeoDataFrame, title: str, out_png: Path, pen: str = "0.03p,black") -> dict | None:
    if gdf is None or gdf.empty:
        return None

    plot_gdf = reduce_building_gdf_for_plot(gdf)
    inside_gdf, outside_gdf = split_inside_outside_gdf(plot_gdf)
    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()

    inside_xy = tmp_dir / "buildings_height_inside.xy"
    outside_xy = tmp_dir / "buildings_height_outside.xy"
    aoi_xy = tmp_dir / "aoi.xy"
    cpt = tmp_dir / "height.cpt"
    centroid_inside_xyz = tmp_dir / "centroid_inside.xyz"
    centroid_outside_xy = tmp_dir / "centroid_outside.xy"

    save_aoi_xy(aoi_xy)
    stats_gdf = get_stats_gdf(gdf)
    make_building_height_cpt(stats_gdf, cpt)

    if not inside_gdf.empty:
        save_building_polygons_value(inside_gdf, "height_m", inside_xy)
        pd.DataFrame({
            "lon": inside_gdf["centroid_lon"],
            "lat": inside_gdf["centroid_lat"],
            "height_m": inside_gdf["height_m"],
        }).to_csv(centroid_inside_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        save_building_polygons_value(outside_gdf, "height_m", outside_xy)
        pd.DataFrame({
            "lon": outside_gdf["centroid_lon"],
            "lat": outside_gdf["centroid_lat"],
        }).to_csv(centroid_outside_xy, sep=" ", index=False, header=False, float_format="%.8f")

    fig = start_map(region, title)
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.plot(data=str(outside_xy), fill=OUTSIDE_AOI_FILL, pen=OUTSIDE_AOI_PEN, label="Outside Hoa Lac polygon")
        fig.plot(data=str(centroid_outside_xy), style="c0.020c", fill=OUTSIDE_AOI_POINT_FILL, pen=None, transparency=OUTSIDE_AOI_POINT_TRANSPARENCY)
    if not inside_gdf.empty:
        fig.plot(data=str(inside_xy), cmap=str(cpt), fill="+z", pen=pen, transparency=20, label="Inside Hoa Lac polygon")
        fig.plot(data=str(centroid_inside_xyz), style="c0.025c", fill="black", pen="0.02p,black", transparency=50)
    fig.plot(data=str(aoi_xy), pen=AOI_PEN)
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c/0c",
        frame=["xaf+lBuilding height", "y+l(m)"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved height map: {out_png}")
    return {
        "dataset": out_png.parent.name,
        "source_file": "building_polygon_gdf",
        "plot_file": str(out_png),
        "plot_type": "building_height_map_mark_outside_aoi",
        "n_records_plotted": len(plot_gdf),
        "n_inside_aoi": int(len(inside_gdf)),
        "n_outside_aoi": int(len(outside_gdf)),
    }


def plot_building_area_map(gdf: gpd.GeoDataFrame, title: str, out_png: Path, pen: str = "0.03p,black") -> dict | None:
    if gdf is None or gdf.empty:
        return None

    plot_gdf = reduce_building_gdf_for_plot(gdf)
    plot_gdf["log_area_m2"] = np.log10(plot_gdf["footprint_area_m2"].clip(lower=1.0))
    inside_gdf, outside_gdf = split_inside_outside_gdf(plot_gdf)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir()
    inside_xy = tmp_dir / "buildings_area_inside.xy"
    outside_xy = tmp_dir / "buildings_area_outside.xy"
    aoi_xy = tmp_dir / "aoi.xy"
    cpt = tmp_dir / "area.cpt"

    save_aoi_xy(aoi_xy)
    stats_gdf = get_stats_gdf(plot_gdf)
    make_log_area_cpt(stats_gdf, cpt)

    if not inside_gdf.empty:
        save_building_polygons_value(inside_gdf, "log_area_m2", inside_xy)
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        save_building_polygons_value(outside_gdf, "log_area_m2", outside_xy)

    fig = start_map(region, title)
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.plot(data=str(outside_xy), fill=OUTSIDE_AOI_FILL, pen=OUTSIDE_AOI_PEN, label="Outside Hoa Lac polygon")
    if not inside_gdf.empty:
        fig.plot(data=str(inside_xy), cmap=str(cpt), fill="+z", pen=pen, transparency=20, label="Inside Hoa Lac polygon")
    fig.plot(data=str(aoi_xy), pen=AOI_PEN)
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c/0c",
        frame=["xaf+llog10 footprint area", "y+l(m@+2@+)"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    if MARK_OUTSIDE_AOI and not outside_gdf.empty:
        fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved area map: {out_png}")
    return {
        "dataset": out_png.parent.name,
        "source_file": "building_polygon_gdf",
        "plot_file": str(out_png),
        "plot_type": "building_area_map_mark_outside_aoi",
        "n_records_plotted": len(plot_gdf),
        "n_inside_aoi": int(len(inside_gdf)),
        "n_outside_aoi": int(len(outside_gdf)),
    }


def plot_height_histogram(gdf: gpd.GeoDataFrame, title: str, out_png: Path) -> dict | None:
    if gdf is None or gdf.empty or "height_m" not in gdf.columns:
        return None

    stats_gdf = get_stats_gdf(gdf)
    values = pd.to_numeric(stats_gdf["height_m"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values >= 0]
    if values.empty:
        return None

    tmp_dir = make_temp_dir()
    values_txt = tmp_dir / "height_values.txt"
    values.to_csv(values_txt, sep=" ", index=False, header=False, float_format="%.8f")

    hmin = 0.0
    hmax = float(values.max())
    if hmax <= 0:
        hmax = 1.0

    bins = 30
    counts, _ = np.histogram(values, bins=bins, range=(hmin, hmax))
    ymax = max(1, int(counts.max() * 1.15))

    mean_h = float(values.mean())
    median_h = float(values.median())
    inside_gdf, outside_gdf = split_inside_outside_gdf(gdf)
    title_suffix = ""
    if STATS_USE_INSIDE_AOI_ONLY and "inside_aoi" in gdf.columns:
        title_suffix = " (inside AOI stats)"

    fig = pygmt.Figure()
    fig.histogram(
        data=str(values_txt),
        region=[hmin, hmax, 0, ymax],
        projection="X14c/8c",
        frame=["xaf+lBuilding height (m)", "yaf+lNumber of buildings", f'WSen+t"{title}{title_suffix}"'],
        series=(hmax - hmin) / bins,
        fill="gray70",
        pen="0.5p,black",
    )
    fig.plot(x=[mean_h, mean_h], y=[0, ymax], pen="1.2p,red,--")
    fig.plot(x=[median_h, median_h], y=[0, ymax], pen="1.2p,blue,.")
    fig.text(x=mean_h, y=ymax * 0.92, text=f"Mean {mean_h:.1f} m", font="9p,Helvetica,red", justify="LM")
    fig.text(x=median_h, y=ymax * 0.70, text=f"Median {median_h:.1f} m", font="9p,Helvetica,blue", justify="LM")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved height histogram: {out_png}")
    return {
        "dataset": out_png.parent.name,
        "source_file": "building_polygon_gdf",
        "plot_file": str(out_png),
        "plot_type": "height_histogram_inside_aoi_stats",
        "n_records_plotted": len(values),
        "n_inside_aoi": int(len(inside_gdf)),
        "n_outside_aoi": int(len(outside_gdf)),
    }

# ============================================================
# GLOBALBUILDINGATLAS LoD1
# ============================================================

def find_gba_building_file() -> Path | None:
    gba_dir = DATASET_DIRS["globalbuildingatlas_lod1"]

    # When outside-AOI marking is requested, prefer bbox/unclipped files first.
    # A file that was already clipped to the polygon cannot show outside data anymore.
    if MARK_OUTSIDE_AOI:
        candidates = [
            gba_dir / "processed" / "gba_lod1_buildings_bbox_filtered_lowram.parquet",
            gba_dir / "processed" / "gba_lod1_buildings_bbox_filtered_lowram.gpkg",
            gba_dir / "processed" / "gba_lod1_buildings_hoalac_clipped.gpkg",
        ]
        search_patterns = [
            "**/*bbox*filtered*.parquet",
            "**/*bbox*filtered*.gpkg",
            "**/*gba*.gpkg",
            "**/*lod1*.gpkg",
            "**/*gba*clipped*.gpkg",
            "**/*lod1*clipped*.gpkg",
            "**/*building*clipped*.gpkg",
        ]
    else:
        candidates = [
            gba_dir / "processed" / "gba_lod1_buildings_hoalac_clipped.gpkg",
            gba_dir / "processed" / "gba_lod1_buildings_bbox_filtered_lowram.parquet",
        ]
        search_patterns = [
            "**/*gba*clipped*.gpkg",
            "**/*lod1*clipped*.gpkg",
            "**/*building*clipped*.gpkg",
            "**/*gba*.gpkg",
            "**/*lod1*.gpkg",
            "**/*bbox*filtered*.parquet",
        ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    for pat in search_patterns:
        found = sorted(gba_dir.glob(pat)) if gba_dir.exists() else []
        for p in found:
            if p.is_file() and not is_forbidden_model_file(p):
                return p
    return None


def load_gba_buildings() -> tuple[gpd.GeoDataFrame | None, Path | None]:
    if not PLOT_GLOBALBUILDINGATLAS_LOD1:
        return None, None

    source_file = find_gba_building_file()
    if source_file is None:
        print("[SKIP] No processed GlobalBuildingAtlas LoD1 building file found.")
        return None, None

    print("\n========== LOAD GLOBALBUILDINGATLAS LoD1 ==========")
    print(f"[INFO] Source: {source_file}")

    try:
        if source_file.suffix.lower() == ".parquet":
            gdf = gpd.read_parquet(source_file)
        else:
            gdf = gpd.read_file(source_file)
    except Exception as exc:
        print(f"[WARN] Could not read GBA file {source_file}: {exc}")
        return None, source_file

    # Keep map-region data, but do not clip/delete data outside Hoa Lac.
    gdf = keep_data_in_map_region(gdf, str(source_file))
    gdf = normalize_building_gdf(gdf, default_height_m=GBA_DEFAULT_BUILDING_HEIGHT_M)
    gdf = add_aoi_flag_to_gdf(gdf, str(source_file))

    if gdf is None or gdf.empty:
        print("[WARN] GBA is empty after map-region filtering / normalization.")
        return None, source_file

    inside_gdf, outside_gdf = split_inside_outside_gdf(gdf)
    stats_gdf = get_stats_gdf(gdf)
    print(f"[INFO] GBA buildings in map region: {len(gdf):,}")
    print(f"[INFO] GBA buildings inside AOI: {len(inside_gdf):,}")
    print(f"[INFO] GBA buildings outside AOI but shown: {len(outside_gdf):,}")
    print(f"[INFO] GBA height range used for stats: {stats_gdf['height_m'].min():.2f} -> {stats_gdf['height_m'].max():.2f} m")
    print(f"[INFO] GBA footprint area total used for stats: {stats_gdf['footprint_area_m2'].sum():,.2f} m2")
    return gdf, source_file


def select_gba_for_3d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if GBA_MAX_3D_BUILDINGS is None or len(gdf) <= GBA_MAX_3D_BUILDINGS:
        return gdf.copy()
    print(f"[INFO] Use largest {GBA_MAX_3D_BUILDINGS:,} GBA buildings for 3D only.")
    return gdf.sort_values("footprint_area_m2", ascending=False).head(GBA_MAX_3D_BUILDINGS).copy()


def plot_gba_3d_pygmt(gdf: gpd.GeoDataFrame, out_png: Path) -> dict | None:
    if gdf is None or gdf.empty:
        return None

    print("\n========== PLOT GBA LoD1 3D PYGMT ==========")
    plot_gdf = select_gba_for_3d(gdf)
    region2d = get_region_from_aoi()
    hmax = float(pd.to_numeric(gdf["height_m"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().max())
    if not np.isfinite(hmax) or hmax <= 0:
        hmax = 10.0
    region3d = [region2d[0], region2d[1], region2d[2], region2d[3], 0, hmax * 1.25]

    tmp_dir = make_temp_dir()
    centroid_xyz = tmp_dir / "gba_centroid_3d.xyz"
    aoi_xyz = tmp_dir / "aoi_3d.xyz"
    cpt = tmp_dir / "gba_height.cpt"

    pd.DataFrame({
        "lon": plot_gdf["centroid_lon"],
        "lat": plot_gdf["centroid_lat"],
        "z": plot_gdf["height_m"],
        "height_m": plot_gdf["height_m"],
    }).to_csv(centroid_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    pd.DataFrame([(x, y, 0.0) for x, y in HOALAC_POLYGON]).to_csv(
        aoi_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    make_building_height_cpt(gdf, cpt)

    fig = pygmt.Figure()
    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd:mmF",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="8p",
    )
    fig.basemap(
        region=region3d,
        projection=PROJECTION,
        perspective=[225, 25],
        zsize="4c",
        frame=[
            'WSneZ+t"GlobalBuildingAtlas LoD1: 3D height overview"',
            "xaf+lLongitude",
            "yaf+lLatitude",
            "zaf+lHeight (m)",
        ],
    )
    fig.plot3d(
        data=str(centroid_xyz),
        region=region3d,
        projection=PROJECTION,
        perspective=[225, 25],
        style="o0.06c",
        cmap=str(cpt),
        fill="+z",
        pen="0.08p,black",
        transparency=25,
    )
    fig.plot3d(
        data=str(aoi_xyz),
        region=region3d,
        projection=PROJECTION,
        perspective=[225, 25],
        pen=AOI_PEN,
    )
    fig.colorbar(
        cmap=str(cpt),
        perspective=[225, 25],
        position="JMR+w6c/0.35c+o0.8c/0c",
        frame=["xaf+lBuilding height", "y+l(m)"],
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved GBA PyGMT 3D: {out_png}")
    return {
        "dataset": "globalbuildingatlas_lod1",
        "source_file": "GBA building polygons",
        "plot_file": str(out_png),
        "plot_type": "gba_3d_pygmt",
        "n_records_plotted": len(plot_gdf),
    }


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is not installed. Install pyvista/vtk or set GBA_PLOT_3D_ENGINE='pygmt'."
        ) from exc
    return pv


def build_gba_pyvista_mesh(gdf: gpd.GeoDataFrame):
    pv = _import_pyvista()
    gdf = select_gba_for_3d(gdf)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise ValueError("No GBA polygons for PyVista mesh.")

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

        height_plot = height_real * GBA_PYVISTA_Z_EXAGGERATION
        has_part = False

        for poly in safe_polygons(geom):
            coords = list(poly.exterior.coords)
            if len(coords) < 4:
                continue
            coords_open = coords[:-1]
            points = np.asarray(
                [[float(x - x0), float(y - y0), 0.0] for x, y in coords_open],
                dtype=float,
            )
            n = len(points)
            if n < 3:
                continue
            faces = np.asarray([n] + list(range(n)), dtype=np.int64)
            footprint = pv.PolyData(points, faces)
            try:
                block = footprint.extrude([0.0, 0.0, float(height_plot)], capping=True)
            except TypeError:
                block = footprint.extrude([0.0, 0.0, float(height_plot)])
            if block.n_cells <= 0:
                continue
            block.cell_data["height_m"] = np.full(block.n_cells, height_real, dtype=float)
            mesh_list.append(block)
            n_parts_used += 1
            has_part = True

        if has_part:
            n_buildings_used += 1

    if not mesh_list:
        raise ValueError("No valid GBA PyVista blocks created.")

    mesh = mesh_list[0]
    for block in mesh_list[1:]:
        mesh = mesh.merge(block)

    info = {
        "utm_crs": str(utm_crs),
        "x_origin_m": float(x0),
        "y_origin_m": float(y0),
        "n_buildings_used": int(n_buildings_used),
        "n_parts_used": int(n_parts_used),
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
    }
    return mesh, info, gdf, utm_crs, x0, y0


def build_pyvista_aoi_boundary(reference_gdf: gpd.GeoDataFrame, utm_crs, x0: float, y0: float):
    pv = _import_pyvista()
    poly = get_aoi_gdf().to_crs(utm_crs).geometry.iloc[0]
    points = [[float(x - x0), float(y - y0), 0.0] for x, y in poly.exterior.coords]
    lines = [len(points)] + list(range(len(points)))
    return pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines, dtype=np.int64))


def plot_gba_3d_pyvista(gdf: gpd.GeoDataFrame, out_png: Path) -> dict | None:
    if gdf is None or gdf.empty:
        return None

    print("\n========== PLOT GBA LoD1 3D PYVISTA ==========")
    pv = _import_pyvista()
    mesh, info, plot_gdf, utm_crs, x0, y0 = build_gba_pyvista_mesh(gdf)

    for key, value in info.items():
        print(f"[INFO] {key}: {value}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    vtp_file = FIG_ROOT / "globalbuildingatlas_lod1" / "05_gba_lod1_3d_height_overview_mesh.vtp"
    html_file = out_png.with_suffix(".html")

    if GBA_PYVISTA_EXPORT_VTP:
        mesh.save(str(vtp_file))
        print(f"[OK] Saved GBA VTP mesh: {vtp_file}")

    plotter = pv.Plotter(off_screen=True, window_size=GBA_PYVISTA_WINDOW_SIZE)
    plotter.set_background(GBA_PYVISTA_BACKGROUND)

    scalar_bar_args = {
        "title": "Building height (m)",
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
        cmap=GBA_PYVISTA_CMAP,
        clim=GBA_PYVISTA_HEIGHT_CBAR_RANGE,
        show_edges=GBA_PYVISTA_SHOW_EDGES,
        edge_color=GBA_PYVISTA_EDGE_COLOR,
        line_width=GBA_PYVISTA_EDGE_WIDTH,
        opacity=GBA_PYVISTA_BUILDING_OPACITY,
        smooth_shading=False,
        scalar_bar_args=scalar_bar_args,
    )

    try:
        boundary = build_pyvista_aoi_boundary(plot_gdf, utm_crs, x0, y0)
        plotter.add_mesh(boundary, color="purple", line_width=5)
    except Exception as exc:
        print(f"[WARN] Could not add PyVista AOI boundary: {exc}")

    plotter.show_axes()
    plotter.add_text("GlobalBuildingAtlas LoD1 - Hoa Lac", position="upper_left", font_size=14, color="black")
    plotter.add_light(pv.Light(light_type="headlight", intensity=0.9))
    plotter.add_light(pv.Light(position=(0, -1, 1), focal_point=(0, 0, 0), intensity=0.35))
    plotter.disable_parallel_projection()
    plotter.view_isometric()
    plotter.camera.Azimuth(GBA_PYVISTA_CAMERA_AZIMUTH)
    plotter.camera.Elevation(GBA_PYVISTA_CAMERA_ELEVATION)
    plotter.camera.Zoom(GBA_PYVISTA_CAMERA_ZOOM)

    plotter.screenshot(str(out_png))
    print(f"[OK] Saved GBA PyVista PNG: {out_png}")

    if GBA_PYVISTA_EXPORT_HTML:
        try:
            plotter.export_html(str(html_file))
            print(f"[OK] Saved GBA PyVista HTML: {html_file}")
        except Exception as exc:
            print(f"[WARN] Could not export PyVista HTML: {exc}")

    plotter.close()

    return {
        "dataset": "globalbuildingatlas_lod1",
        "source_file": "GBA building polygons",
        "plot_file": str(out_png),
        "plot_type": "gba_3d_pyvista",
        "n_records_plotted": info["n_buildings_used"],
    }


def plot_gba_3d(gdf: gpd.GeoDataFrame) -> dict | None:
    if not GBA_PLOT_3D:
        return None
    out_png = FIG_ROOT / "globalbuildingatlas_lod1" / "05_gba_lod1_3d_height_overview.png"
    engine = str(GBA_PLOT_3D_ENGINE).lower().strip()

    if engine == "pyvista":
        try:
            return plot_gba_3d_pyvista(gdf, out_png)
        except Exception as exc:
            print(f"[WARN] GBA PyVista 3D failed: {exc}")
            if not GBA_FALLBACK_TO_PYGMT:
                raise
            print("[INFO] Fallback to PyGMT 3D.")
            return plot_gba_3d_pygmt(gdf, out_png)

    if engine == "pygmt":
        return plot_gba_3d_pygmt(gdf, out_png)

    raise ValueError("GBA_PLOT_3D_ENGINE must be 'pygmt' or 'pyvista'.")


def plot_globalbuildingatlas() -> tuple[gpd.GeoDataFrame | None, list[dict]]:
    summary = []
    gba, source_file = load_gba_buildings()
    if gba is None or gba.empty:
        return gba, summary

    out_dir = FIG_ROOT / "globalbuildingatlas_lod1"
    summary.append(plot_building_dataset_overview(
        gba,
        "GlobalBuildingAtlas LoD1 overview map",
        out_dir / "00_gba_lod1_overview_map.png",
        fill="forestgreen@45",
        pen="0.06p,darkgreen",
    ))
    summary.append(plot_building_height_map(
        gba,
        "GlobalBuildingAtlas LoD1: building height",
        out_dir / "01_gba_lod1_2d_height_overview.png",
        pen=GBA_PEN,
    ))
    summary.append(plot_building_area_map(
        gba,
        "GlobalBuildingAtlas LoD1: footprint area",
        out_dir / "02_gba_lod1_2d_footprint_area.png",
        pen=GBA_PEN,
    ))
    summary.append(plot_height_histogram(
        gba,
        "GlobalBuildingAtlas LoD1 height distribution",
        out_dir / "04_gba_lod1_height_histogram.png",
    ))
    summary.append(plot_gba_3d(gba))

    # Save a compact attribute summary beside figures.
    summary_csv = out_dir / "00_gba_lod1_summary_from_plot_script.csv"
    gba_inside, gba_outside = split_inside_outside_gdf(gba)
    gba_stats = get_stats_gdf(gba)
    pd.DataFrame([{
        "source_file": str(source_file),
        "n_buildings_map_region": int(len(gba)),
        "n_buildings_inside_aoi": int(len(gba_inside)),
        "n_buildings_outside_aoi_shown": int(len(gba_outside)),
        "stats_use_inside_aoi_only": bool(STATS_USE_INSIDE_AOI_ONLY),
        "height_min_m": float(gba_stats["height_m"].min()),
        "height_mean_m": float(gba_stats["height_m"].mean()),
        "height_median_m": float(gba_stats["height_m"].median()),
        "height_max_m": float(gba_stats["height_m"].max()),
        "total_footprint_area_m2": float(gba_stats["footprint_area_m2"].sum()),
        "total_volume_m3": float(gba_stats["volume_m3"].sum()),
    }]).to_csv(summary_csv, index=False)
    print(f"[OK] Saved GBA summary CSV: {summary_csv}")

    return gba, [s for s in summary if s is not None]


# ============================================================
# OPENBUILDINGMAP
# ============================================================

def find_obm_building_file() -> Path | None:
    obm_dir = DATASET_DIRS["openbuildingmap"]

    # Same idea as GBA: prefer bbox/unclipped OBM files if available.
    if MARK_OUTSIDE_AOI:
        candidates = [
            obm_dir / "processed" / "obm_buildings_bbox_filtered.gpkg",
            obm_dir / "bbox" / "obm_buildings_bbox_filtered.gpkg",
            obm_dir / "obm_buildings_bbox_filtered.gpkg",
            obm_dir / "clipped" / "obm_buildings_hoalac_clipped.gpkg",
            obm_dir / "obm_buildings_hoalac_clipped.gpkg",
        ]
        patterns = [
            "**/*bbox*building*.gpkg",
            "**/*bbox*filtered*.gpkg",
            "**/*obm*.gpkg",
            "**/*building*.gpkg",
            "**/*obm*building*clipped*.gpkg",
            "**/*building*clipped*.gpkg",
        ]
    else:
        candidates = [
            obm_dir / "clipped" / "obm_buildings_hoalac_clipped.gpkg",
            obm_dir / "obm_buildings_hoalac_clipped.gpkg",
        ]
        patterns = [
            "**/*obm*building*clipped*.gpkg",
            "**/*building*clipped*.gpkg",
            "**/*obm*.gpkg",
            "**/*building*.gpkg",
        ]

    for p in candidates:
        if p.exists():
            return p

    if not obm_dir.exists():
        return None
    for pat in patterns:
        for p in sorted(obm_dir.glob(pat)):
            if p.is_file() and not is_forbidden_model_file(p):
                return p
    return None


def load_obm_buildings() -> tuple[gpd.GeoDataFrame | None, Path | None]:
    source_file = find_obm_building_file()
    if source_file is None:
        print("[SKIP] No OpenBuildingMap building GPKG found.")
        return None, None

    print("\n========== LOAD OPENBUILDINGMAP ==========")
    print(f"[INFO] Source: {source_file}")
    gdf = load_vector_file(source_file, str(source_file))
    if gdf is None or gdf.empty:
        return None, source_file
    # Keep map-region data, but do not clip/delete data outside Hoa Lac.
    gdf = keep_data_in_map_region(gdf, str(source_file))
    gdf = normalize_building_gdf(gdf, default_height_m=6.0)
    gdf = add_aoi_flag_to_gdf(gdf, str(source_file))
    if gdf.empty:
        return None, source_file
    inside_gdf, outside_gdf = split_inside_outside_gdf(gdf)
    stats_gdf = get_stats_gdf(gdf)
    print(f"[INFO] OBM buildings in map region: {len(gdf):,}")
    print(f"[INFO] OBM buildings inside AOI: {len(inside_gdf):,}")
    print(f"[INFO] OBM buildings outside AOI but shown: {len(outside_gdf):,}")
    print(f"[INFO] OBM height range used for stats: {stats_gdf['height_m'].min():.2f} -> {stats_gdf['height_m'].max():.2f} m")
    return gdf, source_file


def plot_openbuildingmap() -> tuple[gpd.GeoDataFrame | None, list[dict]]:
    summary = []
    obm, source_file = load_obm_buildings()
    out_dir = FIG_ROOT / "openbuildingmap"

    if obm is not None and not obm.empty:
        summary.append(plot_building_dataset_overview(
            obm,
            "OpenBuildingMap overview map",
            out_dir / "00_obm_overview_map.png",
            fill="lightpink@55",
            pen="0.06p,gray40",
        ))
        summary.append(plot_building_height_map(
            obm,
            "OpenBuildingMap: building height",
            out_dir / "01_obm_2d_height_overview.png",
            pen=OBM_PEN,
        ))
        summary.append(plot_building_area_map(
            obm,
            "OpenBuildingMap: footprint area",
            out_dir / "02_obm_2d_footprint_area.png",
            pen=OBM_PEN,
        ))
        summary.append(plot_height_histogram(
            obm,
            "OpenBuildingMap height distribution",
            out_dir / "04_obm_height_histogram.png",
        ))

        summary_csv = out_dir / "00_obm_summary_from_plot_script.csv"
        obm_inside, obm_outside = split_inside_outside_gdf(obm)
        obm_stats = get_stats_gdf(obm)
        pd.DataFrame([{
            "source_file": str(source_file),
            "n_buildings_map_region": int(len(obm)),
            "n_buildings_inside_aoi": int(len(obm_inside)),
            "n_buildings_outside_aoi_shown": int(len(obm_outside)),
            "stats_use_inside_aoi_only": bool(STATS_USE_INSIDE_AOI_ONLY),
            "height_min_m": float(obm_stats["height_m"].min()),
            "height_mean_m": float(obm_stats["height_m"].mean()),
            "height_median_m": float(obm_stats["height_m"].median()),
            "height_max_m": float(obm_stats["height_m"].max()),
            "total_footprint_area_m2": float(obm_stats["footprint_area_m2"].sum()),
            "total_volume_m3": float(obm_stats["volume_m3"].sum()),
        }]).to_csv(summary_csv, index=False)
        print(f"[OK] Saved OBM summary CSV: {summary_csv}")

    # Also quicklook all coordinate table files inside OpenBuildingMap.
    for path in find_table_files(DATASET_DIRS["openbuildingmap"]):
        out_png = out_dir / "quicklook_tables" / f"{path.stem}.png"
        result = plot_table_quicklook(path, "openbuildingmap", out_png)
        if result is not None:
            summary.append(result)

    return obm, [s for s in summary if s is not None]


# ============================================================
# OPENTOPOGRAPHY RASTER HELPERS
# ============================================================

def find_raster_files(folder: Path) -> list[Path]:
    """Find raster/grid files commonly produced by OpenTopography downloads."""
    suffixes = {
        ".tif", ".tiff", ".vrt", ".img", ".bil", ".asc",
        ".grd", ".nc", ".netcdf", ".nc4", ".hgt",
        ".flt", ".dem", ".dt0", ".dt1", ".dt2",
    }
    files = []
    if not folder.exists():
        return files
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes and not is_forbidden_model_file(p):
            files.append(p)
    return files


def _read_raster_with_rasterio(path: Path, max_points: int = MAX_RASTER_QUICKLOOK_POINTS) -> tuple[pd.DataFrame | None, dict]:
    """Read a GeoTIFF/raster file, crop to AOI, downsample, and return lon/lat/value."""
    try:
        import rasterio
        from rasterio.windows import from_bounds, Window
        from rasterio.enums import Resampling
        from rasterio.transform import xy as rasterio_xy
        from rasterio.warp import transform as rio_transform
        from affine import Affine
    except Exception as exc:
        raise ImportError(
            "rasterio is needed for GeoTIFF/OpenTopography raster plotting. "
            "Install with: conda install -c conda-forge rasterio"
        ) from exc

    info = {"reader": "rasterio", "source_file": str(path)}

    with rasterio.open(path) as src:
        info.update({
            "crs": str(src.crs),
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "nodata": src.nodata,
        })

        # Crop to AOI bounds in source CRS when possible.
        use_full_raster = False
        try:
            if src.crs is not None:
                aoi_src = get_aoi_gdf().to_crs(src.crs)
                west, south, east, north = aoi_src.total_bounds
            else:
                west, south, east, north = get_aoi_gdf().total_bounds

            win = from_bounds(west, south, east, north, transform=src.transform)
            win = win.round_offsets().round_lengths()

            col_off = max(0, int(win.col_off))
            row_off = max(0, int(win.row_off))
            width = min(src.width - col_off, int(win.width))
            height = min(src.height - row_off, int(win.height))

            if width <= 0 or height <= 0:
                use_full_raster = True
            else:
                window = Window(col_off, row_off, width, height)
        except Exception as exc:
            print(f"[WARN] Could not crop raster to AOI, use full raster. File: {path} Reason: {exc}")
            use_full_raster = True

        if use_full_raster:
            window = Window(0, 0, src.width, src.height)
            width, height = src.width, src.height

        total_pixels = max(1, int(width * height))
        step = max(1, int(math.ceil(math.sqrt(total_pixels / float(max_points)))))
        out_width = max(1, int(math.ceil(width / step)))
        out_height = max(1, int(math.ceil(height / step)))

        data = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.bilinear,
        )

        transform = src.window_transform(window) * Affine.scale(width / out_width, height / out_height)

        arr = np.asarray(data, dtype=float)
        if np.ma.isMaskedArray(data):
            mask = np.ma.getmaskarray(data)
        else:
            mask = np.zeros(arr.shape, dtype=bool)

        if src.nodata is not None:
            mask |= np.isclose(arr, float(src.nodata), equal_nan=False)

        mask |= ~np.isfinite(arr)

        rows, cols = np.indices(arr.shape)
        xs, ys = rasterio_xy(transform, rows, cols, offset="center")
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        vals = arr.ravel()
        valid = ~mask.ravel()

        xs = xs[valid]
        ys = ys[valid]
        vals = vals[valid]

        if len(vals) == 0:
            return None, info

        if src.crs is not None and str(src.crs).upper() not in ["EPSG:4326", "OGC:CRS84"]:
            lons, lats = rio_transform(src.crs, "EPSG:4326", xs.tolist(), ys.tolist())
            lons = np.asarray(lons, dtype=float)
            lats = np.asarray(lats, dtype=float)
        else:
            lons = xs
            lats = ys

    df = pd.DataFrame({"lon": lons, "lat": lats, "value": vals})
    region = get_region_from_aoi(padding=REGION_PADDING * 2.0)
    df = df[
        df["lon"].between(region[0], region[1])
        & df["lat"].between(region[2], region[3])
    ].copy()

    if df.empty:
        return None, info

    df = add_aoi_flag_to_points_df(df)

    info.update({
        "n_points": int(len(df)),
        "value_min": float(pd.to_numeric(df["value"], errors="coerce").min()),
        "value_max": float(pd.to_numeric(df["value"], errors="coerce").max()),
        "downsample_step": int(step),
    })

    return df, info


def _read_grid_with_pygmt(path: Path, max_points: int = MAX_RASTER_QUICKLOOK_POINTS) -> tuple[pd.DataFrame | None, dict]:
    """Fallback for GMT/NetCDF grids when rasterio cannot read them."""
    info = {"reader": "pygmt.grd2xyz", "source_file": str(path)}
    try:
        xyz = pygmt.grd2xyz(grid=str(path), output_type="pandas")
    except Exception as exc:
        print(f"[WARN] PyGMT could not read grid {path}: {exc}")
        return None, info

    if xyz is None or xyz.empty or xyz.shape[1] < 3:
        return None, info

    xyz = xyz.iloc[:, :3].copy()
    xyz.columns = ["lon", "lat", "value"]
    for col in ["lon", "lat", "value"]:
        xyz[col] = pd.to_numeric(xyz[col], errors="coerce")
    xyz = xyz.dropna(subset=["lon", "lat", "value"]).copy()
    xyz = xyz[xyz["lon"].between(-180, 180) & xyz["lat"].between(-90, 90)].copy()

    region = get_region_from_aoi(padding=REGION_PADDING * 2.0)
    xyz = xyz[
        xyz["lon"].between(region[0], region[1])
        & xyz["lat"].between(region[2], region[3])
    ].copy()

    if xyz.empty:
        return None, info

    xyz = sample_points(xyz, max_points)
    xyz = add_aoi_flag_to_points_df(xyz)
    info.update({
        "n_points": int(len(xyz)),
        "value_min": float(xyz["value"].min()),
        "value_max": float(xyz["value"].max()),
    })
    return xyz, info


def read_raster_coordinate_file(path: Path, max_points: int = MAX_RASTER_QUICKLOOK_POINTS) -> tuple[pd.DataFrame | None, dict]:
    """Read OpenTopography raster/grid as sampled lon/lat/value points."""
    try:
        return _read_raster_with_rasterio(path, max_points=max_points)
    except Exception as exc:
        print(f"[WARN] rasterio reader failed for {path}: {exc}")

    # Try PyGMT for .grd/.nc fallback.
    if path.suffix.lower() in {".grd", ".nc", ".netcdf"}:
        return _read_grid_with_pygmt(path, max_points=max_points)

    return None, {"reader": "none", "source_file": str(path)}

def mask_grid_outside_aoi(grid_file: Path, out_grid: Path) -> Path | None:
    """
    Mask a GMT/PyGMT grid outside HOALAC_POLYGON.

    Outside AOI is set to NaN.
    This avoids using fig.clip(), which is not reliable/available in some PyGMT versions.
    """
    try:
        import xarray as xr
        from matplotlib.path import Path as MplPath
    except Exception as exc:
        print(f"[WARN] Cannot import xarray/matplotlib for grid mask: {exc}")
        return None

    try:
        ds = xr.open_dataset(grid_file)
        var_names = list(ds.data_vars)
        if not var_names:
            print(f"[WARN] No data variable found in grid: {grid_file}")
            return None

        var_name = var_names[0]
        da = ds[var_name]

        # Detect x/y coordinate names.
        coord_names = list(da.coords)
        if "lon" in coord_names:
            x_name = "lon"
        elif "x" in coord_names:
            x_name = "x"
        else:
            x_name = da.dims[-1]

        if "lat" in coord_names:
            y_name = "lat"
        elif "y" in coord_names:
            y_name = "y"
        else:
            y_name = da.dims[-2]

        x = da[x_name].to_numpy()
        y = da[y_name].to_numpy()

        xx, yy = np.meshgrid(x, y)
        poly_path = MplPath(np.asarray(HOALAC_POLYGON, dtype=float))

        inside = poly_path.contains_points(
            np.column_stack([xx.ravel(), yy.ravel()])
        ).reshape(xx.shape)

        mask_da = xr.DataArray(
            inside,
            coords={y_name: y, x_name: x},
            dims=(y_name, x_name),
        )

        # Make sure data order is y/x before masking.
        if da.dims != mask_da.dims:
            da = da.transpose(y_name, x_name)

        da_masked = da.where(mask_da)
        da_masked.name = var_name
        da_masked.to_netcdf(out_grid)

        ds.close()
        return out_grid

    except Exception as exc:
        print(f"[WARN] Failed to mask grid outside AOI: {exc}")
        return None

def plot_raster_quicklook(
    path: Path,
    dataset: str,
    out_png: Path,
    df: pd.DataFrame | None = None,
    info: dict | None = None,
) -> dict | None:
    """
    Plot OpenTopography raster/grid.

    Outside-AOI data are not deleted. In surface mode, the full map-region grid is shown
    faintly first, then the inside-AOI polygon is overlaid in full color.
    """
    if df is None:
        df, info = read_raster_coordinate_file(path)

    if df is None or df.empty:
        return None

    df = add_aoi_flag_to_points_df(df)
    inside_df, outside_df = split_inside_outside_points(df)

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir(prefix="raster_surface_")

    cpt = tmp_dir / "raster_value.cpt"
    grid = tmp_dir / "raster_surface.nc"
    inside_xyz = tmp_dir / "raster_points_inside.xyz"
    outside_xy = tmp_dir / "raster_points_outside.xy"
    aoi_xy = tmp_dir / "aoi.xy"

    name = path.name.lower()
    cmap = TOPO_HILLSHADE_CMAP if "shade" in name or "hill" in name else TOPO_SURFACE_CMAP

    cbar_label = "Raster value"
    if any(k in name for k in ["dem", "terrain", "elevation", "srtm", "topography"]):
        cbar_label = "Elevation (m)"
    elif "slope" in name:
        cbar_label = "Slope"
    elif any(k in name for k in ["hillshade", "shade"]):
        cbar_label = "Hillshade"

    stats_df = get_stats_points_df(df)
    make_value_cpt(stats_df["value"], cpt, cmap=cmap)
    save_aoi_xy(aoi_xy)

    fig = start_map(region, f"{dataset}: {path.name}")

    if PLOT_TOPO_SURFACE:
        grid_file = surface_points_low_memory(
            df=df,
            out_grid=grid,
            region=region,
            spacing=TOPO_SURFACE_SPACING,
            max_points=MAX_TOPO_SURFACE_POINTS,
            tension=TOPO_SURFACE_TENSION,
        )

        if grid_file is None or not grid_file.exists():
            remove_temp_dir(tmp_dir)
            print(f"[WARN] Could not create smooth surface for raster: {path}")
            return None

        try:
            inside_grid = tmp_dir / "raster_surface_inside_aoi.nc"

            if MARK_OUTSIDE_AOI:
                inside_grid_file = mask_grid_outside_aoi(grid_file, inside_grid)

                if inside_grid_file is not None and inside_grid_file.exists():
                    fig.grdimage(
                        grid=str(inside_grid_file),
                        cmap=str(cpt),
                        nan_transparent=True,
                    )
                else:
                    print("[WARN] AOI grid mask failed. Plot full raster instead.")
                    fig.grdimage(
                        grid=str(grid_file),
                        cmap=str(cpt),
                        nan_transparent=True,
                    )
            else:
                fig.grdimage(
                    grid=str(grid_file),
                    cmap=str(cpt),
                    nan_transparent=True,
                )
        except Exception as exc:
            print(f"[WARN] Polygon clip failed for raster surface. Plot without clip. Reason: {exc}")
            fig.grdimage(
                grid=str(grid_file),
                cmap=str(cpt),
                nan_transparent=True,
            )

        plot_type = "raster_surface_quicklook_mark_outside_aoi"
        n_records_plotted = min(len(df), MAX_TOPO_SURFACE_POINTS)
        ok_msg = "smooth raster surface"

    else:
        if MARK_OUTSIDE_AOI and not outside_df.empty:
            outside_df[["lon", "lat"]].to_csv(
                outside_xy,
                sep=" ",
                index=False,
                header=False,
                float_format="%.8f",
            )
            fig.plot(
                data=str(outside_xy),
                style="c0.020c",
                fill=OUTSIDE_AOI_POINT_FILL,
                pen=None,
                transparency=OUTSIDE_AOI_POINT_TRANSPARENCY,
                label="Outside Hoa Lac polygon",
            )
        if not inside_df.empty:
            plot_df = sample_points(inside_df, MAX_RASTER_QUICKLOOK_POINTS)
            plot_df[["lon", "lat", "value"]].to_csv(
                inside_xyz,
                sep=" ",
                index=False,
                header=False,
                float_format="%.8f",
            )
            fig.plot(
                data=str(inside_xyz),
                style="c0.025c",
                cmap=str(cpt),
                fill="+z",
                pen=None,
                transparency=20,
                label="Inside Hoa Lac polygon",
            )

        plot_type = "raster_point_quicklook_mark_outside_aoi"
        n_records_plotted = len(df)
        ok_msg = "raster point quicklook"

    fig.plot(data=str(aoi_xy), pen=AOI_PEN, label="Hoa Lac boundary")
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w7c/0.35c+o0.7c",
        frame=[f"xaf+l{cbar_label}"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    if MARK_OUTSIDE_AOI and not outside_df.empty:
        fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved {ok_msg}: {out_png}")
    return {
        "dataset": dataset,
        "source_file": str(path),
        "plot_file": str(out_png),
        "plot_type": plot_type,
        "n_records_plotted": n_records_plotted,
        "n_inside_aoi": int(len(inside_df)),
        "n_outside_aoi": int(len(outside_df)),
        "value_min": None if info is None else info.get("value_min"),
        "value_max": None if info is None else info.get("value_max"),
        "reader": None if info is None else info.get("reader"),
        "surface_spacing": TOPO_SURFACE_SPACING if PLOT_TOPO_SURFACE else None,
        "surface_tension": TOPO_SURFACE_TENSION if PLOT_TOPO_SURFACE else None,
    }

def surface_points_low_memory(
    df: pd.DataFrame,
    out_grid: Path,
    region: list[float],
    spacing: str = TOPO_SURFACE_SPACING,
    max_points: int = MAX_TOPO_SURFACE_POINTS,
    tension: float = TOPO_SURFACE_TENSION,
) -> Path | None:
    """Create a temporary surface grid from lon/lat/value points using low-memory steps."""
    if df is None or df.empty:
        return None

    work = df[["lon", "lat", "value"]].copy()
    work = work.dropna(subset=["lon", "lat", "value"]).copy()
    if work.empty:
        return None

    work = sample_points(work, max_points)
    tmp_dir = out_grid.parent
    in_xyz = tmp_dir / f"{out_grid.stem}_input.xyz"
    block_xyz = tmp_dir / f"{out_grid.stem}_blockmean.xyz"

    work.to_csv(in_xyz, sep=" ", index=False, header=False, float_format="%.8f")

    try:
        pygmt.blockmean(
            data=str(in_xyz),
            region=region,
            spacing=spacing,
            outfile=str(block_xyz),
            output_type="file",
        )
    except Exception as exc:
        print(f"[WARN] blockmean failed, use sampled XYZ directly for surface. Reason: {exc}")
        block_xyz = in_xyz

    try:
        pygmt.surface(
            data=str(block_xyz),
            region=region,
            spacing=spacing,
            tension=tension,
            outgrid=str(out_grid),
        )
    except Exception as exc:
        print(f"[WARN] surface failed: {exc}")
        return None

    return out_grid


def choose_best_topography_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    def score(item: dict) -> int:
        name = str(item.get("source_file", "")).lower()
        info = item.get("info", {}) or {}
        n = int(len(item.get("df", []))) if item.get("df") is not None else 0
        s = 0
        for k in ["dem", "terrain", "elevation", "srtm", "topography"]:
            if k in name:
                s += 25
        for k in ["slope"]:
            if k in name:
                s += 12
        for k in ["hillshade", "shade"]:
            if k in name:
                s += 8
        if item.get("kind") == "raster":
            s += 20
        s += min(20, n // 5000)
        if info.get("reader") == "rasterio":
            s += 5
        return s

    return sorted(candidates, key=score, reverse=True)[0]


def plot_opentopography_overview_map(candidate: dict, out_png: Path) -> dict | None:
    """Create one overview map for the OpenTopography subdirectory using surface/grdimage."""
    if candidate is None:
        return None

    df = candidate.get("df")
    path = Path(candidate.get("source_file"))
    if df is None or len(df) == 0:
        return None

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir(prefix="topo_surface_")
    grid = tmp_dir / "topography_surface.nc"
    cpt = tmp_dir / "topography_surface.cpt"
    outside_cpt = tmp_dir / "topography_outside_white.cpt"
    aoi_xy = tmp_dir / "aoi.xy"

    grid_file = surface_points_low_memory(df, grid, region)
    if grid_file is None or (not grid_file.exists()):
        remove_temp_dir(tmp_dir)
        return None

    vals = pd.to_numeric(df["value"], errors="coerce")
    name = path.name.lower()
    cmap = TOPO_HILLSHADE_CMAP if "shade" in name or "hill" in name else TOPO_SURFACE_CMAP
    cbar_label = "Raster value"
    if any(k in name for k in ["dem", "terrain", "elevation", "srtm", "topography"]):
        cbar_label = "Elevation (m)"
    elif "slope" in name:
        cbar_label = "Slope"
    elif any(k in name for k in ["hillshade", "shade"]):
        cbar_label = "Hillshade"

    df = add_aoi_flag_to_points_df(df)
    stats_df = get_stats_points_df(df)
    make_value_cpt(stats_df["value"], cpt, cmap=cmap)
    save_aoi_xy(aoi_xy)

    fig = start_map(region, path.stem)
    try:
        inside_grid = tmp_dir / "raster_surface_inside_aoi.nc"

        if MARK_OUTSIDE_AOI:
            inside_grid_file = mask_grid_outside_aoi(grid_file, inside_grid)

            if inside_grid_file is not None and inside_grid_file.exists():
                fig.grdimage(
                    grid=str(inside_grid_file),
                    cmap=str(cpt),
                    nan_transparent=True,
                )
            else:
                print("[WARN] AOI grid mask failed. Plot full raster instead.")
                fig.grdimage(
                    grid=str(grid_file),
                    cmap=str(cpt),
                    nan_transparent=True,
                )
        else:
            fig.grdimage(
                grid=str(grid_file),
                cmap=str(cpt),
                nan_transparent=True,
            )
    except Exception:
        fig.grdimage(grid=str(grid_file), cmap=str(cpt), nan_transparent=True)

    fig.plot(data=str(aoi_xy), pen=AOI_PEN, label="Hoa Lac boundary")
    add_aoi_area_text_box(fig, region)
    fig.colorbar(
        cmap=str(cpt),
        position="JMR+w10c/0.45c+o0.5c/0.0c+v",
        frame=[f"xaf+l{cbar_label}"],
    )
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.legend(position="JBL+jBL+o0.2c/0.2c", box="+gwhite@70+p0.5p,black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved OpenTopography overview map: {out_png}")
    return {
        "dataset": "opentopography",
        "source_file": str(path),
        "plot_file": str(out_png),
        "plot_type": "dataset_overview_map_surface",
        "n_records_plotted": int(len(df)),
    }


# ============================================================
# OPENTOPOGRAPHY
# ============================================================

def write_opentopography_inventory(folder: Path, out_dir: Path) -> pd.DataFrame:
    """Save a detailed OpenTopography file inventory for debugging missing DEM plots."""
    records = []
    if folder.exists():
        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue
            try:
                size_mb = p.stat().st_size / 1024.0 / 1024.0
            except Exception:
                size_mb = np.nan
            records.append({
                "relative_path": str(p.relative_to(folder)),
                "suffix": p.suffix.lower(),
                "size_mb": size_mb,
                "is_raster_candidate": p.suffix.lower() in {
                    ".tif", ".tiff", ".vrt", ".img", ".bil", ".asc", ".grd", ".nc", ".netcdf", ".nc4", ".hgt", ".flt", ".dem", ".dt0", ".dt1", ".dt2",
                },
                "is_table_candidate": p.suffix.lower() in {".xyz", ".csv", ".txt", ".dat"},
                "is_vector_candidate": p.suffix.lower() in {".gpkg", ".geojson", ".json", ".shp"},
                "skipped_summary_metadata_name": table_name_is_summary_or_metadata(p),
                "forbidden_model_file": is_forbidden_model_file(p),
            })

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "00_opentopography_file_inventory.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved OpenTopography inventory: {out_csv}")
    return df


def choose_best_opentopography_map(summary: list[dict]) -> dict | None:
    """Choose the most DEM-like OpenTopography plot for reporting/overview."""
    if not summary:
        return None

    def score(item: dict) -> int:
        name = str(item.get("source_file", "")).lower()
        s = 0
        for k in ["dem", "elevation", "srtm", "terrain", "topography"]:
            if k in name:
                s += 20
        for k in ["slope"]:
            if k in name:
                s += 10
        for k in ["hillshade", "shade"]:
            if k in name:
                s += 5
        if item.get("plot_type") == "raster_quicklook":
            s += 30
        if item.get("plot_type") == "table_coordinate_quicklook":
            s += 15
        s += min(10, int(item.get("n_records_plotted", 0) or 0) // 1000)
        return s

    return sorted(summary, key=score, reverse=True)[0]


def plot_opentopography() -> tuple[list[pd.DataFrame], list[dict]]:
    print("\n========== PLOT OPENTOPOGRAPHY ==========")
    out_dir = FIG_ROOT / "opentopography"
    topo_tables = []
    summary = []
    topo_candidates = []

    folder = DATASET_DIRS["opentopography"]
    if not folder.exists():
        print(f"[SKIP] Missing folder: {folder}")
        return topo_tables, summary

    inventory = write_opentopography_inventory(folder, out_dir)
    if not inventory.empty:
        print("[INFO] OpenTopography files found by suffix:")
        print(inventory.groupby("suffix").size().sort_values(ascending=False).to_string())

    # 1) Raster/grid files: GeoTIFF, NetCDF, GMT grid, etc.
    raster_files = find_raster_files(folder)
    if raster_files:
        print(f"[INFO] Found OpenTopography raster/grid files: {len(raster_files)}")
    else:
        print("[WARN] No OpenTopography raster/grid files detected by suffix.")

    for path in raster_files:
        print(f"[INFO] Try OpenTopography raster/grid: {path}")
        df, info = read_raster_coordinate_file(path, max_points=MAX_RASTER_QUICKLOOK_POINTS)
        if df is None or df.empty:
            print(f"[WARN] Could not extract coordinates from OpenTopography raster: {path}")
            continue

        topo_tables.append(sample_points(df, MAX_RASTER_OVERVIEW_POINTS))
        topo_candidates.append({"source_file": str(path), "df": df.copy(), "info": info, "kind": "raster"})
        out_png = out_dir / "quicklook_rasters" / f"{path.stem}.png"
        result = plot_raster_quicklook(path, "opentopography", out_png, df=df, info=info)
        if result is not None:
            summary.append(result)

    # 2) XYZ/CSV/TXT tables. Skip summary/statistics as spatial maps.
    table_files_all = find_table_files(folder)
    table_files = [p for p in table_files_all if not table_name_is_summary_or_metadata(p)]
    skipped_tables = [p for p in table_files_all if table_name_is_summary_or_metadata(p)]

    if table_files_all:
        print(f"[INFO] Found OpenTopography table files: {len(table_files_all)}")
    if skipped_tables:
        print(f"[INFO] Skip summary/statistics tables as maps: {len(skipped_tables)}")
        for p in skipped_tables[:20]:
            print(f"       - {p}")
    if table_files:
        print(f"[INFO] Try spatial OpenTopography table files: {len(table_files)}")

    for path in table_files:
        print(f"[INFO] Try OpenTopography table: {path}")
        df = read_table_coordinate_file(path)
        if df is None or df.empty:
            print(f"[WARN] Not a spatial coordinate table or outside AOI: {path}")
            continue
        topo_tables.append(sample_points(df, MAX_OVERVIEW_POINTS))
        topo_candidates.append({"source_file": str(path), "df": df.copy(), "info": {"reader": "table"}, "kind": "table"})
        out_png = out_dir / "quicklook_tables" / f"{path.stem}.png"
        result = plot_table_quicklook(path, "opentopography", out_png)
        if result is not None:
            summary.append(result)

    # 3) Vector files, if present.
    vector_files = find_vector_files(folder)
    if vector_files:
        print(f"[INFO] Found OpenTopography vector files: {len(vector_files)}")
    for path in vector_files:
        print(f"[INFO] Try OpenTopography vector: {path}")
        out_png = out_dir / "quicklook_vectors" / f"{path.stem}.png"
        result = plot_vector_quicklook(path, "opentopography", out_png)
        if result is not None:
            summary.append(result)

    best_candidate = choose_best_topography_candidate(topo_candidates)
    if best_candidate is not None:
        result = plot_opentopography_overview_map(
            best_candidate,
            out_dir / "00_opentopography_overview_map.png",
        )
        if result is not None:
            summary.insert(0, result)

    best = choose_best_opentopography_map(summary)
    if best is not None:
        print("[INFO] Best OpenTopography spatial plot:")
        print(f"       source: {best.get('source_file')}")
        print(f"       figure: {best.get('plot_file')}")

    if not summary:
        print("[WARN] No OpenTopography spatial data plotted.")
        print("[INFO] The script did touch/list all OpenTopography files in:")
        print(f"       {out_dir / '00_opentopography_file_inventory.csv'}")
        print("[INFO] If this CSV only contains terrain_statistics/summary files, then the DEM/grid file was not copied into output/01_HoaLac_studies_area/opentopography.")
        print("[INFO] Checked raster/grid suffixes: .tif .tiff .vrt .img .bil .asc .grd .nc .netcdf .nc4 .hgt .flt .dem .dt0 .dt1 .dt2")
        print("[INFO] Checked spatial table suffixes: .xyz .csv .txt .dat, including lon/lat and UTM EPSG:32648 x/y")
        print("[INFO] Checked vector suffixes: .gpkg .geojson .json .shp")

    return topo_tables, summary

# ============================================================
# OSM
# ============================================================

def load_osm_vectors() -> list[tuple[Path, gpd.GeoDataFrame]]:
    folder = DATASET_DIRS["osm"]
    out = []
    if not folder.exists():
        return out

    for path in find_vector_files(folder):
        gdf = load_vector_file(path, str(path))
        if gdf is None or gdf.empty:
            continue

        if REMOVE_OUTSIDE_OSM:
            gdf = clip_gdf_to_aoi_geometry(gdf, str(path))
        else:
            gdf = prepare_vector_for_map_region(gdf, str(path))

        if gdf is None or gdf.empty:
            continue

        out.append((path, gdf))

    return out


def plot_osm() -> tuple[list[tuple[Path, gpd.GeoDataFrame]], list[dict]]:
    print("\n========== PLOT OSM ==========")
    out_dir = FIG_ROOT / "osm"
    summary = []

    vectors = load_osm_vectors()
    for path, gdf in vectors:
        out_png = out_dir / "quicklook_vectors" / f"{path.stem}.png"

        name = path.stem.lower()
        if "extra" in name and "feature" in name:
            result = plot_osm_extra_features_map(path, out_png, gdf=gdf)
        else:
            result = plot_vector_quicklook(path, "osm", out_png)

        if result is not None:
            summary.append(result)

    # Some OSM exports may be xyz/csv road points.
    for path in find_table_files(DATASET_DIRS["osm"]):
        out_png = out_dir / "quicklook_tables" / f"{path.stem}.png"
        result = plot_table_quicklook(path, "osm", out_png)
        if result is not None:
            summary.append(result)

    overview = plot_osm_dataset_overview(
        vectors,
        out_dir / "00_osm_overview_map.png",
    )
    if overview is not None:
        summary.insert(0, overview)

    if not summary:
        print("[WARN] No OSM data plotted.")

    return vectors, summary


# ============================================================
# COMBINED OVERVIEW MAP
# ============================================================


def save_polygons_for_overview(gdf: gpd.GeoDataFrame, out_xy: Path, max_features: int) -> int:
    if gdf is None or gdf.empty:
        out_xy.write_text("", encoding="utf-8")
        return 0
    plot_gdf = gdf.copy()
    if len(plot_gdf) > max_features:
        if "footprint_area_m2" in plot_gdf.columns:
            plot_gdf = plot_gdf.sort_values("footprint_area_m2", ascending=False).head(max_features).copy()
        else:
            plot_gdf = plot_gdf.head(max_features).copy()
    save_geometries_segments(plot_gdf, out_xy)
    return len(plot_gdf)


def save_polygons_by_aoi_for_overview(
    gdf: gpd.GeoDataFrame,
    out_inside_xy: Path,
    out_outside_xy: Path,
    max_features: int,
) -> tuple[int, int]:
    """Save separate inside/outside polygon segments for combined overview."""
    if gdf is None or gdf.empty:
        out_inside_xy.write_text("", encoding="utf-8")
        out_outside_xy.write_text("", encoding="utf-8")
        return 0, 0

    plot_gdf = gdf.copy()
    if len(plot_gdf) > max_features:
        if "footprint_area_m2" in plot_gdf.columns:
            plot_gdf = plot_gdf.sort_values("footprint_area_m2", ascending=False).head(max_features).copy()
        else:
            plot_gdf = plot_gdf.head(max_features).copy()

    inside_gdf, outside_gdf = split_inside_outside_gdf(plot_gdf)
    if not inside_gdf.empty:
        save_geometries_segments(inside_gdf, out_inside_xy)
    else:
        out_inside_xy.write_text("", encoding="utf-8")
    if not outside_gdf.empty:
        save_geometries_segments(outside_gdf, out_outside_xy)
    else:
        out_outside_xy.write_text("", encoding="utf-8")
    return len(inside_gdf), len(outside_gdf)


def save_osm_lines_for_overview(osm_vectors: list[tuple[Path, gpd.GeoDataFrame]], out_xy: Path) -> int:
    n = 0
    with open(out_xy, "w", encoding="utf-8") as f:
        for path, gdf in osm_vectors:
            gdf = safe_to_crs_4326(gdf, str(path))
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type in ["LineString", "LinearRing"]:
                    f.write(">\n")
                    for x, y in geom.coords:
                        f.write(f"{x:.8f} {y:.8f}\n")
                    n += 1
                elif geom.geom_type == "MultiLineString":
                    for line in geom.geoms:
                        f.write(">\n")
                        for x, y in line.coords:
                            f.write(f"{x:.8f} {y:.8f}\n")
                        n += 1
    return n


def save_osm_lines_by_aoi_for_overview(
    osm_vectors: list[tuple[Path, gpd.GeoDataFrame]],
    out_inside_xy: Path,
    out_outside_xy: Path,
) -> tuple[int, int]:
    """
    Save only OSM line pieces inside Hoa Lac.

    Outside OSM data are removed.
    """
    n_inside = 0
    out_outside_xy.write_text("", encoding="utf-8")

    with open(out_inside_xy, "w", encoding="utf-8") as f_in:
        for path, gdf in osm_vectors:
            gdf = clip_gdf_to_aoi_geometry(gdf, str(path))
            if gdf is None or gdf.empty:
                continue

            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                for part in iter_flat_geometries(geom):
                    if part is None or part.is_empty:
                        continue

                    if part.geom_type in ["LineString", "LinearRing"]:
                        coords = list(part.coords)
                        if len(coords) >= 2:
                            f_in.write(">\n")
                            for x, y in coords:
                                f_in.write(f"{x:.8f} {y:.8f}\n")
                            n_inside += 1

    return n_inside, 0

def plot_overview_map(
    gba: gpd.GeoDataFrame | None,
    obm: gpd.GeoDataFrame | None,
    topo_tables: list[pd.DataFrame],
    osm_vectors: list[tuple[Path, gpd.GeoDataFrame]],
) -> dict:
    """
    Combined overview map without topography.

    Layers plotted:
      - GBA buildings
      - OBM buildings
      - OSM lines inside AOI only
      - Hoa Lac boundary

    Topography is intentionally not plotted here.
    """
    print("\n========== PLOT COMBINED INPUT OVERVIEW ==========")
    print("[INFO] Topography layer is disabled for combined overview map.")

    # Keep function signature unchanged because main() still passes topo_tables.
    _ = topo_tables

    region = get_region_from_aoi()
    tmp_dir = make_temp_dir(prefix="combined_overview_")

    aoi_xy = tmp_dir / "aoi.xy"

    gba_inside_xy = tmp_dir / "gba_inside.xy"
    gba_outside_xy = tmp_dir / "gba_outside.xy"

    obm_inside_xy = tmp_dir / "obm_inside.xy"
    obm_outside_xy = tmp_dir / "obm_outside.xy"

    osm_lines_inside_xy = tmp_dir / "osm_lines_inside.xy"
    osm_lines_outside_xy = tmp_dir / "osm_lines_outside.xy"

    save_aoi_xy(aoi_xy)

    n_gba_inside, n_gba_outside = (
        save_polygons_by_aoi_for_overview(
            gba,
            gba_inside_xy,
            gba_outside_xy,
            MAX_OVERVIEW_POLYGONS,
        )
        if gba is not None
        else (0, 0)
    )

    n_obm_inside, n_obm_outside = (
        save_polygons_by_aoi_for_overview(
            obm,
            obm_inside_xy,
            obm_outside_xy,
            MAX_OVERVIEW_POLYGONS,
        )
        if obm is not None
        else (0, 0)
    )

    # With REMOVE_OUTSIDE_OSM=True, this should return outside = 0.
    n_osm_lines_inside, n_osm_lines_outside = save_osm_lines_by_aoi_for_overview(
        osm_vectors,
        osm_lines_inside_xy,
        osm_lines_outside_xy,
    )

    fig = start_map(region, "Downloaded input data overview: Hoa Lac")

    # Outside GBA/OBM only.
    # OSM outside has been removed and is not plotted.
    if MARK_OUTSIDE_AOI:
        if n_gba_outside > 0:
            fig.plot(
                data=str(gba_outside_xy),
                fill=OUTSIDE_AOI_FILL,
                pen=OUTSIDE_AOI_PEN,
                label="Outside Hoa Lac polygon",
            )

        if n_obm_outside > 0:
            fig.plot(
                data=str(obm_outside_xy),
                fill=OUTSIDE_AOI_FILL,
                pen=OUTSIDE_AOI_PEN,
            )

    if n_obm_inside > 0:
        fig.plot(
            data=str(obm_inside_xy),
            fill="red@50",
            pen="0.05p,orange",
            label="OpenBuildingMap inside",
        )

    if n_gba_inside > 0:
        fig.plot(
            data=str(gba_inside_xy),
            fill="green@75",
            pen="0.05p,green",
            label="GBA LoD1 inside",
        )

    if n_osm_lines_inside > 0:
        fig.plot(
            data=str(osm_lines_inside_xy),
            pen="0.8p,blue",
            label="OSM lines inside",
        )

    fig.plot(
        data=str(aoi_xy),
        pen=AOI_PEN,
        label="Hoa Lac boundary",
    )

    add_aoi_area_text_box(fig, region)
    fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l")
    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@15+p0.5p,black",
    )

    out_png = FIG_ROOT / "01_input_data_overview_map.png"
    fig.savefig(str(out_png), dpi=DPI)

    remove_temp_dir(tmp_dir)

    print(f"[OK] Saved overview map: {out_png}")

    if n_osm_lines_outside > 0:
        print(
            f"[WARN] OSM outside line count is {n_osm_lines_outside:,}. "
            "Check save_osm_lines_by_aoi_for_overview() if REMOVE_OUTSIDE_OSM=True."
        )

    area_m2, area_km2, area_ha = get_aoi_area_stats()

    n_records = int(
        n_gba_inside
        + n_gba_outside
        + n_obm_inside
        + n_obm_outside
        + n_osm_lines_inside
    )

    n_inside = int(
        n_gba_inside
        + n_obm_inside
        + n_osm_lines_inside
    )

    n_outside = int(
        n_gba_outside
        + n_obm_outside
    )

    return {
        "dataset": "all",
        "source_file": str(INPUT_ROOT),
        "plot_file": str(out_png),
        "plot_type": "combined_overview_without_topography",
        "n_records_plotted": n_records,
        "n_inside_aoi": n_inside,
        "n_outside_aoi": n_outside,
        "n_osm_outside_aoi": 0,
        "study_area_m2": area_m2,
        "study_area_km2": area_km2,
        "study_area_ha": area_ha,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("\n========== HOA LAC DOWNLOADED INPUT DATA PLOTS ==========")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Figure root: {FIG_ROOT}")
    area_m2, area_km2, area_ha = get_aoi_area_stats()
    print(f"Study area: {area_m2:,.0f} m2 | {area_km2:.4f} km2 | {area_ha:.2f} ha")
    print("[INFO] This script does not read path-finding model files such as raw.xyz.")

    inventory = collect_inventory() if TOUCH_ALL_FILES else pd.DataFrame()

    all_summary: list[dict] = []

    gba, gba_summary = plot_globalbuildingatlas()
    all_summary.extend(gba_summary)

    obm, obm_summary = plot_openbuildingmap()
    all_summary.extend(obm_summary)

    topo_tables, topo_summary = plot_opentopography()
    all_summary.extend(topo_summary)

    osm_vectors, osm_summary = plot_osm()
    all_summary.extend(osm_summary)

    overview_summary = plot_overview_map(gba, obm, topo_tables, osm_vectors)
    all_summary.append(overview_summary)

    write_report(inventory, all_summary)

    print("\n========== DONE ==========")
    print(f"Inventory CSV: {FIG_ROOT / '00_input_data_inventory.csv'}")
    print(f"Report TXT:    {FIG_ROOT / '00_input_data_report.txt'}")
    print(f"Overview map:  {FIG_ROOT / '01_input_data_overview_map.png'}")
    print("Dataset figure folders:")
    for key in DATASET_DIRS:
        print(f"  - {FIG_ROOT / key}")


if __name__ == "__main__":
    main()
