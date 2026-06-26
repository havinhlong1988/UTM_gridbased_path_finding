#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download OpenInfraMap-like infrastructure layers around Hoa Lac and plot/check them.

Important note
--------------
OpenInfraMap is a visualization of infrastructure data mapped in OpenStreetMap.
It does not provide a direct "download this map view" file endpoint. This script
therefore downloads the same type of infrastructure data from OpenStreetMap via
OSMnx / Overpass API, using tags commonly displayed by OpenInfraMap.

Input reference map view:
    https://openinframap.org/#10.2/21.0355/105.4836

Main outputs:
    output_openinframap_hoalac/
    ├── openinframap_view_bbox.gpkg
    ├── hoalac_polygon.gpkg
    ├── bbox_layers/
    │   ├── power_lines_bbox.gpkg
    │   ├── power_substations_bbox.gpkg
    │   ├── power_supports_bbox.gpkg
    │   ├── power_plants_generators_bbox.gpkg
    │   ├── solar_generation_bbox.gpkg
    │   ├── telecom_masts_towers_bbox.gpkg
    │   ├── telecom_facilities_bbox.gpkg
    │   ├── pipelines_bbox.gpkg
    │   ├── oil_gas_facilities_bbox.gpkg
    │   └── water_infrastructure_bbox.gpkg
    ├── hoalac_clipped_layers/
    │   └── *_hoalac.gpkg
    ├── xyz/
    │   └── *_vertices_or_points_*.xyz
    ├── openinframap_layer_availability_summary.csv
    ├── openinframap_layer_availability_summary.txt
    └── figures/
        ├── bbox/
        │   ├── 00_openinframap_all_available_layers_bbox.png
        │   ├── 00a_openinframap_availability_counts_bbox.png
        │   ├── 00b_openinframap_all_layers_gallery_bbox.png
        │   └── 01_...individual layer maps...
        └── hoalac/
            ├── 00_openinframap_all_available_layers_hoalac.png
            ├── 00a_openinframap_availability_counts_hoalac.png
            ├── 00b_openinframap_all_layers_gallery_hoalac.png
            └── 01_...individual layer maps...

XYZ format:
    lon lat value

For line layers, the XYZ value is z_max_agl_m or layer_code depending on the
layer. For point layers, the XYZ value is height_m or layer_code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Approximate Hoa Lac study polygon, format: lon, lat.
# Used for optional clipping and overlay.
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

# OpenInfraMap view URL parameters:
# https://openinframap.org/#10.2/21.0355/105.4836 means:
#   zoom = 10.2, center latitude = 21.0355, center longitude = 105.4836
OIM_ZOOM = 10.2
OIM_CENTER_LAT = 21.0355
OIM_CENTER_LON = 105.4836

# Because the URL does not encode the browser window size, we approximate the
# viewed area with a user-controlled rectangular bbox around the center.
# Increase if you want the wider OpenInfraMap view.
OIM_HALF_WIDTH_KM = 35.0
OIM_HALF_HEIGHT_KM = 25.0

# Output folder.
OUTDIR = "output/01_HoaLac_studies_area/openinframap"

# If True, also clip all layers to Hoa Lac polygon and generate a second set of
# plots for the clipped AOI. The bbox plots are always produced.
CLIP_TO_HOALAC_POLYGON = True

# Export XYZ files for each layer.
EXPORT_XYZ = True

# Save figures.
PLOT_FIGURES = True

# Network settings.
OVERPASS_TIMEOUT_SEC = 240
OVERPASS_SLEEP_BETWEEN_LAYERS_SEC = 2.0

# Assumed heights for UAV obstacle preprocessing when OSM has no explicit height.
# Treat these as conservative placeholders until you verify by DSM/LiDAR/survey.
DEFAULT_HEIGHTS_M = {
    "power_line": 35.0,
    "power_minor_line": 15.0,
    "power_cable": 0.0,
    "power_tower": 35.0,
    "power_pole": 12.0,
    "power_substation": 8.0,
    "power_plant": 20.0,
    "power_generator": 12.0,
    "solar_generation": 3.0,
    "telecom_mast": 35.0,
    "telecom_tower": 45.0,
    "telecom_facility": 15.0,
    "pipeline": 0.0,
    "oil_gas_facility": 12.0,
    "water_tower": 25.0,
    "water_works": 10.0,
    "water_facility": 10.0,
    "unknown": 5.0,
}

# Layer codes for XYZ export and combined plotting.
LAYER_CODE = {
    "power_lines": 11,
    "power_substations": 12,
    "power_supports": 13,
    "power_plants_generators": 14,
    "solar_generation": 15,
    "telecom_masts_towers": 21,
    "telecom_facilities": 22,
    "pipelines": 31,
    "oil_gas_facilities": 32,
    "water_infrastructure": 41,
}

# If a layer has no explicit height data, height_m and z_max_agl_m will be
# estimated from these defaults. Keep height_source for confidence control.


# ============================================================
# LAYER DEFINITIONS
# ============================================================

@dataclass
class LayerSpec:
    name: str
    title: str
    tags: dict
    expected_geometry: str   # "line", "point", "polygon", "mixed"
    plot_value: str          # usually "z_max_agl_m", "height_m", or "layer_code"
    voxel_role: str
    use_now: str
    note: str


# These OSM tags are selected to approximate the OpenInfraMap layers:
# power, solar generation, telecoms, oil/gas pipelines/facilities, and water infrastructure.
LAYER_SPECS = [
    LayerSpec(
        name="power_lines",
        title="Power lines / cables",
        tags={"power": ["line", "minor_line", "cable"]},
        expected_geometry="line",
        plot_value="z_max_agl_m",
        voxel_role="hard obstacle if overhead; cable may be underground",
        use_now="yes_with_height_assumption",
        note="Use power=line/minor_line as UAV obstacle; check cable location before blocking.",
    ),
    LayerSpec(
        name="power_substations",
        title="Power substations / transformers / switchgear",
        tags={"power": ["substation", "transformer", "switchgear", "compensator", "converter"]},
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="hard obstacle / restricted infrastructure risk",
        use_now="yes_with_height_assumption",
        note="Important infrastructure; block structure volume or use as high-risk buffer.",
    ),
    LayerSpec(
        name="power_supports",
        title="Power towers / poles / portals / terminals",
        tags={"power": ["tower", "pole", "portal", "terminal"]},
        expected_geometry="point",
        plot_value="height_m",
        voxel_role="hard obstacle",
        use_now="yes_with_height_assumption",
        note="Good point obstacle layer; height commonly missing.",
    ),
    LayerSpec(
        name="power_plants_generators",
        title="Power plants / generators",
        tags={"power": ["plant", "generator"]},
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="hard obstacle / high-risk infrastructure",
        use_now="yes_with_height_assumption",
        note="Use as infrastructure footprint; actual height varies by facility.",
    ),
    LayerSpec(
        name="solar_generation",
        title="Solar generation",
        tags={"generator:source": "solar", "plant:source": "solar"},
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="low structure obstacle / land-use risk",
        use_now="yes_low_altitude",
        note="Usually low height but can occupy large areas.",
    ),
    LayerSpec(
        name="telecom_masts_towers",
        title="Telecom / communication masts and towers",
        tags={
            "man_made": ["mast", "tower", "communications_tower"],
            "tower:type": ["communication", "communications"],
        },
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="hard obstacle",
        use_now="yes_with_height_assumption",
        note="Very important vertical obstacle layer for UAV flight.",
    ),
    LayerSpec(
        name="telecom_facilities",
        title="Telecom facilities",
        tags={"telecom": True, "communication": True},
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="hard obstacle if mast/tower; otherwise soft infrastructure risk",
        use_now="check_geometry_first",
        note="Tags are inconsistent; inspect before voxel blocking.",
    ),
    LayerSpec(
        name="pipelines",
        title="Pipelines",
        tags={"man_made": "pipeline", "pipeline": True, "route": "pipeline"},
        expected_geometry="line",
        plot_value="layer_code",
        voxel_role="usually not air obstacle; operational risk / construction corridor",
        use_now="soft_risk_or_ignore_for_air_obstacle",
        note="Most pipelines are underground; do not hard-block UAV airspace unless exposed/elevated.",
    ),
    LayerSpec(
        name="oil_gas_facilities",
        title="Oil / gas facilities",
        tags={
            "substance": ["gas", "oil", "petroleum", "natural_gas"],
            "industrial": ["oil", "gas"],
            "man_made": ["gasometer", "petroleum_well", "flare"],
        },
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="high-risk infrastructure / possible hard obstacle footprint",
        use_now="yes_as_restricted_or_high_risk_buffer",
        note="Use conservative buffer; feature meaning can vary.",
    ),
    LayerSpec(
        name="water_infrastructure",
        title="Water infrastructure",
        tags={
            "man_made": ["water_tower", "water_works", "wastewater_plant", "water_well", "reservoir_covered"],
            "water": ["reservoir", "basin"],
            "pipeline": ["water", "sewer", "drain"]
        },
        expected_geometry="mixed",
        plot_value="height_m",
        voxel_role="hard obstacle if tower/tank; otherwise soft infrastructure risk",
        use_now="yes_for_towers_check_others",
        note="Water towers are hard obstacles; underground water/sewer pipelines are not air obstacles.",
    ),
]


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


def make_oim_bbox_gdf() -> gpd.GeoDataFrame:
    """Approximate map-view bbox from OpenInfraMap URL center and half-size in km."""
    lat = OIM_CENTER_LAT
    lon = OIM_CENTER_LON

    dlat = OIM_HALF_HEIGHT_KM / 111.320
    dlon = OIM_HALF_WIDTH_KM / (111.320 * math.cos(math.radians(lat)))

    west = lon - dlon
    east = lon + dlon
    south = lat - dlat
    north = lat + dlat

    geom = box(west, south, east, north)
    return gpd.GeoDataFrame(
        {
            "name": ["OpenInfraMap_view_bbox_approx"],
            "zoom": [OIM_ZOOM],
            "center_lat": [OIM_CENTER_LAT],
            "center_lon": [OIM_CENTER_LON],
            "half_width_km": [OIM_HALF_WIDTH_KM],
            "half_height_km": [OIM_HALF_HEIGHT_KM],
        },
        geometry=[geom],
        crs="EPSG:4326",
    )


def bbox_tuple_from_gdf(gdf: gpd.GeoDataFrame):
    west, south, east, north = gdf.to_crs("EPSG:4326").total_bounds
    return float(west), float(south), float(east), float(north)


def safe_clip(gdf: gpd.GeoDataFrame, clip_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(gdf.copy() if gdf is not None else {}, geometry=[], crs="EPSG:4326")

    try:
        out = gpd.clip(gdf.to_crs("EPSG:4326"), clip_gdf.to_crs("EPSG:4326"))
    except Exception as exc:
        print(f"[WARN] Clip failed: {exc}")
        return gpd.GeoDataFrame(gdf.iloc[0:0].copy(), geometry="geometry", crs=gdf.crs or "EPSG:4326")

    if out.empty:
        return gpd.GeoDataFrame(gdf.iloc[0:0].copy(), geometry="geometry", crs="EPSG:4326")

    return out.reset_index(drop=True).to_crs("EPSG:4326")


# ============================================================
# DATA CLEANING / ATTRIBUTES
# ============================================================

def to_scalar_str(value):
    """Convert list/dict/set values to a safe string for GeoPackage writing."""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    return value


def get_first_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return None
        return list(value)[0]
    return value


def parse_float_from_osm(value):
    value = get_first_value(value)
    if value is None:
        return np.nan

    raw = str(value).lower().strip()
    if raw in ["", "none", "nan", "unknown"]:
        return np.nan

    # Normalize common height/length forms.
    raw = raw.replace("meters", "m")
    raw = raw.replace("meter", "m")
    raw = raw.replace("metres", "m")
    raw = raw.replace("metre", "m")
    raw = raw.replace(",", ".")

    # If feet are explicitly given, convert to meters.
    is_ft = "ft" in raw or "feet" in raw
    for token in ["m", "ft", "feet", "~", "approx", "approximately"]:
        raw = raw.replace(token, "")
    raw = raw.strip()

    # OSM can contain values like "10;12". Use first numeric part.
    raw = raw.split(";")[0].strip()

    try:
        val = float(raw)
    except ValueError:
        return np.nan

    if is_ft:
        val *= 0.3048
    return val


def parse_voltage_kv(value):
    value = get_first_value(value)
    if value is None:
        return np.nan

    raw = str(value).lower().replace("v", "").replace("kv", "").replace(",", ".")
    raw = raw.split(";")[0].strip()
    try:
        val = float(raw)
    except ValueError:
        return np.nan

    # OSM voltage is commonly in volts. Convert large values to kV.
    if val > 1000:
        val /= 1000.0
    return val


def infer_infra_class(row, layer_name: str) -> str:
    power = str(get_first_value(row.get("power")) or "").lower()
    man_made = str(get_first_value(row.get("man_made")) or "").lower()
    telecom = str(get_first_value(row.get("telecom")) or "").lower()
    pipeline = str(get_first_value(row.get("pipeline")) or "").lower()
    substance = str(get_first_value(row.get("substance")) or "").lower()
    generator_source = str(get_first_value(row.get("generator:source")) or "").lower()
    plant_source = str(get_first_value(row.get("plant:source")) or "").lower()

    if layer_name == "power_lines":
        if power == "line":
            return "power_line"
        if power == "minor_line":
            return "power_minor_line"
        if power == "cable":
            return "power_cable"
        return "power_line"

    if layer_name == "power_supports":
        if power == "tower":
            return "power_tower"
        if power == "pole":
            return "power_pole"
        return "power_tower"

    if layer_name == "power_substations":
        return "power_substation"

    if layer_name == "power_plants_generators":
        if power == "plant":
            return "power_plant"
        if power == "generator":
            return "power_generator"
        return "power_generator"

    if layer_name == "solar_generation" or generator_source == "solar" or plant_source == "solar":
        return "solar_generation"

    if layer_name == "telecom_masts_towers":
        if man_made == "tower" or "tower" in man_made:
            return "telecom_tower"
        return "telecom_mast"

    if layer_name == "telecom_facilities" or telecom not in ["", "none", "nan"]:
        return "telecom_facility"

    if layer_name == "pipelines" or pipeline not in ["", "none", "nan"]:
        return "pipeline"

    if layer_name == "oil_gas_facilities" or substance in ["gas", "oil", "petroleum", "natural_gas"]:
        return "oil_gas_facility"

    if layer_name == "water_infrastructure":
        if man_made == "water_tower":
            return "water_tower"
        if man_made in ["water_works", "wastewater_plant"]:
            return "water_works"
        return "water_facility"

    return "unknown"


def estimate_height(row, layer_name: str):
    """Return height_m, z_min_agl_m, z_max_agl_m, height_source."""
    explicit = np.nan
    for key in ["height", "est_height", "tower:height", "building:height"]:
        if key in row:
            explicit = parse_float_from_osm(row.get(key))
            if np.isfinite(explicit):
                break

    infra_class = infer_infra_class(row, layer_name)
    default_h = DEFAULT_HEIGHTS_M.get(infra_class, DEFAULT_HEIGHTS_M["unknown"])

    if np.isfinite(explicit) and explicit > 0:
        height_m = float(explicit)
        source = "osm_height"
    else:
        height_m = float(default_h)
        source = f"assumed_{infra_class}"

    # Special handling for power lines/cables/pipelines.
    if layer_name == "power_lines":
        power = str(get_first_value(row.get("power")) or "").lower()
        location = str(get_first_value(row.get("location")) or "").lower()
        layer = str(get_first_value(row.get("layer")) or "").lower()

        if power == "cable" or location in ["underground", "underwater"] or layer.startswith("-"):
            z_min = 0.0
            z_max = 0.0
            if source.startswith("assumed"):
                source = "assumed_underground_or_cable_not_air_obstacle"
        elif power == "minor_line":
            z_min = 6.0
            z_max = height_m if height_m > 0 else DEFAULT_HEIGHTS_M["power_minor_line"]
        else:
            z_min = 8.0
            z_max = height_m if height_m > 0 else DEFAULT_HEIGHTS_M["power_line"]
        return height_m, z_min, z_max, source

    if layer_name == "pipelines":
        return height_m, 0.0, 0.0, "usually_underground_not_air_obstacle"

    return height_m, 0.0, height_m, source


def sanitize_gdf(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame({"layer_name": pd.Series(dtype="str")}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    gdf = gdf.copy()

    # Drop non-spatial rows and invalid empty geometries.
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        return gpd.GeoDataFrame({"layer_name": pd.Series(dtype="str")}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)

    # Add attributes useful for UAV voxel preprocessing.
    gdf["layer_name"] = layer_name
    gdf["layer_code"] = int(LAYER_CODE.get(layer_name, 99))
    gdf["infra_class"] = gdf.apply(lambda row: infer_infra_class(row, layer_name), axis=1)
    gdf["voltage_kv"] = gdf["voltage"].apply(parse_voltage_kv) if "voltage" in gdf.columns else np.nan

    heights = gdf.apply(lambda row: estimate_height(row, layer_name), axis=1)
    gdf["height_m"] = [h[0] for h in heights]
    gdf["z_min_agl_m"] = [h[1] for h in heights]
    gdf["z_max_agl_m"] = [h[2] for h in heights]
    gdf["height_source"] = [h[3] for h in heights]

    # Mark likely UAV hard obstacle usefulness.
    gdf["uav_obstacle_role"] = "check"
    if layer_name in ["power_lines", "power_supports", "telecom_masts_towers"]:
        gdf["uav_obstacle_role"] = "hard_obstacle"
    elif layer_name in ["power_substations", "power_plants_generators", "oil_gas_facilities"]:
        gdf["uav_obstacle_role"] = "hard_obstacle_or_high_risk_buffer"
    elif layer_name == "pipelines":
        gdf["uav_obstacle_role"] = "not_air_obstacle_usually_soft_risk"
    elif layer_name == "water_infrastructure":
        gdf["uav_obstacle_role"] = "hard_if_tower_check_other_features"
    elif layer_name == "solar_generation":
        gdf["uav_obstacle_role"] = "low_altitude_obstacle_or_landuse_risk"

    # Convert troublesome object/list columns to strings for GPKG writing.
    for col in list(gdf.columns):
        if col == "geometry":
            continue
        if gdf[col].dtype == "object":
            gdf[col] = gdf[col].apply(to_scalar_str)

    return gdf


# ============================================================
# DOWNLOAD LAYERS
# ============================================================

def features_from_bbox_compat(west, south, east, north, tags):
    """OSMnx compatibility wrapper for different versions."""
    try:
        # OSMnx 2.x
        return ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    except TypeError:
        try:
            return ox.features_from_bbox((west, south, east, north), tags=tags)
        except TypeError:
            # Older OSMnx style: north, south, east, west, tags
            return ox.features_from_bbox(north, south, east, west, tags)


def download_layer(spec: LayerSpec, west, south, east, north) -> gpd.GeoDataFrame:
    print(f"\n[INFO] Downloading {spec.title}")
    print(f"       tags = {spec.tags}")

    try:
        gdf = features_from_bbox_compat(west, south, east, north, spec.tags)
    except Exception as exc:
        print(f"[WARN] Failed to download {spec.name}: {exc}")
        return sanitize_gdf(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), spec.name)

    if gdf is None or gdf.empty:
        print(f"[WARN] No features found for {spec.name}.")
        return sanitize_gdf(gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), spec.name)

    gdf = sanitize_gdf(gdf, spec.name)
    print(f"[OK] {spec.name}: {len(gdf)} features")
    return gdf


# ============================================================
# EXPORT HELPERS
# ============================================================

def write_gpkg(gdf: gpd.GeoDataFrame, out_gpkg: Path):
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if gdf is None or gdf.empty:
        empty = gpd.GeoDataFrame(
            {
                "layer_name": pd.Series(dtype="str"),
                "layer_code": pd.Series(dtype="int"),
                "height_m": pd.Series(dtype="float"),
                "z_min_agl_m": pd.Series(dtype="float"),
                "z_max_agl_m": pd.Series(dtype="float"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        print(f"[WARN] Empty GPKG saved: {out_gpkg}")
        return

    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"[OK] Saved: {out_gpkg}")


def geom_to_xyz_records(gdf: gpd.GeoDataFrame, value_col: str):
    if gdf is None or gdf.empty:
        return []

    records = []
    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        try:
            value = float(row.get(value_col, row.get("layer_code", 99)))
        except Exception:
            value = 99.0

        if geom.geom_type == "Point":
            records.append((geom.x, geom.y, value))

        elif geom.geom_type == "MultiPoint":
            for pt in geom.geoms:
                records.append((pt.x, pt.y, value))

        elif geom.geom_type == "LineString":
            for x, y in geom.coords:
                records.append((x, y, value))
            records.append((np.nan, np.nan, np.nan))

        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for x, y in line.coords:
                    records.append((x, y, value))
                records.append((np.nan, np.nan, np.nan))

        elif geom.geom_type == "Polygon":
            for x, y in geom.exterior.coords:
                records.append((x, y, value))
            records.append((np.nan, np.nan, np.nan))

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                for x, y in poly.exterior.coords:
                    records.append((x, y, value))
                records.append((np.nan, np.nan, np.nan))

    return records


def write_xyz(gdf: gpd.GeoDataFrame, out_xyz: Path, value_col: str):
    out_xyz = Path(out_xyz)
    out_xyz.parent.mkdir(parents=True, exist_ok=True)

    records = geom_to_xyz_records(gdf, value_col=value_col)
    if not records:
        out_xyz.write_text("")
        print(f"[WARN] Empty XYZ saved: {out_xyz}")
        return

    df = pd.DataFrame(records, columns=["lon", "lat", value_col])
    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved XYZ: {out_xyz}")


# ============================================================
# SUMMARY / AVAILABILITY
# ============================================================

def geometry_counts(gdf: gpd.GeoDataFrame):
    if gdf is None or gdf.empty:
        return {"point": 0, "line": 0, "polygon": 0, "other": 0}

    geom_types = gdf.geometry.geom_type.value_counts().to_dict()
    return {
        "point": int(geom_types.get("Point", 0) + geom_types.get("MultiPoint", 0)),
        "line": int(geom_types.get("LineString", 0) + geom_types.get("MultiLineString", 0)),
        "polygon": int(geom_types.get("Polygon", 0) + geom_types.get("MultiPolygon", 0)),
        "other": int(sum(v for k, v in geom_types.items() if k not in ["Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"])),
    }


def make_availability_summary(layer_data: dict, specs: list[LayerSpec], scope_name: str) -> pd.DataFrame:
    records = []
    spec_lookup = {s.name: s for s in specs}

    for name, gdf in layer_data.items():
        spec = spec_lookup[name]
        n = int(len(gdf)) if gdf is not None else 0
        counts = geometry_counts(gdf)
        status = "available" if n > 0 else "missing_or_not_mapped_in_area"

        if n > 0 and "height_source" in gdf.columns:
            height_sources = ";".join(sorted(set(str(v) for v in gdf["height_source"].dropna().unique())))
            hmin = float(np.nanmin(gdf["height_m"])) if "height_m" in gdf and len(gdf) else np.nan
            hmax = float(np.nanmax(gdf["height_m"])) if "height_m" in gdf and len(gdf) else np.nan
            zmax = float(np.nanmax(gdf["z_max_agl_m"])) if "z_max_agl_m" in gdf and len(gdf) else np.nan
        else:
            height_sources = "none"
            hmin = np.nan
            hmax = np.nan
            zmax = np.nan

        records.append({
            "scope": scope_name,
            "layer": name,
            "title": spec.title,
            "feature_count": n,
            "status": status,
            "point_count": counts["point"],
            "line_count": counts["line"],
            "polygon_count": counts["polygon"],
            "other_geom_count": counts["other"],
            "height_min_m": hmin,
            "height_max_m": hmax,
            "z_max_agl_max_m": zmax,
            "height_source": height_sources,
            "voxel_role": spec.voxel_role,
            "use_now": spec.use_now,
            "note": spec.note,
        })

    return pd.DataFrame(records)


def save_summary(summary_df: pd.DataFrame, out_csv: Path, out_txt: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved summary CSV: {out_csv}")

    lines = []
    lines.append("========== OPENINFRAMAP / OSM INFRASTRUCTURE LAYER SUMMARY ==========")
    for scope in summary_df["scope"].unique():
        sub = summary_df[summary_df["scope"] == scope]
        lines.append(f"\n--- Scope: {scope} ---")
        for _, row in sub.iterrows():
            lines.append(
                f"{row['layer']:28s} | n={int(row['feature_count']):5d} | "
                f"points={int(row['point_count']):4d} lines={int(row['line_count']):4d} polys={int(row['polygon_count']):4d} | "
                f"status={row['status']} | use={row['use_now']}"
            )
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved summary TXT: {out_txt}")
    print("\n" + "\n".join(lines))


# ============================================================
# PLOTTING HELPERS
# ============================================================

def setup_axis(ax, boundary_gdf=None, hoalac_gdf=None, title=""):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1.2)

    if hoalac_gdf is not None and not hoalac_gdf.empty:
        hoalac_gdf.boundary.plot(ax=ax, linewidth=1.5)


def plot_layer_individual(gdf: gpd.GeoDataFrame, spec: LayerSpec, boundary_gdf, hoalac_gdf, out_png: Path, scope_name: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    setup_axis(ax, boundary_gdf=boundary_gdf, hoalac_gdf=hoalac_gdf, title=f"{spec.title} ({scope_name})")

    if gdf is None or gdf.empty:
        ax.text(0.5, 0.5, "No features found", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    else:
        col = spec.plot_value if spec.plot_value in gdf.columns else "layer_code"
        # Split geometry types so point markers remain visible on top of lines/polygons.
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        points = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]

        legend = True
        if not polys.empty:
            polys.plot(ax=ax, column=col, legend=legend, alpha=0.45, linewidth=0.6, edgecolor="black", cmap="viridis")
            legend = False
        if not lines.empty:
            lines.plot(ax=ax, column=col, legend=legend, linewidth=1.4, cmap="viridis")
            legend = False
        if not points.empty:
            points.plot(ax=ax, column=col, legend=legend, markersize=28, cmap="viridis", edgecolor="black", linewidth=0.3)

        ax.text(
            0.02,
            0.02,
            f"Features: {len(gdf)}\nColor: {col}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_combined_layers(layer_data: dict, specs: list[LayerSpec], boundary_gdf, hoalac_gdf, out_png: Path, scope_name: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    setup_axis(ax, boundary_gdf=boundary_gdf, hoalac_gdf=hoalac_gdf, title=f"All available OpenInfraMap-like layers ({scope_name})")

    plotted = 0
    for i, spec in enumerate(specs):
        gdf = layer_data.get(spec.name)
        if gdf is None or gdf.empty:
            continue

        # Plot without using a fixed style-heavy scheme; default cycle is enough.
        label = f"{spec.title} (n={len(gdf)})"
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        points = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]

        if not polys.empty:
            polys.boundary.plot(ax=ax, linewidth=0.9, label=label)
            label = None
        if not lines.empty:
            lines.plot(ax=ax, linewidth=1.2, label=label)
            label = None
        if not points.empty:
            points.plot(ax=ax, markersize=18, label=label)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No infrastructure layers found", transform=ax.transAxes, ha="center", va="center", fontsize=14)
    else:
        ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    print(f"[OK] Saved combined figure: {out_png}")


def plot_availability_counts(summary_df: pd.DataFrame, out_png: Path, scope_name: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sub = summary_df[summary_df["scope"] == scope_name].copy()
    sub = sub.sort_values("feature_count", ascending=True)

    fig_h = max(5.5, 0.45 * len(sub) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(sub["layer"], sub["feature_count"])
    ax.set_xlabel("Feature count")
    ax.set_title(f"OpenInfraMap-like layer availability ({scope_name})")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)

    for y, val in enumerate(sub["feature_count"]):
        ax.text(val + max(1, sub["feature_count"].max() * 0.01), y, str(int(val)), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved count figure: {out_png}")


def plot_gallery(layer_data: dict, specs: list[LayerSpec], boundary_gdf, hoalac_gdf, out_png: Path, scope_name: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(specs)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, spec in zip(axes, specs):
        gdf = layer_data.get(spec.name)
        setup_axis(ax, boundary_gdf=boundary_gdf, hoalac_gdf=hoalac_gdf, title=f"{spec.title}")

        if gdf is None or gdf.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11)
            continue

        col = spec.plot_value if spec.plot_value in gdf.columns else "layer_code"
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        points = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]

        if not polys.empty:
            polys.plot(ax=ax, column=col, alpha=0.45, linewidth=0.4, edgecolor="black", cmap="viridis")
        if not lines.empty:
            lines.plot(ax=ax, column=col, linewidth=1.0, cmap="viridis")
        if not points.empty:
            points.plot(ax=ax, column=col, markersize=18, cmap="viridis", edgecolor="black", linewidth=0.2)

        ax.text(
            0.02,
            0.02,
            f"n={len(gdf)}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    for ax in axes[len(specs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] Saved gallery figure: {out_png}")


def plot_all_figures(layer_data: dict, specs: list[LayerSpec], summary_df: pd.DataFrame, boundary_gdf, hoalac_gdf, fig_dir: Path, scope_name: str):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_layers(
        layer_data=layer_data,
        specs=specs,
        boundary_gdf=boundary_gdf,
        hoalac_gdf=hoalac_gdf,
        out_png=fig_dir / f"00_openinframap_all_available_layers_{scope_name}.png",
        scope_name=scope_name,
    )

    plot_availability_counts(
        summary_df=summary_df,
        out_png=fig_dir / f"00a_openinframap_availability_counts_{scope_name}.png",
        scope_name=scope_name,
    )

    plot_gallery(
        layer_data=layer_data,
        specs=specs,
        boundary_gdf=boundary_gdf,
        hoalac_gdf=hoalac_gdf,
        out_png=fig_dir / f"00b_openinframap_all_layers_gallery_{scope_name}.png",
        scope_name=scope_name,
    )

    for i, spec in enumerate(specs, start=1):
        safe_title = spec.name
        plot_layer_individual(
            gdf=layer_data.get(spec.name),
            spec=spec,
            boundary_gdf=boundary_gdf,
            hoalac_gdf=hoalac_gdf,
            out_png=fig_dir / f"{i:02d}_{safe_title}_{spec.plot_value}_{scope_name}.png",
            scope_name=scope_name,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== OPENINFRAMAP-LIKE INFRASTRUCTURE DOWNLOAD ==========")
    print("[INFO] Data source: OpenStreetMap via OSMnx / Overpass API")
    print("[INFO] Target map view: https://openinframap.org/#10.2/21.0355/105.4836")
    print(f"[INFO] Approx bbox half-width/height: {OIM_HALF_WIDTH_KM} km x {OIM_HALF_HEIGHT_KM} km")

    # Prepare geometry.
    hoalac_gdf = make_hoalac_polygon_gdf()
    bbox_gdf = make_oim_bbox_gdf()

    hoalac_file = outdir / "hoalac_polygon.gpkg"
    bbox_file = outdir / "openinframap_view_bbox.gpkg"
    hoalac_gdf.to_file(hoalac_file, driver="GPKG")
    bbox_gdf.to_file(bbox_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {hoalac_file}")
    print(f"[OK] Saved OpenInfraMap view bbox: {bbox_file}")

    west, south, east, north = bbox_tuple_from_gdf(bbox_gdf)
    print("\n[INFO] Download bbox:")
    print(f"  WEST  = {west:.6f}")
    print(f"  SOUTH = {south:.6f}")
    print(f"  EAST  = {east:.6f}")
    print(f"  NORTH = {north:.6f}")

    # Configure OSMnx.
    ox.settings.timeout = OVERPASS_TIMEOUT_SEC
    ox.settings.use_cache = True
    ox.settings.log_console = False

    bbox_layers_dir = outdir / "bbox_layers"
    clip_layers_dir = outdir / "hoalac_clipped_layers"
    xyz_dir = outdir / "xyz"

    bbox_data = {}
    hoalac_data = {}

    # Download each layer.
    for spec in LAYER_SPECS:
        gdf_bbox = download_layer(spec, west, south, east, north)
        bbox_data[spec.name] = gdf_bbox

        write_gpkg(gdf_bbox, bbox_layers_dir / f"{spec.name}_bbox.gpkg")

        if EXPORT_XYZ:
            value_col = spec.plot_value if spec.plot_value in gdf_bbox.columns else "layer_code"
            write_xyz(gdf_bbox, xyz_dir / f"{spec.name}_bbox.xyz", value_col=value_col)

        if CLIP_TO_HOALAC_POLYGON:
            gdf_clip = safe_clip(gdf_bbox, hoalac_gdf)
            # Re-sanitize after clipping to ensure CRS and core fields survive.
            if gdf_clip is not None and not gdf_clip.empty:
                gdf_clip = sanitize_gdf(gdf_clip, spec.name)
            hoalac_data[spec.name] = gdf_clip
            write_gpkg(gdf_clip, clip_layers_dir / f"{spec.name}_hoalac.gpkg")

            if EXPORT_XYZ:
                value_col = spec.plot_value if gdf_clip is not None and spec.plot_value in gdf_clip.columns else "layer_code"
                write_xyz(gdf_clip, xyz_dir / f"{spec.name}_hoalac.xyz", value_col=value_col)

        if OVERPASS_SLEEP_BETWEEN_LAYERS_SEC > 0:
            time.sleep(OVERPASS_SLEEP_BETWEEN_LAYERS_SEC)

    # Summaries.
    summary_parts = []
    bbox_summary = make_availability_summary(bbox_data, LAYER_SPECS, scope_name="bbox")
    summary_parts.append(bbox_summary)

    if CLIP_TO_HOALAC_POLYGON:
        hoalac_summary = make_availability_summary(hoalac_data, LAYER_SPECS, scope_name="hoalac")
        summary_parts.append(hoalac_summary)

    summary_df = pd.concat(summary_parts, ignore_index=True)
    save_summary(
        summary_df,
        out_csv=outdir / "openinframap_layer_availability_summary.csv",
        out_txt=outdir / "openinframap_layer_availability_summary.txt",
    )

    # Plot figures.
    if PLOT_FIGURES:
        figures_dir = outdir / "figures"

        plot_all_figures(
            layer_data=bbox_data,
            specs=LAYER_SPECS,
            summary_df=summary_df,
            boundary_gdf=bbox_gdf,
            hoalac_gdf=hoalac_gdf,
            fig_dir=figures_dir / "bbox",
            scope_name="bbox",
        )

        if CLIP_TO_HOALAC_POLYGON:
            plot_all_figures(
                layer_data=hoalac_data,
                specs=LAYER_SPECS,
                summary_df=summary_df,
                boundary_gdf=hoalac_gdf,
                hoalac_gdf=hoalac_gdf,
                fig_dir=figures_dir / "hoalac",
                scope_name="hoalac",
            )

    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")
    print("\nImportant files:")
    print(f"  Summary CSV: {outdir / 'openinframap_layer_availability_summary.csv'}")
    print(f"  Summary TXT: {outdir / 'openinframap_layer_availability_summary.txt'}")
    print(f"  Bbox figures: {outdir / 'figures' / 'bbox'}")
    if CLIP_TO_HOALAC_POLYGON:
        print(f"  Hoa Lac figures: {outdir / 'figures' / 'hoalac'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
