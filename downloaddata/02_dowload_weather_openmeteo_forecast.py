#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download operational Open-Meteo forecast data for Hoa Lac study area
and plot weather surface maps.

Main wind-plot behavior:
    - Wind speed and wind direction are combined.
    - Surface background = wind speed.
    - Arrows = wind direction angle from north.
    - Colorbar = wind speed.
    - Original data points are shown as circles.

Output:
    output/01_HoaLac_studies_area/openmeteo/
        openmeteo_hoalac_next1h_latest.csv
        openmeteo_hoalac_next1h_YYYYMMDD_HHMMSSUTC.csv

    output/01_HoaLac_studies_area/openmeteo/figures/{REQUEST_UTC}/
        lead_01_YYYYMMDD_HHMMUTC_wind_speed_120m_plus_direction_surface.png
        lead_01_YYYYMMDD_HHMMUTC_precipitation_surface.png
        ...

Run:
    python 02_dowload_weather_openmeteo_forecast.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import math
import re

import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point
from scipy.interpolate import griddata
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

# ============================================================
# STUDY AREA
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

OUT_DIR = Path("output/01_HoaLac_studies_area/openmeteo")


# ============================================================
# TIME CONTROL
# ============================================================

# Two modes:
#   "auto_forecast" = use latest Open-Meteo forecast.
#   "fixed_utc"     = request a fixed UTC time window.
TIME_MODE = "auto_forecast"

# Used only when TIME_MODE = "fixed_utc".
# Format:
#   YYYY-MM-DDTHH:MM:SSZ
MANUAL_REQUEST_UTC = "2026-06-16T08:00:00Z"

# If True:
#   now = 08:43 UTC -> request_utc label/output folder = 08:00 UTC
ROUND_AUTO_REQUEST_TO_PREVIOUS_HOUR = True

# User request: forecast for next/current 1 forecast hour.
FORECAST_HOURS = 1

# Use UTC/GMT from Open-Meteo.
TIMEZONE = "GMT"


# ============================================================
# OPEN-METEO PARAMETERS
# ============================================================

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# About 0.01 degree is roughly 1 km in latitude.
# Use 0.005 for denser sampling.
GRID_STEP_DEG = 0.01

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",

    "pressure_msl",
    "surface_pressure",

    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",

    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",

    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",

    "wind_gusts_10m",

    "precipitation",
    "rain",
    "showers",
    "precipitation_probability",

    "weather_code",
    "cape",
    "boundary_layer_height",
]


# ============================================================
# PLOTTING OPTIONS
# ============================================================

# True  = plot every downloaded weather layer.
# False = plot only rain, wind speed+direction, wind gust, humidity.
PLOT_FULL_DATA = False

# If PLOT_FULL_DATA = False, plot only these operational layers.
# Wind speed and wind direction are combined automatically.
SELECTED_PLOT_VARIABLES = [
    # Rain / precipitation
    "precipitation",
    "rain",
    "showers",
    "precipitation_probability",

    # Combined wind maps will be generated from these speed layers
    # plus matching wind_direction_* layers.
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",

    # Gust
    "wind_gusts_10m",

    # Humidity
    "relative_humidity_2m",
    "dew_point_2m",

    # Flight safety
    "visibility",
    "cloud_cover_low",
]

PLOT_EXTRA_FIELDS = [
    "weather_risk_0_1",
    "weather_no_fly",
]

PLOT_DERIVED_FIELDS = True

# Surface plotting
PLOT_SURFACE = True
PLOT_DATA_POINTS = True

FIG_DPI = 180
FIG_SIZE = (8.2, 6.8)

# Higher value = smoother surface, but slower
SURFACE_NX = 500
SURFACE_NY = 500

# Reverse seismic colorbar
SURFACE_CMAP = "seismic"
WIND_SPEED_CMAP = "seismic"

# Smooth interpolated surface
APPLY_SURFACE_SMOOTHING = True

# Higher sigma = smoother surface
# Good range: 1.0 - 3.0
SURFACE_SMOOTH_SIGMA = 2.0

# Keep smoothed values inside original data min/max
CLIP_SMOOTHED_SURFACE = True

# Use cubic for continuous data if possible
CONTINUOUS_INTERPOLATION_METHOD = "cubic"

# Direction colormap is not used as the background in combined maps,
# because background is wind speed. Direction is shown by arrows.
DIRECTION_CMAP = "twilight_shifted"

POINT_SIZE = 45
POINT_EDGE_WIDTH = 0.45

# Variables that should use nearest interpolation.
# Direction and categorical data should not be linearly smoothed.
NEAREST_SURFACE_VARS = {
    "weather_code",
    "weather_no_fly",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
}


# ============================================================
# WIND DIRECTION ARROW OPTIONS
# ============================================================

PLOT_WIND_DIRECTION_ARROWS = True

# "bearing_from_north":
#     Arrow points to the angle measured clockwise from north.
#     0 degree = north/up, 90 degree = east/right.
#
# "wind_flow_to":
#     Open-Meteo wind direction is meteorological direction FROM which
#     wind comes. This mode plots the direction air flows TO.
WIND_ARROW_MODE = "bearing_from_north"

WIND_ARROW_COLOR = "yellow"   # options: "yellow", "gold", "white", "cyan"
 
# Arrow length in degree units.
WIND_ARROW_LENGTH_DEG = 0.0045
WIND_ARROW_WIDTH = 0.004
WIND_ARROW_HEADWIDTH = 3.8
WIND_ARROW_HEADLENGTH = 4.8


# ============================================================
# LABELS / UNITS
# ============================================================

VARIABLE_LABELS = {
    "temperature_2m": "Temperature at 2 m",
    "relative_humidity_2m": "Relative humidity at 2 m",
    "dew_point_2m": "Dew point at 2 m",
    "apparent_temperature": "Apparent temperature",

    "pressure_msl": "Mean sea-level pressure",
    "surface_pressure": "Surface pressure",

    "cloud_cover": "Total cloud cover",
    "cloud_cover_low": "Low cloud cover",
    "cloud_cover_mid": "Mid cloud cover",
    "cloud_cover_high": "High cloud cover",
    "visibility": "Visibility",

    "wind_speed_10m": "Wind speed at 10 m",
    "wind_speed_80m": "Wind speed at 80 m",
    "wind_speed_120m": "Wind speed at 120 m",
    "wind_speed_180m": "Wind speed at 180 m",

    "wind_direction_10m": "Wind direction at 10 m",
    "wind_direction_80m": "Wind direction at 80 m",
    "wind_direction_120m": "Wind direction at 120 m",
    "wind_direction_180m": "Wind direction at 180 m",

    "wind_gusts_10m": "Wind gusts at 10 m",

    "precipitation": "Precipitation",
    "rain": "Rain",
    "showers": "Showers",
    "precipitation_probability": "Precipitation probability",

    "weather_code": "Weather code",
    "cape": "CAPE",
    "boundary_layer_height": "Boundary layer height",

    "weather_risk_0_1": "Weather risk index",
    "weather_no_fly": "Weather no-fly mask",
}

VARIABLE_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "apparent_temperature": "°C",

    "pressure_msl": "hPa",
    "surface_pressure": "hPa",

    "cloud_cover": "%",
    "cloud_cover_low": "%",
    "cloud_cover_mid": "%",
    "cloud_cover_high": "%",
    "visibility": "m",

    "wind_speed_10m": "m/s",
    "wind_speed_80m": "m/s",
    "wind_speed_120m": "m/s",
    "wind_speed_180m": "m/s",

    "wind_direction_10m": "degree",
    "wind_direction_80m": "degree",
    "wind_direction_120m": "degree",
    "wind_direction_180m": "degree",

    "wind_gusts_10m": "m/s",

    "precipitation": "mm",
    "rain": "mm",
    "showers": "mm",
    "precipitation_probability": "%",

    "weather_code": "code",
    "cape": "J/kg",
    "boundary_layer_height": "m",

    "weather_risk_0_1": "0-1",
    "weather_no_fly": "0/1",
}


# ============================================================
# SAMPLING FUNCTIONS
# ============================================================

def build_sampling_points(
    polygon_lonlat: list[tuple[float, float]],
    step_deg: float,
) -> list[tuple[float, float]]:
    """
    Build regular lon/lat sampling points inside the Hoa Lac polygon.

    Returns:
        list of (lon, lat)
    """
    poly = Polygon(polygon_lonlat)
    min_lon, min_lat, max_lon, max_lat = poly.bounds

    points: list[tuple[float, float]] = []

    lat = min_lat
    while lat <= max_lat + 1e-12:
        lon = min_lon
        while lon <= max_lon + 1e-12:
            p = Point(lon, lat)
            if poly.contains(p) or poly.touches(p):
                points.append((round(lon, 6), round(lat, 6)))
            lon += step_deg
        lat += step_deg

    # Add centroid.
    c = poly.centroid
    centroid = (round(c.x, 6), round(c.y, 6))
    if centroid not in points:
        points.append(centroid)

    # Add polygon vertices.
    for lon, lat in polygon_lonlat[:-1]:
        v = (round(lon, 6), round(lat, 6))
        if v not in points:
            points.append(v)

    return points


def chunk_points(
    points: list[tuple[float, float]],
    chunk_size: int = 50,
) -> list[list[tuple[float, float]]]:
    """
    Split points to avoid too-long request URLs.
    """
    return [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]


# ============================================================
# TIME FUNCTIONS
# ============================================================

def parse_utc_time(text: str) -> datetime:
    """
    Parse UTC string:
        YYYY-MM-DDTHH:MM:SSZ
    """
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def get_request_utc() -> datetime:
    """
    Get request UTC time for labels, file names, and fixed request mode.
    """
    if TIME_MODE == "fixed_utc":
        return parse_utc_time(MANUAL_REQUEST_UTC)

    if TIME_MODE == "auto_forecast":
        now_utc = datetime.now(timezone.utc)

        if ROUND_AUTO_REQUEST_TO_PREVIOUS_HOUR:
            now_utc = now_utc.replace(minute=0, second=0, microsecond=0)

        return now_utc

    raise ValueError(
        f"Unknown TIME_MODE = {TIME_MODE}. "
        "Use 'auto_forecast' or 'fixed_utc'."
    )


def build_time_params_for_openmeteo(request_utc: datetime) -> dict:
    """
    Build time parameters for Open-Meteo.

    auto_forecast:
        uses forecast_hours.

    fixed_utc:
        uses start_hour/end_hour in UTC.
    """
    if TIME_MODE == "auto_forecast":
        return {
            "forecast_hours": FORECAST_HOURS,
        }

    if TIME_MODE == "fixed_utc":
        start_utc = request_utc.replace(minute=0, second=0, microsecond=0)
        end_utc = start_utc + timedelta(hours=FORECAST_HOURS - 1)

        return {
            "start_hour": start_utc.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end_utc.strftime("%Y-%m-%dT%H:%M"),
        }

    raise ValueError(f"Unknown TIME_MODE = {TIME_MODE}")


def time_string_to_utc(time_str: str) -> datetime:
    """
    Convert Open-Meteo GMT/UTC time string like:
        2026-06-16T08:00
    to UTC datetime.
    """
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
    return dt.replace(tzinfo=timezone.utc)


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def download_openmeteo_chunk(
    points: list[tuple[float, float]],
    request_utc: datetime,
) -> list[dict]:
    """
    Download one point chunk from Open-Meteo.
    """
    latitudes = ",".join(f"{lat:.6f}" for lon, lat in points)
    longitudes = ",".join(f"{lon:.6f}" for lon, lat in points)

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "land",
        "models": "best_match",
    }

    params.update(build_time_params_for_openmeteo(request_utc))

    r = requests.get(OPENMETEO_URL, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()

    # Multiple coordinates return list.
    # Single coordinate returns dict.
    if isinstance(data, dict):
        data = [data]

    return data


def openmeteo_to_dataframe(
    data_list: list[dict],
    requested_points: list[tuple[float, float]],
    request_utc: datetime,
) -> pd.DataFrame:
    """
    Convert Open-Meteo JSON to long table.
    """
    rows = []

    download_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    request_time_utc = request_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    for loc_idx, data in enumerate(data_list):
        req_lon, req_lat = requested_points[loc_idx]

        model_lat = data.get("latitude")
        model_lon = data.get("longitude")
        model_elevation_m = data.get("elevation")
        generationtime_ms = data.get("generationtime_ms")
        utc_offset_seconds = data.get("utc_offset_seconds")
        timezone_name = data.get("timezone")

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        for i, time_utc in enumerate(times):
            row = {
                "location_id": loc_idx,
                "requested_lon": req_lon,
                "requested_lat": req_lat,
                "model_lon": model_lon,
                "model_lat": model_lat,
                "model_elevation_m": model_elevation_m,
                "time_local": time_utc,
                "time_utc": time_utc,
                "timezone": timezone_name,
                "utc_offset_seconds": utc_offset_seconds,
                "generationtime_ms": generationtime_ms,
                "request_time_utc": request_time_utc,
                "download_time_utc": download_time_utc,
                "time_mode": TIME_MODE,
            }

            for var in HOURLY_VARIABLES:
                values = hourly.get(var)
                row[var] = values[i] if values is not None and i < len(values) else math.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# WEATHER RISK FUNCTIONS
# ============================================================

def add_weather_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple 0-1 weather risk index for UTM path planning.

    Tune thresholds for your UAV type.
    """
    out = df.copy()

    def norm_clip(series, low, high):
        x = pd.to_numeric(series, errors="coerce")
        return ((x - low) / (high - low)).clip(lower=0, upper=1)

    out["risk_wind_120m"] = norm_clip(out["wind_speed_120m"], 5.0, 15.0)
    out["risk_gust_10m"] = norm_clip(out["wind_gusts_10m"], 8.0, 20.0)
    out["risk_precipitation"] = norm_clip(out["precipitation"], 0.1, 5.0)
    out["risk_cape"] = norm_clip(out["cape"], 500.0, 2500.0)

    visibility = pd.to_numeric(out["visibility"], errors="coerce")
    out["risk_visibility"] = ((10000.0 - visibility) / (10000.0 - 1000.0)).clip(
        lower=0,
        upper=1,
    )

    risk_cols = [
        "risk_wind_120m",
        "risk_gust_10m",
        "risk_precipitation",
        "risk_visibility",
        "risk_cape",
    ]

    out["weather_risk_0_1"] = out[risk_cols].max(axis=1, skipna=True)

    out["weather_no_fly"] = (
        (pd.to_numeric(out["wind_speed_120m"], errors="coerce") >= 15.0)
        | (pd.to_numeric(out["wind_gusts_10m"], errors="coerce") >= 20.0)
        | (pd.to_numeric(out["visibility"], errors="coerce") <= 1000.0)
        | (pd.to_numeric(out["precipitation"], errors="coerce") >= 5.0)
        | (pd.to_numeric(out["cape"], errors="coerce") >= 2500.0)
    ).astype(int)

    return out


# ============================================================
# PLOT HELPERS
# ============================================================

def safe_filename(text: str) -> str:
    """
    Make safe file name.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def get_plot_value_range(values: np.ndarray) -> tuple[float, float]:
    """
    Get stable vmin/vmax.
    """
    finite = np.isfinite(values)

    if not finite.any():
        return 0.0, 1.0

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if np.isclose(vmin, vmax):
        if np.isclose(vmin, 0.0):
            vmax = 1.0
        else:
            delta = max(abs(vmin) * 0.05, 0.5)
            vmin -= delta
            vmax += delta

    return vmin, vmax

def smooth_grid_with_nan(
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Smooth a grid that may contain NaN values.

    This avoids NaN spreading during Gaussian filtering.
    """
    grid = grid.astype(float)

    valid = np.isfinite(grid)

    if not valid.any():
        return grid

    grid_filled = np.where(valid, grid, 0.0)
    weight = valid.astype(float)

    smooth_data = gaussian_filter(
        grid_filled,
        sigma=sigma,
        mode="nearest",
    )

    smooth_weight = gaussian_filter(
        weight,
        sigma=sigma,
        mode="nearest",
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = smooth_data / smooth_weight

    smooth[smooth_weight <= 0] = np.nan

    return smooth

def make_surface_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate point values to a smooth surface grid
    and mask outside Hoa Lac polygon.
    """
    poly = Polygon(HOALAC_POLYGON)
    min_lon, min_lat, max_lon, max_lat = poly.bounds

    grid_lon_1d = np.linspace(min_lon, max_lon, SURFACE_NX)
    grid_lat_1d = np.linspace(min_lat, max_lat, SURFACE_NY)

    grid_lon, grid_lat = np.meshgrid(grid_lon_1d, grid_lat_1d)

    finite = np.isfinite(values)
    pts = np.column_stack([lons[finite], lats[finite]])
    vals = values[finite]

    if len(vals) == 0:
        grid_value = np.full_like(grid_lon, np.nan, dtype=float)
        return grid_lon, grid_lat, grid_value

    data_vmin = float(np.nanmin(vals))
    data_vmax = float(np.nanmax(vals))

    # Categorical / direction variables should not be smoothed.
    use_nearest = variable in NEAREST_SURFACE_VARS or len(vals) < 4

    if use_nearest:
        grid_value = griddata(
            pts,
            vals,
            (grid_lon, grid_lat),
            method="nearest",
        )

    else:
        # Cubic gives smoother surfaces than linear.
        # If cubic fails, fallback to linear.
        try:
            grid_interp = griddata(
                pts,
                vals,
                (grid_lon, grid_lat),
                method=CONTINUOUS_INTERPOLATION_METHOD,
            )
        except Exception:
            grid_interp = griddata(
                pts,
                vals,
                (grid_lon, grid_lat),
                method="linear",
            )

        # Fill remaining edge gaps with nearest values.
        grid_nearest = griddata(
            pts,
            vals,
            (grid_lon, grid_lat),
            method="nearest",
        )

        grid_value = np.where(
            np.isfinite(grid_interp),
            grid_interp,
            grid_nearest,
        )

        # Smooth continuous surfaces only.
        if APPLY_SURFACE_SMOOTHING:
            grid_value = smooth_grid_with_nan(
                grid=grid_value,
                sigma=SURFACE_SMOOTH_SIGMA,
            )

        # Prevent smoothing from creating unrealistic min/max overshoot.
        if CLIP_SMOOTHED_SURFACE:
            grid_value = np.clip(
                grid_value,
                data_vmin,
                data_vmax,
            )

    # Mask outside Hoa Lac polygon.
    poly_path = MplPath(np.asarray(HOALAC_POLYGON))
    flat_grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    inside = poly_path.contains_points(flat_grid_points).reshape(grid_lon.shape)

    grid_value = np.where(inside, grid_value, np.nan)

    return grid_lon, grid_lat, grid_value


def is_wind_speed_variable(variable: str) -> bool:
    """
    Check wind speed variable.
    """
    return variable.startswith("wind_speed_")


def is_wind_direction_variable(variable: str) -> bool:
    """
    Check wind direction variable.
    """
    return variable.startswith("wind_direction_")


def get_matching_wind_direction_variable(speed_variable: str) -> str | None:
    """
    Convert wind_speed_120m -> wind_direction_120m.
    """
    if not is_wind_speed_variable(speed_variable):
        return None

    suffix = speed_variable.replace("wind_speed_", "")
    return f"wind_direction_{suffix}"


def wind_direction_to_uv(
    direction_deg: np.ndarray,
    arrow_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert wind direction degrees to x/y arrow components.

    WIND_ARROW_MODE = "bearing_from_north":
        0 deg -> north/up
        90 deg -> east/right

    WIND_ARROW_MODE = "wind_flow_to":
        reverse direction to show where the wind flows to.
    """
    theta = np.deg2rad(direction_deg)

    # Bearing clockwise from north
    u = np.sin(theta)
    v = np.cos(theta)

    if WIND_ARROW_MODE == "wind_flow_to":
        u = -u
        v = -v
    elif WIND_ARROW_MODE != "bearing_from_north":
        raise ValueError(
            f"Unknown WIND_ARROW_MODE = {WIND_ARROW_MODE}. "
            "Use 'bearing_from_north' or 'wind_flow_to'."
        )

    return u * arrow_length, v * arrow_length


def draw_common_map_format(ax, lons: np.ndarray, lats: np.ndarray):
    """
    Common map formatting.
    """
    poly_lons = [p[0] for p in HOALAC_POLYGON]
    poly_lats = [p[1] for p in HOALAC_POLYGON]

    ax.plot(
        poly_lons,
        poly_lats,
        "-k",
        lw=1.5,
        zorder=6,
        label="Hoa Lac polygon",
    )

    min_lon, min_lat, max_lon, max_lat = Polygon(HOALAC_POLYGON).bounds
    pad_lon = max((max_lon - min_lon) * 0.08, 0.003)
    pad_lat = max((max_lat - min_lat) * 0.08, 0.003)

    ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
    ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)

    mean_lat = np.nanmean(lats)
    ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


def add_title(
    ax,
    label: str,
    lead_hour: int,
    valid_utc: datetime,
    request_utc: datetime,
    extra_text: str = "",
):
    """
    Add standard title.
    """
    request_utc_str = request_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    valid_utc_str = valid_utc.strftime("%Y-%m-%d %H:%M UTC")

    if extra_text:
        first_line = f"Hoa Lac Open-Meteo Surface Map: {label} {extra_text}"
    else:
        first_line = f"Hoa Lac Open-Meteo Surface Map: {label}"

    ax.set_title(
        f"{first_line}\n"
        f"Request UTC: {request_utc_str} | "
        f"Forecast step: +{lead_hour}h | "
        f"Valid UTC: {valid_utc_str}",
        fontsize=10,
    )


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_one_standard_layer(
    df_one_time: pd.DataFrame,
    variable: str,
    lead_hour: int,
    valid_utc: datetime,
    request_utc: datetime,
    fig_dir: Path,
):
    """
    Plot one non-combined variable.
    """
    if variable not in df_one_time.columns:
        return

    values = pd.to_numeric(df_one_time[variable], errors="coerce").to_numpy()
    finite = np.isfinite(values)

    if not finite.any():
        print(f"[WARN] Skip plot for {variable}: all values are NaN")
        return

    lons = df_one_time["requested_lon"].to_numpy(dtype=float)
    lats = df_one_time["requested_lat"].to_numpy(dtype=float)

    label = VARIABLE_LABELS.get(variable, variable)
    unit = VARIABLE_UNITS.get(variable, "")

    vmin, vmax = get_plot_value_range(values)

    if variable == "weather_no_fly":
        vmin, vmax = 0.0, 1.0

    if is_wind_direction_variable(variable):
        vmin, vmax = 0.0, 360.0
        cmap_to_use = DIRECTION_CMAP
    else:
        cmap_to_use = SURFACE_CMAP

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    cbar_obj = None

    if PLOT_SURFACE:
        grid_lon, grid_lat, grid_value = make_surface_grid(
            lons=lons,
            lats=lats,
            values=values,
            variable=variable,
        )

        surface = ax.pcolormesh(
            grid_lon,
            grid_lat,
            grid_value,
            shading="auto",
            cmap=cmap_to_use,
            vmin=vmin,
            vmax=vmax,
        )
        cbar_obj = surface

    if PLOT_DATA_POINTS:
        point_plot = ax.scatter(
            lons,
            lats,
            c=values,
            s=POINT_SIZE,
            cmap=cmap_to_use,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=POINT_EDGE_WIDTH,
            marker="o",
            zorder=5,
            label="Open-Meteo sample points",
        )

        if cbar_obj is None:
            cbar_obj = point_plot

    draw_common_map_format(ax, lons, lats)
    add_title(
        ax=ax,
        lead_hour=lead_hour,
        valid_utc=valid_utc,
        request_utc=request_utc,
        label=label,
        )

    if cbar_obj is not None:
        cbar = fig.colorbar(cbar_obj, ax=ax, shrink=0.90)
        cbar.set_label(f"{label} [{unit}]" if unit else label)

    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    valid_tag = valid_utc.strftime("%Y%m%d_%H%MUTC")
    fname = f"lead_{lead_hour:02d}_{valid_tag}_{safe_filename(variable)}_surface.png"
    out_png = fig_dir / fname

    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved plot: {out_png}")


def plot_one_wind_combined_layer(
    df_one_time: pd.DataFrame,
    speed_var: str,
    direction_var: str,
    lead_hour: int,
    valid_utc: datetime,
    request_utc: datetime,
    fig_dir: Path,
):
    """
    Plot combined wind speed + wind direction.

    Surface background:
        wind speed

    Arrows:
        wind direction angle from north

    Colorbar:
        wind speed
    """
    if speed_var not in df_one_time.columns:
        print(f"[WARN] Skip combined wind map: missing {speed_var}")
        return

    if direction_var not in df_one_time.columns:
        print(f"[WARN] Skip combined wind map: missing {direction_var}")
        return

    speed_values = pd.to_numeric(df_one_time[speed_var], errors="coerce").to_numpy()
    direction_values = pd.to_numeric(df_one_time[direction_var], errors="coerce").to_numpy()

    finite_speed = np.isfinite(speed_values)
    finite_dir = np.isfinite(direction_values)

    if not finite_speed.any():
        print(f"[WARN] Skip combined wind map for {speed_var}: all speed values are NaN")
        return

    if not finite_dir.any():
        print(f"[WARN] Skip combined wind map for {direction_var}: all direction values are NaN")
        return

    lons = df_one_time["requested_lon"].to_numpy(dtype=float)
    lats = df_one_time["requested_lat"].to_numpy(dtype=float)

    speed_label = VARIABLE_LABELS.get(speed_var, speed_var)
    direction_label = VARIABLE_LABELS.get(direction_var, direction_var)
    speed_unit = VARIABLE_UNITS.get(speed_var, "")

    vmin, vmax = get_plot_value_range(speed_values)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    cbar_obj = None

    # --------------------------------------------------------
    # Wind speed surface background
    # --------------------------------------------------------
    if PLOT_SURFACE:
        grid_lon, grid_lat, grid_speed = make_surface_grid(
            lons=lons,
            lats=lats,
            values=speed_values,
            variable=speed_var,
        )

        surface = ax.pcolormesh(
            grid_lon,
            grid_lat,
            grid_speed,
            shading="auto",
            cmap=WIND_SPEED_CMAP,
            vmin=vmin,
            vmax=vmax,
        )

        cbar_obj = surface

    # --------------------------------------------------------
    # Data points as circles
    # --------------------------------------------------------
    if PLOT_DATA_POINTS:
        ax.scatter(
            lons[finite_speed],
            lats[finite_speed],
            s=POINT_SIZE,
            facecolors="none",
            edgecolors="black",
            linewidths=POINT_EDGE_WIDTH,
            marker="o",
            zorder=7,
            label="Open-Meteo sample points",
        )

    # --------------------------------------------------------
    # Direction arrows
    # --------------------------------------------------------
    if PLOT_WIND_DIRECTION_ARROWS:
        finite = finite_speed & finite_dir

        u, v = wind_direction_to_uv(
            direction_deg=direction_values[finite],
            arrow_length=WIND_ARROW_LENGTH_DEG,
        )

        ax.quiver(
            lons[finite],
            lats[finite],
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=WIND_ARROW_WIDTH,
            headwidth=WIND_ARROW_HEADWIDTH,
            headlength=WIND_ARROW_HEADLENGTH,
            color=WIND_ARROW_COLOR,
            zorder=8,
            label="Direction from north",
        )

    draw_common_map_format(ax, lons, lats)

    combined_label = f"{speed_label} + {direction_label}"
    add_title(
        ax=ax,
        label=combined_label,
        lead_hour=lead_hour,
        valid_utc=valid_utc,
        request_utc=request_utc,
        extra_text="",
    )

    # Colorbar is for wind speed surface background
    if cbar_obj is not None:
        cbar = fig.colorbar(cbar_obj, ax=ax, shrink=0.90)
        cbar.set_label(f"{speed_label} [{speed_unit}]" if speed_unit else speed_label)

    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    valid_tag = valid_utc.strftime("%Y%m%d_%H%MUTC")
    fname = (
        f"lead_{lead_hour:02d}_{valid_tag}_"
        f"{safe_filename(speed_var)}_plus_{safe_filename(direction_var)}_surface.png"
    )
    out_png = fig_dir / fname

    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved combined wind plot: {out_png}")


def get_variables_to_plot(df: pd.DataFrame) -> list[str]:
    """
    Select variables for plotting.

    Wind direction variables are not plotted alone when matching wind speed
    exists. They are combined with wind speed maps.
    """
    if PLOT_FULL_DATA:
        plot_vars = list(HOURLY_VARIABLES)
    else:
        plot_vars = list(SELECTED_PLOT_VARIABLES)

    if PLOT_DERIVED_FIELDS:
        plot_vars += list(PLOT_EXTRA_FIELDS)

    # Remove wind_direction_* from direct plot list if speed pair exists.
    # Direction will be shown by arrows on wind_speed_* maps.
    filtered = []

    for var in plot_vars:
        if is_wind_direction_variable(var):
            suffix = var.replace("wind_direction_", "")
            speed_var = f"wind_speed_{suffix}"

            if speed_var in df.columns:
                continue

        if var in df.columns:
            filtered.append(var)
        else:
            print(f"[WARN] Requested plot variable missing: {var}")

    # Remove duplicates while preserving order.
    unique = []
    for var in filtered:
        if var not in unique:
            unique.append(var)

    return unique


def plot_all_layers(df: pd.DataFrame):
    """
    Plot selected variables for each forecast hour.
    """
    if df.empty:
        print("[WARN] DataFrame is empty. No plots created.")
        return

    request_utc = pd.to_datetime(
        df["request_time_utc"].iloc[0],
        utc=True,
    ).to_pydatetime()

    request_tag = request_utc.strftime("%Y%m%d_%H%M%SUTC")

    fig_dir = OUT_DIR / "figures" / request_tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_vars = get_variables_to_plot(df)

    unique_times = list(pd.unique(df["time_utc"]))

    print("\n========== PLOT WEATHER SURFACE MAPS ==========")
    print(f"[INFO] Figure directory: {fig_dir}")
    print(f"[INFO] PLOT_FULL_DATA: {PLOT_FULL_DATA}")
    print(f"[INFO] Number of forecast times: {len(unique_times)}")
    print(f"[INFO] Number of plot variables: {len(plot_vars)}")
    print("[INFO] Variables to plot:")

    for var in plot_vars:
        if is_wind_speed_variable(var):
            direction_var = get_matching_wind_direction_variable(var)
            if direction_var in df.columns:
                print(f"       - {var} + {direction_var}")
            else:
                print(f"       - {var}")
        else:
            print(f"       - {var}")

    for lead_hour, t_utc in enumerate(unique_times, start=1):
        df_one_time = df[df["time_utc"] == t_utc].copy()

        if df_one_time.empty:
            continue

        valid_utc = time_string_to_utc(str(t_utc))

        print("\n----------------------------------------")
        print(f"[INFO] Forecast step: +{lead_hour}h")
        print(f"[INFO] Valid UTC:      {valid_utc.strftime('%Y-%m-%d %H:%M UTC')}")

        for var in plot_vars:
            # Combined wind-speed + wind-direction plot
            if is_wind_speed_variable(var):
                direction_var = get_matching_wind_direction_variable(var)

                if direction_var is not None and direction_var in df_one_time.columns:
                    plot_one_wind_combined_layer(
                        df_one_time=df_one_time,
                        speed_var=var,
                        direction_var=direction_var,
                        lead_hour=lead_hour,
                        valid_utc=valid_utc,
                        request_utc=request_utc,
                        fig_dir=fig_dir,
                    )
                    continue

            # Standard plot for all other variables
            plot_one_standard_layer(
                df_one_time=df_one_time,
                variable=var,
                lead_hour=lead_hour,
                valid_utc=valid_utc,
                request_utc=request_utc,
                fig_dir=fig_dir,
            )


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    request_utc = get_request_utc()

    print("\n========== DOWNLOAD OPEN-METEO FORECAST ==========")
    print(f"[INFO] Time mode:       {TIME_MODE}")
    print(f"[INFO] Request UTC:     {request_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"[INFO] Forecast hours:  {FORECAST_HOURS}")
    print(f"[INFO] Output dir:      {OUT_DIR}")

    points = build_sampling_points(
        polygon_lonlat=HOALAC_POLYGON,
        step_deg=GRID_STEP_DEG,
    )

    print(f"[INFO] Sampling points: {len(points)}")

    all_frames = []

    for chunk_id, pts in enumerate(chunk_points(points), start=1):
        print(f"[INFO] Download chunk {chunk_id}: {len(pts)} points")

        data = download_openmeteo_chunk(
            points=pts,
            request_utc=request_utc,
        )

        df_chunk = openmeteo_to_dataframe(
            data_list=data,
            requested_points=pts,
            request_utc=request_utc,
        )

        all_frames.append(df_chunk)

    if not all_frames:
        raise RuntimeError("No Open-Meteo data downloaded.")

    df = pd.concat(all_frames, ignore_index=True)
    df = add_weather_risk_index(df)

    timestamp = request_utc.strftime("%Y%m%d_%H%M%SUTC")

    latest_csv = OUT_DIR / f"openmeteo_hoalac_next{FORECAST_HOURS}h_latest.csv"
    archive_csv = OUT_DIR / f"openmeteo_hoalac_next{FORECAST_HOURS}h_{timestamp}.csv"

    df.to_csv(latest_csv, index=False)
    df.to_csv(archive_csv, index=False)

    print(f"[OK] Saved latest:  {latest_csv}")
    print(f"[OK] Saved archive: {archive_csv}")

    print("\n========== FIRST ROWS ==========")

    show_cols = [
        "requested_lon",
        "requested_lat",
        "time_utc",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_speed_80m",
        "wind_direction_80m",
        "wind_speed_120m",
        "wind_direction_120m",
        "wind_speed_180m",
        "wind_direction_180m",
        "wind_gusts_10m",
        "precipitation",
        "relative_humidity_2m",
        "weather_risk_0_1",
        "weather_no_fly",
    ]

    existing_show_cols = [c for c in show_cols if c in df.columns]
    print(df[existing_show_cols].head(20))

    plot_all_layers(df)


if __name__ == "__main__":
    main()