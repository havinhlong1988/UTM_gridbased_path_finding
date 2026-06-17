#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create UAV cost map / risk map for scenario 1.

HEADER NOTES - UPDATED RISK LOGIC
=================================

This script follows the new LAE-UTM costmap idea:

    R_total = R_obstacle + (1 - R_obstacle) * R_soft

Where:

    R_obstacle = obstacle risk from static spatial constraints
                 currently building/population density + restricted airspace (RA)

    R_soft     = dynamic/support risk
                 currently emergency-absence risk + optional weather risk

    R_emergency = 1 - emergency_support

Important interpretation:

    1. Emergency is NOT an obstacle.
       Emergency availability is a safety/resilience layer.
       Good emergency support near DB / DK / FLZ reduces the soft-risk term.

    2. Lack of emergency support increases soft risk.
       Therefore:

           emergency_support = high near DB / DK / FLZ
           emergency_risk    = 1 - emergency_support

    3. Emergency does not cancel hard no-fly cells.
       Hard no-fly cells are preserved by a hard no-fly mask:

           is_hard_nofly = original no-fly nodes OR RA hard area
           no-fly cost   = 10.0
           flyable base slowness = 0.02
           pathfinder should reject cost >= 10

       RA hard area means:

           a) RA-labeled nodes plus RA_BUFFER_M around them, and/or
           b) the inside of each RA_CIRCLE radius_m.

       Inside this hard RA area, emergency_support is forced to 0.

    4. Weather is optional because weather data are not available yet.
       By default:

           INCLUDE_WEATHER_RISK = False
           W_SOFT_WEATHER      = 0.0

       When weather data are available, set INCLUDE_WEATHER_RISK = True,
       set W_SOFT_WEATHER > 0, and provide WEATHER_RISK_FILE.

    5. The default soft-risk layer therefore uses only emergency absence:

           R_soft = R_emergency

       If weather is enabled:

           R_soft = weighted_mean(R_emergency, R_weather)

    6. The final pathfinding impedance is:

           cost_per_m = W_TIME * slowness + W_RISK * R_total

       The old independent emergency term was removed because emergency is now
       already included inside R_soft.

Obstacle-risk components
------------------------

    density_norm:
        Robust 0-1 normalized building/population density.

    ra_risk:
        RA influence from either:
            a) RA-labeled nodes in the model file, and/or
            b) optional RA circles defined by center lon/lat and radius_m.

    is_ra_hard_nofly:
        True inside RA-labeled nodes + RA_BUFFER_M, or inside RA_CIRCLE radius_m.
        These nodes are forced to no-fly cost = 10 in both final cost and
        pathfinding cost outputs.

    R_obstacle:
        weighted density + RA risk.
        Typical density weight can be 0.6 or 0.7 depending on your definition.

Emergency-support components
----------------------------

    DB  = Drone Base, strongest emergency support
    DK  = Docking station, medium/high support
    FLZ = Forced Landing Zone, emergency landing support

    emergency_support = max(DB_support, DK_support, FLZ_support)
    emergency_risk    = 1 - emergency_support

Weather-risk placeholder
------------------------

    If INCLUDE_WEATHER_RISK = False:
        weather_risk = 0 everywhere.

    If INCLUDE_WEATHER_RISK = True:
        the script tries to read WEATHER_RISK_FILE.

    Preferred weather file format:
        lon lat weather_risk

    Optional weather file format:
        lon lat wind_speed_mps rain_mm_h visibility_m

    All weather values are converted to a 0-1 risk score.

Plot-style notes
----------------

    All map layers use one shared matplotlib colormap controlled by:

        MAP_CMAP = "seismic"

    Colorbars are controlled by:

        COLORBAR_SHRINK    -> shorter/longer colorbar
        COLORBAR_FRACTION  -> thinner/wider colorbar
        COLORBAR_ASPECT    -> visual slenderness
        COLORBAR_PAD       -> gap between map and colorbar

    Fancy font support is controlled by:

        USE_FANCY_FONT = True

    The script selects the first installed font from FANCY_FONT_CANDIDATES.
    If none are found, matplotlib falls back safely to its default font.

Outputs
-------

    output/02_senario1_no_velocity/05_cost_map/
        cost_map_nodes.csv
        cost_map_nodes.xyz
        model_senario1_cost_for_pathfinding.xyz
        cost_map_summary.csv

    figures/02_senario1_no_velocity/05_cost_map/
        obstacle_risk_map.png
        emergency_support_map.png
        emergency_risk_map.png
        weather_risk_map.png
        soft_risk_map.png
        total_risk_map.png
        final_cost_map.png
            Two-panel figure: final slowness map (s/m) + derived velocity map (m/s).
        pathfinding_cost_map.png

Plot note:
    The figure renderer now supports either a smooth map surface
    (recommended, easier on the eyes) or the original point-scatter view.
    Set PLOT_SMOOTH_MAP = True or False below.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from scipy.spatial import cKDTree

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False


# ============================================================
# User settings
# ============================================================

PROJECT_DIR = Path(".").resolve()

MODEL_FILE = (
    PROJECT_DIR
    / "output/02_senario1_no_velocity/04_2D_model_senario_1"
    / "mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz"
)

# Fallback if you already copied the model to algorithm_test_2D style.
FALLBACK_MODEL_FILE = (
    PROJECT_DIR
    / "input/model/senario1/model_senario1_with_label.xyz"
)

OUT_DIR = PROJECT_DIR / "output/02_senario1_no_velocity/05_cost_map"
FIG_DIR = PROJECT_DIR / "figures/02_senario1_no_velocity/05_cost_map"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Operation area polygon: Hoa Lac
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

PLOT_OPERATION_AREA = True
OPERATION_AREA_LINEWIDTH = 2.0

# -------------------------------
# Final pathfinding cost weights
# -------------------------------
# Units: seconds per meter equivalent.
# Keep W_RISK small because slowness already has physical meaning.
# Final cost:
#     cost_per_m = W_TIME * slowness + W_RISK * risk_total
W_TIME = 1.0
W_RISK = 0.10

# -------------------------------
# Obstacle-risk internal weights
# -------------------------------
# R_obstacle = weighted density/building/population + RA influence.
# Set RISK_WEIGHT_DENSITY to 0.60 or 0.70 depending on your definition.
RISK_WEIGHT_DENSITY = 0.60
RISK_WEIGHT_RA = 0.40
RISK_WEIGHT_NOFLY_SOFT = 0.00

# RA influence from RA-labeled nodes.
# Inside the buffer: RA risk = 1.
# Outside the buffer: risk decays smoothly with RA_DECAY_M.
RA_BUFFER_M = 300.0
RA_DECAY_M = 150.0

# Optional RA circles defined directly by center and radius.
# This is useful when RA is known as a circle rather than by RA-labeled nodes.
# Example:
# RA_CIRCLES = [
#     {"name": "RA01", "lon": 105.5400, "lat": 21.0000, "radius_m": 300.0},
# ]
RA_CIRCLES = []
RA_CIRCLE_DECAY_M = 150.0

# -------------------------------
# Soft-risk internal weights
# -------------------------------
# R_soft = weighted emergency-absence risk + optional weather risk.
# Weather is disabled by default because the current model has no weather data.
W_SOFT_EMERGENCY = 1.0
INCLUDE_WEATHER_RISK = False
W_SOFT_WEATHER = 0.0

# Optional weather-risk file.
# Preferred columns: lon lat weather_risk
# Optional columns:   lon lat wind_speed_mps rain_mm_h visibility_m
WEATHER_RISK_FILE = OUT_DIR / "weather_risk_nodes.csv"
WEATHER_NEAREST_MAX_DISTANCE_M = 2000.0
WEATHER_WIND_SPEED_BAD_MPS = 12.0
WEATHER_RAIN_BAD_MM_H = 10.0
WEATHER_VISIBILITY_GOOD_M = 5000.0
WEATHER_WEIGHT_WIND = 0.60
WEATHER_WEIGHT_RAIN = 0.30
WEATHER_WEIGHT_VISIBILITY = 0.10

# -------------------------------
# Emergency support settings
# -------------------------------
# Emergency support is high near DB/DK/FLZ and low far away.
# Emergency risk = 1 - emergency_support.
EMERGENCY_RADIUS_M = 1000.0
EMERGENCY_WEIGHT_DB = 1.0
EMERGENCY_WEIGHT_DK = 0.8
EMERGENCY_WEIGHT_FLZ = 0.6

# Current project convention.
FLYABLE_SLOWNESS = 0.02
NOFLY_SLOWNESS = 10.0

# For final model used by pathfinding:
# no-fly nodes are set to exactly 10.
# Your pathfinder should reject nodes >= 10.
NOFLY_BLOCK_THRESHOLD = 9.999

USE_HARD_NOFLY_OUTPUT = True

# Projection for Hoa Lac, Hanoi: UTM zone 48N.
LOCAL_PROJECTED_CRS = "EPSG:32648"

DPI = 300

# -------------------------------
# Plot style settings
# -------------------------------
# Shared colormap for every plotted layer.
# User requested seismic. Use "seismic_r" if you want the colors reversed.
MAP_CMAP = "seismic"

# Compact colorbar settings.
# Smaller COLORBAR_SHRINK makes the colorbar shorter.
# Smaller COLORBAR_FRACTION makes the colorbar thinner.
COLORBAR_SHRINK = 0.62
COLORBAR_FRACTION = 0.028
COLORBAR_PAD = 0.025
COLORBAR_ASPECT = 35
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 8

# Smooth map plotting options.
# If True, plot a smooth surface using triangulated contour fill.
# If False, fall back to point scatter.
PLOT_SMOOTH_MAP = True
SMOOTH_LEVELS = 200
SMOOTH_OVERLAY_POINTS = False
SMOOTH_POINT_SIZE = 2

# Important smooth-plot rule:
# Hard no-fly cells are discontinuous classes, not smooth continuous values.
# If we smooth/interpolate them together with flyable low-risk nodes, matplotlib creates
# artificial white transition zones because "seismic" is white near 0.5.
# Therefore, for risk/cost maps, hard no-fly cells are removed from the smooth background
# and drawn again as a solid red overlay.
PRESERVE_HARD_NOFLY_RED_ON_SMOOTH = True
# Use a slightly gentler red for hard no-fly overlays so the map is easier on the eyes.
# 1.0 would be the strongest end of the seismic colormap. A slightly lower value keeps
# the visual meaning of "very high risk / no-fly" but looks less harsh.
HARD_NOFLY_OVERLAY_COLOR_VALUE = 0.90
HARD_NOFLY_OVERLAY_ALPHA = 0.88
HARD_NOFLY_OVERLAY_SCATTER_SIZE = 7
HARD_NOFLY_OVERLAY_COLUMNS = {
    "risk_obstacle",
    "emergency_risk",
    "risk_soft",
    "risk_total",
    "risk_norm",
    "cost_per_m",
    "cost_for_pathfinding",
}

# Risk/support layers are dimensionless scores and should always plot from 0 to 1.
# This prevents matplotlib from auto-expanding an all-zero layer, for example
# disabled weather risk, into a misleading negative colorbar range.
FORCE_ZERO_ONE_COLORBAR_FOR_RISK_LAYERS = True
ZERO_ONE_PLOT_COLUMNS = {
    "density_norm",
    "building_population_risk",
    "ra_risk",
    "risk_obstacle",
    "emergency_support",
    "emergency_risk",
    "emergency_penalty",
    "weather_risk",
    "risk_soft",
    "risk_total",
    "risk_norm",
}
ZERO_ONE_COLORBAR_TICKS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Notes:
# - Hard no-fly overlay is also preserved for emergency_risk and risk_soft maps, because
#   inside RA/original no-fly cells we explicitly define emergency_risk = 1 and risk_soft = 1.
# - This avoids artificial white halos in those maps as well.

# Cost/slowness layers should show the full no-fly value on the colorbar.
COST_PLOT_COLUMNS = {
    "cost_per_m",
    "cost_for_pathfinding",
    "final_slowness_s_per_m",
}
COST_COLORBAR_TICKS = [0.0, 2.5, 5.0, 7.5, NOFLY_SLOWNESS]

# Final cost-map visualization.
# The pathfinding model uses an effective slowness/cost value in s/m.
# This two-panel figure plots:
#   1) final slowness in s/m, and
#   2) equivalent velocity in m/s, computed as velocity = 1 / slowness.
# Hard no-fly cells are forced to velocity = 0 for plotting.
PLOT_FINAL_COST_WITH_VELOCITY_PANEL = True
FINAL_SLOWNESS_SOURCE_COLUMN = "cost_for_pathfinding"  # use "slowness" for base slowness only
FINAL_SLOWNESS_COLUMN = "final_slowness_s_per_m"
FINAL_VELOCITY_COLUMN = "final_velocity_mps"
FINAL_VELOCITY_KMH_COLUMN = "final_velocity_kmh"
FINAL_VELOCITY_CMAP = "seismic_r"
FINAL_VELOCITY_HARD_NOFLY_VALUE = 0.0
FINAL_SLOWNESS_COLORBAR_MAX = None  # None = percentile from flyable nodes
FINAL_VELOCITY_COLORBAR_MAX = None  # None = percentile from flyable nodes
FINAL_COLORBAR_PERCENTILE = 99.0
FINAL_FIGSIZE = (8.0, 12.0)
FINAL_PANEL_TITLE_SIZE = 13

# Hard no-fly overlay style.
# Scatter avoids the large artificial red rectangles that can happen when a binary
# no-fly mask is triangulated over a sparse/irregular grid.
HARD_NOFLY_OVERLAY_STYLE = "scatter"  # "scatter" or "contour"
HARD_NOFLY_OVERLAY_MARKER = "s"

# Fancy font settings for title, axis labels, ticks, legend, and colorbar.
# The script will use the first installed font from this list.
USE_FANCY_FONT = True
FANCY_FONT_CANDIDATES = [
    "Times New Roman",
    "DejaVu Serif",
    "Liberation Serif",
    "STIXGeneral",
    "Georgia",
]
TITLE_FONT_SIZE = 14
TITLE_FONT_WEIGHT = "bold"
AXIS_LABEL_FONT_SIZE = 11
AXIS_LABEL_FONT_WEIGHT = "bold"
TICK_LABEL_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
LEGEND_TITLE_FONT_SIZE = 9

# Legend position.
# Put the legend at the top of the map instead of the lower/best position.
# For a horizontal top legend, keep ncol large enough for DB/FLZ/DK/RA/Operation area.
LEGEND_LOC = "upper center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.985)
LEGEND_NCOL = 5
# Make legend fully visible above all other map layers.
LEGEND_FRAME_ALPHA = 1.0
LEGEND_FACE_COLOR = "white"
LEGEND_EDGE_COLOR = "black"
LEGEND_ZORDER = 1000


# ============================================================
# Helpers
# ============================================================

def choose_plot_font() -> str | None:
    """
    Select the first installed fancy font from FANCY_FONT_CANDIDATES.
    Returns None if no requested font is available.
    """
    if not USE_FANCY_FONT:
        return None

    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in FANCY_FONT_CANDIDATES:
        if font_name in installed_fonts:
            return font_name

    print(
        "[WARN] None of the requested fancy fonts were found. "
        "Matplotlib default font will be used."
    )
    return None


def apply_plot_style() -> str | None:
    """
    Apply global matplotlib font settings and return the selected font name.
    """
    selected_font = choose_plot_font()

    if selected_font is not None:
        plt.rcParams["font.family"] = selected_font
        plt.rcParams["mathtext.fontset"] = "stix"
        print(f"[OK] Plot font selected: {selected_font}")

    plt.rcParams["axes.titleweight"] = TITLE_FONT_WEIGHT
    plt.rcParams["axes.labelweight"] = AXIS_LABEL_FONT_WEIGHT
    plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE
    return selected_font


def font_kwargs(font_name: str | None, size: int | None = None, weight: str | None = None) -> dict:
    """Build safe matplotlib font keyword arguments."""
    kwargs = {}
    if font_name is not None:
        kwargs["fontname"] = font_name
    if size is not None:
        kwargs["fontsize"] = size
    if weight is not None:
        kwargs["fontweight"] = weight
    return kwargs


def choose_model_file() -> Path:
    if MODEL_FILE.exists():
        return MODEL_FILE
    if FALLBACK_MODEL_FILE.exists():
        print(f"[WARN] Main model not found. Using fallback:\n  {FALLBACK_MODEL_FILE}")
        return FALLBACK_MODEL_FILE
    raise FileNotFoundError(
        "Cannot find model file.\n"
        f"Checked:\n  {MODEL_FILE}\n  {FALLBACK_MODEL_FILE}"
    )


def read_model_xyz(path: Path) -> pd.DataFrame:
    """
    Flexible reader for files like:
        lon lat elevation slowness
        lon lat elevation slowness category
        lon lat elevation slowness category density
        lon lat elevation slowness category density label

    Extra columns after label are joined into label_text.
    """
    print(f"[INFO] Reading model: {path}")

    raw = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        dtype=str,
        engine="python",
    )

    if raw.shape[1] < 4:
        raise ValueError("Model file must have at least 4 columns: lon lat elevation slowness")

    df = pd.DataFrame()
    df["lon"] = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    df["lat"] = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    df["elevation_m"] = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
    df["slowness"] = pd.to_numeric(raw.iloc[:, 3], errors="coerce")

    if raw.shape[1] >= 5:
        df["category"] = raw.iloc[:, 4].fillna("").astype(str)
    else:
        df["category"] = ""

    if raw.shape[1] >= 6:
        density_candidate = pd.to_numeric(raw.iloc[:, 5], errors="coerce")
        # If the 6th column is mostly non-numeric, do not force it as density.
        if density_candidate.notna().mean() > 0.5:
            df["density"] = density_candidate
        else:
            df["density"] = np.nan
            df["category"] = (
                df["category"].astype(str) + "_" + raw.iloc[:, 5].fillna("").astype(str)
            )
    else:
        df["density"] = np.nan

    if raw.shape[1] >= 7:
        label_parts = []
        for j in range(6, raw.shape[1]):
            label_parts.append(raw.iloc[:, j].fillna("").astype(str))
        df["label"] = pd.concat(label_parts, axis=1).agg("_".join, axis=1)
    else:
        df["label"] = ""

    before = len(df)
    df = df.dropna(subset=["lon", "lat", "elevation_m", "slowness"]).reset_index(drop=True)
    after = len(df)

    if after < before:
        print(f"[WARN] Dropped {before - after} rows with invalid lon/lat/elevation/slowness.")

    df["class_text"] = (
        df["category"].fillna("").astype(str)
        + " "
        + df["label"].fillna("").astype(str)
    ).str.upper()

    print(f"[OK] Loaded nodes: {len(df):,}")
    return df


def lonlat_to_projected_xy(lon, lat, lon0=None, lat0=None):
    """
    Convert lon/lat arrays to local projected x/y in meters.
    Uses EPSG:32648 for Hoa Lac. Falls back to local equirectangular approximation.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if HAS_PYPROJ:
        transformer = Transformer.from_crs("EPSG:4326", LOCAL_PROJECTED_CRS, always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if lon0 is None:
        lon0 = np.nanmean(lon)
    if lat0 is None:
        lat0 = np.nanmean(lat)

    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    meters_per_deg_lat = 110_540.0
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y


def add_projected_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add x_m, y_m in local projected CRS.
    """
    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)

    if not HAS_PYPROJ:
        warnings.warn(
            "pyproj not found. Using local approximate meter conversion. "
            "Install pyproj for better accuracy."
        )

    x, y = lonlat_to_projected_xy(lon, lat)
    df["x_m"] = x
    df["y_m"] = y

    if HAS_PYPROJ:
        print(f"[OK] Projected lon/lat to {LOCAL_PROJECTED_CRS}")
    else:
        print("[WARN] Projected lon/lat using approximate local meter conversion.")

    return df

def robust_norm(values, q_low=1.0, q_high=99.0) -> np.ndarray:
    """
    Robust 0-1 normalization using percentiles.
    NaNs become 0.
    """
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)

    finite = np.isfinite(v)
    if finite.sum() == 0:
        return out

    lo = np.nanpercentile(v[finite], q_low)
    hi = np.nanpercentile(v[finite], q_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out

    out[finite] = (v[finite] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out



def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name, case-insensitive."""
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def safe_weighted_mean(layers: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted mean for risk layers in 0-1 range.
    Layers with weight <= 0 are ignored.
    """
    if not layers:
        return np.array([], dtype=float)

    n = len(layers[0])
    out = np.zeros(n, dtype=float)
    total_w = 0.0

    for layer, weight in zip(layers, weights):
        if weight <= 0:
            continue
        arr = np.asarray(layer, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        out += weight * np.clip(arr, 0.0, 1.0)
        total_w += weight

    if total_w <= 0:
        return np.zeros(n, dtype=float)

    return np.clip(out / total_w, 0.0, 1.0)


def read_weather_table(path: Path) -> pd.DataFrame:
    """
    Read a weather table with automatic separator detection.
    Accepts CSV, whitespace, or tab-separated text.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python", comment="#")


def compute_weather_risk_from_table(weather_df: pd.DataFrame) -> np.ndarray:
    """
    Convert weather variables to one 0-1 weather risk vector.

    Preferred direct column:
        weather_risk

    Otherwise the script tries to combine:
        wind_speed_mps, rain_mm_h, visibility_m
    """
    # Normalize column names only for matching; keep original values accessible.
    weather_df = weather_df.copy()
    weather_df.columns = [str(c).strip() for c in weather_df.columns]

    direct_col = first_existing_column(
        weather_df,
        ["weather_risk", "risk_weather", "r_weather", "weather_norm"],
    )

    if direct_col is not None:
        raw = pd.to_numeric(weather_df[direct_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(raw)
        if finite.sum() == 0:
            return np.zeros(len(weather_df), dtype=float)

        # If already 0-1, only clip. Otherwise robust-normalize.
        if np.nanmin(raw[finite]) >= 0.0 and np.nanmax(raw[finite]) <= 1.0:
            return np.clip(np.nan_to_num(raw, nan=0.0), 0.0, 1.0)
        return robust_norm(raw, 1, 99)

    layers = []
    weights = []

    wind_col = first_existing_column(
        weather_df,
        ["wind_speed_mps", "wind_speed", "windspeed", "wind_speed_10m", "wind_speed_ms"],
    )
    if wind_col is not None:
        wind = pd.to_numeric(weather_df[wind_col], errors="coerce").to_numpy(float)
        wind_risk = np.clip(wind / WEATHER_WIND_SPEED_BAD_MPS, 0.0, 1.0)
        layers.append(wind_risk)
        weights.append(WEATHER_WEIGHT_WIND)

    rain_col = first_existing_column(
        weather_df,
        ["rain_mm_h", "precip_mm_h", "precipitation_mm_h", "rain", "precipitation"],
    )
    if rain_col is not None:
        rain = pd.to_numeric(weather_df[rain_col], errors="coerce").to_numpy(float)
        rain_risk = np.clip(rain / WEATHER_RAIN_BAD_MM_H, 0.0, 1.0)
        layers.append(rain_risk)
        weights.append(WEATHER_WEIGHT_RAIN)

    visibility_col = first_existing_column(
        weather_df,
        ["visibility_m", "visibility", "vis_m"],
    )
    if visibility_col is not None:
        vis = pd.to_numeric(weather_df[visibility_col], errors="coerce").to_numpy(float)
        # Good visibility gives low risk; poor visibility gives high risk.
        vis_risk = 1.0 - np.clip(vis / WEATHER_VISIBILITY_GOOD_M, 0.0, 1.0)
        layers.append(vis_risk)
        weights.append(WEATHER_WEIGHT_VISIBILITY)

    if not layers:
        print(
            "[WARN] Weather file found, but no recognizable weather-risk columns were found. "
            "weather_risk is set to 0."
        )
        return np.zeros(len(weather_df), dtype=float)

    return safe_weighted_mean(layers, weights)


def load_weather_risk_on_model_nodes(df: pd.DataFrame) -> np.ndarray:
    """
    Load optional weather risk and interpolate/assign it to model nodes.

    Default behavior:
        INCLUDE_WEATHER_RISK = False -> return zeros.

    If enabled, the weather file should preferably have:
        lon lat weather_risk

    If the weather file has the same number of rows as the model and no coordinates,
    the risk is assigned row-by-row.
    """
    if not INCLUDE_WEATHER_RISK or W_SOFT_WEATHER <= 0:
        print("[INFO] Weather risk disabled. weather_risk is set to 0.")
        return np.zeros(len(df), dtype=float)

    if WEATHER_RISK_FILE is None or not Path(WEATHER_RISK_FILE).exists():
        print(f"[WARN] Weather risk enabled but file not found: {WEATHER_RISK_FILE}")
        print("[WARN] weather_risk is set to 0 for this run.")
        return np.zeros(len(df), dtype=float)

    weather_df = read_weather_table(Path(WEATHER_RISK_FILE))
    weather_risk_raw = compute_weather_risk_from_table(weather_df)

    lon_col = first_existing_column(weather_df, ["lon", "longitude", "x_lon"])
    lat_col = first_existing_column(weather_df, ["lat", "latitude", "y_lat"])
    x_col = first_existing_column(weather_df, ["x_m", "x", "easting"])
    y_col = first_existing_column(weather_df, ["y_m", "y", "northing"])

    # Case 1: same length and no coordinate columns -> direct row-wise assignment.
    if lon_col is None and lat_col is None and x_col is None and y_col is None:
        if len(weather_df) == len(df):
            print("[OK] Weather risk assigned row-by-row because weather file length matches model nodes.")
            return np.clip(weather_risk_raw, 0.0, 1.0)
        print("[WARN] Weather file has no coordinates and length does not match model nodes.")
        print("[WARN] weather_risk is set to 0 for this run.")
        return np.zeros(len(df), dtype=float)

    # Case 2: lon/lat weather grid -> project to local meters.
    if lon_col is not None and lat_col is not None:
        wlon = pd.to_numeric(weather_df[lon_col], errors="coerce").to_numpy(float)
        wlat = pd.to_numeric(weather_df[lat_col], errors="coerce").to_numpy(float)
        wx, wy = lonlat_to_projected_xy(wlon, wlat)
    # Case 3: already projected weather grid.
    elif x_col is not None and y_col is not None:
        wx = pd.to_numeric(weather_df[x_col], errors="coerce").to_numpy(float)
        wy = pd.to_numeric(weather_df[y_col], errors="coerce").to_numpy(float)
    else:
        print("[WARN] Weather file has incomplete coordinate columns.")
        print("[WARN] weather_risk is set to 0 for this run.")
        return np.zeros(len(df), dtype=float)

    finite_weather = np.isfinite(wx) & np.isfinite(wy) & np.isfinite(weather_risk_raw)
    if finite_weather.sum() == 0:
        print("[WARN] No finite weather coordinates/risk values. weather_risk is set to 0.")
        return np.zeros(len(df), dtype=float)

    weather_coords = np.column_stack([wx[finite_weather], wy[finite_weather]])
    model_coords = df[["x_m", "y_m"]].to_numpy(float)
    tree = cKDTree(weather_coords)
    dist, idx = tree.query(model_coords, k=1)

    risk = weather_risk_raw[finite_weather][idx]
    too_far = dist > WEATHER_NEAREST_MAX_DISTANCE_M
    if too_far.any():
        print(
            f"[WARN] {too_far.sum():,} model nodes are farther than "
            f"{WEATHER_NEAREST_MAX_DISTANCE_M:.1f} m from nearest weather point. "
            "Their weather_risk is set to 0."
        )
        risk[too_far] = 0.0

    print(f"[OK] Weather risk loaded from: {WEATHER_RISK_FILE}")
    return np.clip(np.nan_to_num(risk, nan=0.0), 0.0, 1.0)

def plot_operation_area(ax):
    """
    Plot Hoa Lac operation area polygon boundary.
    Draw a white halo first, then black line on top so it is visible
    on both dark and bright cost-map backgrounds.
    """
    if not PLOT_OPERATION_AREA:
        return

    poly = np.asarray(HOALAC_POLYGON, dtype=float)

    # Make sure polygon is closed.
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    # White halo for visibility.
    ax.plot(
        poly[:, 0],
        poly[:, 1],
        linewidth=OPERATION_AREA_LINEWIDTH + 2.5,
        color="white",
        linestyle="-",
        zorder=49,
    )

    # Main operation-area boundary.
    ax.plot(
        poly[:, 0],
        poly[:, 1],
        linewidth=OPERATION_AREA_LINEWIDTH,
        color="black",
        linestyle="-",
        label="Operation area",
        zorder=50,
    )

def nearest_distance_to_mask(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """
    Distance from every node to nearest node selected by mask.
    Returns inf if no selected nodes.
    """
    coords = df[["x_m", "y_m"]].to_numpy(float)
    mask = np.asarray(mask, dtype=bool)

    if mask.sum() == 0:
        return np.full(len(df), np.inf)

    tree = cKDTree(coords[mask])
    dist, _ = tree.query(coords, k=1)
    return dist


def class_contains_token(text: pd.Series, token: str) -> pd.Series:
    """
    Match class labels such as DB, DB01, DK02, FLZ, RA03 without matching
    unrelated words such as TRAVERSABLE.
    """
    pattern = rf"(?:^|[^A-Z0-9]){token}[0-9]*(?:[^A-Z0-9]|$)"
    return text.str.contains(pattern, regex=True)


def classify_special_nodes(df: pd.DataFrame) -> pd.DataFrame:
    text = df["class_text"].fillna("").astype(str).str.upper()

    df["is_db"] = (
        class_contains_token(text, "DB")
        | text.str.contains("DRONE-BASE", regex=False)
        | text.str.contains("DRONE_BASE", regex=False)
        | text.str.contains("DRONE BASE", regex=False)
    )

    df["is_dk"] = (
        class_contains_token(text, "DK")
        | text.str.contains("DOCKING", regex=False)
        | text.str.contains("DOCK", regex=False)
    )

    df["is_flz"] = (
        class_contains_token(text, "FLZ")
        | text.str.contains("FORCED-LANDING", regex=False)
        | text.str.contains("FORCED_LANDING", regex=False)
        | text.str.contains("FORCED LANDING", regex=False)
    )

    df["is_ra"] = (
        class_contains_token(text, "RA")
        | text.str.contains("RESTRICTED", regex=False)
        | text.str.contains("RESTRICTED_AIRSPACE", regex=False)
        | text.str.contains("RESTRICTED-AIRSPACE", regex=False)
        | text.str.contains("RESTRICTED AIRSPACE", regex=False)
    )

    return df
def compute_ra_circle_risk_and_hard_mask(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RA risk and hard no-fly mask from optional RA_CIRCLES.

    Each RA circle is:
        {"name": "RA01", "lon": ..., "lat": ..., "radius_m": ...}

    Inside radius_m:
        risk = 1
        hard no-fly = True

    Outside radius_m:
        risk decays smoothly with RA_CIRCLE_DECAY_M
        hard no-fly = False
    """
    risk_total = np.zeros(len(df), dtype=float)
    hard_mask = np.zeros(len(df), dtype=bool)

    if not RA_CIRCLES:
        return risk_total, hard_mask

    coords = df[["x_m", "y_m"]].to_numpy(float)

    for item in RA_CIRCLES:
        lon = float(item["lon"])
        lat = float(item["lat"])
        radius_m = float(item.get("radius_m", RA_BUFFER_M))
        decay_m = float(item.get("decay_m", RA_CIRCLE_DECAY_M))

        cx, cy = lonlat_to_projected_xy([lon], [lat])
        dist = np.hypot(coords[:, 0] - cx[0], coords[:, 1] - cy[0])

        risk = np.zeros(len(df), dtype=float)
        inside = dist <= radius_m
        risk[inside] = 1.0
        outside = ~inside
        risk[outside] = np.exp(-((dist[outside] - radius_m) / decay_m) ** 2)

        risk_total = np.maximum(risk_total, risk)
        hard_mask |= inside

    return np.clip(risk_total, 0.0, 1.0), hard_mask


def emergency_support_from_mask(df: pd.DataFrame, mask: np.ndarray, type_weight: float) -> np.ndarray:
    """
    Gaussian emergency support from one emergency class.
    """
    dist = nearest_distance_to_mask(df, mask)
    support = np.zeros(len(df), dtype=float)
    finite = np.isfinite(dist)
    support[finite] = type_weight * np.exp(-0.5 * (dist[finite] / EMERGENCY_RADIUS_M) ** 2)
    return np.clip(support, 0.0, 1.0), dist


def compute_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute updated risk/cost layers:

        risk_density
        ra_risk
        risk_obstacle
        emergency_support
        emergency_risk
        weather_risk
        risk_soft
        risk_total = risk_obstacle + (1 - risk_obstacle) * risk_soft
        is_ra_hard_nofly
        is_hard_nofly
        cost_per_m = W_TIME * slowness + W_RISK * risk_total
        cost_for_pathfinding

    Important hard no-fly rule:
        Original no-fly nodes and RA hard-area nodes are forced to cost 10.
        Emergency support is forced to 0 inside all hard no-fly nodes.
    """
    # -------------------------------
    # Base masks
    # -------------------------------
    df["is_nofly_by_slowness"] = df["slowness"] >= NOFLY_SLOWNESS

    # -------------------------------
    # Density / building / population obstacle risk
    # -------------------------------
    if df["density"].notna().sum() > 0:
        df["density_norm"] = robust_norm(df["density"].to_numpy(float), 1, 99)
    else:
        print("[WARN] No density column found. density_norm is set to 0.")
        df["density_norm"] = 0.0

    df["risk_density"] = np.clip(df["density_norm"].to_numpy(float), 0.0, 1.0)

    # -------------------------------
    # RA obstacle risk from RA-labeled nodes and/or RA circles
    # -------------------------------
    ra_dist = nearest_distance_to_mask(df, df["is_ra"].to_numpy(bool))
    df["distance_to_ra_m"] = ra_dist

    ra_risk_from_nodes = np.zeros(len(df), dtype=float)
    has_ra_distance = np.isfinite(ra_dist)

    # Hard no-fly RA area from labeled RA nodes.
    # This includes the RA nodes themselves plus the requested RA_BUFFER_M.
    ra_hard_from_nodes = has_ra_distance & (ra_dist <= RA_BUFFER_M)

    # Full RA risk inside the hard RA buffer.
    ra_risk_from_nodes[ra_hard_from_nodes] = 1.0

    # Smooth RA influence outside the hard buffer.
    # This is risk influence only, not hard no-fly.
    outside = has_ra_distance & (ra_dist > RA_BUFFER_M)
    ra_risk_from_nodes[outside] = np.exp(-((ra_dist[outside] - RA_BUFFER_M) / RA_DECAY_M) ** 2)

    # RA circles: inside radius_m is hard no-fly; outside radius_m is soft decay.
    ra_risk_from_circles, ra_hard_from_circles = compute_ra_circle_risk_and_hard_mask(df)

    df["ra_risk"] = np.maximum(ra_risk_from_nodes, ra_risk_from_circles)
    df["ra_risk"] = np.clip(df["ra_risk"].to_numpy(float), 0.0, 1.0)

    # Final hard RA mask used by emergency and final no-fly output.
    df["is_ra_hard_nofly"] = ra_hard_from_nodes | ra_hard_from_circles

    # -------------------------------
    # Obstacle risk
    # -------------------------------
    nofly_soft = df["is_nofly_by_slowness"].to_numpy(bool).astype(float)

    obstacle_layers = [
        df["risk_density"].to_numpy(float),
        df["ra_risk"].to_numpy(float),
        nofly_soft,
    ]
    obstacle_weights = [
        RISK_WEIGHT_DENSITY,
        RISK_WEIGHT_RA,
        RISK_WEIGHT_NOFLY_SOFT,
    ]

    df["risk_obstacle"] = safe_weighted_mean(obstacle_layers, obstacle_weights)

    # Hard no-fly mask for the final cost map and pathfinding map.
    # Original no-fly nodes remain no-fly, and RA hard area is also no-fly.
    df["is_hard_nofly"] = (
        df["is_nofly_by_slowness"].to_numpy(bool)
        | df["is_ra_hard_nofly"].to_numpy(bool)
    )

    # Make hard no-fly areas explicit in the risk maps as well.
    # This makes RA area appear as full obstacle risk, not only a weighted soft influence.
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    df.loc[hard_nofly, "risk_obstacle"] = 1.0

    # -------------------------------
    # Emergency support from DB, DK and FLZ
    # -------------------------------
    db_support, db_dist = emergency_support_from_mask(
        df, df["is_db"].to_numpy(bool), EMERGENCY_WEIGHT_DB
    )
    dk_support, dk_dist = emergency_support_from_mask(
        df, df["is_dk"].to_numpy(bool), EMERGENCY_WEIGHT_DK
    )
    flz_support, flz_dist = emergency_support_from_mask(
        df, df["is_flz"].to_numpy(bool), EMERGENCY_WEIGHT_FLZ
    )

    df["distance_to_db_m"] = db_dist
    df["distance_to_dk_m"] = dk_dist
    df["distance_to_flz_m"] = flz_dist

    # Minimum finite distance to any emergency type.
    # If no DB/DK/FLZ exists, the distance remains inf.
    dist_stack = np.vstack([db_dist, dk_dist, flz_dist])
    dist_stack = np.where(np.isfinite(dist_stack), dist_stack, np.inf)
    df["distance_to_emergency_m"] = np.min(dist_stack, axis=0)

    df["emergency_support_db"] = db_support
    df["emergency_support_dk"] = dk_support
    df["emergency_support_flz"] = flz_support
    df["emergency_support"] = np.maximum.reduce([db_support, dk_support, flz_support])
    df["emergency_support"] = np.clip(df["emergency_support"].to_numpy(float), 0.0, 1.0)

    # RA and other hard no-fly areas cannot provide usable emergency support.
    # This is the key rule requested: inside RA hard area, support = 0 and risk = 1.
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    df.loc[hard_nofly, "emergency_support_db"] = 0.0
    df.loc[hard_nofly, "emergency_support_dk"] = 0.0
    df.loc[hard_nofly, "emergency_support_flz"] = 0.0
    df.loc[hard_nofly, "emergency_support"] = 0.0

    df["emergency_risk"] = 1.0 - df["emergency_support"]

    # Backward-compatible old name.
    df["emergency_penalty"] = df["emergency_risk"]

    # -------------------------------
    # Optional weather risk
    # -------------------------------
    df["weather_risk"] = load_weather_risk_on_model_nodes(df)

    # -------------------------------
    # Soft risk
    # -------------------------------
    effective_weather_weight = W_SOFT_WEATHER if INCLUDE_WEATHER_RISK else 0.0
    soft_layers = [
        df["emergency_risk"].to_numpy(float),
        df["weather_risk"].to_numpy(float),
    ]
    soft_weights = [
        W_SOFT_EMERGENCY,
        effective_weather_weight,
    ]

    df["risk_soft"] = safe_weighted_mean(soft_layers, soft_weights)

    # -------------------------------
    # Updated total risk formula
    # -------------------------------
    obstacle = df["risk_obstacle"].to_numpy(float)
    soft = df["risk_soft"].to_numpy(float)
    df["risk_total"] = obstacle + (1.0 - obstacle) * soft
    df["risk_total"] = np.clip(df["risk_total"].to_numpy(float), 0.0, 1.0)

    # Hard no-fly areas are full total risk.
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    df.loc[hard_nofly, "risk_total"] = 1.0

    # Backward-compatible old name used by older plotting/downstream scripts.
    df["risk_norm"] = df["risk_total"]

    # -------------------------------
    # Final impedance per meter
    # -------------------------------
    df["cost_per_m"] = (
        W_TIME * df["slowness"].to_numpy(float)
        + W_RISK * df["risk_total"].to_numpy(float)
    )

    # In the final cost map, hard no-fly remains exactly 10.
    # This keeps original no-fly cells unchanged and makes RA hard area no-fly too.
    if USE_HARD_NOFLY_OUTPUT:
        df.loc[hard_nofly, "cost_per_m"] = NOFLY_SLOWNESS

    # -------------------------------
    # Pathfinding cost/slowness output
    # -------------------------------
    df["cost_for_pathfinding"] = df["cost_per_m"].to_numpy(float)

    if USE_HARD_NOFLY_OUTPUT:
        nofly = df["is_hard_nofly"].to_numpy(bool)

        # Keep flyable nodes below no-fly threshold.
        df.loc[~nofly, "cost_for_pathfinding"] = np.minimum(
            df.loc[~nofly, "cost_for_pathfinding"].to_numpy(float),
            NOFLY_BLOCK_THRESHOLD,
        )

        # Set hard no-fly nodes to exactly 10.
        # This includes original no-fly cells and RA hard-area cells.
        df.loc[nofly, "cost_for_pathfinding"] = NOFLY_SLOWNESS

    # -------------------------------
    # Final slowness and velocity for visualization
    # -------------------------------
    # The final slowness column is the value that should be interpreted by the
    # pathfinder as s/m. By default it is cost_for_pathfinding, because this is
    # the final impedance after risk/emergency/RA logic. Set
    # FINAL_SLOWNESS_SOURCE_COLUMN = "slowness" if you want to visualize only
    # the original base slowness.
    if FINAL_SLOWNESS_SOURCE_COLUMN not in df.columns:
        raise KeyError(
            f"FINAL_SLOWNESS_SOURCE_COLUMN={FINAL_SLOWNESS_SOURCE_COLUMN!r} not found in dataframe."
        )

    final_slowness = df[FINAL_SLOWNESS_SOURCE_COLUMN].to_numpy(float)
    df[FINAL_SLOWNESS_COLUMN] = final_slowness

    final_velocity = np.zeros(len(df), dtype=float)
    valid_velocity = np.isfinite(final_slowness) & (final_slowness > 0.0)
    final_velocity[valid_velocity] = 1.0 / final_slowness[valid_velocity]

    # Hard no-fly cells are not physically traversable. Plot them as 0 velocity.
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    final_velocity[hard_nofly] = FINAL_VELOCITY_HARD_NOFLY_VALUE

    df[FINAL_VELOCITY_COLUMN] = final_velocity
    df[FINAL_VELOCITY_KMH_COLUMN] = final_velocity * 3.6

    return df
def save_outputs(df: pd.DataFrame) -> None:
    csv_file = OUT_DIR / "cost_map_nodes.csv"
    xyz_file = OUT_DIR / "cost_map_nodes.xyz"
    pf_file = OUT_DIR / "model_senario1_cost_for_pathfinding.xyz"
    summary_file = OUT_DIR / "cost_map_summary.csv"

    out_cols = [
        "lon",
        "lat",
        "elevation_m",
        "slowness",
        "density",
        "density_norm",
        "risk_density",
        "ra_risk",
        "is_ra_hard_nofly",
        "is_hard_nofly",
        "risk_obstacle",
        "emergency_support_db",
        "emergency_support_dk",
        "emergency_support_flz",
        "emergency_support",
        "emergency_risk",
        "weather_risk",
        "risk_soft",
        "risk_total",
        "risk_norm",
        "cost_per_m",
        "cost_for_pathfinding",
        FINAL_SLOWNESS_COLUMN,
        FINAL_VELOCITY_COLUMN,
        FINAL_VELOCITY_KMH_COLUMN,
        "distance_to_ra_m",
        "distance_to_db_m",
        "distance_to_dk_m",
        "distance_to_flz_m",
        "distance_to_emergency_m",
        "category",
        "label",
    ]

    df[out_cols].to_csv(csv_file, index=False, float_format="%.8f")

    # Headered xyz for checking / plotting.
    df[out_cols].to_csv(
        xyz_file,
        sep=" ",
        index=False,
        header=True,
        float_format="%.8f",
    )

    # No-header pathfinding model.
    # Replace slowness column by cost_for_pathfinding.
    pf_cols = [
        "lon",
        "lat",
        "elevation_m",
        "cost_for_pathfinding",
        "category",
        "density",
        "label",
    ]

    df[pf_cols].to_csv(
        pf_file,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    summary = {
        "nodes": len(df),
        "db_nodes": int(df["is_db"].sum()),
        "dk_nodes": int(df["is_dk"].sum()),
        "flz_nodes": int(df["is_flz"].sum()),
        "ra_nodes": int(df["is_ra"].sum()),
        "ra_circles": len(RA_CIRCLES),
        "ra_hard_nofly_nodes": int(df["is_ra_hard_nofly"].sum()),
        "nofly_nodes_by_slowness": int(df["is_nofly_by_slowness"].sum()),
        "hard_nofly_nodes_total": int(df["is_hard_nofly"].sum()),
        "risk_obstacle_min": float(np.nanmin(df["risk_obstacle"])),
        "risk_obstacle_max": float(np.nanmax(df["risk_obstacle"])),
        "risk_obstacle_mean": float(np.nanmean(df["risk_obstacle"])),
        "risk_soft_min": float(np.nanmin(df["risk_soft"])),
        "risk_soft_max": float(np.nanmax(df["risk_soft"])),
        "risk_soft_mean": float(np.nanmean(df["risk_soft"])),
        "risk_total_min": float(np.nanmin(df["risk_total"])),
        "risk_total_max": float(np.nanmax(df["risk_total"])),
        "risk_total_mean": float(np.nanmean(df["risk_total"])),
        "weather_risk_min": float(np.nanmin(df["weather_risk"])),
        "weather_risk_max": float(np.nanmax(df["weather_risk"])),
        "weather_risk_mean": float(np.nanmean(df["weather_risk"])),
        "emergency_support_min": float(np.nanmin(df["emergency_support"])),
        "emergency_support_max": float(np.nanmax(df["emergency_support"])),
        "emergency_support_mean": float(np.nanmean(df["emergency_support"])),
        "cost_per_m_min": float(np.nanmin(df["cost_per_m"])),
        "cost_per_m_max": float(np.nanmax(df["cost_per_m"])),
        "cost_per_m_mean": float(np.nanmean(df["cost_per_m"])),
        "final_slowness_source_column": FINAL_SLOWNESS_SOURCE_COLUMN,
        "final_slowness_min": float(np.nanmin(df[FINAL_SLOWNESS_COLUMN])),
        "final_slowness_max": float(np.nanmax(df[FINAL_SLOWNESS_COLUMN])),
        "final_slowness_mean": float(np.nanmean(df[FINAL_SLOWNESS_COLUMN])),
        "final_velocity_mps_min": float(np.nanmin(df[FINAL_VELOCITY_COLUMN])),
        "final_velocity_mps_max": float(np.nanmax(df[FINAL_VELOCITY_COLUMN])),
        "final_velocity_mps_mean": float(np.nanmean(df[FINAL_VELOCITY_COLUMN])),
        "W_TIME": W_TIME,
        "W_RISK": W_RISK,
        "RISK_WEIGHT_DENSITY": RISK_WEIGHT_DENSITY,
        "RISK_WEIGHT_RA": RISK_WEIGHT_RA,
        "W_SOFT_EMERGENCY": W_SOFT_EMERGENCY,
        "INCLUDE_WEATHER_RISK": INCLUDE_WEATHER_RISK,
        "W_SOFT_WEATHER": W_SOFT_WEATHER,
        "RA_BUFFER_M": RA_BUFFER_M,
        "RA_DECAY_M": RA_DECAY_M,
        "EMERGENCY_RADIUS_M": EMERGENCY_RADIUS_M,
        "NOFLY_SLOWNESS": NOFLY_SLOWNESS,
        "PLOT_SMOOTH_MAP": bool(PLOT_SMOOTH_MAP),
        "SMOOTH_LEVELS": int(SMOOTH_LEVELS),
        "FORCE_ZERO_ONE_COLORBAR_FOR_RISK_LAYERS": bool(FORCE_ZERO_ONE_COLORBAR_FOR_RISK_LAYERS),
        "PRESERVE_HARD_NOFLY_RED_ON_SMOOTH": bool(PRESERVE_HARD_NOFLY_RED_ON_SMOOTH),
    }

    pd.DataFrame([summary]).to_csv(summary_file, index=False)

    print(f"[OK] Saved CSV: {csv_file}")
    print(f"[OK] Saved XYZ: {xyz_file}")
    print(f"[OK] Saved pathfinding model: {pf_file}")
    print(f"[OK] Saved summary: {summary_file}")
def get_plot_limits(column: str, value: np.ndarray) -> tuple[float | None, float | None, np.ndarray | int | None, list[float] | None]:
    """
    Decide color scale limits for each plotted layer.

    Risk/support layers are always 0-1 scores. Keeping vmin=0 and vmax=1
    avoids misleading colorbars such as -1 to 1 when the layer is constant,
    especially when weather risk is disabled and therefore all zeros.

    Cost layers keep automatic scaling because their units are cost/slowness.
    """
    if FORCE_ZERO_ONE_COLORBAR_FOR_RISK_LAYERS and column in ZERO_ONE_PLOT_COLUMNS:
        levels = np.linspace(0.0, 1.0, int(SMOOTH_LEVELS))
        return 0.0, 1.0, levels, ZERO_ONE_COLORBAR_TICKS

    if column in COST_PLOT_COLUMNS:
        levels = np.linspace(0.0, float(NOFLY_SLOWNESS), int(SMOOTH_LEVELS))
        return 0.0, float(NOFLY_SLOWNESS), levels, COST_COLORBAR_TICKS

    return None, None, SMOOTH_LEVELS, None


def should_overlay_hard_nofly(column: str) -> bool:
    """
    Decide whether a layer should preserve hard no-fly cells as a solid red overlay.

    This prevents artificial white halos caused by smoothing a discontinuous
    hard no-fly class together with nearby low-risk flyable nodes.
    """
    return (
        PRESERVE_HARD_NOFLY_RED_ON_SMOOTH
        and column in HARD_NOFLY_OVERLAY_COLUMNS
    )


def plot_hard_nofly_overlay(ax, df: pd.DataFrame, hard_mask: np.ndarray) -> None:
    """
    Draw hard no-fly cells as a gentle red overlay on top of the smooth background.

    The color is taken from the high end of MAP_CMAP, but not the absolute maximum,
    so it still means "very high risk / no-fly" while looking a little softer.
    """
    hard_mask = np.asarray(hard_mask, dtype=bool)
    if hard_mask.sum() == 0:
        return

    red_color = plt.get_cmap(MAP_CMAP)(float(HARD_NOFLY_OVERLAY_COLOR_VALUE))

    lon = df["lon"].to_numpy(float)
    lat = df["lat"].to_numpy(float)
    hard_value = hard_mask.astype(float)

    if HARD_NOFLY_OVERLAY_STYLE.lower() == "contour" and PLOT_SMOOTH_MAP and len(df) >= 3:
        try:
            ax.tricontourf(
                lon,
                lat,
                hard_value,
                levels=[0.5, 1.5],
                colors=[red_color],
                alpha=HARD_NOFLY_OVERLAY_ALPHA,
                zorder=20,
            )
            return
        except Exception as exc:
            print(f"[WARN] Hard no-fly overlay contour failed. Fallback to scatter. Reason: {exc}")

    ax.scatter(
        lon[hard_mask],
        lat[hard_mask],
        c=[red_color],
        s=HARD_NOFLY_OVERLAY_SCATTER_SIZE,
        marker=HARD_NOFLY_OVERLAY_MARKER,
        linewidths=0,
        alpha=HARD_NOFLY_OVERLAY_ALPHA,
        zorder=20,
    )


def plot_layer(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_file: Path,
    label: str,
    mask_nofly: bool = False,
    plot_font: str | None = None,
):
    value = df[column].to_numpy(float).copy()
    hard_nofly = df.get(
        "is_hard_nofly",
        pd.Series(False, index=df.index),
    ).to_numpy(bool)

    if mask_nofly:
        value[hard_nofly] = np.nan

    # Enforce valid risk/support range before plotting.
    if column in ZERO_ONE_PLOT_COLUMNS:
        value = np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0)

    # Do not smooth hard no-fly cells into the flyable background.
    # They will be drawn as a solid red overlay below.
    overlay_hard = should_overlay_hard_nofly(column)
    if overlay_hard:
        value[hard_nofly] = np.nan

    finite = np.isfinite(value)
    plot_vmin, plot_vmax, plot_levels, cbar_ticks = get_plot_limits(column, value)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Smooth surface is easier on the eyes than dense point clouds.
    # Fall back to scatter automatically if triangulated contouring fails.
    sc = None
    if np.count_nonzero(finite) == 0:
        # This can happen if a layer contains only hard no-fly cells after masking.
        # Create a mappable only for the colorbar; the map content is handled by overlay.
        norm = mpl.colors.Normalize(vmin=plot_vmin, vmax=plot_vmax)
        sc = mpl.cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
        sc.set_array([])
    elif PLOT_SMOOTH_MAP and np.count_nonzero(finite) >= 3:
        try:
            sc = ax.tricontourf(
                df.loc[finite, "lon"].to_numpy(float),
                df.loc[finite, "lat"].to_numpy(float),
                value[finite],
                levels=plot_levels,
                cmap=MAP_CMAP,
                vmin=plot_vmin,
                vmax=plot_vmax,
            )
            if SMOOTH_OVERLAY_POINTS:
                ax.scatter(
                    df.loc[finite, "lon"],
                    df.loc[finite, "lat"],
                    c=value[finite],
                    s=SMOOTH_POINT_SIZE,
                    cmap=MAP_CMAP,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                    linewidths=0,
                    alpha=0.35,
                    zorder=10,
                )
        except Exception as exc:
            print(f"[WARN] Smooth plot failed for {column}. Fallback to scatter. Reason: {exc}")
            sc = ax.scatter(
                df.loc[finite, "lon"],
                df.loc[finite, "lat"],
                c=value[finite],
                s=4,
                cmap=MAP_CMAP,
                vmin=plot_vmin,
                vmax=plot_vmax,
                linewidths=0,
            )
    else:
        sc = ax.scatter(
            df.loc[finite, "lon"],
            df.loc[finite, "lat"],
            c=value[finite],
            s=4,
            cmap=MAP_CMAP,
            vmin=plot_vmin,
            vmax=plot_vmax,
            linewidths=0,
        )

    # Preserve RA / original no-fly cells as solid red after smoothing.
    # This removes artificial white transition zones around hard obstacles.
    if overlay_hard:
        plot_hard_nofly_overlay(ax, df, hard_nofly)

    # Plot special nodes on top.
    if df["is_db"].sum() > 0:
        ax.scatter(
            df.loc[df["is_db"], "lon"],
            df.loc[df["is_db"], "lat"],
            marker="s",
            s=60,
            edgecolors="black",
            facecolors="none",
            label="DB",
        )

    if df["is_flz"].sum() > 0:
        ax.scatter(
            df.loc[df["is_flz"], "lon"],
            df.loc[df["is_flz"], "lat"],
            marker="^",
            s=60,
            edgecolors="black",
            facecolors="none",
            label="FLZ",
        )

    if df["is_dk"].sum() > 0:
        ax.scatter(
            df.loc[df["is_dk"], "lon"],
            df.loc[df["is_dk"], "lat"],
            marker="D",
            s=50,
            edgecolors="black",
            facecolors="none",
            label="DK",
        )

    if df["is_ra"].sum() > 0:
        ax.scatter(
            df.loc[df["is_ra"], "lon"],
            df.loc[df["is_ra"], "lat"],
            marker="x",
            s=40,
            label="RA",
            zorder=30,
        )

    # Plot Hoa Lac operation area boundary.
    plot_operation_area(ax)

    ax.set_title(
        title,
        **font_kwargs(plot_font, TITLE_FONT_SIZE, TITLE_FONT_WEIGHT),
    )
    ax.set_xlabel(
        "Longitude",
        **font_kwargs(plot_font, AXIS_LABEL_FONT_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )
    ax.set_ylabel(
        "Latitude",
        **font_kwargs(plot_font, AXIS_LABEL_FONT_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        if plot_font is not None:
            tick_label.set_fontname(plot_font)
        tick_label.set_fontsize(TICK_LABEL_FONT_SIZE)

    if PLOT_OPERATION_AREA:
        poly = np.asarray(HOALAC_POLYGON, dtype=float)

        lon_min, lat_min = poly.min(axis=0)
        lon_max, lat_max = poly.max(axis=0)

        pad_lon = 0.003
        pad_lat = 0.003

        ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    cbar = plt.colorbar(
        sc,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        ticks=cbar_ticks,
    )
    cbar.set_label(
        label,
        **font_kwargs(plot_font, COLORBAR_LABEL_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for tick_label in cbar.ax.get_yticklabels():
        if plot_font is not None:
            tick_label.set_fontname(plot_font)
        tick_label.set_fontsize(COLORBAR_TICK_SIZE)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            ncol=LEGEND_NCOL,
            fontsize=LEGEND_FONT_SIZE,
            frameon=True,
            fancybox=True,
            framealpha=LEGEND_FRAME_ALPHA,
        )
        legend.set_zorder(LEGEND_ZORDER)
        legend.get_frame().set_facecolor(LEGEND_FACE_COLOR)
        legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
        legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
        for text_obj in legend.get_texts():
            if plot_font is not None:
                text_obj.set_fontname(plot_font)
            text_obj.set_fontsize(LEGEND_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved figure: {output_file}")



def percentile_plot_max(values: np.ndarray, mask: np.ndarray, user_max: float | None) -> float:
    """
    Return a clean colorbar maximum for the final slowness/velocity panels.
    Hard no-fly cells are excluded so the flyable part is not visually flattened.
    """
    if user_max is not None:
        return float(user_max)

    v = np.asarray(values, dtype=float)
    m = np.asarray(mask, dtype=bool)
    good = np.isfinite(v) & ~m
    if good.sum() == 0:
        good = np.isfinite(v)
    if good.sum() == 0:
        return 1.0

    vmax = float(np.nanpercentile(v[good], FINAL_COLORBAR_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(v[good]))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def plot_special_nodes_and_legend(ax, df: pd.DataFrame, plot_font: str | None = None, show_legend: bool = True) -> None:
    """
    Plot DB / DK / FLZ / RA markers, operation-area outline, and optional legend.
    This is used by the two-panel final slowness/velocity figure.
    """
    if df["is_db"].sum() > 0:
        ax.scatter(
            df.loc[df["is_db"], "lon"],
            df.loc[df["is_db"], "lat"],
            marker="s",
            s=60,
            edgecolors="black",
            facecolors="none",
            label="DB",
            zorder=40,
        )

    if df["is_flz"].sum() > 0:
        ax.scatter(
            df.loc[df["is_flz"], "lon"],
            df.loc[df["is_flz"], "lat"],
            marker="^",
            s=60,
            edgecolors="black",
            facecolors="none",
            label="FLZ",
            zorder=40,
        )

    if df["is_dk"].sum() > 0:
        ax.scatter(
            df.loc[df["is_dk"], "lon"],
            df.loc[df["is_dk"], "lat"],
            marker="D",
            s=50,
            edgecolors="black",
            facecolors="none",
            label="DK",
            zorder=40,
        )

    if df["is_ra"].sum() > 0:
        ax.scatter(
            df.loc[df["is_ra"], "lon"],
            df.loc[df["is_ra"], "lat"],
            marker="x",
            s=40,
            label="RA",
            zorder=45,
        )

    plot_operation_area(ax)

    if show_legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                loc=LEGEND_LOC,
                bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
                ncol=LEGEND_NCOL,
                fontsize=LEGEND_FONT_SIZE,
                frameon=True,
                fancybox=True,
                framealpha=LEGEND_FRAME_ALPHA,
            )
            legend.set_zorder(LEGEND_ZORDER)
            legend.get_frame().set_facecolor(LEGEND_FACE_COLOR)
            legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
            legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
            for text_obj in legend.get_texts():
                if plot_font is not None:
                    text_obj.set_fontname(plot_font)
                text_obj.set_fontsize(LEGEND_FONT_SIZE)


def format_map_axis(ax, title: str, plot_font: str | None = None) -> None:
    """
    Common map formatting for the final slowness/velocity panels.
    """
    ax.set_title(
        title,
        **font_kwargs(plot_font, FINAL_PANEL_TITLE_SIZE, TITLE_FONT_WEIGHT),
    )
    ax.set_xlabel(
        "Longitude",
        **font_kwargs(plot_font, AXIS_LABEL_FONT_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )
    ax.set_ylabel(
        "Latitude",
        **font_kwargs(plot_font, AXIS_LABEL_FONT_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        if plot_font is not None:
            tick_label.set_fontname(plot_font)
        tick_label.set_fontsize(TICK_LABEL_FONT_SIZE)

    if PLOT_OPERATION_AREA:
        poly = np.asarray(HOALAC_POLYGON, dtype=float)
        lon_min, lat_min = poly.min(axis=0)
        lon_max, lat_max = poly.max(axis=0)
        pad_lon = 0.003
        pad_lat = 0.003
        ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
        ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def plot_panel_surface(
    ax,
    df: pd.DataFrame,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """
    Draw one smooth/scatter panel while excluding hard no-fly cells from interpolation.
    Hard no-fly cells are drawn later with the overlay so they remain forced areas.
    """
    values = np.asarray(values, dtype=float).copy()
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    values[hard_nofly] = np.nan
    finite = np.isfinite(values)
    levels = np.linspace(vmin, vmax, int(SMOOTH_LEVELS))

    if np.count_nonzero(finite) == 0:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sc = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sc.set_array([])
        return sc

    if PLOT_SMOOTH_MAP and np.count_nonzero(finite) >= 3:
        try:
            return ax.tricontourf(
                df.loc[finite, "lon"].to_numpy(float),
                df.loc[finite, "lat"].to_numpy(float),
                values[finite],
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception as exc:
            print(f"[WARN] Final panel smooth plot failed. Fallback to scatter. Reason: {exc}")

    return ax.scatter(
        df.loc[finite, "lon"],
        df.loc[finite, "lat"],
        c=values[finite],
        s=4,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )


def add_panel_colorbar(fig, ax, mappable, label: str, ticks: np.ndarray, plot_font: str | None = None) -> None:
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=COLORBAR_SHRINK,
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        ticks=ticks,
    )
    cbar.set_label(
        label,
        **font_kwargs(plot_font, COLORBAR_LABEL_SIZE, AXIS_LABEL_FONT_WEIGHT),
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for tick_label in cbar.ax.get_yticklabels():
        if plot_font is not None:
            tick_label.set_fontname(plot_font)
        tick_label.set_fontsize(COLORBAR_TICK_SIZE)


def plot_final_slowness_velocity_map(df: pd.DataFrame, output_file: Path, plot_font: str | None = None) -> None:
    """
    Final cost-map figure for pathfinding.

    Top panel:
        final effective slowness used by the pathfinder, in s/m.

    Bottom panel:
        equivalent effective velocity, velocity = 1 / slowness, in m/s.
        Hard no-fly cells are plotted as 0 velocity and overlaid as forced areas.
    """
    hard_nofly = df["is_hard_nofly"].to_numpy(bool)
    slowness = df[FINAL_SLOWNESS_COLUMN].to_numpy(float)
    velocity = df[FINAL_VELOCITY_COLUMN].to_numpy(float)

    s_vmin = 0.0
    s_vmax = percentile_plot_max(slowness, hard_nofly, FINAL_SLOWNESS_COLORBAR_MAX)
    v_vmin = 0.0
    v_vmax = percentile_plot_max(velocity, hard_nofly, FINAL_VELOCITY_COLORBAR_MAX)

    s_ticks = np.linspace(s_vmin, s_vmax, 5)
    v_ticks = np.linspace(v_vmin, v_vmax, 5)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FINAL_FIGSIZE,
        sharex=False,
        sharey=False,
    )

    # Slowness panel: higher slowness is worse, so it uses MAP_CMAP.
    sc1 = plot_panel_surface(
        axes[0],
        df,
        slowness,
        cmap=MAP_CMAP,
        vmin=s_vmin,
        vmax=s_vmax,
    )
    plot_hard_nofly_overlay(axes[0], df, hard_nofly)
    plot_special_nodes_and_legend(axes[0], df, plot_font=plot_font, show_legend=True)
    format_map_axis(
        axes[0],
        f"Final Slowness Map ({FINAL_SLOWNESS_SOURCE_COLUMN})",
        plot_font=plot_font,
    )
    add_panel_colorbar(
        fig,
        axes[0],
        sc1,
        "Slowness (s/m)",
        s_ticks,
        plot_font=plot_font,
    )

    # Velocity panel: reverse the colormap so low velocity/no-fly is red and
    # high velocity is blue, matching the slowness-risk interpretation.
    sc2 = plot_panel_surface(
        axes[1],
        df,
        velocity,
        cmap=FINAL_VELOCITY_CMAP,
        vmin=v_vmin,
        vmax=v_vmax,
    )
    plot_hard_nofly_overlay(axes[1], df, hard_nofly)
    plot_special_nodes_and_legend(axes[1], df, plot_font=plot_font, show_legend=False)
    format_map_axis(
        axes[1],
        "Equivalent Velocity Map: velocity = 1 / slowness",
        plot_font=plot_font,
    )
    add_panel_colorbar(
        fig,
        axes[1],
        sc2,
        "Velocity (m/s)",
        v_ticks,
        plot_font=plot_font,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved final slowness/velocity figure: {output_file}")

def make_plots(df: pd.DataFrame) -> None:
    plot_font = apply_plot_style()

    plot_layer(
        df,
        "risk_obstacle",
        "Obstacle Risk Map: Density + RA",
        FIG_DIR / "obstacle_risk_map.png",
        "Obstacle risk",
        plot_font=plot_font,
    )

    plot_layer(
        df,
        "emergency_support",
        "Emergency Support Map: DB / DK / FLZ Influence",
        FIG_DIR / "emergency_support_map.png",
        "Emergency support",
        plot_font=plot_font,
    )

    plot_layer(
        df,
        "emergency_risk",
        "Emergency Risk Map: 1 - Emergency Support",
        FIG_DIR / "emergency_risk_map.png",
        "Emergency risk",
        plot_font=plot_font,
    )

    plot_layer(
        df,
        "weather_risk",
        "Weather Risk Map: Optional Layer",
        FIG_DIR / "weather_risk_map.png",
        "Weather risk",
        plot_font=plot_font,
    )

    plot_layer(
        df,
        "risk_soft",
        "Soft Risk Map: Emergency Absence + Optional Weather",
        FIG_DIR / "soft_risk_map.png",
        "Soft risk",
        plot_font=plot_font,
    )

    plot_layer(
        df,
        "risk_total",
        "Total Risk Map: R_obstacle + (1 - R_obstacle) × R_soft",
        FIG_DIR / "total_risk_map.png",
        "Total risk",
        plot_font=plot_font,
    )

    # Backward-compatible filename for older workflow.
    plot_layer(
        df,
        "risk_total",
        "Risk Map: Total Risk",
        FIG_DIR / "risk_map.png",
        "Total risk",
        plot_font=plot_font,
    )

    if PLOT_FINAL_COST_WITH_VELOCITY_PANEL:
        plot_final_slowness_velocity_map(
            df,
            FIG_DIR / "final_cost_map.png",
            plot_font=plot_font,
        )
    else:
        plot_layer(
            df,
            FINAL_SLOWNESS_COLUMN,
            "Final Slowness Map",
            FIG_DIR / "final_cost_map.png",
            "Slowness (s/m)",
            mask_nofly=False,
            plot_font=plot_font,
        )

    plot_layer(
        df,
        "cost_for_pathfinding",
        "Pathfinding Cost Map",
        FIG_DIR / "pathfinding_cost_map.png",
        "Pathfinding cost/slowness",
        mask_nofly=False,
        plot_font=plot_font,
    )
def print_report(df: pd.DataFrame) -> None:
    print("\n========== COST MAP REPORT ==========")
    print(f"Nodes:                    {len(df):,}")
    print(f"DB nodes:                 {int(df['is_db'].sum()):,}")
    print(f"DK nodes:                 {int(df['is_dk'].sum()):,}")
    print(f"FLZ nodes:                {int(df['is_flz'].sum()):,}")
    print(f"RA nodes:                 {int(df['is_ra'].sum()):,}")
    print(f"RA circles:               {len(RA_CIRCLES):,}")
    print(f"RA hard no-fly nodes:     {int(df['is_ra_hard_nofly'].sum()):,}")
    print(f"Original no-fly nodes:    {int(df['is_nofly_by_slowness'].sum()):,}")
    print(f"Total hard no-fly nodes:  {int(df['is_hard_nofly'].sum()):,}")
    print(f"Obstacle risk min/max/mean:{df['risk_obstacle'].min():.4f} / {df['risk_obstacle'].max():.4f} / {df['risk_obstacle'].mean():.4f}")
    print(f"Emergency support min/max/mean:{df['emergency_support'].min():.4f} / {df['emergency_support'].max():.4f} / {df['emergency_support'].mean():.4f}")
    print(f"Weather enabled:          {INCLUDE_WEATHER_RISK}")
    print(f"Weather weight:           {W_SOFT_WEATHER:.4f}")
    print(f"Weather risk min/max/mean:{df['weather_risk'].min():.4f} / {df['weather_risk'].max():.4f} / {df['weather_risk'].mean():.4f}")
    print(f"Soft risk min/max/mean:   {df['risk_soft'].min():.4f} / {df['risk_soft'].max():.4f} / {df['risk_soft'].mean():.4f}")
    print(f"Total risk min/max/mean:  {df['risk_total'].min():.4f} / {df['risk_total'].max():.4f} / {df['risk_total'].mean():.4f}")
    print(f"Cost min/max/mean:        {df['cost_per_m'].min():.6f} / {df['cost_per_m'].max():.6f} / {df['cost_per_m'].mean():.6f}")
    print("\n[NOTE] Risk formula:")
    print("       risk_total = risk_obstacle + (1 - risk_obstacle) * risk_soft")
    print("[NOTE] Emergency is inside risk_soft as emergency_risk = 1 - emergency_support.")
    print("[NOTE] Weather is optional. Default: INCLUDE_WEATHER_RISK=False and W_SOFT_WEATHER=0.")
    print("[NOTE] Original no-fly nodes and RA hard area are forced to cost = 10.")
    print("[NOTE] RA hard area also forces emergency_support = 0.")
    print("[NOTE] If no-fly = 10, set your pathfinder traversable rule to cost < 10.")


# ============================================================
# Optional edge-cost functions for your A* code
# ============================================================

def edge_cost_meters(x1, y1, cost1, x2, y2, cost2) -> float:
    """
    Use this inside your graph/A* algorithm.

    Edge cost = metric distance * average endpoint cost.
    """
    d = math.hypot(x2 - x1, y2 - y1)
    return d * 0.5 * (cost1 + cost2)


def admissible_heuristic_meters(x, y, gx, gy, min_cost_per_m) -> float:
    """
    A* heuristic that remains conservative:
        straight-line distance * minimum possible cost per meter.
    """
    return math.hypot(gx - x, gy - y) * min_cost_per_m


# ============================================================
# Main
# ============================================================

def main():
    model_file = choose_model_file()

    df = read_model_xyz(model_file)
    df = add_projected_xy(df)
    df = classify_special_nodes(df)
    df = compute_layers(df)

    save_outputs(df)
    make_plots(df)
    print_report(df)

    print("\n[DONE] Cost map calculation complete.")


if __name__ == "__main__":
    main()