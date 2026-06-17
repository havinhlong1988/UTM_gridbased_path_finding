#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare and plot 2D model for scenario 1 path-finding test.

Input model:
  output/02_senario1_no_velocity/03_mixed_model/extracted_2d_models/
      mixed_model_2d_req_10m_sel_0m.xyz

Expected model format:
  lon lat z slowness

Input plan data:
  input/02_data_senario1_no_velocity/kml_plan/*.xyz

Special plan file:
  setup_fly_control.xyz

This file defines the fly-control / restricted circle area.
All model nodes inside this area are forced to no-fly.

Final numeric path-finding model:
  mixed_model_2d_after_fly_control_for_pathfinding.xyz
  format: lon lat z slowness

Final labeled path-finding model:
  mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz
  format: lon lat z slowness label

Labels:
  N     = normal model node
  DB01  = Drone-Base-01
  DK01  = Docking-01
  FLZ01 = FLZ-01
  RA01  = Restricted_airspace-01

Figures:
  1. before collision model only
  2. after collision zoom:
       - no fly-control circle plotted
       - only red nodes represent fly-control collision
       - objectives are plotted
  3. after collision full map:
       - fly-control circle is plotted
       - fly-control center is plotted
       - fly-control radius line and radius label are plotted
       - Hoa Lac polygon is plotted
       - no path-connected plan lines
       - objectives are plotted
"""

from pathlib import Path
import shutil
import warnings
import re

import numpy as np
import pandas as pd
import pygmt

from matplotlib.path import Path as MplPath
from pyproj import Geod


# ============================================================
# User settings
# ============================================================

PROJECT_DIR = Path(".").resolve()

MODEL_FILE = (
    PROJECT_DIR
    / "output"
    / "02_senario1_no_velocity"
    / "03_mixed_model"
    / "extracted_2d_models"
    / "mixed_model_2d_req_10m_sel_0m.xyz"
)

PLAN_DIR = (
    PROJECT_DIR
    / "input"
    / "02_data_senario1_no_velocity"
    / "kml_plan"
)

OUT_DIR = (
    PROJECT_DIR
    / "output"
    / "02_senario1_no_velocity"
    / "04_2D_model_senario_1"
)

FIG_DIR = (
    PROJECT_DIR
    / "figures"
    / "02_senario1_no_velocity"
    / "04_2D_model_senario_1"
)

CLEAR_OUTPUT = True

REGION_PADDING = 0.003
PROJECTION = "M15c"
DPI = 300

GEOD = Geod(ellps="WGS84")


# ============================================================
# Hoa Lac polygon, lon/lat
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


# ============================================================
# Slowness classification
# ============================================================

# Nodes with slowness >= this value are treated as no-fly.
NO_FLY_SLOWNESS_THRESHOLD = 1

# Forced no-fly slowness value for nodes inside setup_fly_control.xyz.
FORCED_NOFLY_SLOWNESS_VALUE = 10

SLOWNESS_ROUND_DECIMALS = 12


# ============================================================
# Fly-control collision settings
# ============================================================

FLY_CONTROL_SOURCE_FILES = [
    "setup_fly_control.xyz",
]

REMOVE_CENTER_POINT_FROM_FLY_CONTROL = True


# ============================================================
# Plot settings
# ============================================================

MODEL_NODE_STYLE = "s0.045c"

FLYABLE_COLOR = "blue"
NOFLY_COLOR = "red"

HOALAC_POLYGON_PEN = "2.0p,purple"
HOALAC_POLYGON_FILL = "purple@92"

FLY_CONTROL_LINE_PEN = "2.0p,red"
FLY_CONTROL_FILL = "red@88"

FLY_CONTROL_CENTER_STYLE = "c0.28c"
FLY_CONTROL_CENTER_FILL = "white"
FLY_CONTROL_CENTER_PEN = "1.2p,red"

FLY_CONTROL_RADIUS_LINE_PEN = "1.2p,red,--"
FLY_CONTROL_RADIUS_TEXT_FONT = "9p,Helvetica-Bold,red"

MAX_LABEL_POINTS = 80

OBJECTIVE_STYLES = {
    "Drone-Base": {
        "style": "a0.35c",
        "fill": "green",
        "pen": "0.8p,black",
        "label": "Drone Base",
    },
    "Docking": {
        "style": "t0.32c",
        "fill": "orange",
        "pen": "0.8p,black",
        "label": "Docking",
    },
    "FLZ": {
        "style": "d0.30c",
        "fill": "purple",
        "pen": "0.8p,black",
        "label": "FLZ",
    },
    "Restricted_airspace": {
        "style": "s0.28c",
        "fill": "magenta",
        "pen": "0.8p,black",
        "label": "Restricted airspace",
    },
}


# ============================================================
# Utilities
# ============================================================

def ensure_clean_dir(path: Path, clear: bool = False):
    path.mkdir(parents=True, exist_ok=True)

    if clear:
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def make_region(lon, lat, padding=0.003):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    if len(lon) == 0 or len(lat) == 0:
        raise ValueError("Cannot make region because lon/lat arrays are empty.")

    return [
        float(lon.min() - padding),
        float(lon.max() + padding),
        float(lat.min() - padding),
        float(lat.max() + padding),
    ]


def polygon_to_dataframe(poly, name="polygon") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [name] * len(poly),
            "lon": [p[0] for p in poly],
            "lat": [p[1] for p in poly],
        }
    )


def make_objective_short_label(name: str, objective_class: str) -> str:
    """
    Convert objective name to short label with index.

    Examples:
      Drone-Base-01             -> DB01
      Drone-Base-1              -> DB01
      Docking-01                -> DK01
      Docking01                 -> DK01
      FLZ-3                     -> FLZ03
      Restricted_airspace-02    -> RA02
    """
    prefix_map = {
        "Drone-Base": "DB",
        "Docking": "DK",
        "FLZ": "FLZ",
        "Restricted_airspace": "RA",
    }

    short_prefix = prefix_map.get(objective_class, "OBJ")
    name_str = str(name)

    nums = re.findall(r"\d+", name_str)

    if len(nums) > 0:
        idx = int(nums[-1])
        return f"{short_prefix}{idx:02d}"

    return short_prefix


# ============================================================
# Read model
# ============================================================

def read_model_xyz(model_file: Path) -> pd.DataFrame:
    """
    Read selected 2D model.

    Expected format:
      lon lat z slowness

    If more columns exist, only first 4 columns are used.
    """
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = pd.read_csv(
        model_file,
        sep=r"\s+|,",
        engine="python",
        comment="#",
        header=None,
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 4:
        raise ValueError(
            f"Model file must have at least 4 columns: lon lat z slowness\n"
            f"Found {df.shape[1]} columns in {model_file}"
        )

    df = df.iloc[:, :4].copy()
    df.columns = ["lon", "lat", "z", "slowness"]

    for col in ["lon", "lat", "z", "slowness"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lon", "lat", "z", "slowness"]).copy()
    df = df.drop_duplicates(subset=["lon", "lat"], keep="last")

    df["slowness_initial"] = df["slowness"].copy()
    df["slowness_rounded"] = df["slowness"].round(SLOWNESS_ROUND_DECIMALS)

    unique_slow = np.sort(df["slowness_rounded"].unique())

    print("\n========== SLOWNESS UNIQUE CHECK ==========")
    print("Unique slowness values:")
    for val in unique_slow:
        count = int((df["slowness_rounded"] == val).sum())
        print(f"  {val} | nodes: {count:,}")

    df["category_initial"] = 0
    df.loc[df["slowness_rounded"] >= NO_FLY_SLOWNESS_THRESHOLD, "category_initial"] = 1

    df["category"] = df["category_initial"].copy()

    print("\n========== INITIAL CLASSIFICATION ==========")
    print(f"No-fly threshold : {NO_FLY_SLOWNESS_THRESHOLD}")
    print(f"Flyable nodes    : {(df['category_initial'] == 0).sum():,}")
    print(f"No-fly nodes     : {(df['category_initial'] == 1).sum():,}")

    print("\nInitial slowness class summary:")
    for val in unique_slow:
        tmp = df[df["slowness_rounded"] == val]
        n_total = len(tmp)
        n_flyable = int((tmp["category_initial"] == 0).sum())
        n_nofly = int((tmp["category_initial"] == 1).sum())

        print(
            f"  slowness={val} | total={n_total:,} | "
            f"flyable={n_flyable:,} | no-fly={n_nofly:,}"
        )

    return df


# ============================================================
# Read plan data
# ============================================================

def read_one_plan_xyz(path: Path) -> pd.DataFrame:
    """
    Read one plan xyz.

    Preferred format:
      name lon lat elevation

    Also supports:
      lon lat elevation
      lon lat
    """
    df = pd.read_csv(
        path,
        sep=r"\s+|,",
        engine="python",
        comment="#",
        header=None,
        dtype=str,
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] >= 4:
        out = pd.DataFrame(
            {
                "name": df.iloc[:, 0].astype(str),
                "lon": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                "lat": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
                "z": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            }
        )

    elif df.shape[1] == 3:
        out = pd.DataFrame(
            {
                "name": [path.stem] * len(df),
                "lon": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                "lat": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                "z": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
            }
        )

    elif df.shape[1] == 2:
        out = pd.DataFrame(
            {
                "name": [path.stem] * len(df),
                "lon": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                "lat": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                "z": 0.0,
            }
        )

    else:
        raise ValueError(f"Unsupported plan XYZ format: {path}")

    out["source_file"] = path.name
    out["z"] = out["z"].fillna(0.0)
    out = out.dropna(subset=["lon", "lat"]).copy()

    return out


def classify_objective_names(name_series: pd.Series) -> pd.Series:
    """
    Classify plan points by object name prefix.
    """
    out = pd.Series(["Other"] * len(name_series), index=name_series.index, dtype=object)

    name_lower = name_series.astype(str).str.lower()

    out[name_lower.str.startswith("drone-base")] = "Drone-Base"
    out[name_lower.str.startswith("docking")] = "Docking"
    out[name_lower.str.startswith("flz")] = "FLZ"
    out[name_lower.str.startswith("restricted_airspace")] = "Restricted_airspace"

    return out


def read_plan_xyz_files(plan_dir: Path) -> pd.DataFrame:
    if not plan_dir.exists():
        warnings.warn(f"Plan directory does not exist: {plan_dir}")
        return pd.DataFrame(
            columns=["name", "lon", "lat", "z", "source_file", "objective_class"]
        )

    files = sorted(plan_dir.glob("*.xyz"))

    if len(files) == 0:
        warnings.warn(f"No plan XYZ files found in: {plan_dir}")
        return pd.DataFrame(
            columns=["name", "lon", "lat", "z", "source_file", "objective_class"]
        )

    all_data = []

    for f in files:
        try:
            tmp = read_one_plan_xyz(f)
            all_data.append(tmp)
            print(f"[OK] Loaded plan: {f} | points: {len(tmp)}")
        except Exception as e:
            print(f"[WARNING] Cannot read plan file: {f}")
            print(f"          {e}")

    if len(all_data) == 0:
        return pd.DataFrame(
            columns=["name", "lon", "lat", "z", "source_file", "objective_class"]
        )

    plan_df = pd.concat(all_data, ignore_index=True)
    plan_df["objective_class"] = classify_objective_names(plan_df["name"])

    return plan_df


# ============================================================
# Fly-control polygon and radius
# ============================================================

def get_fly_control_points(plan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract polygon/ring points from setup_fly_control.xyz.

    If the file contains one centroid point and many ring points,
    this function tries to remove the centroid automatically.
    """
    if len(plan_df) == 0:
        return pd.DataFrame(columns=plan_df.columns)

    fly_df = plan_df[plan_df["source_file"].isin(FLY_CONTROL_SOURCE_FILES)].copy()

    if len(fly_df) == 0:
        print("\n[WARNING] No fly-control file found.")
        print(f"Expected one of: {FLY_CONTROL_SOURCE_FILES}")
        return fly_df

    fly_df = fly_df.sort_index().copy()

    name_lower = fly_df["name"].astype(str).str.lower()
    mask_centroid_name = (
        name_lower.str.contains("centroid")
        | name_lower.str.contains("center")
        | name_lower.str.contains("centre")
    )

    if mask_centroid_name.any():
        fly_df = fly_df[~mask_centroid_name].copy()

    name_lower = fly_df["name"].astype(str).str.lower()
    mask_ring_name = (
        name_lower.str.contains("ring")
        | name_lower.str.contains("circle")
        | name_lower.str.contains("perimeter")
    )

    if mask_ring_name.any():
        fly_df = fly_df[mask_ring_name].copy()

    if REMOVE_CENTER_POINT_FROM_FLY_CONTROL and len(fly_df) > 20:
        lon = fly_df["lon"].to_numpy(float)
        lat = fly_df["lat"].to_numpy(float)

        lon_med = np.median(lon)
        lat_med = np.median(lat)

        dist = np.sqrt((lon - lon_med) ** 2 + (lat - lat_med) ** 2)

        positive_dist = dist[dist > 0]

        if len(positive_dist) > 0:
            med_dist = np.median(positive_dist)

            keep = dist > 0.2 * med_dist

            if keep.sum() >= 3:
                removed = len(fly_df) - int(keep.sum())

                if removed > 0:
                    print(
                        f"[INFO] Removed {removed} centroid-like point(s) "
                        f"from fly-control polygon."
                    )

                fly_df = fly_df.loc[keep].copy()

    if len(fly_df) < 3:
        print("[WARNING] Fly-control polygon has fewer than 3 points.")
        return pd.DataFrame(columns=plan_df.columns)

    return fly_df


def estimate_fly_control_center_radius(fly_control_df: pd.DataFrame):
    """
    Estimate fly-control center and radius.

    Returns:
      center_lon, center_lat, radius_m
    """
    if len(fly_control_df) < 3:
        return None, None, None

    lon = fly_control_df["lon"].to_numpy(float)
    lat = fly_control_df["lat"].to_numpy(float)

    center_lon = float(np.mean(lon))
    center_lat = float(np.mean(lat))

    _, _, dist_m = GEOD.inv(
        np.full_like(lon, center_lon),
        np.full_like(lat, center_lat),
        lon,
        lat,
    )

    radius_m = float(np.median(dist_m))

    return center_lon, center_lat, radius_m


def find_radius_endpoint(
    fly_control_df: pd.DataFrame,
    center_lon: float,
    center_lat: float,
):
    """
    Find one ring point closest to the median radius.
    Used to draw a radius line.
    """
    lon = fly_control_df["lon"].to_numpy(float)
    lat = fly_control_df["lat"].to_numpy(float)

    _, _, dist_m = GEOD.inv(
        np.full_like(lon, center_lon),
        np.full_like(lat, center_lat),
        lon,
        lat,
    )

    radius_m = np.median(dist_m)
    idx = int(np.argmin(np.abs(dist_m - radius_m)))

    return float(lon[idx]), float(lat[idx])


def force_nofly_by_fly_control(
    model: pd.DataFrame,
    fly_control_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Force all model nodes inside fly-control polygon to no-fly.

    This updates both:
      - category
      - slowness
    """
    model = model.copy()

    model["inside_fly_control"] = False
    model["collision_forced_nofly"] = False
    model["slowness_before_fly_control"] = model["slowness"].copy()

    if len(fly_control_df) < 3:
        print("\n========== FLY-CONTROL COLLISION ==========")
        print("[WARNING] No valid fly-control polygon. Skip collision update.")

        model["slowness_final"] = model["slowness"].copy()
        model["class_name"] = np.where(model["category"] == 1, "no-fly", "flyable")

        return model

    polygon = fly_control_df[["lon", "lat"]].to_numpy(float)

    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    path = MplPath(polygon)

    points = model[["lon", "lat"]].to_numpy(float)
    inside = path.contains_points(points, radius=1e-12)

    model["inside_fly_control"] = inside

    before_nofly = int((model["category"] == 1).sum())

    model.loc[inside, "category"] = 1
    model.loc[inside, "slowness"] = FORCED_NOFLY_SLOWNESS_VALUE

    model["slowness_final"] = model["slowness"].copy()

    after_nofly = int((model["category"] == 1).sum())

    model["collision_forced_nofly"] = (
        (model["category_initial"] == 0) & (model["inside_fly_control"])
    )

    forced_count = int(model["collision_forced_nofly"].sum())

    model["class_name"] = np.where(model["category"] == 1, "no-fly", "flyable")

    print("\n========== FLY-CONTROL COLLISION ==========")
    print(f"Fly-control polygon points : {len(fly_control_df):,}")
    print(f"Model nodes inside control : {int(inside.sum()):,}")
    print(f"No-fly before collision    : {before_nofly:,}")
    print(f"No-fly after collision     : {after_nofly:,}")
    print(f"Newly forced no-fly nodes  : {forced_count:,}")
    print(f"Forced no-fly slowness     : {FORCED_NOFLY_SLOWNESS_VALUE}")

    print("\nFinal slowness class summary:")
    final_unique = np.sort(
        model["slowness_final"].round(SLOWNESS_ROUND_DECIMALS).unique()
    )

    for val in final_unique:
        tmp = model[
            model["slowness_final"].round(SLOWNESS_ROUND_DECIMALS) == val
        ]
        n_total = len(tmp)
        n_flyable = int((tmp["category"] == 0).sum())
        n_nofly = int((tmp["category"] == 1).sum())

        print(
            f"  slowness={val} | total={n_total:,} | "
            f"flyable={n_flyable:,} | no-fly={n_nofly:,}"
        )

    return model


# ============================================================
# Save outputs
# ============================================================

def save_clean_outputs(
    model_before: pd.DataFrame,
    model_after: pd.DataFrame,
    plan_df: pd.DataFrame,
    fly_control_df: pd.DataFrame,
):
    before_out = OUT_DIR / "mixed_model_2d_before_fly_control_classified.xyz"
    after_out = OUT_DIR / "mixed_model_2d_after_fly_control_classified.xyz"

    pathfinding_out = OUT_DIR / "mixed_model_2d_after_fly_control_for_pathfinding.xyz"

    pathfinding_label_out = (
        OUT_DIR / "mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz"
    )

    model_before[
        [
            "lon",
            "lat",
            "z",
            "slowness_initial",
            "slowness",
            "category_initial",
        ]
    ].to_csv(
        before_out,
        sep=" ",
        index=False,
        header=True,
        float_format="%.10f",
    )

    model_after[
        [
            "lon",
            "lat",
            "z",
            "slowness_initial",
            "slowness_before_fly_control",
            "slowness_final",
            "category_initial",
            "inside_fly_control",
            "collision_forced_nofly",
            "category",
            "class_name",
        ]
    ].to_csv(
        after_out,
        sep=" ",
        index=False,
        header=True,
        float_format="%.10f",
    )

    pathfinding_df = model_after[
        [
            "lon",
            "lat",
            "z",
            "slowness_final",
        ]
    ].rename(
        columns={"slowness_final": "slowness"}
    ).copy()

    pathfinding_df.to_csv(
        pathfinding_out,
        sep=" ",
        index=False,
        header=False,
        float_format="%.10f",
    )

    print(f"[OK] Saved before fly-control model : {before_out}")
    print(f"[OK] Saved after fly-control model  : {after_out}")
    print(f"[OK] Saved path-finding xyz model   : {pathfinding_out}")

    labeled_parts = []

    nodes_labeled = pathfinding_df.copy()
    nodes_labeled["label"] = "N"
    labeled_parts.append(nodes_labeled)

    if len(plan_df) > 0:
        objective_classes = [
            "Drone-Base",
            "Docking",
            "FLZ",
            "Restricted_airspace",
        ]

        objective_df = plan_df[
            plan_df["objective_class"].isin(objective_classes)
        ].copy()

        if len(objective_df) > 0:
            objective_labeled = pd.DataFrame(
                {
                    "lon": objective_df["lon"],
                    "lat": objective_df["lat"],
                    "z": objective_df["z"],
                    "slowness": 0.0,
                    "label": [
                        make_objective_short_label(name, obj_class)
                        for name, obj_class in zip(
                            objective_df["name"],
                            objective_df["objective_class"],
                        )
                    ],
                }
            )

            labeled_parts.append(objective_labeled)

    pathfinding_label_df = pd.concat(labeled_parts, ignore_index=True)

    pathfinding_label_df.to_csv(
        pathfinding_label_out,
        sep=" ",
        index=False,
        header=False,
        float_format="%.10f",
    )

    print(f"[OK] Saved labeled path-finding xyz : {pathfinding_label_out}")

    copied_model = OUT_DIR / MODEL_FILE.name
    shutil.copy2(MODEL_FILE, copied_model)
    print(f"[OK] Copied original model: {copied_model}")

    if len(plan_df) > 0:
        plan_out = OUT_DIR / "kml_plan_all_clean_lon_lat.xyz"

        plan_df[
            ["name", "lon", "lat", "z", "source_file", "objective_class"]
        ].to_csv(
            plan_out,
            sep=" ",
            index=False,
            header=True,
            float_format="%.10f",
        )

        print(f"[OK] Saved combined plan: {plan_out}")

        plan_clean_dir = OUT_DIR / "plan_cleaned"
        plan_clean_dir.mkdir(parents=True, exist_ok=True)

        for source_file, group in plan_df.groupby("source_file"):
            out_file = plan_clean_dir / source_file

            group[["name", "lon", "lat", "z"]].to_csv(
                out_file,
                sep=" ",
                index=False,
                header=False,
                float_format="%.10f",
            )

            print(f"[OK] Saved cleaned plan: {out_file}")

    if len(fly_control_df) > 0:
        fly_control_out = OUT_DIR / "fly_control_polygon_used_for_collision.xyz"

        fly_control_df[["name", "lon", "lat", "z", "source_file"]].to_csv(
            fly_control_out,
            sep=" ",
            index=False,
            header=True,
            float_format="%.10f",
        )

        print(f"[OK] Saved fly-control polygon used for collision: {fly_control_out}")

        center_lon, center_lat, radius_m = estimate_fly_control_center_radius(fly_control_df)
        fly_control_info_out = OUT_DIR / "fly_control_center_radius.txt"

        with open(fly_control_info_out, "w", encoding="utf-8") as f:
            f.write(f"center_lon {center_lon:.10f}\n")
            f.write(f"center_lat {center_lat:.10f}\n")
            f.write(f"radius_m {radius_m:.3f}\n")
            f.write(f"radius_km {radius_m / 1000.0:.3f}\n")

        print(f"[OK] Saved fly-control center/radius: {fly_control_info_out}")

    hoalac_out = OUT_DIR / "hoalac_polygon.xyz"
    hoalac_df = polygon_to_dataframe(HOALAC_POLYGON, name="Hoa_Lac_polygon")
    hoalac_df.to_csv(
        hoalac_out,
        sep=" ",
        index=False,
        header=True,
        float_format="%.10f",
    )
    print(f"[OK] Saved Hoa Lac polygon: {hoalac_out}")


# ============================================================
# Plot helpers
# ============================================================

def plot_flyable_nofly_nodes(fig: pygmt.Figure, model: pd.DataFrame):
    flyable = model[model["category"] == 0]
    nofly = model[model["category"] == 1]

    if len(flyable) > 0:
        fig.plot(
            x=flyable["lon"],
            y=flyable["lat"],
            style=MODEL_NODE_STYLE,
            fill=FLYABLE_COLOR,
            pen=None,
            label="Flyable",
        )

    if len(nofly) > 0:
        fig.plot(
            x=nofly["lon"],
            y=nofly["lat"],
            style=MODEL_NODE_STYLE,
            fill=NOFLY_COLOR,
            pen=None,
            label="No-fly",
        )


def plot_hoalac_polygon(fig: pygmt.Figure):
    hoalac_df = polygon_to_dataframe(HOALAC_POLYGON, name="Hoa_Lac_polygon")

    fig.plot(
        x=hoalac_df["lon"],
        y=hoalac_df["lat"],
        pen=HOALAC_POLYGON_PEN,
        fill=HOALAC_POLYGON_FILL,
        label="Hoa Lac polygon",
    )


def plot_fly_control_polygon(fig: pygmt.Figure, fly_control_df: pd.DataFrame):
    """
    Plot fly-control circle/polygon, center, radius line, and radius label.

    Used only in:
      mixed_model_2d_after_fly_control_collision_with_plan.png
    """
    if len(fly_control_df) < 3:
        return

    poly = fly_control_df.copy()

    if not (
        np.isclose(poly.iloc[0]["lon"], poly.iloc[-1]["lon"])
        and np.isclose(poly.iloc[0]["lat"], poly.iloc[-1]["lat"])
    ):
        poly = pd.concat([poly, poly.iloc[[0]]], ignore_index=True)

    fig.plot(
        x=poly["lon"],
        y=poly["lat"],
        pen=FLY_CONTROL_LINE_PEN,
        fill=FLY_CONTROL_FILL,
        label="Fly-control no-fly zone",
    )

    center_lon, center_lat, radius_m = estimate_fly_control_center_radius(fly_control_df)

    if center_lon is None or center_lat is None or radius_m is None:
        return

    radius_km = radius_m / 1000.0

    end_lon, end_lat = find_radius_endpoint(
        fly_control_df,
        center_lon=center_lon,
        center_lat=center_lat,
    )

    fig.plot(
        x=[center_lon, end_lon],
        y=[center_lat, end_lat],
        pen=FLY_CONTROL_RADIUS_LINE_PEN,
        label="Fly-control radius",
    )

    fig.plot(
        x=[center_lon],
        y=[center_lat],
        style=FLY_CONTROL_CENTER_STYLE,
        fill=FLY_CONTROL_CENTER_FILL,
        pen=FLY_CONTROL_CENTER_PEN,
        label="Fly-control center",
    )

    mid_lon = 0.5 * (center_lon + end_lon)
    mid_lat = 0.5 * (center_lat + end_lat)

    fig.text(
        x=mid_lon,
        y=mid_lat,
        text=f"R = {radius_km:.2f} km",
        font=FLY_CONTROL_RADIUS_TEXT_FONT,
        justify="CM",
        offset="0.1c/0.1c",
        fill="white@20",
        pen="0.4p,red",
    )


def plot_objectives(fig: pygmt.Figure, plan_df: pd.DataFrame):
    if len(plan_df) == 0:
        return

    for obj_class, style in OBJECTIVE_STYLES.items():
        sub = plan_df[plan_df["objective_class"] == obj_class]

        if len(sub) == 0:
            continue

        fig.plot(
            x=sub["lon"],
            y=sub["lat"],
            style=style["style"],
            fill=style["fill"],
            pen=style["pen"],
            label=style["label"],
        )

        if len(sub) <= MAX_LABEL_POINTS:
            for _, row in sub.iterrows():
                fig.text(
                    x=row["lon"],
                    y=row["lat"],
                    text=str(row["name"]),
                    font="7p,Helvetica,black",
                    justify="LM",
                    offset="0.08c/0.05c",
                    fill="white@25",
                    pen="0.5p,black",
                )


# ============================================================
# Plot figures
# ============================================================

def plot_model_before_collision(model_before: pd.DataFrame):
    """
    Original model-only plot before fly-control collision.
    """
    model_plot = model_before.copy()
    model_plot["category"] = model_plot["category_initial"]

    region = make_region(
        model_plot["lon"].values,
        model_plot["lat"].values,
        padding=REGION_PADDING,
    )

    print(f"[INFO] Before-collision model-only region: {region}")

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
        FONT_LABEL="11p",
        FONT_ANNOT_PRIMARY="9p",
        FONT_TITLE="13p,Helvetica-Bold",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            'WSen+t"2D model before fly-control collision"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    plot_flyable_nofly_nodes(fig, model_plot)

    fig.legend(
        position="JTL+jTL+o0.2c",
        box="+gwhite+p0.5p",
    )

    fig_file = FIG_DIR / "mixed_model_2d_before_fly_control_collision_model_only.png"
    fig.savefig(fig_file, dpi=DPI)

    print(f"[OK] Saved before-collision figure: {fig_file}")


def plot_model_only_after_collision(
    model_after: pd.DataFrame,
    plan_df: pd.DataFrame,
):
    """
    Zoomed model-only style figure after fly-control collision.

    This figure does NOT plot the fly-control circle.
    The fly-control effect is represented by red no-fly nodes.
    """
    region = make_region(
        model_after["lon"].values,
        model_after["lat"].values,
        padding=REGION_PADDING,
    )

    print(f"[INFO] Collision model-only zoom region: {region}")

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
        FONT_LABEL="11p",
        FONT_ANNOT_PRIMARY="9p",
        FONT_TITLE="13p,Helvetica-Bold",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            'WSen+t"2D model after fly-control collision"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    plot_flyable_nofly_nodes(fig, model_after)
    plot_objectives(fig, plan_df)

    fig.legend(
        position="JTL+jTL+o0.2c",
        box="+gwhite+p0.5p",
    )

    fig_file = FIG_DIR / "mixed_model_2d_after_fly_control_collision_zoom.png"
    fig.savefig(fig_file, dpi=DPI)

    print(f"[OK] Saved collision zoom figure: {fig_file}")


def plot_model_with_plan_after_collision(
    model_after: pd.DataFrame,
    plan_df: pd.DataFrame,
    fly_control_df: pd.DataFrame,
):
    """
    Full map with model, Hoa Lac polygon, fly-control circle, and objectives.

    Important:
      - Fly-control circle IS plotted here.
      - Fly-control center IS plotted here.
      - Fly-control radius IS plotted here.
      - Path-connected lines from plan files are NOT plotted.
      - Hoa Lac polygon is plotted.
    """
    hoalac_df = polygon_to_dataframe(HOALAC_POLYGON, name="Hoa_Lac_polygon")

    lon_parts = [model_after["lon"].values, hoalac_df["lon"].values]
    lat_parts = [model_after["lat"].values, hoalac_df["lat"].values]

    if len(plan_df) > 0:
        lon_parts.append(plan_df["lon"].values)
        lat_parts.append(plan_df["lat"].values)

    if len(fly_control_df) > 0:
        lon_parts.append(fly_control_df["lon"].values)
        lat_parts.append(fly_control_df["lat"].values)

    all_lon = np.concatenate(lon_parts)
    all_lat = np.concatenate(lat_parts)

    region = make_region(all_lon, all_lat, padding=REGION_PADDING)

    print(f"[INFO] Full map after collision region: {region}")

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
        FONT_LABEL="11p",
        FONT_ANNOT_PRIMARY="9p",
        FONT_TITLE="13p,Helvetica-Bold",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            'WSen+t"2D path-finding model after fly-control collision"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    plot_hoalac_polygon(fig)
    plot_fly_control_polygon(fig, fly_control_df)

    plot_flyable_nofly_nodes(fig, model_after)

    plot_objectives(fig, plan_df)

    fig.legend(
        position="JTL+jTL+o0.2c",
        box="+gwhite+p0.5p",
    )

    fig_file = FIG_DIR / "mixed_model_2d_after_fly_control_collision_with_plan.png"
    fig.savefig(fig_file, dpi=DPI)

    print(f"[OK] Saved full collision figure: {fig_file}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_clean_dir(OUT_DIR, clear=CLEAR_OUTPUT)
    ensure_clean_dir(FIG_DIR, clear=CLEAR_OUTPUT)

    print("========== INPUT ==========")
    print(f"Model file : {MODEL_FILE}")
    print(f"Plan dir   : {PLAN_DIR}")

    print("\n========== LOAD MODEL ==========")
    model_before = read_model_xyz(MODEL_FILE)

    print(f"Model nodes: {len(model_before):,}")
    print("Model columns:", list(model_before.columns))

    print("\n========== LOAD PLAN DATA ==========")
    plan_df = read_plan_xyz_files(PLAN_DIR)
    print(f"Plan points: {len(plan_df):,}")

    if len(plan_df) > 0:
        print("\n========== OBJECTIVE SUMMARY ==========")
        print(plan_df["objective_class"].value_counts())

    print("\n========== EXTRACT FLY-CONTROL POLYGON ==========")
    fly_control_df = get_fly_control_points(plan_df)
    print(f"Fly-control points used for collision: {len(fly_control_df):,}")

    if len(fly_control_df) >= 3:
        center_lon, center_lat, radius_m = estimate_fly_control_center_radius(fly_control_df)
        print("\n========== FLY-CONTROL CENTER / RADIUS ==========")
        print(f"Center lon : {center_lon:.8f}")
        print(f"Center lat : {center_lat:.8f}")
        print(f"Radius     : {radius_m / 1000.0:.3f} km")

    print("\n========== APPLY FLY-CONTROL COLLISION ==========")
    model_after = force_nofly_by_fly_control(model_before, fly_control_df)

    print("\n========== SAVE CLEANED OUTPUT ==========")
    save_clean_outputs(
        model_before=model_before,
        model_after=model_after,
        plan_df=plan_df,
        fly_control_df=fly_control_df,
    )

    print("\n========== PLOT BEFORE COLLISION ==========")
    plot_model_before_collision(model_before)

    print("\n========== PLOT AFTER COLLISION ZOOM ==========")
    plot_model_only_after_collision(
        model_after=model_after,
        plan_df=plan_df,
    )

    print("\n========== PLOT AFTER COLLISION FULL MAP ==========")
    plot_model_with_plan_after_collision(
        model_after=model_after,
        plan_df=plan_df,
        fly_control_df=fly_control_df,
    )

    print("\n========== DONE ==========")
    print(f"Output folder : {OUT_DIR}")
    print(f"Figure folder : {FIG_DIR}")


if __name__ == "__main__":
    main()