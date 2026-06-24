#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/plotting.py

Plotting utilities for Scenario 1 path finding.

Important:
    This file must NOT import from src.plotting.
    main.py imports functions from this file.

Includes:
    - initiate model plot
    - path report plot
    - multiple ranked path plot
    - traveltime histogram
    - cost/slowness diagnostic plots
    - side-by-side input model vs slowness model
    - path-corridor zoom diagnostic for adjacent-node checking
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import pygmt


# ============================================================
# Binary flyable/no-fly plot style
# ============================================================

# Keep no-fly cells visually distinct from high slowness/cost classes.
# GMT accepts names such as "black@15"; CPT uses RGB strings.
FLYABLE_GMT_FILL = "seagreen"
FLYABLE_MPL_COLOR = "#79c79a"
NO_FLY_GMT_FILL = "black"
NO_FLY_GMT_FILL_LIGHT = "black@15"
NO_FLY_GMT_PEN = "0.15p,black"
NO_FLY_GMT_PEN_THIN = "0.10p,black"
NO_FLY_RGB = "0/0/0"
NO_FLY_MPL_COLOR = "black"


# ============================================================
# Basic helpers
# ============================================================

def detect_lonlat(model: pd.DataFrame) -> bool:
    """Return True if x/y look like longitude/latitude degrees."""
    if model is None or len(model) == 0:
        return False

    x = pd.to_numeric(model["x"], errors="coerce")
    y = pd.to_numeric(model["y"], errors="coerce")

    if x.dropna().empty or y.dropna().empty:
        return False

    return (
        x.dropna().between(-180.0, 180.0).all()
        and y.dropna().between(-90.0, 90.0).all()
    )


def alpha_to_transparency(alpha: float) -> int:
    """Convert matplotlib-like alpha to GMT transparency percentage."""
    alpha = max(0.0, min(1.0, float(alpha)))
    return int(round((1.0 - alpha) * 100.0))


def ensure_label_prefix(model: pd.DataFrame) -> pd.DataFrame:
    """Make sure label_prefix exists."""
    out = model.copy()
    if "label_prefix" not in out.columns:
        if "label" in out.columns:
            out["label_prefix"] = (
                out["label"].astype(str).str.extract(r"^([A-Za-z_]+)", expand=False).fillna("")
            )
        else:
            out["label_prefix"] = ""
    return out


def sample_model_for_plot(
    model: pd.DataFrame,
    max_model_points: int = 300000,
) -> pd.DataFrame:
    """Sample model for faster plotting while preserving special nodes."""
    model = ensure_label_prefix(model)

    if len(model) <= int(max_model_points):
        return model.copy()

    special_prefixes = {"DB", "DK", "FLZ", "RA"}
    special = model[model["label_prefix"].astype(str).str.upper().isin(special_prefixes)].copy()
    normal = model[~model.index.isin(special.index)].copy()

    remain = max(int(max_model_points) - len(special), 1000)
    if len(normal) > remain:
        normal = normal.sample(n=remain, random_state=42)

    out = pd.concat([normal, special], axis=0).sort_index()
    return out


def get_xy_region_with_padding(
    model: pd.DataFrame,
    path_df: pd.DataFrame | None = None,
    padding_ratio: float = 0.04,
) -> list[float]:
    """Get [xmin, xmax, ymin, ymax] with padding."""
    frames = []
    if model is not None and len(model) > 0:
        frames.append(model[["x", "y"]])
    if path_df is not None and len(path_df) > 0:
        frames.append(path_df[["x", "y"]])

    if not frames:
        raise ValueError("No x/y points available for region calculation.")

    xy = pd.concat(frames, axis=0, ignore_index=True)
    x = pd.to_numeric(xy["x"], errors="coerce")
    y = pd.to_numeric(xy["y"], errors="coerce")

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 or not np.isfinite(dx):
        dx = 1e-6 if detect_lonlat(xy.rename(columns={"x": "x", "y": "y"})) else 1.0
    if dy <= 0 or not np.isfinite(dy):
        dy = 1e-6 if detect_lonlat(xy.rename(columns={"x": "x", "y": "y"})) else 1.0

    padx = dx * float(padding_ratio)
    pady = dy * float(padding_ratio)

    return [xmin - padx, xmax + padx, ymin - pady, ymax + pady]


def get_profile_region(x, y, y_padding_ratio: float = 0.12) -> list[float]:
    """Get profile plot region."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]

    if len(finite_x) == 0:
        finite_x = np.array([0.0, 1.0])
    if len(finite_y) == 0:
        finite_y = np.array([0.0, 1.0])

    xmin = float(np.nanmin(finite_x))
    xmax = float(np.nanmax(finite_x))
    ymin = float(np.nanmin(finite_y))
    ymax = float(np.nanmax(finite_y))

    if np.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0

    dy = ymax - ymin
    if dy <= 0 or not np.isfinite(dy):
        dy = max(abs(ymin) * 0.1, 1.0)

    ymin -= dy * float(y_padding_ratio)
    ymax += dy * float(y_padding_ratio)

    return [xmin, xmax, ymin, ymax]


def _meters_to_lonlat_delta(lat_degree: float, distance_m: float) -> tuple[float, float]:
    """Approximate lon/lat degree deltas for a distance in meters."""
    lat_rad = math.radians(float(lat_degree))
    cos_lat = max(abs(math.cos(lat_rad)), 1e-8)
    dy = float(distance_m) / 111_320.0
    dx = float(distance_m) / (111_320.0 * cos_lat)
    return dx, dy


def _segment_distance_m(model: pd.DataFrame, idx_a: int, idx_b: int, use_z: bool = True) -> float:
    """Distance between two model rows in meters."""
    x1 = float(model.loc[int(idx_a), "x"])
    y1 = float(model.loc[int(idx_a), "y"])
    x2 = float(model.loc[int(idx_b), "x"])
    y2 = float(model.loc[int(idx_b), "y"])

    if detect_lonlat(model):
        lat0 = math.radians(0.5 * (y1 + y2))
        dx = (x2 - x1) * 111_320.0 * math.cos(lat0)
        dy = (y2 - y1) * 110_540.0
    else:
        dx = x2 - x1
        dy = y2 - y1

    dz = 0.0
    if use_z and "z" in model.columns:
        try:
            dz = float(model.loc[int(idx_b), "z"] - model.loc[int(idx_a), "z"])
        except Exception:
            dz = 0.0

    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


# ============================================================
# Flyable / no-fly classification
# ============================================================

def classify_flyable_nofly(
    model: pd.DataFrame,
    no_fly_prefixes=(),
    no_fly_slowness_threshold: float = 10.0,
    always_flyable_prefixes=(),
):
    """Classify nodes using slowness threshold and optional label prefixes."""
    model = ensure_label_prefix(model)

    no_fly_prefixes = tuple(str(p).upper() for p in (no_fly_prefixes or ()))
    always_flyable_prefixes = tuple(str(p).upper() for p in (always_flyable_prefixes or ()))

    prefix = model["label_prefix"].astype(str).str.upper()
    slow = pd.to_numeric(model["slowness"], errors="coerce").fillna(np.inf)

    prefix_mask = prefix.isin(no_fly_prefixes) if no_fly_prefixes else np.zeros(len(model), dtype=bool)
    slow_mask = slow >= float(no_fly_slowness_threshold)
    always_mask = prefix.isin(always_flyable_prefixes) if always_flyable_prefixes else np.zeros(len(model), dtype=bool)

    nofly_mask = (np.asarray(prefix_mask, dtype=bool) | np.asarray(slow_mask, dtype=bool)) & (~np.asarray(always_mask, dtype=bool))
    flyable_mask = ~nofly_mask

    return model[flyable_mask].copy(), model[nofly_mask].copy()


def format_slowness_value(value: float) -> str:
    """Nice slowness formatting."""
    value = float(value)
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.1e}"
    if abs(value) < 1:
        return f"{value:.3g}"
    if value.is_integer():
        return f"{value:.0f}"
    return f"{value:.3g}"


def get_representative_slowness_text(df: pd.DataFrame) -> str:
    """Return compact representative slowness text for legend."""
    if df is None or df.empty or "slowness" not in df.columns:
        return "N/A"

    slow = pd.to_numeric(df["slowness"], errors="coerce")
    slow = slow[np.isfinite(slow)]

    if len(slow) == 0:
        return "N/A"

    smin = float(slow.min())
    smax = float(slow.max())
    if np.isclose(smin, smax):
        return format_slowness_value(smin)
    return f"{format_slowness_value(smin)}-{format_slowness_value(smax)}"


def plot_model_flyable_nofly(
    fig: pygmt.Figure,
    model: pd.DataFrame,
    model_marker_size: float = 2.0,
    model_alpha: float = 0.45,
    no_fly_prefixes=(),
    no_fly_slowness_threshold: float = 10.0,
    show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
):
    """Plot binary flyable/no-fly nodes."""
    model = ensure_label_prefix(model)
    flyable, nofly = classify_flyable_nofly(
        model=model,
        no_fly_prefixes=no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    flyable_slow_text = get_representative_slowness_text(flyable)
    nofly_slow_text = get_representative_slowness_text(nofly)

    if not flyable.empty:
        fig.plot(
            x=flyable["x"],
            y=flyable["y"],
            style=f"c{max(model_marker_size / 35.0, 0.03):.3f}c",
            fill=f"{FLYABLE_GMT_FILL}@{alpha_to_transparency(model_alpha)}",
            pen=None,
            label=f"Flyable: s={flyable_slow_text}",
        )

    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style=f"s{max(model_marker_size / 28.0, 0.05):.3f}c",
            fill=NO_FLY_GMT_FILL_LIGHT,
            pen=NO_FLY_GMT_PEN,
            label=f"No-fly: s={nofly_slow_text}",
        )

    if show_flz_overlay:
        flz = model[model["label_prefix"].astype(str).str.upper() == "FLZ"].copy()
        if not flz.empty:
            fig.plot(
                x=flz["x"],
                y=flz["y"],
                style=f"c{max(model_marker_size / 18.0, 0.06):.3f}c",
                fill="orange@25",
                pen="0.2p,black",
                label="FLZ",
            )


def _plot_start_end_markers(
    fig: pygmt.Figure,
    model: pd.DataFrame,
    start_idx: int | None = None,
    end_idx: int | None = None,
    label: bool = True,
):
    """Plot start/end markers if indices are available."""
    if start_idx is None or end_idx is None:
        return

    start = model.loc[int(start_idx)]
    end = model.loc[int(end_idx)]

    fig.plot(
        x=[start["x"]],
        y=[start["y"]],
        style="a0.45c",
        fill="yellow",
        pen="0.7p,black",
        label=f"Start: {start['label']}",
        transparency=50,
    )
    fig.plot(
        x=[end["x"]],
        y=[end["y"]],
        style="s0.42c",
        fill="grey@50",
        pen="1.4p,blue",
        label=f"End: {end['label']}",
        transparency=50,
    )

    if label:
        fig.text(
            x=[start["x"], end["x"]],
            y=[start["y"], end["y"]],
            text=[str(start["label"]), str(end["label"])],
            font="9p,Helvetica-Bold,black",
            justify="LM",
            offset="0.15c/0.15c",
        )


# ============================================================
# Path table helpers
# ============================================================

def add_path_traveltime_columns(
    model: pd.DataFrame,
    path_df: pd.DataFrame,
    path_indices: list[int],
) -> pd.DataFrame:
    """Add segment/cumulative distance and traveltime columns."""
    out = path_df.copy()
    indices = [int(i) for i in path_indices]

    segment_distance_m = np.zeros(len(indices), dtype=float)
    segment_traveltime_s = np.zeros(len(indices), dtype=float)

    slow_values = pd.to_numeric(model.loc[indices, "slowness"], errors="coerce").to_numpy(float)
    positive_s = slow_values[np.isfinite(slow_values) & (slow_values > 0)]
    min_positive_s = float(np.min(positive_s)) if len(positive_s) else 1.0

    for i in range(1, len(indices)):
        idx0 = indices[i - 1]
        idx1 = indices[i]
        dist_m = _segment_distance_m(model, idx0, idx1, use_z=True)

        s0 = float(model.loc[idx0, "slowness"])
        s1 = float(model.loc[idx1, "slowness"])
        if s0 <= 0 or not np.isfinite(s0):
            s0 = min_positive_s
        if s1 <= 0 or not np.isfinite(s1):
            s1 = min_positive_s

        segment_distance_m[i] = dist_m
        segment_traveltime_s[i] = dist_m * 0.5 * (s0 + s1)

    out["segment_distance_m"] = segment_distance_m
    out["cumulative_distance_m"] = np.cumsum(segment_distance_m)
    out["cumulative_distance_km"] = out["cumulative_distance_m"] / 1000.0
    out["segment_traveltime_s"] = segment_traveltime_s
    out["cumulative_traveltime_s"] = np.cumsum(segment_traveltime_s)
    out["cumulative_traveltime_min"] = out["cumulative_traveltime_s"] / 60.0
    return out


def add_report_text_box_to_profile(
    fig: pygmt.Figure,
    region: list[float],
    algorithm_name: str,
    path_df: pd.DataFrame,
    result: dict | None = None,
):
    """Add compact report text to the profile panel."""
    if result is None:
        result = {}

    xmin, xmax, ymin, ymax = region
    path_nodes = len(path_df)

    distance_km = result.get("output_path_distance_km", None)
    traveltime_min = result.get("output_estimated_traveltime_min", None)
    runtime_s = result.get("runtime_seconds", None)
    expanded_nodes = result.get("expanded_nodes", None)

    if distance_km is None and "cumulative_distance_km" in path_df.columns:
        distance_km = float(path_df["cumulative_distance_km"].iloc[-1])
    if traveltime_min is None and "cumulative_traveltime_min" in path_df.columns:
        traveltime_min = float(path_df["cumulative_traveltime_min"].iloc[-1])

    text0 = f"Algorithm: {algorithm_name} | Path nodes: {path_nodes:,} | Distance: {float(distance_km):.3f} km"
    text1 = (
        f"Travel time: {float(traveltime_min):.2f} min | "
        f"Expanded: {int(expanded_nodes):,}" if expanded_nodes is not None else
        f"Travel time: {float(traveltime_min):.2f} min | Expanded: N/A"
    )
    if runtime_s is not None:
        text1 += f" | Runtime: {float(runtime_s):.3f} s"

    x_text = xmax - 0.15 * (xmax - xmin)
    y_text0 = ymax - 0.10 * (ymax - ymin)
    y_text1 = ymax - 0.22 * (ymax - ymin)

    for text, y_text in [(text0, y_text0), (text1, y_text1)]:
        fig.text(
            x=x_text,
            y=y_text,
            text=text,
            font="8p,Helvetica-Bold,black",
            justify="TR",
            fill="white@50",
            pen="0.4p,black",
            clearance="0.10c/0.10c",
        )

# ============================================================
# Clean up the connector
# ============================================================

def cleanup_plot_temp_files(*paths):
    """
    Remove temporary CPT/GRD files and possible GMT sidecar files.
    """
    for path in paths:
        try:
            path = Path(path)

            # Remove exact file
            if path.exists():
                path.unlink()

            # Remove possible sidecar files with same base name
            parent = path.parent
            stem = path.stem
            for extra in parent.glob(stem + "*"):
                if extra.is_file() and extra.suffix.lower() in {
                    ".cpt", ".grd", ".nc", ".tmp", ".xyz", ".dat"
                }:
                    try:
                        extra.unlink()
                    except Exception:
                        pass

        except Exception as exc:
            print(f"[WARNING] Could not cleanup temporary file {path}. Reason: {exc}") 

# ============================================================
# Main path report
# ============================================================

def _parse_spacing_xy(spacing) -> tuple[float, float]:
    """
    Parse GMT spacing value.

    Examples:
        20 -> (20, 20)
        "20" -> (20, 20)
        "0.00018/0.00018" -> (0.00018, 0.00018)
    """
    if isinstance(spacing, (int, float)):
        dx = dy = float(spacing)
    else:
        text = str(spacing).strip()
        if "/" in text:
            a, b = text.split("/", 1)
            dx = float(a)
            dy = float(b)
        else:
            dx = dy = float(text)

    if dx <= 0 or dy <= 0:
        raise ValueError(f"Invalid GMT spacing: {spacing}")

    return dx, dy


def _adjust_region_to_spacing(region, spacing) -> list[float]:
    """
    Expand region so GMT surface accepts the selected spacing.

    GMT surface requires:
        xmax - xmin = NX * dx
        ymax - ymin = NY * dy

    This helper keeps xmin/ymin fixed and expands xmax/ymax outward.
    """
    xmin, xmax, ymin, ymax = [float(v) for v in region]
    dx, dy = _parse_spacing_xy(spacing)

    width = max(xmax - xmin, dx)
    height = max(ymax - ymin, dy)

    nx = max(1, int(np.ceil((width / dx) - 1e-12)))
    ny = max(1, int(np.ceil((height / dy) - 1e-12)))

    new_xmax = xmin + nx * dx
    new_ymax = ymin + ny * dy

    return [xmin, new_xmax, ymin, new_ymax]


def plot_path_report(
    model: pd.DataFrame,
    path_indices: list[int],
    figure_file: Path,
    algorithm_name: str,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 2.0,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=(),
    plot_no_fly_slowness_threshold: float = 10.0,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
    result: dict | None = None,
    plot_surface: bool = False,
    surface_spacing_m: float = 20.0,
    surface_alpha: int = 0,
    show_map_colorbar: bool = False,
    cleanup_temp: bool = True,
    no_fly_slowness_threshold: float | None = None,
    slowness_discrete_bounds=None,
    filled_cell_marker_size: float | None = None,
):
    """
    Plot path report:
        1. map with filled slowness/cost background
        2. path slowness profile
        3. cumulative traveltime profile

    Safe version:
        - Does NOT call pygmt.surface().
        - Uses slowness-colored square cells instead.
        - This avoids GMT native crashes such as:
              free(): invalid next size
              Aborted (core dumped)

    Notes:
        - plot_surface, surface_spacing_m, and surface_alpha are kept in the
          function signature only for compatibility with existing main.py calls.
        - No-fly nodes are still overlaid explicitly as black squares.
    """

    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if len(path_indices) == 0:
        raise ValueError("Cannot plot empty path.")

    model = ensure_label_prefix(model)
    path_indices = [int(i) for i in path_indices]

    if no_fly_slowness_threshold is None:
        no_fly_threshold = float(plot_no_fly_slowness_threshold)
    else:
        no_fly_threshold = float(no_fly_slowness_threshold)
        plot_no_fly_slowness_threshold = float(no_fly_slowness_threshold)

    path_df = model.loc[path_indices].copy().reset_index(drop=False)

    if "node_index" in path_df.columns:
        path_df = path_df.rename(columns={"node_index": "original_node_index"})
    elif "index" in path_df.columns:
        path_df = path_df.rename(columns={"index": "original_node_index"})

    path_df["path_step"] = np.arange(len(path_df))
    path_df = add_path_traveltime_columns(
        model=model,
        path_df=path_df,
        path_indices=path_indices,
    )

    plot_model = sample_model_for_plot(
        model,
        max_model_points=max_model_points,
    )

    region = get_xy_region_with_padding(
        plot_model,
        path_df,
        padding_ratio=0.04,
    )

    is_lonlat = detect_lonlat(plot_model)

    # ------------------------------------------------------------
    # CPT for slowness/cost map background
    # ------------------------------------------------------------
    cpt_file = figure_file.with_suffix(".path_report_slowness_discrete.cpt")
    cpt_result = make_discrete_slowness_cpt(
        cpt_file=cpt_file,
        slowness_values=plot_model["slowness"],
        bounds=slowness_discrete_bounds,
        cmap="turbo",
        vmin=0.0,
        vmax=float(no_fly_threshold),
        no_fly_threshold=float(no_fly_threshold),
        n_steps=50,
        cpt_mode="data_nonuniform",
        round_decimals=4,
        max_bounds=120,
    )

    # Compatible with both helper versions:
    #   new: returns (cpt_file, bounds)
    #   old: returns cpt_file only
    if isinstance(cpt_result, tuple):
        cpt_file, bounds = cpt_result
    else:
        cpt_file = Path(cpt_result)
        bounds = None

    fig = pygmt.Figure()

    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    # ============================================================
    # Map panel
    # ============================================================
    fig.basemap(
        region=region,
        projection="M16c",
        frame=[
            f"WSne+tScenario 1 path report - {algorithm_name}",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    # ------------------------------------------------------------
    # Filled background using square slowness cells.
    # This replaces pygmt.surface() because surface can crash GMT when
    # -R/-I are not perfectly compatible.
    # ------------------------------------------------------------
    if filled_cell_marker_size is None:
        # Larger than the old point marker to make a filled raster-like map.
        cell_size = max(float(model_marker_size) / 30.0, 0.05)
    else:
        cell_size = float(filled_cell_marker_size)

    plot_slowness = pd.to_numeric(plot_model["slowness"], errors="coerce")
    valid = (
        pd.to_numeric(plot_model["x"], errors="coerce").notna()
        & pd.to_numeric(plot_model["y"], errors="coerce").notna()
        & plot_slowness.notna()
    )
    cell_df = plot_model.loc[valid].copy()
    cell_slowness = plot_slowness.loc[valid]

    if not cell_df.empty:
        fig.plot(
            x=cell_df["x"],
            y=cell_df["y"],
            style=f"s{cell_size:.3f}c",
            fill=cell_slowness,
            cmap=str(cpt_file),
            pen=None,
        )

    # Optional smaller node overlay. This keeps node locations readable.
    # Set model_marker_size very small if you want only the filled-cell look.
    if model_marker_size > 0:
        _plot_slowness_points_with_discrete_cpt(
            fig=fig,
            plot_model=plot_model,
            cpt_file=cpt_file,
            marker_size=max(float(model_marker_size) * 0.75, 0.5),
        )

    # Overlay no-fly nodes explicitly
    _, nofly = classify_flyable_nofly(
        plot_model,
        no_fly_prefixes=plot_no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_threshold,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style=f"s{max(cell_size * 0.95, 0.06):.3f}c",
            fill=NO_FLY_GMT_FILL,
            pen=NO_FLY_GMT_PEN_THIN,
            label=f"No-fly: s >= {no_fly_threshold:g}",
        )

    # Optional FLZ overlay
    if plot_show_flz_overlay:
        flz = plot_model[
            plot_model["label_prefix"].astype(str).str.upper() == "FLZ"
        ].copy()

        if not flz.empty:
            fig.plot(
                x=flz["x"],
                y=flz["y"],
                style=f"c{max(model_marker_size / 18.0, 0.06):.3f}c",
                fill="orange@25",
                pen="0.2p,black",
                label="FLZ",
            )

    # Optional map colorbar.
    # Avoid frame with equalsize because some GMT versions reject -L + -B.
    if show_map_colorbar:
        try:
            fig.colorbar(
                cmap=str(cpt_file),
                position="JBC+w8c/0.25c+o0c/0.65c+h",
                equalsize="0.18c",
            )
        except Exception as exc:
            print(f"[WARNING] Could not draw path-report map colorbar. Reason: {exc}")

    # Path line and path nodes
    fig.plot(
        x=path_df["x"],
        y=path_df["y"],
        pen=f"{path_line_width}p,black",
        label=f"Path - {algorithm_name}",
    )

    fig.plot(
        x=path_df["x"],
        y=path_df["y"],
        style="c0.10c",
        fill="green",
        pen="0.1p,black",
    )

    _plot_start_end_markers(
        fig,
        model,
        path_indices[0],
        path_indices[-1],
        label=False,
    )

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    try:
        fig.legend(
            position="JTL+jTL+o0.15c/0.15c",
            box="+gwhite@25+p0.5p,black",
        )
    except Exception:
        pass

    # ============================================================
    # Slowness profile panel
    # ============================================================
    fig.shift_origin(xshift="16.5c", yshift="7.4c")

    slow_region = get_profile_region(
        path_df["path_step"].values,
        path_df["slowness"].values,
        y_padding_ratio=0.12,
    )

    fig.basemap(
        region=slow_region,
        projection="X10c/4.0c",
        frame=[
            "wSnE+tPath slowness profile",
            "xaf+lPath step",
            "yaf+lSlowness",
        ],
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["slowness"],
        pen="1.2p,black,--",
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["slowness"],
        style="c0.15c",
        fill="green",
        pen="0.1p,black",
    )

    add_report_text_box_to_profile(
        fig,
        slow_region,
        algorithm_name,
        path_df,
        result=result,
    )

    # ============================================================
    # Cumulative traveltime profile panel
    # ============================================================
    fig.shift_origin(yshift="-7.4c")

    time_region = get_profile_region(
        path_df["path_step"].values,
        path_df["cumulative_traveltime_s"].values,
        y_padding_ratio=0.12,
    )

    fig.basemap(
        region=time_region,
        projection="X10c/4.0c",
        frame=[
            "wSnE+tCumulative traveltime profile",
            "xaf+lPath step",
            "yaf+lTraveltime (s)",
        ],
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["cumulative_traveltime_s"],
        pen="1.2p,black,--",
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["cumulative_traveltime_s"],
        style="c0.15c",
        fill="green",
        pen="0.1p,black",
    )

    # ============================================================
    # Optional altitude profile
    # ============================================================
    if "z" in model.columns and model["z"].nunique() > 1:
        fig.shift_origin(yshift="-4.2c")

        z_region = get_profile_region(
            path_df["path_step"].values,
            path_df["z"].values,
            y_padding_ratio=0.12,
        )

        fig.basemap(
            region=z_region,
            projection="X10c/3.0c",
            frame=[
                "WSne+tPath altitude profile",
                "xaf+lPath step",
                "yaf+lZ / altitude",
            ],
        )

        fig.plot(
            x=path_df["path_step"],
            y=path_df["z"],
            pen="1.2p,black",
        )

        fig.plot(
            x=path_df["path_step"],
            y=path_df["z"],
            style="c0.05c",
            fill="black",
            pen="0.1p,black",
        )

    fig.savefig(str(figure_file), dpi=dpi)

    if cleanup_temp:
        try:
            cleanup_plot_temp_files(cpt_file)
        except Exception:
            pass

    return figure_file



# ============================================================
# Initiate plot
# ============================================================

def plot_initiate_model(
    model: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    figure_file: Path,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=(),
    plot_no_fly_slowness_threshold: float = 10.0,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
):
    """Plot the initial model with start/end markers."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    model = ensure_label_prefix(model)
    plot_model = sample_model_for_plot(model, max_model_points=max_model_points)
    start = model.loc[int(start_idx)]
    end = model.loc[int(end_idx)]

    point_df = pd.DataFrame({"x": [start["x"], end["x"]], "y": [start["y"], end["y"]]})
    region = get_xy_region_with_padding(plot_model, point_df, padding_ratio=0.04)
    is_lonlat = detect_lonlat(plot_model)

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    fig.basemap(
        region=region,
        projection="M16c",
        frame=[
            "WSne+t00 Initiate model check",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    plot_model_flyable_nofly(
        fig=fig,
        model=plot_model,
        model_marker_size=model_marker_size,
        model_alpha=model_alpha,
        no_fly_prefixes=plot_no_fly_prefixes,
        no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
        show_flz_overlay=plot_show_flz_overlay,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    _plot_start_end_markers(fig, model, start_idx, end_idx, label=True)

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    try:
        fig.legend(position="JTL+jTL+o0.15c/0.15c", box="+gwhite@25+p0.5p,black")
    except Exception:
        pass

    fig.savefig(str(figure_file), dpi=dpi)
    return figure_file


# ============================================================
# Multiple ranked paths report
# ============================================================

def _clean_path_indices(path_indices):
    """Return a flat list[int] from common path-index containers."""
    if path_indices is None:
        return []
    try:
        values = list(path_indices)
    except TypeError:
        values = [path_indices]

    out = []
    for value in values:
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            out.extend(_clean_path_indices(value))
        else:
            out.append(int(value))
    return out


def normalize_ranked_paths_for_plot(ranked_paths):
    """Normalize ranked path input to [(rank, path_indices), ...]."""
    if ranked_paths is None:
        return []

    ranked_items = []

    if isinstance(ranked_paths, dict):
        for rank, path_indices in ranked_paths.items():
            ranked_items.append((int(rank), _clean_path_indices(path_indices)))
    elif isinstance(ranked_paths, list):
        for i, item in enumerate(ranked_paths, start=1):
            if isinstance(item, dict):
                rank = int(item.get("rank", i))
                path_indices = item.get("path_indices", item.get("path", []))
                ranked_items.append((rank, _clean_path_indices(path_indices)))
            elif isinstance(item, tuple) and len(item) == 2:
                rank, path_indices = item
                ranked_items.append((int(rank), _clean_path_indices(path_indices)))
            elif isinstance(item, list) and len(item) == 2 and isinstance(item[0], (int, float, str)):
                rank, path_indices = item
                ranked_items.append((int(rank), _clean_path_indices(path_indices)))
            else:
                ranked_items.append((int(i), _clean_path_indices(item)))
    else:
        raise TypeError("Unsupported ranked_paths format.")

    ranked_items = [(r, p) for r, p in ranked_items if len(p) > 0]
    ranked_items = sorted(ranked_items, key=lambda x: x[0])

    seen = set()
    unique = []
    for rank, path_indices in ranked_items:
        if rank in seen:
            continue
        seen.add(rank)
        unique.append((rank, path_indices))
    return unique



def _as_plot_path_items(ranked_paths):
    """Normalize ranked path dictionaries while preserving path-offset metadata."""
    items = []
    if ranked_paths is None:
        return items

    if isinstance(ranked_paths, dict):
        for rank, path_indices in ranked_paths.items():
            items.append({"rank": int(rank), "path_indices": _clean_path_indices(path_indices)})
        return items

    try:
        iterable = list(ranked_paths)
    except TypeError:
        return items

    for i, item in enumerate(iterable, start=1):
        if isinstance(item, dict):
            q = dict(item)
            q["rank"] = int(q.get("rank", i))
            q["path_indices"] = _clean_path_indices(q.get("path_indices", q.get("path", [])))
            items.append(q)
        elif isinstance(item, tuple) and len(item) == 2:
            rank, path_indices = item
            items.append({"rank": int(rank), "path_indices": _clean_path_indices(path_indices)})
        else:
            items.append({"rank": int(i), "path_indices": _clean_path_indices(item)})

    items = [q for q in items if q.get("path_indices")]
    items.sort(key=lambda q: int(q.get("rank", 0)))
    return items


def plot_all_facility_paths_report(
    model: pd.DataFrame,
    ranked_paths,
    figure_file: Path,
    algorithm_name: str,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 1.2,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=(),
    plot_no_fly_slowness_threshold: float = 10.0,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
    result: dict | None = None,
    forward_bg_color: str = "yellow",      # yellow
    backward_bg_color: str = "#d9d9d9",     # light gray
    direction_bg_alpha: float = 0.32,
    direction_bg_width_factor: float = 5.5,
    direction_bg_min_width: float = 5.5,
    traffic_link_bg_alpha: float = 0.18,
    traffic_link_width_factor: float = 7.0,
    traffic_link_min_width: float = 6.0,
    backup_dash_pattern=(6, 4),
    flz_buffer_m: float = 200.0,
    flz_buffer_facecolor: str = "#4da3ff",
    flz_buffer_alpha: float = 0.22,
    flz_buffer_edgecolor: str = "#1f5fbf",
    flz_buffer_edgewidth: float = 0.8,
):
    """Plot all FMM2D facility path-offset lanes with direction/role styling.

    Visual convention:
      - forward lane background  : light yellow
      - backward lane background : light gray
      - main lane                : solid line
      - backup lane              : dashed line
      - traffic-link fallback    : extra clouded/shared-corridor overlay
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    model = ensure_label_prefix(model)

    if "x" in model.columns and "y" in model.columns:
        xcol, ycol = "x", "y"
    elif "lon" in model.columns and "lat" in model.columns:
        xcol, ycol = "lon", "lat"
    else:
        raise ValueError("Cannot plot all facility paths: model must contain x/y or lon/lat columns.")

    x = pd.to_numeric(model[xcol], errors="coerce").to_numpy(dtype=float, copy=True)
    y = pd.to_numeric(model[ycol], errors="coerce").to_numpy(dtype=float, copy=True)
    n_model = len(model)

    finite = np.isfinite(x) & np.isfinite(y)
    if n_model > int(max_model_points) > 0:
        keep_idx = np.flatnonzero(finite)
        step = max(1, int(np.ceil(len(keep_idx) / float(max_model_points))))
        plot_idx = keep_idx[::step]
    else:
        plot_idx = np.flatnonzero(finite)

    valid_paths = []
    for item in _as_plot_path_items(ranked_paths):
        path = [int(v) for v in item.get("path_indices", []) if 0 <= int(v) < n_model]
        if len(path) >= 2:
            q = dict(item)
            q["path_indices"] = path
            valid_paths.append(q)

    if not valid_paths:
        raise ValueError("No valid ranked_paths available for all-facility plot.")

    fig, ax = plt.subplots(figsize=(14, 10), dpi=int(dpi))

    if plot_model_as_flyable_nofly and "slowness" in model.columns:
        slow = pd.to_numeric(model["slowness"], errors="coerce").to_numpy(dtype=float, copy=True)
        fly = np.isfinite(slow) & (slow < float(plot_no_fly_slowness_threshold))
        nofly = np.isfinite(slow) & (slow >= float(plot_no_fly_slowness_threshold))
        pidx = plot_idx
        ax.scatter(
            x[pidx[fly[pidx]]], y[pidx[fly[pidx]]],
            s=float(model_marker_size), marker="o", alpha=float(model_alpha),
            color=FLYABLE_MPL_COLOR, edgecolors="none",
            label=f"Flyable: s < {plot_no_fly_slowness_threshold:g}", zorder=1,
        )
        ax.scatter(
            x[pidx[nofly[pidx]]], y[pidx[nofly[pidx]]],
            s=float(model_marker_size), marker="s", alpha=float(model_alpha),
            color=NO_FLY_MPL_COLOR, edgecolors="none",
            label=f"No-fly: s >= {plot_no_fly_slowness_threshold:g}", zorder=1,
        )
    else:
        ax.scatter(
            x[plot_idx], y[plot_idx],
            s=float(model_marker_size), alpha=float(model_alpha),
            color="0.7", edgecolors="none", label="Model nodes", zorder=1,
        )

    # FLZ zone overlay with 200 m buffer (or configured value).
    # Draw as a transparent blue covered area.
    flz_label_drawn = False
    if plot_show_flz_overlay and "label_prefix" in model.columns:
        flz = model.loc[model["label_prefix"].astype(str).str.upper() == "FLZ"].copy()
        if not flz.empty and float(flz_buffer_m) > 0.0:
            is_lonlat_plot = detect_lonlat(model[[xcol, ycol]].rename(columns={xcol: "x", ycol: "y"}))
            from matplotlib.patches import Circle, Ellipse
            for _, row in flz.iterrows():
                try:
                    fx = float(row[xcol])
                    fy = float(row[ycol])
                except Exception:
                    continue
                if not (np.isfinite(fx) and np.isfinite(fy)):
                    continue
                if is_lonlat_plot:
                    dx_deg, dy_deg = _meters_to_lonlat_delta(fy, float(flz_buffer_m))
                    patch = Ellipse(
                        (fx, fy),
                        width=2.0 * float(dx_deg),
                        height=2.0 * float(dy_deg),
                        facecolor=str(flz_buffer_facecolor),
                        edgecolor=str(flz_buffer_edgecolor),
                        linewidth=float(flz_buffer_edgewidth),
                        alpha=float(flz_buffer_alpha),
                        zorder=1.8,
                        label=(f"FLZ buffer ({float(flz_buffer_m):.0f} m)" if not flz_label_drawn else None),
                    )
                else:
                    patch = Circle(
                        (fx, fy),
                        radius=float(flz_buffer_m),
                        facecolor=str(flz_buffer_facecolor),
                        edgecolor=str(flz_buffer_edgecolor),
                        linewidth=float(flz_buffer_edgewidth),
                        alpha=float(flz_buffer_alpha),
                        zorder=1.8,
                        label=(f"FLZ buffer ({float(flz_buffer_m):.0f} m)" if not flz_label_drawn else None),
                    )
                ax.add_patch(patch)
                flz_label_drawn = True

    max_rank = max(int(item.get("rank", i + 1)) for i, item in enumerate(valid_paths))
    norm = Normalize(vmin=1, vmax=max(1, max_rank))
    cmap = plt.get_cmap("turbo") if max_rank > 1 else plt.get_cmap("viridis")

    source_indices = set()
    target_indices = set()
    labels_by_idx = {}

    def _direction_bg_color(item):
        direction = str(item.get("path_offset_direction", "")).strip().lower()
        if direction == "backward":
            return str(backward_bg_color)
        return str(forward_bg_color)

    forward_bg_drawn = False
    backward_bg_drawn = False
    traffic_label_drawn = False

    # Direction background corridors.
    for item in valid_paths:
        path = item["path_indices"]
        direction = str(item.get("path_offset_direction", "")).strip().lower()
        bg_color = _direction_bg_color(item)
        label = None
        if direction == "backward" and not backward_bg_drawn:
            label = "Backward corridor"
            backward_bg_drawn = True
        elif direction != "backward" and not forward_bg_drawn:
            label = "Forward corridor"
            forward_bg_drawn = True

        ax.plot(
            x[path], y[path],
            color=bg_color,
            linewidth=max(float(direction_bg_min_width), float(path_line_width) * float(direction_bg_width_factor)),
            alpha=float(direction_bg_alpha),
            solid_capstyle="round",
            zorder=2,
            label=label,
        )

        if bool(item.get("traffic_link_required", False)):
            ax.plot(
                x[path], y[path],
                color="0.35",
                linewidth=max(float(traffic_link_min_width), float(path_line_width) * float(traffic_link_width_factor)),
                alpha=float(traffic_link_bg_alpha),
                solid_capstyle="round",
                zorder=3,
                label="Traffic-link buffer" if not traffic_label_drawn else None,
            )
            traffic_label_drawn = True

    # Colored path centerlines. Main is solid; backup is dashed.
    for i, item in enumerate(valid_paths):
        rank = int(item.get("rank", i + 1))
        path = item["path_indices"]
        color = cmap(norm(rank))
        alpha = 0.88 if len(valid_paths) <= 80 else 0.55
        role = str(item.get("path_offset_role", "main")).strip().lower()
        is_backup = role == "backup"
        linestyle = "--" if is_backup else "-"

        line, = ax.plot(
            x[path], y[path],
            color=color,
            linewidth=max(0.9, float(path_line_width) * (0.70 if len(valid_paths) > 80 else 1.10)),
            alpha=alpha,
            linestyle=linestyle,
            zorder=4,
        )
        if is_backup:
            try:
                line.set_dashes(tuple(backup_dash_pattern))
            except Exception:
                pass

        src = int(item.get("source_idx", path[0]))
        dst = int(item.get("target_idx", path[-1]))
        if src != dst and 0 <= src < n_model and 0 <= dst < n_model:
            source_indices.add(src)
            target_indices.add(dst)
            labels_by_idx[src] = str(item.get("source_label", model.loc[src, "label"] if "label" in model.columns else src))
            labels_by_idx[dst] = str(item.get("target_label", model.loc[dst, "label"] if "label" in model.columns else dst))

    both_indices = source_indices & target_indices
    source_only = sorted(source_indices - both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))
    target_only = sorted(target_indices - both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))
    both_sorted = sorted(both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))

    def _scatter_indices(indices, marker, label, face, edge, size, z):
        if not indices:
            return
        idx = np.asarray(indices, dtype=int)
        ax.scatter(
            x[idx], y[idx],
            s=size, marker=marker,
            facecolors=face, edgecolors=edge,
            linewidths=1.7,
            label=label,
            zorder=z,
        )

    _scatter_indices(source_only, "*", "Start facilities", "yellow", "black", 180, 8)
    _scatter_indices(target_only, "s", "End facilities", "none", "blue", 120, 8)
    if both_sorted:
        _scatter_indices(both_sorted, "D", "Start + end facilities", "white", "black", 105, 9)
        _scatter_indices(both_sorted, "*", None, "yellow", "black", 155, 10)

    for pos, idx in enumerate(sorted(source_indices | target_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))):
        if idx < 0 or idx >= n_model or not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
            continue
        dx = 5 if pos % 2 == 0 else -5
        dy = 5 if (pos // 2) % 2 == 0 else -7
        ax.annotate(
            labels_by_idx.get(idx, str(idx)),
            (x[idx], y[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            zorder=12,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.83, pad=0.02)
    cbar.set_label("Rank index")

    unique_sources = len(source_indices)
    unique_targets = len(target_indices)
    pair_count = len(valid_paths)
    traffic_link_count = len({str(item.get("traffic_link_id", "")) for item in valid_paths if str(item.get("traffic_link_id", ""))})
    strict_count = sum(1 for item in valid_paths if bool(item.get("path_offset_strict", False)))

    ax.set_title(f"Scenario 1 facility path-offset routes - {algorithm_name}", fontsize=18, fontweight="bold")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(False)

    style_handles = [
        Patch(facecolor=str(forward_bg_color), edgecolor="none", alpha=float(direction_bg_alpha), label="Forward corridor"),
        Patch(facecolor=str(backward_bg_color), edgecolor="none", alpha=float(direction_bg_alpha), label="Backward corridor"),
        Patch(facecolor=str(flz_buffer_facecolor), edgecolor=str(flz_buffer_edgecolor), alpha=float(flz_buffer_alpha), label=f"FLZ buffer ({float(flz_buffer_m):.0f} m)"),
        Line2D([0], [0], color="black", lw=max(1.2, float(path_line_width) * 1.10), linestyle="-", label="Main lane"),
        Line2D([0], [0], color="black", lw=max(1.2, float(path_line_width) * 1.10), linestyle="--", label="Backup lane"),
    ]
    if traffic_label_drawn:
        style_handles.append(
            Line2D([0], [0], color="0.35", lw=max(float(traffic_link_min_width), float(path_line_width) * 2.0), alpha=float(traffic_link_bg_alpha), label="Traffic-link buffer")
        )

    handles, labels = ax.get_legend_handles_labels()
    merged_handles = []
    merged_labels = []
    for h, l in list(zip(handles, labels)) + [(h, h.get_label()) for h in style_handles]:
        if not l or l in merged_labels:
            continue
        merged_handles.append(h)
        merged_labels.append(l)
    ax.legend(merged_handles, merged_labels, loc="upper left", frameon=True, fancybox=False, edgecolor="black", fontsize=9)

    text = (
        f"Paths plotted: {pair_count}\n"
        f"Strict paths: {strict_count}\n"
        f"Traffic links: {traffic_link_count}\n"
        f"Unique starts: {unique_sources}\n"
        f"Unique ends: {unique_targets}"
    )
    ax.text(
        0.99, 0.99, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.82),
        zorder=20,
    )

    fig.tight_layout()
    fig.savefig(figure_file, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return figure_file

def _safe_filename_token(value: str) -> str:
    """Return a filesystem-safe compact token."""
    import re
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown"



def group_path_items_by_pair(ranked_paths):
    """Group ranked path dictionaries by undirected facility pair."""
    groups = {}
    for item in _as_plot_path_items(ranked_paths):
        src = str(item.get("source_label", "") or "").strip()
        dst = str(item.get("target_label", "") or "").strip()
        route_id = str(item.get("pair_undirected_key", item.get("route_id", "")) or "").strip()
        if route_id:
            key = route_id
        else:
            a, b = sorted([src or "A", dst or "B"])
            key = f"PAIR--{a}--{b}"
        groups.setdefault(key, []).append(dict(item))

    for key in list(groups.keys()):
        groups[key] = sorted(groups[key], key=lambda q: int(q.get("rank", 0)))
    return groups



def plot_facility_pair_path_reports(
    model: pd.DataFrame,
    ranked_paths,
    figure_dir: Path,
    algorithm_name: str,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 1.2,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=(),
    plot_no_fly_slowness_threshold: float = 10.0,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
    result: dict | None = None,
    forward_bg_color: str = "yellow",
    backward_bg_color: str = "#d9d9d9",
    direction_bg_alpha: float = 0.32,
    direction_bg_width_factor: float = 5.5,
    direction_bg_min_width: float = 5.5,
    traffic_link_bg_alpha: float = 0.18,
    traffic_link_width_factor: float = 7.0,
    traffic_link_min_width: float = 6.0,
    backup_dash_pattern=(6, 4),
    flz_buffer_m: float = 200.0,
    flz_buffer_facecolor: str = "#4da3ff",
    flz_buffer_alpha: float = 0.22,
    flz_buffer_edgecolor: str = "#1f5fbf",
    flz_buffer_edgewidth: float = 0.8,
):
    """Plot one lane map per undirected facility pair into figure_dir.

    Each figure usually contains up to 4 paths for the same base pair:
        A -> B main, A -> B backup, B -> A main, B -> A backup.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    grouped = group_path_items_by_pair(ranked_paths)
    if not grouped:
        return outputs

    for pair_key, items in grouped.items():
        labels = []
        for item in items:
            for key in ("source_label", "target_label"):
                val = str(item.get(key, "") or "").strip()
                if val and val not in labels:
                    labels.append(val)
        if len(labels) >= 2:
            pair_title = f"{labels[0]} <-> {labels[1]}"
            fname_a = _safe_filename_token(labels[0])
            fname_b = _safe_filename_token(labels[1])
        elif len(labels) == 1:
            pair_title = labels[0]
            fname_a = _safe_filename_token(labels[0])
            fname_b = "pair"
        else:
            pair_title = str(pair_key)
            fname_a = _safe_filename_token(pair_key)
            fname_b = "pair"

        figure_file = figure_dir / f"path_report_{_safe_filename_token(algorithm_name)}_pair_{fname_a}_to_{fname_b}.png"

        plot_all_facility_paths_report(
            model=model,
            ranked_paths=items,
            figure_file=figure_file,
            algorithm_name=f"{algorithm_name} | {pair_title}",
            max_model_points=max_model_points,
            dpi=dpi,
            model_alpha=model_alpha,
            model_marker_size=model_marker_size,
            path_line_width=path_line_width,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=always_flyable_prefixes,
            result=result,
            forward_bg_color=forward_bg_color,
            backward_bg_color=backward_bg_color,
            direction_bg_alpha=direction_bg_alpha,
            direction_bg_width_factor=direction_bg_width_factor,
            direction_bg_min_width=direction_bg_min_width,
            traffic_link_bg_alpha=traffic_link_bg_alpha,
            traffic_link_width_factor=traffic_link_width_factor,
            traffic_link_min_width=traffic_link_min_width,
            backup_dash_pattern=backup_dash_pattern,
            flz_buffer_m=flz_buffer_m,
            flz_buffer_facecolor=flz_buffer_facecolor,
            flz_buffer_alpha=flz_buffer_alpha,
            flz_buffer_edgecolor=flz_buffer_edgecolor,
            flz_buffer_edgewidth=flz_buffer_edgewidth,
        )
        outputs.append(str(figure_file))

    return outputs


def plot_multiple_paths_report(
    model: pd.DataFrame,
    ranked_paths,
    figure_file: Path,
    algorithm_name: str,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 1.2,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=(),
    plot_no_fly_slowness_threshold: float = 10.0,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=(),
    result: dict | None = None,
    rank_min: int | None = None,
    rank_max: int | None = None,
):
    """Plot all ranked paths in one map."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    model = ensure_label_prefix(model)
    ranked_items = normalize_ranked_paths_for_plot(ranked_paths)
    if not ranked_items:
        raise ValueError("No ranked paths available to plot.")

    if rank_min is None:
        rank_min = min(rank for rank, _ in ranked_items)
    if rank_max is None:
        rank_max = max(rank for rank, _ in ranked_items)

    plot_model = sample_model_for_plot(model, max_model_points=max_model_points)
    all_path_rows = []
    for rank, path_indices in ranked_items:
        tmp = model.loc[path_indices, ["x", "y", "z", "label"]].copy()
        tmp["rank"] = rank
        all_path_rows.append(tmp)
    all_path_df = pd.concat(all_path_rows, axis=0, ignore_index=True)

    region = get_xy_region_with_padding(plot_model, all_path_df, padding_ratio=0.04)
    is_lonlat = detect_lonlat(plot_model)

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    fig.basemap(
        region=region,
        projection="M16c",
        frame=[
            f"WSne+tScenario 1 ranked path report - {algorithm_name}",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    plot_model_flyable_nofly(
        fig=fig,
        model=plot_model,
        model_marker_size=model_marker_size,
        model_alpha=model_alpha,
        no_fly_prefixes=plot_no_fly_prefixes,
        no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
        show_flz_overlay=plot_show_flz_overlay,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    seg_file = figure_file.parent / f"{figure_file.stem}_all_ranks_segments.txt"
    with seg_file.open("w", encoding="utf-8") as f:
        for rank, path_indices in ranked_items:
            f.write(f"> -Z{rank}\n")
            path_df = model.loc[path_indices, ["x", "y"]]
            for _, row in path_df.iterrows():
                f.write(f"{row['x']} {row['y']}\n")

    pygmt.makecpt(cmap="turbo", series=[rank_min, rank_max, 1], continuous=True)
    fig.plot(data=str(seg_file), pen=f"{path_line_width}p,+z", cmap=True, transparency=20)

    best_rank, best_path_indices = ranked_items[0]
    best_df = model.loc[best_path_indices]
    fig.plot(x=best_df["x"], y=best_df["y"], pen=f"{max(path_line_width + 0.8, 2.0)}p,black", label="Fastest path")

    _plot_start_end_markers(fig, model, best_path_indices[0], best_path_indices[-1], label=True)

    fig.colorbar(cmap=True, position="JMR+w11.5c/0.350c+o0.8c/0c+v", frame=["xaf+lRank index"])

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    try:
        fig.legend(position="JTL+jTL+o0.15c/0.15c", box="+gwhite@25+p0.5p,black")
    except Exception:
        pass

    fig.savefig(str(figure_file), dpi=dpi)

    try:
        seg_file.unlink()
    except Exception:
        pass

    return figure_file


def normalize_ranked_paths(paths: list[list[int]] | dict[int, list[int]]) -> list[tuple[int, list[int]]]:
    """Backward-compatible ranked path normalizer."""
    return normalize_ranked_paths_for_plot(paths)


# ============================================================
# Traveltime histogram
# ============================================================

def plot_multiple_path_time_histogram(
    time_table: pd.DataFrame,
    figure_file: Path,
    algorithm_name: str,
    fastest_n: int = 10,
    time_column: str = "traveltime_s",
    rank_column: str = "rank",
    dpi: int = 300,
    bin_count: int = 20,
):
    """Plot traveltime histogram for multiple ranked paths."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if time_table is None or len(time_table) == 0:
        raise ValueError("time_table is empty; cannot plot traveltime histogram.")

    df = time_table.copy()
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if rank_column not in df.columns:
        df[rank_column] = np.arange(1, len(df) + 1)

    df[time_column] = pd.to_numeric(df[time_column], errors="coerce")
    df[rank_column] = pd.to_numeric(df[rank_column], errors="coerce")
    df = df.dropna(subset=[time_column, rank_column]).copy()
    if df.empty:
        raise ValueError("No valid traveltime values to plot.")

    df[rank_column] = df[rank_column].astype(int)
    df = df.sort_values([time_column, rank_column]).reset_index(drop=True)

    fastest_n = max(1, min(int(fastest_n), len(df)))
    fastest_df = df.head(fastest_n).copy()

    all_times = df[time_column].to_numpy(float)
    fast_times = fastest_df[time_column].to_numpy(float)

    tmin = float(np.nanmin(all_times))
    tmax = float(np.nanmax(all_times))
    if np.isclose(tmin, tmax):
        dt = max(abs(tmin) * 0.05, 1.0)
        tmin -= dt
        tmax += dt

    use_min = tmax >= 120.0
    if use_min:
        all_plot = all_times / 60.0
        fast_plot = fast_times / 60.0
        xmin, xmax = tmin / 60.0, tmax / 60.0
        xlabel = "Traveltime (min)"
        fastest_value = float(np.nanmin(all_times) / 60.0)
    else:
        all_plot = all_times
        fast_plot = fast_times
        xmin, xmax = tmin, tmax
        xlabel = "Traveltime (s)"
        fastest_value = float(np.nanmin(all_times))

    if np.isclose(xmin, xmax):
        dx = max(abs(xmin) * 0.05, 1.0)
        xmin -= dx
        xmax += dx

    bin_count = max(5, int(bin_count))
    bin_width = (xmax - xmin) / float(bin_count)
    if bin_width <= 0 or not np.isfinite(bin_width):
        bin_width = 1.0

    counts_all, _ = np.histogram(all_plot, bins=bin_count, range=(xmin, xmax))
    counts_fast, _ = np.histogram(fast_plot, bins=bin_count, range=(xmin, xmax))
    ymax = max(1, int(np.ceil(max(counts_all.max(), counts_fast.max()) * 1.25)))

    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="13p,Helvetica-Bold", FONT_LABEL="10p,Helvetica", FONT_ANNOT_PRIMARY="8p,Helvetica", MAP_FRAME_TYPE="plain")

    fig.basemap(
        region=[float(xmin), float(xmax), 0, float(ymax)],
        projection="X15c/8c",
        frame=[f"WSne+tTraveltime distribution - {algorithm_name}", f"xaf+l{xlabel}", "yaf+lNumber of paths"],
    )

    fig.histogram(data=all_plot, series=bin_width, fill="gray@45", pen="0.6p,black", label=f"All paths (n={len(df)})")
    fig.histogram(data=fast_plot, series=bin_width, fill="orange@25", pen="0.8p,orange", label=f"Fastest {fastest_n} paths")
    fig.plot(x=[fastest_value, fastest_value], y=[0, ymax], pen="1.4p,blue,--", label="Fastest path time")

    try:
        fig.legend(position="JTR+jTR+o0.15c/0.15c", box="+gwhite@25+p0.5p,black")
    except Exception:
        pass

    fig.savefig(str(figure_file), dpi=dpi)
    return figure_file


# ============================================================
# Discrete slowness CPT helpers
# ============================================================

def _as_float_list(value):
    """Return clean list[float] from None/list/tuple/comma-separated text."""
    if value is None:
        return None

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        values = [float(p) for p in parts]
    else:
        try:
            values = [float(v) for v in value]
        except TypeError:
            values = [float(value)]

    values = [v for v in values if np.isfinite(v)]
    return values or None


def build_slowness_cpt_bounds(
    slowness_values=None,
    mode: str = "data_nonuniform",
    manual_bounds=None,
    vmin: float = 0.0,
    vmax: float = 10.0,
    no_fly_threshold: float = 10.0,
    n_steps: int = 50,
    round_decimals: int = 4,
    max_bounds: int = 120,
):
    """
    Build discrete CPT bounds for slowness/cost.

    Modes:
        data_nonuniform / quantile:
            Build non-uniform bounds from true flyable slowness values.
            Recommended when flyable values are small, e.g. 0.02-0.1,
            but no-fly cells are 10.

        uniform:
            Build equal steps from vmin to vmax using n_steps.

        manual:
            Use manual_bounds directly.

        true_values:
            Use actual unique flyable slowness values. If there are too many,
            reduce them by quantiles using max_bounds.
    """
    mode = str(mode or "data_nonuniform").strip().lower()
    if mode == "quantile":
        mode = "data_nonuniform"

    vmin = float(vmin)
    vmax = float(vmax)
    no_fly_threshold = float(no_fly_threshold)
    n_steps = max(1, int(n_steps))
    round_decimals = int(round_decimals)
    max_bounds = max(3, int(max_bounds))

    if vmax <= vmin:
        vmax = vmin + 1.0

    # ------------------------------------------------------------
    # Manual mode
    # ------------------------------------------------------------
    if mode == "manual" and manual_bounds is not None:
        bounds = _as_float_list(manual_bounds)
        if bounds is None or len(bounds) < 2:
            bounds = [vmin, vmax]

        bounds = np.asarray(bounds, dtype=float)
        bounds = bounds[np.isfinite(bounds)]
        bounds = np.clip(bounds, vmin, vmax)
        bounds = np.unique(np.round(bounds, round_decimals))

        if len(bounds) < 2:
            bounds = np.array([vmin, vmax], dtype=float)

        return [float(v) for v in bounds]

    # ------------------------------------------------------------
    # Uniform mode: fixed range and fixed n-step
    # ------------------------------------------------------------
    if mode == "uniform":
        bounds = np.linspace(vmin, vmax, n_steps + 1)
        bounds = np.unique(np.round(bounds, round_decimals))
        if len(bounds) < 2:
            bounds = np.array([vmin, vmax], dtype=float)
        return [float(v) for v in bounds]

    # ------------------------------------------------------------
    # Data-based modes
    # ------------------------------------------------------------
    vals = pd.to_numeric(pd.Series(slowness_values), errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if vals.size == 0:
        bounds = np.linspace(vmin, vmax, n_steps + 1)
        return [float(v) for v in np.unique(np.round(bounds, round_decimals))]

    # Important:
    # Build the gradient only from flyable values below the no-fly threshold.
    # Otherwise value 10 dominates the scale and small flyable values collapse
    # into one color.
    flyable = vals[
        (vals >= vmin)
        & (vals < no_fly_threshold)
        & np.isfinite(vals)
    ]

    if flyable.size == 0:
        bounds = np.array([vmin, no_fly_threshold], dtype=float)
        bounds = np.clip(bounds, vmin, vmax)
        bounds = np.unique(np.round(bounds, round_decimals))
        if len(bounds) < 2:
            bounds = np.array([vmin, vmax], dtype=float)
        return [float(v) for v in bounds]

    if mode == "true_values":
        bounds = np.unique(np.round(flyable, round_decimals))

        if len(bounds) > max_bounds:
            q = np.linspace(0.0, 1.0, max_bounds)
            bounds = np.quantile(flyable, q)
            bounds = np.unique(np.round(bounds, round_decimals))
    else:
        # data_nonuniform / quantile
        q = np.linspace(0.0, 1.0, n_steps + 1)
        bounds = np.quantile(flyable, q)
        bounds = np.unique(np.round(bounds, round_decimals))

    bounds = bounds[np.isfinite(bounds)]

    # Force lower bound.
    bounds = np.unique(
        np.concatenate(
            [
                [vmin],
                bounds,
            ]
        )
    )

    # Add one bound just below the no-fly threshold so 10 is separated
    # as the final high-cost/no-fly class.
    eps = max(abs(no_fly_threshold) * 1e-6, 1e-6)
    below_threshold = no_fly_threshold - eps

    if bounds[-1] < below_threshold:
        bounds = np.concatenate([bounds, [below_threshold]])

    # Add exact no-fly threshold / vmax endpoint.
    high_endpoint = min(max(no_fly_threshold, vmin), vmax)
    if bounds[-1] < high_endpoint or not np.isclose(bounds[-1], high_endpoint):
        bounds = np.concatenate([bounds, [high_endpoint]])

    bounds = np.clip(bounds, vmin, vmax)
    bounds = np.unique(np.round(bounds, round_decimals))

    if len(bounds) < 2:
        bounds = np.array([vmin, vmax], dtype=float)

    return [float(v) for v in bounds]


def _sample_cmap_rgb_strings(cmap: str, n: int):
    """
    Sample n RGB colors from a matplotlib colormap.

    This is used to make an interval-indexed CPT. The important point is that
    colors are distributed by class/bin number, not by the absolute z value.
    That is what makes non-uniform slowness bounds useful.
    """
    n = max(1, int(n))

    try:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap(cmap)
        if n == 1:
            positions = [0.5]
        else:
            # Avoid the very dark/pale extremes of some colormaps.
            positions = np.linspace(0.06, 0.94, n)

        colors = []
        for p in positions:
            r, g, b, _ = cm(float(p))
            colors.append(f"{int(round(r * 255))}/{int(round(g * 255))}/{int(round(b * 255))}")
        return colors

    except Exception:
        # Fallback palette, no matplotlib needed.
        base = [
            "68/1/84",
            "72/35/116",
            "64/67/135",
            "52/94/141",
            "41/120/142",
            "32/144/140",
            "34/167/132",
            "68/190/112",
            "121/209/81",
            "189/223/38",
            "253/231/37",
        ]

        idx = np.linspace(0, len(base) - 1, n).round().astype(int)
        return [base[i] for i in idx]


def _write_interval_indexed_cpt(
    cpt_file: Path,
    bounds,
    cmap: str = "turbo",
    no_fly_threshold: float = 10.0,
    no_fly_color: str = NO_FLY_RGB,
):
    """
    Write a GMT CPT where colors are assigned by interval index.

    Why not only pygmt.makecpt(series=nonuniform_bounds)?
    -----------------------------------------------------
    GMT still samples the colormap using the real z values. If z ranges from
    0 to 10, then values 0.02-0.10 still use almost the same low-end color.
    This function instead assigns colors sequentially across the bins, so
    non-uniform bounds reveal gradients in the small flyable slowness range.
    """
    cpt_file = Path(cpt_file)

    bounds = [float(v) for v in bounds if np.isfinite(float(v))]
    bounds = sorted(set(bounds))

    threshold = float(no_fly_threshold)
    eps = max(abs(threshold) * 1e-6, 1e-6)

    # Make sure exact no-fly value, e.g. 10.0, can get its own black no-fly class.
    if threshold not in bounds:
        bounds.append(threshold)

    if max(bounds) <= threshold:
        bounds.append(threshold + eps)

    bounds = sorted(set(float(v) for v in bounds if np.isfinite(v)))
    n_intervals = max(1, len(bounds) - 1)

    flyable_interval_indices = []
    for i in range(n_intervals):
        z0 = bounds[i]
        if z0 < threshold:
            flyable_interval_indices.append(i)

    fly_colors = _sample_cmap_rgb_strings(cmap, max(1, len(flyable_interval_indices)))

    lines = []
    fly_i = 0

    for i in range(n_intervals):
        z0 = bounds[i]
        z1 = bounds[i + 1]

        if z0 >= threshold:
            color = no_fly_color
        else:
            color = fly_colors[min(fly_i, len(fly_colors) - 1)]
            fly_i += 1

        lines.append(f"{z0:.12g} {color} {z1:.12g} {color}")

    # B = below, F = above/foreground, N = NaN.
    # Set F to no-fly color so exact/high no-fly values stay black.
    lines.append("B 245/245/245")
    lines.append(f"F {no_fly_color}")
    lines.append("N 180/180/180")

    cpt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cpt_file, bounds


def make_discrete_slowness_cpt(
    cpt_file: Path,
    slowness_values=None,
    bounds=None,
    cmap: str = "turbo",
    vmin: float = 0.0,
    vmax: float = 10.0,
    no_fly_threshold: float = 10.0,
    n_steps: int = 50,
    cpt_mode: str = "data_nonuniform",
    round_decimals: int = 4,
    max_bounds: int = 120,
    no_fly_color: str = NO_FLY_RGB,
):
    """
    Make a discrete CPT for slowness/cost.

    Modes:
        cpt_mode="data_nonuniform"
            Build non-uniform bins from true flyable slowness values.
            Colors are assigned by bin index, so small flyable differences
            are visible even when no-fly slowness is 10.

        cpt_mode="uniform"
            Use fixed vmin/vmax/n_steps.

        cpt_mode="manual"
            Use bounds provided by slowness_discrete_bounds.

        cpt_mode="true_values"
            Use actual unique flyable values, limited by max_bounds.

    Important:
        This function writes a custom CPT instead of relying only on
        pygmt.makecpt(series=nonuniform_bounds). That is necessary because
        GMT samples colors by absolute z value, which makes 0.02-0.10 all
        appear nearly identical when the total scale is 0-10.
    """
    cpt_file = Path(cpt_file)

    if bounds is not None:
        final_bounds = build_slowness_cpt_bounds(
            slowness_values=slowness_values,
            mode="manual",
            manual_bounds=bounds,
            vmin=vmin,
            vmax=vmax,
            no_fly_threshold=no_fly_threshold,
            n_steps=n_steps,
            round_decimals=round_decimals,
            max_bounds=max_bounds,
        )
    else:
        final_bounds = build_slowness_cpt_bounds(
            slowness_values=slowness_values,
            mode=cpt_mode,
            manual_bounds=None,
            vmin=vmin,
            vmax=vmax,
            no_fly_threshold=no_fly_threshold,
            n_steps=n_steps,
            round_decimals=round_decimals,
            max_bounds=max_bounds,
        )

    cpt_file, final_bounds = _write_interval_indexed_cpt(
        cpt_file=cpt_file,
        bounds=final_bounds,
        cmap=cmap,
        no_fly_threshold=no_fly_threshold,
        no_fly_color=no_fly_color,
    )

    return cpt_file, final_bounds



def _slowness_colorbar_frame(bounds=None, label: str = "Slowness / cost"):
    """
    GMT-6/PyGMT-safe colorbar frame.

    Do not manually build GMT-4 style labels such as:
        0/0,0.05/0.05

    That causes:
        Mixing of GMT 4 and 5 level syntax is not possible
    """
    return [f"xaf+l{label}"]


def _plot_slowness_points_with_discrete_cpt(
    fig: pygmt.Figure,
    plot_model: pd.DataFrame,
    cpt_file: Path,
    marker_size: float,
    dot_pen: str | None = None,
):
    """Plot slowness-colored points with a discrete CPT."""
    fig.plot(
        x=plot_model["x"],
        y=plot_model["y"],
        style=f"c{max(marker_size / 28.0, 0.04):.3f}c",
        fill=pd.to_numeric(plot_model["slowness"], errors="coerce"),
        cmap=str(cpt_file),
        pen=dot_pen,
    )


# ============================================================
# Side-by-side input model vs slowness model
# ============================================================

def plot_model_slowness_side_by_side(
    model: pd.DataFrame,
    figure_file: Path,
    start_idx: int | None = None,
    end_idx: int | None = None,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 1.0,
    no_fly_prefixes=(),
    no_fly_slowness_threshold: float = 10.0,
    always_flyable_prefixes=(),
    show_flz_overlay: bool = True,
    slowness_discrete_bounds=None,
    cleanup_temp: bool = True,
):
    """Plot input flyable/no-fly model and slowness/cost model side by side."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    model = ensure_label_prefix(model)
    plot_model = sample_model_for_plot(model, max_model_points=max_model_points)
    if plot_model.empty:
        raise ValueError("No model points available for side-by-side plot.")

    point_rows = []
    if start_idx is not None:
        point_rows.append(model.loc[int(start_idx), ["x", "y"]])
    if end_idx is not None:
        point_rows.append(model.loc[int(end_idx), ["x", "y"]])
    point_df = pd.DataFrame(point_rows) if point_rows else plot_model[["x", "y"]].copy()

    region = get_xy_region_with_padding(plot_model, point_df, padding_ratio=0.04)
    is_lonlat = detect_lonlat(plot_model)

    cpt_file = figure_file.with_suffix(".slowness_discrete.cpt")
    cpt_file, bounds = make_discrete_slowness_cpt(
        cpt_file=cpt_file,
        slowness_values=plot_model["slowness"],
        bounds=slowness_discrete_bounds,      # None = auto from true flyable values
        cmap="turbo",
        vmin=0.0,
        vmax=float(no_fly_slowness_threshold),
        no_fly_threshold=float(no_fly_slowness_threshold),
        n_steps=50,
        cpt_mode="data_nonuniform",           # use "uniform" for fixed equal steps
        round_decimals=4,
        max_bounds=120,
    )

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="12p,Helvetica-Bold",
        FONT_LABEL="9p,Helvetica",
        FONT_ANNOT_PRIMARY="7p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    # Left panel: binary input model
    fig.basemap(
        region=region,
        projection="M12c",
        frame=[
            "WSne+tInput model: flyable / no-fly",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    plot_model_flyable_nofly(
        fig=fig,
        model=plot_model,
        model_marker_size=model_marker_size,
        model_alpha=model_alpha,
        no_fly_prefixes=no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        show_flz_overlay=show_flz_overlay,
        always_flyable_prefixes=always_flyable_prefixes,
    )
    _plot_start_end_markers(fig, model, start_idx, end_idx, label=True)

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    try:
        fig.legend(position="JTL+jTL+o0.12c/0.12c", box="+gwhite@25+p0.4p,black")
    except Exception:
        pass

    # Right panel: slowness/cost model
    fig.shift_origin(xshift="13.0c")
    fig.basemap(
        region=region,
        projection="M12c",
        frame=[
            "WSne+tSlowness / cost model",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    _plot_slowness_points_with_discrete_cpt(fig, plot_model, cpt_file, marker_size=model_marker_size)
    _plot_start_end_markers(fig, model, start_idx, end_idx, label=True)

    # Overlay no-fly nodes to make threshold easy to verify.
    _, nofly = classify_flyable_nofly(
        plot_model,
        no_fly_prefixes=no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        always_flyable_prefixes=always_flyable_prefixes,
    )
    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style=f"s{max(model_marker_size / 22.0, 0.055):.3f}c",
            fill=NO_FLY_GMT_FILL,
            pen=NO_FLY_GMT_PEN_THIN,
            label=f"No-fly: s >= {no_fly_slowness_threshold:g}",
        )

    # fig.colorbar(
    #     cmap=str(cpt_file),
    #     position="JMR+w9.5c/0.32c+o0.7c/0c+v",
    #     frame=_slowness_colorbar_frame(bounds, label="Slowness / cost"),
    # )
    fig.colorbar(
        cmap=str(cpt_file),
        position="JMR+w9c/0.45c+o0.8c/0c+v",
        equalsize="0.5c",
        # frame=["x+lSlowness / cost"],
    )
    try:
        fig.legend(position="JTL+jTL+o0.12c/0.12c", box="+gwhite@25+p0.4p,black")
    except Exception:
        pass

    fig.savefig(str(figure_file), dpi=dpi)

    if cleanup_temp:
        try:
            cleanup_plot_temp_files(cpt_file)
        except Exception:
            pass

    return figure_file


# ============================================================
# Path corridor zoom diagnostic
# ============================================================

def _path_zoom_region(model: pd.DataFrame, path_df: pd.DataFrame, buffer_m: float) -> list[float]:
    """Build zoom region around path using metric buffer."""
    xmin = float(path_df["x"].min())
    xmax = float(path_df["x"].max())
    ymin = float(path_df["y"].min())
    ymax = float(path_df["y"].max())

    if detect_lonlat(model):
        lat0 = float(path_df["y"].mean())
        dx, dy = _meters_to_lonlat_delta(lat0, buffer_m)
    else:
        dx = dy = float(buffer_m)

    if np.isclose(xmin, xmax):
        xmin -= dx
        xmax += dx
    if np.isclose(ymin, ymax):
        ymin -= dy
        ymax += dy

    return [xmin - dx, xmax + dx, ymin - dy, ymax + dy]


def _model_inside_region(model: pd.DataFrame, region: list[float]) -> pd.DataFrame:
    xmin, xmax, ymin, ymax = region
    return model[
        (pd.to_numeric(model["x"], errors="coerce") >= xmin)
        & (pd.to_numeric(model["x"], errors="coerce") <= xmax)
        & (pd.to_numeric(model["y"], errors="coerce") >= ymin)
        & (pd.to_numeric(model["y"], errors="coerce") <= ymax)
    ].copy()


def _get_graph_neighbors(graph, idx: int):
    """Robustly extract neighbors of one node from common graph structures."""
    idx = int(idx)

    # Common dict-of-dicts or dict-of-lists keys.
    for key in ("neighbors", "adjacency", "adj", "edges"):
        obj = graph.get(key, None) if isinstance(graph, dict) else None
        if obj is None:
            continue
        try:
            vals = obj.get(idx, [])
        except AttributeError:
            try:
                vals = obj[idx]
            except Exception:
                vals = []

        out = []
        for item in vals:
            if isinstance(item, (tuple, list)) and len(item) > 0:
                out.append(int(item[0]))
            else:
                try:
                    out.append(int(item))
                except Exception:
                    pass
        if out:
            return out

    return []


def _write_neighbor_edge_file(
    model: pd.DataFrame,
    graph,
    path_indices: list[int],
    local_index_set: set[int],
    out_file: Path,
):
    """Write GMT multi-segment neighbor edges around path nodes."""
    n_edges = 0
    out_file = Path(out_file)
    with out_file.open("w", encoding="utf-8") as f:
        for idx in path_indices:
            idx = int(idx)
            for nb in _get_graph_neighbors(graph, idx):
                nb = int(nb)
                if nb not in local_index_set:
                    continue
                if idx not in model.index or nb not in model.index:
                    continue
                f.write(">\n")
                f.write(f"{model.loc[idx, 'x']} {model.loc[idx, 'y']}\n")
                f.write(f"{model.loc[nb, 'x']} {model.loc[nb, 'y']}\n")
                n_edges += 1
    return out_file, n_edges


def _copy_with_plot_xy(
    df: pd.DataFrame,
    x_values,
    y_values,
) -> pd.DataFrame:
    """Return a copy whose x/y columns are replaced by plotting coordinates."""
    out = df.copy()
    out["x_original"] = out["x"]
    out["y_original"] = out["y"]
    out["x"] = np.asarray(x_values, dtype=float)
    out["y"] = np.asarray(y_values, dtype=float)
    return out


def _relative_xy_from_lower_left(
    df: pd.DataFrame,
    origin_x: float,
    origin_y: float,
    lat_ref: float | None = None,
    is_lonlat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert x/y to relative coordinates from the lower-left reference point."""
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)

    if is_lonlat:
        if lat_ref is None:
            lat_ref = float(np.nanmean(y))
        meters_per_deg_lon = 111_320.0 * max(math.cos(math.radians(float(lat_ref))), 1e-8)
        meters_per_deg_lat = 110_540.0
        return (x - float(origin_x)) * meters_per_deg_lon, (y - float(origin_y)) * meters_per_deg_lat

    return x - float(origin_x), y - float(origin_y)


def _plot_rectangle_ab(
    fig: pygmt.Figure,
    region: list[float],
    pen: str = "1.0p,blue,--",
    label: bool = True,
):
    """Draw and label the A-B zoom rectangle."""
    xmin, xmax, ymin, ymax = [float(v) for v in region]

    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]

    fig.plot(x=xs, y=ys, pen=pen)

    if label:
        fig.text(
            x=xmin,
            y=ymin,
            text="A",
            font="10p,Helvetica-Bold,blue",
            justify="LB",
            offset="0.08c/0.08c",
        )

        fig.text(
            x=xmax,
            y=ymax,
            text="B",
            font="10p,Helvetica-Bold,blue",
            justify="RT",
            offset="-0.08c/-0.08c",
        )


def _plot_path_direction_segments(
    fig: pygmt.Figure,
    path_df: pd.DataFrame,
    path_line_width: float = 1.2,
    arrow_every: int = 1,
):
    """Plot path segments with arrow heads to show step direction."""
    if len(path_df) < 2:
        return

    arrow_every = max(1, int(arrow_every))
    x = path_df["x"].to_numpy(dtype=float)
    y = path_df["y"].to_numpy(dtype=float)

    for i in range(len(path_df) - 1):
        # Draw every segment as a line, but only add arrow heads at the requested interval.
        if i % arrow_every == 0:
            pen = f"{path_line_width}p,black+ve0.18c"
        else:
            pen = f"{path_line_width}p,black"
        fig.plot(x=[x[i], x[i + 1]], y=[y[i], y[i + 1]], pen=pen)


def plot_path_zoom_diagnostic(
    model: pd.DataFrame,
    graph,
    path_indices: list[int],
    figure_file: Path,
    algorithm_name: str,
    buffer_m: float = 250.0,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_marker_size: float = 1.5,
    path_line_width: float = 1.2,
    no_fly_prefixes=("RA",),
    no_fly_slowness_threshold: float = 10.0,
    always_flyable_prefixes=("DB", "DK"),
    show_neighbor_edges: bool = False,
    show_adjacent_nodes: bool = True,
    slowness_discrete_bounds=None,
    cleanup_temp: bool = True,
    coordinate_mode: str = "relative_m",
    label_path_steps: bool = True,
    label_step_every: int = 1,
    arrow_every: int = 1,
    plot_surface: bool = False,
    surface_spacing_m: float = 20.0,
    surface_alpha: int = 0,
):
    """Plot a simple rectangular zoom around the path and show step direction.

    coordinate_mode:
        "relative_m"  : use distance in meters from the lower-left corner A.
        "map" or "xy" : use original model x/y coordinates.

    The rectangle is defined from A = lower-left of the zoom window to
    B = upper-right of the zoom window.  In relative_m mode, A is (0, 0).
    """
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if not path_indices:
        raise ValueError("Cannot plot zoom diagnostic for empty path.")

    model = ensure_label_prefix(model)
    path_indices = [int(i) for i in path_indices]
    path_df_original = model.loc[path_indices].copy()

    # The zoom rectangle is only the path bounding box plus a metric buffer.
    map_region = _path_zoom_region(model, path_df_original, buffer_m=buffer_m)
    local_model_original = _model_inside_region(model, map_region)
    if local_model_original.empty:
        raise ValueError("No model nodes inside path zoom rectangle.")

    if len(local_model_original) > int(max_model_points):
        local_model_original = sample_model_for_plot(
            local_model_original,
            max_model_points=max_model_points,
        )

    local_index_set = {int(i) for i in local_model_original.index}
    path_index_set = {int(i) for i in path_indices}

    adjacent_indices = set()
    if show_adjacent_nodes or show_neighbor_edges:
        for idx in path_indices:
            for nb in _get_graph_neighbors(graph, int(idx)):
                nb = int(nb)
                if nb in local_index_set and nb not in path_index_set:
                    adjacent_indices.add(nb)

    is_lonlat = detect_lonlat(model)
    coordinate_mode = str(coordinate_mode or "relative_m").strip().lower()
    use_relative = coordinate_mode in ("relative", "relative_m", "meter", "meters", "local_m")

    if use_relative:
        origin_x = float(map_region[0])
        origin_y = float(map_region[2])
        lat_ref = float(path_df_original["y"].mean()) if is_lonlat else None

        local_x, local_y = _relative_xy_from_lower_left(
            local_model_original,
            origin_x=origin_x,
            origin_y=origin_y,
            lat_ref=lat_ref,
            is_lonlat=is_lonlat,
        )
        path_x, path_y = _relative_xy_from_lower_left(
            path_df_original,
            origin_x=origin_x,
            origin_y=origin_y,
            lat_ref=lat_ref,
            is_lonlat=is_lonlat,
        )

        local_model = _copy_with_plot_xy(local_model_original, local_x, local_y)
        path_df = _copy_with_plot_xy(path_df_original, path_x, path_y)

        # Region from A=(0,0) to B=(width,height).
        if is_lonlat:
            width_m, height_m = _relative_xy_from_lower_left(
                pd.DataFrame({"x": [map_region[1]], "y": [map_region[3]]}),
                origin_x=origin_x,
                origin_y=origin_y,
                lat_ref=lat_ref,
                is_lonlat=True,
            )
            plot_region = [0.0, float(width_m[0]), 0.0, float(height_m[0])]
        else:
            plot_region = [0.0, map_region[1] - map_region[0], 0.0, map_region[3] - map_region[2]]

        projection = "X16c/14c"
        frame = [
            f"WSne+tPath step direction zoom - {algorithm_name}",
            "xaf+lDistance east from A (m)",
            "yaf+lDistance north from A (m)",
        ]
        info_coord_text = "Coordinate: relative meters from A"
    else:
        local_model = local_model_original.copy()
        path_df = path_df_original.copy()
        plot_region = map_region
        projection = "M16c"
        frame = [
            f"WSne+tPath step direction zoom - {algorithm_name}",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ]
        info_coord_text = "Coordinate: original x/y"

    # # Make discrete CPT. If slowness_discrete_bounds is None, default is 50 steps.
    # cpt_file = figure_file.with_suffix(".zoom_slowness_discrete.cpt")
    # make_discrete_slowness_cpt(
    #     cpt_file=cpt_file,
    #     bounds=slowness_discrete_bounds,
    #     cmap="turbo",
    #     vmin=0.0,
    #     vmax=float(no_fly_slowness_threshold),
    #     n_steps=50,
    # )

    cpt_file = figure_file.with_suffix(".zoom_slowness_discrete.cpt")
    cpt_file, bounds = make_discrete_slowness_cpt(
        cpt_file=cpt_file,
        slowness_values=local_model["slowness"],
        bounds=slowness_discrete_bounds,      # None = auto from true flyable values
        cmap="turbo",
        vmin=0.0,
        vmax=float(no_fly_slowness_threshold),
        no_fly_threshold=float(no_fly_slowness_threshold),
        n_steps=50,
        cpt_mode="data_nonuniform",           # use "uniform" for fixed equal steps
        round_decimals=4,
        max_bounds=120,
    )

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="12p,Helvetica-Bold",
        FONT_LABEL="9p,Helvetica",
        FONT_ANNOT_PRIMARY="7p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    fig.basemap(region=plot_region, projection=projection, frame=frame)
    # ------------------------------------------------------------
    # Surface slowness background
    # ------------------------------------------------------------
    grid_file = figure_file.with_suffix(".zoom_surface.grd")

    if plot_surface:
        surf_df = local_model[["x", "y", "slowness"]].copy()

        surf_df["x"] = pd.to_numeric(surf_df["x"], errors="coerce")
        surf_df["y"] = pd.to_numeric(surf_df["y"], errors="coerce")
        surf_df["slowness"] = pd.to_numeric(surf_df["slowness"], errors="coerce")
        surf_df = surf_df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(surf_df) >= 4:
            try:
                if use_relative:
                    spacing = f"{float(surface_spacing_m):.6f}"
                else:
                    spacing = _surface_spacing_string(
                        local_model,
                        surface_spacing_m,
                    )

                pygmt.surface(
                    data=surf_df,
                    region=plot_region,
                    spacing=spacing,
                    outgrid=str(grid_file),
                )

                fig.grdimage(
                    grid=str(grid_file),
                    cmap=str(cpt_file),
                    nan_transparent=True,
                    transparency=int(surface_alpha),
                )

            except Exception as exc:
                print(
                    "[WARNING] Zoom surface plot failed, "
                    f"continue without surface. Reason: {exc}"
                )
    
    # Rectangle A-B. In relative mode this is the full plotting box, but the labels
    # make the reference point explicit.
    _plot_rectangle_ab(fig, plot_region, pen="1.0p,blue,--", label=True)

    _plot_slowness_points_with_discrete_cpt(
        fig,
        local_model,
        cpt_file,
        marker_size=model_marker_size*5,
    )

    _, nofly = classify_flyable_nofly(
        local_model,
        no_fly_prefixes=no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        always_flyable_prefixes=always_flyable_prefixes,
    )
    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style=f"s{max(model_marker_size / 20.0, 0.055):.3f}c",
            fill=NO_FLY_GMT_FILL,
            pen=NO_FLY_GMT_PEN_THIN,
            label=f"No-fly: s >= {no_fly_slowness_threshold:g}",
        )

    # Optional neighbor edges around each path node, only to see local graph connectivity.
    if show_neighbor_edges:
        for idx in path_indices:
            idx = int(idx)
            if idx not in path_df.index:
                continue
            x0 = float(path_df.loc[idx, "x"])
            y0 = float(path_df.loc[idx, "y"])
            for nb in _get_graph_neighbors(graph, idx):
                nb = int(nb)
                if nb not in local_model.index:
                    continue
                x1 = float(local_model.loc[nb, "x"])
                y1 = float(local_model.loc[nb, "y"])
                fig.plot(x=[x0, x1], y=[y0, y1], pen="0.20p,gray@70")

    if show_adjacent_nodes and adjacent_indices:
        adj_original = model.loc[sorted(adjacent_indices)].copy()
        if use_relative:
            adj_x, adj_y = _relative_xy_from_lower_left(
                adj_original,
                origin_x=origin_x,
                origin_y=origin_y,
                lat_ref=lat_ref,
                is_lonlat=is_lonlat,
            )
            adj_df = _copy_with_plot_xy(adj_original, adj_x, adj_y)
        else:
            adj_df = adj_original
        fig.plot(
            x=adj_df["x"],
            y=adj_df["y"],
            style=f"c{max(model_marker_size / 12.0, 0.075):.3f}c",
            fill="cyan@15",
            pen="0.25p,black",
            label=f"Adjacent nodes: {len(adj_df):,}",
        )

    # Draw path with arrows so each step direction can be checked.
    _plot_path_direction_segments(
        fig,
        path_df,
        path_line_width=path_line_width,
        arrow_every=arrow_every,
    )

    fig.plot(
        x=path_df["x"],
        y=path_df["y"],
        style=f"c{max(model_marker_size / 10.0, 0.09):.3f}c",
        # fill="purple@50",
        pen="0.2p,black",
        label=f"Path nodes: {len(path_df):,}",
    )

    # Step labels for checking movement order.
    if label_path_steps:
        label_step_every = max(1, int(label_step_every))
        step_numbers = np.arange(len(path_df), dtype=int)
        label_mask = (step_numbers % label_step_every) == 0
        # Always include last step.
        label_mask[-1] = True
        label_df = path_df.iloc[label_mask].copy()
        label_text = [str(i) for i in step_numbers[label_mask]]
        fig.text(
            x=label_df["x"],
            y=label_df["y"],
            text=label_text,
            font="6.5p,Helvetica-Bold,black",
            justify="CM",
            fill="white@35",
            clearance="0.04c/0.04c",
        )

    # Start/end markers in the plotting coordinate system.
    start = path_df.iloc[0]
    end = path_df.iloc[-1]
    fig.plot(
        x=[start["x"]],
        y=[start["y"]],
        style="a0.35c",
        fill="yellow",
        pen="0.7p,black",
        label=f"Start: {start['label']}",
    )
    fig.plot(
        x=[end["x"]],
        y=[end["y"]],
        style="s0.32c",
        fill="grey@35",
        pen="1.1p,blue",
        label=f"End: {end['label']}",
    )

    #  ------------- Color bar for background slowness ----------------
    # fig.colorbar(
    #     cmap=str(cpt_file),
    #     position="JMR+w8c/0.35c+o0.8c/0c+v",
    #     frame=_slowness_colorbar_frame(bounds, label="Slowness / cost"),
    # )
    fig.colorbar(
        cmap=str(cpt_file),
        position="JMR+w13c/0.25c+o0.8c/0c+v",
        equalsize="0.5c",
        # frame=["x+lSlowness / cost"],
    )

    if (not use_relative) and is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w500+f+l500 m")
        except Exception:
            pass

    xmin, xmax, ymin, ymax = [float(v) for v in plot_region]
    text = (
        f"A: lower-left reference | "
        f"B: upper-right rectangle | "
        f"{info_coord_text} | "
    )
    fig.text(
        x=xmin + 0.02 * (xmax - xmin),
        y=ymin + 0.09 * (ymax - ymin),
        text=text,
        font="8p,Helvetica-Bold,black",
        justify="BL",
        fill="white@50",
        pen="0.4p,black",
        clearance="0.11c/0.11c",
    )
    text = (
        f"Buffer: {float(buffer_m):.0f} m  | "
        f"Path nodes: {len(path_df):,} | "
        f"Local nodes: {len(local_model):,} | "
        f"Adjacent nodes: {len(adjacent_indices):,}"
    )
    fig.text(
        x=xmin + 0.02 * (xmax - xmin),
        y=ymin + 0.04 * (ymax - ymin),
        text=text,
        font="8p,Helvetica-Bold,black",
        justify="BL",
        fill="white@50",
        pen="0.4p,black",
        clearance="0.11c/0.11c",
    )

    try:
        fig.legend(position="JTL+jTL+o0.15c/0.15c", box="+gwhite@25+p0.5p,black")
    except Exception:
        pass

    fig.savefig(str(figure_file), dpi=dpi)

    if cleanup_temp:
        try:
            cleanup_plot_temp_files(cpt_file, grid_file)
        except Exception:
            pass

    return figure_file


# ============================================================
# Costmap diagnostic plots
# ============================================================

def _get_dynamic_plot_range(values, robust_percentiles=(2, 98), force_min=None, force_max=None):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 1.0

    vmin = float(force_min) if force_min is not None else float(np.nanpercentile(arr, robust_percentiles[0]))
    vmax = float(force_max) if force_max is not None else float(np.nanpercentile(arr, robust_percentiles[1]))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        dv = max(abs(vmin) * 0.05, 1e-6)
        vmin -= dv
        vmax += dv
    return min(vmin, vmax), max(vmin, vmax)


def _estimate_lonlat_spacing_degree(model: pd.DataFrame, spacing_m: float) -> tuple[float, float]:
    lat0 = float(pd.to_numeric(model["y"], errors="coerce").mean())
    lat0_rad = np.deg2rad(lat0)
    cos_lat = max(float(np.cos(lat0_rad)), 1e-8)
    dy_deg = float(spacing_m) / 111_320.0
    dx_deg = float(spacing_m) / (111_320.0 * cos_lat)
    return dx_deg, dy_deg


def _surface_spacing_string(model: pd.DataFrame, spacing_m: float) -> str:
    if detect_lonlat(model):
        dx_deg, dy_deg = _estimate_lonlat_spacing_degree(model, spacing_m)
        return f"{dx_deg:.10f}/{dy_deg:.10f}"
    return f"{float(spacing_m):.6f}"


def plot_costmap_layer(
    model: pd.DataFrame,
    column: str,
    figure_file: Path,
    title: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "turbo",
    max_model_points: int = 300000,
    dpi: int = 300,
    marker_size: float = 2.0,
    robust_percentiles=(2, 98),
    force_min: float | None = None,
    force_max: float | None = None,
    plot_no_fly_overlay: bool = True,
    no_fly_prefixes=(),
    no_fly_slowness_threshold: float = 10.0,
    always_flyable_prefixes=(),
    plot_surface: bool = True,
    surface_spacing_m: float = 20.0,
    surface_tension: float | None = None,
    surface_alpha: int = 0,
    plot_dots: bool = True,
    dot_pen: str | None = None,
    cleanup_temp: bool = True,
):
    """Plot one scalar costmap/slowness layer."""
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if column not in model.columns:
        raise ValueError(f"Column '{column}' not found in model.")

    model = ensure_label_prefix(model)
    full_df = model[["x", "y", column]].copy()
    full_df["x"] = pd.to_numeric(full_df["x"], errors="coerce")
    full_df["y"] = pd.to_numeric(full_df["y"], errors="coerce")
    full_df[column] = pd.to_numeric(full_df[column], errors="coerce")
    full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna()

    if full_df.empty:
        raise ValueError(f"No valid values in column '{column}' to plot.")

    vmin, vmax = _get_dynamic_plot_range(full_df[column], robust_percentiles=robust_percentiles, force_min=force_min, force_max=force_max)

    plot_model = model.copy()
    plot_model[column] = pd.to_numeric(plot_model[column], errors="coerce")
    plot_model = plot_model[np.isfinite(plot_model[column])].copy()
    plot_model = sample_model_for_plot(plot_model, max_model_points=max_model_points)

    region = get_xy_region_with_padding(plot_model, plot_model[["x", "y"]], padding_ratio=0.04)
    is_lonlat = detect_lonlat(plot_model)

    cpt_file = figure_file.with_suffix(".cpt")
    pygmt.makecpt(cmap=cmap, series=[vmin, vmax], continuous=True, output=str(cpt_file))

    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="13p,Helvetica-Bold", FONT_LABEL="10p,Helvetica", FONT_ANNOT_PRIMARY="8p,Helvetica", MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd.xxx")

    if title is None:
        title = column
    if colorbar_label is None:
        colorbar_label = column

    fig.basemap(
        region=region,
        projection="M16c",
        frame=["WSne+t" + str(title), "xaf+lLongitude" if is_lonlat else "xaf+lX", "yaf+lLatitude" if is_lonlat else "yaf+lY"],
    )

    grid_file = None
    if plot_surface:
        grid_file = figure_file.with_suffix(".grd")
        spacing = _surface_spacing_string(full_df, surface_spacing_m)
        surface_kwargs = dict(data=full_df[["x", "y", column]], region=region, spacing=spacing, outgrid=str(grid_file))
        if surface_tension is not None:
            surface_kwargs["tension"] = float(surface_tension)
        try:
            pygmt.surface(**surface_kwargs)
            fig.grdimage(grid=str(grid_file), cmap=str(cpt_file), nan_transparent=True, transparency=int(surface_alpha))
        except Exception as exc:
            print(f"[WARNING] Surface plot failed for {column}; continue with dots only. Reason: {exc}")

    if plot_dots:
        fig.plot(
            x=plot_model["x"],
            y=plot_model["y"],
            style=f"c{max(marker_size / 28.0, 0.04):.3f}c",
            fill=plot_model[column],
            cmap=str(cpt_file),
            pen=dot_pen,
        )

    if plot_no_fly_overlay:
        try:
            _, nofly = classify_flyable_nofly(plot_model, no_fly_prefixes=no_fly_prefixes, no_fly_slowness_threshold=no_fly_slowness_threshold, always_flyable_prefixes=always_flyable_prefixes)
            if not nofly.empty:
                fig.plot(x=nofly["x"], y=nofly["y"], style=f"s{max(marker_size / 18.0, 0.06):.3f}c", fill=NO_FLY_GMT_FILL, pen=NO_FLY_GMT_PEN, label="No-fly")
        except Exception:
            pass

    fig.colorbar(cmap=str(cpt_file), position="JMR+w11.5c/0.35c+o0.8c/0c+v", frame=[f"xaf+l{colorbar_label}"])

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    fig.savefig(str(figure_file), dpi=dpi)

    if cleanup_temp:
        try:
            cleanup_plot_temp_files(cpt_file, grid_file)
        except Exception:
            pass

    return figure_file


def plot_costmap_diagnostics(
    model: pd.DataFrame,
    figure_dir: Path,
    max_model_points: int = 300000,
    dpi: int = 300,
    marker_size: float = 2.0,
    no_fly_prefixes=(),
    no_fly_slowness_threshold: float = 10.0,
    always_flyable_prefixes=(),
    robust_percentiles=(2, 98),
    plot_surface: bool = True,
    surface_spacing_m: float = 20.0,
):
    """Plot available costmap diagnostics."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    spacing_tag = f"{int(surface_spacing_m)}m" if float(surface_spacing_m).is_integer() else f"{surface_spacing_m:g}m"
    outputs = {}

    layer_specs = [
        {"key": "slowness", "column": "slowness", "filename": f"00_slowness_surface_{spacing_tag}.png" if plot_surface else "00_slowness_map.png", "title": "Input slowness / cost map", "label": "Slowness / cost", "cmap": "hot"},
        {"key": "risk_map", "column": "risk_map", "filename": f"01_risk_map_surface_{spacing_tag}.png" if plot_surface else "01_risk_map.png", "title": "Risk map", "label": "Risk index", "cmap": "turbo"},
        {"key": "emergency_score", "column": "emergency_score", "filename": f"02_emergency_score_surface_{spacing_tag}.png" if plot_surface else "02_emergency_score_map.png", "title": "Emergency accessibility map", "label": "Emergency score", "cmap": "viridis"},
        {"key": "base_slowness", "column": "base_slowness", "filename": f"03_base_slowness_surface_{spacing_tag}.png" if plot_surface else "03_base_slowness_map.png", "title": "Base slowness map", "label": "Base slowness", "cmap": "hot"},
        {"key": "effective_slowness", "column": "effective_slowness", "filename": f"04_final_cost_map_effective_slowness_surface_{spacing_tag}.png" if plot_surface else "04_final_cost_map_effective_slowness.png", "title": "Final cost map / effective slowness", "label": "Effective slowness / final cost", "cmap": "hot"},
        {"key": "cost_multiplier", "column": "cost_multiplier", "filename": f"05_cost_multiplier_surface_{spacing_tag}.png" if plot_surface else "05_cost_multiplier_map.png", "title": "Cost multiplier map", "label": "Cost multiplier", "cmap": "turbo"},
        {"key": "emergency_distance_m", "column": "emergency_distance_m", "filename": f"06_emergency_distance_surface_{spacing_tag}.png" if plot_surface else "06_emergency_distance_map.png", "title": "Distance to nearest emergency/safe node", "label": "Emergency distance (m)", "cmap": "turbo"},
    ]

    for spec in layer_specs:
        col = spec["column"]
        if col not in model.columns:
            continue
        fig_file = figure_dir / spec["filename"]
        try:
            print(f"      Plot costmap layer: {fig_file}")
            plot_costmap_layer(
                model=model,
                column=col,
                figure_file=fig_file,
                title=spec["title"],
                colorbar_label=spec["label"],
                cmap=spec["cmap"],
                max_model_points=max_model_points,
                dpi=dpi,
                marker_size=marker_size,
                robust_percentiles=robust_percentiles,
                plot_no_fly_overlay=True,
                no_fly_prefixes=no_fly_prefixes,
                no_fly_slowness_threshold=no_fly_slowness_threshold,
                always_flyable_prefixes=always_flyable_prefixes,
                plot_surface=plot_surface,
                surface_spacing_m=surface_spacing_m,
                plot_dots=True,
            )
            outputs[spec["key"]] = str(fig_file)
        except Exception as exc:
            print(f"[WARNING] Could not plot costmap layer '{col}': {exc}")

    return outputs


def plot_costmap_surface_outputs(
    model: pd.DataFrame,
    figure_dir: Path,
    spacing_m: float = 20.0,
    dpi: int = 300,
    no_fly_slowness_threshold: float = 10.0,
    max_model_points: int = 300000,
    marker_size: float = 2.0,
):
    """Backward-compatible name imported by main.py."""
    return plot_costmap_diagnostics(
        model=model,
        figure_dir=figure_dir,
        max_model_points=max_model_points,
        dpi=dpi,
        marker_size=marker_size,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        plot_surface=True,
        surface_spacing_m=spacing_m,
    )


# ============================================================
# Collision-avoidance time-offset report
# ============================================================

def _safe_float(value, default=np.nan):
    """Convert a value to float; return default for blank/non-finite values."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value, default=0):
    """Convert a value to int; return default when not possible."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _collision_pair_label(item: dict, rank: int) -> str:
    """Compact pair label for collision timeline figures."""
    pair_key = str(item.get("pair_key", "") or "").strip()
    if pair_key:
        return pair_key
    src = str(item.get("source_label", "") or "").strip()
    dst = str(item.get("target_label", "") or "").strip()
    if src or dst:
        return f"{src}->{dst}"
    return f"rank {rank}"


def _collision_results_to_dataframe(path_results, max_paths_plot: int | None = None) -> pd.DataFrame:
    """Convert path_results with time-offset metadata to a clean dataframe."""
    rows = []
    for i, item in enumerate(path_results or [], start=1):
        if not isinstance(item, dict):
            continue

        rank = _safe_int(item.get("rank", i), i)
        schedule_order = _safe_int(item.get("schedule_order", rank), rank)

        duration_s = _safe_float(
            item.get("collision_duration_s", item.get("travel_cost", item.get("total_cost", np.nan)))
        )
        departure_s = _safe_float(item.get("departure_time_s", 0.0), 0.0)
        arrival_s = _safe_float(item.get("arrival_time_s", departure_s + duration_s), departure_s + duration_s)
        delay_s = _safe_float(item.get("collision_delay_s", departure_s), departure_s)

        if not np.isfinite(duration_s) and np.isfinite(arrival_s) and np.isfinite(departure_s):
            duration_s = max(0.0, arrival_s - departure_s)
        if not np.isfinite(arrival_s) and np.isfinite(departure_s) and np.isfinite(duration_s):
            arrival_s = departure_s + duration_s
        if not np.isfinite(delay_s):
            delay_s = max(0.0, departure_s)

        if not (np.isfinite(duration_s) and np.isfinite(departure_s) and np.isfinite(arrival_s)):
            continue

        rows.append(
            {
                "rank": rank,
                "schedule_order": schedule_order,
                "pair_key": _collision_pair_label(item, rank),
                "source_label": item.get("source_label", ""),
                "target_label": item.get("target_label", ""),
                "departure_time_s": float(departure_s),
                "arrival_time_s": float(arrival_s),
                "duration_s": float(duration_s),
                "collision_delay_s": float(max(0.0, delay_s)),
                "departure_time_min": float(departure_s / 60.0),
                "arrival_time_min": float(arrival_s / 60.0),
                "duration_min": float(duration_s / 60.0),
                "collision_delay_min": float(max(0.0, delay_s) / 60.0),
                "collision_free": item.get("collision_free", ""),
                "collision_min_distance_m": item.get("collision_min_distance_m", ""),
                "collision_blocking_pair_key": item.get("collision_blocking_pair_key", ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["schedule_order", "rank", "departure_time_s"]).reset_index(drop=True)
    if max_paths_plot is not None:
        try:
            max_paths_plot = int(max_paths_plot)
            if max_paths_plot > 0:
                df = df.head(max_paths_plot).copy()
        except Exception:
            pass

    # y_plot puts the first scheduled path at the top of the timeline.
    n = len(df)
    df["y_plot"] = np.arange(n, 0, -1, dtype=float)
    df["plot_id"] = np.arange(1, n + 1, dtype=int)
    df["label"] = df.apply(
        lambda r: f"{int(r['rank']):03d} {r['pair_key']}", axis=1
    )
    return df


def plot_collision_time_offset_report(
    path_results,
    figure_file: Path,
    algorithm_name: str,
    dpi: int = 300,
    max_paths_plot: int | None = None,
):
    """Plot start-time offsets and travel-time increase from collision avoidance.

    This figure is intended for FMM2D_COLLISION_AVOIDANCE_MODE="time_offset".

    Top panel:
        horizontal schedule timeline for each path.
        dashed orange segment = waiting/departure delay.
        black segment = flight interval from departure to arrival.

    Bottom panel:
        collision-delay / travel-time increase by path rank.
        Because the spatial path is unchanged, the increase in scheduled
        completion time equals the assigned departure delay.
    """
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    df = _collision_results_to_dataframe(path_results, max_paths_plot=max_paths_plot)
    if df.empty:
        raise ValueError("No path_results with collision timing columns available to plot.")

    n = len(df)
    max_arrival_min = float(max(df["arrival_time_min"].max(), 1.0))
    max_delay_min = float(max(df["collision_delay_min"].max(), 1.0 / 60.0))
    xpad = max(max_arrival_min * 0.05, 0.5)
    x_max = max_arrival_min + xpad

    fig = pygmt.Figure()
    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
    )

    # Dynamic panel height: readable for few paths, not absurdly tall for many paths.
    timeline_height = min(max(5.5, 0.22 * n + 2.0), 18.0)

    fig.basemap(
        region=[0.0, x_max, 0.5, n + 0.5],
        projection=f"X18c/{timeline_height:.2f}c",
        frame=[
            f"WSne+tCollision-avoidance schedule - {algorithm_name}",
            "xaf+lTime from mission start (min)",
            "yaf+lScheduled path order",
        ],
    )

    # Draw each scheduled path.
    for _, row in df.iterrows():
        y = float(row["y_plot"])
        dep = float(row["departure_time_min"])
        arr = float(row["arrival_time_min"])
        dur = float(row["duration_min"])
        delay = float(row["collision_delay_min"])

        if delay > 0.0:
            fig.plot(x=[0.0, dep], y=[y, y], pen="1.0p,orange,--")

        fig.plot(x=[dep, arr], y=[y, y], pen="2.2p,black")
        fig.plot(x=[dep], y=[y], style="t0.18c", fill="yellow", pen="0.35p,black")
        fig.plot(x=[arr], y=[y], style="s0.15c", fill="dodgerblue", pen="0.35p,black")

        # Label only when not too crowded.
        if n <= 80:
            label = str(row["label"])
            if len(label) > 28:
                label = label[:25] + "..."
            fig.text(
                x=0.02 * x_max,
                y=y + 0.18,
                text=label,
                font="6.5p,Helvetica-Bold,black",
                justify="LB",
                fill="white@55",
                clearance="0.03c/0.03c",
            )

        if n <= 50 and (delay > 0.0 or dur > 0.0):
            txt = f"start={dep:.1f}m, dur={dur:.1f}m"
            fig.text(
                x=min(arr + 0.01 * x_max, x_max - 0.01 * x_max),
                y=y,
                text=txt,
                font="6p,Helvetica,black",
                justify="LM",
            )

    # Legend-like text box.
    avoided = int(np.count_nonzero(pd.to_numeric(df["collision_delay_s"], errors="coerce") > 0.0))
    max_delay = float(df["collision_delay_min"].max())
    txt = (
        f"Paths: {n:,} | delayed paths: {avoided:,} | "
        f"max start delay: {max_delay:.2f} min | "
        "orange dashed = waiting, black = flight"
    )
    fig.text(
        x=0.02 * x_max,
        y=0.80,
        text=txt,
        font="8p,Helvetica-Bold,black",
        justify="LB",
        fill="white@35",
        pen="0.35p,black",
        clearance="0.08c/0.08c",
    )

    # ------------------------------------------------------------
    # Bottom panel: delay/increase by rank/order
    # ------------------------------------------------------------
    fig.shift_origin(yshift=f"-{timeline_height + 1.1:.2f}c")

    y_max = max(max_delay_min * 1.20, 1.0)
    fig.basemap(
        region=[0.5, n + 0.5, 0.0, y_max],
        projection="X18c/5.0c",
        frame=[
            "WSne+tTravel-time increase from collision avoidance",
            "xaf+lScheduled path order",
            "yaf+lIncrease / start delay (min)",
        ],
    )

    # Vertical bars from 0 to delay.
    for _, row in df.iterrows():
        x = float(row["plot_id"])
        delay = float(row["collision_delay_min"])
        pen = "1.8p,red" if delay > 0.0 else "0.8p,gray"
        fig.plot(x=[x, x], y=[0.0, delay], pen=pen)

    fig.plot(
        x=df["plot_id"],
        y=df["collision_delay_min"],
        style="c0.10c",
        fill="red",
        pen="0.2p,black",
    )

    if max_delay_min <= 0.0:
        fig.text(
            x=0.5 + 0.5 * n,
            y=0.5 * y_max,
            text="No time increase: all paths can start at the base time",
            font="10p,Helvetica-Bold,black",
            justify="CM",
            fill="white@35",
            pen="0.35p,black",
            clearance="0.10c/0.10c",
        )

    fig.savefig(str(figure_file), dpi=dpi)
    return figure_file
