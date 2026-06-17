#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Path connection builder for LAE-UTM.

This module:
  1. Reads the input pathfinding model.
  2. Extracts DB and DK nodes.
  3. Builds possible one-way connections:
       - DB to DK
       - DK to DK
       - optional DB to DB
  4. Saves:
       - output/senario1/paths.csv
       - figure/senario1/paths.png
"""

from __future__ import annotations

import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm, colors

# ============================================================
# Small parameter helpers
# ============================================================

def _get_param(params: Any, names: list[str], default: Any = None) -> Any:
    """
    Read parameter value from a module/object/dict using several possible names.
    """
    if params is None:
        return default

    if isinstance(params, dict):
        for name in names:
            if name in params:
                return params[name]
        return default

    for name in names:
        if hasattr(params, name):
            return getattr(params, name)

    return default


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


# ============================================================
# Input model reading
# ============================================================

def _first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                return s

    raise ValueError(f"No valid data line found in: {path}")


def read_model_table(
    model_path: str | Path,
    model_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Read model file automatically.

    Supports:
      - whitespace-separated XYZ-like file
      - comma-separated CSV file
      - header or no header

    Important:
      If model_columns is given, use it for headerless model files.
      For your current model, this should be:
          ("lon", "lat", "z", "slowness", "label")
    """
    model_path = _as_path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Input model file does not exist: {model_path}")

    first_line = _first_data_line(model_path)
    sep = "," if "," in first_line else r"\s+"

    df0 = pd.read_csv(
        model_path,
        sep=sep,
        comment="#",
        header=None,
        engine="python",
    )

    if df0.empty:
        raise ValueError(f"Input model is empty: {model_path}")

    first_row = [str(v).strip().lower() for v in df0.iloc[0].tolist()]

    header_keywords = {
        "id",
        "idx",
        "index",
        "node",
        "node_id",
        "node_label",
        "label",
        "class",
        "type",
        "name",
        "x",
        "y",
        "z",
        "lon",
        "lat",
        "longitude",
        "latitude",
        "easting",
        "northing",
        "slowness",
        "velocity",
    }

    has_header = any(v in header_keywords for v in first_row)
    has_header = has_header or any("label" in v for v in first_row)
    has_header = has_header or any("class" in v for v in first_row)

    if has_header:
        df = pd.read_csv(
            model_path,
            sep=sep,
            comment="#",
            header=0,
            engine="python",
        )
        df.columns = [str(c).strip() for c in df.columns]
        return df

    df = df0.copy()

    if model_columns is not None:
        model_columns = [str(c).strip() for c in model_columns]
        n_cols = df.shape[1]

        if n_cols == len(model_columns):
            df.columns = model_columns

        elif n_cols == len(model_columns) + 1:
            # Common alternative:
            # index lon lat z slowness label
            df.columns = ["index"] + model_columns

        elif n_cols > len(model_columns):
            extra_cols = [f"extra_{i}" for i in range(n_cols - len(model_columns))]
            df.columns = model_columns + extra_cols

        else:
            print(
                "[WARN] MODEL_COLUMNS length is larger than data columns. "
                "Use automatic c0, c1, c2... column names."
            )
            df.columns = [f"c{i}" for i in range(n_cols)]

    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]

    return df


# ============================================================
# Column inference
# ============================================================

def _numeric_valid_ratio(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").notna().mean()


def _looks_like_lon_lat_pair(x: pd.Series, y: pd.Series) -> bool:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")

    if x_num.notna().mean() < 0.8 or y_num.notna().mean() < 0.8:
        return False

    x_min, x_max = x_num.min(), x_num.max()
    y_min, y_max = y_num.min(), y_num.max()

    return (
        -180.0 <= x_min <= 180.0
        and -180.0 <= x_max <= 180.0
        and -90.0 <= y_min <= 90.0
        and -90.0 <= y_max <= 90.0
    )


def infer_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Infer x/y columns.

    Priority:
      1. Header names: lon/lat, x/y, easting/northing.
      2. Headerless lon/lat model: c0/c1.
      3. Headerless indexed model: c1/c2.
      4. First two numeric columns.
    """
    lower_map = {str(c).lower(): str(c) for c in df.columns}

    x_candidates = ["lon", "longitude", "x", "utm_x", "easting", "east"]
    y_candidates = ["lat", "latitude", "y", "utm_y", "northing", "north"]

    x_col = None
    y_col = None

    for name in x_candidates:
        if name in lower_map:
            x_col = lower_map[name]
            break

    for name in y_candidates:
        if name in lower_map:
            y_col = lower_map[name]
            break

    if x_col is not None and y_col is not None:
        return x_col, y_col

    numeric_cols = [
        str(c)
        for c in df.columns
        if _numeric_valid_ratio(df[c]) > 0.8
    ]

    if len(numeric_cols) < 2:
        raise ValueError("Cannot infer x/y columns. Need at least two numeric columns.")

    # Headerless lon/lat model:
    # c0 = lon
    # c1 = lat
    # c2 = z
    if "c0" in df.columns and "c1" in df.columns:
        if _looks_like_lon_lat_pair(df["c0"], df["c1"]):
            return "c0", "c1"

    # Headerless indexed lon/lat model:
    # c0 = index
    # c1 = lon
    # c2 = lat
    if "c1" in df.columns and "c2" in df.columns:
        if _looks_like_lon_lat_pair(df["c1"], df["c2"]):
            return "c1", "c2"

    # Headerless indexed x/y model:
    # c0 = index
    # c1 = x
    # c2 = y
    if "c0" in df.columns and "c1" in df.columns and "c2" in df.columns:
        c0 = pd.to_numeric(df["c0"], errors="coerce")
        c1 = pd.to_numeric(df["c1"], errors="coerce")
        c2 = pd.to_numeric(df["c2"], errors="coerce")

        if c0.notna().mean() > 0.8 and c1.notna().mean() > 0.8 and c2.notna().mean() > 0.8:
            return "c1", "c2"

    return numeric_cols[0], numeric_cols[1]


def infer_index_column(df: pd.DataFrame) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}

    for name in ["index", "idx", "id", "node_index"]:
        if name in lower_map:
            return lower_map[name]

    if "c0" in df.columns and _numeric_valid_ratio(df["c0"]) > 0.8:
        return "c0"

    return None


def infer_label_column(df: pd.DataFrame) -> str | None:
    """
    Find column containing DB/DK labels.
    """
    lower_map = {str(c).lower(): str(c) for c in df.columns}

    preferred = [
        "label",
        "node_label",
        "node_id",
        "name",
        "class",
        "type",
        "category",
    ]

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    best_col = None
    best_score = 0

    for c in df.columns:
        text = df[c].astype(str).str.strip()

        score = text.str.match(r"^(DB|DK)\d*$", case=False, na=False).sum()
        score += text.str.contains("Drone-Base|Drone_Base|Docking", case=False, na=False).sum()

        if score > best_score:
            best_score = int(score)
            best_col = str(c)

    if best_score > 0:
        return best_col

    return None


# ============================================================
# Node extraction
# ============================================================

def classify_node(value: Any) -> str | None:
    """
    Return:
      DB, DK, or None
    """
    text = str(value).strip().upper()

    if re.match(r"^DB\d*$", text):
        return "DB"

    if re.match(r"^DK\d*$", text):
        return "DK"

    if text in {"DRONE-BASE", "DRONE_BASE", "DRONEBASE"}:
        return "DB"

    if text == "DOCKING":
        return "DK"

    return None


def normalize_node_id(raw_label: Any, node_type: str, counter: int) -> str:
    raw = str(raw_label).strip().upper()

    if re.match(r"^(DB|DK)\d+$", raw):
        return raw

    return f"{node_type}{counter:02d}"


def extract_db_dk_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract DB and DK nodes from model dataframe.

    Returns dataframe with:
      node_id, node_type, model_index, x, y, source_label
    """
    x_col, y_col = infer_xy_columns(df)
    index_col = infer_index_column(df)
    label_col = infer_label_column(df)

    if label_col is None:
        raise ValueError(
            "Cannot find DB/DK label column. "
            "The model should contain labels such as DB01, DK01, "
            "or class names such as Drone-Base and Docking."
        )

    records = []
    counters = {"DB": 0, "DK": 0}

    for _, row in df.iterrows():
        raw_label = row[label_col]
        node_type = classify_node(raw_label)

        if node_type is None:
            continue

        x = pd.to_numeric(row[x_col], errors="coerce")
        y = pd.to_numeric(row[y_col], errors="coerce")

        if pd.isna(x) or pd.isna(y):
            continue

        counters[node_type] += 1
        node_id = normalize_node_id(raw_label, node_type, counters[node_type])

        model_index = row[index_col] if index_col is not None else None

        records.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "model_index": model_index,
                "x": float(x),
                "y": float(y),
                "source_label": str(raw_label),
            }
        )

    nodes = pd.DataFrame(records)

    if nodes.empty:
        raise ValueError("No DB or DK nodes were found in the model.")

    nodes = nodes.drop_duplicates(subset=["node_id"], keep="first")
    nodes = nodes.sort_values(["node_type", "node_id"]).reset_index(drop=True)

    return nodes


# ============================================================
# Path connection generation
# ============================================================

def _distance_m(a: pd.Series, b: pd.Series) -> float:
    return float(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def _make_path_record(
    path_number: int,
    start_node: pd.Series,
    end_node: pd.Series,
    path_type: str,
) -> dict[str, Any]:
    """
    Build one path row.
    """

    return {
        "path_index": path_number,
        "path_id": f"P{path_number:05d}",
        "connection_id": f"{start_node['node_id']}__{end_node['node_id']}",
        "path_type": path_type,

        "start_id": start_node["node_id"],
        "end_id": end_node["node_id"],

        "start_type": start_node["node_type"],
        "end_type": end_node["node_type"],

        "start_model_index": start_node["model_index"],
        "end_model_index": end_node["model_index"],

        "start_x": start_node["x"],
        "start_y": start_node["y"],
        "end_x": end_node["x"],
        "end_y": end_node["y"],

        "distance_m": _distance_m(start_node, end_node),
    }


def build_connection_table(
    nodes: pd.DataFrame,
    include_db_dk: bool = True,
    include_dk_dk: bool = True,
    include_db_db: bool = True,
) -> pd.DataFrame:
    """
    Estimate all possible one-way connections.

    One-way rule:
      DB-DK: DB -> DK only
      DK-DK: combinations only, no reverse duplicate
      DB-DB: combinations only, no reverse duplicate
    """
    db = nodes[nodes["node_type"] == "DB"].sort_values("node_id").reset_index(drop=True)
    dk = nodes[nodes["node_type"] == "DK"].sort_values("node_id").reset_index(drop=True)

    records = []
    path_number = 1

    if include_db_dk:
        for _, start_node in db.iterrows():
            for _, end_node in dk.iterrows():
                records.append(
                    _make_path_record(
                        path_number,
                        start_node,
                        end_node,
                        "DB_DK",
                    )
                )
                path_number += 1

    if include_dk_dk:
        for i, j in combinations(dk.index, 2):
            records.append(
                _make_path_record(
                    path_number,
                    dk.loc[i],
                    dk.loc[j],
                    "DK_DK",
                )
            )
            path_number += 1

    if include_db_db:
        for i, j in combinations(db.index, 2):
            records.append(
                _make_path_record(
                    path_number,
                    db.loc[i],
                    db.loc[j],
                    "DB_DB",
                )
            )
            path_number += 1

    paths = pd.DataFrame(records)

    if paths.empty:
        raise ValueError("No paths were generated. Please check DB/DK nodes.")

    return paths


# ============================================================
# Plotting
# ============================================================

def _set_plot_region(
    ax,
    nodes: pd.DataFrame,
    model_df: pd.DataFrame | None = None,
    model_x_col: str | None = None,
    model_y_col: str | None = None,
    pad_fraction: float = 0.03,
) -> None:
    """
    Set plot region.

    Prefer full model extent if available.
    Otherwise use DB/DK node extent.
    """
    if model_df is not None and model_x_col is not None and model_y_col is not None:
        x = pd.to_numeric(model_df[model_x_col], errors="coerce")
        y = pd.to_numeric(model_df[model_y_col], errors="coerce")
    else:
        x = pd.to_numeric(nodes["x"], errors="coerce")
        y = pd.to_numeric(nodes["y"], errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if x.empty or y.empty:
        return

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    dx = xmax - xmin
    dy = ymax - ymin

    if dx <= 0:
        dx = max(abs(xmin) * 1e-6, 1e-6)
    if dy <= 0:
        dy = max(abs(ymin) * 1e-6, 1e-6)

    xpad = dx * pad_fraction
    ypad = dy * pad_fraction

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def _set_plot_region(
    ax,
    nodes: pd.DataFrame,
    model_df: pd.DataFrame | None = None,
    model_x_col: str | None = None,
    model_y_col: str | None = None,
    pad_fraction: float = 0.03,
) -> None:
    """
    Set plot region.

    Prefer full model extent if available.
    Otherwise use DB/DK node extent.
    """
    if model_df is not None and model_x_col is not None and model_y_col is not None:
        x = pd.to_numeric(model_df[model_x_col], errors="coerce")
        y = pd.to_numeric(model_df[model_y_col], errors="coerce")
    else:
        x = pd.to_numeric(nodes["x"], errors="coerce")
        y = pd.to_numeric(nodes["y"], errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if x.empty or y.empty:
        return

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    dx = xmax - xmin
    dy = ymax - ymin

    if dx <= 0:
        dx = max(abs(xmin) * 1e-6, 1e-6)
    if dy <= 0:
        dy = max(abs(ymin) * 1e-6, 1e-6)

    xpad = dx * pad_fraction
    ypad = dy * pad_fraction

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def plot_paths(
    nodes: pd.DataFrame,
    paths: pd.DataFrame,
    output_fig: str | Path,
    title: str = "Possible One-Way Path Connections",
    dpi: int = 300,
    model_df: pd.DataFrame | None = None,
    model_x_col: str | None = None,
    model_y_col: str | None = None,
    region_pad_fraction: float = 0.03,
    cmap_name: str = "viridis",
    font_family: str = "STIXGeneral",
    cbar_shrink: float = 0.82,
    cbar_fraction: float = 0.045,
    cbar_pad: float = 0.02,
    cbar_aspect: int = 28,
) -> None:
    """
    Plot nodes and path connections.

    - line color is controlled by discrete path_index
    - colorbar is discrete
    - use a fancy serif font
    """
    output_fig = _as_path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    if paths.empty:
        raise ValueError("No paths available for plotting.")

    paths = paths.copy()

    if "path_index" not in paths.columns:
        paths["path_index"] = np.arange(1, len(paths) + 1)

    paths["path_index"] = pd.to_numeric(paths["path_index"], errors="coerce")
    paths = paths.dropna(subset=["path_index"]).copy()
    paths["path_index"] = paths["path_index"].astype(int)

    idx_min = int(paths["path_index"].min())
    idx_max = int(paths["path_index"].max())

    # --------------------------------------------------------
    # Discrete colormap / norm
    # --------------------------------------------------------
    n_paths = idx_max - idx_min + 1

    # Get exactly n_paths discrete colors
    cmap = cm.get_cmap(cmap_name, n_paths)

    # Boundaries centered on integers:
    # path 1 => [0.5, 1.5], path 2 => [1.5, 2.5], ...
    boundaries = np.arange(idx_min - 0.5, idx_max + 1.5, 1.0)
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    # Plot colored lines by path_index
    for _, p in paths.iterrows():
        idx = int(p["path_index"])
        line_color = cmap(idx - idx_min)

        ax.plot(
            [p["start_x"], p["end_x"]],
            [p["start_y"], p["end_y"]],
            color=line_color,
            linewidth=1.6,
            alpha=0.92,
            zorder=2,
        )

    # Plot nodes
    db = nodes[nodes["node_type"] == "DB"]
    dk = nodes[nodes["node_type"] == "DK"]

    if not dk.empty:
        ax.scatter(
            dk["x"],
            dk["y"],
            s=70,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            label="DK",
            zorder=5,
        )

    if not db.empty:
        ax.scatter(
            db["x"],
            db["y"],
            s=120,
            marker="s",
            facecolor="white",
            edgecolor="black",
            linewidth=2.0,
            label="DB",
            zorder=6,
        )

    # Node labels
    for _, n in nodes.iterrows():
        ax.text(
            n["x"],
            n["y"],
            f" {n['node_id']}",
            fontsize=9,
            ha="left",
            va="center",
            zorder=7,
            fontfamily=font_family,
        )

    _set_plot_region(
        ax=ax,
        nodes=nodes,
        model_df=model_df,
        model_x_col=model_x_col,
        model_y_col=model_y_col,
        pad_fraction=region_pad_fraction,
    )

    ax.set_title(title, fontsize=16, fontfamily=font_family, fontweight="bold")

    if model_x_col is not None and str(model_x_col).lower() in {"lon", "longitude"}:
        ax.set_xlabel("Longitude", fontsize=13, fontfamily=font_family)
    else:
        ax.set_xlabel("X / Easting", fontsize=13, fontfamily=font_family)

    if model_y_col is not None and str(model_y_col).lower() in {"lat", "latitude"}:
        ax.set_ylabel("Latitude", fontsize=13, fontfamily=font_family)
    else:
        ax.set_ylabel("Y / Northing", fontsize=13, fontfamily=font_family)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(10)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(10)

    legend = ax.legend(loc="best", frameon=True, fontsize=10)
    for txt in legend.get_texts():
        txt.set_fontfamily(font_family)

    # --------------------------------------------------------
    # Discrete colorbar
    # --------------------------------------------------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    tick_step = max(1, int(np.ceil(n_paths / 15)))
    ticks = np.arange(idx_min, idx_max + 1, tick_step)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        boundaries=boundaries,
        ticks=ticks,
        spacing="proportional",
        pad=cbar_pad,
        shrink=cbar_shrink,
        fraction=cbar_fraction,
        aspect=cbar_aspect,
        drawedges=True,
    )
    cbar.set_label("Path index", fontsize=12, fontfamily=font_family)

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily(font_family)
        tick.set_fontsize(10)

    fig.tight_layout()
    fig.savefig(output_fig, dpi=dpi)
    plt.close(fig)

# ============================================================
# Main public function
# ============================================================

def create_possible_paths(
    params: Any | None = None,
    model_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_fig: str | Path | None = None,
    include_db_dk: bool | None = None,
    include_dk_dk: bool | None = None,
    include_db_db: bool | None = None,
    make_figure: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main callable function.

    Can be called from main.py:

        import parameters as prm
        from src.paths import create_possible_paths

        paths, nodes = create_possible_paths(prm)

    Returns:
      paths, nodes
    """

    model_path = model_path or _get_param(
        params,
        [
            "PATH_INPUT_MODEL",
            "MODEL_FILE",
            "INPUT_MODEL_PATH",
            "INPUT_MODEL_FILE",
            "INPUT_MODEL_XYZ",
            "PATHFINDING_MODEL",
            "PATHFINDING_MODEL_XYZ",
        ],
        Path("model") / "senario1" / "model_senario1_cost_for_pathfinding.xyz",
    )

    output_csv = output_csv or _get_param(
        params,
        ["PATH_OUTPUT_CSV", "PATHS_OUTPUT_CSV", "OUTPUT_PATHS_CSV"],
        Path("output") / "senario1" / "paths.csv",
    )

    output_fig = output_fig or _get_param(
        params,
        ["PATH_OUTPUT_FIG", "PATHS_OUTPUT_FIG", "OUTPUT_PATHS_FIG"],
        Path("figure") / "senario1" / "paths.png",
    )

    include_db_dk = (
        include_db_dk
        if include_db_dk is not None
        else _get_param(params, ["PATH_INCLUDE_DB_DK"], True)
    )

    include_dk_dk = (
        include_dk_dk
        if include_dk_dk is not None
        else _get_param(params, ["PATH_INCLUDE_DK_DK"], True)
    )

    include_db_db = (
        include_db_db
        if include_db_db is not None
        else _get_param(params, ["PATH_INCLUDE_DB_DB"], True)
    )

    make_figure = (
        make_figure
        if make_figure is not None
        else _get_param(params, ["PATH_MAKE_FIGURE"], True)
    )

    model_path = _as_path(model_path)
    output_csv = _as_path(output_csv)
    output_fig = _as_path(output_fig)

    print("\n========== CREATE POSSIBLE CONNECTION PATHS ==========")
    print(f"Input model : {model_path}")
    print(f"Output CSV  : {output_csv}")
    print(f"Output fig  : {output_fig}")

    model_columns = _get_param(
        params,
        ["PATH_MODEL_COLUMNS", "MODEL_COLUMNS"],
        None,
    )

    model_df = read_model_table(
        model_path,
        model_columns=model_columns,
    )

    model_x_col, model_y_col = infer_xy_columns(model_df)
    nodes = extract_db_dk_nodes(model_df)

    paths = build_connection_table(
        nodes,
        include_db_dk=bool(include_db_dk),
        include_dk_dk=bool(include_dk_dk),
        include_db_db=bool(include_db_db),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output_csv, index=False)

    if make_figure:
        plot_paths(
            nodes=nodes,
            paths=paths,
            output_fig=output_fig,
            model_df=model_df,
            model_x_col=model_x_col,
            model_y_col=model_y_col,
            region_pad_fraction=float(
                _get_param(params, ["PATH_REGION_PAD_FRACTION"], 0.03)
            ),
            cmap_name=str(
                _get_param(params, ["PATH_LINE_CMAP"], "viridis")
            ),
            font_family=str(
                _get_param(params, ["PATH_FANCY_FONT"], "STIXGeneral")
            ),
            cbar_shrink=float(
                _get_param(params, ["PATH_COLORBAR_SHRINK"], 0.82)
            ),
            cbar_fraction=float(
                _get_param(params, ["PATH_COLORBAR_FRACTION"], 0.045)
            ),
            cbar_pad=float(
                _get_param(params, ["PATH_COLORBAR_PAD"], 0.02)
            ),
            cbar_aspect=int(
                _get_param(params, ["PATH_COLORBAR_ASPECT"], 28)
            ),
        )

    print("\n========== PATH SUMMARY ==========")
    print(f"DB nodes    : {(nodes['node_type'] == 'DB').sum()}")
    print(f"DK nodes    : {(nodes['node_type'] == 'DK').sum()}")
    print(f"Total paths : {len(paths)}")
    print(paths["path_type"].value_counts().to_string())

    print("\n[OK] Saved:")
    print(f"  CSV : {output_csv}")
    if make_figure:
        print(f"  FIG : {output_fig}")

    return paths, nodes