#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_v4.py

LAE-UTM node-based Theta* master route planner, v4.

Purpose
-------
This script is the master planning wrapper. It does NOT implement Theta*.
It reads a node-based 2D riskmap, prepares route constraints, builds the graph,
then calls the existing Theta* backend:

    src/thetastar.py  -> run(model, graph, start_idx, end_idx, **kwargs)

Route concept in v1
-------------------
- DB and DK are normal route endpoints.
- FLZ is not a normal endpoint by default. It is used as emergency-safety
  attraction, so routes prefer to stay closer to FLZ when possible.
- RA and obstacle nodes remain hard no-fly.
- Each DB/DK pair gets configurable route sets per direction:
    forward_main_01, forward_backup_01, ...
    backward_main_01, backward_backup_01, ...

Important safety fix in v1
--------------------------
Theta* may export any-angle segments. This script validates every segment using
Bresenham grid traversal. If a segment crosses obstacle/RA/no-fly cells, the
route is rejected and retried in safe A*-delegate mode.

Run
---
    python main_v4.py --param-file params/thetastar.params

Expected input model columns
----------------------------
Required:
    x y slowness label label_prefix
Optional:
    node_id z risk_obstacle risk_ra risk_total obstacle_flag ra_flag objective_flag

Outputs
-------
    OUTPUT_DIR/
        objective_table.csv
        planning_model_with_flz_support.xyz
        route_summary.csv
        all_route_edges.csv
        route_nodes/*.csv
        figures/*.png
        00_all_theta_routes_overview_v4.png
"""

from __future__ import annotations

import argparse
import ast
import inspect
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# Import project backend
# ======================================================================

try:
    from src.thetastar import run as theta_run
except Exception:
    try:
        from thetastar import run as theta_run
    except Exception as exc:
        raise ImportError(
            "Cannot import Theta* backend. Put thetastar.py in src/thetastar.py "
            "or in the same directory as this main_v4.py."
        ) from exc

try:
    from src.model_io import build_grid_graph
except Exception as exc:
    raise ImportError(
        "Cannot import src.model_io.build_grid_graph. This planner expects the "
        "existing LAE-UTM project structure with src/model_io.py."
    ) from exc


VERSION = "v4"


# ======================================================================
# Parameter loading
# ======================================================================

def parse_value(raw: str) -> Any:
    raw = raw.strip()

    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "none":
        return None

    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip('"').strip("'")


def load_params(param_file: str | Path) -> dict[str, Any]:
    param_file = Path(param_file)

    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    params: dict[str, Any] = {}

    with open(param_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "#" in line:
                line = line.split("#", 1)[0].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            params[key.strip()] = parse_value(value.strip())

    return params


def pget(params: dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAE-UTM Theta* master route planner v2")
    parser.add_argument(
        "--param-file",
        default="params/thetastar.params",
        help="Path to params/thetastar.params",
    )
    return parser.parse_args()


# ======================================================================
# Model utilities
# ======================================================================

def read_node_model(model_file: str | Path) -> pd.DataFrame:
    model_file = Path(model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = pd.read_csv(model_file, sep=r"\s+", engine="python")
    df = df.reset_index(drop=True)

    if "node_id" not in df.columns:
        df.insert(0, "node_id", np.arange(len(df), dtype=int))

    if "z" not in df.columns:
        df["z"] = 0.0

    if "label" not in df.columns:
        df["label"] = "NONE"

    if "label_prefix" not in df.columns:
        df["label_prefix"] = "NONE"

    if "objective_flag" not in df.columns:
        df["objective_flag"] = (df["label_prefix"].astype(str) != "NONE").astype(int)

    required = ["x", "y", "z", "slowness", "label", "label_prefix"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input model is missing required column: {col}")

    df["label"] = df["label"].fillna("NONE").astype(str)
    df["label_prefix"] = df["label_prefix"].fillna("NONE").astype(str)

    return df


def infer_grid_spacing(values: np.ndarray) -> float:
    vals = np.unique(np.round(values.astype(float), 6))
    if len(vals) < 2:
        return 1.0

    diffs = np.diff(vals)
    diffs = diffs[diffs > 1.0e-9]

    if len(diffs) == 0:
        return 1.0

    return float(np.median(diffs))


def add_grid_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, int], int], float]:
    dx = infer_grid_spacing(df["x"].to_numpy(float))
    dy = infer_grid_spacing(df["y"].to_numpy(float))
    grid_m = float(np.median([dx, dy]))

    xmin = float(df["x"].min())
    ymin = float(df["y"].min())

    work = df.copy()
    work["_ix"] = np.rint((work["x"].to_numpy(float) - xmin) / grid_m).astype(int)
    work["_iy"] = np.rint((work["y"].to_numpy(float) - ymin) / grid_m).astype(int)

    cell_to_idx: dict[tuple[int, int], int] = {}
    for idx, row in work.iterrows():
        cell_to_idx[(int(row["_ix"]), int(row["_iy"]))] = int(idx)

    return work, cell_to_idx, grid_m


def build_graph_for_model(model: pd.DataFrame, params: dict[str, Any]):
    """Build graph using the existing project build_grid_graph()."""
    graph_kwargs: dict[str, Any] = {}

    possible_keys = [
        "CONNECTIVITY",
        "NEIGHBOR_MODE",
        "NEIGHBOR_RADIUS_M",
        "NOFLY_SLOWNESS_THRESHOLD",
        "SLOWNESS_NOFLY_THRESHOLD",
    ]

    for key in possible_keys:
        if key in params:
            graph_kwargs[key] = params[key]
            graph_kwargs[key.lower()] = params[key]

    try:
        sig = inspect.signature(build_grid_graph)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if accepts_kwargs:
            return build_grid_graph(model, **graph_kwargs)

        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in graph_kwargs.items() if k in allowed}

        if "model" in allowed:
            return build_grid_graph(model=model, **filtered)

        return build_grid_graph(model, **filtered)

    except TypeError:
        return build_grid_graph(model)


def apply_allowed_mask_to_model(
    model: pd.DataFrame,
    allowed_mask: np.ndarray,
    nofly_value: float,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> pd.DataFrame:
    work = model.copy()
    allowed = allowed_mask.copy()

    if start_idx is not None:
        allowed[int(start_idx)] = True
    if end_idx is not None:
        allowed[int(end_idx)] = True

    work.loc[~allowed, "slowness"] = float(nofly_value)

    return work


# ======================================================================
# FLZ emergency support
# ======================================================================

def compute_flz_support(df: pd.DataFrame, sigma_m: float) -> np.ndarray:
    flz = df[df["label_prefix"] == "FLZ"]

    if len(flz) == 0:
        return np.zeros(len(df), dtype=float)

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    support = np.zeros(len(df), dtype=float)
    sigma_m = max(float(sigma_m), 1.0)

    for _, row in flz.iterrows():
        dx = x - float(row["x"])
        dy = y - float(row["y"])
        d2 = dx * dx + dy * dy
        s = np.exp(-d2 / (2.0 * sigma_m * sigma_m))
        support = np.maximum(support, s)

    return support


def apply_flz_safety_attraction(
    df: pd.DataFrame,
    params: dict[str, Any],
    nofly_threshold: float,
) -> pd.DataFrame:
    """
    Reduce flyable-node slowness near FLZ.

    This makes routes prefer emergency-support corridors without making no-fly
    cells flyable.
    """
    use_flz = bool(pget(params, "USE_FLZ_SAFETY_ATTRACTION", True))
    if not use_flz:
        return df

    work = df.copy()

    sigma_m = float(pget(params, "FLZ_SUPPORT_SIGMA_M", 700.0))
    weight = float(pget(params, "FLZ_SAFETY_WEIGHT", 0.30))
    min_factor = float(pget(params, "MIN_FLZ_SLOWNESS_FACTOR", 0.70))

    support = compute_flz_support(work, sigma_m=sigma_m)
    emergency_risk = 1.0 - support

    factor = 1.0 - weight * support
    factor = np.maximum(factor, min_factor)

    original_slowness = work["slowness"].to_numpy(float)
    modified_slowness = original_slowness.copy()

    flyable = original_slowness < nofly_threshold
    modified_slowness[flyable] = original_slowness[flyable] * factor[flyable]
    modified_slowness[~flyable] = original_slowness[~flyable]

    work["flz_support"] = support
    work["emergency_risk"] = emergency_risk
    work["slowness_raw"] = original_slowness
    work["slowness"] = modified_slowness

    return work


# ======================================================================
# Pair-aware swarm traffic avoidance
# ======================================================================

def compute_other_terminal_avoidance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    prefixes: list[str],
    sigma_m: float,
) -> np.ndarray:
    """
    Return a 0..1 traffic-attraction/avoidance field around DB/DK terminals
    that are NOT part of the current route pair.

    1.0 = very close to another DB/DK terminal
    0.0 = far from other DB/DK terminals
    """
    if "label_prefix" not in df.columns:
        return np.zeros(len(df), dtype=float)

    prefixes = [str(v) for v in prefixes]
    terminal_mask = df["label_prefix"].astype(str).isin(prefixes)
    terminal_mask &= df["label"].astype(str) != "NONE"

    # Exclude current pair terminals. For DB01->DK02, DB01 and DK02 are allowed;
    # all other DB/DK nodes become traffic-density areas to avoid.
    terminal_mask.iloc[int(start_idx)] = False
    terminal_mask.iloc[int(end_idx)] = False

    terminals = df.loc[terminal_mask]
    if len(terminals) == 0:
        return np.zeros(len(df), dtype=float)

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    sigma_m = max(float(sigma_m), 1.0)

    avoidance = np.zeros(len(df), dtype=float)
    for _, row in terminals.iterrows():
        dx = x - float(row["x"])
        dy = y - float(row["y"])
        d2 = dx * dx + dy * dy
        score = np.exp(-d2 / (2.0 * sigma_m * sigma_m))
        avoidance = np.maximum(avoidance, score)

    return avoidance


def apply_pair_terminal_traffic_avoidance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    nofly_threshold: float,
) -> pd.DataFrame:
    """
    Increase flyable-node slowness near DB/DK terminals that are not part of
    the current pair. This discourages paths from passing near unrelated
    bases/docking stations, reducing future swarm-traffic density.
    """
    use_avoid = bool(pget(params, "USE_OTHER_DB_DK_TRAFFIC_AVOIDANCE", True))
    if not use_avoid:
        return df

    work = df.copy()
    prefixes = list(pget(params, "OTHER_TERMINAL_AVOID_PREFIXES", ["DB", "DK"]))
    sigma_m = float(pget(params, "OTHER_TERMINAL_AVOID_SIGMA_M", 450.0))
    weight = float(pget(params, "OTHER_TERMINAL_AVOID_WEIGHT", 0.80))
    max_factor = float(pget(params, "MAX_OTHER_TERMINAL_SLOWNESS_FACTOR", 2.50))

    avoidance = compute_other_terminal_avoidance(
        work,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        prefixes=prefixes,
        sigma_m=sigma_m,
    )

    original_slowness = work["slowness"].to_numpy(float)
    modified_slowness = original_slowness.copy()
    flyable = original_slowness < nofly_threshold

    traffic_factor = 1.0 + weight * avoidance
    traffic_factor = np.minimum(traffic_factor, max_factor)

    modified_slowness[flyable] = original_slowness[flyable] * traffic_factor[flyable]
    modified_slowness[~flyable] = original_slowness[~flyable]

    work["other_terminal_avoidance"] = avoidance
    work["traffic_penalty_factor"] = traffic_factor
    work["slowness"] = modified_slowness

    return work


def path_mean_column(df: pd.DataFrame, path_indices: list[int], column: str) -> float:
    if not path_indices or column not in df.columns:
        return np.nan
    try:
        vals = df.loc[[int(v) for v in path_indices], column].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan
    except Exception:
        return np.nan


# ======================================================================
# Objective and pair utilities
# ======================================================================

def clean_route_prefixes(params: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    v1 default:
      - DB/DK are route endpoints.
      - FLZ is emergency support, not route endpoint, unless FLZ_AS_ROUTE_ENDPOINT=True.
    """
    raw_prefixes = list(pget(params, "ROUTE_OBJECTIVE_PREFIXES", ["DB", "DK"]))
    exclude = list(pget(params, "EXCLUDE_ROUTE_OBJECTIVE_PREFIXES", ["RA"]))

    flz_as_endpoint = bool(pget(params, "FLZ_AS_ROUTE_ENDPOINT", False))

    prefixes: list[str] = []
    for p in raw_prefixes:
        p = str(p)
        if p == "FLZ" and not flz_as_endpoint:
            if "FLZ" not in exclude:
                exclude.append("FLZ")
            continue
        if p not in prefixes:
            prefixes.append(p)

    # If user supplied only FLZ accidentally, recover the intended master-route endpoints.
    if not prefixes:
        prefixes = ["DB", "DK"]

    return prefixes, exclude


def objective_table(
    df: pd.DataFrame,
    route_prefixes: list[str],
    exclude_prefixes: list[str],
) -> pd.DataFrame:
    mask = df["label_prefix"].isin(route_prefixes)
    mask &= ~df["label_prefix"].isin(exclude_prefixes)
    mask &= df["label"].astype(str) != "NONE"

    obj = df.loc[mask, ["node_id", "x", "y", "z", "label", "label_prefix"]].copy()
    obj = obj.reset_index().rename(columns={"index": "idx"})
    obj = obj.sort_values(["label_prefix", "label"]).reset_index(drop=True)

    return obj


def make_pairs(
    obj: pd.DataFrame,
    pair_mode: str,
    skip_same_prefix: bool,
    max_pair_distance_m: float,
) -> list[dict[str, Any]]:
    pair_mode = pair_mode.lower().strip()
    pairs: list[dict[str, Any]] = []

    for i in range(len(obj)):
        for j in range(len(obj)):
            if i == j:
                continue
            if pair_mode == "unordered" and j <= i:
                continue

            a = obj.iloc[i]
            b = obj.iloc[j]

            if skip_same_prefix and a["label_prefix"] == b["label_prefix"]:
                continue

            dist = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

            if max_pair_distance_m > 0 and dist > max_pair_distance_m:
                continue

            pairs.append({
                "a_idx": int(a["idx"]),
                "b_idx": int(b["idx"]),
                "a_label": str(a["label"]),
                "b_label": str(b["label"]),
                "a_prefix": str(a["label_prefix"]),
                "b_prefix": str(b["label_prefix"]),
                "straight_distance_m": float(dist),
            })

    return pairs


def safe_name(text: str) -> str:
    text = str(text)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]"]:
        text = text.replace(ch, "_")
    return text


# ======================================================================
# Geometry, masks, and validation
# ======================================================================

def path_distance_m(df: pd.DataFrame, path_indices: list[int]) -> float:
    if not path_indices or len(path_indices) < 2:
        return 0.0

    xy = df.loc[path_indices, ["x", "y"]].to_numpy(float)
    d = np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2)
    return float(np.sum(d))


def path_to_xy(df: pd.DataFrame, path_indices: list[int]) -> np.ndarray:
    if not path_indices:
        return np.empty((0, 2), dtype=float)
    return df.loc[path_indices, ["x", "y"]].to_numpy(float)


def distance_to_path(df: pd.DataFrame, path_indices: list[int]) -> np.ndarray:
    pts = df[["x", "y"]].to_numpy(float)
    path_xy = path_to_xy(df, path_indices)

    if len(path_xy) == 0:
        return np.full(len(df), np.inf)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(path_xy)
        dist, _ = tree.query(pts, k=1)
        return dist.astype(float)
    except Exception:
        dist = np.full(len(df), np.inf)
        for px, py in path_xy:
            d = np.sqrt((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
            dist = np.minimum(dist, d)
        return dist


def bbox_mask(df: pd.DataFrame, start_idx: int, end_idx: int, buffer_m: float) -> np.ndarray:
    if buffer_m <= 0:
        return np.ones(len(df), dtype=bool)

    sx = float(df.at[start_idx, "x"])
    sy = float(df.at[start_idx, "y"])
    ex = float(df.at[end_idx, "x"])
    ey = float(df.at[end_idx, "y"])

    xmin = min(sx, ex) - buffer_m
    xmax = max(sx, ex) + buffer_m
    ymin = min(sy, ey) - buffer_m
    ymax = max(sy, ey) + buffer_m

    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def endpoint_buffer_mask(df: pd.DataFrame, start_idx: int, end_idx: int, radius_m: float) -> np.ndarray:
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    sx = float(df.at[start_idx, "x"])
    sy = float(df.at[start_idx, "y"])
    ex = float(df.at[end_idx, "x"])
    ey = float(df.at[end_idx, "y"])

    ds = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    de = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)

    return (ds <= radius_m) | (de <= radius_m)


def bresenham_cells(ix0: int, iy0: int, ix1: int, iy1: int) -> list[tuple[int, int]]:
    ix0 = int(ix0); iy0 = int(iy0); ix1 = int(ix1); iy1 = int(iy1)

    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)

    sx = 1 if ix0 < ix1 else (-1 if ix0 > ix1 else 0)
    sy = 1 if iy0 < iy1 else (-1 if iy0 > iy1 else 0)

    x = ix0
    y = iy0
    cells = [(x, y)]

    if dx >= dy:
        err = dx / 2.0
        while x != ix1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            cells.append((x, y))
    else:
        err = dy / 2.0
        while y != iy1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            cells.append((x, y))

    return cells


def expand_and_validate_path_by_grid_los(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    path_indices: list[int],
    allowed_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """
    Expand every segment by Bresenham cells and reject any obstacle crossing.
    """
    if not path_indices:
        return [], []

    if len(path_indices) == 1:
        return [int(path_indices[0])], []

    expanded: list[int] = []
    bad_segments: list[dict[str, Any]] = []

    for a, b in zip(path_indices[:-1], path_indices[1:]):
        a = int(a)
        b = int(b)

        ix0 = int(df.at[a, "_ix"])
        iy0 = int(df.at[a, "_iy"])
        ix1 = int(df.at[b, "_ix"])
        iy1 = int(df.at[b, "_iy"])

        cells = bresenham_cells(ix0, iy0, ix1, iy1)

        segment_indices: list[int] = []
        blocked_hits: list[tuple[int, int]] = []

        for cell in cells:
            idx = cell_to_idx.get(cell, None)

            if idx is None:
                blocked_hits.append(cell)
                continue

            idx = int(idx)

            # Allow exact terminal cells because objective cells may be forced flyable.
            if idx not in (int(start_idx), int(end_idx)) and not bool(allowed_mask[idx]):
                blocked_hits.append(cell)
                continue

            segment_indices.append(idx)

        if blocked_hits:
            bad_segments.append({
                "from_idx": a,
                "to_idx": b,
                "from_node_id": int(df.at[a, "node_id"]) if "node_id" in df.columns else a,
                "to_node_id": int(df.at[b, "node_id"]) if "node_id" in df.columns else b,
                "n_blocked_cells": len(blocked_hits),
                "blocked_cells_preview": blocked_hits[:10],
            })

        if not segment_indices:
            continue

        if expanded and segment_indices[0] == expanded[-1]:
            expanded.extend(segment_indices[1:])
        else:
            expanded.extend(segment_indices)

    cleaned: list[int] = []
    for idx in expanded:
        idx = int(idx)
        if not cleaned or cleaned[-1] != idx:
            cleaned.append(idx)

    return cleaned, bad_segments


def closeness_to_reference_m(df: pd.DataFrame, path: list[int], ref_path: list[int]) -> float:
    if not path or not ref_path:
        return np.nan

    path_xy = path_to_xy(df, path)
    ref_xy = path_to_xy(df, ref_path)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_xy)
        dist, _ = tree.query(path_xy, k=1)
        return float(np.mean(dist))
    except Exception:
        out = []
        for x, y in path_xy:
            d = np.sqrt((ref_xy[:, 0] - x) ** 2 + (ref_xy[:, 1] - y) ** 2)
            out.append(float(np.min(d)))
        return float(np.mean(out))


# ======================================================================
# Theta* wrapper
# ======================================================================

def theta_kwargs_from_params(params: dict[str, Any], safe_fallback: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in params.items():
        if key.startswith("THETASTAR_"):
            out[key] = value

    if "THETASTAR_HEURISTIC_WEIGHT" in params:
        out["heuristic_weight"] = params["THETASTAR_HEURISTIC_WEIGHT"]

    if "THETASTAR_MAX_EXPANSIONS" in params:
        out["max_expansions"] = params["THETASTAR_MAX_EXPANSIONS"]

    if safe_fallback:
        # Pure A* delegate through the Theta* module. This avoids any-angle shortcuts.
        out["THETASTAR_ASTAR_FIRST_LOS_SMOOTH"] = False
        out["THETASTAR_ALLOW_ANY_ANGLE"] = False
        out["THETASTAR_PURE_ASTAR_DELEGATE"] = True
        out["THETASTAR_OUTPUT_SAMPLED_PATH"] = True
        out["THETASTAR_LOS_STRAIGHT_PATH_FALLBACK"] = False

    return out


def result_runtime(result: dict[str, Any]) -> float:
    for key in ["runtime_seconds", "runtime_sec", "runtime"]:
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except Exception:
                pass
    return np.nan


def result_expanded(result: dict[str, Any]) -> int:
    for key in ["expanded_nodes", "expanded_states", "visited_nodes"]:
        if key in result and result[key] is not None:
            try:
                return int(result[key])
            except Exception:
                pass
    return -1


def run_theta_once(
    base_model: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    safe_fallback: bool,
) -> dict[str, Any]:
    nofly_value = float(pget(params, "NOFLY_SLOWNESS_VALUE", 10.0))

    work_model = apply_allowed_mask_to_model(
        model=base_model,
        allowed_mask=mask,
        nofly_value=nofly_value,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Pair-aware traffic criterion:
    # keep the current pair terminals usable, but penalize corridors close to
    # all other DB/DK terminals so the final swarm network does not overload
    # unrelated bases/docking stations.
    work_model = apply_pair_terminal_traffic_avoidance(
        df=work_model,
        start_idx=start_idx,
        end_idx=end_idx,
        params=params,
        nofly_threshold=float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0)),
    )

    graph = build_graph_for_model(work_model, params)

    result = theta_run(
        model=work_model,
        graph=graph,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        **theta_kwargs_from_params(params, safe_fallback=safe_fallback),
    )

    if not isinstance(result, dict):
        result = {
            "success": False,
            "message": "src.thetastar.run() did not return a dict.",
            "path_indices": [],
        }

    raw_path = [int(v) for v in result.get("path_indices", [])]

    invalid_nodes = [idx for idx in raw_path if idx < 0 or idx >= len(mask) or not bool(mask[idx])]
    if invalid_nodes:
        result["success"] = False
        result["message"] = (
            str(result.get("message", "")) +
            f" Path contains {len(invalid_nodes)} blocked path nodes."
        )
        result["path_indices"] = []
        raw_path = []
    else:
        expanded_path, bad_segments = expand_and_validate_path_by_grid_los(
            df=base_model,
            cell_to_idx=cell_to_idx,
            path_indices=raw_path,
            allowed_mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        if bad_segments:
            result["success"] = False
            result["message"] = (
                str(result.get("message", "")) +
                f" Theta* segment crossed blocked cells. "
                f"Bad segments={len(bad_segments)}; first={bad_segments[0]}"
            )
            result["path_indices"] = []
            raw_path = []
        else:
            raw_path = expanded_path
            result["path_indices"] = raw_path
            result["grid_los_validated"] = True
            result["grid_los_expanded_nodes"] = len(expanded_path)

    success = bool(result.get("success", False)) and len(raw_path) > 0
    result["success"] = success
    result["route_name"] = route_name
    result["path_indices"] = raw_path
    result["distance_m"] = path_distance_m(base_model, raw_path) if raw_path else np.nan
    result["mean_flz_support"] = path_mean_column(work_model, raw_path, "flz_support")
    result["mean_emergency_risk"] = path_mean_column(work_model, raw_path, "emergency_risk")
    result["mean_other_terminal_avoidance"] = path_mean_column(work_model, raw_path, "other_terminal_avoidance")
    result["mean_traffic_penalty_factor"] = path_mean_column(work_model, raw_path, "traffic_penalty_factor")
    result["runtime_sec"] = result_runtime(result)
    result["expanded_nodes"] = result_expanded(result)
    result["safe_fallback"] = bool(safe_fallback)

    return result


def run_project_theta(
    base_model: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    extra_allowed_mask: np.ndarray | None = None,
    block_mask: np.ndarray | None = None,
    route_name: str = "",
) -> dict[str, Any]:
    bbox_buffer = float(pget(params, "SEARCH_BBOX_BUFFER_M", 0.0))
    retry_full = bool(pget(params, "RETRY_FULL_MAP_IF_FAILED", True))

    allowed = base_allowed_mask.copy()

    if extra_allowed_mask is not None:
        allowed &= extra_allowed_mask

    if block_mask is not None:
        allowed &= ~block_mask

    allowed[int(start_idx)] = True
    allowed[int(end_idx)] = True

    masks_to_try: list[tuple[np.ndarray, bool]] = []
    if bbox_buffer > 0:
        mask_bbox = allowed & bbox_mask(base_model, start_idx, end_idx, bbox_buffer)
        mask_bbox[int(start_idx)] = True
        mask_bbox[int(end_idx)] = True
        masks_to_try.append((mask_bbox, False))
        if retry_full:
            masks_to_try.append((allowed, True))
    else:
        masks_to_try.append((allowed, False))

    last_result: dict[str, Any] | None = None

    for mask, used_full_retry in masks_to_try:
        # First, try normal configured Theta*.
        result = run_theta_once(
            base_model=base_model,
            cell_to_idx=cell_to_idx,
            mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
            safe_fallback=False,
        )
        result["used_full_map_retry"] = bool(used_full_retry)

        if result.get("success", False):
            return result

        last_result = result

        # If any-angle route is invalid, retry pure A* delegate through Theta*.
        result_safe = run_theta_once(
            base_model=base_model,
            cell_to_idx=cell_to_idx,
            mask=mask,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
            safe_fallback=True,
        )
        result_safe["used_full_map_retry"] = bool(used_full_retry)
        result_safe["message"] = str(result_safe.get("message", "")) + " [safe fallback attempted]"

        if result_safe.get("success", False):
            return result_safe

        last_result = result_safe

    if last_result is None:
        return {
            "success": False,
            "message": "No search attempted.",
            "path_indices": [],
            "route_name": route_name,
        }

    return last_result


# ======================================================================
# Route generation
# ======================================================================

def sorted_corridor_list(params: dict[str, Any]) -> list[float]:
    vals = [float(v) for v in list(pget(params, "BACKUP_CORRIDOR_M_LIST", [150.0, 250.0, 400.0, 700.0]))]
    vals = sorted(set(v for v in vals if v >= 0.0))
    return vals or [150.0, 250.0, 400.0, 700.0]


def sorted_avoid_list(params: dict[str, Any]) -> list[float]:
    vals = [float(v) for v in list(pget(params, "BACKWARD_AVOID_FORWARD_RADIUS_M_LIST", [150.0, 100.0, 50.0, 0.0]))]
    vals = sorted(set(v for v in vals if v >= 0.0), reverse=True)
    if 0.0 not in vals:
        vals.append(0.0)
    return vals


def route_sets_per_direction(params: dict[str, Any]) -> int:
    """Number of main/backup route sets to generate per direction.

    Example:
        ROUTE_SETS_PER_DIRECTION = 2

    gives:
        forward_main_01, forward_backup_01
        forward_main_02, forward_backup_02
        backward_main_01, backward_backup_01
        backward_main_02, backward_backup_02
    """
    value = pget(
        params,
        "ROUTE_SETS_PER_DIRECTION",
        pget(params, "N_ROUTES_EACH_DIRECTION", pget(params, "ROUTE_COUNT_EACH_DIRECTION", 1)),
    )
    try:
        return max(1, int(float(value)))
    except Exception:
        return 1


def sorted_main_avoid_list(params: dict[str, Any]) -> list[float]:
    """Avoid-radius list for secondary main routes.

    main_01 is the fastest route.
    main_02, main_03, ... are generated by blocking previous main routes
    with these radii.  Larger values force stronger separation and usually
    make the secondary route longer.
    """
    default = pget(params, "BACKWARD_AVOID_FORWARD_RADIUS_M_LIST", [200.0, 150.0, 100.0, 50.0, 0.0])
    vals = [float(v) for v in list(pget(params, "MAIN_ROUTE_AVOID_PREVIOUS_RADIUS_M_LIST", default))]
    vals = sorted(set(v for v in vals if v >= 0.0), reverse=True)
    if 0.0 not in vals:
        vals.append(0.0)
    return vals


def combined_distance_to_paths(df: pd.DataFrame, paths: list[list[int]]) -> np.ndarray:
    """Minimum distance from each node to any node on any reference path."""
    valid_paths = [p for p in paths if p]
    if not valid_paths:
        return np.full(len(df), np.inf, dtype=float)

    dist = np.full(len(df), np.inf, dtype=float)
    for path in valid_paths:
        dist = np.minimum(dist, distance_to_path(df, path))
    return dist


def unique_path_indices(path: list[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for idx in path:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def lock_settings(params: dict[str, Any]) -> tuple[bool, float, float, bool]:
    enabled = bool(pget(params, "ENABLE_ROUTE_NODE_LOCK", True))
    max_overlap_pct = float(pget(params, "MAX_ROUTE_NODE_OVERLAP_PERCENT", 10.0))
    lock_radius_m = float(pget(params, "LOCK_NODE_RADIUS_M", 0.0))
    strict_first = bool(pget(params, "STRICT_LOCK_BLOCK_FIRST", True))
    return enabled, max_overlap_pct, lock_radius_m, strict_first


def build_locked_node_mask(
    df: pd.DataFrame,
    locked_paths: list[list[int]],
    endpoint_mask: np.ndarray,
    lock_radius_m: float,
) -> np.ndarray:
    if not locked_paths:
        return np.zeros(len(df), dtype=bool)
    dist_to_locked = combined_distance_to_paths(df, locked_paths)
    if float(lock_radius_m) <= 0.0:
        mask = np.isfinite(dist_to_locked) & (dist_to_locked <= 1.0e-9)
    else:
        mask = dist_to_locked <= float(lock_radius_m)
    return mask & (~endpoint_mask)


def route_overlap_stats(
    path_indices: list[int],
    locked_mask: np.ndarray,
    endpoint_mask: np.ndarray,
) -> tuple[int, int, float]:
    if not path_indices:
        return 0, 0, 0.0
    idxs = np.array(unique_path_indices(path_indices), dtype=int)
    if idxs.size == 0:
        return 0, 0, 0.0
    valid = ~endpoint_mask[idxs]
    idxs = idxs[valid]
    total = int(idxs.size)
    if total <= 0:
        return 0, 0, 0.0
    overlap = int(np.count_nonzero(locked_mask[idxs]))
    pct = 100.0 * overlap / float(total)
    return overlap, total, pct


def route_respects_lock_limit(
    result: dict[str, Any],
    locked_mask: np.ndarray,
    endpoint_mask: np.ndarray,
    max_overlap_pct: float,
) -> tuple[bool, dict[str, Any]]:
    path = [int(v) for v in result.get("path_indices", [])]
    overlap_nodes, checked_nodes, overlap_pct = route_overlap_stats(path, locked_mask, endpoint_mask)
    result["locked_overlap_nodes"] = int(overlap_nodes)
    result["locked_overlap_checked_nodes"] = int(checked_nodes)
    result["locked_overlap_percent"] = float(overlap_pct)
    ok = overlap_pct <= float(max_overlap_pct) + 1.0e-9
    if not ok:
        result["message"] = (
            str(result.get("message", ""))
            + f" Overlap with locked routes = {overlap_nodes}/{checked_nodes} nodes ({overlap_pct:.2f}%),"
            + f" exceeds limit {float(max_overlap_pct):.2f}%."
        ).strip()
    return ok, result


def duplicate_route_result(
    df: pd.DataFrame,
    source_result: dict[str, Any],
    route_name: str,
    message: str,
) -> dict[str, Any]:
    path = [int(v) for v in source_result.get("path_indices", [])]
    return {
        "success": bool(path),
        "message": message,
        "route_name": route_name,
        "path_indices": path,
        "distance_m": path_distance_m(df, path) if path else np.nan,
        "total_cost": np.nan,
        "runtime_sec": 0.0,
        "expanded_nodes": 0,
        "duplicated_from_main": True,
        "duplicated_from_previous_main": True,
        "safe_fallback": False,
    }


def make_main_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    route_name: str,
    previous_main_paths: list[list[int]] | None = None,
    opposite_direction_paths: list[list[int]] | None = None,
    locked_paths: list[list[int]] | None = None,
    rank: int = 1,
) -> dict[str, Any]:
    """Generate one main route.

    rank=1: fastest route with only the base constraints.
    rank>1: alternative main route. It avoids previous main routes and,
            when provided, opposite-direction routes. The alternative does
            not need to be fastest; with ALTERNATIVE_MAIN_SELECTION_MODE =
            "longest", the script keeps the longest successful candidate.
    """
    previous_main_paths = previous_main_paths or []
    opposite_direction_paths = opposite_direction_paths or []
    locked_paths = locked_paths or []
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, float(pget(params, "ENDPOINT_BUFFER_M", 150.0)))
    lock_enabled, max_overlap_pct, lock_radius_m, strict_lock_first = lock_settings(params)
    locked_mask = build_locked_node_mask(df, locked_paths, endpoint_mask, lock_radius_m)

    # First main route is the fastest normal route.
    if int(rank) <= 1 and not opposite_direction_paths and not locked_paths:
        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
        )
        result["main_rank"] = int(rank)
        result["alternative_main"] = False
        return result

    avoid_paths = list(previous_main_paths) + list(opposite_direction_paths)
    if not avoid_paths and not locked_paths:
        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=route_name,
        )
        result["main_rank"] = int(rank)
        result["alternative_main"] = False
        return result

    avoid_list = sorted_main_avoid_list(params)
    selection_mode = str(pget(params, "ALTERNATIVE_MAIN_SELECTION_MODE", "longest")).strip().lower()
    allow_duplicate = bool(pget(params, "ALLOW_MAIN_DUPLICATE_IF_FAILED", False))

    dist_to_avoid = combined_distance_to_paths(df, avoid_paths)

    candidates: list[dict[str, Any]] = []
    best_failed: dict[str, Any] | None = None

    for avoid_m in avoid_list:
        base_block_mask = None if float(avoid_m) <= 0.0 else ((dist_to_avoid <= float(avoid_m)) & (~endpoint_mask))

        candidate_attempts = []
        if lock_enabled and np.any(locked_mask):
            if strict_lock_first:
                candidate_attempts.append(("strict_lock", locked_mask if base_block_mask is None else (base_block_mask | locked_mask)))
            candidate_attempts.append(("soft_lock", base_block_mask))
        else:
            candidate_attempts.append(("no_lock", base_block_mask))

        for lock_mode, block_mask in candidate_attempts:
            result = run_project_theta(
                base_model=df,
                cell_to_idx=cell_to_idx,
                base_allowed_mask=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                params=params,
                block_mask=block_mask,
                route_name=route_name,
            )
            result["main_rank"] = int(rank)
            result["alternative_main"] = True
            result["main_avoid_previous_radius_m"] = float(avoid_m)
            result["lock_mode"] = lock_mode

            if result.get("success", False):
                ok_overlap, result = route_respects_lock_limit(result, locked_mask, endpoint_mask, max_overlap_pct)
                if ok_overlap:
                    candidates.append(result)
                    if selection_mode not in ("longest", "max_distance", "long"):
                        return result
                else:
                    result["success"] = False

            best_failed = result

    if candidates:
        if selection_mode in ("longest", "max_distance", "long"):
            chosen = max(candidates, key=lambda r: float(r.get("distance_m", -np.inf)))
        else:
            chosen = candidates[0]
        chosen["alternative_main_selection_mode"] = selection_mode
        return chosen

    if allow_duplicate and previous_main_paths:
        fake_source = {"path_indices": previous_main_paths[-1]}
        return duplicate_route_result(
            df=df,
            source_result=fake_source,
            route_name=route_name,
            message="Alternative main failed; duplicated previous main route because ALLOW_MAIN_DUPLICATE_IF_FAILED=True.",
        )

    if best_failed is None:
        best_failed = {"success": False, "message": "Alternative main not attempted.", "path_indices": []}

    best_failed["main_rank"] = int(rank)
    best_failed["alternative_main"] = True
    return best_failed


def generate_direction_route_sets(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    params: dict[str, Any],
    pair_name: str,
    direction: str,
    n_sets: int,
    opposite_direction_paths: list[list[int]] | None = None,
    prelocked_paths: list[list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate K main + K backup routes for one direction."""
    routes: dict[str, dict[str, Any]] = {}
    previous_main_paths: list[list[int]] = []
    opposite_direction_paths = opposite_direction_paths or []
    locked_paths: list[list[int]] = list(prelocked_paths or [])

    for rank in range(1, int(n_sets) + 1):
        main_key = f"{direction}_main_{rank:02d}"
        backup_key = f"{direction}_backup_{rank:02d}"

        main_result = make_main_route(
            df=df,
            cell_to_idx=cell_to_idx,
            base_allowed=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            route_name=f"{pair_name}_{main_key}",
            previous_main_paths=previous_main_paths,
            opposite_direction_paths=opposite_direction_paths,
            locked_paths=locked_paths,
            rank=rank,
        )
        main_result["direction"] = direction
        main_result["route_type"] = "main"
        main_result["route_rank"] = int(rank)
        routes[main_key] = main_result

        if main_result.get("success", False):
            main_path = [int(v) for v in main_result.get("path_indices", [])]
            previous_main_paths.append(main_path)
            locked_paths.append(main_path)

            backup_result = make_backup_route(
                df=df,
                cell_to_idx=cell_to_idx,
                base_allowed=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                main_path=main_path,
                params=params,
                route_name=f"{pair_name}_{backup_key}",
                locked_paths=locked_paths,
            )
        else:
            backup_result = {
                "success": False,
                "message": f"Skipped because {main_key} failed.",
                "path_indices": [],
                "route_name": f"{pair_name}_{backup_key}",
            }

        backup_result["direction"] = direction
        backup_result["route_type"] = "backup"
        backup_result["route_rank"] = int(rank)
        routes[backup_key] = backup_result
        if backup_result.get("success", False):
            locked_paths.append([int(v) for v in backup_result.get("path_indices", [])])

    return routes


def make_backup_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    main_path: list[int],
    params: dict[str, Any],
    route_name: str,
    locked_paths: list[list[int]] | None = None,
) -> dict[str, Any]:
    corridor_list = sorted_corridor_list(params)
    block_radius = float(pget(params, "BACKUP_BLOCK_MAIN_RADIUS_M", 30.0))
    endpoint_buffer_m = float(pget(params, "ENDPOINT_BUFFER_M", 150.0))
    allow_duplicate = bool(pget(params, "ALLOW_BACKUP_DUPLICATE_IF_FAILED", True))
    locked_paths = locked_paths or []
    lock_enabled, max_overlap_pct, lock_radius_m, strict_lock_first = lock_settings(params)

    dist_to_main = distance_to_path(df, main_path)
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, endpoint_buffer_m)
    block_mask = (dist_to_main <= block_radius) & (~endpoint_mask)
    locked_mask = build_locked_node_mask(df, locked_paths, endpoint_mask, lock_radius_m)

    best_failed: dict[str, Any] | None = None

    for corridor_m in corridor_list:
        corridor_mask = dist_to_main <= float(corridor_m)

        candidate_attempts = []
        if lock_enabled and np.any(locked_mask):
            if strict_lock_first:
                candidate_attempts.append(("strict_lock", block_mask | locked_mask))
            candidate_attempts.append(("soft_lock", block_mask))
        else:
            candidate_attempts.append(("no_lock", block_mask))

        for lock_mode, this_block_mask in candidate_attempts:
            result = run_project_theta(
                base_model=df,
                cell_to_idx=cell_to_idx,
                base_allowed_mask=base_allowed,
                start_idx=start_idx,
                end_idx=end_idx,
                params=params,
                extra_allowed_mask=corridor_mask,
                block_mask=this_block_mask,
                route_name=route_name,
            )

            result["backup_corridor_m"] = float(corridor_m)
            result["backup_block_main_radius_m"] = block_radius
            result["lock_mode"] = lock_mode

            if result.get("success", False):
                ok_overlap, result = route_respects_lock_limit(result, locked_mask, endpoint_mask, max_overlap_pct)
                if ok_overlap:
                    result["duplicated_from_main"] = False
                    return result
                result["success"] = False

            best_failed = result

    if allow_duplicate:
        return {
            "success": True,
            "message": "Backup failed; duplicated main route because ALLOW_BACKUP_DUPLICATE_IF_FAILED=True.",
            "route_name": route_name,
            "path_indices": list(main_path),
            "distance_m": path_distance_m(df, main_path),
            "total_cost": np.nan,
            "runtime_sec": 0.0,
            "expanded_nodes": 0,
            "duplicated_from_main": True,
            "safe_fallback": False,
        }

    if best_failed is None:
        best_failed = {"success": False, "message": "Backup not attempted.", "path_indices": []}

    best_failed["duplicated_from_main"] = False
    return best_failed


def make_backward_main_route(
    df: pd.DataFrame,
    cell_to_idx: dict[tuple[int, int], int],
    base_allowed: np.ndarray,
    start_idx: int,
    end_idx: int,
    forward_main_path: list[int],
    params: dict[str, Any],
    route_name: str,
) -> dict[str, Any]:
    avoid_list = sorted_avoid_list(params)
    endpoint_buffer_m = float(pget(params, "ENDPOINT_BUFFER_M", 150.0))

    dist_to_forward = distance_to_path(df, forward_main_path)
    endpoint_mask = endpoint_buffer_mask(df, start_idx, end_idx, endpoint_buffer_m)

    best_failed: dict[str, Any] | None = None

    for avoid_m in avoid_list:
        if float(avoid_m) <= 0:
            block_mask = None
        else:
            block_mask = (dist_to_forward <= float(avoid_m)) & (~endpoint_mask)

        result = run_project_theta(
            base_model=df,
            cell_to_idx=cell_to_idx,
            base_allowed_mask=base_allowed,
            start_idx=start_idx,
            end_idx=end_idx,
            params=params,
            block_mask=block_mask,
            route_name=route_name,
        )

        result["backward_avoid_forward_radius_m"] = float(avoid_m)

        if result.get("success", False):
            return result

        best_failed = result

    if best_failed is None:
        best_failed = {"success": False, "message": "Backward route not attempted.", "path_indices": []}

    return best_failed


# ======================================================================
# Output and plotting
# ======================================================================

def save_route_nodes(
    df: pd.DataFrame,
    path_indices: list[int],
    output_file: Path,
    pair_name: str,
    direction: str,
    route_type: str,
) -> None:
    if not path_indices:
        return

    cols = [
        "node_id", "x", "y", "z", "slowness", "slowness_raw",
        "flz_support", "emergency_risk",
        "risk_obstacle", "risk_ra", "risk_total",
        "obstacle_flag", "ra_flag", "objective_flag", "label", "label_prefix",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df.loc[path_indices, cols].copy()
    out.insert(0, "seq", np.arange(len(out), dtype=int))
    out.insert(1, "pair", pair_name)
    out.insert(2, "direction", direction)
    out.insert(3, "route_type", route_type)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)


def route_edges_dataframe(
    df: pd.DataFrame,
    path_indices: list[int],
    pair_name: str,
    direction: str,
    route_type: str,
) -> pd.DataFrame:
    rows = []

    for k in range(len(path_indices) - 1):
        i = int(path_indices[k])
        j = int(path_indices[k + 1])

        x1 = float(df.at[i, "x"])
        y1 = float(df.at[i, "y"])
        x2 = float(df.at[j, "x"])
        y2 = float(df.at[j, "y"])

        rows.append({
            "pair": pair_name,
            "direction": direction,
            "route_type": route_type,
            "edge_seq": k,
            "from_node_id": int(df.at[i, "node_id"]),
            "to_node_id": int(df.at[j, "node_id"]),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "edge_distance_m": math.hypot(x2 - x1, y2 - y1),
        })

    return pd.DataFrame(rows)


def plot_pair_routes(
    df: pd.DataFrame,
    routes: dict[str, dict[str, Any]],
    pair_name: str,
    output_file: Path,
    nofly_threshold: float,
    plot_node_size: float,
    line_width: float,
    params: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    font_family = str(pget(params, "PLOT_FONT_FAMILY", "DejaVu Serif"))
    title_size = float(pget(params, "PLOT_TITLE_FONT_SIZE", 16))
    label_size = float(pget(params, "PLOT_LABEL_FONT_SIZE", 13))
    legend_size = float(pget(params, "PLOT_LEGEND_FONT_SIZE", 10))
    text_size = float(pget(params, "PLOT_TEXT_FONT_SIZE", 8))

    nofly = df["slowness"].to_numpy(float) >= nofly_threshold

    ax.scatter(df.loc[~nofly, "x"], df.loc[~nofly, "y"], s=plot_node_size, c="lightgray", linewidths=0, label="flyable")
    ax.scatter(df.loc[nofly, "x"], df.loc[nofly, "y"], s=plot_node_size, c="black", linewidths=0, label="no-fly")

    marker_info = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 140),
        "RA": ("X", "purple", 100),
    }

    for prefix, info in marker_info.items():
        marker, color, size = info
        sub = df[df["label_prefix"] == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, zorder=10, label=prefix)
        for _, row in sub.iterrows():
            ax.text(row["x"], row["y"], str(row["label"]), fontsize=text_size, color=color, weight="bold", zorder=11, fontfamily=font_family)

    seen_route_labels: set[str] = set()
    for key, result in routes.items():
        path = result.get("path_indices", [])
        if not path:
            continue

        direction = str(result.get("direction", "forward" if key.startswith("forward") else "backward"))
        route_type = str(result.get("route_type", "backup" if "backup" in key else "main"))
        route_rank = int(result.get("route_rank", 1))

        xy = path_to_xy(df, path)
        color = "red" if direction == "forward" else "blue"
        linestyle = "-" if route_type == "main" else "--"
        alpha = max(0.30, 0.95 - 0.12 * (route_rank - 1))
        lw = line_width if route_rank == 1 else max(1.0, line_width * 0.75)
        label = f"{direction} {route_type}"
        if label in seen_route_labels:
            label = "_nolegend_"
        else:
            seen_route_labels.add(label)

        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
            label=label,
            zorder=20,
        )

    ax.set_title(pair_name, fontfamily=font_family, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("X coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_ylabel("Y coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(fontsize=legend_size, loc="upper right", frameon=True)
    for text_obj in legend.get_texts():
        text_obj.set_fontfamily(font_family)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=250)
    plt.close(fig)


def plot_overview(
    df: pd.DataFrame,
    route_records: list[dict[str, Any]],
    output_file: Path,
    nofly_threshold: float,
    plot_node_size: float,
    params: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    font_family = str(pget(params, "PLOT_FONT_FAMILY", "DejaVu Serif"))
    title_size = float(pget(params, "PLOT_TITLE_FONT_SIZE", 16))
    label_size = float(pget(params, "PLOT_LABEL_FONT_SIZE", 13))
    legend_size = float(pget(params, "PLOT_LEGEND_FONT_SIZE", 10))
    text_size = float(pget(params, "PLOT_TEXT_FONT_SIZE", 8))

    nofly = df["slowness"].to_numpy(float) >= nofly_threshold

    ax.scatter(df.loc[~nofly, "x"], df.loc[~nofly, "y"], s=plot_node_size, c="lightgray", linewidths=0, label="flyable")
    ax.scatter(df.loc[nofly, "x"], df.loc[nofly, "y"], s=plot_node_size, c="black", linewidths=0, label="no-fly")

    for rec in route_records:
        path = rec.get("path_indices", [])
        if not path:
            continue
        xy = path_to_xy(df, path)
        direction = rec.get("direction", "")
        route_type = rec.get("route_type", "")
        color = "red" if direction == "forward" else "blue"
        linestyle = "-" if route_type == "main" else "--"
        alpha = 0.75 if route_type == "main" else 0.45
        ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, linewidth=1.2, alpha=alpha)

    marker_info = {
        "DB": ("^", "blue", 90),
        "DK": ("s", "green", 80),
        "FLZ": ("*", "orange", 140),
        "RA": ("X", "purple", 100),
    }

    for prefix, info in marker_info.items():
        marker, color, size = info
        sub = df[df["label_prefix"] == prefix]
        if len(sub) == 0:
            continue
        ax.scatter(sub["x"], sub["y"], marker=marker, s=size, c=color, edgecolors="white", linewidths=0.8, zorder=10, label=prefix)
        for _, row in sub.iterrows():
            ax.text(row["x"], row["y"], str(row["label"]), fontsize=text_size, color=color, weight="bold", zorder=11, fontfamily=font_family)

    ax.set_title("All Theta* master routes v4", fontfamily=font_family, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("X coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_ylabel("Y coordinate (m)", fontfamily=font_family, fontsize=label_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(fontsize=legend_size, loc="upper right", frameon=True)
    for text_obj in legend.get_texts():
        text_obj.set_fontfamily(font_family)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=250)
    plt.close(fig)


# ======================================================================
# Main workflow
# ======================================================================

def main() -> None:
    args = parse_args()
    params = load_params(args.param_file)

    model_file = Path(str(pget(params, "MODEL_FILE", "")))
    output_dir = Path(str(pget(params, "OUTPUT_DIR", "output/thetastar_master_plan")))

    route_dir = output_dir / "route_nodes"
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"THETA* MASTER OBJECTIVE-PAIR PLANNER {VERSION}")
    print("=" * 80)
    print(f"Param file      : {args.param_file}")
    print(f"Model file      : {model_file}")
    print(f"Output directory: {output_dir}")

    nofly_threshold = float(pget(params, "NOFLY_SLOWNESS_THRESHOLD", 10.0))

    df = read_node_model(model_file)
    df, cell_to_idx, grid_m = add_grid_index(df)

    # FLZ support modifies slowness only on flyable cells.
    df = apply_flz_safety_attraction(df, params, nofly_threshold=nofly_threshold)

    planning_model_file = output_dir / "planning_model_with_flz_support.xyz"
    save_cols = [c for c in df.columns if not c.startswith("_")]
    df[save_cols].to_csv(planning_model_file, sep=" ", index=False, float_format="%.6f")

    print(f"Nodes           : {len(df):,}")
    print(f"Inferred grid    : {grid_m:.3f} m")
    print(f"Planning model   : {planning_model_file}")

    base_allowed = df["slowness"].to_numpy(float) < nofly_threshold

    route_prefixes, exclude_prefixes = clean_route_prefixes(params)
    print(f"Route prefixes  : {route_prefixes}")
    print(f"Excluded prefix : {exclude_prefixes}")

    obj = objective_table(df=df, route_prefixes=route_prefixes, exclude_prefixes=exclude_prefixes)

    if len(obj) < 2:
        raise RuntimeError("Need at least two DB/DK route objectives.")

    if bool(pget(params, "FORCE_ROUTE_OBJECTIVES_FLYABLE", False)):
        for idx in obj["idx"].to_numpy(int):
            base_allowed[int(idx)] = True

    obj.to_csv(output_dir / "objective_table.csv", index=False)

    pair_mode = str(pget(params, "PAIR_MODE", "unordered"))
    skip_same_prefix = bool(pget(params, "SKIP_SAME_PREFIX", False))
    max_pair_distance_m = float(pget(params, "MAX_PAIR_DISTANCE_M", 0.0))

    pairs = make_pairs(
        obj=obj,
        pair_mode=pair_mode,
        skip_same_prefix=skip_same_prefix,
        max_pair_distance_m=max_pair_distance_m,
    )

    print(f"Route objectives: {len(obj)}")
    print(obj[["label", "label_prefix", "x", "y"]].to_string(index=False))
    print(f"Route pairs     : {len(pairs)}")
    print(f"Backup corridor : {sorted_corridor_list(params)}")
    print(f"Backward avoid  : {sorted_avoid_list(params)}")
    print("-" * 80)

    summary_rows: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    edge_tables: list[pd.DataFrame] = []

    n_route_sets = route_sets_per_direction(params)
    print(f"Route sets/dir  : {n_route_sets}")
    print(f"Main avoid list : {sorted_main_avoid_list(params)}")
    print(f"Node lock       : {lock_settings(params)}")
    print(f"FLZ attraction  : {bool(pget(params, 'USE_FLZ_SAFETY_ATTRACTION', True))}, sigma={float(pget(params, 'FLZ_SUPPORT_SIGMA_M', 700.0)):.1f} m, weight={float(pget(params, 'FLZ_SAFETY_WEIGHT', 0.30)):.2f}")
    print(f"DB/DK avoidance : {bool(pget(params, 'USE_OTHER_DB_DK_TRAFFIC_AVOIDANCE', True))}, sigma={float(pget(params, 'OTHER_TERMINAL_AVOID_SIGMA_M', 450.0)):.1f} m, weight={float(pget(params, 'OTHER_TERMINAL_AVOID_WEIGHT', 0.80)):.2f}")

    for pair_id, pair in enumerate(pairs, start=1):
        a_idx = int(pair["a_idx"])
        b_idx = int(pair["b_idx"])
        a_label = str(pair["a_label"])
        b_label = str(pair["b_label"])
        pair_name = f"{safe_name(a_label)}_to_{safe_name(b_label)}"

        print(f"[{pair_id}/{len(pairs)}] {a_label} ↔ {b_label}")

        # Forward: A -> B.
        # main_01 is fastest. main_02+ avoid previous forward main routes.
        forward_routes = generate_direction_route_sets(
            df=df,
            cell_to_idx=cell_to_idx,
            base_allowed=base_allowed,
            start_idx=a_idx,
            end_idx=b_idx,
            params=params,
            pair_name=pair_name,
            direction="forward",
            n_sets=n_route_sets,
            opposite_direction_paths=[],
            prelocked_paths=[],
        )

        forward_main_paths = [
            [int(v) for v in r.get("path_indices", [])]
            for k, r in forward_routes.items()
            if r.get("route_type") == "main" and r.get("success", False)
        ]

        # Backward: B -> A.
        # main_01 tries to avoid all forward main routes first. main_02+ also
        # avoid previous backward main routes.
        backward_routes = generate_direction_route_sets(
            df=df,
            cell_to_idx=cell_to_idx,
            base_allowed=base_allowed,
            start_idx=b_idx,
            end_idx=a_idx,
            params=params,
            pair_name=pair_name,
            direction="backward",
            n_sets=n_route_sets,
            opposite_direction_paths=forward_main_paths,
            prelocked_paths=forward_main_paths,
        )

        routes = {}
        routes.update(forward_routes)
        routes.update(backward_routes)

        for route_key, result in routes.items():
            direction = str(result.get("direction", "forward" if route_key.startswith("forward") else "backward"))
            route_type = str(result.get("route_type", "backup" if "backup" in route_key else "main"))
            route_rank = int(result.get("route_rank", 1))
            path = [int(v) for v in result.get("path_indices", [])]
            route_file = route_dir / f"{pair_name}_{route_key}.csv"

            if path:
                save_route_nodes(
                    df=df,
                    path_indices=path,
                    output_file=route_file,
                    pair_name=pair_name,
                    direction=direction,
                    route_type=route_type,
                )
                edge_table = route_edges_dataframe(
                    df=df,
                    path_indices=path,
                    pair_name=pair_name,
                    direction=direction,
                    route_type=route_type,
                )
                if len(edge_table):
                    edge_table["route_rank"] = route_rank
                    edge_table["route_key"] = route_key
                edge_tables.append(edge_table)

            if route_type == "backup":
                ref_key = f"{direction}_main_{route_rank:02d}"
                ref_path = routes.get(ref_key, {}).get("path_indices", [])
                mean_distance_to_main = closeness_to_reference_m(df, path, ref_path)
            else:
                mean_distance_to_main = np.nan

            total_cost = result.get("total_cost", result.get("cost", np.nan))

            row = {
                "version": VERSION,
                "pair": pair_name,
                "route_key": route_key,
                "start_label": a_label if direction == "forward" else b_label,
                "end_label": b_label if direction == "forward" else a_label,
                "direction": direction,
                "route_type": route_type,
                "route_rank": route_rank,
                "success": bool(result.get("success", False)),
                "message": result.get("message", result.get("status", "")),
                "distance_m": float(result.get("distance_m", np.nan)),
                "distance_km": float(result.get("distance_m", np.nan)) / 1000.0
                    if not pd.isna(result.get("distance_m", np.nan)) else np.nan,
                "total_cost": total_cost,
                "n_path_nodes": len(path),
                "expanded_nodes": int(result.get("expanded_nodes", -1)),
                "runtime_sec": float(result.get("runtime_sec", np.nan)),
                "mean_distance_to_main_m": mean_distance_to_main,
                "duplicated_from_main": bool(result.get("duplicated_from_main", False)),
                "duplicated_from_previous_main": bool(result.get("duplicated_from_previous_main", False)),
                "used_full_map_retry": bool(result.get("used_full_map_retry", False)),
                "safe_fallback": bool(result.get("safe_fallback", False)),
                "main_avoid_previous_radius_m": result.get("main_avoid_previous_radius_m", np.nan),
                "backup_corridor_m": result.get("backup_corridor_m", np.nan),
                "lock_mode": result.get("lock_mode", ""),
                "locked_overlap_nodes": int(result.get("locked_overlap_nodes", 0)),
                "locked_overlap_checked_nodes": int(result.get("locked_overlap_checked_nodes", 0)),
                "locked_overlap_percent": float(result.get("locked_overlap_percent", 0.0)),
                "mean_flz_support": float(result.get("mean_flz_support", np.nan)),
                "mean_emergency_risk": float(result.get("mean_emergency_risk", np.nan)),
                "mean_other_terminal_avoidance": float(result.get("mean_other_terminal_avoidance", np.nan)),
                "mean_traffic_penalty_factor": float(result.get("mean_traffic_penalty_factor", np.nan)),
                "route_file": str(route_file) if path else "",
            }

            summary_rows.append(row)

            record = dict(row)
            record["path_indices"] = path
            route_records.append(record)

            print(
                f"  {route_key:20s} | "
                f"success={str(row['success']):5s} | "
                f"dist={row['distance_m']:9.1f} m | "
                f"nodes={row['n_path_nodes']:5d} | "
                f"expanded={row['expanded_nodes']:8d} | "
                f"fallback={str(row['safe_fallback']):5s} | "
                f"time={row['runtime_sec']:7.3f} s"
            )

        if bool(pget(params, "SAVE_PAIR_FIGURES", True)):
            plot_pair_routes(
                df=df,
                routes=routes,
                pair_name=pair_name,
                output_file=figure_dir / f"{pair_name}_theta_routes_v4.png",
                nofly_threshold=nofly_threshold,
                plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
                line_width=float(pget(params, "PLOT_ROUTE_LINEWIDTH", 2.2)),
                params=params,
            )

        print("-" * 80)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / "route_summary_v4.csv"
    summary_df.to_csv(summary_file, index=False)

    if edge_tables:
        edge_df = pd.concat(edge_tables, ignore_index=True)
    else:
        edge_df = pd.DataFrame()

    edge_file = output_dir / "all_route_edges_v4.csv"
    edge_df.to_csv(edge_file, index=False)

    if bool(pget(params, "SAVE_OVERVIEW_FIGURE", True)):
        # Save one overview inside figures/ and one directly in OUTPUT_DIR.
        # The parent-directory copy is easier to find after batch processing.
        overview_figures = figure_dir / "00_all_theta_routes_overview_v4.png"
        overview_parent = output_dir / "00_all_theta_routes_overview_v4.png"

        plot_overview(
            df=df,
            route_records=route_records,
            output_file=overview_figures,
            nofly_threshold=nofly_threshold,
            plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
            params=params,
        )

        plot_overview(
            df=df,
            route_records=route_records,
            output_file=overview_parent,
            nofly_threshold=nofly_threshold,
            plot_node_size=float(pget(params, "PLOT_NODE_SIZE", 4.0)),
            params=params,
        )

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Objective table : {output_dir / 'objective_table.csv'}")
    print(f"Planning model  : {planning_model_file}")
    print(f"Route summary   : {summary_file}")
    print(f"All route edges : {edge_file}")
    print(f"Route nodes     : {route_dir}")
    print(f"Figures         : {figure_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        raise
