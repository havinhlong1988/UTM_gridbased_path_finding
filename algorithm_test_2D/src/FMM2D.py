#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMM2D.py

Fast-Marching-style 2D path tracing on node media for LAE-UTM.

This script reads a 2D model node file, applies the current flyable/no-fly
rule, and traces multiple feasible paths from A to B.

Current LAE-UTM rule implemented here:
    - slowness < SLOWNESS_NOFLY_THRESHOLD is flyable
    - slowness >= SLOWNESS_NOFLY_THRESHOLD is no-fly
    - DB / DK / BD / selected start/end nodes are allowed to be forced flyable
      even if they are located on a no-fly node, BUT only if they are not
      completely isolated by no-fly surrounding 8-neighbor nodes.

Important note:
    A true FMM gives one minimum-arrival-time path for one speed/slowness map.
    To obtain many possible alternatives, this script repeatedly runs the
    marching solver and penalizes or blocks a buffer around paths already found.

Input formats supported:
    Headered file with columns like:
        index x y z slowness label
        x y z slowness label
        lon lat z slowness label

    Headerless whitespace or CSV file, common forms:
        index x y z slowness label
        x y z slowness label
        index x y z slowness
        x y z slowness
        x y slowness

Outputs:
    output/fmm2d/*.csv
    figures/fmm2d/*.png
"""

from __future__ import annotations

import heapq
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================
# USER SETTINGS
# ======================================================================

PROJECT_DIR = Path(".").resolve()

# Change this to your real model file if needed.
MODEL_FILE = (
    PROJECT_DIR
    / "input"
    / "model"
    / "senario1"
    / "mixed_model_2d_after_fly_control_for_pathfinding_with_label.xyz"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "fmm2d"
FIGURE_DIR = PROJECT_DIR / "figures" / "fmm2d"

RUN_FMM_PATH_CALCULATION = True
PLOT_RESULT = True

# Start and end can be given by labels or by coordinates.
# If START_LABEL / END_LABEL is not None, label search has priority.
START_LABEL = "DB01"
END_LABEL = "DK01"

# If label is not found, or you prefer coordinates, set labels to None and use these.
# Coordinates must be in the same coordinate system as the model x/y columns.
START_XY: tuple[float, float] | None = None
END_XY: tuple[float, float] | None = None

# Flyable / no-fly rule.
SLOWNESS_NOFLY_THRESHOLD = 10.0
NOFLY_MODE = "greater_equal"  # currently only greater_equal is used

# Value assigned to a forced-flyable DB/DK/BD endpoint if its stored slowness is no-fly.
# If possible, the script uses the median of nearby flyable neighbors; otherwise this value.
FLYABLE_ENDPOINT_SLOWNESS_FALLBACK = 0.085

# Labels that may be forced flyable when sitting on a no-fly node.
# DB is Drone Base, DK is Docking, BD is included because some old files/notes may use BD.
FORCE_FLYABLE_LABEL_PREFIXES = ("DB", "DK", "BD")
FORCE_SELECTED_START_END_FLYABLE = True
ALLOW_FORCE_FLYABLE_ONLY_IF_NOT_ISOLATED = True

# Connectivity for the marching graph.
# 8 is recommended for 2D diagonal + horizontal/vertical movement.
CONNECTIVITY = 8  # 4 or 8

# Multiple path generation.
MAX_PATHS = 30

# How to reduce overlap after a path is found:
#   "penalty"    : keep nodes open, but make cost higher in the previous path buffer
#   "hard_block" : block nodes inside previous path buffer
#   "none"       : no overlap control; usually repeats the same path
PREVIOUS_PATH_ACTION = "penalty"
PATH_BUFFER_M = 150.0
PENALTY_MULTIPLIER = 4.0
MAX_PENALTY_FACTOR = 1.0e6

# Never block/penalize nodes near start and end within this distance.
ENDPOINT_PROTECTION_RADIUS_M = 250.0

# Stop if the new path is too similar to all previous paths.
# Similarity is Jaccard overlap between node sets.
MAX_ALLOWED_NODE_OVERLAP_RATIO = 0.85
MAX_REPEATED_ATTEMPTS = 10

# Optional search corridor to reduce computation.
# If None, all flyable nodes can be searched.
# If a number, after the first path is found, later runs only search inside this
# distance from the first path. This can speed up low-RAM machines but can remove
# far alternative paths.
SEARCH_CORRIDOR_FROM_FIRST_PATH_M: float | None = None

# Plot controls.
PLOT_MAX_BACKGROUND_POINTS = 200_000
PLOT_NODE_SIZE = 2.0
PLOT_PATH_LINEWIDTH = 1.8
PLOT_DPI = 220

# Save full arrival-time field for each path. This can be large.
SAVE_ARRIVAL_TIME_TABLES = False


# ======================================================================
# DATA STRUCTURES
# ======================================================================

@dataclass
class ModelData:
    df: pd.DataFrame
    x_orig: np.ndarray
    y_orig: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    is_lonlat: bool
    ix: np.ndarray
    iy: np.ndarray
    node_at_cell: dict[tuple[int, int], int]


@dataclass
class PathResult:
    path_id: int
    node_indices: list[int]
    travel_time_s: float
    distance_m: float
    status: str
    overlap_ratio: float
    message: str
    arrival_time: np.ndarray | None = None


# ======================================================================
# INPUT READING
# ======================================================================

def _split_tokens(line: str) -> list[str]:
    return [t for t in re.split(r"[\s,]+", line.strip()) if t]


def _is_float_token(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            return s
    raise ValueError(f"No data line found in {path}")


def _standardize_header_columns(columns: Iterable[object]) -> dict[object, str]:
    """Return a rename mapping from input columns to canonical names."""
    mapping: dict[object, str] = {}

    for col in columns:
        low = str(col).strip().lower()
        low_clean = low.replace("-", "_").replace(" ", "_")

        if low_clean in {"id", "idx", "index", "node", "node_id", "node_index"}:
            mapping[col] = "original_index"
        elif low_clean in {"x", "lon", "long", "longitude", "easting", "utm_x"}:
            mapping[col] = "x"
        elif low_clean in {"y", "lat", "latitude", "northing", "utm_y"}:
            mapping[col] = "y"
        elif low_clean in {"z", "elev", "elevation", "height", "alt", "altitude"}:
            mapping[col] = "z"
        elif low_clean in {
            "slow",
            "slowness",
            "slowness_s_m",
            "cost",
            "cost_s_m",
            "travel_slowness",
        }:
            mapping[col] = "slowness"
        elif low_clean in {"label", "labels", "name", "class", "node_label", "type"}:
            mapping[col] = "label"

    return mapping


def read_model_file(model_file: Path) -> pd.DataFrame:
    """Read a model file and return canonical columns.

    Required output columns:
        original_index, x, y, z, slowness, label
    """
    model_file = Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found:\n  {model_file}\n"
            "Please change MODEL_FILE at the top of FMM2D.py."
        )

    first = _first_data_line(model_file)
    tokens = _split_tokens(first)
    first_line_has_header = any(not _is_float_token(t) for t in tokens[:-1])

    # If the first line contains known column names, treat it as a header.
    known_header_words = {
        "x", "y", "z", "lon", "lat", "longitude", "latitude", "slowness",
        "slow", "label", "class", "index", "node", "node_id"
    }
    if any(t.strip().lower() in known_header_words for t in tokens):
        first_line_has_header = True

    if first_line_has_header:
        df = pd.read_csv(
            model_file,
            sep=r"\s+|,",
            engine="python",
            comment="#",
        )
        rename = _standardize_header_columns(df.columns)
        df = df.rename(columns=rename)
    else:
        raw = pd.read_csv(
            model_file,
            sep=r"\s+|,",
            engine="python",
            comment="#",
            header=None,
        )
        ncol = raw.shape[1]
        last_is_label = not _is_float_token(str(raw.iloc[0, ncol - 1]))

        if last_is_label:
            if ncol >= 6:
                # index x y z slowness label
                raw = raw.iloc[:, :6]
                raw.columns = ["original_index", "x", "y", "z", "slowness", "label"]
            elif ncol == 5:
                # x y z slowness label
                raw.columns = ["x", "y", "z", "slowness", "label"]
            elif ncol == 4:
                # x y slowness label
                raw.columns = ["x", "y", "slowness", "label"]
                raw["z"] = 0.0
            else:
                raise ValueError(f"Unsupported headerless format with {ncol} columns: {model_file}")
        else:
            if ncol >= 5:
                # index x y z slowness
                raw = raw.iloc[:, :5]
                raw.columns = ["original_index", "x", "y", "z", "slowness"]
                raw["label"] = "N"
            elif ncol == 4:
                # x y z slowness
                raw.columns = ["x", "y", "z", "slowness"]
                raw["label"] = "N"
            elif ncol == 3:
                # x y slowness
                raw.columns = ["x", "y", "slowness"]
                raw["z"] = 0.0
                raw["label"] = "N"
            else:
                raise ValueError(f"Unsupported headerless format with {ncol} columns: {model_file}")
        df = raw

    required = {"x", "y", "slowness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {model_file}.\n"
            f"Detected columns: {list(df.columns)}"
        )

    if "original_index" not in df.columns:
        df["original_index"] = np.arange(len(df), dtype=int)
    if "z" not in df.columns:
        df["z"] = 0.0
    if "label" not in df.columns:
        df["label"] = "N"

    keep = ["original_index", "x", "y", "z", "slowness", "label"]
    df = df[keep].copy()

    for col in ["x", "y", "z", "slowness"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = df["label"].fillna("N").astype(str).str.strip()
    df["original_index"] = df["original_index"].fillna(np.arange(len(df))).astype(str)

    before = len(df)
    df = df.dropna(subset=["x", "y", "slowness"]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows with invalid x/y/slowness values.")

    return df


# ======================================================================
# COORDINATES AND GRID INDEXING
# ======================================================================

def looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    return (
        np.nanmin(x) >= -180.0
        and np.nanmax(x) <= 180.0
        and np.nanmin(y) >= -90.0
        and np.nanmax(y) <= 90.0
        and (np.nanmax(x) - np.nanmin(x)) < 5.0
        and (np.nanmax(y) - np.nanmin(y)) < 5.0
    )


def lonlat_to_metric(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat to meters. Use pyproj UTM if available; otherwise local approximation."""
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))

    try:
        from pyproj import Transformer

        zone = int(math.floor((lon0 + 180.0) / 6.0) + 1)
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xm, ym = transformer.transform(lon, lat)
        return np.asarray(xm, dtype=float), np.asarray(ym, dtype=float)
    except Exception:
        # Good enough for a small local study area.
        rad = math.pi / 180.0
        x_m = (lon - lon0) * 111_320.0 * math.cos(lat0 * rad)
        y_m = (lat - lat0) * 110_540.0
        return np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)


def build_model_data(df: pd.DataFrame) -> ModelData:
    x_orig = df["x"].to_numpy(float)
    y_orig = df["y"].to_numpy(float)
    is_ll = looks_like_lonlat(x_orig, y_orig)

    if is_ll:
        x_m, y_m = lonlat_to_metric(x_orig, y_orig)
        print("[INFO] x/y look like lon/lat. Internally projected to meters for distance/cost.")
    else:
        x_m, y_m = x_orig.copy(), y_orig.copy()
        print("[INFO] x/y treated as projected/metric coordinates.")

    # Build regular-grid integer coordinates from sorted unique x/y values.
    # Rounding protects against tiny floating point noise.
    decimals = 10 if is_ll else 6
    xr = np.round(x_orig, decimals=decimals)
    yr = np.round(y_orig, decimals=decimals)

    unique_x = np.array(sorted(pd.unique(xr)))
    unique_y = np.array(sorted(pd.unique(yr)))

    x_to_ix = {v: i for i, v in enumerate(unique_x)}
    y_to_iy = {v: i for i, v in enumerate(unique_y)}

    ix = np.fromiter((x_to_ix[v] for v in xr), dtype=np.int64, count=len(df))
    iy = np.fromiter((y_to_iy[v] for v in yr), dtype=np.int64, count=len(df))

    node_at_cell: dict[tuple[int, int], int] = {}
    duplicate_count = 0
    for node_i, cell in enumerate(zip(iy, ix)):
        key = (int(cell[0]), int(cell[1]))
        if key in node_at_cell:
            duplicate_count += 1
            # Keep the first; duplicate rows should normally not exist.
            continue
        node_at_cell[key] = node_i

    if duplicate_count:
        print(f"[WARN] Found {duplicate_count} duplicate grid cells. First node kept for each cell.")

    nx = len(unique_x)
    ny = len(unique_y)
    fill_ratio = len(node_at_cell) / max(nx * ny, 1)
    print(f"[INFO] Grid index: nx={nx:,}, ny={ny:,}, nodes={len(df):,}, fill={fill_ratio:.3f}")

    return ModelData(
        df=df,
        x_orig=x_orig,
        y_orig=y_orig,
        x_m=x_m,
        y_m=y_m,
        is_lonlat=is_ll,
        ix=ix,
        iy=iy,
        node_at_cell=node_at_cell,
    )


# ======================================================================
# FLYABLE / NO-FLY LOGIC
# ======================================================================

def label_has_prefix(label: str, prefixes: tuple[str, ...]) -> bool:
    lab = str(label).strip().upper()
    return any(lab.startswith(p.upper()) for p in prefixes)


def base_flyable_mask(slowness: np.ndarray) -> np.ndarray:
    if NOFLY_MODE != "greater_equal":
        raise ValueError("Only NOFLY_MODE='greater_equal' is currently implemented.")
    return np.isfinite(slowness) & (slowness < SLOWNESS_NOFLY_THRESHOLD)


def neighbor_offsets(connectivity: int = 8) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    raise ValueError("CONNECTIVITY must be 4 or 8")


def get_neighbor_indices(model: ModelData, node_idx: int, connectivity: int = 8) -> list[int]:
    iy = int(model.iy[node_idx])
    ix = int(model.ix[node_idx])
    out: list[int] = []
    for dy, dx in neighbor_offsets(connectivity):
        j = model.node_at_cell.get((iy + dy, ix + dx))
        if j is not None:
            out.append(j)
    return out


def is_node_isolated_by_nofly(
    model: ModelData,
    node_idx: int,
    base_flyable: np.ndarray,
) -> bool:
    """True if none of the surrounding 8-neighbor nodes is flyable."""
    neigh = get_neighbor_indices(model, node_idx, connectivity=8)
    if not neigh:
        return True
    return not bool(np.any(base_flyable[np.asarray(neigh, dtype=int)]))


def local_endpoint_slowness(
    model: ModelData,
    node_idx: int,
    slowness: np.ndarray,
    base_flyable: np.ndarray,
) -> float:
    neigh = get_neighbor_indices(model, node_idx, connectivity=8)
    if neigh:
        vals = slowness[np.asarray(neigh, dtype=int)]
        vals = vals[np.isfinite(vals) & base_flyable[np.asarray(neigh, dtype=int)]]
        vals = vals[(vals > 0.0) & (vals < SLOWNESS_NOFLY_THRESHOLD)]
        if len(vals):
            return float(np.median(vals))
    return float(FLYABLE_ENDPOINT_SLOWNESS_FALLBACK)


def apply_flyable_rules(
    model: ModelData,
    start_idx: int | None,
    end_idx: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    df = model.df
    slowness = df["slowness"].to_numpy(float)
    base = base_flyable_mask(slowness)
    flyable = base.copy()
    effective_slowness = slowness.copy()

    labels = df["label"].astype(str).to_numpy()
    forced_candidates: set[int] = set()

    for i, lab in enumerate(labels):
        if label_has_prefix(lab, FORCE_FLYABLE_LABEL_PREFIXES):
            forced_candidates.add(i)

    if FORCE_SELECTED_START_END_FLYABLE:
        if start_idx is not None:
            forced_candidates.add(int(start_idx))
        if end_idx is not None:
            forced_candidates.add(int(end_idx))

    stats = {
        "base_flyable": int(np.count_nonzero(base)),
        "base_nofly": int(len(base) - np.count_nonzero(base)),
        "forced_endpoint_flyable": 0,
        "forced_endpoint_kept_blocked_isolated": 0,
        "forced_endpoint_already_flyable": 0,
    }

    for i in sorted(forced_candidates):
        if base[i]:
            stats["forced_endpoint_already_flyable"] += 1
            continue

        isolated = is_node_isolated_by_nofly(model, i, base)
        if ALLOW_FORCE_FLYABLE_ONLY_IF_NOT_ISOLATED and isolated:
            flyable[i] = False
            stats["forced_endpoint_kept_blocked_isolated"] += 1
            continue

        flyable[i] = True
        effective_slowness[i] = local_endpoint_slowness(model, i, slowness, base)
        stats["forced_endpoint_flyable"] += 1

    # Clean any invalid or nonpositive effective slowness on flyable nodes.
    bad = flyable & (~np.isfinite(effective_slowness) | (effective_slowness <= 0.0))
    if np.any(bad):
        effective_slowness[bad] = FLYABLE_ENDPOINT_SLOWNESS_FALLBACK

    return flyable, effective_slowness, stats


# ======================================================================
# START / END SELECTION
# ======================================================================

def find_node_by_label(df: pd.DataFrame, wanted: str | None) -> int | None:
    if wanted is None:
        return None
    wanted_norm = str(wanted).strip().upper()
    labels = df["label"].astype(str).str.strip().str.upper()

    exact = np.flatnonzero(labels.to_numpy() == wanted_norm)
    if len(exact):
        return int(exact[0])

    # If the user writes DB1 but file has DB01, or vice versa, compare compact forms.
    def compact(s: str) -> str:
        m = re.match(r"^([A-Z_\-]+)0*([0-9]+)$", s)
        if m:
            return f"{m.group(1)}{int(m.group(2))}"
        return s

    wanted_compact = compact(wanted_norm)
    compact_labels = np.array([compact(x) for x in labels.to_numpy()])
    match = np.flatnonzero(compact_labels == wanted_compact)
    if len(match):
        return int(match[0])

    return None


def find_nearest_node(model: ModelData, xy: tuple[float, float] | None) -> int | None:
    if xy is None:
        return None

    x, y = xy
    if model.is_lonlat:
        x_arr = np.array([x], dtype=float)
        y_arr = np.array([y], dtype=float)
        xm, ym = lonlat_to_metric(x_arr, y_arr)
        qx, qy = float(xm[0]), float(ym[0])
        # The local projection from lonlat_to_metric above is based only on the point;
        # for nearest-node search in small areas, use original lon/lat squared distance instead.
        d2 = (model.x_orig - x) ** 2 + (model.y_orig - y) ** 2
    else:
        qx, qy = x, y
        d2 = (model.x_m - qx) ** 2 + (model.y_m - qy) ** 2
    return int(np.nanargmin(d2))


def resolve_start_end(model: ModelData) -> tuple[int, int]:
    df = model.df

    start_idx = find_node_by_label(df, START_LABEL)
    end_idx = find_node_by_label(df, END_LABEL)

    if start_idx is None:
        start_idx = find_nearest_node(model, START_XY)
    if end_idx is None:
        end_idx = find_nearest_node(model, END_XY)

    if start_idx is None:
        raise ValueError(
            "Could not resolve START node. Set START_LABEL to an existing label "
            "or set START_XY=(x, y)."
        )
    if end_idx is None:
        raise ValueError(
            "Could not resolve END node. Set END_LABEL to an existing label "
            "or set END_XY=(x, y)."
        )

    return int(start_idx), int(end_idx)


# ======================================================================
# FAST MARCHING GRAPH SOLVER
# ======================================================================

def build_neighbor_list(model: ModelData, connectivity: int) -> list[list[int]]:
    n = len(model.df)
    neigh: list[list[int]] = [[] for _ in range(n)]
    offsets = neighbor_offsets(connectivity)
    for i in range(n):
        iy = int(model.iy[i])
        ix = int(model.ix[i])
        out: list[int] = []
        for dy, dx in offsets:
            j = model.node_at_cell.get((iy + dy, ix + dx))
            if j is not None:
                out.append(j)
        neigh[i] = out
    return neigh


def edge_cost_s(
    model: ModelData,
    i: int,
    j: int,
    effective_slowness: np.ndarray,
    penalty_factor: np.ndarray,
) -> float:
    dx = model.x_m[j] - model.x_m[i]
    dy = model.y_m[j] - model.y_m[i]
    dist_m = math.hypot(float(dx), float(dy))

    slow = 0.5 * (effective_slowness[i] + effective_slowness[j])
    penalty = 0.5 * (penalty_factor[i] + penalty_factor[j])
    return dist_m * slow * penalty


def fmm_shortest_arrival(
    model: ModelData,
    neighbors: list[list[int]],
    start_idx: int,
    end_idx: int,
    flyable: np.ndarray,
    effective_slowness: np.ndarray,
    penalty_factor: np.ndarray,
    extra_blocked: np.ndarray | None = None,
    corridor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fast-Marching-style label-setting propagation on the 2D node graph.

    This is equivalent to Dijkstra/FMM on a graph with positive edge costs.
    It is robust for node media and supports holes/no-fly cells.
    """
    n = len(model.df)
    active = flyable.copy()

    if extra_blocked is not None:
        active &= ~extra_blocked
    if corridor_mask is not None:
        active &= corridor_mask

    # Always allow the selected start/end if the flyable rule accepted them.
    active[start_idx] = bool(flyable[start_idx])
    active[end_idx] = bool(flyable[end_idx])

    if not active[start_idx]:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64), "start_blocked"
    if not active[end_idx]:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64), "end_blocked"

    T = np.full(n, np.inf, dtype=float)
    parent = np.full(n, -1, dtype=np.int64)
    accepted = np.zeros(n, dtype=bool)

    T[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]

    while heap:
        t_i, i = heapq.heappop(heap)
        if accepted[i]:
            continue
        if t_i > T[i]:
            continue

        accepted[i] = True
        if i == end_idx:
            return T, parent, "ok"

        for j in neighbors[i]:
            if accepted[j] or not active[j]:
                continue
            c = edge_cost_s(model, i, j, effective_slowness, penalty_factor)
            if not np.isfinite(c) or c <= 0.0:
                continue
            cand = t_i + c
            if cand < T[j]:
                T[j] = cand
                parent[j] = i
                heapq.heappush(heap, (cand, int(j)))

    return T, parent, "unreachable"


def reconstruct_path(parent: np.ndarray, start_idx: int, end_idx: int) -> list[int]:
    if start_idx == end_idx:
        return [int(start_idx)]
    if parent[end_idx] < 0:
        return []

    path = [int(end_idx)]
    current = int(end_idx)
    max_steps = len(parent) + 5
    for _ in range(max_steps):
        current = int(parent[current])
        if current < 0:
            return []
        path.append(current)
        if current == start_idx:
            path.reverse()
            return path
    return []


def path_distance_m(model: ModelData, path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    idx = np.asarray(path, dtype=int)
    dx = np.diff(model.x_m[idx])
    dy = np.diff(model.y_m[idx])
    return float(np.sum(np.hypot(dx, dy)))


def path_overlap_ratio(path: list[int], previous_paths: list[list[int]]) -> float:
    if not previous_paths:
        return 0.0
    s = set(path)
    if not s:
        return 1.0
    best = 0.0
    for p in previous_paths:
        q = set(p)
        inter = len(s & q)
        union = len(s | q)
        if union:
            best = max(best, inter / union)
    return float(best)


# ======================================================================
# PATH BUFFER / PENALTY
# ======================================================================

def _kd_query_within_path(
    model: ModelData,
    path: list[int],
    radius_m: float,
) -> np.ndarray:
    """Return mask of nodes within radius from any node in path."""
    n = len(model.df)
    if radius_m <= 0.0 or not path:
        return np.zeros(n, dtype=bool)

    coords_all = np.column_stack([model.x_m, model.y_m])
    path_coords = coords_all[np.asarray(path, dtype=int)]

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(path_coords)
        dist, _ = tree.query(coords_all, k=1, distance_upper_bound=radius_m)
        return np.isfinite(dist)
    except Exception:
        # Fallback without scipy. Chunk to avoid high memory.
        mask = np.zeros(n, dtype=bool)
        r2 = radius_m * radius_m
        chunk = 25_000
        px = path_coords[:, 0]
        py = path_coords[:, 1]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            ax = coords_all[start:stop, 0][:, None]
            ay = coords_all[start:stop, 1][:, None]
            d2 = (ax - px[None, :]) ** 2 + (ay - py[None, :]) ** 2
            mask[start:stop] = np.any(d2 <= r2, axis=1)
        return mask


def protected_endpoint_mask(
    model: ModelData,
    start_idx: int,
    end_idx: int,
    radius_m: float,
) -> np.ndarray:
    n = len(model.df)
    if radius_m <= 0.0:
        mask = np.zeros(n, dtype=bool)
        mask[start_idx] = True
        mask[end_idx] = True
        return mask

    sx, sy = model.x_m[start_idx], model.y_m[start_idx]
    ex, ey = model.x_m[end_idx], model.y_m[end_idx]
    ds = np.hypot(model.x_m - sx, model.y_m - sy)
    de = np.hypot(model.x_m - ex, model.y_m - ey)
    return (ds <= radius_m) | (de <= radius_m)


def update_overlap_control(
    model: ModelData,
    path: list[int],
    start_idx: int,
    end_idx: int,
    penalty_factor: np.ndarray,
    extra_blocked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if PREVIOUS_PATH_ACTION.lower() == "none":
        return penalty_factor, extra_blocked, 0

    mask = _kd_query_within_path(model, path, PATH_BUFFER_M)
    protected = protected_endpoint_mask(model, start_idx, end_idx, ENDPOINT_PROTECTION_RADIUS_M)
    mask &= ~protected

    n_changed = int(np.count_nonzero(mask))

    action = PREVIOUS_PATH_ACTION.lower()
    if action == "penalty":
        penalty_factor[mask] = np.minimum(
            penalty_factor[mask] * float(PENALTY_MULTIPLIER),
            float(MAX_PENALTY_FACTOR),
        )
    elif action == "hard_block":
        extra_blocked[mask] = True
    else:
        raise ValueError("PREVIOUS_PATH_ACTION must be 'penalty', 'hard_block', or 'none'")

    return penalty_factor, extra_blocked, n_changed


def make_corridor_mask_from_first_path(model: ModelData, first_path: list[int]) -> np.ndarray | None:
    if SEARCH_CORRIDOR_FROM_FIRST_PATH_M is None:
        return None
    mask = _kd_query_within_path(model, first_path, float(SEARCH_CORRIDOR_FROM_FIRST_PATH_M))
    print(
        f"[INFO] Search corridor enabled: {np.count_nonzero(mask):,} nodes "
        f"within {SEARCH_CORRIDOR_FROM_FIRST_PATH_M:.1f} m from first path."
    )
    return mask


# ======================================================================
# MULTI-PATH DRIVER
# ======================================================================

def run_multiple_fmm_paths(
    model: ModelData,
    flyable: np.ndarray,
    effective_slowness: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> list[PathResult]:
    print("\n========== BUILD NEIGHBOR LIST ==========")
    neighbors = build_neighbor_list(model, CONNECTIVITY)
    print(f"[OK] Neighbor list built with {CONNECTIVITY}-connectivity.")

    n = len(model.df)
    penalty_factor = np.ones(n, dtype=float)
    extra_blocked = np.zeros(n, dtype=bool)
    corridor_mask: np.ndarray | None = None

    accepted_paths: list[list[int]] = []
    results: list[PathResult] = []
    repeated_attempts = 0

    print("\n========== RUN FMM MULTI-PATH SEARCH ==========")
    print(f"Start node: {start_idx} | label={model.df.loc[start_idx, 'label']}")
    print(f"End node  : {end_idx} | label={model.df.loc[end_idx, 'label']}")

    for attempt in range(1, MAX_PATHS + MAX_REPEATED_ATTEMPTS + 1):
        T, parent, status = fmm_shortest_arrival(
            model=model,
            neighbors=neighbors,
            start_idx=start_idx,
            end_idx=end_idx,
            flyable=flyable,
            effective_slowness=effective_slowness,
            penalty_factor=penalty_factor,
            extra_blocked=extra_blocked,
            corridor_mask=corridor_mask,
        )

        if status != "ok":
            msg = f"FMM stopped: {status}"
            print(f"[STOP] {msg}")
            results.append(
                PathResult(
                    path_id=len(accepted_paths) + 1,
                    node_indices=[],
                    travel_time_s=float("nan"),
                    distance_m=float("nan"),
                    status=status,
                    overlap_ratio=float("nan"),
                    message=msg,
                    arrival_time=T if SAVE_ARRIVAL_TIME_TABLES else None,
                )
            )
            break

        path = reconstruct_path(parent, start_idx, end_idx)
        if not path:
            msg = "Could not reconstruct path although end was reached."
            print(f"[STOP] {msg}")
            break

        distance_m = path_distance_m(model, path)
        travel_time_s = float(T[end_idx])
        overlap = path_overlap_ratio(path, accepted_paths)

        is_unique = overlap <= MAX_ALLOWED_NODE_OVERLAP_RATIO or not accepted_paths

        if is_unique:
            path_id = len(accepted_paths) + 1
            accepted_paths.append(path)
            result = PathResult(
                path_id=path_id,
                node_indices=path,
                travel_time_s=travel_time_s,
                distance_m=distance_m,
                status="ok",
                overlap_ratio=overlap,
                message="accepted",
                arrival_time=T if SAVE_ARRIVAL_TIME_TABLES else None,
            )
            results.append(result)
            repeated_attempts = 0

            print(
                f"[PATH {path_id:03d}] nodes={len(path):,} | "
                f"distance={distance_m/1000.0:.3f} km | "
                f"time={travel_time_s:.2f} s | overlap={overlap:.3f}"
            )

            if path_id == 1:
                corridor_mask = make_corridor_mask_from_first_path(model, path)

            penalty_factor, extra_blocked, changed = update_overlap_control(
                model, path, start_idx, end_idx, penalty_factor, extra_blocked
            )
            if PREVIOUS_PATH_ACTION.lower() != "none":
                print(
                    f"         overlap-control={PREVIOUS_PATH_ACTION}, "
                    f"buffer_nodes_changed={changed:,}"
                )

            if len(accepted_paths) >= MAX_PATHS:
                print(f"[STOP] Reached MAX_PATHS={MAX_PATHS}.")
                break
        else:
            repeated_attempts += 1
            print(
                f"[SKIP] attempt={attempt}, too similar to previous paths "
                f"(overlap={overlap:.3f}). Increasing overlap control."
            )
            penalty_factor, extra_blocked, changed = update_overlap_control(
                model, path, start_idx, end_idx, penalty_factor, extra_blocked
            )
            if changed == 0 or repeated_attempts >= MAX_REPEATED_ATTEMPTS:
                print("[STOP] No more sufficiently different paths found.")
                break

    return [r for r in results if r.status == "ok" and r.node_indices]


# ======================================================================
# OUTPUT
# ======================================================================

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def path_to_dataframe(model: ModelData, result: PathResult) -> pd.DataFrame:
    idx = np.asarray(result.node_indices, dtype=int)
    df = model.df.iloc[idx].copy().reset_index(drop=True)
    df.insert(0, "path_id", result.path_id)
    df.insert(1, "step", np.arange(len(df), dtype=int))
    df.insert(2, "node_internal_index", idx)

    # Add segment distance and cumulative distance.
    seg = np.zeros(len(df), dtype=float)
    if len(idx) > 1:
        dx = np.diff(model.x_m[idx])
        dy = np.diff(model.y_m[idx])
        seg[1:] = np.hypot(dx, dy)
    df["segment_distance_m"] = seg
    df["cumulative_distance_m"] = np.cumsum(seg)
    df["path_distance_m"] = result.distance_m
    df["path_travel_time_s"] = result.travel_time_s
    df["overlap_ratio"] = result.overlap_ratio
    return df


def save_outputs(model: ModelData, results: list[PathResult], start_idx: int, end_idx: int) -> None:
    ensure_dirs()
    start_name = str(model.df.loc[start_idx, "label"])
    end_name = str(model.df.loc[end_idx, "label"])
    if not start_name or start_name == "N":
        start_name = f"node{start_idx}"
    if not end_name or end_name == "N":
        end_name = f"node{end_idx}"

    safe_start = re.sub(r"[^A-Za-z0-9_\-]+", "_", start_name)
    safe_end = re.sub(r"[^A-Za-z0-9_\-]+", "_", end_name)
    prefix = f"FMM2D_from_{safe_start}_to_{safe_end}"

    summary_rows = []
    all_path_frames = []

    for result in results:
        path_df = path_to_dataframe(model, result)
        all_path_frames.append(path_df)
        path_file = OUTPUT_DIR / f"{prefix}_path_{result.path_id:03d}.csv"
        path_df.to_csv(path_file, index=False)
        print(f"[OK] Saved path: {path_file}")

        summary_rows.append(
            {
                "path_id": result.path_id,
                "status": result.status,
                "nodes": len(result.node_indices),
                "distance_m": result.distance_m,
                "distance_km": result.distance_m / 1000.0,
                "travel_time_s": result.travel_time_s,
                "mean_speed_m_s": result.distance_m / result.travel_time_s
                if result.travel_time_s > 0.0 else np.nan,
                "overlap_ratio": result.overlap_ratio,
                "message": result.message,
            }
        )

        if SAVE_ARRIVAL_TIME_TABLES and result.arrival_time is not None:
            arr_df = model.df.copy()
            arr_df["arrival_time_s"] = result.arrival_time
            arr_file = OUTPUT_DIR / f"{prefix}_arrival_time_{result.path_id:03d}.csv"
            arr_df.to_csv(arr_file, index=False)
            print(f"[OK] Saved arrival-time field: {arr_file}")

    summary = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / f"{prefix}_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"[OK] Saved summary: {summary_file}")

    if all_path_frames:
        all_paths = pd.concat(all_path_frames, ignore_index=True)
        all_paths_file = OUTPUT_DIR / f"{prefix}_all_paths.csv"
        all_paths.to_csv(all_paths_file, index=False)
        print(f"[OK] Saved all paths: {all_paths_file}")


def plot_results(
    model: ModelData,
    flyable: np.ndarray,
    results: list[PathResult],
    start_idx: int,
    end_idx: int,
) -> None:
    if not PLOT_RESULT:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib is not available. Skip plotting. Error: {exc}")
        return

    ensure_dirs()

    n = len(model.df)
    rng = np.random.default_rng(12345)
    if n > PLOT_MAX_BACKGROUND_POINTS:
        bg_idx = rng.choice(n, size=PLOT_MAX_BACKGROUND_POINTS, replace=False)
        bg_idx.sort()
    else:
        bg_idx = np.arange(n)

    x = model.x_orig
    y = model.y_orig

    fig, ax = plt.subplots(figsize=(10, 9))

    bg_fly = bg_idx[flyable[bg_idx]]
    bg_nofly = bg_idx[~flyable[bg_idx]]

    # User requested blue flyable and red no-fly dots in previous plotting workflow.
    ax.scatter(x[bg_fly], y[bg_fly], s=PLOT_NODE_SIZE, c="tab:blue", alpha=0.35, label="Flyable")
    ax.scatter(x[bg_nofly], y[bg_nofly], s=PLOT_NODE_SIZE, c="tab:red", alpha=0.35, label="No-fly")

    for result in results:
        idx = np.asarray(result.node_indices, dtype=int)
        ax.plot(
            x[idx],
            y[idx],
            linewidth=PLOT_PATH_LINEWIDTH,
            label=f"FMM path {result.path_id:02d}",
        )

    ax.scatter(
        [x[start_idx]], [y[start_idx]],
        s=80, marker="*", c="gold", edgecolors="black", linewidths=0.8,
        label=f"Start {model.df.loc[start_idx, 'label']}", zorder=10,
    )
    ax.scatter(
        [x[end_idx]], [y[end_idx]],
        s=80, marker="X", c="lime", edgecolors="black", linewidths=0.8,
        label=f"End {model.df.loc[end_idx, 'label']}", zorder=10,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude" if model.is_lonlat else "X")
    ax.set_ylabel("Latitude" if model.is_lonlat else "Y")
    ax.set_title("FMM2D multiple feasible paths")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    total_paths = len(results)
    best_distance = min((r.distance_m for r in results), default=float("nan"))
    text = (
        f"Flyable nodes : {np.count_nonzero(flyable):,}\n"
        f"No-fly nodes  : {np.count_nonzero(~flyable):,}\n"
        f"Total paths   : {total_paths:,}\n"
        f"Best distance : {best_distance/1000.0:.3f} km\n"
        f"Rule          : slowness < {SLOWNESS_NOFLY_THRESHOLD:g}"
    )
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.4"),
    )

    # Keep legend compact.
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 18:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    else:
        ax.legend(handles[:12], labels[:12], loc="lower left", fontsize=8, framealpha=0.85)

    start_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(model.df.loc[start_idx, "label"]))
    end_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(model.df.loc[end_idx, "label"]))
    fig_file = FIGURE_DIR / f"FMM2D_from_{start_name}_to_{end_name}.png"
    fig.tight_layout()
    fig.savefig(fig_file, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[OK] Saved figure: {fig_file}")


# ======================================================================
# MAIN
# ======================================================================

def print_node_info(model: ModelData, idx: int, name: str) -> None:
    row = model.df.iloc[idx]
    print(
        f"{name}: internal={idx}, original_index={row['original_index']}, "
        f"label={row['label']}, x={row['x']}, y={row['y']}, "
        f"slowness={row['slowness']}"
    )


def main() -> None:
    print("=" * 70)
    print("FMM2D PATH TRACING")
    print("=" * 70)
    print(f"Model file : {MODEL_FILE}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Figure dir : {FIGURE_DIR}")

    df = read_model_file(MODEL_FILE)
    model = build_model_data(df)

    start_idx, end_idx = resolve_start_end(model)
    print("\n========== START / END ==========")
    print_node_info(model, start_idx, "START")
    print_node_info(model, end_idx, "END")

    flyable, effective_slowness, fly_stats = apply_flyable_rules(model, start_idx, end_idx)

    print("\n========== FLYABLE RULE ==========")
    print(f"No-fly threshold                    : slowness >= {SLOWNESS_NOFLY_THRESHOLD:g}")
    print(f"Base flyable nodes                  : {fly_stats['base_flyable']:,}")
    print(f"Base no-fly nodes                   : {fly_stats['base_nofly']:,}")
    print(f"Forced DB/DK/BD already flyable     : {fly_stats['forced_endpoint_already_flyable']:,}")
    print(f"Forced DB/DK/BD changed to flyable  : {fly_stats['forced_endpoint_flyable']:,}")
    print(f"Forced DB/DK/BD kept no-fly isolated: {fly_stats['forced_endpoint_kept_blocked_isolated']:,}")
    print(f"Final flyable nodes                 : {np.count_nonzero(flyable):,}")
    print(f"Final no-fly nodes                  : {np.count_nonzero(~flyable):,}")

    if not flyable[start_idx]:
        print("\n[ERROR] Start node is no-fly after isolation check. No path can start.")
        return
    if not flyable[end_idx]:
        print("\n[ERROR] End node is no-fly after isolation check. No path can end.")
        return

    results: list[PathResult] = []
    if RUN_FMM_PATH_CALCULATION:
        results = run_multiple_fmm_paths(
            model=model,
            flyable=flyable,
            effective_slowness=effective_slowness,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        if results:
            save_outputs(model, results, start_idx, end_idx)
        else:
            print("[WARN] No path results to save.")
    else:
        print("[INFO] RUN_FMM_PATH_CALCULATION=False, skip path search.")

    if PLOT_RESULT:
        plot_results(model, flyable, results, start_idx, end_idx)

    print("\n[DONE] FMM2D finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
