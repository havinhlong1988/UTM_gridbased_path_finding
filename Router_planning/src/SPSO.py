#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node-based Spherical Vector-based Particle Swarm Optimization (SPSO).

This module adapts the MATLAB SPSO idea by Phung & Ha to a node-based
slowness model:

1. A particle is a sequence of spherical movement vectors (r, psi, phi).
2. The vector sequence is converted into continuous Cartesian path points.
3. Each segment is sampled and attached to the nearest node of the input model.
4. The cost function combines distance, travel time from slowness, no-fly
   penalties, outside-grid penalties, and smoothness penalties.

The input model is expected to have columns:
    x/lon  y/lat  z  slowness  label  [label_prefix]

Important project convention:
    slowness >= NOFLY_SLOWNESS_THRESHOLD is treated as hard no-fly by default.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover - scipy is recommended
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


EARTH_RADIUS_M = 6_371_000.0


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------

@dataclass
class NodeModel:
    df: pd.DataFrame
    coord_mode: str
    x_ref: float
    y_ref: float
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    points_m: np.ndarray
    slowness: np.ndarray
    labels: np.ndarray
    label_prefix: np.ndarray
    flyable: np.ndarray
    kdtree: Any
    params: Dict[str, Any]

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def bounds_xy(self) -> Tuple[float, float, float, float]:
        return (
            float(np.nanmin(self.x_m)),
            float(np.nanmax(self.x_m)),
            float(np.nanmin(self.y_m)),
            float(np.nanmax(self.y_m)),
        )

    def to_original_xy(self, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert local meters back to input x/y or lon/lat."""
        x_m = np.asarray(x_m, dtype=float)
        y_m = np.asarray(y_m, dtype=float)
        if self.coord_mode == "lonlat":
            lat0 = math.radians(self.y_ref)
            lon = self.x_ref + np.degrees(x_m / (EARTH_RADIUS_M * math.cos(lat0)))
            lat = self.y_ref + np.degrees(y_m / EARTH_RADIUS_M)
            return lon, lat
        return x_m + self.x_ref, y_m + self.y_ref


@dataclass
class RouteResult:
    start_label: str
    end_label: str
    start_idx: int
    end_idx: int
    success: bool
    best_cost: float
    distance_m: float
    travel_time_s: float
    nofly_hits: int
    outside_hits: int
    repeat_penalty: int
    smooth_penalty: float
    path_node_indices: List[int]
    path_points_m: np.ndarray
    sampled_node_indices: np.ndarray
    sampled_points_m: np.ndarray
    best_cost_history: List[float]
    runtime_s: float
    message: str = ""
    direction: str = "forward"
    route_rank: int = 1
    pair_name: str = ""
    overlap_ratio: float = 0.0
    overlap_nodes: int = 0
    overlap_samples: int = 0


# ----------------------------------------------------------------------
# Model loading and labels
# ----------------------------------------------------------------------

def _is_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return False
    xx = x[finite]
    yy = y[finite]
    return (
        np.nanmin(xx) >= -180.0 and np.nanmax(xx) <= 180.0
        and np.nanmin(yy) >= -90.0 and np.nanmax(yy) <= 90.0
        and (np.nanmax(xx) - np.nanmin(xx)) < 5.0
        and (np.nanmax(yy) - np.nanmin(yy)) < 5.0
    )


def _make_flyable(slowness: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    threshold = float(params.get("NOFLY_SLOWNESS_THRESHOLD", 10.0))
    mode = str(params.get("NOFLY_THRESHOLD_MODE", "greater_equal")).lower()
    if mode == "greater_equal":
        nofly = slowness >= threshold
    elif mode == "greater":
        nofly = slowness > threshold
    elif mode == "equal":
        nofly = np.isclose(slowness, threshold)
    else:
        raise ValueError(f"Unsupported NOFLY_THRESHOLD_MODE: {mode}")
    return np.isfinite(slowness) & (~nofly)


def load_node_model(model_file: str | Path, params: Dict[str, Any]) -> NodeModel:
    """Read a node-based xyz/slowness/label model file."""
    if cKDTree is None:
        raise ImportError(
            "scipy is required for nearest-node queries. Install scipy. "
            f"Original import error: {_SCIPY_IMPORT_ERROR}"
        )

    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    raw = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    if raw.shape[1] < 4:
        raise ValueError(
            "Model file must contain at least 4 columns: x y z slowness [label] [label_prefix]"
        )

    cols = ["x", "y", "z", "slowness", "label", "label_prefix"][: raw.shape[1]]
    raw.columns = cols
    if "label" not in raw.columns:
        raw["label"] = "N"
    if "label_prefix" not in raw.columns:
        raw["label_prefix"] = ""

    for c in ["x", "y", "z", "slowness"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["label"] = raw["label"].fillna("N").astype(str)
    raw["label_prefix"] = raw["label_prefix"].fillna("").astype(str)

    raw = raw.dropna(subset=["x", "y", "z", "slowness"]).reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"No valid numeric rows found in model file: {path}")

    x = raw["x"].to_numpy(float)
    y = raw["y"].to_numpy(float)
    z = raw["z"].to_numpy(float)
    slowness = raw["slowness"].to_numpy(float)

    coord_mode = str(params.get("COORDINATE_MODE", "auto")).lower()
    if coord_mode == "auto":
        coord_mode = "lonlat" if _is_lonlat(x, y) else "xy"
    if coord_mode not in {"lonlat", "xy"}:
        raise ValueError(f"COORDINATE_MODE must be auto, lonlat, or xy. Got: {coord_mode}")

    if coord_mode == "lonlat":
        x_ref = float(np.nanmean(x))
        y_ref = float(np.nanmean(y))
        lat0 = math.radians(y_ref)
        x_m = np.radians(x - x_ref) * math.cos(lat0) * EARTH_RADIUS_M
        y_m = np.radians(y - y_ref) * EARTH_RADIUS_M
    else:
        x_ref = float(np.nanmean(x))
        y_ref = float(np.nanmean(y))
        x_m = x - x_ref
        y_m = y - y_ref

    points_m = np.column_stack([x_m, y_m, z])
    flyable = _make_flyable(slowness, params)

    # Operational endpoints may be allowed even when their slowness is hard-coded.
    if bool(params.get("FORCE_ENDPOINTS_FLYABLE", True)):
        endpoint_prefixes = tuple(str(v) for v in params.get("ENDPOINT_FORCE_PREFIXES", []))
        labels = raw["label"].astype(str).to_numpy()
        for i, lab in enumerate(labels):
            if lab != "N" and lab.startswith(endpoint_prefixes):
                flyable[i] = True

    kdtree = cKDTree(points_m[:, :2])

    return NodeModel(
        df=raw,
        coord_mode=coord_mode,
        x_ref=x_ref,
        y_ref=y_ref,
        x_m=x_m,
        y_m=y_m,
        z_m=z,
        points_m=points_m,
        slowness=slowness,
        labels=raw["label"].to_numpy(str),
        label_prefix=raw["label_prefix"].to_numpy(str),
        flyable=flyable,
        kdtree=kdtree,
        params=params,
    )


def resolve_label_query(model: NodeModel, query: str, mode: str = "auto") -> List[int]:
    """Resolve one label query to row indices."""
    query = str(query)
    mode = str(mode).lower()
    labels = model.labels.astype(str)

    if mode == "exact":
        idx = np.where(labels == query)[0]
    elif mode == "prefix":
        idx = np.array([i for i, lab in enumerate(labels) if lab.startswith(query) and lab != "N"], dtype=int)
    elif mode == "auto":
        exact = np.where(labels == query)[0]
        if len(exact):
            idx = exact
        else:
            idx = np.array([i for i, lab in enumerate(labels) if lab.startswith(query) and lab != "N"], dtype=int)
    else:
        raise ValueError(f"LABEL_MATCH_MODE must be exact, prefix, or auto. Got: {mode}")

    return [int(i) for i in idx]


def build_route_pairs(model: NodeModel, params: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Build all start/end node-index pairs from params."""
    mode = str(params.get("LABEL_MATCH_MODE", "auto"))
    explicit_pairs = params.get("ROUTE_PAIRS", []) or []
    pairs: List[Tuple[int, int]] = []

    if explicit_pairs:
        for pair in explicit_pairs:
            if len(pair) != 2:
                raise ValueError(f"Invalid ROUTE_PAIRS entry: {pair!r}")
            s_query, e_query = pair
            s_idx = resolve_label_query(model, str(s_query), mode)
            e_idx = resolve_label_query(model, str(e_query), mode)
            if not s_idx:
                raise ValueError(f"No start label matched: {s_query}")
            if not e_idx:
                raise ValueError(f"No end label matched: {e_query}")
            for si in s_idx:
                for ei in e_idx:
                    if si != ei:
                        pairs.append((si, ei))
    else:
        start_indices: List[int] = []
        end_indices: List[int] = []
        for q in params.get("START_LABELS", []):
            start_indices.extend(resolve_label_query(model, str(q), mode))
        for q in params.get("END_LABELS", []):
            end_indices.extend(resolve_label_query(model, str(q), mode))
        start_indices = sorted(set(start_indices))
        end_indices = sorted(set(end_indices))
        if not start_indices:
            raise ValueError(f"No start labels matched START_LABELS={params.get('START_LABELS')!r}")
        if not end_indices:
            raise ValueError(f"No end labels matched END_LABELS={params.get('END_LABELS')!r}")
        for si in start_indices:
            for ei in end_indices:
                if si != ei:
                    pairs.append((si, ei))

    max_routes = int(params.get("MAX_ROUTES", 0) or 0)
    if max_routes > 0:
        pairs = pairs[:max_routes]
    return pairs


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _dedupe_preserve_order(indices: Sequence[int]) -> List[int]:
    out: List[int] = []
    last = None
    for idx in indices:
        idx = int(idx)
        if idx != last:
            out.append(idx)
            last = idx
    return out


def _simplify_collinear(points: np.ndarray, indices: List[int], angle_tol_deg: float = 2.0) -> List[int]:
    if len(indices) <= 2:
        return indices
    keep = [indices[0]]
    for i in range(1, len(indices) - 1):
        p0 = points[indices[i - 1], :2]
        p1 = points[indices[i], :2]
        p2 = points[indices[i + 1], :2]
        a = p1 - p0
        b = p2 - p1
        angle = _angle_between(a, b)
        if angle > angle_tol_deg:
            keep.append(indices[i])
    keep.append(indices[-1])
    return keep


def spherical_to_cartesian(
    r: np.ndarray,
    psi: np.ndarray,
    phi: np.ndarray,
    start_xyz: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """
    Convert spherical movement vectors to Cartesian intermediate points.

    Conventional orientation is used here:
        dx = r*cos(psi)*cos(phi)
        dy = r*cos(psi)*sin(phi)
        dz = r*sin(psi)

    This keeps the MATLAB SPSO idea of vectorized movement increments while
    using the common atan2(dy, dx) angle convention for projected coordinates.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    n = len(r)
    pts = np.zeros((n, 3), dtype=float)
    current = np.array(start_xyz, dtype=float).copy()

    for i in range(n):
        dx = r[i] * math.cos(psi[i]) * math.cos(phi[i])
        dy = r[i] * math.cos(psi[i]) * math.sin(phi[i])
        dz = r[i] * math.sin(psi[i])
        current = current + np.array([dx, dy, dz], dtype=float)
        current[0] = min(max(current[0], xmin), xmax)
        current[1] = min(max(current[1], ymin), ymax)
        current[2] = min(max(current[2], zmin), zmax)
        pts[i] = current
    return pts


def sample_polyline(points: np.ndarray, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a polyline. Returns sampled points and approximate segment lengths
    between consecutive sampled points.
    """
    step_m = max(float(step_m), 1.0)
    samples: List[np.ndarray] = []
    seg_lengths: List[float] = []

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        vec = p1 - p0
        length = float(np.linalg.norm(vec[:2])) if points.shape[1] >= 2 else float(np.linalg.norm(vec))
        nseg = max(1, int(math.ceil(length / step_m)))
        for k in range(nseg):
            t = k / nseg
            if i > 0 and k == 0:
                continue
            samples.append(p0 + t * vec)
            seg_lengths.append(length / nseg)
        if i == len(points) - 2:
            samples.append(p1.copy())
            seg_lengths.append(0.0)

    if not samples:
        return points.copy(), np.zeros(len(points), dtype=float)
    return np.vstack(samples), np.asarray(seg_lengths, dtype=float)


# ----------------------------------------------------------------------
# Cost function
# ----------------------------------------------------------------------

def evaluate_path(
    model: NodeModel,
    path_points_m: np.ndarray,
    params: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    avoid_nodes: Optional[set[int]] = None,
) -> Dict[str, Any]:
    """Evaluate one continuous path against the node model."""
    sample_step_m = float(params.get("SAMPLE_STEP_M", 25.0))
    max_dist_m = float(params.get("NEAREST_NODE_MAX_DIST_M", 80.0))

    sampled, sample_lengths = sample_polyline(path_points_m, sample_step_m)
    nn_dist, nn_idx = model.kdtree.query(sampled[:, :2], k=1)
    nn_idx = nn_idx.astype(int)

    outside = nn_dist > max_dist_m
    flyable = model.flyable[nn_idx].copy()

    if bool(params.get("FORCE_ENDPOINTS_FLYABLE", True)):
        # Avoid false no-fly hits right at operational start/end nodes.
        near_start = nn_idx == int(start_idx)
        near_end = nn_idx == int(end_idx)
        flyable[near_start | near_end] = True

    nofly = ~flyable
    nofly_hits = int(np.count_nonzero(nofly))
    outside_hits = int(np.count_nonzero(outside))

    # Travel time = integral slowness ds. For no-fly samples, keep the raw
    # slowness contribution but rely on the hard no-fly penalty for rejection.
    slowness = model.slowness[nn_idx]
    travel_time_s = float(np.sum(slowness * sample_lengths))

    diffs = np.diff(path_points_m[:, :2], axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    distance_m = float(np.sum(seg_lens))

    # Smoothness cost follows the MATLAB idea: penalize turns above a threshold.
    turning_max = float(params.get("TURNING_MAX_DEG", 45.0))
    smooth_penalty = 0.0
    for i in range(1, len(path_points_m) - 1):
        v1 = path_points_m[i, :2] - path_points_m[i - 1, :2]
        v2 = path_points_m[i + 1, :2] - path_points_m[i, :2]
        angle = _angle_between(v1, v2)
        if angle > turning_max:
            smooth_penalty += angle - turning_max

    # Climb smoothness is only meaningful for 3D.
    if str(params.get("PATH_DIMENSION", "2d")).lower() == "3d":
        climb_max = float(params.get("CLIMB_MAX_DEG", 45.0))
        for i in range(1, len(path_points_m) - 1):
            h1 = float(np.linalg.norm(path_points_m[i, :2] - path_points_m[i - 1, :2]))
            h2 = float(np.linalg.norm(path_points_m[i + 1, :2] - path_points_m[i, :2]))
            a1 = math.degrees(math.atan2(path_points_m[i, 2] - path_points_m[i - 1, 2], max(h1, 1e-9)))
            a2 = math.degrees(math.atan2(path_points_m[i + 1, 2] - path_points_m[i, 2], max(h2, 1e-9)))
            da = abs(a2 - a1)
            if da > climb_max:
                smooth_penalty += da - climb_max

    # Repeated snapped nodes mean the path wasted samples in the same cell.
    unique_nodes = np.unique(nn_idx)
    unique_count = len(unique_nodes)
    repeat_penalty = int(max(0, len(nn_idx) - unique_count))

    # Multi-path overlap penalty. The overlap set is built from already accepted
    # paths for the same A-B pair. Endpoint buffer nodes are removed before this
    # function is called, so shared DB/DK/FLZ terminal regions can be allowed.
    if avoid_nodes:
        avoid_arr = np.fromiter((int(v) for v in avoid_nodes), dtype=int)
        avoid_lookup = set(int(v) for v in avoid_arr.tolist())
        overlap_mask = np.array([int(v) in avoid_lookup for v in nn_idx], dtype=bool)
        overlap_samples = int(np.count_nonzero(overlap_mask))
        overlap_nodes = int(sum(1 for v in unique_nodes.tolist() if int(v) in avoid_lookup))
        overlap_ratio = float(overlap_nodes / max(1, unique_count))
    else:
        overlap_samples = 0
        overlap_nodes = 0
        overlap_ratio = 0.0

    max_overlap_ratio = float(params.get("MAX_OVERLAP_RATIO", 0.10))
    w_overlap = float(params.get("W_OVERLAP", 5.0e5))
    overlap_excess = max(0.0, overlap_ratio - max_overlap_ratio)

    w_length = float(params.get("W_LENGTH", 1.0))
    w_time = float(params.get("W_TIME", 1.0))
    w_nofly = float(params.get("W_NOFLY", 1.0e7))
    w_outside = float(params.get("W_OUTSIDE", 1.0e6))
    w_smooth = float(params.get("W_SMOOTH", 25.0))
    w_repeat = float(params.get("W_REPEAT_NODE", 100.0))

    cost = (
        w_length * distance_m
        + w_time * travel_time_s
        + w_nofly * nofly_hits
        + w_outside * outside_hits
        + w_smooth * smooth_penalty
        + w_repeat * repeat_penalty
        + w_overlap * overlap_excess * max(1, unique_count)
    )

    return {
        "cost": float(cost),
        "distance_m": distance_m,
        "travel_time_s": travel_time_s,
        "nofly_hits": nofly_hits,
        "outside_hits": outside_hits,
        "repeat_penalty": repeat_penalty,
        "smooth_penalty": float(smooth_penalty),
        "overlap_ratio": float(overlap_ratio),
        "overlap_nodes": int(overlap_nodes),
        "overlap_samples": int(overlap_samples),
        "sampled_points_m": sampled,
        "sampled_node_indices": nn_idx,
        "sampled_node_dist_m": nn_dist,
    }


# ----------------------------------------------------------------------
# SPSO optimizer
# ----------------------------------------------------------------------

def _initial_particle(
    rng: np.random.Generator,
    n: int,
    var_min: Dict[str, float],
    var_max: Dict[str, float],
    path_dimension: str,
) -> Dict[str, np.ndarray]:
    r = rng.uniform(var_min["r"], var_max["r"], size=n)
    phi = rng.uniform(var_min["phi"], var_max["phi"], size=n)
    if path_dimension == "3d":
        psi = rng.uniform(var_min["psi"], var_max["psi"], size=n)
    else:
        psi = np.zeros(n, dtype=float)
    return {"r": r, "psi": psi, "phi": phi}


def _clip_with_mirror(
    position: np.ndarray,
    velocity: np.ndarray,
    lo: float,
    hi: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = (position < lo) | (position > hi)
    velocity = velocity.copy()
    velocity[out] *= -1.0
    position = np.clip(position, lo, hi)
    return position, velocity


def run(
    model: NodeModel,
    start_idx: int,
    end_idx: int,
    params: Dict[str, Any],
    avoid_nodes: Optional[set[int]] = None,
    direction: str = "forward",
    route_rank: int = 1,
    pair_name: str = "",
    seed_offset: int = 0,
) -> RouteResult:
    """Run node-based SPSO for one start/end pair."""
    t0 = time.perf_counter()
    seed = (
        int(params.get("SEED", 42))
        + int(start_idx) * 1009
        + int(end_idx) * 9176
        + int(route_rank) * 100_003
        + int(seed_offset) * 1_000_003
    )
    rng = np.random.default_rng(seed)

    n = int(params.get("N_WAYPOINTS", 12))
    n_pop = int(params.get("N_POP", 160))
    max_it = int(params.get("MAX_IT", 120))
    if n <= 0:
        raise ValueError("N_WAYPOINTS must be > 0")
    if n_pop <= 0:
        raise ValueError("N_POP must be > 0")
    if max_it <= 0:
        raise ValueError("MAX_IT must be > 0")

    path_dimension = str(params.get("PATH_DIMENSION", "2d")).lower()
    if path_dimension not in {"2d", "3d"}:
        raise ValueError("PATH_DIMENSION must be '2d' or '3d'")

    start_xyz = model.points_m[int(start_idx)].copy()
    end_xyz = model.points_m[int(end_idx)].copy()
    if path_dimension == "2d":
        start_xyz[2] = 0.0
        end_xyz[2] = 0.0

    xmin, xmax, ymin, ymax = model.bounds_xy
    zmin = float(params.get("Z_MIN_M", 0.0))
    zmax = float(params.get("Z_MAX_M", 0.0))
    if path_dimension == "2d":
        zmin = zmax = 0.0
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

    direct_vec = end_xyz - start_xyz
    direct_dist = float(np.linalg.norm(direct_vec[:2]))
    if direct_dist <= 0.0:
        return RouteResult(
            start_label=str(model.labels[start_idx]),
            end_label=str(model.labels[end_idx]),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            success=False,
            best_cost=float("inf"),
            distance_m=0.0,
            travel_time_s=0.0,
            nofly_hits=0,
            outside_hits=0,
            repeat_penalty=0,
            smooth_penalty=0.0,
            path_node_indices=[int(start_idx)],
            path_points_m=np.array([start_xyz]),
            sampled_node_indices=np.array([int(start_idx)], dtype=int),
            sampled_points_m=np.array([start_xyz]),
            best_cost_history=[],
            runtime_s=time.perf_counter() - t0,
            message="Start and end are identical.",
        )

    phi0 = math.atan2(float(direct_vec[1]), float(direct_vec[0]))
    angle_range = math.radians(float(params.get("ANGLE_RANGE_DEG", 75.0)))
    elev_range = math.radians(float(params.get("ELEVATION_ANGLE_RANGE_DEG", 20.0)))
    r_max = float(params.get("R_MAX_FACTOR", 2.0)) * direct_dist / n

    var_min = {"r": 0.0, "psi": -elev_range, "phi": phi0 - angle_range}
    var_max = {"r": r_max, "psi": elev_range, "phi": phi0 + angle_range}

    alpha = float(params.get("VELOCITY_ALPHA", 0.5))
    vel_min = {k: -alpha * (var_max[k] - var_min[k]) for k in var_min}
    vel_max = {k: alpha * (var_max[k] - var_min[k]) for k in var_min}

    # Particle arrays.
    positions: List[Dict[str, np.ndarray]] = []
    velocities: List[Dict[str, np.ndarray]] = []
    pbest_pos: List[Dict[str, np.ndarray]] = []
    pbest_cost = np.full(n_pop, np.inf, dtype=float)

    gbest_position: Optional[Dict[str, np.ndarray]] = None
    gbest_cost = float("inf")
    gbest_eval: Optional[Dict[str, Any]] = None
    gbest_path: Optional[np.ndarray] = None

    for _ in range(n_pop):
        pos = _initial_particle(rng, n, var_min, var_max, path_dimension)
        vel = {
            "r": np.zeros(n, dtype=float),
            "psi": np.zeros(n, dtype=float),
            "phi": np.zeros(n, dtype=float),
        }
        middle = spherical_to_cartesian(pos["r"], pos["psi"], pos["phi"], start_xyz, bounds)
        path = np.vstack([start_xyz, middle, end_xyz])
        ev = evaluate_path(model, path, params, start_idx, end_idx, avoid_nodes=avoid_nodes)
        cost = ev["cost"]

        positions.append(pos)
        velocities.append(vel)
        pbest_pos.append({k: v.copy() for k, v in pos.items()})
        pbest_cost[len(positions) - 1] = cost

        if cost < gbest_cost:
            gbest_cost = cost
            gbest_position = {k: v.copy() for k, v in pos.items()}
            gbest_eval = ev
            gbest_path = path

    if gbest_position is None or gbest_eval is None or gbest_path is None:
        raise RuntimeError("SPSO initialization failed to create any particle.")

    w = float(params.get("INERTIA_WEIGHT", 1.0))
    wdamp = float(params.get("INERTIA_DAMPING", 0.98))
    c1 = float(params.get("C1", 1.5))
    c2 = float(params.get("C2", 1.5))
    verbose = bool(params.get("VERBOSE", True))
    print_every = max(1, int(params.get("PRINT_EVERY", 10)))
    early_stop_iters = int(params.get("EARLY_STOP_ITERS", 35) or 0)

    best_history: List[float] = []
    no_improve = 0
    last_best = gbest_cost

    for it in range(1, max_it + 1):
        for i in range(n_pop):
            for key in ["r", "phi"] + (["psi"] if path_dimension == "3d" else []):
                r1 = rng.random(n)
                r2 = rng.random(n)
                velocities[i][key] = (
                    w * velocities[i][key]
                    + c1 * r1 * (pbest_pos[i][key] - positions[i][key])
                    + c2 * r2 * (gbest_position[key] - positions[i][key])
                )
                velocities[i][key] = np.clip(velocities[i][key], vel_min[key], vel_max[key])
                positions[i][key] = positions[i][key] + velocities[i][key]
                positions[i][key], velocities[i][key] = _clip_with_mirror(
                    positions[i][key], velocities[i][key], var_min[key], var_max[key]
                )

            if path_dimension == "2d":
                positions[i]["psi"][:] = 0.0
                velocities[i]["psi"][:] = 0.0

            middle = spherical_to_cartesian(
                positions[i]["r"], positions[i]["psi"], positions[i]["phi"], start_xyz, bounds
            )
            path = np.vstack([start_xyz, middle, end_xyz])
            ev = evaluate_path(model, path, params, start_idx, end_idx, avoid_nodes=avoid_nodes)
            cost = ev["cost"]

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i] = {k: v.copy() for k, v in positions[i].items()}

                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_position = {k: v.copy() for k, v in positions[i].items()}
                    gbest_eval = ev
                    gbest_path = path

        w *= wdamp
        best_history.append(float(gbest_cost))

        if gbest_cost < last_best - 1e-9:
            no_improve = 0
            last_best = gbest_cost
        else:
            no_improve += 1

        if verbose and (it == 1 or it % print_every == 0 or it == max_it):
            print(
                f"    it={it:4d}/{max_it:<4d} best={gbest_cost:.3f} "
                f"dist={gbest_eval['distance_m']:.1f}m time={gbest_eval['travel_time_s']:.2f}s "
                f"nofly={gbest_eval['nofly_hits']} outside={gbest_eval['outside_hits']} "
                f"overlap={gbest_eval.get('overlap_ratio', 0.0):.3f}"
            )

        if early_stop_iters > 0 and no_improve >= early_stop_iters:
            if verbose:
                print(f"    early stop: no improvement for {early_stop_iters} iterations")
            break

    # Build final node path from sampled nearest nodes.
    sampled_node_indices = np.asarray(gbest_eval["sampled_node_indices"], dtype=int)
    node_path = _dedupe_preserve_order(sampled_node_indices.tolist())
    if bool(params.get("REMOVE_DUPLICATE_NODES", True)):
        node_path = _dedupe_preserve_order(node_path)
    if bool(params.get("SIMPLIFY_COLLINEAR_NODES", False)):
        node_path = _simplify_collinear(model.points_m, node_path)

    # Ensure exact endpoints are present.
    if not node_path or node_path[0] != int(start_idx):
        node_path = [int(start_idx)] + node_path
    if node_path[-1] != int(end_idx):
        node_path.append(int(end_idx))

    success = (int(gbest_eval["nofly_hits"]) == 0) and (int(gbest_eval["outside_hits"]) == 0)
    message = "OK" if success else "Best solution still touches no-fly or outside-grid samples. Increase N_POP/MAX_IT or adjust weights."

    return RouteResult(
        start_label=str(model.labels[start_idx]),
        end_label=str(model.labels[end_idx]),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        success=success,
        best_cost=float(gbest_cost),
        distance_m=float(gbest_eval["distance_m"]),
        travel_time_s=float(gbest_eval["travel_time_s"]),
        nofly_hits=int(gbest_eval["nofly_hits"]),
        outside_hits=int(gbest_eval["outside_hits"]),
        repeat_penalty=int(gbest_eval["repeat_penalty"]),
        smooth_penalty=float(gbest_eval["smooth_penalty"]),
        path_node_indices=[int(v) for v in node_path],
        path_points_m=gbest_path.copy(),
        sampled_node_indices=sampled_node_indices.copy(),
        sampled_points_m=np.asarray(gbest_eval["sampled_points_m"], dtype=float).copy(),
        best_cost_history=best_history,
        runtime_s=time.perf_counter() - t0,
        message=message,
        direction=str(direction),
        route_rank=int(route_rank),
        pair_name=str(pair_name),
        overlap_ratio=float(gbest_eval.get("overlap_ratio", 0.0)),
        overlap_nodes=int(gbest_eval.get("overlap_nodes", 0)),
        overlap_samples=int(gbest_eval.get("overlap_samples", 0)),
    )


# ----------------------------------------------------------------------
# Multi-path helpers and output helpers
# ----------------------------------------------------------------------

def _endpoint_buffer_nodes(model: NodeModel, start_idx: int, end_idx: int, params: Dict[str, Any]) -> set[int]:
    """Return nodes allowed to overlap around terminal operating areas."""
    radius = float(params.get("ENDPOINT_OVERLAP_IGNORE_RADIUS_M", 200.0) or 0.0)
    if radius <= 0.0:
        return {int(start_idx), int(end_idx)}
    nodes: set[int] = set()
    for idx in (int(start_idx), int(end_idx)):
        found = model.kdtree.query_ball_point(model.points_m[idx, :2], r=radius)
        nodes.update(int(v) for v in found)
    return nodes


def _core_node_set(model: NodeModel, result: RouteResult, params: Dict[str, Any]) -> set[int]:
    """Nodes used by a route, excluding the endpoint buffer where overlap is allowed."""
    raw = set(int(v) for v in np.asarray(result.sampled_node_indices, dtype=int).tolist())
    raw.update(int(v) for v in result.path_node_indices)
    raw.difference_update(_endpoint_buffer_nodes(model, result.start_idx, result.end_idx, params))
    return raw


def _union_node_sets(node_sets: Sequence[set[int]]) -> set[int]:
    out: set[int] = set()
    for s in node_sets:
        out.update(int(v) for v in s)
    return out


def _route_file_stem(result: RouteResult) -> str:
    pair = result.pair_name or f"{result.start_label}_to_{result.end_label}"
    return f"{pair}_{result.direction}_path{int(result.route_rank):02d}"


def result_to_summary_row(result: RouteResult) -> Dict[str, Any]:
    return {
        "pair_name": result.pair_name or f"{result.start_label}_to_{result.end_label}",
        "direction": result.direction,
        "route_rank": result.route_rank,
        "start_label": result.start_label,
        "end_label": result.end_label,
        "success": result.success,
        "best_cost": result.best_cost,
        "distance_m": result.distance_m,
        "distance_km": result.distance_m / 1000.0,
        "travel_time_s": result.travel_time_s,
        "nofly_hits": result.nofly_hits,
        "outside_hits": result.outside_hits,
        "overlap_ratio": result.overlap_ratio,
        "overlap_nodes": result.overlap_nodes,
        "overlap_samples": result.overlap_samples,
        "repeat_penalty": result.repeat_penalty,
        "smooth_penalty": result.smooth_penalty,
        "path_nodes": len(result.path_node_indices),
        "sampled_nodes": len(result.sampled_node_indices),
        "runtime_s": result.runtime_s,
        "message": result.message,
    }


def save_route_outputs(model: NodeModel, result: RouteResult, out_dir: str | Path, params: Dict[str, Any]) -> None:
    """Save per-route CSV files and optional plot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    route_name = _route_file_stem(result)

    if bool(params.get("SAVE_ROUTE_NODE_CSV", True)):
        rows = []
        for order, idx in enumerate(result.path_node_indices):
            row = model.df.iloc[int(idx)].to_dict()
            row.update({
                "order": order,
                "node_index": int(idx),
                "pair_name": result.pair_name,
                "direction": result.direction,
                "route_rank": int(result.route_rank),
                "x_m": float(model.x_m[int(idx)]),
                "y_m": float(model.y_m[int(idx)]),
                "flyable": bool(model.flyable[int(idx)]),
            })
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"{route_name}_nodes.csv", index=False)

    if bool(params.get("SAVE_ROUTE_POINT_CSV", True)):
        lon_or_x, lat_or_y = model.to_original_xy(result.path_points_m[:, 0], result.path_points_m[:, 1])
        point_df = pd.DataFrame({
            "order": np.arange(len(result.path_points_m)),
            "pair_name": result.pair_name,
            "direction": result.direction,
            "route_rank": int(result.route_rank),
            "x_or_lon": lon_or_x,
            "y_or_lat": lat_or_y,
            "x_m": result.path_points_m[:, 0],
            "y_m": result.path_points_m[:, 1],
            "z_m": result.path_points_m[:, 2],
        })
        point_df.to_csv(out / f"{route_name}_continuous_points.csv", index=False)

    if bool(params.get("PLOT_ROUTES", True)):
        plot_route(model, result, out / f"{route_name}.png", dpi=int(params.get("PLOT_DPI", 220)))


def plot_route(model: NodeModel, result: RouteResult, out_png: str | Path, dpi: int = 220) -> None:
    """Plot slowness nodes and one SPSO path."""
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = model.x_m
    y = model.y_m
    nofly = ~model.flyable
    fly = model.flyable

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    sc = ax.scatter(x[fly], y[fly], c=model.slowness[fly], s=4, linewidths=0, alpha=0.85)
    ax.scatter(x[nofly], y[nofly], c="black", s=4, linewidths=0, alpha=0.85, label="No-fly")

    labeled = np.where(model.labels != "N")[0]
    for idx in labeled:
        lab = str(model.labels[idx])
        ax.scatter(model.x_m[idx], model.y_m[idx], s=36, marker="o", edgecolors="k", facecolors="white", zorder=5)
        ax.text(model.x_m[idx], model.y_m[idx], f" {lab}", fontsize=7, zorder=6)

    pts = result.path_points_m
    linestyle = "-" if result.direction == "forward" else "--"
    ax.plot(pts[:, 0], pts[:, 1], linestyle, linewidth=2.0,
            label=f"{result.direction} path {result.route_rank:02d}", zorder=7)
    node_pts = model.points_m[result.path_node_indices]
    ax.plot(node_pts[:, 0], node_pts[:, 1], ":", linewidth=1.0, label="Snapped node path", zorder=8)
    ax.scatter(pts[0, 0], pts[0, 1], marker="s", s=70, zorder=9, label="Start")
    ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=130, zorder=9, label="End")

    ax.set_title(
        f"SPSO: {result.start_label} → {result.end_label} | {result.direction} path {result.route_rank:02d}\n"
        f"distance={result.distance_m/1000:.3f} km, time={result.travel_time_s:.2f} s, "
        f"cost={result.best_cost:.2f}, overlap={result.overlap_ratio:.3f}, success={result.success}"
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Slowness (s/m)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _route_bounds_xy(model: NodeModel, results: Sequence[RouteResult], margin_m: float) -> Tuple[float, float, float, float]:
    """Return x/y limits around all continuous and snapped route points."""
    arrays: List[np.ndarray] = []
    for res in results:
        if res.path_points_m is not None and len(res.path_points_m):
            arrays.append(np.asarray(res.path_points_m[:, :2], dtype=float))
        if res.path_node_indices:
            arrays.append(np.asarray(model.points_m[res.path_node_indices, :2], dtype=float))

    if arrays:
        pts = np.vstack(arrays)
        xmin, ymin = np.nanmin(pts, axis=0)
        xmax, ymax = np.nanmax(pts, axis=0)
    else:
        xmin, xmax, ymin, ymax = model.bounds_xy

    margin_m = max(float(margin_m), 0.0)
    if xmax - xmin < 1.0:
        xmin -= 50.0
        xmax += 50.0
    if ymax - ymin < 1.0:
        ymin -= 50.0
        ymax += 50.0

    return float(xmin - margin_m), float(xmax + margin_m), float(ymin - margin_m), float(ymax + margin_m)


def plot_pair_report(
    model: NodeModel,
    results: Sequence[RouteResult],
    out_png: str | Path,
    params: Dict[str, Any],
    zoom: bool = False,
) -> None:
    """Plot all forward/backward route alternatives for one A-B pair.

    When zoom=True, the same report is saved with axis limits focused on
    the route corridor instead of the whole model extent.
    """
    if not results or not bool(params.get("PLOT_ROUTES", True)):
        return

    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = model.x_m
    y = model.y_m
    nofly = ~model.flyable
    fly = model.flyable

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sc = ax.scatter(x[fly], y[fly], c=model.slowness[fly], s=4, linewidths=0, alpha=0.70)
    ax.scatter(x[nofly], y[nofly], c="black", s=4, linewidths=0, alpha=0.85, label="No-fly")

    if zoom:
        xmin, xmax, ymin, ymax = _route_bounds_xy(
            model, results, float(params.get("ZOOM_MARGIN_M", 250.0))
        )
        label_mask = (
            (model.x_m >= xmin) & (model.x_m <= xmax)
            & (model.y_m >= ymin) & (model.y_m <= ymax)
            & (model.labels != "N")
        )
        labeled = np.where(label_mask)[0]
    else:
        labeled = np.where(model.labels != "N")[0]

    for idx in labeled:
        lab = str(model.labels[idx])
        ax.scatter(model.x_m[idx], model.y_m[idx], s=36, marker="o", edgecolors="k", facecolors="white", zorder=5)
        ax.text(model.x_m[idx], model.y_m[idx], f" {lab}", fontsize=7, zorder=6)

    legend_max = max(0, int(params.get("LEGEND_MAX_ROUTES", 12)))
    for i, res in enumerate(results):
        pts = res.path_points_m
        linestyle = "-" if res.direction == "forward" else "--"
        linewidth = 2.2 if res.route_rank == 1 else 1.5
        if legend_max == 0 or i >= legend_max:
            label = "_nolegend_"
        else:
            label = (
                f"{res.direction} {res.route_rank:02d}: "
                f"{res.start_label}→{res.end_label}, ov={res.overlap_ratio:.2f}"
            )
        ax.plot(pts[:, 0], pts[:, 1], linestyle, linewidth=linewidth, label=label, zorder=7)

    pair_name = results[0].pair_name or f"{results[0].start_label}_to_{results[0].end_label}"
    max_overlap = float(params.get("MAX_OVERLAP_RATIO", 0.10))
    suffix = " zoom" if zoom else ""
    ax.set_title(f"SPSO multiple-path report{suffix}: {pair_name}\nallowed overlap ratio ≤ {max_overlap:.2f}")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    if zoom:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Slowness (s/m)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(params.get("PLOT_DPI", 220)))
    plt.close(fig)


def save_summary(results: Sequence[RouteResult], out_dir: str | Path, params: Dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_name = str(params.get("ROUTE_SUMMARY_CSV", "SPSO_route_summary.csv"))
    summary_path = out / summary_name
    pd.DataFrame([result_to_summary_row(r) for r in results]).to_csv(summary_path, index=False)
    return summary_path


def _run_rank_with_retries(
    model: NodeModel,
    start_idx: int,
    end_idx: int,
    params: Dict[str, Any],
    avoid_nodes: set[int],
    direction: str,
    route_rank: int,
    pair_name: str,
) -> RouteResult:
    """Run one alternative path, retrying with stronger overlap penalty if needed."""
    max_overlap = float(params.get("MAX_OVERLAP_RATIO", 0.10))
    attempts = max(1, int(params.get("MULTI_PATH_ATTEMPTS_PER_RANK", 3)))
    base_w_overlap = float(params.get("W_OVERLAP", 5.0e5))
    base_angle_range = float(params.get("ANGLE_RANGE_DEG", 75.0))

    candidates: List[RouteResult] = []
    for attempt in range(attempts):
        local_params = dict(params)
        local_params["W_OVERLAP"] = base_w_overlap * ((attempt + 1) ** 2)
        local_params["ANGLE_RANGE_DEG"] = min(180.0, base_angle_range + attempt * 10.0)

        res = run(
            model,
            start_idx,
            end_idx,
            local_params,
            avoid_nodes=avoid_nodes,
            direction=direction,
            route_rank=route_rank,
            pair_name=pair_name,
            seed_offset=attempt,
        )
        candidates.append(res)

        if res.success and res.overlap_ratio <= max_overlap:
            return res

    def score(r: RouteResult) -> Tuple[int, int, float, float]:
        # Prefer valid/no-overlap first, then lower overlap, then lower cost.
        return (
            0 if r.success else 1,
            0 if r.overlap_ratio <= max_overlap else 1,
            float(r.overlap_ratio),
            float(r.best_cost),
        )

    best = sorted(candidates, key=score)[0]
    if best.overlap_ratio > max_overlap:
        best.message = (
            best.message
            + f" | WARNING: overlap_ratio={best.overlap_ratio:.3f} exceeds MAX_OVERLAP_RATIO={max_overlap:.3f}."
        )
    return best


def _get_n_route(params: Dict[str, Any]) -> int:
    """Return number of route alternatives per enabled direction.

    N_ROUTE is the clearer user-facing name. N_PATHS_PER_DIRECTION is kept
    for compatibility with earlier versions of this project.
    """
    if params.get("N_ROUTE") is not None:
        return max(1, int(params.get("N_ROUTE")))
    return max(1, int(params.get("N_PATHS_PER_DIRECTION", 1)))


def run_pair_multi(model: NodeModel, start_idx: int, end_idx: int, params: Dict[str, Any]) -> List[RouteResult]:
    """Generate multiple forward/backward routes for one A-B pair."""
    n_paths = _get_n_route(params)
    run_forward = bool(params.get("RUN_FORWARD_PATHS", True))
    run_backward = bool(params.get("RUN_BACKWARD_PATHS", False))
    compare_forward_backward = bool(params.get("OVERLAP_COMPARE_FORWARD_BACKWARD", True))

    a_label = str(model.labels[start_idx])
    b_label = str(model.labels[end_idx])
    pair_name = f"{a_label}_to_{b_label}"

    directions: List[Tuple[str, int, int]] = []
    if run_forward:
        directions.append(("forward", int(start_idx), int(end_idx)))
    if run_backward:
        directions.append(("backward", int(end_idx), int(start_idx)))

    results: List[RouteResult] = []
    accepted_all: List[set[int]] = []
    accepted_by_direction: Dict[str, List[set[int]]] = {"forward": [], "backward": []}

    verbose = bool(params.get("VERBOSE", True))
    max_overlap = float(params.get("MAX_OVERLAP_RATIO", 0.10))

    for direction, si, ei in directions:
        for rank in range(1, n_paths + 1):
            if compare_forward_backward:
                avoid_nodes = _union_node_sets(accepted_all)
            else:
                avoid_nodes = _union_node_sets(accepted_by_direction[direction])

            if verbose:
                print(
                    f"\n  {pair_name} | {direction} path {rank}/{n_paths} "
                    f"(avoid_nodes={len(avoid_nodes):,}, max_overlap={max_overlap:.2f})"
                )

            res = _run_rank_with_retries(
                model=model,
                start_idx=si,
                end_idx=ei,
                params=params,
                avoid_nodes=avoid_nodes,
                direction=direction,
                route_rank=rank,
                pair_name=pair_name,
            )
            results.append(res)

            core = _core_node_set(model, res, params)
            accepted_all.append(core)
            accepted_by_direction[direction].append(core)

            if verbose:
                print(
                    f"    selected: success={res.success}, overlap={res.overlap_ratio:.3f}, "
                    f"cost={res.best_cost:.3f}, distance={res.distance_m/1000:.3f} km, "
                    f"time={res.travel_time_s:.2f} s, nodes={len(res.path_node_indices)}, "
                    f"runtime={res.runtime_s:.2f} s"
                )

    return results


def run_all(model: NodeModel, pairs: Sequence[Tuple[int, int]], params: Dict[str, Any]) -> List[RouteResult]:
    """Run SPSO for all route pairs, including multi-path and backward options."""
    results: List[RouteResult] = []
    out_dir = Path(str(params.get("OUTPUT_DIR", "output/SPSO")))
    verbose = bool(params.get("VERBOSE", True))

    for k, (start_idx, end_idx) in enumerate(pairs, start=1):
        start_label = str(model.labels[start_idx])
        end_label = str(model.labels[end_idx])
        pair_name = f"{start_label}_to_{end_label}"
        pair_dir = out_dir / "routes" / pair_name

        if verbose:
            print("\n" + "=" * 70)
            print(f"SPSO pair {k}/{len(pairs)}: {pair_name}")
            print("=" * 70)

        pair_results = run_pair_multi(model, start_idx, end_idx, params)

        for res in pair_results:
            save_route_outputs(model, res, pair_dir, params)
            results.append(res)

        pair_summary = pair_dir / f"{pair_name}_multiple_path_summary.csv"
        pd.DataFrame([result_to_summary_row(r) for r in pair_results]).to_csv(pair_summary, index=False)

        if bool(params.get("PLOT_FULL_REPORT", True)):
            plot_pair_report(model, pair_results, pair_dir / f"{pair_name}_multiple_path_report.png", params, zoom=False)
        if bool(params.get("PLOT_ZOOM_REPORT", True)):
            plot_pair_report(model, pair_results, pair_dir / f"{pair_name}_multiple_path_report_zoom.png", params, zoom=True)

        if verbose:
            ok = sum(1 for r in pair_results if r.success)
            print(f"  pair report: {pair_summary}")
            print(f"  pair done  : {ok}/{len(pair_results)} successful route alternatives")

    save_summary(results, out_dir, params)
    return results
