#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/FMM2D.py

FMM/Fast-Marching-style path search module for the LAE-UTM main.py protocol.

Fast-Marching-style / Dijkstra label-setting on a 2D node graph Not true continuous Fast Marching Method.


This file is NOT a standalone runner.  It is called by main.py as:

    result = src.FMM2D.run(model=model, graph=graph, start_idx=i, end_idx=j, **kwargs)

It uses the model and graph already prepared by main.py, including the shared
flyability/no-fly rule:

    slowness < 10.0   -> flyable
    slowness >= 10.0  -> no-fly

DB/DK/BD/FLZ or selected endpoints may be forced flyable by main.py, but this
module can reject a forced special node when all surrounding neighbours are
blocked/no-fly.

Returned result format is compatible with the existing export/plot section in
main.py:
    - result["path_indices"]          : best path
    - result["path_results"]          : ranked paths, if multiple paths found
    - result["total_cost"]            : best path travel time/cost
    - result["k_paths_found"]         : number of accepted paths
    - result["expanded_states"]       : total expanded nodes over searches
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


# ============================================================
# Parameter helpers
# ============================================================


def _parameters_module():
    try:
        import parameters as P  # type: ignore
        return P
    except Exception:
        return None


def _param(kwargs: dict[str, Any], name: str, default: Any = None) -> Any:
    """Read from kwargs first, then parameters.py, then default."""
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    P = _parameters_module()
    if P is not None and hasattr(P, name):
        return getattr(P, name)
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _is_none_like(value: Any) -> bool:
    """Return True for None / text values used to request auto-until-exhausted mode."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("none", "null", "auto", "all", "unlimited")
    return False


def _optional_positive_int(value: Any, default: int) -> tuple[int, bool]:
    """Parse max-path style values.

    Returns
    -------
    parsed_value : int
        Positive integer to use internally.
    is_auto : bool
        True when the user requested None/auto/all mode.
    """
    if _is_none_like(value):
        return int(default), True
    ivalue = int(value)
    return max(1, ivalue), False


# ============================================================
# Coordinate / distance helpers
# ============================================================


def _get_xy_columns(model: pd.DataFrame) -> tuple[str, str]:
    if {"x", "y"}.issubset(model.columns):
        return "x", "y"
    if {"lon", "lat"}.issubset(model.columns):
        return "lon", "lat"
    raise ValueError("FMM2D requires model columns x/y or lon/lat.")


def _looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    try:
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return False
        xf = x[finite]
        yf = y[finite]
        return (
            np.nanmin(xf) >= -180.0
            and np.nanmax(xf) <= 180.0
            and np.nanmin(yf) >= -90.0
            and np.nanmax(yf) <= 90.0
            and (np.nanmax(xf) - np.nanmin(xf)) < 5.0
            and (np.nanmax(yf) - np.nanmin(yf)) < 5.0
        )
    except Exception:
        return False


def _xy_to_metric(model: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
    xcol, ycol = _get_xy_columns(model)
    x = pd.to_numeric(model[xcol], errors="coerce").to_numpy(dtype=float, copy=True)
    y = pd.to_numeric(model[ycol], errors="coerce").to_numpy(dtype=float, copy=True)

    is_lonlat = _looks_like_lonlat(x, y)
    if not is_lonlat:
        return x, y, False

    # Use a local equirectangular approximation to avoid a hard pyproj dependency.
    lon0 = float(np.nanmean(x))
    lat0 = float(np.nanmean(y))
    rad = math.pi / 180.0
    xm = (x - lon0) * 111_320.0 * math.cos(lat0 * rad)
    ym = (y - lat0) * 110_540.0
    return xm.astype(float), ym.astype(float), True


def _distance_m(xm: np.ndarray, ym: np.ndarray, i: int, j: int) -> float:
    return float(math.hypot(float(xm[j] - xm[i]), float(ym[j] - ym[i])))


# ============================================================
# Label / flyability helpers
# ============================================================


def _label_array(model: pd.DataFrame) -> np.ndarray:
    if "label" in model.columns:
        return model["label"].fillna("N").astype(str).to_numpy(copy=True)
    return np.full(len(model), "N", dtype=object)


def _label_prefix_array(model: pd.DataFrame) -> np.ndarray:
    if "label_prefix" in model.columns:
        return model["label_prefix"].fillna("").astype(str).to_numpy(copy=True)

    labels = _label_array(model)
    out = []
    for lab in labels:
        text = str(lab)
        prefix = ""
        for ch in text:
            if ch.isalpha() or ch == "_":
                prefix += ch
            else:
                break
        out.append(prefix or text)
    return np.asarray(out, dtype=object)


def _starts_with_any(text: str, prefixes: Iterable[str]) -> bool:
    t = str(text).upper()
    return any(t.startswith(str(p).upper()) for p in prefixes)


def _special_mask(model: pd.DataFrame, prefixes: tuple[str, ...]) -> np.ndarray:
    labels = _label_array(model)
    label_prefix = _label_prefix_array(model)
    mask = np.zeros(len(model), dtype=bool)
    for p in prefixes:
        pp = str(p).upper()
        if not pp:
            continue
        mask |= np.char.startswith(labels.astype(str), pp)
        mask |= np.char.startswith(label_prefix.astype(str), pp)
    return mask


def _slowness_array(model: pd.DataFrame, fallback: float) -> np.ndarray:
    if "slowness" not in model.columns:
        return np.full(len(model), float(fallback), dtype=float)
    slow = pd.to_numeric(model["slowness"], errors="coerce").to_numpy(dtype=float, copy=True)
    slow[~np.isfinite(slow)] = float(fallback)
    return slow


def _valid_mask_from_graph(model: pd.DataFrame, graph: dict[str, Any]) -> np.ndarray:
    n = len(model)
    valid = graph.get("valid_indices", None)
    if valid is None:
        if "is_flyable" in model.columns:
            return model["is_flyable"].astype(bool).to_numpy(copy=True)
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    for i in valid:
        ii = int(i)
        if 0 <= ii < n:
            mask[ii] = True
    return mask


# ============================================================
# Graph neighbour extraction
# ============================================================


def _parse_neighbor_item(item: Any) -> int | None:
    """Extract neighbour index from several common adjacency formats."""
    if item is None:
        return None

    if isinstance(item, (int, np.integer)):
        return int(item)

    if isinstance(item, dict):
        for key in ("to", "target", "node", "idx", "index", "neighbor"):
            if key in item:
                try:
                    return int(item[key])
                except Exception:
                    return None
        return None

    if isinstance(item, (tuple, list, np.ndarray)) and len(item) > 0:
        try:
            return int(item[0])
        except Exception:
            return None

    try:
        return int(item)
    except Exception:
        return None


def _find_adjacency_object(graph: dict[str, Any]) -> Any | None:
    """Find an adjacency object inside graph, if build_grid_graph stored one."""
    candidate_keys = (
        "neighbors",
        "neighbours",
        "adjacency",
        "adj",
        "edges",
        "neighbor_indices",
        "graph",
    )
    for key in candidate_keys:
        obj = graph.get(key, None)
        if obj is not None:
            return obj
    return None


def _neighbor_function_from_graph(
    model: pd.DataFrame,
    graph: dict[str, Any],
    xm: np.ndarray,
    ym: np.ndarray,
    valid_mask: np.ndarray,
    connectivity: int,
) -> Callable[[int], list[int]]:
    """Return a function i -> neighbour indices.

    First tries graph adjacency.  If unavailable, builds a low-memory KDTree
    neighbour query using the same radius/max-neighbour hints stored in graph.
    """
    n = len(model)
    adjacency = _find_adjacency_object(graph)

    if adjacency is not None:
        if isinstance(adjacency, dict):
            def neigh_from_dict(i: int) -> list[int]:
                raw = adjacency.get(i, adjacency.get(str(i), []))
                if isinstance(raw, dict):
                    items = raw.keys()
                else:
                    items = raw
                out: list[int] = []
                for item in items:
                    j = _parse_neighbor_item(item)
                    if j is not None and 0 <= j < n:
                        out.append(j)
                return out
            return neigh_from_dict

        if isinstance(adjacency, (list, tuple)) and len(adjacency) >= n:
            def neigh_from_list(i: int) -> list[int]:
                raw = adjacency[int(i)]
                if isinstance(raw, dict):
                    items = raw.keys()
                else:
                    items = raw
                out: list[int] = []
                for item in items:
                    j = _parse_neighbor_item(item)
                    if j is not None and 0 <= j < n:
                        out.append(j)
                return out
            return neigh_from_list

    # Fallback: query by radius from coordinates.
    # This is still low-memory compared with building a dense full graph.
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "FMM2D could not find graph adjacency, and scipy.spatial.cKDTree "
            "is not available for fallback neighbour search. Please make sure "
            "build_grid_graph() stores graph['neighbors'] or install scipy."
        ) from exc

    coords = np.column_stack([xm, ym]).copy()
    tree = cKDTree(coords)

    radius_m = graph.get("neighbor_radius_m", None)
    if radius_m is None or float(radius_m) <= 0.0:
        spacing = graph.get("grid_spacing_m", None)
        if spacing is not None and float(spacing) > 0.0:
            radius_m = 1.60 * float(spacing)
        else:
            # Estimate from nearest neighbour distance on a small sample.
            sample_n = min(len(coords), 5000)
            sample = coords[:sample_n]
            sample_tree = cKDTree(sample)
            d, _ = sample_tree.query(sample, k=2)
            nn = d[:, 1]
            nn = nn[np.isfinite(nn) & (nn > 0.0)]
            radius_m = 1.60 * float(np.median(nn)) if len(nn) else 1.0

    max_neighbors = int(graph.get("max_neighbors", 8 if connectivity == 8 else 4))
    max_neighbors = max(max_neighbors, 8 if connectivity == 8 else 4)

    def neigh_from_kdtree(i: int) -> list[int]:
        inds = tree.query_ball_point(coords[int(i)], r=float(radius_m))
        inds = [int(j) for j in inds if int(j) != int(i)]
        if not inds:
            return []
        # Keep closest candidates only.
        inds.sort(key=lambda j: (coords[j, 0] - coords[i, 0]) ** 2 + (coords[j, 1] - coords[i, 1]) ** 2)
        return inds[:max_neighbors]

    return neigh_from_kdtree


def _count_active_neighbors(neighbor_fn: Callable[[int], list[int]], active: np.ndarray, idx: int) -> int:
    count = 0
    for j in neighbor_fn(int(idx)):
        if 0 <= int(j) < len(active) and bool(active[int(j)]):
            count += 1
    return count


# ============================================================
# Slowness correction for forced endpoints/facilities
# ============================================================


def _make_effective_slowness(
    model: pd.DataFrame,
    neighbor_fn: Callable[[int], list[int]],
    active: np.ndarray,
    start_idx: int,
    end_idx: int,
    kwargs: dict[str, Any],
) -> np.ndarray:
    threshold = float(_param(kwargs, "NO_FLY_SLOWNESS_THRESHOLD", 10.0))
    fallback = float(_param(kwargs, "FMM2D_FLYABLE_ENDPOINT_SLOWNESS_FALLBACK", _param(kwargs, "FLYABLE_SLOWNESS", 0.085)))
    special_prefixes = tuple(_param(kwargs, "ALWAYS_FLYABLE_PREFIXES", ("DB", "DK", "BD", "FLZ")))

    slow = _slowness_array(model, fallback=fallback)
    eff = slow.copy()

    special = _special_mask(model, special_prefixes)
    special[int(start_idx)] = True
    special[int(end_idx)] = True

    # If a forced special endpoint/facility has no-fly slowness, do not use the
    # no-fly value as travel time. Use median flyable neighbour slowness.
    forced = active & special & (~np.isfinite(eff) | (eff >= threshold) | (eff <= 0.0))
    for i in np.flatnonzero(forced):
        neigh = neighbor_fn(int(i))
        vals = []
        for j in neigh:
            jj = int(j)
            if 0 <= jj < len(slow) and active[jj] and np.isfinite(slow[jj]) and 0.0 < slow[jj] < threshold:
                vals.append(float(slow[jj]))
        eff[i] = float(np.median(vals)) if vals else fallback

    bad = active & (~np.isfinite(eff) | (eff <= 0.0))
    eff[bad] = fallback
    return eff


def _apply_isolated_special_rule(
    model: pd.DataFrame,
    neighbor_fn: Callable[[int], list[int]],
    active: np.ndarray,
    start_idx: int,
    end_idx: int,
    kwargs: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    """Block special nodes with no active 8-neighbour access if requested."""
    use_rule = _as_bool(_param(kwargs, "SPECIAL_NODE_BLOCK_IF_ALL_8_NEIGHBORS_NOFLY", True))
    if not use_rule:
        return active, []

    prefixes = tuple(_param(kwargs, "ALWAYS_FLYABLE_PREFIXES", ("DB", "DK", "BD", "FLZ")))
    special = _special_mask(model, prefixes)

    # Always check start/end even if their label is unusual.
    special[int(start_idx)] = True
    special[int(end_idx)] = True

    active2 = active.copy()
    blocked: list[int] = []
    for i in np.flatnonzero(special & active):
        # Temporarily ignore the node itself.  A node with no active neighbours
        # cannot connect to the flyable graph even if main.py forced it valid.
        n_active = _count_active_neighbors(neighbor_fn, active, int(i))
        if n_active <= 0:
            active2[int(i)] = False
            blocked.append(int(i))

    return active2, blocked


# ============================================================
# FMM / Dijkstra label-setting solver
# ============================================================


def _edge_cost_s(
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    penalty: np.ndarray,
    i: int,
    j: int,
) -> float:
    dist = _distance_m(xm, ym, int(i), int(j))
    if dist <= 0.0 or not np.isfinite(dist):
        return float("inf")
    s = 0.5 * (float(slow[int(i)]) + float(slow[int(j)]))
    p = 0.5 * (float(penalty[int(i)]) + float(penalty[int(j)]))
    c = dist * s * p
    return float(c) if np.isfinite(c) and c > 0.0 else float("inf")


def _fmm_search(
    *,
    n: int,
    neighbor_fn: Callable[[int], list[int]],
    active: np.ndarray,
    blocked: np.ndarray,
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    penalty: np.ndarray,
    start_idx: int,
    end_idx: int,
    max_expanded_nodes: int | None,
    inf_time: float,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    usable = active & (~blocked)
    usable[int(start_idx)] = bool(active[int(start_idx)])
    usable[int(end_idx)] = bool(active[int(end_idx)])

    T = np.full(n, float(inf_time), dtype=float)
    parent = np.full(n, -1, dtype=np.int64)

    if not usable[int(start_idx)]:
        return T, parent, "start_blocked", 0
    if not usable[int(end_idx)]:
        return T, parent, "end_blocked", 0

    accepted = np.zeros(n, dtype=bool)
    T[int(start_idx)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]
    expanded = 0

    while heap:
        t_i, i = heapq.heappop(heap)
        if accepted[i]:
            continue
        if t_i > T[i]:
            continue

        accepted[i] = True
        expanded += 1

        if i == int(end_idx):
            return T, parent, "ok", expanded

        if max_expanded_nodes is not None and expanded >= int(max_expanded_nodes):
            return T, parent, "max_expanded_nodes", expanded

        for j in neighbor_fn(i):
            jj = int(j)
            if jj < 0 or jj >= n:
                continue
            if accepted[jj] or not usable[jj]:
                continue
            c = _edge_cost_s(xm, ym, slow, penalty, i, jj)
            if not np.isfinite(c):
                continue
            cand = t_i + c
            if cand < T[jj]:
                T[jj] = cand
                parent[jj] = int(i)
                heapq.heappush(heap, (float(cand), jj))

    return T, parent, "unreachable", expanded


def _reconstruct_path(parent: np.ndarray, start_idx: int, end_idx: int) -> list[int]:
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if start_idx == end_idx:
        return [start_idx]
    if parent[end_idx] < 0:
        return []

    path = [end_idx]
    current = end_idx
    for _ in range(len(parent) + 5):
        current = int(parent[current])
        if current < 0:
            return []
        path.append(current)
        if current == start_idx:
            path.reverse()
            return [int(i) for i in path]
    return []


def _path_distance_m(xm: np.ndarray, ym: np.ndarray, path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    idx = np.asarray(path, dtype=int)
    dx = np.diff(xm[idx])
    dy = np.diff(ym[idx])
    return float(np.sum(np.hypot(dx, dy)))


def _path_overlap_ratio(path: list[int], previous_paths: list[list[int]]) -> float:
    if not previous_paths:
        return 0.0
    s = set(int(i) for i in path)
    if not s:
        return 1.0
    best = 0.0
    for old in previous_paths:
        q = set(int(i) for i in old)
        union = len(s | q)
        if union:
            best = max(best, len(s & q) / union)
    return float(best)


# ============================================================
# Multiple-path overlap control
# ============================================================


def _nodes_within_path_buffer(
    tree: Any,
    coords: np.ndarray,
    path: list[int],
    radius_m: float,
    low_memory: bool,
) -> np.ndarray:
    n = len(coords)
    mask = np.zeros(n, dtype=bool)
    if radius_m <= 0.0 or not path:
        return mask

    path_idx = np.asarray(path, dtype=int)

    # Query path nodes in chunks to avoid one very large Python list on low RAM.
    chunk = 256 if low_memory else 2048
    for start in range(0, len(path_idx), chunk):
        sub = coords[path_idx[start:start + chunk]]
        found = tree.query_ball_point(sub, r=float(radius_m))
        for item in found:
            if item:
                mask[np.asarray(item, dtype=int)] = True
    return mask


def _endpoint_protection_mask(coords: np.ndarray, start_idx: int, end_idx: int, radius_m: float) -> np.ndarray:
    n = len(coords)
    if radius_m <= 0.0:
        mask = np.zeros(n, dtype=bool)
        mask[int(start_idx)] = True
        mask[int(end_idx)] = True
        return mask
    ds = np.hypot(coords[:, 0] - coords[int(start_idx), 0], coords[:, 1] - coords[int(start_idx), 1])
    de = np.hypot(coords[:, 0] - coords[int(end_idx), 0], coords[:, 1] - coords[int(end_idx), 1])
    return (ds <= float(radius_m)) | (de <= float(radius_m))


def _update_overlap_control(
    *,
    tree: Any,
    coords: np.ndarray,
    path: list[int],
    start_idx: int,
    end_idx: int,
    penalty: np.ndarray,
    blocked: np.ndarray,
    kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    action = str(_param(kwargs, "FMM2D_PREVIOUS_PATH_ACTION", "penalty")).strip().lower()
    if action in ("none", "allow"):
        return penalty, blocked, 0

    buffer_m = float(_param(kwargs, "FMM2D_PATH_BUFFER_M", _param(kwargs, "PATH_BUFFER_M", _param(kwargs, "MULTI_PATH_NON_OVERLAP_BUFFER_RADIUS_M", 150.0))))
    endpoint_radius_m = float(_param(kwargs, "FMM2D_ENDPOINT_PROTECTION_RADIUS_M", max(buffer_m, 150.0)))
    low_memory = _as_bool(_param(kwargs, "FMM2D_LOW_MEMORY_MODE", _param(kwargs, "LOW_MEMORY_MODE", True)))

    mask = _nodes_within_path_buffer(tree, coords, path, buffer_m, low_memory=low_memory)
    mask &= ~_endpoint_protection_mask(coords, start_idx, end_idx, endpoint_radius_m)

    changed = int(np.count_nonzero(mask))

    if action in ("penalty", "penalize", "soft"):
        factor = float(_param(kwargs, "FMM2D_PREVIOUS_PATH_PENALTY_FACTOR", 5.0))
        max_factor = float(_param(kwargs, "FMM2D_MAX_PENALTY_FACTOR", 1.0e6))
        penalty[mask] = np.minimum(penalty[mask] * factor, max_factor)
    elif action in ("block", "hard_block", "non_overlap"):
        blocked[mask] = True
    else:
        raise ValueError(
            "FMM2D_PREVIOUS_PATH_ACTION must be 'penalty', 'block', 'hard_block', or 'none'."
        )

    blocked[int(start_idx)] = False
    blocked[int(end_idx)] = False
    return penalty, blocked, changed


# ============================================================
# Public algorithm entry point required by main.py
# ============================================================


def run(model: pd.DataFrame, graph: dict[str, Any], start_idx: int, end_idx: int, **kwargs) -> dict[str, Any]:
    """Run FMM2D using the model/graph already prepared by main.py.

    Parameters are read from params/FMM2D.params through parameters.py when
    available.  Generic MULTI_PATH_* kwargs from main.py are accepted but do
    not need to include FMM2D-specific values.
    """
    n = len(model)
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if n <= 0:
        return {"success": False, "path_indices": [], "message": "empty model"}

    # One FMM2D module supports both fastest and multiple modes.
    #
    # Recommended control in params/FMM2D.params:
    #   FMM2D_MODE = "fastest"  -> force one fastest path only
    #   FMM2D_MODE = "multiple" -> use FMM2D_MAX_PATHS, including None/auto
    #   FMM2D_MODE = "auto"     -> infer from FMM2D_MAX_PATHS
    #
    # FMM2D must not silently inherit MULTI_PATH_K_PATHS from astar_multiple.
    # Default is one fastest path unless params/FMM2D.params explicitly sets
    # FMM2D_MODE="multiple" and/or FMM2D_MAX_PATHS.
    fmm2d_mode = str(_param(kwargs, "FMM2D_MODE", "auto")).strip().lower()

    if fmm2d_mode in ("fastest", "single", "one", "best", "shortest_time"):
        raw_max_paths = 1
    elif fmm2d_mode in ("multiple", "multi", "alternatives", "alternative"):
        raw_max_paths = _param(kwargs, "FMM2D_MAX_PATHS", None)
    elif fmm2d_mode in ("auto", "from_max_paths", "default"):
        raw_max_paths = _param(kwargs, "FMM2D_MAX_PATHS", 1)
    else:
        raise ValueError(
            "FMM2D_MODE must be 'fastest', 'multiple', or 'auto'. "
            f"Got: {fmm2d_mode!r}"
        )
    safety_max_paths = int(_param(kwargs, "FMM2D_MAX_PATHS_SAFETY", 100))
    max_paths, auto_until_exhausted = _optional_positive_int(raw_max_paths, safety_max_paths)

    max_rounds_per_path = int(_param(kwargs, "FMM2D_MAX_ROUNDS_PER_PATH", kwargs.get("max_rounds_per_path", 3)))
    max_rounds_per_path = max(1, max_rounds_per_path)

    max_total_attempts_default = max_paths * max_rounds_per_path
    if auto_until_exhausted:
        max_total_attempts_default = max(max_total_attempts_default, 2 * max_paths)
    max_total_attempts = int(_param(kwargs, "FMM2D_MAX_TOTAL_ATTEMPTS", max_total_attempts_default))
    max_total_attempts = max(1, max_total_attempts)

    max_repeated_attempts = int(_param(kwargs, "FMM2D_MAX_REPEATED_ATTEMPTS", max_rounds_per_path))
    max_repeated_attempts = max(1, max_repeated_attempts)

    # Clause requested for clean fastest-path testing:
    # if only one path is requested, do not apply any previous-path penalty/block.
    previous_path_action = str(_param(kwargs, "FMM2D_PREVIOUS_PATH_ACTION", "penalty")).strip().lower()
    if max_paths == 1 and not auto_until_exhausted:
        previous_path_action = "none"
        kwargs = dict(kwargs)
        kwargs["FMM2D_PREVIOUS_PATH_ACTION"] = "none"

    max_overlap = float(_param(kwargs, "FMM2D_MAX_ALLOWED_NODE_OVERLAP_RATIO", 0.85))
    inf_time = float(_param(kwargs, "FMM2D_INF_TIME", 1.0e30))
    max_expanded_nodes = _param(kwargs, "FMM2D_MAX_EXPANDED_NODES", kwargs.get("max_expansions", None))
    if max_expanded_nodes is not None:
        max_expanded_nodes = int(max_expanded_nodes)

    connectivity = int(_param(kwargs, "CONNECTIVITY_2D", graph.get("connectivity", 8)))
    low_memory = _as_bool(_param(kwargs, "FMM2D_LOW_MEMORY_MODE", _param(kwargs, "LOW_MEMORY_MODE", True)))
    verbose = _as_bool(_param(kwargs, "FMM2D_VERBOSE", kwargs.get("verbose", True)))

    xm, ym, is_lonlat = _xy_to_metric(model)
    coords = np.column_stack([xm, ym]).copy()

    active = np.asarray(_valid_mask_from_graph(model, graph), dtype=bool).copy()
    active[start_idx] = bool(active[start_idx])
    active[end_idx] = bool(active[end_idx])

    neighbor_fn = _neighbor_function_from_graph(
        model=model,
        graph=graph,
        xm=xm,
        ym=ym,
        valid_mask=active,
        connectivity=connectivity,
    )

    active, isolated_blocked = _apply_isolated_special_rule(
        model=model,
        neighbor_fn=neighbor_fn,
        active=active,
        start_idx=start_idx,
        end_idx=end_idx,
        kwargs=kwargs,
    )

    if start_idx in isolated_blocked:
        return {
            "success": False,
            "path_indices": [],
            "path_results": [],
            "total_cost": float("inf"),
            "k_paths_requested": None if auto_until_exhausted else int(max_paths),
            "k_paths_safety_cap": int(max_paths),
            "fmm2d_auto_until_exhausted": bool(auto_until_exhausted),
            "k_paths_found": 0,
            "expanded_states": 0,
            "message": "FMM2D start node is special/forced but all surrounding neighbours are no-fly.",
            "isolated_special_blocked": isolated_blocked,
        }

    if end_idx in isolated_blocked:
        return {
            "success": False,
            "path_indices": [],
            "path_results": [],
            "total_cost": float("inf"),
            "k_paths_requested": max_paths,
            "k_paths_found": 0,
            "expanded_states": 0,
            "message": "FMM2D end node is special/forced but all surrounding neighbours are no-fly.",
            "isolated_special_blocked": isolated_blocked,
        }

    effective_slowness = _make_effective_slowness(
        model=model,
        neighbor_fn=neighbor_fn,
        active=active,
        start_idx=start_idx,
        end_idx=end_idx,
        kwargs=kwargs,
    )
    effective_slowness = np.asarray(effective_slowness, dtype=float).copy()

    # ------------------------------------------------------------
    # Fast path: when only one path is requested, do exactly one FMM
    # propagation and return.  Do not build cKDTree for path buffers,
    # do not apply overlap control, and do not enter the multi-path loop.
    # ------------------------------------------------------------
    if max_paths == 1 and not auto_until_exhausted:
        previous_path_action = "none"
        penalty = np.ones(n, dtype=float)
        blocked = np.zeros(n, dtype=bool)

        if verbose:
            print("      FMM2D FASTEST-PATH MODE:")
            print("        max paths        : 1")
            print("        overlap action   : none")
            print("        note             : no multi-path penalty/blocking is used")
            print(f"        low memory       : {low_memory}")
            print(f"        connectivity     : {connectivity}")
            print(f"        active nodes     : {int(np.count_nonzero(active)):,}")

        T, parent, status, expanded = _fmm_search(
            n=n,
            neighbor_fn=neighbor_fn,
            active=active,
            blocked=blocked,
            xm=xm,
            ym=ym,
            slow=effective_slowness,
            penalty=penalty,
            start_idx=start_idx,
            end_idx=end_idx,
            max_expanded_nodes=max_expanded_nodes,
            inf_time=inf_time,
        )

        if status != "ok":
            return {
                "success": False,
                "algorithm": "FMM2D",
                "path_indices": [],
                "path_results": [],
                "ranked_paths": [],
                "total_cost": float("inf"),
                "travel_cost": float("inf"),
                "k_paths_requested": 1,
                "k_paths_found": 0,
                "expanded_states": int(expanded),
                "message": f"FMM2D failed: {status}",
                "isolated_special_blocked": isolated_blocked,
                "is_lonlat": bool(is_lonlat),
                "fmm2d_previous_path_action": "none",
            }

        path = _reconstruct_path(parent, start_idx, end_idx)
        if not path:
            return {
                "success": False,
                "algorithm": "FMM2D",
                "path_indices": [],
                "path_results": [],
                "ranked_paths": [],
                "total_cost": float("inf"),
                "travel_cost": float("inf"),
                "k_paths_requested": 1,
                "k_paths_found": 0,
                "expanded_states": int(expanded),
                "message": "FMM2D failed: path reconstruction failed",
                "isolated_special_blocked": isolated_blocked,
                "is_lonlat": bool(is_lonlat),
                "fmm2d_previous_path_action": "none",
            }

        distance_m = _path_distance_m(xm, ym, path)
        travel_time_s = float(T[end_idx])
        item = {
            "rank": 1,
            "path_indices": [int(i) for i in path],
            "total_cost": float(travel_time_s),
            "cost": float(travel_time_s),
            "travel_cost": float(travel_time_s),
            "turn_cost": 0.0,
            "turn_count": 0,
            "total_turn_angle_degree": 0.0,
            "nodes": int(len(path)),
            "distance_m": float(distance_m),
            "distance_km": float(distance_m / 1000.0),
            "estimated_traveltime_s": float(travel_time_s),
            "estimated_traveltime_min": float(travel_time_s / 60.0),
            "overlap_ratio": 0.0,
            "expanded_states": int(expanded),
            "round_id": 1,
            "status": "ok",
        }

        if verbose:
            print(
                f"      FMM2D fastest path: nodes={len(path):,}, "
                f"distance={distance_m/1000.0:.4f} km, "
                f"time={travel_time_s:.2f} s"
            )

        return {
            "success": True,
            "algorithm": "FMM2D",
            "path_indices": item["path_indices"],
            "path_results": [item],
            "ranked_paths": [item],
            "total_cost": float(item["total_cost"]),
            "travel_cost": float(item["travel_cost"]),
            "turn_cost": 0.0,
            "turn_count": 0,
            "total_turn_angle_degree": 0.0,
            "nodes": int(item["nodes"]),
            "distance_m": float(item["distance_m"]),
            "distance_km": float(item["distance_km"]),
            "estimated_traveltime_s": float(item["estimated_traveltime_s"]),
            "estimated_traveltime_min": float(item["estimated_traveltime_min"]),
            "k_paths_requested": 1,
            "k_paths_safety_cap": 1,
            "fmm2d_auto_until_exhausted": False,
            "fmm2d_total_attempts": 1,
            "fmm2d_max_total_attempts": 1,
            "k_paths_found": 1,
            "expanded_states": int(expanded),
            "message": "ok",
            "isolated_special_blocked": isolated_blocked,
            "is_lonlat": bool(is_lonlat),
            "fmm2d_previous_path_action": "none",
            "fmm2d_mode": str(fmm2d_mode),
            "fmm2d_fastest_path_mode": True,
        }

    tree = None
    if previous_path_action not in ("none", "allow"):
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
        except Exception as exc:
            raise RuntimeError("FMM2D requires scipy.spatial.cKDTree for path-buffer overlap control.") from exc

    penalty = np.ones(n, dtype=float)
    blocked = np.zeros(n, dtype=bool)

    accepted_paths: list[list[int]] = []
    path_results: list[dict[str, Any]] = []
    total_expanded = 0
    stop_reason = ""

    if verbose:
        print("      FMM2D settings:")
        print(f"        max paths        : {'auto/None -> ' + str(max_paths) + ' safety cap' if auto_until_exhausted else max_paths}")
        print(f"        rounds per path  : {max_rounds_per_path}")
        print(f"        total attempts   : {max_total_attempts}")
        print(f"        low memory       : {low_memory}")
        print(f"        connectivity     : {connectivity}")
        print(f"        active nodes     : {int(np.count_nonzero(active)):,}")
        print(f"        isolated special : {len(isolated_blocked):,}")
        print(f"        overlap action   : {previous_path_action}")
        if previous_path_action == "none" and max_paths == 1 and not auto_until_exhausted:
            print("        note             : one-path mode forces overlap action = none")

    # Try to accept max_paths.  If FMM2D_MAX_PATHS=None, max_paths is a safety
    # cap and the loop stops naturally when no new acceptable path can be found.
    # If a candidate is too similar, update penalty/block and retry for the same
    # rank up to max_rounds_per_path / max_repeated_attempts / max_total_attempts.
    rank = 1
    total_attempts = 0
    repeated_attempts = 0
    while rank <= max_paths and total_attempts < max_total_attempts:
        accepted_this_rank = False
        last_status = ""

        for round_id in range(1, max_rounds_per_path + 1):
            if total_attempts >= max_total_attempts:
                stop_reason = "max_total_attempts"
                break
            total_attempts += 1

            T, parent, status, expanded = _fmm_search(
                n=n,
                neighbor_fn=neighbor_fn,
                active=active,
                blocked=blocked,
                xm=xm,
                ym=ym,
                slow=effective_slowness,
                penalty=penalty,
                start_idx=start_idx,
                end_idx=end_idx,
                max_expanded_nodes=max_expanded_nodes,
                inf_time=inf_time,
            )
            total_expanded += int(expanded)
            last_status = status

            if status != "ok":
                stop_reason = status
                break

            path = _reconstruct_path(parent, start_idx, end_idx)
            if not path:
                stop_reason = "path_reconstruction_failed"
                break

            overlap = _path_overlap_ratio(path, accepted_paths)
            if accepted_paths and overlap > max_overlap:
                repeated_attempts += 1
                # Penalize/block this too-similar path and retry the same rank.
                if tree is None:
                    changed = 0
                else:
                    penalty, blocked, changed = _update_overlap_control(
                        tree=tree,
                        coords=coords,
                        path=path,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        penalty=penalty,
                        blocked=blocked,
                        kwargs=kwargs,
                    )
                if verbose:
                    print(
                        f"      FMM2D rank {rank:03d}, round {round_id}: "
                        f"skip overlap={overlap:.3f}, changed={changed:,}, "
                        f"repeated={repeated_attempts}/{max_repeated_attempts}"
                    )
                if changed <= 0:
                    stop_reason = "too_similar_no_more_buffer_nodes"
                    break
                if repeated_attempts >= max_repeated_attempts:
                    stop_reason = "max_repeated_attempts"
                    break
                continue

            distance_m = _path_distance_m(xm, ym, path)
            travel_time_s = float(T[end_idx])

            item = {
                "rank": int(rank),
                "path_indices": [int(i) for i in path],
                "total_cost": float(travel_time_s),
                "cost": float(travel_time_s),
                "travel_cost": float(travel_time_s),
                "turn_cost": 0.0,
                "turn_count": 0,
                "total_turn_angle_degree": 0.0,
                "nodes": int(len(path)),
                "distance_m": float(distance_m),
                "distance_km": float(distance_m / 1000.0),
                "estimated_traveltime_s": float(travel_time_s),
                "estimated_traveltime_min": float(travel_time_s / 60.0),
                "overlap_ratio": float(overlap),
                "expanded_states": int(expanded),
                "round_id": int(round_id),
                "status": "ok",
            }
            path_results.append(item)
            accepted_paths.append(path)
            repeated_attempts = 0

            # No need to update overlap control after the final requested path.
            if tree is not None and rank < max_paths:
                penalty, blocked, changed = _update_overlap_control(
                    tree=tree,
                    coords=coords,
                    path=path,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    penalty=penalty,
                    blocked=blocked,
                    kwargs=kwargs,
                )
            else:
                changed = 0

            if verbose:
                print(
                    f"      FMM2D path {rank:03d}: nodes={len(path):,}, "
                    f"distance={distance_m/1000.0:.4f} km, "
                    f"time={travel_time_s:.2f} s, overlap={overlap:.3f}, "
                    f"buffer_changed={changed:,}"
                )

            accepted_this_rank = True
            rank += 1
            break

        if not accepted_this_rank:
            if not stop_reason:
                stop_reason = last_status or "no_more_paths"
            break

    if not path_results:
        return {
            "success": False,
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_requested": int(max_paths),
            "k_paths_found": 0,
            "expanded_states": int(total_expanded),
            "message": f"FMM2D failed: {stop_reason or 'no path found'}",
            "isolated_special_blocked": isolated_blocked,
            "is_lonlat": bool(is_lonlat),
        }

    best = path_results[0]
    result = {
        "success": True,
        "algorithm": "FMM2D",
        "path_indices": best["path_indices"],
        "path_results": path_results,
        "ranked_paths": path_results,
        "total_cost": float(best["total_cost"]),
        "travel_cost": float(best["travel_cost"]),
        "turn_cost": 0.0,
        "turn_count": 0,
        "total_turn_angle_degree": 0.0,
        "nodes": int(best["nodes"]),
        "distance_m": float(best["distance_m"]),
        "distance_km": float(best["distance_km"]),
        "estimated_traveltime_s": float(best["estimated_traveltime_s"]),
        "estimated_traveltime_min": float(best["estimated_traveltime_min"]),
        "k_paths_requested": None if auto_until_exhausted else int(max_paths),
        "k_paths_safety_cap": int(max_paths),
        "fmm2d_auto_until_exhausted": bool(auto_until_exhausted),
        "fmm2d_total_attempts": int(total_attempts),
        "fmm2d_max_total_attempts": int(max_total_attempts),
        "k_paths_found": int(len(path_results)),
        "expanded_states": int(total_expanded),
        "message": "ok" if len(path_results) >= max_paths else f"stopped: {stop_reason or 'max paths reached'}",
        "isolated_special_blocked": isolated_blocked,
        "is_lonlat": bool(is_lonlat),
        "fmm2d_previous_path_action": str(previous_path_action),
        "fmm2d_mode": str(fmm2d_mode),
        "fmm2d_fastest_path_mode": False,
    }
    return result
