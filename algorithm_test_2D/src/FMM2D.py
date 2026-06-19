#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/FMM2D.py

FMM/Fast-Marching-style path search module for the LAE-UTM main.py protocol.

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
main.py. It supports selected-pair mode and facility-pair library mode.

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
# Post-path collision avoidance / time deconfliction
# ============================================================



def _collision_mode(kwargs: dict[str, Any]) -> str:
    """Read spatial collision-avoidance mode.

    The old time-offset scheduler has been removed.  Collision avoidance is now
    spatial: make path-offset alternatives first, and use traffic links only
    when strict spatial separation cannot provide all requested paths.
    """
    value = _param(
        kwargs,
        "FMM2D_COLLISION_AVOIDANCE_MODE",
        _param(kwargs, "FMM2D_COLISION_AVOIDANCE_MODE", "none"),
    )
    if isinstance(value, bool):
        return "path_offset" if value else "none"

    text = str(value).strip().lower()
    if text in ("", "none", "off", "false", "0", "no"):
        return "none"

    if text in (
        "on", "true", "1", "yes", "avoid", "collision", "collision_avoidance",
        "path_offset", "path-offset", "spatial", "spatial_offset", "non_overlap",
        "non-overlap", "offset", "traffic_link", "traffic-links",
    ):
        return "path_offset"

    # Backward compatibility: old parameter files may still say time_offset.
    # Treat it as a request for the new spatial path-offset behavior instead
    # of silently scheduling departure delays.
    if text in ("time_offset", "schedule", "delay", "deconflict"):
        return "path_offset"

    return text



def _path_cumulative_times_s(
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    path: list[int],
) -> np.ndarray:
    """Cumulative travel time along one path using distance * slowness."""
    if not path:
        return np.asarray([], dtype=float)
    times = np.zeros(len(path), dtype=float)
    for k in range(1, len(path)):
        i = int(path[k - 1])
        j = int(path[k])
        dist = _distance_m(xm, ym, i, j)
        s = 0.5 * (float(slow[i]) + float(slow[j]))
        dt = dist * s
        if not np.isfinite(dt) or dt < 0.0:
            dt = 0.0
        times[k] = times[k - 1] + float(dt)
    return times

def _path_offset_duration_metadata(
    *,
    path_results: list[dict[str, Any]],
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach lightweight duration metadata without time scheduling.

    This replaces the old time-offset collision layer.  It does not assign
    departure delays.  Spatial deconfliction must happen during path search.
    """
    mode = _collision_mode(kwargs)
    out: list[dict[str, Any]] = []
    for item in path_results or []:
        q = dict(item)
        pth = [int(i) for i in q.get("path_indices", [])]
        q["path_indices"] = pth
        cum_t = _path_cumulative_times_s(xm, ym, slow, pth)
        duration = float(cum_t[-1]) if len(cum_t) else float(q.get("travel_cost", q.get("total_cost", 0.0)))
        if not np.isfinite(duration) or duration < 0.0:
            duration = float(q.get("estimated_traveltime_s", q.get("total_cost", 0.0)))
        q["collision_duration_s"] = float(duration)
        q["collision_avoidance_mode"] = mode
        q["collision_checked"] = bool(mode == "path_offset")
        # Keep old column names harmlessly populated for backward-compatible CSV readers.
        q["collision_free"] = bool(q.get("path_offset_strict", True)) if mode == "path_offset" else "not_checked"
        q["collision_delay_s"] = 0.0
        q["collision_delay_min"] = 0.0
        q["departure_time_s"] = ""
        q["departure_time_min"] = ""
        q["arrival_time_s"] = ""
        q["arrival_time_min"] = ""
        q["schedule_order"] = ""
        q["collision_min_distance_m"] = q.get("path_offset_min_separation_m", "")
        q["collision_blocking_rank"] = ""
        q["collision_blocking_pair_key"] = ""
        q["collision_conflict_time_s"] = ""
        out.append(q)

    return out, {
        "fmm2d_collision_avoidance_mode": mode,
        "fmm2d_collision_checked": bool(mode == "path_offset"),
        "fmm2d_collision_safety_distance_m": float(_param(kwargs, "FMM2D_PATH_OFFSET_BUFFER_M", _param(kwargs, "FMM2D_COLLISION_SAFETY_DISTANCE_M", 200.0))),
        "fmm2d_collision_avoided_count": int(sum(1 for q in out if bool(q.get("path_offset_strict", False)))),
        "fmm2d_collision_unresolved_count": int(sum(1 for q in out if bool(q.get("traffic_link_required", False)))),
    }


# Backward-compatible function name used lower in this module.
def _apply_collision_avoidance_to_path_results(
    *,
    path_results: list[dict[str, Any]],
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _path_offset_duration_metadata(
        path_results=path_results,
        xm=xm,
        ym=ym,
        slow=slow,
        kwargs=kwargs,
    )


# Old time-offset helper names intentionally remain absent from the execution
# path.  Spatial conflict handling is implemented in the path-offset facility
# library below.

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
# Facility-pair spatial path-offset mode
# ============================================================


def _facility_zone_protection_mask(
    *,
    model: pd.DataFrame,
    coords: np.ndarray,
    start_idx: int,
    end_idx: int,
    radius_m: float,
    prefixes: tuple[str, ...],
    low_memory: bool,
) -> np.ndarray:
    """Nodes where path overlap is operationally allowed.

    Overlap is allowed around the route endpoints and around DB/DK/FLZ service
    nodes.  This is the 200 m exception zone requested for takeoff/landing and
    facility access.
    """
    n = len(model)
    mask = np.zeros(n, dtype=bool)
    radius_m = float(radius_m)

    seeds = {int(start_idx), int(end_idx)}
    if prefixes:
        special = _special_mask(model, prefixes)
        seeds.update(int(i) for i in np.flatnonzero(special))

    valid_seeds = np.asarray([i for i in seeds if 0 <= i < n], dtype=int)
    if len(valid_seeds) == 0:
        return mask

    if radius_m <= 0.0:
        mask[valid_seeds] = True
        return mask

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        chunk = 128 if low_memory else 1024
        for pos in range(0, len(valid_seeds), chunk):
            found = tree.query_ball_point(coords[valid_seeds[pos:pos + chunk]], r=radius_m)
            for item in found:
                if item:
                    mask[np.asarray(item, dtype=int)] = True
    except Exception:
        # Conservative fallback: protect only seed nodes.
        mask[valid_seeds] = True

    return mask


def _buffer_mask_for_path_excluding_allowed(
    *,
    tree: Any,
    coords: np.ndarray,
    path: list[int],
    path_buffer_m: float,
    allowed_mask: np.ndarray,
    low_memory: bool,
) -> np.ndarray:
    mask = _nodes_within_path_buffer(
        tree=tree,
        coords=coords,
        path=path,
        radius_m=float(path_buffer_m),
        low_memory=low_memory,
    )
    if allowed_mask is not None and len(allowed_mask) == len(mask):
        mask &= ~np.asarray(allowed_mask, dtype=bool)
    return mask


def _penalty_from_locked_mask(
    *,
    locked_mask: np.ndarray,
    factor: float,
    n: int,
) -> np.ndarray:
    penalty = np.ones(int(n), dtype=float)
    if locked_mask is not None and len(locked_mask) == n:
        penalty[np.asarray(locked_mask, dtype=bool)] = max(1.0, float(factor))
    return penalty


def _connected_components_from_mask(
    *,
    tree: Any,
    coords: np.ndarray,
    mask: np.ndarray,
    radius_m: float,
) -> list[list[int]]:
    """Group nearby overlapped-buffer nodes into a minimal set of traffic links."""
    nodes = [int(i) for i in np.flatnonzero(np.asarray(mask, dtype=bool))]
    if not nodes:
        return []

    node_set = set(nodes)
    seen: set[int] = set()
    comps: list[list[int]] = []
    link_radius = max(float(radius_m) * 2.05, 1.0)

    for seed in nodes:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            try:
                neigh = tree.query_ball_point(coords[int(i)], r=link_radius)
            except Exception:
                neigh = []
            for j in neigh:
                jj = int(j)
                if jj in node_set and jj not in seen:
                    seen.add(jj)
                    stack.append(jj)
        comps.append(sorted(comp))

    comps.sort(key=len, reverse=True)
    return comps



def _distance_to_path_nodes_m(
    *,
    coords: np.ndarray,
    path: list[int],
    low_memory: bool,
) -> np.ndarray:
    """Distance from every model node to the nearest node of a reference path.

    Used to keep a backup lane close to its same-direction main lane while
    still respecting the hard no-overlap/path-offset buffer.
    """
    n = len(coords)
    out = np.full(n, float("inf"), dtype=float)
    if not path:
        return out

    path_idx = np.asarray([int(i) for i in path if 0 <= int(i) < n], dtype=int)
    if len(path_idx) == 0:
        return out

    try:
        from scipy.spatial import cKDTree
        ref_tree = cKDTree(coords[path_idx])
        chunk = 10_000 if low_memory else 100_000
        for start in range(0, n, chunk):
            d, _ = ref_tree.query(coords[start:start + chunk], k=1)
            out[start:start + chunk] = np.asarray(d, dtype=float)
    except Exception:
        # Conservative fallback: compute in chunks against reference nodes.
        chunk = 2_000 if low_memory else 20_000
        ref = coords[path_idx]
        for start in range(0, n, chunk):
            sub = coords[start:start + chunk]
            d2_min = np.full(len(sub), float("inf"), dtype=float)
            for r0 in range(0, len(ref), 512):
                rr = ref[r0:r0 + 512]
                dx = sub[:, None, 0] - rr[None, :, 0]
                dy = sub[:, None, 1] - rr[None, :, 1]
                d2_min = np.minimum(d2_min, np.min(dx * dx + dy * dy, axis=1))
            out[start:start + chunk] = np.sqrt(d2_min)
    return out


def _lane_pair_preference(
    *,
    coords: np.ndarray,
    reference_path: list[int],
    path_buffer_m: float,
    allowed_mask: np.ndarray,
    kwargs: dict[str, Any],
    low_memory: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return penalty and optional hard mask for close/parallel backup lanes.

    The main lane is solved first.  For backup lanes in the same direction,
    this function adds a soft distance penalty so the next path is the nearest
    feasible non-overlapping lane beside the main lane, not a random distant
    alternative.  Forward and backward directions do not share this attraction;
    each direction has its own main/backup lane pair.
    """
    n = len(coords)
    penalty = np.ones(n, dtype=float)
    hard_block = np.zeros(n, dtype=bool)
    dist = np.full(n, float("inf"), dtype=float)

    enabled = _as_bool(_param(kwargs, "FMM2D_LANE_PAIR_CLOSE_PARALLEL_PRIORITY", True))
    if (not enabled) or not reference_path:
        return penalty, dist, hard_block

    dist = _distance_to_path_nodes_m(
        coords=coords,
        path=reference_path,
        low_memory=low_memory,
    )

    preferred_max_m = float(
        _param(
            kwargs,
            "FMM2D_LANE_PAIR_PREFERRED_MAX_DISTANCE_M",
            max(float(path_buffer_m) * 2.5, float(path_buffer_m) + 150.0),
        )
    )
    preferred_max_m = max(preferred_max_m, float(path_buffer_m) + 1.0)

    weight = float(_param(kwargs, "FMM2D_LANE_PAIR_DISTANCE_WEIGHT", 1.5))
    max_penalty = float(_param(kwargs, "FMM2D_LANE_PAIR_MAX_PENALTY_FACTOR", 25.0))
    max_penalty = max(1.0, max_penalty)

    # Nodes just outside the hard no-overlap buffer get almost no extra cost.
    # Nodes farther from the main lane become progressively less attractive.
    excess = np.maximum(0.0, dist - float(path_buffer_m))
    scale = max(preferred_max_m - float(path_buffer_m), 1.0)
    penalty = 1.0 + max(0.0, weight) * (excess / scale)
    penalty = np.minimum(penalty, max_penalty)
    penalty[~np.isfinite(penalty)] = max_penalty

    hard_max_raw = _param(kwargs, "FMM2D_LANE_PAIR_HARD_MAX_DISTANCE_M", None)
    if not _is_none_like(hard_max_raw):
        hard_max_m = float(hard_max_raw)
        if hard_max_m > 0.0:
            hard_block = dist > hard_max_m
            if allowed_mask is not None and len(allowed_mask) == n:
                hard_block &= ~np.asarray(allowed_mask, dtype=bool)

    return penalty, dist, hard_block


def _lane_pair_distance_stats(lane_distance_m: np.ndarray, path: list[int]) -> tuple[float, float]:
    """Mean/max distance from this lane to its same-direction reference lane."""
    if lane_distance_m is None or len(path) == 0:
        return float("nan"), float("nan")
    try:
        vals = lane_distance_m[np.asarray(path, dtype=int)]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.max(vals))
    except Exception:
        return float("nan"), float("nan")

def _make_path_offset_item(
    *,
    rank: int,
    path: list[int],
    travel_time_s: float,
    expanded: int,
    xm: np.ndarray,
    ym: np.ndarray,
    source_idx: int,
    target_idx: int,
    source_label: str,
    target_label: str,
    route_id: str,
    route_pair_type: str,
    direction: str,
    role: str,
    role_index: int,
    strict: bool,
    status: str,
    overlap_nodes: int,
    traffic_link_id: str,
    traffic_link_buffer_m: float,
    conflict_with: str,
    direct_distance_m: float,
    lane_pair_reference_rank: int | None = None,
    lane_pair_reference_pair_key: str = "",
    lane_pair_mean_distance_m: float = float("nan"),
    lane_pair_max_distance_m: float = float("nan"),
) -> dict[str, Any]:
    distance_m = _path_distance_m(xm, ym, path)
    pair_key = f"{source_label}->{target_label} {direction}_{role}"
    return {
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
        "overlap_ratio": 0.0,
        "expanded_states": int(expanded),
        "round_id": int(role_index),
        "status": str(status),
        "source_idx": int(source_idx),
        "target_idx": int(target_idx),
        "source_label": str(source_label),
        "target_label": str(target_label),
        "pair_type": str(route_pair_type),
        "pair_key": pair_key,
        "pair_undirected_key": str(route_id),
        "route_id": str(route_id),
        "direct_distance_m": float(direct_distance_m) if np.isfinite(direct_distance_m) else float("nan"),
        "path_offset_direction": str(direction),
        "path_offset_role": str(role),
        "path_offset_role_index": int(role_index),
        "path_offset_strict": bool(strict),
        "path_offset_status": str(status),
        "path_offset_overlap_nodes": int(overlap_nodes),
        "path_offset_conflict_with": str(conflict_with),
        "lane_pair_reference_rank": int(lane_pair_reference_rank) if lane_pair_reference_rank is not None else "",
        "lane_pair_reference_pair_key": str(lane_pair_reference_pair_key),
        "lane_pair_mean_distance_m": float(lane_pair_mean_distance_m) if np.isfinite(lane_pair_mean_distance_m) else "",
        "lane_pair_max_distance_m": float(lane_pair_max_distance_m) if np.isfinite(lane_pair_max_distance_m) else "",
        "lane_pair_close_parallel_priority": bool(lane_pair_reference_pair_key),
        "traffic_link_required": bool(traffic_link_id),
        "traffic_link_id": str(traffic_link_id),
        "traffic_link_buffer_m": float(traffic_link_buffer_m) if traffic_link_id else 0.0,
        "traffic_link_node_count": int(overlap_nodes) if traffic_link_id else 0,
        "collision_avoidance_mode": "path_offset",
        "collision_checked": True,
        "collision_free": bool(strict),
        "collision_delay_s": 0.0,
        "collision_delay_min": 0.0,
        "collision_duration_s": float(travel_time_s),
    }


def _route_id_from_labels(a: str, b: str, pair_type: str) -> str:
    aa = _normalize_label_text(a) or str(a)
    bb = _normalize_label_text(b) or str(b)
    x, y = sorted((aa, bb))
    return f"{str(pair_type).upper()}--{x}--{y}"


def _run_facility_pair_path_offset_library(
    model: pd.DataFrame,
    graph: dict[str, Any],
    start_idx: int,
    end_idx: int,
    **kwargs,
) -> dict[str, Any]:
    """Generate 2 forward + 2 backward spatially separated paths per route.

    Search order per base route:
        A->B main, A->B backup, B->A main, B->A backup.

    A corridor buffer is hard-locked after each strict path.  The lock is not
    applied inside the allowed service-zone buffer around DB/DK/FLZ and the two
    current endpoints.  If strict separation fails, a fallback path is searched
    with a high penalty through the locked corridor and is tagged as a traffic
    link.  The traffic-link geometry is represented by the overlapped buffer
    nodes and grouped into the smallest connected set practical for the route.
    """
    n = len(model)
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    db_prefixes = tuple(_param(kwargs, "FMM2D_PAIR_DB_PREFIXES", ("DB",)))
    dk_prefixes = tuple(_param(kwargs, "FMM2D_PAIR_DK_PREFIXES", ("DK",)))
    include_db_dk = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DB_DK", True))
    include_db_db = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DB_DB", True))
    include_dk_dk = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DK_DK", True))

    forward_count = max(1, int(_param(kwargs, "FMM2D_PATH_OFFSET_FORWARD_PATHS", 2)))
    backward_count = max(1, int(_param(kwargs, "FMM2D_PATH_OFFSET_BACKWARD_PATHS", 2)))
    path_buffer_m = float(_param(kwargs, "FMM2D_PATH_OFFSET_BUFFER_M", _param(kwargs, "FMM2D_COLLISION_SAFETY_DISTANCE_M", 200.0)))
    allowed_buffer_m = float(_param(kwargs, "FMM2D_PATH_OFFSET_ALLOWED_BUFFER_M", _param(kwargs, "MULTI_PATH_NON_OVERLAP_BUFFER_RADIUS_M", 200.0)))
    allowed_prefixes = tuple(_param(kwargs, "FMM2D_PATH_OFFSET_ALLOWED_PREFIXES", _param(kwargs, "MULTI_PATH_NON_OVERLAP_ALLOWED_PREFIXES", ("DB", "DK", "FLZ"))))
    traffic_link_buffer_m = float(_param(kwargs, "FMM2D_TRAFFIC_LINK_BUFFER_M", path_buffer_m))
    traffic_penalty_factor = float(_param(kwargs, "FMM2D_TRAFFIC_LINK_PENALTY_FACTOR", 50.0))
    lane_pair_priority = _as_bool(_param(kwargs, "FMM2D_LANE_PAIR_CLOSE_PARALLEL_PRIORITY", True))
    lane_pair_preferred_max_m = float(_param(kwargs, "FMM2D_LANE_PAIR_PREFERRED_MAX_DISTANCE_M", max(path_buffer_m * 2.5, path_buffer_m + 150.0)))
    lane_pair_distance_weight = float(_param(kwargs, "FMM2D_LANE_PAIR_DISTANCE_WEIGHT", 1.5))
    strict_before_fallback = _as_bool(_param(kwargs, "FMM2D_PATH_OFFSET_STRICT_BEFORE_TRAFFIC_LINK", True))
    max_pair_results_raw = _param(kwargs, "FMM2D_PAIR_MAX_RESULTS", None)
    max_pair_results = None if _is_none_like(max_pair_results_raw) else int(max_pair_results_raw)

    pair_min_distance_m = float(_param(kwargs, "FMM2D_PAIR_MIN_DISTANCE_M", 0.0))
    pair_skip_same_label = _as_bool(_param(kwargs, "FMM2D_PAIR_SKIP_SAME_LABEL", True))
    pair_skip_same_coord = _as_bool(_param(kwargs, "FMM2D_PAIR_SKIP_SAME_COORD", True))
    pair_same_coord_tolerance_m = float(_param(kwargs, "FMM2D_PAIR_SAME_COORD_TOLERANCE_M", 1.0))

    inf_time = float(_param(kwargs, "FMM2D_INF_TIME", 1.0e30))
    max_expanded_nodes = _param(kwargs, "FMM2D_PAIR_MAX_EXPANDED_NODES", _param(kwargs, "FMM2D_MAX_EXPANDED_NODES", kwargs.get("max_expansions", None)))
    if max_expanded_nodes is not None:
        max_expanded_nodes = int(max_expanded_nodes)
    connectivity = int(_param(kwargs, "CONNECTIVITY_2D", graph.get("connectivity", 8)))
    low_memory = _as_bool(_param(kwargs, "FMM2D_LOW_MEMORY_MODE", _param(kwargs, "LOW_MEMORY_MODE", True)))
    verbose = _as_bool(_param(kwargs, "FMM2D_VERBOSE", kwargs.get("verbose", True)))

    xm, ym, is_lonlat = _xy_to_metric(model)
    coords = np.column_stack([xm, ym]).copy()
    active = _valid_mask_from_graph(model, graph)

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

    effective_slowness = _make_effective_slowness(
        model=model,
        neighbor_fn=neighbor_fn,
        active=active,
        start_idx=start_idx,
        end_idx=end_idx,
        kwargs=kwargs,
    )

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
    except Exception as exc:
        raise RuntimeError("FMM2D path_offset mode requires scipy.spatial.cKDTree.") from exc

    db_nodes_all = _facility_indices(model, db_prefixes)
    dk_nodes_all = _facility_indices(model, dk_prefixes)
    db_nodes = [i for i in db_nodes_all if bool(active[int(i)])]
    dk_nodes = [i for i in dk_nodes_all if bool(active[int(i)])]

    # Build one canonical direction per route. Reverse-direction paths are
    # generated explicitly below, so two-way de-duplication stays enabled here.
    base_pairs = _build_facility_pairs(
        model=model,
        db_indices=db_nodes,
        dk_indices=dk_nodes,
        include_db_dk=include_db_dk,
        include_db_db=include_db_db,
        include_dk_dk=include_dk_dk,
        include_reverse=False,
        min_distance_m=pair_min_distance_m,
        skip_same_label=pair_skip_same_label,
        skip_same_coord=pair_skip_same_coord,
        same_coord_tolerance_m=pair_same_coord_tolerance_m,
        dedup_two_way=True,
    )

    if verbose:
        print("      FMM2D PATH-OFFSET FACILITY MODE:")
        print(f"        active DB nodes      : {len(db_nodes):,} / {len(db_nodes_all):,}")
        print(f"        active DK nodes      : {len(dk_nodes):,} / {len(dk_nodes_all):,}")
        print(f"        base routes          : {len(base_pairs):,}")
        print(f"        requested per route  : {forward_count} forward + {backward_count} backward")
        print(f"        path buffer          : {path_buffer_m:.2f} m")
        print(f"        DB/DK/FLZ buffer     : {allowed_buffer_m:.2f} m")
        print(f"        allowed prefixes     : {allowed_prefixes}")
        print(f"        fallback             : traffic_link, penalty={traffic_penalty_factor:g}")
        print(f"        lane pair priority   : {lane_pair_priority}")
        if lane_pair_priority:
            print(f"        lane preferred max   : {lane_pair_preferred_max_m:.2f} m")
            print(f"        lane distance weight : {lane_pair_distance_weight:g}")

    if not base_pairs:
        return {
            "success": False,
            "algorithm": "FMM2D",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_found": 0,
            "expanded_states": 0,
            "message": "FMM2D path_offset mode found no active DB/DK routes.",
            "isolated_special_blocked": isolated_blocked,
            "fmm2d_pair_mode": True,
            "fmm2d_collision_avoidance_mode": "path_offset",
        }

    path_results: list[dict[str, Any]] = []
    missing_paths: list[dict[str, Any]] = []
    total_expanded = 0
    traffic_link_counter = 0
    traffic_link_routes: set[str] = set()

    rank = 1
    for route_pos, base in enumerate(base_pairs, start=1):
        a = int(base["source_idx"])
        b = int(base["target_idx"])
        a_label = str(base["source_label"])
        b_label = str(base["target_label"])
        route_pair_type = str(base["pair_type"])
        route_id = _route_id_from_labels(a_label, b_label, route_pair_type)
        direct_distance_m = float(base.get("direct_distance_m", float("nan")))

        allowed_mask = _facility_zone_protection_mask(
            model=model,
            coords=coords,
            start_idx=a,
            end_idx=b,
            radius_m=allowed_buffer_m,
            prefixes=allowed_prefixes,
            low_memory=low_memory,
        )

        locked_mask = np.zeros(n, dtype=bool)
        accepted_for_route: list[dict[str, Any]] = []
        # For each direction, backup lanes are attracted toward their own main lane.
        # Forward and backward directions intentionally do not share a closeness target.
        same_direction_main_path: dict[str, list[int]] = {}
        same_direction_main_rank: dict[str, int] = {}
        same_direction_main_pair_key: dict[str, str] = {}

        role_specs: list[tuple[int, int, str, str, int]] = []
        for k in range(1, forward_count + 1):
            role_specs.append((a, b, "forward", "main" if k == 1 else "backup", k))
        for k in range(1, backward_count + 1):
            role_specs.append((b, a, "backward", "main" if k == 1 else "backup", k))

        if verbose:
            print(f"        route {route_pos:03d}: {a_label} <-> {b_label} | {len(role_specs)} paths")

        for src, dst, direction, role, role_index in role_specs:
            src_label = _node_label(model, src)
            dst_label = _node_label(model, dst)

            # Backup lanes should stay close/parallel to the main lane of the SAME direction.
            # This is a soft preference, so the search can still find a solution if terrain/no-fly
            # cells make a nearby lane impossible.
            reference_path = [] if role_index == 1 else same_direction_main_path.get(direction, [])
            reference_rank = same_direction_main_rank.get(direction, None) if reference_path else None
            reference_pair_key = same_direction_main_pair_key.get(direction, "") if reference_path else ""
            lane_penalty, lane_distance_m, lane_hard_block = _lane_pair_preference(
                coords=coords,
                reference_path=reference_path,
                path_buffer_m=path_buffer_m,
                allowed_mask=allowed_mask,
                kwargs=kwargs,
                low_memory=low_memory,
            )

            search_attempts = []
            if strict_before_fallback:
                strict_blocked = locked_mask.copy() | lane_hard_block
                strict_penalty = lane_penalty.copy()
                search_attempts.append(("strict", strict_blocked, strict_penalty))

            traffic_blocked = lane_hard_block.copy() if _as_bool(_param(kwargs, "FMM2D_LANE_PAIR_HARD_LIMIT_FOR_TRAFFIC_LINK", False)) else np.zeros(n, dtype=bool)
            traffic_penalty = _penalty_from_locked_mask(
                locked_mask=locked_mask,
                factor=traffic_penalty_factor,
                n=n,
            ) * lane_penalty
            search_attempts.append(("traffic_link", traffic_blocked, traffic_penalty))

            chosen = None
            chosen_mode = ""
            chosen_overlap_mask = np.zeros(n, dtype=bool)

            for attempt_mode, attempt_blocked, attempt_penalty in search_attempts:
                # Start/end of the current direction must never be blocked by a previous corridor.
                attempt_blocked[int(src)] = False
                attempt_blocked[int(dst)] = False

                T, parent, status, expanded = _fmm_search(
                    n=n,
                    neighbor_fn=neighbor_fn,
                    active=active,
                    blocked=attempt_blocked,
                    xm=xm,
                    ym=ym,
                    slow=effective_slowness,
                    penalty=attempt_penalty,
                    start_idx=src,
                    end_idx=dst,
                    max_expanded_nodes=max_expanded_nodes,
                    inf_time=inf_time,
                )
                total_expanded += int(expanded)

                if status != "ok":
                    continue
                path = _reconstruct_path(parent, src, dst)
                if not path:
                    continue

                path_mask = _buffer_mask_for_path_excluding_allowed(
                    tree=tree,
                    coords=coords,
                    path=path,
                    path_buffer_m=path_buffer_m,
                    allowed_mask=allowed_mask,
                    low_memory=low_memory,
                )
                overlap_mask = path_mask & locked_mask

                if attempt_mode == "strict" and np.any(overlap_mask):
                    # Should be rare because locked nodes are blocked, but keep a guard.
                    continue

                chosen = (path, float(T[int(dst)]), int(expanded), path_mask, status)
                chosen_mode = attempt_mode
                chosen_overlap_mask = overlap_mask
                break

            if chosen is None:
                missing_paths.append({
                    "route_id": route_id,
                    "source_label": src_label,
                    "target_label": dst_label,
                    "direction": direction,
                    "role": role,
                    "role_index": role_index,
                    "status": "missing_no_strict_or_traffic_link_path",
                })
                if verbose:
                    print(f"          missing {direction}_{role}: {src_label}->{dst_label}")
                continue

            path, travel_time_s, expanded, path_mask, status = chosen
            strict = chosen_mode == "strict"
            overlap_nodes = int(np.count_nonzero(chosen_overlap_mask))
            lane_mean_distance_m, lane_max_distance_m = _lane_pair_distance_stats(lane_distance_m, path)
            traffic_link_id = ""
            conflict_with = ""

            if not strict:
                # Minimize traffic links by grouping the fallback overlap into
                # connected components, then using one route-level link ID for
                # all components belonging to this route.  This prevents a very
                # large number of small link IDs from being created.
                comps = _connected_components_from_mask(
                    tree=tree,
                    coords=coords,
                    mask=chosen_overlap_mask,
                    radius_m=traffic_link_buffer_m,
                )
                if comps or overlap_nodes > 0:
                    if route_id not in traffic_link_routes:
                        traffic_link_counter += 1
                        traffic_link_routes.add(route_id)
                    traffic_link_id = f"TL{traffic_link_counter:03d}"
                    conflict_keys = []
                    path_node_set = set(path)
                    for prev in accepted_for_route:
                        prev_nodes = set(prev.get("path_indices", []))
                        if path_node_set & prev_nodes:
                            conflict_keys.append(str(prev.get("pair_key", prev.get("path_offset_role", ""))))
                    conflict_with = ";".join(conflict_keys)

            item = _make_path_offset_item(
                rank=rank,
                path=path,
                travel_time_s=travel_time_s,
                expanded=expanded,
                xm=xm,
                ym=ym,
                source_idx=src,
                target_idx=dst,
                source_label=src_label,
                target_label=dst_label,
                route_id=route_id,
                route_pair_type=route_pair_type,
                direction=direction,
                role=role,
                role_index=role_index,
                strict=strict,
                status="ok_strict" if strict else "fallback_traffic_link",
                overlap_nodes=overlap_nodes,
                traffic_link_id=traffic_link_id,
                traffic_link_buffer_m=traffic_link_buffer_m,
                conflict_with=conflict_with,
                direct_distance_m=direct_distance_m,
                lane_pair_reference_rank=reference_rank,
                lane_pair_reference_pair_key=reference_pair_key,
                lane_pair_mean_distance_m=lane_mean_distance_m,
                lane_pair_max_distance_m=lane_max_distance_m,
            )
            path_results.append(item)
            accepted_for_route.append(item)
            if role_index == 1:
                same_direction_main_path[direction] = [int(v) for v in path]
                same_direction_main_rank[direction] = int(item["rank"])
                same_direction_main_pair_key[direction] = str(item.get("pair_key", ""))
            rank += 1

            # Lock the non-allowed corridor of every accepted path.  Even a
            # fallback traffic-link path contributes its free corridor to the
            # lock; future fallback searches may still pass through locked areas
            # only with penalty and will be marked as traffic-link use.
            locked_mask |= path_mask

            if verbose:
                print(
                    f"          {direction}_{role:<6s}: {src_label}->{dst_label} | "
                    f"{len(path):,} nodes | {travel_time_s:.2f} s | "
                    f"strict={strict} | overlap_nodes={overlap_nodes:,} | link={traffic_link_id or '-'}"
                    + (f" | lane_mean={lane_mean_distance_m:.1f} m" if np.isfinite(lane_mean_distance_m) else "")
                )

            if max_pair_results is not None and len(path_results) >= max_pair_results:
                break
        if max_pair_results is not None and len(path_results) >= max_pair_results:
            break

    if not path_results:
        return {
            "success": False,
            "algorithm": "FMM2D",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_found": 0,
            "expanded_states": int(total_expanded),
            "message": "FMM2D path_offset mode found no reachable route paths.",
            "isolated_special_blocked": isolated_blocked,
            "is_lonlat": bool(is_lonlat),
            "fmm2d_pair_mode": True,
            "fmm2d_collision_avoidance_mode": "path_offset",
            "fmm2d_path_offset_missing_paths": int(len(missing_paths)),
        }

    path_results, collision_summary = _path_offset_duration_metadata(
        path_results=path_results,
        xm=xm,
        ym=ym,
        slow=effective_slowness,
        kwargs={**kwargs, "FMM2D_COLLISION_AVOIDANCE_MODE": "path_offset"},
    )

    best = path_results[0]
    expected_paths = len(base_pairs) * (forward_count + backward_count)
    strict_count = int(sum(1 for item in path_results if bool(item.get("path_offset_strict", False))))
    traffic_link_count = len({str(item.get("traffic_link_id", "")) for item in path_results if str(item.get("traffic_link_id", ""))})

    if verbose:
        print("      FMM2D path-offset result:")
        print(f"        returned paths       : {len(path_results):,} / expected {expected_paths:,}")
        print(f"        strict paths         : {strict_count:,}")
        print(f"        traffic links        : {traffic_link_count:,}")
        print(f"        missing paths        : {len(missing_paths):,}")

    return {
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
        "k_paths_requested": int(expected_paths),
        "k_paths_found": int(len(path_results)),
        "expanded_states": int(total_expanded),
        "message": "ok" if len(path_results) == expected_paths else f"partial: missing {len(missing_paths)} requested paths",
        "isolated_special_blocked": isolated_blocked,
        "is_lonlat": bool(is_lonlat),
        "fmm2d_pair_mode": True,
        "fmm2d_pair_return_mode": "all",
        "fmm2d_pair_strategy": "path_offset_four_paths_per_route",
        "fmm2d_collision_avoidance_mode": "path_offset",
        "fmm2d_path_offset_forward_paths": int(forward_count),
        "fmm2d_path_offset_backward_paths": int(backward_count),
        "fmm2d_path_offset_buffer_m": float(path_buffer_m),
        "fmm2d_path_offset_allowed_buffer_m": float(allowed_buffer_m),
        "fmm2d_path_offset_allowed_prefixes": ",".join(str(v) for v in allowed_prefixes),
        "fmm2d_lane_pair_close_parallel_priority": bool(lane_pair_priority),
        "fmm2d_lane_pair_preferred_max_distance_m": float(lane_pair_preferred_max_m),
        "fmm2d_lane_pair_distance_weight": float(lane_pair_distance_weight),
        "fmm2d_path_offset_expected_paths": int(expected_paths),
        "fmm2d_path_offset_strict_paths": int(strict_count),
        "fmm2d_path_offset_traffic_link_paths": int(sum(1 for item in path_results if bool(item.get("traffic_link_required", False)))),
        "fmm2d_path_offset_traffic_link_count": int(traffic_link_count),
        "fmm2d_path_offset_missing_paths": int(len(missing_paths)),
        "fmm2d_path_offset_missing_details": missing_paths,
        **collision_summary,
        "best_pair_key": str(best.get("pair_key", "")),
        "best_pair_type": str(best.get("pair_type", "")),
        "best_source_label": str(best.get("source_label", "")),
        "best_target_label": str(best.get("target_label", "")),
    }


# ============================================================
# Facility-pair library mode
# ============================================================


def _normalize_label_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("none", "null", ""):
        return ""
    return text.upper()


def _node_label(model: pd.DataFrame, idx: int) -> str:
    labels = _label_array(model)
    if 0 <= int(idx) < len(labels):
        return str(labels[int(idx)])
    return f"node{idx}"


def _facility_indices(model: pd.DataFrame, prefixes: tuple[str, ...]) -> list[int]:
    """Return model indices whose label or label_prefix starts with one of prefixes."""
    prefixes = tuple(str(p).upper() for p in prefixes if str(p))
    if not prefixes:
        return []

    labels = _label_array(model).astype(str)
    label_prefixes = _label_prefix_array(model).astype(str)
    mask = np.zeros(len(model), dtype=bool)
    for p in prefixes:
        mask |= np.char.startswith(np.char.upper(labels), p)
        mask |= np.char.startswith(np.char.upper(label_prefixes), p)

    indices = [int(i) for i in np.flatnonzero(mask)]
    indices.sort(key=lambda i: (_node_label(model, i), i))
    return indices


def _build_facility_pairs(
    model: pd.DataFrame,
    db_indices: list[int],
    dk_indices: list[int],
    include_db_dk: bool,
    include_db_db: bool,
    include_dk_dk: bool,
    include_reverse: bool,
    *,
    min_distance_m: float = 0.0,
    skip_same_label: bool = True,
    skip_same_coord: bool = True,
    same_coord_tolerance_m: float = 1.0,
    dedup_two_way: bool = True,
) -> list[dict[str, Any]]:
    """Build requested facility source-target pair definitions.

    The facility-library mode should never create a path from a node/facility
    to itself. This also catches duplicate labels such as DK04 -> DK04 when the
    model contains more than one row with the same operational label.
    """
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    seen_two_way: set[tuple[str, str, str]] = set()

    try:
        xm_pair, ym_pair, _ = _xy_to_metric(model)
    except Exception:
        xm_pair = ym_pair = None

    def too_close_or_same(src: int, dst: int) -> tuple[bool, float]:
        src = int(src)
        dst = int(dst)
        if src == dst:
            return True, 0.0

        src_label_norm = _normalize_label_text(_node_label(model, src))
        dst_label_norm = _normalize_label_text(_node_label(model, dst))
        if skip_same_label and src_label_norm and src_label_norm == dst_label_norm:
            return True, 0.0

        dist_m = float("nan")
        if xm_pair is not None and ym_pair is not None:
            try:
                dist_m = float(math.hypot(float(xm_pair[dst] - xm_pair[src]), float(ym_pair[dst] - ym_pair[src])))
            except Exception:
                dist_m = float("nan")

        if np.isfinite(dist_m):
            if skip_same_coord and dist_m <= float(same_coord_tolerance_m):
                return True, dist_m
            if float(min_distance_m) > 0.0 and dist_m < float(min_distance_m):
                return True, dist_m

        return False, dist_m

    def two_way_key(src: int, dst: int, pair_type: str) -> tuple[str, str, str]:
        """Canonical key used to drop duplicate reverse-direction paths.

        Example: DB01->DK01 and DK01->DB01 share the same key and only
        the first one is kept.  Labels are used when available so duplicate
        facility rows are also de-duplicated cleanly.
        """
        src_label = _normalize_label_text(_node_label(model, int(src))) or str(int(src))
        dst_label = _normalize_label_text(_node_label(model, int(dst))) or str(int(dst))

        # Treat DB_DK and DK_DB as the same operational pair family.
        pt = str(pair_type).upper()
        if pt in ("DB_DK", "DK_DB"):
            family = "DB_DK"
        else:
            family = pt

        a, b = sorted((src_label, dst_label))
        return (family, a, b)

    def add(src: int, dst: int, pair_type: str) -> None:
        src = int(src)
        dst = int(dst)

        skip, direct_distance_m = too_close_or_same(src, dst)
        if skip:
            return

        if dedup_two_way:
            ukey = two_way_key(src, dst, pair_type)
            if ukey in seen_two_way:
                return
            seen_two_way.add(ukey)

        key = (src, dst, pair_type)
        if key in seen:
            return
        seen.add(key)

        src_label = _node_label(model, src)
        dst_label = _node_label(model, dst)
        pairs.append({
            "source_idx": src,
            "target_idx": dst,
            "source_label": src_label,
            "target_label": dst_label,
            "pair_type": pair_type,
            "pair_key": f"{src_label}->{dst_label}",
            "pair_undirected_key": "--".join(two_way_key(src, dst, pair_type)),
            "direct_distance_m": float(direct_distance_m) if np.isfinite(direct_distance_m) else float("nan"),
        })

    if include_db_dk:
        for db in db_indices:
            for dk in dk_indices:
                add(db, dk, "DB_DK")
                if include_reverse:
                    add(dk, db, "DK_DB")

    if include_db_db:
        for a_pos, a in enumerate(db_indices):
            for b in db_indices[a_pos + 1:]:
                add(a, b, "DB_DB")
                if include_reverse:
                    add(b, a, "DB_DB")

    if include_dk_dk:
        for a_pos, a in enumerate(dk_indices):
            for b in dk_indices[a_pos + 1:]:
                add(a, b, "DK_DK")
                if include_reverse:
                    add(b, a, "DK_DK")

    return pairs

def _fmm_search_from_source_to_targets(
    *,
    n: int,
    neighbor_fn: Callable[[int], list[int]],
    active: np.ndarray,
    blocked: np.ndarray,
    xm: np.ndarray,
    ym: np.ndarray,
    slow: np.ndarray,
    penalty: np.ndarray,
    source_idx: int,
    target_indices: Iterable[int],
    max_expanded_nodes: int | None,
    inf_time: float,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """One-source FMM/Dijkstra propagation stopped after all targets are accepted.

    This is the efficient facility-pair mode: one propagation from DB01 can
    extract DB01->DK01, DB01->DK02, DB01->DB02, etc., instead of rerunning for
    every source-target pair.
    """
    source_idx = int(source_idx)
    targets = {int(t) for t in target_indices if 0 <= int(t) < n and int(t) != source_idx}
    usable = active.copy() & (~blocked)

    T = np.full(n, float(inf_time), dtype=float)
    parent = np.full(n, -1, dtype=np.int64)

    if source_idx < 0 or source_idx >= n or not bool(usable[source_idx]):
        return T, parent, "source_blocked", 0

    targets = {t for t in targets if bool(usable[t])}
    if not targets:
        return T, parent, "no_active_targets", 0

    accepted = np.zeros(n, dtype=bool)
    T[source_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, source_idx)]
    remaining = set(targets)
    expanded = 0

    while heap:
        t_i, i = heapq.heappop(heap)
        if accepted[i]:
            continue
        if t_i > T[i]:
            continue

        accepted[i] = True
        expanded += 1

        if i in remaining:
            remaining.remove(i)
            if not remaining:
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

    return T, parent, "partial" if len(remaining) < len(targets) else "unreachable", expanded


def _select_requested_pair_results(
    model: pd.DataFrame,
    pair_results: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter all pair results to the requested START_LABEL -> END_LABEL pair."""
    req_start_label = _normalize_label_text(_param(kwargs, "START_LABEL", None))
    req_end_label = _normalize_label_text(_param(kwargs, "END_LABEL", None))

    if req_start_label and req_end_label:
        selected = [
            item for item in pair_results
            if _normalize_label_text(item.get("source_label")) == req_start_label
            and _normalize_label_text(item.get("target_label")) == req_end_label
        ]
        if selected:
            return selected

    return [
        item for item in pair_results
        if int(item.get("source_idx", -1)) == int(start_idx)
        and int(item.get("target_idx", -1)) == int(end_idx)
    ]


def _run_facility_pair_library(
    model: pd.DataFrame,
    graph: dict[str, Any],
    start_idx: int,
    end_idx: int,
    **kwargs,
) -> dict[str, Any]:
    """Compute fastest paths for all DB/DK facility pairs in one-source batches.

    Returned paths are normal `path_results`, so existing main.py can export
    them as ranked path CSVs. This mode computes one fastest path per pair;
    alternative paths for a single selected pair are still handled by normal
    FMM2D multiple mode.
    """
    n = len(model)
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    db_prefixes = tuple(_param(kwargs, "FMM2D_PAIR_DB_PREFIXES", ("DB",)))
    dk_prefixes = tuple(_param(kwargs, "FMM2D_PAIR_DK_PREFIXES", ("DK",)))

    include_db_dk = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DB_DK", True))
    include_db_db = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DB_DB", True))
    include_dk_dk = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_DK_DK", True))
    include_reverse = _as_bool(_param(kwargs, "FMM2D_PAIR_INCLUDE_REVERSE", False))

    return_mode = str(_param(kwargs, "FMM2D_PAIR_RETURN_MODE", "requested")).strip().lower()
    if return_mode in ("auto", "default"):
        s_label = _normalize_label_text(_param(kwargs, "START_LABEL", None))
        e_label = _normalize_label_text(_param(kwargs, "END_LABEL", None))
        return_mode = "requested" if (s_label and e_label) else "all"

    max_pair_results_raw = _param(kwargs, "FMM2D_PAIR_MAX_RESULTS", None)
    max_pair_results = None if _is_none_like(max_pair_results_raw) else int(max_pair_results_raw)

    inf_time = float(_param(kwargs, "FMM2D_INF_TIME", 1.0e30))
    max_expanded_nodes = _param(kwargs, "FMM2D_PAIR_MAX_EXPANDED_NODES", _param(kwargs, "FMM2D_MAX_EXPANDED_NODES", kwargs.get("max_expansions", None)))
    if max_expanded_nodes is not None:
        max_expanded_nodes = int(max_expanded_nodes)
    connectivity = int(_param(kwargs, "CONNECTIVITY_2D", graph.get("connectivity", 8)))
    low_memory = _as_bool(_param(kwargs, "FMM2D_LOW_MEMORY_MODE", _param(kwargs, "LOW_MEMORY_MODE", True)))
    verbose = _as_bool(_param(kwargs, "FMM2D_VERBOSE", kwargs.get("verbose", True)))

    xm, ym, is_lonlat = _xy_to_metric(model)
    active = _valid_mask_from_graph(model, graph)

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

    effective_slowness = _make_effective_slowness(
        model=model,
        neighbor_fn=neighbor_fn,
        active=active,
        start_idx=start_idx,
        end_idx=end_idx,
        kwargs=kwargs,
    )

    db_nodes_all = _facility_indices(model, db_prefixes)
    dk_nodes_all = _facility_indices(model, dk_prefixes)
    db_nodes = [i for i in db_nodes_all if bool(active[int(i)])]
    dk_nodes = [i for i in dk_nodes_all if bool(active[int(i)])]

    pair_min_distance_m = float(_param(kwargs, "FMM2D_PAIR_MIN_DISTANCE_M", 0.0))
    pair_skip_same_label = _as_bool(_param(kwargs, "FMM2D_PAIR_SKIP_SAME_LABEL", True))
    pair_skip_same_coord = _as_bool(_param(kwargs, "FMM2D_PAIR_SKIP_SAME_COORD", True))
    pair_same_coord_tolerance_m = float(_param(kwargs, "FMM2D_PAIR_SAME_COORD_TOLERANCE_M", 1.0))
    pair_dedup_two_way = _as_bool(_param(kwargs, "FMM2D_PAIR_DEDUP_TWO_WAY", True))

    pair_defs = _build_facility_pairs(
        model=model,
        db_indices=db_nodes,
        dk_indices=dk_nodes,
        include_db_dk=include_db_dk,
        include_db_db=include_db_db,
        include_dk_dk=include_dk_dk,
        include_reverse=include_reverse,
        min_distance_m=pair_min_distance_m,
        skip_same_label=pair_skip_same_label,
        skip_same_coord=pair_skip_same_coord,
        same_coord_tolerance_m=pair_same_coord_tolerance_m,
        dedup_two_way=pair_dedup_two_way,
    )

    if verbose:
        print("      FMM2D FACILITY-PAIR LIBRARY MODE:")
        print(f"        DB prefixes       : {db_prefixes}")
        print(f"        DK prefixes       : {dk_prefixes}")
        print(f"        active DB nodes   : {len(db_nodes):,} / {len(db_nodes_all):,}")
        print(f"        active DK nodes   : {len(dk_nodes):,} / {len(dk_nodes_all):,}")
        print(f"        candidate pairs   : {len(pair_defs):,}")
        print(f"        return mode       : {return_mode}")
        print(f"        skip same label   : {pair_skip_same_label}")
        print(f"        skip same coord   : {pair_skip_same_coord} <= {pair_same_coord_tolerance_m:.3f} m")
        print(f"        de-dup 2-way     : {pair_dedup_two_way}")
        print(f"        min pair distance : {pair_min_distance_m:.3f} m")
        print(f"        low memory        : {low_memory}")
        print("        strategy          : one FMM propagation per unique source, then extract all targets")

    if not pair_defs:
        return {
            "success": False,
            "algorithm": "FMM2D",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_found": 0,
            "expanded_states": 0,
            "message": "FMM2D facility-pair mode found no active DB/DK pairs.",
            "isolated_special_blocked": isolated_blocked,
            "fmm2d_pair_mode": True,
        }

    targets_by_source: dict[int, list[int]] = {}
    pair_lookup: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for pair in pair_defs:
        src = int(pair["source_idx"])
        dst = int(pair["target_idx"])
        targets_by_source.setdefault(src, []).append(dst)
        pair_lookup.setdefault((src, dst), []).append(pair)

    penalty = np.ones(n, dtype=float)
    blocked = np.zeros(n, dtype=bool)
    total_expanded = 0
    path_results: list[dict[str, Any]] = []
    unreachable_pairs: list[dict[str, Any]] = []

    for source_rank, (src, targets) in enumerate(sorted(targets_by_source.items(), key=lambda kv: (_node_label(model, kv[0]), kv[0])), start=1):
        unique_targets = sorted(set(int(t) for t in targets), key=lambda i: (_node_label(model, i), i))
        if verbose:
            print(f"        source {source_rank:03d}: {_node_label(model, src)} -> {len(unique_targets):,} targets")

        T, parent, status, expanded = _fmm_search_from_source_to_targets(
            n=n,
            neighbor_fn=neighbor_fn,
            active=active,
            blocked=blocked,
            xm=xm,
            ym=ym,
            slow=effective_slowness,
            penalty=penalty,
            source_idx=src,
            target_indices=unique_targets,
            max_expanded_nodes=max_expanded_nodes,
            inf_time=inf_time,
        )
        total_expanded += int(expanded)

        for dst in unique_targets:
            pair_items = pair_lookup.get((src, dst), [])
            if not pair_items:
                continue
            if not np.isfinite(T[int(dst)]) or T[int(dst)] >= inf_time or parent[int(dst)] < 0:
                for pair in pair_items:
                    unreachable_pairs.append(dict(pair, status="unreachable", source_status=status))
                continue

            path = _reconstruct_path(parent, src, dst)
            if not path:
                for pair in pair_items:
                    unreachable_pairs.append(dict(pair, status="path_reconstruction_failed", source_status=status))
                continue

            distance_m = _path_distance_m(xm, ym, path)
            travel_time_s = float(T[int(dst)])
            for pair in pair_items:
                item = {
                    "rank": 0,
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
                    "source_idx": int(src),
                    "target_idx": int(dst),
                    "source_label": pair["source_label"],
                    "target_label": pair["target_label"],
                    "pair_type": pair["pair_type"],
                    "pair_key": pair["pair_key"],
                    "pair_undirected_key": str(pair.get("pair_undirected_key", "")),
                    "direct_distance_m": float(pair.get("direct_distance_m", float("nan"))),
                    "source_fmm_status": status,
                }
                path_results.append(item)

    path_results.sort(key=lambda item: (float(item["total_cost"]), item["pair_type"], item["source_label"], item["target_label"]))
    for rank, item in enumerate(path_results, start=1):
        item["rank"] = int(rank)

    if max_pair_results is not None:
        path_results = path_results[:max_pair_results]

    all_pair_count = len(path_results)
    if return_mode in ("requested", "selected", "request", "one"):
        selected = _select_requested_pair_results(model, path_results, start_idx, end_idx, kwargs)
        if selected:
            path_results = selected
            for rank, item in enumerate(path_results, start=1):
                item["rank"] = int(rank)
        else:
            return {
                "success": False,
                "algorithm": "FMM2D",
                "path_indices": [],
                "path_results": [],
                "ranked_paths": [],
                "total_cost": float("inf"),
                "travel_cost": float("inf"),
                "k_paths_found": 0,
                "expanded_states": int(total_expanded),
                "message": "FMM2D facility-pair mode computed the library, but the requested START_LABEL -> END_LABEL pair was not found/reachable.",
                "unreachable_pairs_count": int(len(unreachable_pairs)),
                "all_pair_paths_found": int(all_pair_count),
                "isolated_special_blocked": isolated_blocked,
                "is_lonlat": bool(is_lonlat),
                "fmm2d_pair_mode": True,
                "fmm2d_pair_return_mode": str(return_mode),
            }
    elif return_mode not in ("all", "library", "all_pairs"):
        raise ValueError("FMM2D_PAIR_RETURN_MODE must be 'requested', 'all', or 'auto'.")

    if not path_results:
        return {
            "success": False,
            "algorithm": "FMM2D",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_found": 0,
            "expanded_states": int(total_expanded),
            "message": "FMM2D facility-pair mode found no reachable pair paths.",
            "unreachable_pairs_count": int(len(unreachable_pairs)),
            "isolated_special_blocked": isolated_blocked,
            "is_lonlat": bool(is_lonlat),
            "fmm2d_pair_mode": True,
        }

    path_results, collision_summary = _apply_collision_avoidance_to_path_results(
        path_results=path_results,
        xm=xm,
        ym=ym,
        slow=effective_slowness,
        kwargs=kwargs,
    )

    best = path_results[0]
    if verbose:
        print("      FMM2D facility-pair result:")
        print(f"        returned paths    : {len(path_results):,}")
        print(f"        all found paths   : {all_pair_count:,}")
        print(f"        unreachable pairs : {len(unreachable_pairs):,}")
        print(f"        first path        : {best.get('pair_key')} | time={best.get('total_cost'):.2f} s")

    return {
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
        "k_paths_requested": int(len(path_results)),
        "k_paths_found": int(len(path_results)),
        "expanded_states": int(total_expanded),
        "message": "ok",
        "isolated_special_blocked": isolated_blocked,
        "is_lonlat": bool(is_lonlat),
        "fmm2d_pair_mode": True,
        "fmm2d_pair_return_mode": str(return_mode),
        "all_pair_paths_found": int(all_pair_count),
        "unreachable_pairs_count": int(len(unreachable_pairs)),
        "fmm2d_pair_strategy": "one FMM propagation per unique source",
        "fmm2d_pair_skip_same_label": bool(pair_skip_same_label),
        "fmm2d_pair_skip_same_coord": bool(pair_skip_same_coord),
        "fmm2d_pair_dedup_two_way": bool(pair_dedup_two_way),
        "fmm2d_pair_same_coord_tolerance_m": float(pair_same_coord_tolerance_m),
        "fmm2d_pair_min_distance_m": float(pair_min_distance_m),
        **collision_summary,
        "best_pair_key": str(best.get("pair_key", "")),
        "best_pair_type": str(best.get("pair_type", "")),
        "best_source_label": str(best.get("source_label", "")),
        "best_target_label": str(best.get("target_label", "")),
    }


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

    pair_mode = str(_param(kwargs, "FMM2D_PAIR_MODE", "selected")).strip().lower()
    if pair_mode in ("facility", "facility_pairs", "facility_library", "all_facility_pairs", "all_pairs"):
        if _collision_mode(kwargs) == "path_offset":
            return _run_facility_pair_path_offset_library(
                model=model,
                graph=graph,
                start_idx=start_idx,
                end_idx=end_idx,
                **kwargs,
            )
        return _run_facility_pair_library(
            model=model,
            graph=graph,
            start_idx=start_idx,
            end_idx=end_idx,
            **kwargs,
        )

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

    path_results, collision_summary = _apply_collision_avoidance_to_path_results(
        path_results=path_results,
        xm=xm,
        ym=ym,
        slow=effective_slowness,
        kwargs=kwargs,
    )

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
        **collision_summary,
    }
    return result
