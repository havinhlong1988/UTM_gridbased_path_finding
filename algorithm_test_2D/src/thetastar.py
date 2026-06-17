#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic Theta* any-angle path-finding algorithm.

Required interface:
    run(model, graph, start_idx, end_idx) -> dict

This file is designed to sit beside astar.py and use the same external
interface. It is compatible with the graph utilities from src.model_io:

    iter_neighbors
    heuristic_cost

Theta* difference from A*:
    A* sets parent(neighbor) = current.
    Theta* first checks whether parent(current) can directly see neighbor.
    If yes, it sets parent(neighbor) = parent(current), producing an
    any-angle path instead of a path constrained to grid edges.

Important for LAE-UTM:
    - The line-of-sight check samples along the straight segment.
    - Every sampled cell/node must be inside graph["valid_indices"].
    - Therefore RA / no-fly / blocked cells are not crossed if they are
      removed from valid_indices by build_grid_graph().
    - The direct segment cost is computed as integral slowness * distance,
      approximated by sampling the model slowness along the line.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.model_io import iter_neighbors, heuristic_cost


# ---------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------


def _as_float(value, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _get_row(model, idx: int):
    """
    Return model row for a node index.

    This supports both cases:
        1. idx is a pandas index label
        2. idx is an integer row position
    """
    if idx in model.index:
        return model.loc[idx]
    return model.iloc[int(idx)]


def _coord_tuple(model, idx: int) -> tuple[float, float, float]:
    """Return x, y, z coordinate for one model node."""
    row = _get_row(model, int(idx))
    x = _as_float(row["x"])
    y = _as_float(row["y"])
    z = _as_float(row["z"], 0.0) if "z" in model.columns else 0.0
    return x, y, z


def _euclidean_distance(model, idx1: int, idx2: int) -> float:
    """Euclidean distance between two model nodes."""
    x1, y1, z1 = _coord_tuple(model, idx1)
    x2, y2, z2 = _coord_tuple(model, idx2)
    return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))


def _slowness(model, idx: int) -> float:
    """Return positive slowness value for one node."""
    row = _get_row(model, int(idx))
    s = _as_float(row["slowness"], math.inf)
    if not math.isfinite(s) or s <= 0.0:
        return math.inf
    return float(s)


# ---------------------------------------------------------------------
# Grid cache for fast nearest-cell lookup along line-of-sight rays
# ---------------------------------------------------------------------


@dataclass
class GridCache:
    """
    Cached grid metadata used by line-of-sight sampling.

    x_values, y_values, z_values:
        Sorted unique coordinate vectors.

    xyz_to_idx:
        Dictionary mapping rounded coordinate tuple to model index.

    step:
        Sampling interval along a straight segment.
        Smaller step is safer near thin obstacles but slower.
    """

    x_values: np.ndarray
    y_values: np.ndarray
    z_values: np.ndarray
    xyz_to_idx: dict[tuple[float, float, float], int]
    valid_indices: set[int]
    step: float
    ndims: int


def _median_positive_spacing(values: np.ndarray) -> float | None:
    """Median positive spacing of sorted unique values."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None

    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]

    if len(diffs) == 0:
        return None

    return float(np.median(diffs))


def _make_grid_cache(model, graph) -> GridCache:
    """
    Build a cache that maps sampled x/y/z positions to model indices.

    The path model is normally a structured grid. This function still works
    for most nearly-regular grids because sampled coordinates are snapped to
    the nearest existing x/y/z coordinate before lookup.
    """
    x_values = np.sort(model["x"].astype(float).unique())
    y_values = np.sort(model["y"].astype(float).unique())

    if "z" in model.columns:
        z_values = np.sort(model["z"].astype(float).unique())
    else:
        z_values = np.array([0.0], dtype=float)

    # Determine whether this is effectively 2D or 3D.
    ndims = 2 if len(z_values) <= 1 else 3

    dx = _median_positive_spacing(x_values)
    dy = _median_positive_spacing(y_values)
    dz = _median_positive_spacing(z_values) if ndims == 3 else None

    spacings = [v for v in (dx, dy, dz) if v is not None and math.isfinite(v) and v > 0.0]
    if spacings:
        # Half-cell sampling is conservative enough for line-of-sight checking.
        step = 0.5 * min(spacings)
    else:
        # Fallback for unusual data. This is slow but safe.
        step = 1.0

    xyz_to_idx: dict[tuple[float, float, float], int] = {}

    has_z = "z" in model.columns
    for idx, row in model.iterrows():
        x = round(float(row["x"]), 8)
        y = round(float(row["y"]), 8)
        z = round(float(row["z"]), 8) if has_z else 0.0
        xyz_to_idx[(x, y, z)] = int(idx)

    valid_indices = set(int(v) for v in graph.get("valid_indices", []))

    return GridCache(
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        xyz_to_idx=xyz_to_idx,
        valid_indices=valid_indices,
        step=float(step),
        ndims=int(ndims),
    )


def _nearest_value(values: np.ndarray, value: float) -> float:
    """Nearest coordinate value from a sorted coordinate vector."""
    if len(values) == 1:
        return float(values[0])

    pos = int(np.searchsorted(values, value))

    if pos <= 0:
        return float(values[0])
    if pos >= len(values):
        return float(values[-1])

    before = float(values[pos - 1])
    after = float(values[pos])

    if abs(value - before) <= abs(value - after):
        return before
    return after


def _nearest_model_index(cache: GridCache, x: float, y: float, z: float) -> int | None:
    """Return nearest model index for sampled coordinate."""
    xn = round(_nearest_value(cache.x_values, x), 8)
    yn = round(_nearest_value(cache.y_values, y), 8)

    if cache.ndims == 3:
        zn = round(_nearest_value(cache.z_values, z), 8)
    else:
        zn = round(float(cache.z_values[0]), 8)

    return cache.xyz_to_idx.get((xn, yn, zn), None)


def _sample_segment_indices(
    model,
    cache: GridCache,
    idx1: int,
    idx2: int,
) -> list[int] | None:
    """
    Sample the straight segment from idx1 to idx2 and return crossed node indices.

    Returns None if any sampled point cannot be mapped to a model node.
    """
    x1, y1, z1 = _coord_tuple(model, int(idx1))
    x2, y2, z2 = _coord_tuple(model, int(idx2))

    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    if not math.isfinite(length):
        return None

    if length <= 0.0:
        return [int(idx1)]

    n_steps = max(1, int(math.ceil(length / max(cache.step, 1.0e-12))))

    sampled: list[int] = []
    last_idx: int | None = None

    for k in range(n_steps + 1):
        t = k / n_steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        z = z1 + t * (z2 - z1)

        nearest_idx = _nearest_model_index(cache, x, y, z)
        if nearest_idx is None:
            return None

        nearest_idx = int(nearest_idx)

        if nearest_idx != last_idx:
            sampled.append(nearest_idx)
            last_idx = nearest_idx

    # Make sure exact endpoints are included.
    if sampled[0] != int(idx1):
        sampled.insert(0, int(idx1))
    if sampled[-1] != int(idx2):
        sampled.append(int(idx2))

    return sampled


def _line_of_sight(
    model,
    graph,
    cache: GridCache,
    idx1: int,
    idx2: int,
    los_cache: dict[tuple[int, int], bool],
) -> bool:
    """
    Return True if the straight segment idx1 -> idx2 crosses only valid cells.

    This is the key Theta* check. The segment is accepted only when all sampled
    cells are traversable according to graph["valid_indices"].
    """
    idx1 = int(idx1)
    idx2 = int(idx2)

    key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
    if key in los_cache:
        return los_cache[key]

    if idx1 not in cache.valid_indices or idx2 not in cache.valid_indices:
        los_cache[key] = False
        return False

    sampled = _sample_segment_indices(model, cache, idx1, idx2)

    if sampled is None:
        los_cache[key] = False
        return False

    ok = all(int(i) in cache.valid_indices for i in sampled)
    los_cache[key] = bool(ok)
    return bool(ok)


def _direct_segment_cost(
    model,
    cache: GridCache,
    idx1: int,
    idx2: int,
    cost_cache: dict[tuple[int, int], float],
) -> float:
    """
    Approximate the cost of a direct straight segment.

    Cost definition:
        cost = integral(slowness ds)

    Numerically:
        - sample the segment through the grid,
        - average slowness along each small section,
        - multiply by section length.

    If slowness is in s/m, the cost is travel time in seconds.
    """
    idx1 = int(idx1)
    idx2 = int(idx2)

    key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
    if key in cost_cache:
        return cost_cache[key]

    sampled = _sample_segment_indices(model, cache, idx1, idx2)

    if sampled is None or len(sampled) == 0:
        cost_cache[key] = math.inf
        return math.inf

    # Single-cell case.
    if len(sampled) == 1:
        cost_cache[key] = 0.0
        return 0.0

    total = 0.0

    for a, b in zip(sampled[:-1], sampled[1:]):
        d = _euclidean_distance(model, int(a), int(b))
        s1 = _slowness(model, int(a))
        s2 = _slowness(model, int(b))

        if not math.isfinite(d) or not math.isfinite(s1) or not math.isfinite(s2):
            total = math.inf
            break

        total += d * 0.5 * (s1 + s2)

    cost_cache[key] = float(total)
    return float(total)


def _safe_heuristic(model, graph, idx: int, end_idx: int) -> float:
    """
    Use project heuristic_cost() when possible.

    If the imported heuristic raises an error because of argument naming or
    graph metadata differences, fall back to Euclidean distance multiplied by
    the minimum positive slowness. This fallback remains admissible when
    min_slowness is the lowest traversal slowness in the model.
    """
    try:
        h = heuristic_cost(model=model, graph=graph, idx=int(idx), end_idx=int(end_idx))
        h = float(h)
        if math.isfinite(h):
            return h
    except TypeError:
        try:
            h = heuristic_cost(model, graph, int(idx), int(end_idx))
            h = float(h)
            if math.isfinite(h):
                return h
        except Exception:
            pass
    except Exception:
        pass

    # Fallback heuristic.
    distance = _euclidean_distance(model, int(idx), int(end_idx))

    try:
        vals = model["slowness"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        min_slow = float(np.min(vals)) if len(vals) else 0.0
    except Exception:
        min_slow = 0.0

    return float(distance * min_slow)


# ---------------------------------------------------------------------
# Theta* main algorithm
# ---------------------------------------------------------------------


def run(model, graph, start_idx: int, end_idx: int) -> dict:
    """
    Run Basic Theta* search.

    Parameters
    ----------
    model : pandas.DataFrame
        Model table with columns x, y, z, slowness, label, label_prefix.
    graph : dict
        Graph metadata built by build_grid_graph().
        Must contain graph["valid_indices"].
    start_idx : int
        Search start node index.
    end_idx : int
        Search end node index.

    Returns
    -------
    result : dict
        Contains path_indices, total_cost, expanded_nodes, runtime, etc.
    """

    t0 = time.time()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    valid_indices = set(int(v) for v in graph.get("valid_indices", []))

    if start_idx not in valid_indices:
        return {
            "success": False,
            "algorithm": "thetastar",
            "message": "Start node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    if end_idx not in valid_indices:
        return {
            "success": False,
            "algorithm": "thetastar",
            "message": "End node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    cache = _make_grid_cache(model, graph)

    open_heap: list[tuple[float, int, int]] = []
    heap_counter = 0

    g_score: dict[int, float] = {start_idx: 0.0}

    # In Theta*, the start node is its own parent.
    came_from: dict[int, int] = {start_idx: start_idx}

    f_start = _safe_heuristic(model, graph, start_idx, end_idx)
    heapq.heappush(open_heap, (f_start, heap_counter, start_idx))

    visited: set[int] = set()
    expanded_nodes = 0

    los_cache: dict[tuple[int, int], bool] = {}
    cost_cache: dict[tuple[int, int], float] = {}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        current = int(current)

        if current in visited:
            continue

        visited.add(current)
        expanded_nodes += 1

        if current == end_idx:
            path = reconstruct_path(came_from, current)
            total_cost = float(g_score[current])

            return {
                "success": True,
                "algorithm": "thetastar",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": total_cost,
                "expanded_nodes": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "runtime_seconds": float(time.time() - t0),
            }

        current_parent = int(came_from.get(current, current))

        for neighbor in iter_neighbors(model, graph, current):
            neighbor = int(neighbor)

            if neighbor in visited:
                continue

            # ---------------------------------------------------------
            # Basic Theta* update:
            #
            # If parent(current) has line of sight to neighbor, try:
            #     parent(neighbor) = parent(current)
            #
            # Otherwise, fall back to normal A*-like:
            #     parent(neighbor) = current
            # ---------------------------------------------------------
            use_parent_shortcut = _line_of_sight(
                model=model,
                graph=graph,
                cache=cache,
                idx1=current_parent,
                idx2=neighbor,
                los_cache=los_cache,
            )

            if use_parent_shortcut:
                candidate_parent = current_parent
                base_g = g_score.get(current_parent, math.inf)
                segment_cost = _direct_segment_cost(
                    model=model,
                    cache=cache,
                    idx1=current_parent,
                    idx2=neighbor,
                    cost_cache=cost_cache,
                )
            else:
                candidate_parent = current
                base_g = g_score.get(current, math.inf)
                segment_cost = _direct_segment_cost(
                    model=model,
                    cache=cache,
                    idx1=current,
                    idx2=neighbor,
                    cost_cache=cost_cache,
                )

            tentative_g = base_g + segment_cost

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = int(candidate_parent)
                g_score[neighbor] = float(tentative_g)

                f = tentative_g + _safe_heuristic(model, graph, neighbor, end_idx)

                heap_counter += 1
                heapq.heappush(open_heap, (float(f), heap_counter, neighbor))

    return {
        "success": False,
        "algorithm": "thetastar",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "visited_nodes": int(len(visited)),
        "runtime_seconds": float(time.time() - t0),
    }


def reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    """
    Reconstruct path from came_from dictionary.

    For Theta*, some jumps may skip intermediate grid nodes because the path
    is any-angle. The returned path contains waypoint indices, not every cell
    crossed by each straight segment.
    """
    current = int(current)
    path = [current]

    while current in came_from:
        parent = int(came_from[current])

        if parent == current:
            break

        current = parent
        path.append(current)

    path.reverse()
    return path
