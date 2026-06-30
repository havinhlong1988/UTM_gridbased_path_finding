#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
floodfill.py

Flood-fill / BFS path finder for the LAE-UTM pathfinding controller.

Protocol
--------
This module follows the existing main.py algorithm protocol:

    src/{ALGORITHM}.py must provide:
        run(model, graph, start_idx, end_idx, **kwargs) -> dict

Use from parameters.py:

    ALGORITHM = ["floodfill"]

Optional algorithm-specific settings are loaded from:

    params/floodfill.params

Purpose
-------
Flood-fill ignores slowness magnitude and finds the path with the smallest
number of graph steps. It is useful as a pure connectivity diagnostic:

    - Can the snapped start reach the snapped end?
    - Are missing paths caused by graph connectivity or by weighted-cost logic?
    - Are DB/DK endpoints actually connected to the traversable graph?

The graph itself is still created by main.py, so normal project rules still
apply before this algorithm runs:

    slowness < NO_FLY_SLOWNESS_THRESHOLD  -> traversable
    slowness >= NO_FLY_SLOWNESS_THRESHOLD -> blocked/no-fly
    DB/DK/FLZ may be forced flyable if parameters.py requests it
"""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Iterable

try:
    import parameters as prm
except Exception:  # pragma: no cover - keeps module importable in small tests
    prm = None

from src.model_io import iter_neighbors


ALGORITHM_NAME = "floodfill"


def _get_param(name: str, default=None):
    """Read parameters.py / params/floodfill.params values safely."""
    if prm is None:
        return default
    return getattr(prm, name, default)


def _as_optional_int(value, default=None):
    """Convert None-like values to default, otherwise int(value)."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "false"):
        return default
    return int(value)


def _is_lonlat_xy(model) -> bool:
    """Return True when x/y look like longitude/latitude degrees."""
    try:
        x = model["x"] if "x" in model.columns else model["lon"]
        y = model["y"] if "y" in model.columns else model["lat"]
        return (
            x.dropna().between(-180.0, 180.0).all()
            and y.dropna().between(-90.0, 90.0).all()
        )
    except Exception:
        return False


def _coord_xy(model, idx: int):
    """Return x/y or lon/lat coordinate pair for a node."""
    if "x" in model.columns and "y" in model.columns:
        return float(model.loc[int(idx), "x"]), float(model.loc[int(idx), "y"])
    return float(model.loc[int(idx), "lon"]), float(model.loc[int(idx), "lat"])


def _distance_to_goal_key(model, goal_idx: int, use_lonlat_scale: bool):
    """Return a sorting key that orders candidate neighbors by distance to goal."""
    gx, gy = _coord_xy(model, goal_idx)

    if use_lonlat_scale:
        lat0 = math.radians(float(gy))
        sx = 111_320.0 * math.cos(lat0)
        sy = 110_540.0
    else:
        sx = 1.0
        sy = 1.0

    def key(idx: int) -> float:
        x, y = _coord_xy(model, int(idx))
        dx = (x - gx) * sx
        dy = (y - gy) * sy
        return dx * dx + dy * dy

    return key


def _iter_flood_neighbors(model, graph, current: int, end_idx: int) -> Iterable[int]:
    """Yield neighbors, optionally sorted for deterministic/goal-oriented BFS."""
    neighbors = [int(n) for n in iter_neighbors(model, graph, int(current))]

    # Keep default False for speed and to preserve graph order.  When True,
    # BFS still finds the minimum number of graph steps, but ties are resolved
    # by moving closer to the goal first.  This often gives a cleaner diagnostic
    # path without changing the connectivity result.
    if bool(_get_param("FLOODFILL_SORT_NEIGHBORS_BY_GOAL", False)):
        use_lonlat_scale = bool(_get_param("FLOODFILL_LONLAT_DISTANCE_SCALE", True)) and _is_lonlat_xy(model)
        neighbors.sort(key=_distance_to_goal_key(model, int(end_idx), use_lonlat_scale))

    return neighbors


def _failure_result(
    message: str,
    start_idx: int,
    end_idx: int,
    t0: float,
    expanded_nodes: int = 0,
    visited_nodes: int = 0,
) -> dict:
    """Return a standard failure dictionary accepted by main.py."""
    return {
        "success": False,
        "algorithm": ALGORITHM_NAME,
        "message": str(message),
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "expanded_states": int(expanded_nodes),
        "visited_nodes": int(visited_nodes),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "k_paths_requested": 1,
        "k_paths_found": 0,
        "runtime_seconds": float(time.time() - t0),
    }


def run(model, graph, start_idx: int, end_idx: int, **kwargs) -> dict:
    """
    Run flood-fill / BFS search.

    Parameters
    ----------
    model : pandas.DataFrame
        Loaded model table from main.py.
    graph : dict
        Graph built by main.py. Must contain graph["valid_indices"].
    start_idx, end_idx : int
        Snapped search-node indices supplied by main.py.
    **kwargs
        Extra arguments from main.py are accepted for protocol compatibility.

    Returns
    -------
    dict
        Standard result dictionary consumed by main.py.
    """

    t0 = time.time()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    valid_indices = {int(i) for i in graph.get("valid_indices", set())}
    if not valid_indices:
        return _failure_result(
            "Graph has no traversable nodes in graph['valid_indices'].",
            start_idx=start_idx,
            end_idx=end_idx,
            t0=t0,
        )

    if start_idx not in valid_indices:
        return _failure_result(
            "Start node is blocked or not traversable.",
            start_idx=start_idx,
            end_idx=end_idx,
            t0=t0,
        )

    if end_idx not in valid_indices:
        return _failure_result(
            "End node is blocked or not traversable.",
            start_idx=start_idx,
            end_idx=end_idx,
            t0=t0,
        )

    # params/floodfill.params can set FLOODFILL_MAX_EXPANSIONS.  If it is None,
    # use main.py's generic max_expansions if passed.  This prevents accidental
    # unbounded BFS on very large models while keeping normal behavior simple.
    max_expansions = _get_param("FLOODFILL_MAX_EXPANSIONS", None)
    if max_expansions is None:
        max_expansions = kwargs.get("max_expansions", None)
    max_expansions = _as_optional_int(max_expansions, default=None)

    verbose = bool(_get_param("FLOODFILL_VERBOSE", False))
    progress_interval = int(_get_param("FLOODFILL_PROGRESS_INTERVAL", 100_000))

    if verbose:
        print("      [floodfill] BFS connectivity search")
        print(f"        start_idx      : {start_idx}")
        print(f"        end_idx        : {end_idx}")
        print(f"        valid nodes    : {len(valid_indices):,}")
        print(f"        max expansions : {max_expansions}")

    queue = deque([start_idx])
    came_from = {start_idx: None}
    visited = {start_idx}
    expanded_nodes = 0

    while queue:
        current = int(queue.popleft())
        expanded_nodes += 1

        if max_expansions is not None and expanded_nodes > max_expansions:
            return _failure_result(
                f"Stopped after reaching FLOODFILL_MAX_EXPANSIONS={max_expansions}.",
                start_idx=start_idx,
                end_idx=end_idx,
                t0=t0,
                expanded_nodes=expanded_nodes,
                visited_nodes=len(visited),
            )

        if verbose and progress_interval > 0 and expanded_nodes % progress_interval == 0:
            print(
                f"        [floodfill] expanded={expanded_nodes:,}, "
                f"visited={len(visited):,}, queue={len(queue):,}"
            )

        if current == end_idx:
            path = reconstruct_path(came_from, end_idx)
            runtime_seconds = float(time.time() - t0)

            return {
                "success": True,
                "algorithm": ALGORITHM_NAME,
                "message": "Path found by flood-fill/BFS.",
                "path_indices": path,
                # BFS minimizes graph steps, so the algorithm cost is number of edges.
                # main.py later computes real distance and traveltime from the path.
                "total_cost": float(max(0, len(path) - 1)),
                "expanded_nodes": int(expanded_nodes),
                "expanded_states": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "path_nodes": int(len(path)),
                "path_edges": int(max(0, len(path) - 1)),
                "k_paths_requested": 1,
                "k_paths_found": 1,
                "runtime_seconds": runtime_seconds,
            }

        for neighbor in _iter_flood_neighbors(model, graph, current, end_idx):
            neighbor = int(neighbor)

            # iter_neighbors should already obey graph['valid_indices'], but this
            # guard keeps floodfill safe if a future graph backend changes.
            if neighbor not in valid_indices:
                continue

            if neighbor in visited:
                continue

            visited.add(neighbor)
            came_from[neighbor] = current
            queue.append(neighbor)

    return _failure_result(
        "No path found by flood-fill/BFS.",
        start_idx=start_idx,
        end_idx=end_idx,
        t0=t0,
        expanded_nodes=expanded_nodes,
        visited_nodes=len(visited),
    )


def reconstruct_path(came_from: dict, end_idx: int) -> list[int]:
    """Reconstruct BFS path from the came_from dictionary."""
    path = []
    current = int(end_idx)

    while current is not None:
        path.append(int(current))
        current = came_from[current]

    path.reverse()
    return path
