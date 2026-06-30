#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/FMM.py

True 2-D Fast Marching Method (FMM) for the LAE-UTM main.py protocol.

This module is called by main.py as:

    result = src.FMM.run(model=model, graph=graph, start_idx=i, end_idx=j, **kwargs)

Purpose
-------
This is different from graph A*/Dijkstra.  It solves the isotropic Eikonal
arrival-time equation on a regular 2-D grid using a first-order upwind FMM:

    |grad T(x, y)| = s(x, y)

where T is arrival time and s is slowness in s/m.  The ray/path is then traced
back from B to A by following the negative gradient / descending arrival-time
field.

No-fly handling
---------------
Two modes are supported:

    FMM_NOFLY_MODE = "hard"
        No-fly cells are blocked and FMM will not enter them.
        This is recommended for operational UAV no-fly zones.

    FMM_NOFLY_MODE = "high_slowness"
        No-fly cells remain in the grid but their slowness is replaced by a
        very high value.  The raypath will avoid them because they are reached
        very late / have high travel time, but they are not mathematically
        forbidden.  This is useful for testing soft barriers.

Compatibility
-------------
Returns the same dictionary style as other LAE-UTM algorithms:
    result["path_indices"]
    result["path_results"]
    result["total_cost"]
    result["k_paths_found"]
    result["expanded_states"]

Notes
-----
This implementation assumes the model nodes form a mostly regular 2-D grid.
Missing cells and hard no-fly cells are allowed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable

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
    """Read parameter from explicit kwargs, then parameters.py, then default."""
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


def _none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("none", "null", "auto", "all", "unlimited")
    return False


def _optional_positive_int(value: Any, default: int) -> tuple[int, bool]:
    if _none_like(value):
        return max(1, int(default)), True
    return max(1, int(value)), False


# ============================================================
# Grid data container
# ============================================================


@dataclass
class FMMGrid:
    x_col: str
    y_col: str
    x_orig: np.ndarray
    y_orig: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    is_lonlat: bool
    ix: np.ndarray
    iy: np.ndarray
    unique_x_key: np.ndarray
    unique_y_key: np.ndarray
    unique_x_m: np.ndarray
    unique_y_m: np.ndarray
    dx_m: float
    dy_m: float
    ny: int
    nx: int
    node_at_cell: np.ndarray
    cell_of_node: dict[int, tuple[int, int]]


# ============================================================
# Coordinate and grid helpers
# ============================================================


def _get_xy_columns(model: pd.DataFrame) -> tuple[str, str]:
    if {"x", "y"}.issubset(model.columns):
        return "x", "y"
    if {"lon", "lat"}.issubset(model.columns):
        return "lon", "lat"
    raise ValueError("FMM requires model columns x/y or lon/lat.")


def _looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
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


def _xy_to_metric_arrays(model: pd.DataFrame) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    x_col, y_col = _get_xy_columns(model)
    x_orig = pd.to_numeric(model[x_col], errors="coerce").to_numpy(dtype=float, copy=True)
    y_orig = pd.to_numeric(model[y_col], errors="coerce").to_numpy(dtype=float, copy=True)

    is_lonlat = _looks_like_lonlat(x_orig, y_orig)
    if not is_lonlat:
        return x_col, y_col, x_orig, y_orig, x_orig.copy(), y_orig.copy(), False

    lon0 = float(np.nanmean(x_orig))
    lat0 = float(np.nanmean(y_orig))
    rad = math.pi / 180.0
    x_m = (x_orig - lon0) * 111_320.0 * math.cos(lat0 * rad)
    y_m = (y_orig - lat0) * 110_540.0
    return x_col, y_col, x_orig, y_orig, x_m.astype(float), y_m.astype(float), True


def _build_regular_grid(model: pd.DataFrame, round_decimals_lonlat: int = 10, round_decimals_xy: int = 6) -> FMMGrid:
    x_col, y_col, x_orig, y_orig, x_m, y_m, is_lonlat = _xy_to_metric_arrays(model)
    decimals = round_decimals_lonlat if is_lonlat else round_decimals_xy

    x_key = np.round(x_orig, decimals=decimals)
    y_key = np.round(y_orig, decimals=decimals)

    unique_x_key = np.array(sorted(pd.unique(x_key)))
    unique_y_key = np.array(sorted(pd.unique(y_key)))
    nx = int(len(unique_x_key))
    ny = int(len(unique_y_key))

    x_to_ix = {v: i for i, v in enumerate(unique_x_key)}
    y_to_iy = {v: i for i, v in enumerate(unique_y_key)}

    ix = np.fromiter((x_to_ix[v] for v in x_key), dtype=np.int64, count=len(model))
    iy = np.fromiter((y_to_iy[v] for v in y_key), dtype=np.int64, count=len(model))

    node_at_cell = np.full((ny, nx), -1, dtype=np.int64)
    cell_of_node: dict[int, tuple[int, int]] = {}
    duplicate_count = 0

    for node_i, (r, c) in enumerate(zip(iy, ix)):
        rr = int(r)
        cc = int(c)
        if node_at_cell[rr, cc] >= 0:
            duplicate_count += 1
            continue
        node_at_cell[rr, cc] = int(node_i)
        cell_of_node[int(node_i)] = (rr, cc)

    if duplicate_count:
        print(f"      [FMM WARNING] duplicate grid cells: {duplicate_count:,}; first node kept.")

    unique_x_m = np.full(nx, np.nan, dtype=float)
    unique_y_m = np.full(ny, np.nan, dtype=float)
    for c in range(nx):
        vals = x_m[ix == c]
        unique_x_m[c] = float(np.nanmean(vals)) if len(vals) else np.nan
    for r in range(ny):
        vals = y_m[iy == r]
        unique_y_m[r] = float(np.nanmean(vals)) if len(vals) else np.nan

    dx_vals = np.diff(np.sort(unique_x_m[np.isfinite(unique_x_m)]))
    dy_vals = np.diff(np.sort(unique_y_m[np.isfinite(unique_y_m)]))
    dx_vals = dx_vals[np.isfinite(dx_vals) & (dx_vals > 0.0)]
    dy_vals = dy_vals[np.isfinite(dy_vals) & (dy_vals > 0.0)]

    if len(dx_vals) == 0 or len(dy_vals) == 0:
        raise ValueError("FMM requires at least two unique x and y grid coordinates.")

    dx_m = float(np.median(dx_vals))
    dy_m = float(np.median(dy_vals))

    return FMMGrid(
        x_col=x_col,
        y_col=y_col,
        x_orig=x_orig,
        y_orig=y_orig,
        x_m=x_m,
        y_m=y_m,
        is_lonlat=is_lonlat,
        ix=ix,
        iy=iy,
        unique_x_key=unique_x_key,
        unique_y_key=unique_y_key,
        unique_x_m=unique_x_m,
        unique_y_m=unique_y_m,
        dx_m=dx_m,
        dy_m=dy_m,
        ny=ny,
        nx=nx,
        node_at_cell=node_at_cell,
        cell_of_node=cell_of_node,
    )


# ============================================================
# Label / flyability / slowness helpers
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


def _special_mask(model: pd.DataFrame, prefixes: Iterable[str]) -> np.ndarray:
    labels = _label_array(model).astype(str)
    label_prefixes = _label_prefix_array(model).astype(str)
    mask = np.zeros(len(model), dtype=bool)
    for p in prefixes:
        pp = str(p).upper()
        if not pp:
            continue
        mask |= np.char.startswith(np.char.upper(labels), pp)
        mask |= np.char.startswith(np.char.upper(label_prefixes), pp)
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
    for item in valid:
        i = int(item)
        if 0 <= i < n:
            mask[i] = True
    return mask


def _grid_arrays_for_fmm(
    *,
    model: pd.DataFrame,
    graph: dict[str, Any],
    grid: FMMGrid,
    start_idx: int,
    end_idx: int,
    kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build slowness grid and active mask for FMM."""
    threshold = float(_param(kwargs, "NO_FLY_SLOWNESS_THRESHOLD", 10.0))
    fallback = float(_param(kwargs, "FMM_FLYABLE_SLOWNESS_FALLBACK", _param(kwargs, "FLYABLE_SLOWNESS", 0.085)))
    nofly_mode = str(_param(kwargs, "FMM_NOFLY_MODE", "hard")).strip().lower()
    high_slow = float(_param(kwargs, "FMM_NOFLY_HIGH_SLOWNESS", 10_000.0))
    hard_block_nan = _as_bool(_param(kwargs, "FMM_BLOCK_NONFINITE_SLOWNESS", True))
    always_prefixes = tuple(_param(kwargs, "ALWAYS_FLYABLE_PREFIXES", ("DB", "DK", "BD", "FLZ")))

    n = len(model)
    slow_node = _slowness_array(model, fallback=fallback)
    finite_node = np.isfinite(slow_node)

    graph_valid = _valid_mask_from_graph(model, graph)
    base_nofly = (~finite_node) | (slow_node >= threshold)
    special = _special_mask(model, always_prefixes)
    special[int(start_idx)] = True
    special[int(end_idx)] = True

    if nofly_mode in ("hard", "block", "blocked"):
        active_node = graph_valid.copy()
        # main.py may already force start/end valid. Keep only if graph says valid.
        active_node[int(start_idx)] = bool(graph_valid[int(start_idx)])
        active_node[int(end_idx)] = bool(graph_valid[int(end_idx)])
    elif nofly_mode in ("high_slowness", "high", "soft", "penalty"):
        # Soft no-fly: include all model cells, but no-fly gets very high slowness.
        active_node = np.ones(n, dtype=bool)
        if hard_block_nan:
            active_node &= finite_node
        slow_node[base_nofly] = high_slow
        # Special DB/DK/BD/FLZ should use a reasonable local/fallback slowness.
        forced_special = special & base_nofly
        slow_node[forced_special] = fallback
    else:
        raise ValueError("FMM_NOFLY_MODE must be 'hard' or 'high_slowness'.")

    # If forced start/end have no-fly slowness, do not allow their high no-fly
    # value to dominate the first/last segment.
    for idx in (int(start_idx), int(end_idx)):
        if not np.isfinite(slow_node[idx]) or slow_node[idx] <= 0.0 or slow_node[idx] >= threshold:
            slow_node[idx] = fallback

    slowness_grid = np.full((grid.ny, grid.nx), np.inf, dtype=float)
    active_grid = np.zeros((grid.ny, grid.nx), dtype=bool)
    original_nofly_grid = np.zeros((grid.ny, grid.nx), dtype=bool)
    special_grid = np.zeros((grid.ny, grid.nx), dtype=bool)

    for node_i, (r, c) in grid.cell_of_node.items():
        slowness_grid[r, c] = float(slow_node[node_i])
        active_grid[r, c] = bool(active_node[node_i])
        original_nofly_grid[r, c] = bool(base_nofly[node_i])
        special_grid[r, c] = bool(special[node_i])

    # Missing cells are inactive.
    active_grid &= grid.node_at_cell >= 0

    # Non-positive slowness cannot be used by the Eikonal update.
    bad_active = active_grid & (~np.isfinite(slowness_grid) | (slowness_grid <= 0.0))
    slowness_grid[bad_active] = fallback

    meta = {
        "nofly_mode": nofly_mode,
        "threshold": threshold,
        "high_slowness": high_slow,
        "fallback_slowness": fallback,
        "graph_valid_nodes": int(np.count_nonzero(graph_valid)),
        "active_nodes": int(np.count_nonzero(active_grid)),
        "original_nofly_nodes": int(np.count_nonzero(original_nofly_grid)),
        "special_nodes": int(np.count_nonzero(special_grid)),
    }
    return slowness_grid, active_grid, original_nofly_grid, special_grid, meta


# ============================================================
# True FMM solver
# ============================================================


FAR = np.uint8(0)
TRIAL = np.uint8(1)
ACCEPTED = np.uint8(2)
BLOCKED = np.uint8(3)


def _accepted_axis_min(T: np.ndarray, status: np.ndarray, r: int, c: int, axis: str) -> float | None:
    vals: list[float] = []
    ny, nx = T.shape
    if axis == "x":
        for cc in (c - 1, c + 1):
            if 0 <= cc < nx and status[r, cc] == ACCEPTED and np.isfinite(T[r, cc]):
                vals.append(float(T[r, cc]))
    else:
        for rr in (r - 1, r + 1):
            if 0 <= rr < ny and status[rr, c] == ACCEPTED and np.isfinite(T[rr, c]):
                vals.append(float(T[rr, c]))
    if not vals:
        return None
    return float(min(vals))


def _upwind_update(
    T: np.ndarray,
    status: np.ndarray,
    slowness: np.ndarray,
    active: np.ndarray,
    r: int,
    c: int,
    dx: float,
    dy: float,
) -> float:
    """First-order upwind FMM update solving |grad T| = s."""
    if not active[r, c]:
        return float("inf")

    s = float(slowness[r, c])
    if not np.isfinite(s) or s <= 0.0:
        return float("inf")

    ax = _accepted_axis_min(T, status, r, c, "x")
    ay = _accepted_axis_min(T, status, r, c, "y")

    candidates: list[float] = []
    if ax is not None:
        candidates.append(ax + s * dx)
    if ay is not None:
        candidates.append(ay + s * dy)

    if ax is not None and ay is not None:
        # Solve: ((T-ax)/dx)^2 + ((T-ay)/dy)^2 = s^2
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        A = inv_dx2 + inv_dy2
        B = -2.0 * (ax * inv_dx2 + ay * inv_dy2)
        C = ax * ax * inv_dx2 + ay * ay * inv_dy2 - s * s
        disc = B * B - 4.0 * A * C
        if disc >= 0.0:
            t = (-B + math.sqrt(disc)) / (2.0 * A)
            # Causality condition.  If violated, use one-sided update.
            if t >= max(ax, ay):
                candidates.append(float(t))

    if not candidates:
        return float("inf")
    return float(min(candidates))


def _fmm_solve(
    *,
    slowness: np.ndarray,
    active: np.ndarray,
    start_cell: tuple[int, int],
    end_cell: tuple[int, int],
    dx: float,
    dy: float,
    inf_time: float,
    stop_at_end: bool,
    max_accepted: int | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    ny, nx = slowness.shape
    T = np.full((ny, nx), float(inf_time), dtype=float)
    status = np.full((ny, nx), FAR, dtype=np.uint8)
    status[~active] = BLOCKED

    sr, sc = start_cell
    er, ec = end_cell

    if not active[sr, sc]:
        return T, status, "start_blocked", 0
    if not active[er, ec]:
        return T, status, "end_blocked", 0

    T[sr, sc] = 0.0
    status[sr, sc] = ACCEPTED
    accepted_count = 1

    heap: list[tuple[float, int, int]] = []
    four = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def push_update(rr: int, cc: int) -> None:
        if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
            return
        if status[rr, cc] == ACCEPTED or status[rr, cc] == BLOCKED:
            return
        t_new = _upwind_update(T, status, slowness, active, rr, cc, dx, dy)
        if np.isfinite(t_new) and t_new < T[rr, cc]:
            T[rr, cc] = t_new
            status[rr, cc] = TRIAL
            heapq.heappush(heap, (float(t_new), int(rr), int(cc)))

    for dr, dc in four:
        push_update(sr + dr, sc + dc)

    while heap:
        t, r, c = heapq.heappop(heap)
        if status[r, c] == ACCEPTED:
            continue
        if t > T[r, c]:
            continue

        status[r, c] = ACCEPTED
        accepted_count += 1

        if stop_at_end and (r, c) == (er, ec):
            return T, status, "ok", accepted_count

        if max_accepted is not None and accepted_count >= int(max_accepted):
            return T, status, "max_accepted", accepted_count

        for dr, dc in four:
            push_update(r + dr, c + dc)

    if np.isfinite(T[er, ec]):
        return T, status, "ok", accepted_count
    return T, status, "unreachable", accepted_count


# ============================================================
# Ray tracing from arrival-time field
# ============================================================


def _best_lower_neighbor(T: np.ndarray, active: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    ny, nx = T.shape
    current = float(T[r, c])
    best_cell: tuple[int, int] | None = None
    best_t = current
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            if not active[rr, cc] or not np.isfinite(T[rr, cc]):
                continue
            if T[rr, cc] < best_t:
                best_t = float(T[rr, cc])
                best_cell = (int(rr), int(cc))
    return best_cell


def _nearest_cell_from_xy(grid: FMMGrid, x: float, y: float) -> tuple[int, int]:
    c = int(np.searchsorted(grid.unique_x_m, x))
    if c <= 0:
        cc = 0
    elif c >= grid.nx:
        cc = grid.nx - 1
    else:
        cc = c if abs(grid.unique_x_m[c] - x) < abs(grid.unique_x_m[c - 1] - x) else c - 1

    r = int(np.searchsorted(grid.unique_y_m, y))
    if r <= 0:
        rr = 0
    elif r >= grid.ny:
        rr = grid.ny - 1
    else:
        rr = r if abs(grid.unique_y_m[r] - y) < abs(grid.unique_y_m[r - 1] - y) else r - 1
    return int(rr), int(cc)


def _finite_diff_gradient(T: np.ndarray, active: np.ndarray, r: int, c: int, dx: float, dy: float) -> tuple[float, float] | None:
    ny, nx = T.shape
    t0 = float(T[r, c])
    if not np.isfinite(t0):
        return None

    # dT/dx
    gx_vals: list[float] = []
    if c + 1 < nx and active[r, c + 1] and np.isfinite(T[r, c + 1]):
        gx_vals.append((float(T[r, c + 1]) - t0) / dx)
    if c - 1 >= 0 and active[r, c - 1] and np.isfinite(T[r, c - 1]):
        gx_vals.append((t0 - float(T[r, c - 1])) / dx)

    gy_vals: list[float] = []
    if r + 1 < ny and active[r + 1, c] and np.isfinite(T[r + 1, c]):
        gy_vals.append((float(T[r + 1, c]) - t0) / dy)
    if r - 1 >= 0 and active[r - 1, c] and np.isfinite(T[r - 1, c]):
        gy_vals.append((t0 - float(T[r - 1, c])) / dy)

    if not gx_vals and not gy_vals:
        return None
    gx = float(np.mean(gx_vals)) if gx_vals else 0.0
    gy = float(np.mean(gy_vals)) if gy_vals else 0.0
    if not np.isfinite(gx) or not np.isfinite(gy):
        return None
    if math.hypot(gx, gy) <= 0.0:
        return None
    return gx, gy


def _trace_ray_gradient(
    *,
    T: np.ndarray,
    active: np.ndarray,
    grid: FMMGrid,
    start_cell: tuple[int, int],
    end_cell: tuple[int, int],
    step_factor: float,
    max_steps: int,
) -> list[int]:
    """Trace ray from end to start by descending T.

    The continuous gradient descent samples cells and converts them to node
    indices.  If a local gradient is not usable, it falls back to steepest
    lower neighbouring cell.  This keeps the trace robust around blocked holes.
    """
    sr, sc = start_cell
    er, ec = end_cell

    x = float(grid.unique_x_m[ec])
    y = float(grid.unique_y_m[er])
    x_start = float(grid.unique_x_m[sc])
    y_start = float(grid.unique_y_m[sr])

    step_m = max(1.0e-6, float(step_factor) * min(grid.dx_m, grid.dy_m))
    close_m = 0.75 * max(grid.dx_m, grid.dy_m)

    path_nodes_reverse: list[int] = []
    visited_cells: set[tuple[int, int]] = set()

    r, c = er, ec
    for _ in range(max_steps):
        if not (0 <= r < grid.ny and 0 <= c < grid.nx):
            break
        node_i = int(grid.node_at_cell[r, c])
        if node_i >= 0 and (not path_nodes_reverse or path_nodes_reverse[-1] != node_i):
            path_nodes_reverse.append(node_i)

        if (r, c) == (sr, sc):
            break
        if math.hypot(x - x_start, y - y_start) <= close_m:
            start_node = int(grid.node_at_cell[sr, sc])
            if start_node >= 0 and (not path_nodes_reverse or path_nodes_reverse[-1] != start_node):
                path_nodes_reverse.append(start_node)
            break

        grad = _finite_diff_gradient(T, active, r, c, grid.dx_m, grid.dy_m)
        moved_by_gradient = False
        if grad is not None:
            gx, gy = grad
            norm = math.hypot(gx, gy)
            if norm > 0.0:
                # Move opposite to grad(T).
                x_new = x - step_m * gx / norm
                y_new = y - step_m * gy / norm
                rr, cc = _nearest_cell_from_xy(grid, x_new, y_new)
                if active[rr, cc] and np.isfinite(T[rr, cc]) and T[rr, cc] <= T[r, c] + 1.0e-9:
                    x, y = x_new, y_new
                    r, c = rr, cc
                    moved_by_gradient = True

        if not moved_by_gradient:
            nxt = _best_lower_neighbor(T, active, r, c)
            if nxt is None:
                break
            r, c = nxt
            x = float(grid.unique_x_m[c])
            y = float(grid.unique_y_m[r])

        cell = (int(r), int(c))
        if cell in visited_cells:
            # Rare plateau/loop fallback: use best lower neighbour once more.
            nxt = _best_lower_neighbor(T, active, r, c)
            if nxt is None or nxt == cell:
                break
            r, c = nxt
            x = float(grid.unique_x_m[c])
            y = float(grid.unique_y_m[r])
        visited_cells.add(cell)

    # Reverse to start -> end order and remove duplicate nodes while preserving order.
    path = list(reversed(path_nodes_reverse))
    cleaned: list[int] = []
    seen_last = None
    for idx in path:
        if idx != seen_last:
            cleaned.append(int(idx))
            seen_last = idx

    # Ensure endpoints.
    start_node = int(grid.node_at_cell[sr, sc])
    end_node = int(grid.node_at_cell[er, ec])
    if cleaned and cleaned[0] != start_node:
        cleaned.insert(0, start_node)
    if cleaned and cleaned[-1] != end_node:
        cleaned.append(end_node)
    return cleaned


def _trace_ray_steepest_neighbor(
    *,
    T: np.ndarray,
    active: np.ndarray,
    grid: FMMGrid,
    start_cell: tuple[int, int],
    end_cell: tuple[int, int],
    max_steps: int,
) -> list[int]:
    sr, sc = start_cell
    r, c = end_cell
    path_rev: list[int] = []
    seen: set[tuple[int, int]] = set()

    for _ in range(max_steps):
        if (r, c) in seen:
            break
        seen.add((r, c))
        node_i = int(grid.node_at_cell[r, c])
        if node_i >= 0:
            path_rev.append(node_i)
        if (r, c) == (sr, sc):
            break
        nxt = _best_lower_neighbor(T, active, r, c)
        if nxt is None:
            break
        r, c = nxt

    path = list(reversed(path_rev))
    start_node = int(grid.node_at_cell[sr, sc])
    end_node = int(grid.node_at_cell[end_cell[0], end_cell[1]])
    if path and path[0] != start_node:
        path.insert(0, start_node)
    if path and path[-1] != end_node:
        path.append(end_node)
    return [int(i) for i in path]


def _path_distance_m(grid: FMMGrid, path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    idx = np.asarray(path, dtype=int)
    dx = np.diff(grid.x_m[idx])
    dy = np.diff(grid.y_m[idx])
    return float(np.sum(np.hypot(dx, dy)))


def _path_nofly_count(path: list[int], model: pd.DataFrame, threshold: float) -> int:
    if not path or "slowness" not in model.columns:
        return 0
    slow = pd.to_numeric(model.loc[path, "slowness"], errors="coerce").to_numpy(dtype=float, copy=True)
    return int(np.count_nonzero(~np.isfinite(slow) | (slow >= float(threshold))))


# ============================================================
# Multiple path via repeated FMM with high-cost corridor penalty
# ============================================================


def _path_buffer_mask(grid: FMMGrid, path: list[int], radius_m: float) -> np.ndarray:
    mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    if radius_m <= 0.0 or not path:
        return mask

    coords_all = np.column_stack([grid.x_m, grid.y_m])
    path_coords = coords_all[np.asarray(path, dtype=int)]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(path_coords)
        dist, _ = tree.query(coords_all, k=1, distance_upper_bound=float(radius_m))
        node_mask = np.isfinite(dist)
    except Exception:
        node_mask = np.zeros(len(coords_all), dtype=bool)
        r2 = float(radius_m) * float(radius_m)
        chunk = 20_000
        px = path_coords[:, 0]
        py = path_coords[:, 1]
        for s in range(0, len(coords_all), chunk):
            e = min(s + chunk, len(coords_all))
            ax = coords_all[s:e, 0][:, None]
            ay = coords_all[s:e, 1][:, None]
            d2 = (ax - px[None, :]) ** 2 + (ay - py[None, :]) ** 2
            node_mask[s:e] = np.any(d2 <= r2, axis=1)

    for node_i in np.flatnonzero(node_mask):
        cell = grid.cell_of_node.get(int(node_i))
        if cell is not None:
            mask[cell] = True
    return mask


def _endpoint_protection_mask(grid: FMMGrid, start_cell: tuple[int, int], end_cell: tuple[int, int], radius_m: float) -> np.ndarray:
    mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    sr, sc = start_cell
    er, ec = end_cell
    if radius_m <= 0.0:
        mask[sr, sc] = True
        mask[er, ec] = True
        return mask

    xs = grid.unique_x_m[None, :]
    ys = grid.unique_y_m[:, None]
    x0s = float(grid.unique_x_m[sc])
    y0s = float(grid.unique_y_m[sr])
    x0e = float(grid.unique_x_m[ec])
    y0e = float(grid.unique_y_m[er])
    ds = np.hypot(xs - x0s, ys - y0s)
    de = np.hypot(xs - x0e, ys - y0e)
    return (ds <= float(radius_m)) | (de <= float(radius_m))


def _overlap_ratio(path: list[int], old_paths: list[list[int]]) -> float:
    if not old_paths:
        return 0.0
    s = set(path)
    if not s:
        return 1.0
    best = 0.0
    for old in old_paths:
        q = set(old)
        union = len(s | q)
        if union:
            best = max(best, len(s & q) / union)
    return float(best)


# ============================================================
# Public run() required by main.py
# ============================================================


def run(model: pd.DataFrame, graph: dict[str, Any], start_idx: int, end_idx: int, **kwargs) -> dict[str, Any]:
    """Run true grid-based FMM and return a main.py-compatible result."""
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    n = len(model)

    if n <= 0:
        return {"success": False, "path_indices": [], "message": "empty model"}

    verbose = _as_bool(_param(kwargs, "FMM_VERBOSE", kwargs.get("verbose", True)))
    raw_max_paths = _param(kwargs, "FMM_MAX_PATHS", 1)
    max_paths_safety = int(_param(kwargs, "FMM_MAX_PATHS_SAFETY", 50))
    max_paths, auto_mode = _optional_positive_int(raw_max_paths, max_paths_safety)

    mode = str(_param(kwargs, "FMM_MODE", "fastest")).strip().lower()
    if mode == "fastest":
        max_paths = 1
        auto_mode = False

    inf_time = float(_param(kwargs, "FMM_INF_TIME", 1.0e30))
    stop_at_end = _as_bool(_param(kwargs, "FMM_STOP_AT_END", True))
    max_accepted = _param(kwargs, "FMM_MAX_ACCEPTED_CELLS", None)
    if max_accepted is not None:
        max_accepted = int(max_accepted)

    backtrace_mode = str(_param(kwargs, "FMM_BACKTRACE_MODE", "gradient")).strip().lower()
    step_factor = float(_param(kwargs, "FMM_BACKTRACE_STEP_FACTOR", 0.50))
    max_trace_steps = int(_param(kwargs, "FMM_MAX_TRACE_STEPS", max(10_000, 10 * n)))

    max_rounds = int(_param(kwargs, "FMM_MAX_ROUNDS_PER_PATH", 3))
    max_rounds = max(1, max_rounds)
    max_total_attempts_default = max_paths * max_rounds
    if auto_mode:
        max_total_attempts_default = max(max_total_attempts_default, 2 * max_paths)
    max_total_attempts = int(_param(kwargs, "FMM_MAX_TOTAL_ATTEMPTS", max_total_attempts_default))
    max_repeated_attempts = int(_param(kwargs, "FMM_MAX_REPEATED_ATTEMPTS", max_rounds))
    max_overlap = float(_param(kwargs, "FMM_MAX_ALLOWED_NODE_OVERLAP_RATIO", 0.85))

    prev_action = str(_param(kwargs, "FMM_PREVIOUS_PATH_ACTION", "penalty")).strip().lower()
    if max_paths == 1 and not auto_mode:
        prev_action = "none"

    grid = _build_regular_grid(model)
    if start_idx not in grid.cell_of_node:
        return {"success": False, "path_indices": [], "message": "start node is not on FMM grid"}
    if end_idx not in grid.cell_of_node:
        return {"success": False, "path_indices": [], "message": "end node is not on FMM grid"}

    start_cell = grid.cell_of_node[start_idx]
    end_cell = grid.cell_of_node[end_idx]

    base_slowness_grid, base_active_grid, original_nofly_grid, special_grid, meta = _grid_arrays_for_fmm(
        model=model,
        graph=graph,
        grid=grid,
        start_idx=start_idx,
        end_idx=end_idx,
        kwargs=kwargs,
    )

    # In hard mode, start/end can still be rejected if all surrounding cells are blocked.
    special_isolation_rule = _as_bool(_param(kwargs, "SPECIAL_NODE_BLOCK_IF_ALL_8_NEIGHBORS_NOFLY", True))
    if special_isolation_rule:
        for label, cell in (("start", start_cell), ("end", end_cell)):
            r, c = cell
            n_active = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < grid.ny and 0 <= cc < grid.nx and base_active_grid[rr, cc]:
                        n_active += 1
            if n_active == 0:
                return {
                    "success": False,
                    "algorithm": "FMM",
                    "path_indices": [],
                    "path_results": [],
                    "ranked_paths": [],
                    "total_cost": float("inf"),
                    "k_paths_found": 0,
                    "expanded_states": 0,
                    "message": f"FMM {label} node is isolated by no-fly/blocked neighbours.",
                    "fmm_grid_shape": [int(grid.ny), int(grid.nx)],
                    "fmm_meta": meta,
                }

    if verbose:
        print("      TRUE FMM2D / EIKONAL MODE:")
        print(f"        module           : src/FMM.py")
        print(f"        equation         : |grad T| = slowness")
        print(f"        grid shape       : ny={grid.ny:,}, nx={grid.nx:,}")
        print(f"        dx, dy           : {grid.dx_m:.2f} m, {grid.dy_m:.2f} m")
        print(f"        no-fly mode      : {meta['nofly_mode']}")
        if meta["nofly_mode"] != "hard":
            print(f"        no-fly slowness  : {meta['high_slowness']:.6g} s/m")
        print(f"        active cells     : {meta['active_nodes']:,}")
        print(f"        max paths        : {'auto/None -> ' + str(max_paths) + ' safety cap' if auto_mode else max_paths}")
        print(f"        previous action  : {prev_action}")
        print(f"        backtrace mode   : {backtrace_mode}")

    accepted_paths: list[list[int]] = []
    path_results: list[dict[str, Any]] = []
    total_accepted = 0
    total_attempts = 0
    repeated_attempts = 0
    stop_reason = ""

    slowness_grid = base_slowness_grid.copy()
    active_grid = base_active_grid.copy()

    rank = 1
    while rank <= max_paths and total_attempts < max_total_attempts:
        accepted_this_rank = False

        for round_id in range(1, max_rounds + 1):
            if total_attempts >= max_total_attempts:
                stop_reason = "max_total_attempts"
                break
            total_attempts += 1

            T, status, solve_status, accepted_count = _fmm_solve(
                slowness=slowness_grid,
                active=active_grid,
                start_cell=start_cell,
                end_cell=end_cell,
                dx=grid.dx_m,
                dy=grid.dy_m,
                inf_time=inf_time,
                stop_at_end=stop_at_end,
                max_accepted=max_accepted,
                verbose=verbose,
            )
            total_accepted += int(accepted_count)

            if solve_status != "ok":
                stop_reason = solve_status
                break

            if backtrace_mode in ("gradient", "ray", "raypath"):
                path = _trace_ray_gradient(
                    T=T,
                    active=active_grid,
                    grid=grid,
                    start_cell=start_cell,
                    end_cell=end_cell,
                    step_factor=step_factor,
                    max_steps=max_trace_steps,
                )
            elif backtrace_mode in ("steepest", "neighbor", "neighbour"):
                path = _trace_ray_steepest_neighbor(
                    T=T,
                    active=active_grid,
                    grid=grid,
                    start_cell=start_cell,
                    end_cell=end_cell,
                    max_steps=max_trace_steps,
                )
            else:
                raise ValueError("FMM_BACKTRACE_MODE must be 'gradient' or 'steepest'.")

            if not path or len(path) < 2:
                stop_reason = "ray_trace_failed"
                break

            overlap = _overlap_ratio(path, accepted_paths)
            if accepted_paths and overlap > max_overlap:
                repeated_attempts += 1
                if verbose:
                    print(
                        f"      FMM rank {rank:03d}, round {round_id}: "
                        f"skip overlap={overlap:.3f}, repeated={repeated_attempts}/{max_repeated_attempts}"
                    )
                if repeated_attempts >= max_repeated_attempts:
                    stop_reason = "max_repeated_attempts"
                    break
            else:
                distance_m = _path_distance_m(grid, path)
                travel_time_s = float(T[end_cell])
                threshold = float(meta["threshold"])
                nofly_count = _path_nofly_count(path, model, threshold)

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
                    "expanded_states": int(accepted_count),
                    "accepted_cells": int(accepted_count),
                    "round_id": int(round_id),
                    "status": "ok",
                    "nofly_nodes_on_path": int(nofly_count),
                }
                path_results.append(item)
                accepted_paths.append(path)
                repeated_attempts = 0

                if verbose:
                    print(
                        f"      FMM path {rank:03d}: nodes={len(path):,}, "
                        f"distance={distance_m/1000.0:.4f} km, "
                        f"arrival={travel_time_s:.2f} s, overlap={overlap:.3f}, "
                        f"nofly_nodes={nofly_count}"
                    )

                accepted_this_rank = True
                rank += 1

            # Update path penalty/block after accepted path or too-similar path.
            # This enables multiple FMM alternatives without pretending these are
            # exact mathematical "all paths".
            if prev_action not in ("none", "allow") and path:
                buffer_m = float(_param(kwargs, "FMM_PATH_BUFFER_M", _param(kwargs, "PATH_BUFFER_M", 150.0)))
                endpoint_radius_m = float(_param(kwargs, "FMM_ENDPOINT_PROTECTION_RADIUS_M", max(buffer_m, 150.0)))
                penalty_factor = float(_param(kwargs, "FMM_PREVIOUS_PATH_PENALTY_FACTOR", 5.0))
                max_slow = float(_param(kwargs, "FMM_MAX_SLOWNESS", 1.0e12))
                mask = _path_buffer_mask(grid, path, buffer_m)
                mask &= ~_endpoint_protection_mask(grid, start_cell, end_cell, endpoint_radius_m)
                if prev_action in ("penalty", "penalize", "soft"):
                    slowness_grid[mask] = np.minimum(slowness_grid[mask] * penalty_factor, max_slow)
                elif prev_action in ("block", "hard_block", "non_overlap"):
                    active_grid[mask] = False
                    active_grid[start_cell] = True
                    active_grid[end_cell] = True
                else:
                    raise ValueError("FMM_PREVIOUS_PATH_ACTION must be 'none', 'penalty', or 'block'.")

            if accepted_this_rank:
                break

        if not accepted_this_rank:
            if not stop_reason:
                stop_reason = "no_more_paths"
            break

    if not path_results:
        return {
            "success": False,
            "algorithm": "FMM",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": float("inf"),
            "travel_cost": float("inf"),
            "k_paths_requested": None if auto_mode else int(max_paths),
            "k_paths_found": 0,
            "expanded_states": int(total_accepted),
            "message": f"FMM failed: {stop_reason or 'no path found'}",
            "fmm_grid_shape": [int(grid.ny), int(grid.nx)],
            "fmm_meta": meta,
        }

    best = path_results[0]
    return {
        "success": True,
        "algorithm": "FMM",
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
        "k_paths_requested": None if auto_mode else int(max_paths),
        "k_paths_safety_cap": int(max_paths),
        "k_paths_found": int(len(path_results)),
        "expanded_states": int(total_accepted),
        "message": "ok" if len(path_results) >= max_paths else f"stopped: {stop_reason or 'done'}",
        "fmm_mode": str(mode),
        "fmm_nofly_mode": str(meta["nofly_mode"]),
        "fmm_auto_until_exhausted": bool(auto_mode),
        "fmm_total_attempts": int(total_attempts),
        "fmm_max_total_attempts": int(max_total_attempts),
        "fmm_grid_shape": [int(grid.ny), int(grid.nx)],
        "fmm_dx_m": float(grid.dx_m),
        "fmm_dy_m": float(grid.dy_m),
        "fmm_backtrace_mode": str(backtrace_mode),
        "fmm_previous_path_action": str(prev_action),
        "fmm_meta": meta,
        "nofly_nodes_on_best_path": int(best.get("nofly_nodes_on_path", 0)),
    }
