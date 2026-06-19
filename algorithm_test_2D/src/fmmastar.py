#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/fmmastar.py

Graph/KDTree FMM-clearance A* for the LAE-UTM main.py protocol.

Use in parameters.py:
    ALGORITHM = ["fmmastar"]

This version does NOT require a compact regular grid.  It uses the graph already
built by main.py when possible, and falls back to a lightweight KDTree neighbor
search over graph["valid_indices"] when the graph adjacency structure is not
available.

Concept:
    1. main.py already decides flyable nodes using the official slowness rule:
           slowness < 10.0   -> flyable
           slowness >= 10.0  -> no-fly
    2. fmmastar estimates clearance_m = Euclidean distance to nearest no-fly node.
    3. A* minimizes travel-time cost plus a clearance penalty, so paths prefer
       wider/safer corridors.

Required public function:
    run(model, graph, start_idx, end_idx, **kwargs) -> dict
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Parameter handling
# ============================================================


def _get_param(name: str, default: Any = None) -> Any:
    """Read shared or params/fmmastar.params values from parameters.py."""
    try:
        import parameters as prm  # type: ignore
        return getattr(prm, name, default)
    except Exception:
        return default


@dataclass
class FmmAstarConfig:
    no_fly_threshold: float = 10.0
    threshold_mode: str = "greater_equal"
    use_graph_valid_indices: bool = True
    force_start_end_flyable: bool = True

    # Neighbor construction
    neighbor_source: str = "graph_or_kdtree"  # graph_or_kdtree, graph, kdtree
    connectivity: int = 8
    use_z_distance: bool = False
    kdtree_radius_factor: float = 1.60
    kdtree_max_neighbors_2d: int = 8
    kdtree_max_neighbors_3d: int = 26

    # A* search
    max_expansions: Optional[int] = None
    heuristic_weight: float = 1.0

    # Clearance penalty
    clearance_method: str = "nearest_blocked_kdtree"
    safe_clearance_m: float = 300.0
    clearance_weight: float = 2.0
    clearance_power: float = 2.0
    min_clearance_to_enter_m: float = 0.0

    # Optional post smoothing.  Off by default because main.py expects node-by-node path.
    smooth_path: bool = False
    smooth_require_clearance_m: float = 0.0

    verbose: bool = True


def _build_config(kwargs: Dict[str, Any]) -> FmmAstarConfig:
    max_expansions = _get_param("FMM_MAX_EXPANSIONS", None)
    if max_expansions is None:
        max_expansions = kwargs.get("max_expansions", _get_param("MULTI_PATH_MAX_EXPANSIONS", None))

    heuristic_weight = _get_param("FMM_HEURISTIC_WEIGHT", None)
    if heuristic_weight is None:
        heuristic_weight = kwargs.get("heuristic_weight", _get_param("MULTI_PATH_HEURISTIC_WEIGHT", 1.0))

    verbose = _get_param("FMM_VERBOSE", None)
    if verbose is None:
        verbose = kwargs.get("verbose", _get_param("MULTI_PATH_VERBOSE", True))

    return FmmAstarConfig(
        no_fly_threshold=float(_get_param("FMM_NO_FLY_THRESHOLD", _get_param("NO_FLY_SLOWNESS_THRESHOLD", 10.0))),
        threshold_mode=str(_get_param("FMM_THRESHOLD_MODE", _get_param("NO_FLY_THRESHOLD_MODE", "greater_equal"))).strip().lower(),
        use_graph_valid_indices=bool(_get_param("FMM_USE_GRAPH_VALID_INDICES", True)),
        force_start_end_flyable=bool(_get_param("FMM_FORCE_START_END_FLYABLE", _get_param("FORCE_SEARCH_START_END_FLYABLE", True))),
        neighbor_source=str(_get_param("FMM_NEIGHBOR_SOURCE", "graph_or_kdtree")).strip().lower(),
        connectivity=int(_get_param("FMM_CONNECTIVITY", _get_param("CONNECTIVITY_2D", 8))),
        use_z_distance=bool(_get_param("FMM_USE_Z_DISTANCE", False)),
        kdtree_radius_factor=float(_get_param("FMM_KDTREE_RADIUS_FACTOR", _get_param("KDTREE_RADIUS_FACTOR", 1.60))),
        kdtree_max_neighbors_2d=int(_get_param("FMM_KDTREE_MAX_NEIGHBORS_2D", _get_param("KDTREE_MAX_NEIGHBORS_2D", 8))),
        kdtree_max_neighbors_3d=int(_get_param("FMM_KDTREE_MAX_NEIGHBORS_3D", _get_param("KDTREE_MAX_NEIGHBORS_3D", 26))),
        max_expansions=None if max_expansions is None else int(max_expansions),
        heuristic_weight=float(heuristic_weight),
        clearance_method=str(_get_param("FMM_CLEARANCE_METHOD", "nearest_blocked_kdtree")).strip().lower(),
        safe_clearance_m=float(_get_param("FMM_SAFE_CLEARANCE_M", 300.0)),
        clearance_weight=float(_get_param("FMM_CLEARANCE_WEIGHT", 2.0)),
        clearance_power=float(_get_param("FMM_CLEARANCE_POWER", 2.0)),
        min_clearance_to_enter_m=float(_get_param("FMM_MIN_CLEARANCE_TO_ENTER_M", 0.0)),
        smooth_path=bool(_get_param("FMM_SMOOTH_PATH", False)),
        smooth_require_clearance_m=float(_get_param("FMM_SMOOTH_REQUIRE_CLEARANCE_M", 0.0)),
        verbose=bool(verbose),
    )


# ============================================================
# Coordinates / masks
# ============================================================


def _xy_columns(model: pd.DataFrame) -> Tuple[str, str]:
    if {"x", "y"}.issubset(model.columns):
        return "x", "y"
    if {"lon", "lat"}.issubset(model.columns):
        return "lon", "lat"
    raise ValueError("fmmastar requires model columns x/y or lon/lat.")


def _looks_lonlat(model: pd.DataFrame, xcol: str, ycol: str) -> bool:
    try:
        x = pd.to_numeric(model[xcol], errors="coerce")
        y = pd.to_numeric(model[ycol], errors="coerce")
        return bool(
            x.dropna().between(-180.0, 180.0).all()
            and y.dropna().between(-90.0, 90.0).all()
        )
    except Exception:
        return False


def _coords_m(model: pd.DataFrame, use_z: bool = False) -> np.ndarray:
    """Return metric coordinates for distance calculations."""
    xcol, ycol = _xy_columns(model)
    x = pd.to_numeric(model[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(model[ycol], errors="coerce").to_numpy(dtype=float)

    if _looks_lonlat(model, xcol, ycol):
        finite_y = y[np.isfinite(y)]
        lat0 = float(np.nanmean(finite_y)) if len(finite_y) else 0.0
        lon0 = float(np.nanmean(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
        x_m = (x - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        y_m = (y - lat0) * 110_540.0
    else:
        x_m = x
        y_m = y

    if use_z and "z" in model.columns:
        z = pd.to_numeric(model["z"], errors="coerce").to_numpy(dtype=float)
        z = np.where(np.isfinite(z), z, 0.0)
        return np.column_stack([x_m, y_m, z])

    return np.column_stack([x_m, y_m])


def _threshold_blocked(model: pd.DataFrame, cfg: FmmAstarConfig) -> np.ndarray:
    if "slowness" not in model.columns:
        raise ValueError("fmmastar requires a 'slowness' column.")
    slow = pd.to_numeric(model["slowness"], errors="coerce").to_numpy(dtype=float)
    mode = cfg.threshold_mode
    if mode in ("greater_equal", "ge", ">=", "threshold"):
        blocked = slow >= cfg.no_fly_threshold
    elif mode in ("greater", "gt", ">"):
        blocked = slow > cfg.no_fly_threshold
    elif mode in ("less_equal", "le", "<="):
        blocked = slow <= cfg.no_fly_threshold
    elif mode in ("less", "lt", "<"):
        blocked = slow < cfg.no_fly_threshold
    else:
        raise ValueError("Unsupported FMM_THRESHOLD_MODE. Use 'greater_equal' for this model.")
    return np.asarray(blocked | (~np.isfinite(slow)), dtype=bool)


def _valid_index_set(model: pd.DataFrame, graph: Optional[Dict[str, Any]], cfg: FmmAstarConfig) -> set[int]:
    if cfg.use_graph_valid_indices and isinstance(graph, dict) and "valid_indices" in graph:
        try:
            valid = {int(v) for v in graph.get("valid_indices", set())}
            if valid:
                return valid
        except Exception:
            pass

    blocked = _threshold_blocked(model, cfg)
    return {int(idx) for idx, is_blocked in zip(model.index.tolist(), blocked) if not bool(is_blocked)}


def _index_position_map(model: pd.DataFrame) -> Dict[int, int]:
    return {int(idx): pos for pos, idx in enumerate(model.index.tolist())}


# ============================================================
# Neighbor provider
# ============================================================


class NeighborProvider:
    """Generic graph-neighbor interface with KDTree fallback."""

    def __init__(
        self,
        model: pd.DataFrame,
        graph: Optional[Dict[str, Any]],
        valid_indices: Sequence[int],
        coords: np.ndarray,
        cfg: FmmAstarConfig,
    ) -> None:
        self.model = model
        self.graph = graph if isinstance(graph, dict) else {}
        self.valid = {int(v) for v in valid_indices}
        self.coords = coords
        self.cfg = cfg
        self.pos_by_idx = _index_position_map(model)
        self.valid_sorted = np.array(sorted(self.valid), dtype=int)
        self.valid_positions = np.array([self.pos_by_idx[int(i)] for i in self.valid_sorted if int(i) in self.pos_by_idx], dtype=int)
        self.valid_sorted = np.array([int(i) for i in self.valid_sorted if int(i) in self.pos_by_idx], dtype=int)
        self.valid_lookup = {int(idx): k for k, idx in enumerate(self.valid_sorted.tolist())}
        self._tree = None
        self._valid_coords = None
        self._radius_m = None
        self._max_neighbors = None
        self._graph_adj = self._extract_graph_adjacency()

        if self.cfg.neighbor_source in ("kdtree", "graph_or_kdtree") and self._graph_adj is None:
            self._build_tree()
        elif self.cfg.neighbor_source == "kdtree":
            self._graph_adj = None
            self._build_tree()

    def _extract_graph_adjacency(self) -> Optional[Dict[int, Any]]:
        if self.cfg.neighbor_source == "kdtree":
            return None
        if not self.graph:
            return None

        # Common dictionary names used by graph builders.
        for key in (
            "adjacency", "neighbors", "neighbor_indices", "edges", "graph", "adj",
        ):
            obj = self.graph.get(key, None)
            if isinstance(obj, dict) and obj:
                return obj

        # NetworkX-like graph object.
        obj = self.graph.get("G", None)
        if obj is not None and hasattr(obj, "neighbors"):
            return {int(n): list(obj.neighbors(n)) for n in obj.nodes}

        return None

    def _infer_spacing_m(self) -> float:
        if "grid_spacing_m" in self.graph:
            try:
                value = float(self.graph.get("grid_spacing_m"))
                if value > 0:
                    return value
            except Exception:
                pass

        if len(self.valid_positions) < 2:
            return 1.0

        # Use a small sample to infer nearest-neighbor spacing.
        try:
            from scipy.spatial import cKDTree
            sample_pos = self.valid_positions[: min(len(self.valid_positions), 5000)]
            sample = self.coords[sample_pos]
            sample = sample[np.all(np.isfinite(sample), axis=1)]
            if len(sample) >= 2:
                tree = cKDTree(sample)
                dist, _ = tree.query(sample, k=2)
                nn = dist[:, 1]
                nn = nn[np.isfinite(nn) & (nn > 0)]
                if len(nn):
                    return float(np.median(nn))
        except Exception:
            pass

        return 50.0

    def _build_tree(self) -> None:
        try:
            from scipy.spatial import cKDTree
        except Exception as exc:
            raise ImportError(
                "fmmastar KDTree fallback requires scipy. Install scipy or provide graph adjacency."
            ) from exc

        valid_coords = self.coords[self.valid_positions]
        finite = np.all(np.isfinite(valid_coords), axis=1)
        self.valid_positions = self.valid_positions[finite]
        self.valid_sorted = self.valid_sorted[finite]
        self.valid_lookup = {int(idx): k for k, idx in enumerate(self.valid_sorted.tolist())}
        self._valid_coords = self.coords[self.valid_positions]
        self._tree = cKDTree(self._valid_coords)

        if "neighbor_radius_m" in self.graph:
            try:
                radius = float(self.graph.get("neighbor_radius_m"))
            except Exception:
                radius = 0.0
        else:
            radius = 0.0

        if not np.isfinite(radius) or radius <= 0:
            radius = self._infer_spacing_m() * float(self.cfg.kdtree_radius_factor)

        self._radius_m = float(radius)
        if self.coords.shape[1] >= 3:
            self._max_neighbors = int(self.cfg.kdtree_max_neighbors_3d)
        else:
            self._max_neighbors = int(self.cfg.kdtree_max_neighbors_2d)
        self._max_neighbors = max(1, self._max_neighbors)

    def _normalize_neighbor_item(self, item: Any) -> Optional[int]:
        # Supported examples:
        #   neighbor_idx
        #   (neighbor_idx, cost)
        #   {"to": neighbor_idx}
        #   {"idx": neighbor_idx}
        if isinstance(item, dict):
            for key in ("to", "target", "neighbor", "idx", "index", "node"):
                if key in item:
                    try:
                        return int(item[key])
                    except Exception:
                        return None
            return None

        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                return None
            try:
                return int(item[0])
            except Exception:
                return None

        try:
            return int(item)
        except Exception:
            return None

    def neighbors(self, idx: int) -> List[int]:
        idx = int(idx)
        if idx not in self.valid:
            return []

        if self._graph_adj is not None:
            raw = self._graph_adj.get(idx, None)
            if raw is None:
                raw = self._graph_adj.get(str(idx), None)
            if raw is not None:
                out = []
                for item in raw:
                    nb = self._normalize_neighbor_item(item)
                    if nb is not None and nb in self.valid and nb != idx:
                        out.append(int(nb))
                if out:
                    return out
                if self.cfg.neighbor_source == "graph":
                    return []

        # KDTree fallback or fallback when graph adjacency for this node is empty.
        if self._tree is None:
            self._build_tree()
        assert self._tree is not None
        assert self._valid_coords is not None
        assert self._radius_m is not None
        assert self._max_neighbors is not None

        kpos = self.valid_lookup.get(idx, None)
        if kpos is None:
            return []
        query_point = self._valid_coords[kpos]

        # query_ball_point gives exact radius neighbors.  Limit after sorting by distance.
        candidates = self._tree.query_ball_point(query_point, r=self._radius_m)
        if not candidates:
            return []

        neigh = []
        for j in candidates:
            nb = int(self.valid_sorted[int(j)])
            if nb == idx:
                continue
            d = float(np.linalg.norm(self.coords[self.pos_by_idx[nb]] - self.coords[self.pos_by_idx[idx]]))
            if np.isfinite(d) and d > 0:
                neigh.append((d, nb))
        neigh.sort(key=lambda x: x[0])
        return [nb for _, nb in neigh[: self._max_neighbors]]

    def distance(self, idx_a: int, idx_b: int) -> float:
        pa = self.pos_by_idx.get(int(idx_a))
        pb = self.pos_by_idx.get(int(idx_b))
        if pa is None or pb is None:
            return math.inf
        d = float(np.linalg.norm(self.coords[pa] - self.coords[pb]))
        return d if np.isfinite(d) else math.inf


# ============================================================
# Clearance
# ============================================================


def _compute_clearance_m(
    model: pd.DataFrame,
    coords: np.ndarray,
    valid_indices: Sequence[int],
    start_idx: int,
    end_idx: int,
    cfg: FmmAstarConfig,
) -> Dict[int, float]:
    """Nearest no-fly/blocked-node clearance for each valid index."""
    blocked = _threshold_blocked(model, cfg)
    pos_by_idx = _index_position_map(model)
    valid_set = {int(v) for v in valid_indices}

    # main.py can force DB/DK/FLZ endpoints flyable.  Keep them valid, but do
    # not let their own high slowness make their clearance zero.
    for idx in (int(start_idx), int(end_idx)):
        pos = pos_by_idx.get(idx)
        if pos is not None and cfg.force_start_end_flyable:
            blocked[pos] = False

    blocked_positions = np.flatnonzero(blocked)
    blocked_coords = coords[blocked_positions]
    finite_blocked = np.all(np.isfinite(blocked_coords), axis=1)
    blocked_coords = blocked_coords[finite_blocked]

    clearance: Dict[int, float] = {}

    if len(blocked_coords) == 0:
        for idx in valid_set:
            clearance[int(idx)] = math.inf
        return clearance

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(blocked_coords)
        query_positions = np.array([pos_by_idx[int(i)] for i in valid_set if int(i) in pos_by_idx], dtype=int)
        query_indices = [int(i) for i in valid_set if int(i) in pos_by_idx]
        query_coords = coords[query_positions]
        finite_query = np.all(np.isfinite(query_coords), axis=1)
        dists = np.full(len(query_indices), math.inf, dtype=float)
        if np.any(finite_query):
            d, _ = tree.query(query_coords[finite_query], k=1)
            dists[finite_query] = d
        for idx, d in zip(query_indices, dists):
            clearance[int(idx)] = float(d)
    except Exception:
        # Safe fallback, slower but okay for small tests.
        for idx in valid_set:
            pos = pos_by_idx.get(int(idx))
            if pos is None or not np.all(np.isfinite(coords[pos])):
                clearance[int(idx)] = math.inf
                continue
            d = np.linalg.norm(blocked_coords - coords[pos], axis=1)
            clearance[int(idx)] = float(np.nanmin(d)) if len(d) else math.inf

    return clearance


def _clearance_penalty_value(clearance_m: float, cfg: FmmAstarConfig) -> float:
    if cfg.safe_clearance_m <= 0.0 or cfg.clearance_weight <= 0.0:
        return 0.0
    if not np.isfinite(clearance_m):
        return 0.0
    p = (cfg.safe_clearance_m - float(clearance_m)) / cfg.safe_clearance_m
    p = max(0.0, min(1.0, p))
    if cfg.clearance_power != 1.0:
        p = p ** cfg.clearance_power
    return float(p)


# ============================================================
# A* core
# ============================================================


def _reconstruct(parent: Dict[int, int], start_idx: int, end_idx: int) -> List[int]:
    path = []
    cur = int(end_idx)
    while True:
        path.append(cur)
        if cur == int(start_idx):
            break
        if cur not in parent:
            return []
        cur = int(parent[cur])
    path.reverse()
    return path


def _path_distance_time(model: pd.DataFrame, provider: NeighborProvider, path: Sequence[int]) -> Tuple[float, float]:
    if len(path) < 2:
        return 0.0, 0.0
    slow = pd.to_numeric(model["slowness"], errors="coerce")
    dist = 0.0
    travel = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = provider.distance(int(a), int(b))
        if not np.isfinite(d):
            continue
        try:
            sa = float(slow.loc[int(a)])
            sb = float(slow.loc[int(b)])
        except Exception:
            sa = sb = 0.0
        avg_slow = 0.5 * (sa + sb) if np.isfinite(sa) and np.isfinite(sb) else 0.0
        dist += d
        travel += avg_slow * d
    return float(dist), float(travel)


def _astar(
    model: pd.DataFrame,
    coords: np.ndarray,
    provider: NeighborProvider,
    valid_indices: Sequence[int],
    clearance_m: Dict[int, float],
    start_idx: int,
    end_idx: int,
    cfg: FmmAstarConfig,
) -> Dict[str, Any]:
    valid_set = {int(v) for v in valid_indices}
    if start_idx not in valid_set:
        return {"success": False, "message": f"start_idx {start_idx} not in valid graph", "path_indices": []}
    if end_idx not in valid_set:
        return {"success": False, "message": f"end_idx {end_idx} not in valid graph", "path_indices": []}

    pos_by_idx = provider.pos_by_idx
    slow_series = pd.to_numeric(model["slowness"], errors="coerce")
    fly_slow = []
    for idx in valid_set:
        try:
            s = float(slow_series.loc[idx])
            if np.isfinite(s):
                fly_slow.append(s)
        except Exception:
            pass
    min_slow = max(min(fly_slow) if fly_slow else 1.0e-9, 1.0e-12)

    def heuristic(idx: int) -> float:
        pa = pos_by_idx.get(int(idx))
        pb = pos_by_idx.get(int(end_idx))
        if pa is None or pb is None:
            return 0.0
        d = float(np.linalg.norm(coords[pa] - coords[pb]))
        return d * min_slow if np.isfinite(d) else 0.0

    g_cost: Dict[int, float] = {int(start_idx): 0.0}
    travel_cost: Dict[int, float] = {int(start_idx): 0.0}
    distance_cost: Dict[int, float] = {int(start_idx): 0.0}
    parent: Dict[int, int] = {}
    closed: set[int] = set()

    heap: List[Tuple[float, float, int, int]] = []
    counter = 0
    h0 = heuristic(start_idx)
    heapq.heappush(heap, (float(cfg.heuristic_weight) * h0, h0, counter, int(start_idx)))

    expanded = 0
    while heap:
        _, _, _, current = heapq.heappop(heap)
        current = int(current)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == int(end_idx):
            path = _reconstruct(parent, start_idx, end_idx)
            dist_m = distance_cost.get(end_idx, 0.0)
            travel_s = travel_cost.get(end_idx, 0.0)
            return {
                "success": True,
                "message": "ok",
                "path_indices": path,
                "total_cost": float(g_cost[end_idx]),
                "travel_cost": float(travel_s),
                "clearance_cost": float(g_cost[end_idx] - travel_s),
                "distance_m": float(dist_m),
                "expanded_states": int(expanded),
            }

        if cfg.max_expansions is not None and expanded >= cfg.max_expansions:
            return {
                "success": False,
                "message": "FMM_MAX_EXPANSIONS reached",
                "path_indices": [],
                "expanded_states": int(expanded),
            }

        for nb in provider.neighbors(current):
            nb = int(nb)
            if nb in closed:
                continue
            if nb not in valid_set:
                continue
            if cfg.min_clearance_to_enter_m > 0.0 and nb not in (start_idx, end_idx):
                if clearance_m.get(nb, math.inf) < cfg.min_clearance_to_enter_m:
                    continue

            d = provider.distance(current, nb)
            if not np.isfinite(d) or d <= 0.0:
                continue

            try:
                sa = float(slow_series.loc[current])
                sb = float(slow_series.loc[nb])
            except Exception:
                continue
            if not np.isfinite(sa) or not np.isfinite(sb):
                continue

            base_time = 0.5 * (sa + sb) * d
            p_cur = _clearance_penalty_value(clearance_m.get(current, math.inf), cfg)
            p_nb = _clearance_penalty_value(clearance_m.get(nb, math.inf), cfg)
            avg_penalty = 0.5 * (p_cur + p_nb)
            edge_cost = base_time * (1.0 + float(cfg.clearance_weight) * avg_penalty)

            new_g = g_cost[current] + edge_cost
            if new_g < g_cost.get(nb, math.inf):
                g_cost[nb] = float(new_g)
                travel_cost[nb] = float(travel_cost[current] + base_time)
                distance_cost[nb] = float(distance_cost[current] + d)
                parent[nb] = current
                h = heuristic(nb)
                counter += 1
                heapq.heappush(heap, (new_g + float(cfg.heuristic_weight) * h, h, counter, nb))

    return {
        "success": False,
        "message": "no path found",
        "path_indices": [],
        "expanded_states": int(expanded),
    }


# ============================================================
# Optional simple line smoothing over graph path
# ============================================================


def _smooth_by_skipping(
    model: pd.DataFrame,
    provider: NeighborProvider,
    path: List[int],
    clearance_m: Dict[int, float],
    cfg: FmmAstarConfig,
) -> List[int]:
    """Conservative smoothing: only remove B from A-B-C when A and C are direct neighbors."""
    if not cfg.smooth_path or len(path) <= 2:
        return path

    neighbor_cache: Dict[int, set[int]] = {}

    def direct(a: int, c: int) -> bool:
        if a not in neighbor_cache:
            neighbor_cache[a] = set(provider.neighbors(a))
        if c not in neighbor_cache[a]:
            return False
        if cfg.smooth_require_clearance_m > 0.0:
            if clearance_m.get(c, math.inf) < cfg.smooth_require_clearance_m:
                return False
        return True

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        if i + 2 < len(path) and direct(path[i], path[i + 2]):
            out.append(path[i + 2])
            i += 2
        else:
            out.append(path[i + 1])
            i += 1
    return out


# ============================================================
# Public main.py protocol
# ============================================================


def run(
    model: pd.DataFrame,
    graph: Optional[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run fmmastar using the protocol expected by main.py."""
    t0 = time.perf_counter()
    cfg = _build_config(kwargs)
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    coords = _coords_m(model, use_z=cfg.use_z_distance)
    valid = _valid_index_set(model, graph, cfg)

    if cfg.force_start_end_flyable:
        valid.add(start_idx)
        valid.add(end_idx)

    provider = NeighborProvider(
        model=model,
        graph=graph,
        valid_indices=sorted(valid),
        coords=coords,
        cfg=cfg,
    )

    # If some valid nodes had non-finite coordinates, NeighborProvider removes
    # them from KDTree. Keep start/end if possible; otherwise report clearly.
    valid = set(provider.valid_sorted.tolist()) if len(provider.valid_sorted) else set(valid)
    if cfg.force_start_end_flyable:
        # Keep endpoints in valid only if they have finite coords.
        pos_by_idx = _index_position_map(model)
        for idx in (start_idx, end_idx):
            pos = pos_by_idx.get(idx)
            if pos is not None and np.all(np.isfinite(coords[pos])):
                valid.add(idx)

    clearance = _compute_clearance_m(
        model=model,
        coords=coords,
        valid_indices=sorted(valid),
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
    )

    if cfg.verbose:
        n_valid = len(valid)
        threshold_blocked = _threshold_blocked(model, cfg)
        n_blocked = int(np.count_nonzero(threshold_blocked))
        graph_adj_available = provider._graph_adj is not None
        print("\n========== FMMASTAR ==========")
        print("Mode                   : graph/KDTree nodal FMM-A*")
        print(f"Valid/search nodes     : {n_valid:,}")
        print(f"Blocked/no-fly nodes   : {n_blocked:,}")
        print(f"Start idx              : {start_idx}")
        print(f"End idx                : {end_idx}")
        print(f"Neighbor source        : {cfg.neighbor_source}")
        print(f"Graph adjacency found  : {graph_adj_available}")
        if provider._radius_m is not None:
            print(f"KDTree radius          : {provider._radius_m:.2f} m")
        print(f"Connectivity/max nb    : {cfg.connectivity} / {provider._max_neighbors}")
        print(f"Clearance method       : {cfg.clearance_method}")
        print(f"Safe clearance         : {cfg.safe_clearance_m:.1f} m")
        print(f"Clearance weight       : {cfg.clearance_weight}")

    search = _astar(
        model=model,
        coords=coords,
        provider=provider,
        valid_indices=sorted(valid),
        clearance_m=clearance,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
    )

    path_indices = [int(i) for i in search.get("path_indices", [])]
    raw_nodes = len(path_indices)
    if search.get("success") and cfg.smooth_path:
        path_indices = _smooth_by_skipping(model, provider, path_indices, clearance, cfg)
        if len(path_indices) != raw_nodes:
            dist_m, travel_s = _path_distance_time(model, provider, path_indices)
            search["distance_m"] = dist_m
            search["travel_cost"] = travel_s

    runtime_s = time.perf_counter() - t0

    if path_indices:
        path_clearance = np.array([clearance.get(int(i), math.inf) for i in path_indices], dtype=float)
        finite_clearance = path_clearance[np.isfinite(path_clearance)]
        min_clearance = float(np.min(finite_clearance)) if len(finite_clearance) else math.inf
        mean_clearance = float(np.mean(finite_clearance)) if len(finite_clearance) else math.inf
    else:
        min_clearance = math.nan
        mean_clearance = math.nan

    distance_m = float(search.get("distance_m", 0.0))
    travel_s = float(search.get("travel_cost", math.inf))
    total_cost = float(search.get("total_cost", math.inf))

    result: Dict[str, Any] = {
        "success": bool(search.get("success", False)),
        "message": str(search.get("message", "")),
        "algorithm": "fmmastar",
        "path_indices": path_indices,
        "total_cost": total_cost,
        "travel_cost": travel_s,
        "clearance_cost": float(search.get("clearance_cost", total_cost - travel_s if np.isfinite(total_cost) and np.isfinite(travel_s) else 0.0)),
        "distance_m": distance_m,
        "distance_km": distance_m / 1000.0,
        "estimated_traveltime_s": travel_s,
        "estimated_traveltime_min": travel_s / 60.0 if np.isfinite(travel_s) else math.inf,
        "nodes": int(len(path_indices)),
        "expanded_states": int(search.get("expanded_states", 0)),
        "runtime_s": float(runtime_s),
        "raw_path_nodes_before_smoothing": int(raw_nodes),
        "smoothed_path": bool(cfg.smooth_path and len(path_indices) < raw_nodes),
        "min_clearance_on_path_m": min_clearance,
        "mean_clearance_on_path_m": mean_clearance,
        "fmmastar_mode": "graph_kdtree_nodal",
        "fmmastar_neighbor_source": str(cfg.neighbor_source),
        "fmmastar_graph_adjacency_found": bool(provider._graph_adj is not None),
        "fmmastar_kdtree_radius_m": None if provider._radius_m is None else float(provider._radius_m),
        "fmmastar_valid_node_count": int(len(valid)),
        "fmmastar_clearance_method": str(cfg.clearance_method),
        "fmmastar_safe_clearance_m": float(cfg.safe_clearance_m),
        "fmmastar_clearance_weight": float(cfg.clearance_weight),
        "fmmastar_clearance_power": float(cfg.clearance_power),
        "fmmastar_min_clearance_to_enter_m": float(cfg.min_clearance_to_enter_m),
        "fmmastar_connectivity": int(cfg.connectivity),
        "fmmastar_heuristic_weight": float(cfg.heuristic_weight),
        "fmmastar_uses_graph_valid_indices": bool(cfg.use_graph_valid_indices),
    }

    if cfg.verbose:
        print("---------- FMMASTAR result ----------")
        print(f"Success                : {result['success']}")
        print(f"Message                : {result['message']}")
        print(f"Expanded states        : {result['expanded_states']:,}")
        print(f"Path nodes             : {result['nodes']:,}")
        print(f"Distance               : {result['distance_m']:.2f} m")
        print(f"Travel time            : {result['estimated_traveltime_s']:.2f} s")
        print(f"Total cost             : {result['total_cost']:.6g}")
        print(f"Min clearance on path  : {result['min_clearance_on_path_m']:.2f} m")
        print(f"Mean clearance on path : {result['mean_clearance_on_path_m']:.2f} m")
        print(f"Runtime                : {result['runtime_s']:.3f} s")

    return result


# Aliases for manual testing/import convenience.
fmmastar = run
find_path = run
run_algorithm = run
