#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Top-K path finder with turn minimization.

This module is designed for the current main.py interface:

    result = run(model, graph, start_idx, end_idx, ...)

Returned result always contains:
    path_indices : best path node indices
    total_cost   : best total cost

If k_paths > 1, it also contains:
    path_results : ranked list of path dictionaries

Cost function:
    total_cost = travel_cost + turn_cost

where:
    travel_cost = distance * average_slowness * optional_label_factor
    turn_cost   = turn_weight * turning_angle_degree / 180

Increase turn_weight to produce smoother paths with fewer turns.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ============================================================
# Optional project parameters
# ============================================================

try:
    import parameters as _parameter
except Exception:  # pragma: no cover
    try:
        import parameter as _parameter
    except Exception:
        _parameter = None


def _get_param(name: str, default: Any = None) -> Any:
    if _parameter is None:
        return default
    return getattr(_parameter, name, default)


# ============================================================
# Basic helpers
# ============================================================

def _as_int(value: Any) -> int:
    return int(value)


def _label_prefix(label: Any) -> str:
    text = str(label).strip().upper()
    out = []
    for char in text:
        if char.isalpha():
            out.append(char)
        else:
            break
    return "".join(out)


def _get_xyz(model, idx: int, dimension: int = 2) -> np.ndarray:
    if dimension >= 3 and "z" in model.columns:
        return np.array(
            [model.at[idx, "x"], model.at[idx, "y"], model.at[idx, "z"]],
            dtype=float,
        )
    return np.array([model.at[idx, "x"], model.at[idx, "y"]], dtype=float)


def _distance(model, a: int, b: int, dimension: int = 2) -> float:
    pa = _get_xyz(model, a, dimension)
    pb = _get_xyz(model, b, dimension)
    return float(np.linalg.norm(pb - pa))


def _direction_angle(model, a: int, b: int) -> float:
    dx = float(model.at[b, "x"] - model.at[a, "x"])
    dy = float(model.at[b, "y"] - model.at[a, "y"])
    return math.atan2(dy, dx)


def _angle_difference_degree(angle1: float, angle2: float) -> float:
    diff = abs(angle1 - angle2)
    while diff > math.pi:
        diff = abs(diff - 2.0 * math.pi)
    return math.degrees(diff)


def _direction_bucket(angle: float, n_buckets: int = 32) -> int:
    return int(round((angle + math.pi) / (2.0 * math.pi) * n_buckets)) % n_buckets


def _model_slowness(model, idx: int) -> float:
    if "slowness" in model.columns:
        return float(model.at[idx, "slowness"])
    if "slow" in model.columns:
        return float(model.at[idx, "slow"])
    return 1.0


def _node_label_factor(model, idx: int) -> float:
    """
    Optional cost multiplier for labels such as FLZ.
    The graph builder may already handle this, but this keeps src/astar.py
    correct when it uses its own neighbor/cost fallback.
    """
    high_cost_prefixes = tuple(_get_param("HIGH_COST_LABEL_PREFIXES", ("FLZ",)))
    high_cost_factor = float(_get_param("FLZ_COST_FACTOR", 1.0))

    if "label" not in model.columns:
        return 1.0

    prefix = _label_prefix(model.at[idx, "label"])
    if prefix in high_cost_prefixes:
        return high_cost_factor
    return 1.0


def _edge_travel_cost(model, a: int, b: int, dimension: int = 2) -> float:
    dist = _distance(model, a, b, dimension)
    slow_avg = 0.5 * (_model_slowness(model, a) + _model_slowness(model, b))
    factor_avg = 0.5 * (_node_label_factor(model, a) + _node_label_factor(model, b))
    return float(dist * slow_avg * factor_avg)


def _extract_neighbor_index(item: Any) -> int:
    """
    Accept neighbor formats:
        12
        (12, cost)
        [12, cost]
        {"index": 12}
        {"node": 12}
        {"to": 12}
    """
    if isinstance(item, dict):
        for key in ("index", "node", "to", "target", "idx"):
            if key in item:
                return int(item[key])
        raise ValueError(f"Cannot extract neighbor index from dict: {item}")

    if isinstance(item, (tuple, list, np.ndarray)):
        if len(item) == 0:
            raise ValueError("Empty neighbor item")
        return int(item[0])

    return int(item)


# ============================================================
# Neighbor provider
# ============================================================

class NeighborProvider:
    def __init__(self, model, graph: dict):
        self.model = model
        self.graph = graph
        self.dimension = int(graph.get("dimension", 2))
        self.valid_indices = set(int(i) for i in graph.get("valid_indices", model.index))
        self.cache: dict[int, list[int]] = {}

        self.adjacency = None
        for key in ("neighbors", "adjacency", "adj", "edges", "graph"):
            if key in graph and isinstance(graph[key], dict):
                self.adjacency = graph[key]
                break

        self.tree = None
        self.tree_indices = None
        self.radius = None
        self.max_neighbors = None

        if self.adjacency is None:
            self._build_kdtree_fallback()

    def _build_kdtree_fallback(self) -> None:
        if cKDTree is None:
            raise ImportError(
                "scipy is required for KDTree neighbor fallback, but scipy.spatial.cKDTree is not available."
            )

        self.tree_indices = np.array(sorted(self.valid_indices), dtype=int)
        coords = np.vstack([
            _get_xyz(self.model, int(idx), self.dimension)
            for idx in self.tree_indices
        ])

        self.tree = cKDTree(coords)

        self.radius = float(self.graph.get("neighbor_radius_m", 0.0) or 0.0)
        if self.radius <= 0:
            self.radius = self._estimate_neighbor_radius(coords)

        self.max_neighbors = int(
            self.graph.get(
                "max_neighbors",
                26 if self.dimension >= 3 else 8,
            )
        )

    def _estimate_neighbor_radius(self, coords: np.ndarray) -> float:
        if len(coords) < 2:
            return 1.0

        sample = coords
        if len(coords) > 5000:
            sample = coords[np.linspace(0, len(coords) - 1, 5000).astype(int)]

        tree = cKDTree(sample)
        dists, _ = tree.query(sample, k=2)
        nearest = dists[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

        if len(nearest) == 0:
            return 1.0

        grid_spacing = float(np.median(nearest))
        radius_factor = float(self.graph.get("kdtree_radius_factor", 1.60))
        return grid_spacing * radius_factor

    def neighbors(self, node: int) -> list[int]:
        node = int(node)
        if node in self.cache:
            return self.cache[node]

        if self.adjacency is not None:
            raw_neighbors = self.adjacency.get(node, self.adjacency.get(str(node), []))
            out = []
            for item in raw_neighbors:
                try:
                    nb = _extract_neighbor_index(item)
                except Exception:
                    continue
                if nb in self.valid_indices and nb != node:
                    out.append(nb)
            self.cache[node] = out
            return out

        # KDTree fallback.
        p = _get_xyz(self.model, node, self.dimension)
        candidate_positions = self.tree.query_ball_point(p, r=self.radius)
        candidate_indices = [int(self.tree_indices[pos]) for pos in candidate_positions]
        candidate_indices = [idx for idx in candidate_indices if idx != node]
        candidate_indices.sort(key=lambda idx: _distance(self.model, node, idx, self.dimension))

        if self.max_neighbors > 0:
            candidate_indices = candidate_indices[:self.max_neighbors]

        self.cache[node] = candidate_indices
        return candidate_indices


# ============================================================
# Metrics
# ============================================================

def _path_turn_metrics(model, path: list[int], threshold_degree: float) -> tuple[int, float]:
    if len(path) < 3:
        return 0, 0.0

    turn_count = 0
    total_angle = 0.0
    prev_angle = _direction_angle(model, path[0], path[1])

    for a, b in zip(path[1:-1], path[2:]):
        angle = _direction_angle(model, a, b)
        delta = _angle_difference_degree(prev_angle, angle)
        total_angle += delta
        if delta > threshold_degree:
            turn_count += 1
        prev_angle = angle

    return int(turn_count), float(total_angle)


def _path_travel_cost(model, path: list[int], dimension: int) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(_edge_travel_cost(model, a, b, dimension) for a, b in zip(path[:-1], path[1:])))


def _build_path_item(
    model,
    path: list[int],
    rank: int,
    total_cost: float,
    travel_cost: float,
    turn_cost: float,
    threshold_degree: float,
    reached_goal: bool = True,
) -> dict[str, Any]:
    turn_count, total_turn_angle_degree = _path_turn_metrics(model, path, threshold_degree)
    return {
        "rank": int(rank),
        "path_indices": [int(i) for i in path],
        "nodes": int(len(path)),
        "total_cost": float(total_cost),
        "cost": float(total_cost),
        "travel_cost": float(travel_cost),
        "turn_cost": float(turn_cost),
        "turn_count": int(turn_count),
        "total_turn_angle_degree": float(total_turn_angle_degree),
        "reached_goal": bool(reached_goal),
    }


# ============================================================
# Path overlap helpers
# ============================================================

def _is_lonlat_xy(model) -> bool:
    """Return True when model x/y look like longitude/latitude degrees."""
    try:
        x = np.asarray(model["x"], dtype=float)
        y = np.asarray(model["y"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return False
        return (
            np.nanmin(x[finite]) >= -180.0
            and np.nanmax(x[finite]) <= 180.0
            and np.nanmin(y[finite]) >= -90.0
            and np.nanmax(y[finite]) <= 90.0
        )
    except Exception:
        return False


def _coords_for_metric_buffer(model, indices: list[int], dimension: int = 2) -> np.ndarray:
    """Coordinates for buffer tests in meters when x/y are lon/lat."""
    indices = [int(i) for i in indices]
    coords = np.vstack([_get_xyz(model, idx, dimension) for idx in indices]).astype(float)

    if len(indices) == 0:
        return coords

    if _is_lonlat_xy(model):
        # Local equirectangular scaling.  Good enough for small UTM operation areas.
        y_values = coords[:, 1]
        lat0 = math.radians(float(np.nanmean(y_values)))
        scale = np.ones(coords.shape[1], dtype=float)
        scale[0] = 111_320.0 * math.cos(lat0)
        scale[1] = 110_540.0
        coords = coords * scale

    return coords


def _label_prefix_mask(model, prefixes) -> np.ndarray:
    """Return a boolean mask where label starts with any requested prefix."""
    prefixes = tuple(str(p).strip().upper() for p in (prefixes or ()) if str(p).strip())
    if not prefixes or "label" not in model.columns:
        return np.zeros(len(model), dtype=bool)

    labels = model["label"].astype(str).str.upper()
    mask = np.zeros(len(model), dtype=bool)
    for prefix in prefixes:
        mask |= labels.str.startswith(prefix).to_numpy(bool)
    return np.asarray(mask, dtype=bool)


def _build_allowed_overlap_nodes(
    model,
    valid_indices: set[int],
    start_idx: int,
    end_idx: int,
    dimension: int,
    buffer_radius_m: float,
    allowed_label_prefixes=("DB", "DK", "FLZ"),
) -> set[int]:
    """Nodes where different ranked paths are allowed to overlap.

    Overlap is allowed inside a buffer around:
      - search start
      - search end
      - any DB/DK/FLZ node, or prefixes supplied by allowed_label_prefixes

    This keeps paths separated in normal grid cells while still letting them
    share facility/service areas.
    """
    valid = {int(i) for i in valid_indices}
    seed_indices = {int(start_idx), int(end_idx)}

    if "label" in model.columns:
        facility_mask = _label_prefix_mask(model, allowed_label_prefixes)
        try:
            seed_indices.update(int(i) for i in model.index[np.asarray(facility_mask, dtype=bool)])
        except Exception:
            # Fallback if model.index does not support boolean selection in the expected way.
            seed_indices.update(int(i) for i, keep in zip(model.index, facility_mask) if keep)

    # Exact seed nodes are always allowed even when the buffer radius is zero.
    allowed = set(seed_indices)

    buffer_radius_m = float(buffer_radius_m or 0.0)
    if buffer_radius_m <= 0:
        return allowed

    candidate_indices = sorted(valid | seed_indices)
    seed_indices = sorted(seed_indices)

    if not candidate_indices or not seed_indices:
        return allowed

    candidate_coords = _coords_for_metric_buffer(model, candidate_indices, dimension)
    seed_coords = _coords_for_metric_buffer(model, seed_indices, dimension)

    if cKDTree is not None:
        tree = cKDTree(candidate_coords)
        hits = tree.query_ball_point(seed_coords, r=buffer_radius_m)
        for hit_list in hits:
            allowed.update(candidate_indices[int(pos)] for pos in hit_list)
    else:  # pragma: no cover - scipy is normally available for this project
        for s in seed_coords:
            d = np.linalg.norm(candidate_coords - s, axis=1)
            for pos in np.where(d <= buffer_radius_m)[0]:
                allowed.add(candidate_indices[int(pos)])

    return {int(i) for i in allowed}


def _edge_key(a: int, b: int) -> tuple[int, int]:
    """Undirected edge key."""
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _blocked_nodes_from_path(path: list[int], allowed_overlap_nodes: set[int]) -> set[int]:
    """Nodes from a selected path that later paths are not allowed to reuse."""
    allowed_overlap_nodes = {int(i) for i in allowed_overlap_nodes}
    return {int(i) for i in path if int(i) not in allowed_overlap_nodes}


def _blocked_edges_from_path(path: list[int], allowed_overlap_nodes: set[int]) -> set[tuple[int, int]]:
    """Edges from a selected path that later paths are not allowed to reuse."""
    allowed_overlap_nodes = {int(i) for i in allowed_overlap_nodes}
    blocked = set()
    for a, b in zip(path[:-1], path[1:]):
        a = int(a)
        b = int(b)
        # If both edge endpoints are inside an allowed overlap zone, the edge is allowed.
        if a in allowed_overlap_nodes and b in allowed_overlap_nodes:
            continue
        blocked.add(_edge_key(a, b))
    return blocked


def _astar_search_one(
    model,
    provider: NeighborProvider,
    start_idx: int,
    end_idx: int,
    dimension: int,
    min_slowness: float,
    turn_weight: float,
    turn_angle_threshold_degree: float,
    max_expansions: int,
    max_states_per_node_direction: int,
    heuristic_weight: float,
    use_turn_penalty: bool,
    forbidden_nodes: set[int] | None = None,
    forbidden_edges: set[tuple[int, int]] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any] | None, int, str]:
    """Find one best path while respecting optional forbidden nodes/edges."""
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    forbidden_nodes = {int(i) for i in (forbidden_nodes or set())}
    forbidden_edges = set(forbidden_edges or set())

    # Never forbid the requested endpoints themselves.
    forbidden_nodes.discard(start_idx)
    forbidden_nodes.discard(end_idx)

    def heuristic(node: int) -> float:
        return heuristic_weight * _distance(model, node, end_idx, dimension) * min_slowness

    heap: list[tuple[float, float, float, float, int, float | None, tuple[int, ...]]] = []
    heapq.heappush(heap, (heuristic(start_idx), 0.0, 0.0, 0.0, start_idx, None, (start_idx,)))

    state_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    expansions = 0

    while heap:
        _f_cost, total_cost, travel_cost, turn_cost, node, prev_angle, path_tuple = heapq.heappop(heap)
        expansions += 1

        if expansions > max_expansions:
            return None, expansions, f"Reached max_expansions={max_expansions:,}."

        if node == end_idx:
            path = list(path_tuple)
            return (
                _build_path_item(
                    model=model,
                    path=path,
                    rank=1,
                    total_cost=total_cost,
                    travel_cost=travel_cost,
                    turn_cost=turn_cost,
                    threshold_degree=turn_angle_threshold_degree,
                    reached_goal=True,
                ),
                expansions,
                "Found path.",
            )

        for nb in provider.neighbors(node):
            nb = int(nb)

            if nb in path_tuple:
                # Keep each path loopless.
                continue
            if nb in forbidden_nodes:
                continue
            if _edge_key(node, nb) in forbidden_edges:
                continue

            step_travel_cost = _edge_travel_cost(model, node, nb, dimension)
            new_angle = _direction_angle(model, node, nb)

            step_turn_cost = 0.0
            if use_turn_penalty and prev_angle is not None and turn_weight > 0:
                delta_angle = _angle_difference_degree(prev_angle, new_angle)
                step_turn_cost = turn_weight * delta_angle / 180.0

            new_travel_cost = travel_cost + step_travel_cost
            new_turn_cost = turn_cost + step_turn_cost
            new_total_cost = total_cost + step_travel_cost + step_turn_cost

            bucket = _direction_bucket(new_angle)
            state_key = (nb, bucket)
            if state_counts[state_key] >= max_states_per_node_direction:
                continue
            state_counts[state_key] += 1

            new_path_tuple = path_tuple + (nb,)
            new_f = new_total_cost + heuristic(nb)
            heapq.heappush(
                heap,
                (
                    new_f,
                    new_total_cost,
                    new_travel_cost,
                    new_turn_cost,
                    nb,
                    new_angle,
                    new_path_tuple,
                ),
            )

    return None, expansions, "No path found."


# ============================================================
# Top-K path search with turn penalty
# ============================================================

def run(
    model,
    graph: dict,
    start_idx: int,
    end_idx: int,
    k_paths: int = 1,
    turn_weight: float = 0.0,
    turn_angle_threshold_degree: float = 1.0,
    max_expansions: int = 5_000_000,
    max_states_per_node_direction: int = 150,
    heuristic_weight: float = 1.0,
    use_turn_penalty: bool = True,
    save_all_k_paths: bool = True,
    verbose: bool = True,
    path_overlap_mode: str = "allow",
    non_overlap_buffer_radius_m: float = 150.0,
    non_overlap_allowed_prefixes=("DB", "DK", "FLZ"),
    non_overlap_block_edges: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Run Top-K path search with optional turn penalty and optional non-overlap control.

    path_overlap_mode:
        "allow" / "overlap" / "normal"
            Keep the original behavior: return the best K paths, which may share
            nodes or edges.

        "non_overlap" / "no_overlap" / "disjoint"
            Build paths sequentially. After each selected path, its normal grid
            nodes are blocked for later paths. Overlap remains allowed inside a
            buffer around start/end and around DB/DK/FLZ service nodes.
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    k_paths = max(1, int(k_paths))
    turn_weight = float(turn_weight)
    heuristic_weight = float(heuristic_weight)
    max_expansions = int(max_expansions)
    max_states_per_node_direction = max(1, int(max_states_per_node_direction))
    turn_angle_threshold_degree = float(turn_angle_threshold_degree)
    path_overlap_mode = str(path_overlap_mode or "allow").strip().lower()
    non_overlap_buffer_radius_m = float(non_overlap_buffer_radius_m or 0.0)
    non_overlap_allowed_prefixes = tuple(non_overlap_allowed_prefixes or ())
    non_overlap_block_edges = bool(non_overlap_block_edges)

    dimension = int(graph.get("dimension", 2))
    provider = NeighborProvider(model, graph)

    valid_indices = provider.valid_indices
    valid_slowness = [
        _model_slowness(model, idx)
        for idx in valid_indices
        if np.isfinite(_model_slowness(model, idx)) and _model_slowness(model, idx) > 0
    ]
    min_slowness = float(min(valid_slowness)) if valid_slowness else 1.0

    non_overlap_modes = {
        "non_overlap",
        "no_overlap",
        "non-overlap",
        "no-overlap",
        "disjoint",
        "node_disjoint",
        "node-disjoint",
    }
    use_non_overlap = path_overlap_mode in non_overlap_modes

    # ============================================================
    # Original behavior: enumerate best K paths, allowing overlap.
    # ============================================================
    if not use_non_overlap:
        def heuristic(node: int) -> float:
            return heuristic_weight * _distance(model, node, end_idx, dimension) * min_slowness

        # Heap item:
        #   f, total_cost, travel_cost, turn_cost, node, prev_angle, path_tuple
        heap: list[tuple[float, float, float, float, int, float | None, tuple[int, ...]]] = []

        start_path = (start_idx,)
        heapq.heappush(heap, (heuristic(start_idx), 0.0, 0.0, 0.0, start_idx, None, start_path))

        found: list[dict[str, Any]] = []
        found_path_set: set[tuple[int, ...]] = set()
        state_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

        expansions = 0

        while heap and len(found) < k_paths:
            f_cost, total_cost, travel_cost, turn_cost, node, prev_angle, path_tuple = heapq.heappop(heap)
            expansions += 1

            if expansions > max_expansions:
                if verbose:
                    print(f"[WARNING] Path search reached max_expansions={max_expansions:,}")
                break

            if node == end_idx:
                if path_tuple not in found_path_set:
                    found_path_set.add(path_tuple)
                    path = list(path_tuple)
                    found.append(
                        _build_path_item(
                            model=model,
                            path=path,
                            rank=len(found) + 1,
                            total_cost=total_cost,
                            travel_cost=travel_cost,
                            turn_cost=turn_cost,
                            threshold_degree=turn_angle_threshold_degree,
                            reached_goal=True,
                        )
                    )
                    if verbose:
                        item = found[-1]
                        print(
                            f"[OK] path {item['rank']:03d}/{k_paths}: "
                            f"total={item['total_cost']:.6g}, "
                            f"travel={item['travel_cost']:.6g}, "
                            f"turn={item['turn_cost']:.6g}, "
                            f"turns={item['turn_count']}, nodes={item['nodes']}"
                        )
                continue

            for nb in provider.neighbors(node):
                nb = int(nb)
                if nb in path_tuple:
                    # Keep paths loopless.
                    continue

                step_travel_cost = _edge_travel_cost(model, node, nb, dimension)
                new_angle = _direction_angle(model, node, nb)

                step_turn_cost = 0.0
                if use_turn_penalty and prev_angle is not None and turn_weight > 0:
                    delta_angle = _angle_difference_degree(prev_angle, new_angle)
                    step_turn_cost = turn_weight * delta_angle / 180.0

                new_travel_cost = travel_cost + step_travel_cost
                new_turn_cost = turn_cost + step_turn_cost
                new_total_cost = total_cost + step_travel_cost + step_turn_cost

                bucket = _direction_bucket(new_angle)
                state_key = (nb, bucket)
                if state_counts[state_key] >= max_states_per_node_direction:
                    continue
                state_counts[state_key] += 1

                new_path_tuple = path_tuple + (nb,)
                new_f = new_total_cost + heuristic(nb)

                heapq.heappush(
                    heap,
                    (
                        new_f,
                        new_total_cost,
                        new_travel_cost,
                        new_turn_cost,
                        nb,
                        new_angle,
                        new_path_tuple,
                    ),
                )

        overlap_allowed_nodes_count = 0
        forbidden_nodes_count = 0
        forbidden_edges_count = 0

    # ============================================================
    # New behavior: sequential non-overlapping path population.
    # ============================================================
    else:
        allowed_overlap_nodes = _build_allowed_overlap_nodes(
            model=model,
            valid_indices=valid_indices,
            start_idx=start_idx,
            end_idx=end_idx,
            dimension=dimension,
            buffer_radius_m=non_overlap_buffer_radius_m,
            allowed_label_prefixes=non_overlap_allowed_prefixes,
        )

        found = []
        found_path_set: set[tuple[int, ...]] = set()
        forbidden_nodes: set[int] = set()
        forbidden_edges: set[tuple[int, int]] = set()
        expansions = 0
        last_message = ""

        if verbose:
            print("[INFO] path search multiple overlap mode: non_overlap")
            print(f"       allowed overlap buffer : {non_overlap_buffer_radius_m:.2f} m")
            print(f"       allowed prefixes       : {non_overlap_allowed_prefixes}")
            print(f"       allowed overlap nodes   : {len(allowed_overlap_nodes):,}")

        for rank in range(1, k_paths + 1):
            remaining_expansions = max(1, max_expansions - expansions)
            item, used_expansions, message = _astar_search_one(
                model=model,
                provider=provider,
                start_idx=start_idx,
                end_idx=end_idx,
                dimension=dimension,
                min_slowness=min_slowness,
                turn_weight=turn_weight,
                turn_angle_threshold_degree=turn_angle_threshold_degree,
                max_expansions=remaining_expansions,
                max_states_per_node_direction=max_states_per_node_direction,
                heuristic_weight=heuristic_weight,
                use_turn_penalty=use_turn_penalty,
                forbidden_nodes=forbidden_nodes,
                forbidden_edges=forbidden_edges if non_overlap_block_edges else set(),
                verbose=verbose,
            )
            expansions += int(used_expansions)
            last_message = str(message)

            if item is None:
                if verbose:
                    print(
                        f"[WARNING] path search non-overlap stopped at rank {rank:03d}/{k_paths}: "
                        f"{message}"
                    )
                break

            path_tuple = tuple(int(i) for i in item["path_indices"])
            if path_tuple in found_path_set:
                if verbose:
                    print(
                        f"[WARNING] path search non-overlap produced a duplicate at rank {rank:03d}; stopping."
                    )
                break

            item["rank"] = int(rank)
            found_path_set.add(path_tuple)
            found.append(item)

            newly_blocked_nodes = _blocked_nodes_from_path(
                item["path_indices"],
                allowed_overlap_nodes=allowed_overlap_nodes,
            )
            forbidden_nodes.update(newly_blocked_nodes)

            if non_overlap_block_edges:
                newly_blocked_edges = _blocked_edges_from_path(
                    item["path_indices"],
                    allowed_overlap_nodes=allowed_overlap_nodes,
                )
                forbidden_edges.update(newly_blocked_edges)

            if verbose:
                print(
                    f"[OK] path {rank:03d}/{k_paths}: "
                    f"total={item['total_cost']:.6g}, "
                    f"travel={item['travel_cost']:.6g}, "
                    f"turn={item['turn_cost']:.6g}, "
                    f"turns={item['turn_count']}, nodes={item['nodes']}, "
                    f"blocked_nodes_for_next={len(forbidden_nodes):,}"
                )

            if expansions >= max_expansions:
                if verbose:
                    print(f"[WARNING] Path search reached max_expansions={max_expansions:,}")
                break

        overlap_allowed_nodes_count = len(allowed_overlap_nodes)
        forbidden_nodes_count = len(forbidden_nodes)
        forbidden_edges_count = len(forbidden_edges)

    if not found:
        return {
            "algorithm": "astar_multiple",
            "path_indices": [],
            "path_results": [],
            "ranked_paths": [],
            "total_cost": None,
            "success": False,
            "message": "No path found.",
            "k_paths_requested": int(k_paths),
            "k_paths_found": 0,
            "expanded_states": int(expansions),
            "turn_weight": float(turn_weight),
            "use_turn_penalty": bool(use_turn_penalty),
            "path_overlap_mode": str(path_overlap_mode),
            "non_overlap_buffer_radius_m": float(non_overlap_buffer_radius_m),
            "non_overlap_allowed_prefixes": list(non_overlap_allowed_prefixes),
            "non_overlap_allowed_nodes": int(overlap_allowed_nodes_count),
            "non_overlap_forbidden_nodes": int(forbidden_nodes_count),
            "non_overlap_forbidden_edges": int(forbidden_edges_count),
        }

    best = found[0]

    return {
        "algorithm": "astar_multiple",
        "path_indices": best["path_indices"],
        "path_results": found if save_all_k_paths else [best],
        "ranked_paths": found if save_all_k_paths else [best],
        "total_cost": best["total_cost"],
        "travel_cost": best["travel_cost"],
        "turn_cost": best["turn_cost"],
        "turn_count": best["turn_count"],
        "total_turn_angle_degree": best["total_turn_angle_degree"],
        "success": True,
        "message": f"Found {len(found)} path(s).",
        "k_paths_requested": int(k_paths),
        "k_paths_found": int(len(found)),
        "expanded_states": int(expansions),
        "turn_weight": float(turn_weight),
        "turn_angle_threshold_degree": float(turn_angle_threshold_degree),
        "use_turn_penalty": bool(use_turn_penalty),
        "heuristic_weight": float(heuristic_weight),
        "max_states_per_node_direction": int(max_states_per_node_direction),
        "path_overlap_mode": str(path_overlap_mode),
        "non_overlap_buffer_radius_m": float(non_overlap_buffer_radius_m),
        "non_overlap_allowed_prefixes": list(non_overlap_allowed_prefixes),
        "non_overlap_allowed_nodes": int(overlap_allowed_nodes_count),
        "non_overlap_forbidden_nodes": int(forbidden_nodes_count),
        "non_overlap_forbidden_edges": int(forbidden_edges_count),
    }
