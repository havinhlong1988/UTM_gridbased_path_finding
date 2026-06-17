#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Top-K A* path finder with turn minimization.

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
# Top-K A* with turn penalty
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
    **kwargs,
) -> dict[str, Any]:
    """
    Run Top-K A* with turn penalty.

    Parameters are intentionally simple so main.py can pass values directly
    from parameters.py.
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    k_paths = max(1, int(k_paths))
    turn_weight = float(turn_weight)
    heuristic_weight = float(heuristic_weight)
    max_expansions = int(max_expansions)
    max_states_per_node_direction = max(1, int(max_states_per_node_direction))
    turn_angle_threshold_degree = float(turn_angle_threshold_degree)

    dimension = int(graph.get("dimension", 2))
    provider = NeighborProvider(model, graph)

    valid_indices = provider.valid_indices
    valid_slowness = [
        _model_slowness(model, idx)
        for idx in valid_indices
        if np.isfinite(_model_slowness(model, idx)) and _model_slowness(model, idx) > 0
    ]
    min_slowness = float(min(valid_slowness)) if valid_slowness else 1.0

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
                print(f"[WARNING] A* reached max_expansions={max_expansions:,}")
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
                        f"[OK] A* path {item['rank']:03d}/{k_paths}: "
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

    if not found:
        return {
            "algorithm": "astar_multiple",
            "path_indices": [],
            "path_results": [],
            "total_cost": None,
            "success": False,
            "message": "No path found.",
            "k_paths_requested": int(k_paths),
            "k_paths_found": 0,
            "expanded_states": int(expansions),
            "turn_weight": float(turn_weight),
            "use_turn_penalty": bool(use_turn_penalty),
        }

    best = found[0]

    return {
        "algorithm": "astar_multiple",
        "path_indices": best["path_indices"],
        "path_results": found if save_all_k_paths else [best],
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
    }
