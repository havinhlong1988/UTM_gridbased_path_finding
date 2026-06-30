#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dijkstra path finder.

This uses graph connectivity plus movement cost/slowness.
It finds the minimum accumulated-cost path between start and end.

Return structure is kept similar to flood_fill.py.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from src.model_io import iter_neighbors


def run(model, graph, start_idx: int, end_idx: int) -> dict:
    """
    Run Dijkstra search.

    Parameters
    ----------
    model : dict-like
        Model object/data used by iter_neighbors and cost extraction.

    graph : dict-like
        Graph object that must include:
            graph["valid_indices"]

        Optional useful fields:
            graph["coords"]
            graph["xyz"]
            graph["points"]
            graph["slowness"]
            graph["cost"]

    start_idx : int
        Start node index.

    end_idx : int
        End node index.

    Returns
    -------
    dict
        Same style as flood_fill.py:
            success
            algorithm
            message
            path_indices
            total_cost
            expanded_nodes
            visited_nodes
            runtime_seconds
    """

    t0 = time.time()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if start_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "algorithm": "dijkstra",
            "message": "Start node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    if end_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "algorithm": "dijkstra",
            "message": "End node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    # Priority queue item:
    #     (accumulated_cost, node_index)
    queue = []
    heapq.heappush(queue, (0.0, start_idx))

    came_from = {start_idx: None}
    cost_so_far = {start_idx: 0.0}
    visited = set()

    expanded_nodes = 0

    while queue:
        current_cost, current = heapq.heappop(queue)
        current = int(current)

        if current in visited:
            continue

        visited.add(current)
        expanded_nodes += 1

        if current == end_idx:
            path = reconstruct_path(came_from, end_idx)

            return {
                "success": True,
                "algorithm": "dijkstra",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": float(cost_so_far[end_idx]),
                "expanded_nodes": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "runtime_seconds": float(time.time() - t0),
            }

        for neighbor in iter_neighbors(model, graph, current):
            neighbor = int(neighbor)

            if neighbor in visited:
                continue

            step_cost = get_step_cost(model, graph, current, neighbor)

            if not math.isfinite(step_cost):
                continue

            new_cost = current_cost + step_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = float(new_cost)
                came_from[neighbor] = current
                heapq.heappush(queue, (float(new_cost), neighbor))

    return {
        "success": False,
        "algorithm": "dijkstra",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "visited_nodes": int(len(visited)),
        "runtime_seconds": float(time.time() - t0),
    }


def get_step_cost(model, graph, current_idx: int, neighbor_idx: int) -> float:
    """
    Compute movement cost from current_idx to neighbor_idx.

    Default logic:
        step_cost = average_node_cost * geometric_distance

    It tries to read node cost/slowness from graph or model.

    Priority:
        1. graph["slowness"]
        2. graph["cost"]
        3. model["slowness"]
        4. model["cost"]
        5. fallback = 1.0

    Distance priority:
        1. graph["coords"]
        2. graph["xyz"]
        3. graph["points"]
        4. model["coords"]
        5. model["xyz"]
        6. model["points"]
        7. fallback = 1.0
    """

    current_idx = int(current_idx)
    neighbor_idx = int(neighbor_idx)

    c0 = get_node_cost(model, graph, current_idx)
    c1 = get_node_cost(model, graph, neighbor_idx)

    if not math.isfinite(c0) or not math.isfinite(c1):
        return float("inf")

    average_cost = 0.5 * (float(c0) + float(c1))
    distance = get_node_distance(model, graph, current_idx, neighbor_idx)

    if not math.isfinite(distance):
        return float("inf")

    return float(average_cost * distance)


def get_node_cost(model, graph, idx: int) -> float:
    """
    Return node cost/slowness value.
    """

    idx = int(idx)

    for container in (graph, model):
        for key in ("slowness", "cost"):
            if isinstance(container, dict) and key in container:
                values = container[key]
                return float(values[idx])

    # Fallback: pure shortest path in weighted graph step distance.
    return 1.0


def get_node_distance(model, graph, idx0: int, idx1: int) -> float:
    """
    Return geometric distance between two nodes.

    If no coordinate array is found, use distance = 1.0.
    """

    idx0 = int(idx0)
    idx1 = int(idx1)

    for container in (graph, model):
        for key in ("coords", "xyz", "points"):
            if isinstance(container, dict) and key in container:
                arr = np.asarray(container[key], dtype=float)

                p0 = arr[idx0]
                p1 = arr[idx1]

                return float(np.linalg.norm(p1 - p0))

    return 1.0


def reconstruct_path(came_from: dict, end_idx: int) -> list[int]:
    """
    Reconstruct Dijkstra path.
    """

    path = []
    current = int(end_idx)

    while current is not None:
        path.append(int(current))
        current = came_from[current]

    path.reverse()
    return path