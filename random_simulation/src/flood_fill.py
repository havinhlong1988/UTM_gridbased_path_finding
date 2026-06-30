#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flood-fill / BFS path finder.

This ignores slowness cost and finds a path with the smallest number
of graph steps. Useful for checking pure connectivity.
"""

from __future__ import annotations

from collections import deque
import time

from src.model_io import iter_neighbors


def run(model, graph, start_idx: int, end_idx: int) -> dict:
    """
    Run flood-fill / BFS search.

    Returns a connectivity path if one exists.
    """

    t0 = time.time()

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if start_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "algorithm": "flood_fill",
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
            "algorithm": "flood_fill",
            "message": "End node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    queue = deque([start_idx])
    came_from = {start_idx: None}
    visited = {start_idx}

    expanded_nodes = 0

    while queue:
        current = queue.popleft()
        expanded_nodes += 1

        if current == end_idx:
            path = reconstruct_path(came_from, end_idx)

            return {
                "success": True,
                "algorithm": "flood_fill",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": float(len(path) - 1),
                "expanded_nodes": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "runtime_seconds": float(time.time() - t0),
            }

        for neighbor in iter_neighbors(model, graph, current):
            neighbor = int(neighbor)

            if neighbor in visited:
                continue

            visited.add(neighbor)
            came_from[neighbor] = current
            queue.append(neighbor)

    return {
        "success": False,
        "algorithm": "flood_fill",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "visited_nodes": int(len(visited)),
        "runtime_seconds": float(time.time() - t0),
    }


def reconstruct_path(came_from: dict, end_idx: int) -> list[int]:
    """
    Reconstruct BFS path.
    """

    path = []
    current = int(end_idx)

    while current is not None:
        path.append(int(current))
        current = came_from[current]

    path.reverse()
    return path