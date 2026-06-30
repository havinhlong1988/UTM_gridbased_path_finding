#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A* path-finding algorithm.

Required interface:
    run(model, graph, start_idx, end_idx) -> dict

The graph utilities are provided by src.model_io:
    iter_neighbors
    edge_cost
    heuristic_cost
"""

from __future__ import annotations

import heapq
import math
import time

from src.model_io import iter_neighbors, edge_cost, heuristic_cost


def run(model, graph, start_idx: int, end_idx: int) -> dict:
    """
    Run A* search.

    Parameters
    ----------
    model : pandas.DataFrame
        Model table with columns x, y, z, slowness, label, label_prefix.
    graph : dict
        Graph metadata built by build_grid_graph().
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

    if start_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "algorithm": "astar",
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
            "algorithm": "astar",
            "message": "End node is blocked or not traversable.",
            "path_indices": [],
            "total_cost": None,
            "expanded_nodes": 0,
            "visited_nodes": 0,
            "runtime_seconds": time.time() - t0,
        }

    open_heap = []
    heap_counter = 0

    g_score = {start_idx: 0.0}
    f_start = heuristic_cost(model, graph, start_idx, end_idx)

    heapq.heappush(open_heap, (f_start, heap_counter, start_idx))

    came_from = {}
    visited = set()

    expanded_nodes = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        expanded_nodes += 1

        if current == end_idx:
            path = reconstruct_path(came_from, current)
            total_cost = float(g_score[current])

            return {
                "success": True,
                "algorithm": "astar",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": total_cost,
                "expanded_nodes": int(expanded_nodes),
                "visited_nodes": int(len(visited)),
                "runtime_seconds": float(time.time() - t0),
            }

        for neighbor in iter_neighbors(model, graph, current):
            neighbor = int(neighbor)

            tentative_g = g_score[current] + edge_cost(
                model=model,
                graph=graph,
                idx1=current,
                idx2=neighbor,
            )

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f = tentative_g + heuristic_cost(
                    model=model,
                    graph=graph,
                    idx=neighbor,
                    end_idx=end_idx,
                )

                heap_counter += 1
                heapq.heappush(open_heap, (f, heap_counter, neighbor))

    return {
        "success": False,
        "algorithm": "astar",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "visited_nodes": int(len(visited)),
        "runtime_seconds": float(time.time() - t0),
    }


def reconstruct_path(came_from: dict, current: int) -> list[int]:
    """
    Reconstruct path from came_from dictionary.
    """

    current = int(current)
    path = [current]

    while current in came_from:
        current = int(came_from[current])
        path.append(current)

    path.reverse()
    return path