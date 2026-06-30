#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
D* path-finding algorithm for the LAE-UTM grid-node graph.

Required interface used by main.py:
    run(model, graph, start_idx, end_idx, **kwargs) -> dict

This implementation follows the classic D* idea used in the supplied MATLAB
example: propagate cost-to-go backward from the goal, store a parent/successor
pointer for every reached node, then extract the route from start to goal.

The planner works directly on the graph created by src.model_io.build_grid_graph,
so it does not require a compact matrix grid.  Obstacles/no-fly nodes should
already be excluded from graph["valid_indices"] by main.py.  Optional dynamic
blocked nodes can also be supplied for a future re-planning workflow.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.model_io import iter_neighbors, edge_cost

try:
    import parameters as prm
except Exception:  # pragma: no cover - keeps module usable in isolated tests
    prm = None


# D* node tags. These match the MATLAB sample convention.
NEW = 0
OPEN = 1
CLOSED = 2


def _get_param(name: str, default=None):
    """Read optional settings from parameters.py / params/dstar.params."""
    if prm is None:
        return default
    return getattr(prm, name, default)


def _as_int_set(values) -> Set[int]:
    """Convert a scalar/list/tuple/set of node indices to a clean int set."""
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        text = values.decode() if isinstance(values, bytes) else values
        text = text.strip()
        if not text:
            return set()
        return {int(v.strip()) for v in text.split(",") if v.strip()}
    try:
        return {int(v) for v in values}
    except TypeError:
        return {int(values)}


def _safe_edge_cost(model, graph, idx1: int, idx2: int, blocked: Set[int]) -> float:
    """Return finite edge cost, or inf if the edge is blocked/invalid."""
    idx1 = int(idx1)
    idx2 = int(idx2)

    if idx1 in blocked or idx2 in blocked:
        return math.inf
    if idx1 not in graph.get("valid_indices", set()):
        return math.inf
    if idx2 not in graph.get("valid_indices", set()):
        return math.inf

    try:
        cost = float(edge_cost(model=model, graph=graph, idx1=idx1, idx2=idx2))
    except Exception:
        return math.inf

    if not math.isfinite(cost) or cost < 0.0:
        return math.inf
    return cost


def _iter_valid_neighbors(model, graph, idx: int, blocked: Set[int]) -> Iterable[int]:
    """Yield traversable neighbors after dynamic blocking is applied."""
    valid = graph.get("valid_indices", set())
    for neighbor in iter_neighbors(model, graph, int(idx)):
        neighbor = int(neighbor)
        if neighbor in blocked:
            continue
        if neighbor not in valid:
            continue
        yield neighbor


def _insert(
    node: int,
    h_new: float,
    tag: Dict[int, int],
    h: Dict[int, float],
    k: Dict[int, float],
    open_heap: List[Tuple[float, int, int]],
    counter: List[int],
) -> None:
    """Insert/update a state in OPEN using the classic D* key rule."""
    node = int(node)
    h_new = float(h_new)
    old_tag = tag.get(node, NEW)

    if old_tag == NEW:
        k_new = h_new
    elif old_tag == OPEN:
        k_new = min(k.get(node, math.inf), h_new)
    else:  # CLOSED
        k_new = min(h.get(node, math.inf), h_new)

    h[node] = h_new
    k[node] = k_new
    tag[node] = OPEN

    counter[0] += 1
    heapq.heappush(open_heap, (k_new, counter[0], node))


def _pop_min_open(
    open_heap: List[Tuple[float, int, int]],
    tag: Dict[int, int],
    k: Dict[int, float],
) -> Tuple[float, Optional[int]]:
    """Pop the current minimum OPEN state, skipping stale heap entries."""
    while open_heap:
        key, _, node = heapq.heappop(open_heap)
        node = int(node)
        if tag.get(node, NEW) != OPEN:
            continue
        if not math.isclose(float(key), float(k.get(node, math.inf)), rel_tol=1e-12, abs_tol=1e-12):
            continue
        return float(key), node
    return math.inf, None


def _current_min_open_key(
    open_heap: List[Tuple[float, int, int]],
    tag: Dict[int, int],
    k: Dict[int, float],
) -> float:
    """Return the current minimum OPEN key without removing a valid entry."""
    while open_heap:
        key, _, node = open_heap[0]
        node = int(node)
        if tag.get(node, NEW) == OPEN and math.isclose(float(key), float(k.get(node, math.inf)), rel_tol=1e-12, abs_tol=1e-12):
            return float(key)
        heapq.heappop(open_heap)
    return math.inf


def _process_state(
    model,
    graph,
    blocked: Set[int],
    tag: Dict[int, int],
    h: Dict[int, float],
    k: Dict[int, float],
    parent: Dict[int, int],
    open_heap: List[Tuple[float, int, int]],
    counter: List[int],
    expanded_order: Optional[List[int]] = None,
) -> Tuple[float, Optional[int]]:
    """Process one D* OPEN state and return the next minimum OPEN key."""
    k_old, current = _pop_min_open(open_heap, tag, k)
    if current is None:
        return -1.0, None

    tag[current] = CLOSED

    if expanded_order is not None:
        expanded_order.append(int(current))

    h_current = float(h.get(current, math.inf))

    if k_old < h_current and not math.isclose(k_old, h_current, rel_tol=1e-12, abs_tol=1e-12):
        for neighbor in _iter_valid_neighbors(model, graph, current, blocked):
            c = _safe_edge_cost(model, graph, current, neighbor, blocked)
            if not math.isfinite(c):
                continue

            h_neighbor = float(h.get(neighbor, math.inf))
            if (
                tag.get(neighbor, NEW) != NEW
                and h_neighbor <= k_old
                and h_current > h_neighbor + c
            ):
                parent[current] = int(neighbor)
                h_current = h_neighbor + c
                h[current] = h_current

    if math.isclose(k_old, h_current, rel_tol=1e-12, abs_tol=1e-12):
        for neighbor in _iter_valid_neighbors(model, graph, current, blocked):
            c = _safe_edge_cost(model, graph, current, neighbor, blocked)
            if not math.isfinite(c):
                continue

            h_neighbor = float(h.get(neighbor, math.inf))
            neighbor_parent = parent.get(neighbor, None)
            proposed = h_current + c

            if (
                tag.get(neighbor, NEW) == NEW
                or (neighbor_parent == current and not math.isclose(h_neighbor, proposed, rel_tol=1e-12, abs_tol=1e-12))
                or (neighbor_parent != current and h_neighbor > proposed)
            ):
                parent[neighbor] = int(current)
                _insert(neighbor, proposed, tag, h, k, open_heap, counter)
    else:
        for neighbor in _iter_valid_neighbors(model, graph, current, blocked):
            c = _safe_edge_cost(model, graph, current, neighbor, blocked)
            if not math.isfinite(c):
                continue

            h_neighbor = float(h.get(neighbor, math.inf))
            neighbor_parent = parent.get(neighbor, None)
            proposed = h_current + c

            if (
                tag.get(neighbor, NEW) == NEW
                or (neighbor_parent == current and not math.isclose(h_neighbor, proposed, rel_tol=1e-12, abs_tol=1e-12))
            ):
                parent[neighbor] = int(current)
                _insert(neighbor, proposed, tag, h, k, open_heap, counter)
            elif neighbor_parent != current and h_neighbor > proposed:
                _insert(current, h_current, tag, h, k, open_heap, counter)
            elif (
                neighbor_parent != current
                and h_current > h_neighbor + c
                and tag.get(neighbor, NEW) == CLOSED
                and h_neighbor > k_old
            ):
                _insert(neighbor, h_neighbor, tag, h, k, open_heap, counter)

    next_key = _current_min_open_key(open_heap, tag, k)
    if not math.isfinite(next_key):
        return -1.0, current
    return next_key, current


def _extract_path(
    model,
    graph,
    start_idx: int,
    end_idx: int,
    parent: Dict[int, int],
    blocked: Set[int],
    max_steps: int,
) -> Tuple[List[int], float, str]:
    """Extract start -> goal path from D* parent/successor pointers."""
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    path = [start_idx]
    total_cost = 0.0
    current = start_idx
    seen = {start_idx}

    for _ in range(int(max_steps)):
        if current == end_idx:
            return path, float(total_cost), "Path found."

        if current not in parent:
            return [], math.inf, "Parent chain stopped before reaching the goal."

        nxt = int(parent[current])
        if nxt in seen:
            return [], math.inf, "Parent chain contains a loop."

        c = _safe_edge_cost(model, graph, current, nxt, blocked)
        if not math.isfinite(c):
            return [], math.inf, "Parent chain contains a blocked or invalid edge."

        total_cost += c
        current = nxt
        path.append(current)
        seen.add(current)

    return [], math.inf, "Path extraction exceeded maximum step limit."


def _failure_result(
    message: str,
    start_idx: int,
    end_idx: int,
    expanded_nodes: int,
    visited_nodes: int,
    t0: float,
    dynamic_blocked_count: int = 0,
) -> dict:
    """Return a result dictionary compatible with main.py."""
    return {
        "success": False,
        "algorithm": "dstar",
        "message": str(message),
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": int(expanded_nodes),
        "expanded_states": int(expanded_nodes),
        "visited_nodes": int(visited_nodes),
        "runtime_seconds": float(time.time() - t0),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "dynamic_blocked_nodes": int(dynamic_blocked_count),
    }


def run(model, graph, start_idx: int, end_idx: int, **kwargs) -> dict:
    """
    Run classic D* planning on the existing grid-node graph.

    Parameters
    ----------
    model : pandas.DataFrame
        Model table with x/y/z/slowness/label columns after main.py loading.
    graph : dict
        Graph metadata from build_grid_graph(). Must include valid_indices.
    start_idx, end_idx : int
        Search-node indices already snapped/validated by main.py.
    **kwargs : dict
        Extra arguments are accepted so main.py can pass shared options without
        breaking this algorithm.

    Returns
    -------
    dict
        Compatible with the existing export/plot workflow.
    """
    t0 = time.time()
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    valid_indices = graph.get("valid_indices", set())
    if not isinstance(valid_indices, set):
        valid_indices = {int(v) for v in valid_indices}

    # Algorithm-specific controls. DSTAR_* values come from params/dstar.params.
    max_expansions = int(kwargs.get(
        "DSTAR_MAX_EXPANSIONS",
        kwargs.get("max_expansions", _get_param("DSTAR_MAX_EXPANSIONS", 5_000_000)),
    ))
    verbose = bool(kwargs.get("DSTAR_VERBOSE", _get_param("DSTAR_VERBOSE", True)))
    stop_when_start_closed = bool(_get_param("DSTAR_STOP_WHEN_START_CLOSED", True))
    return_expand_list = bool(_get_param("DSTAR_RETURN_EXPAND_LIST", False))

    dynamic_blocked = _as_int_set(kwargs.get(
        "dynamic_blocked_indices",
        _get_param("DSTAR_DYNAMIC_BLOCKED_INDICES", ()),
    ))
    dynamic_blocked = {idx for idx in dynamic_blocked if idx in valid_indices}

    if start_idx not in valid_indices:
        return _failure_result(
            "Start node is blocked or not traversable.",
            start_idx,
            end_idx,
            expanded_nodes=0,
            visited_nodes=0,
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    if end_idx not in valid_indices:
        return _failure_result(
            "End node is blocked or not traversable.",
            start_idx,
            end_idx,
            expanded_nodes=0,
            visited_nodes=0,
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    if start_idx in dynamic_blocked:
        return _failure_result(
            "Start node is dynamically blocked.",
            start_idx,
            end_idx,
            expanded_nodes=0,
            visited_nodes=0,
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    if end_idx in dynamic_blocked:
        return _failure_result(
            "End node is dynamically blocked.",
            start_idx,
            end_idx,
            expanded_nodes=0,
            visited_nodes=0,
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    tag: Dict[int, int] = {}
    h: Dict[int, float] = {}
    k: Dict[int, float] = {}
    parent: Dict[int, int] = {}
    open_heap: List[Tuple[float, int, int]] = []
    counter = [0]
    expanded_order: Optional[List[int]] = [] if return_expand_list else None

    # Reverse propagation starts from the goal, exactly like the MATLAB sample.
    _insert(end_idx, 0.0, tag, h, k, open_heap, counter)

    expanded_nodes = 0
    last_key = math.inf

    while open_heap:
        if expanded_nodes >= max_expansions:
            return _failure_result(
                f"D* reached DSTAR_MAX_EXPANSIONS={max_expansions:,} before finding a path.",
                start_idx,
                end_idx,
                expanded_nodes=expanded_nodes,
                visited_nodes=sum(1 for v in tag.values() if v != NEW),
                t0=t0,
                dynamic_blocked_count=len(dynamic_blocked),
            )

        last_key, processed = _process_state(
            model=model,
            graph=graph,
            blocked=dynamic_blocked,
            tag=tag,
            h=h,
            k=k,
            parent=parent,
            open_heap=open_heap,
            counter=counter,
            expanded_order=expanded_order,
        )

        if processed is None or last_key == -1.0:
            break

        expanded_nodes += 1

        if stop_when_start_closed and tag.get(start_idx, NEW) == CLOSED:
            break

    if tag.get(start_idx, NEW) != CLOSED:
        return _failure_result(
            "No path found. Start node was not reached by backward D* propagation.",
            start_idx,
            end_idx,
            expanded_nodes=expanded_nodes,
            visited_nodes=sum(1 for v in tag.values() if v != NEW),
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    max_path_steps = int(_get_param("DSTAR_MAX_PATH_STEPS", 0) or 0)
    if max_path_steps <= 0:
        # A simple path should not visit more nodes than the traversable graph.
        max_path_steps = max(2, len(valid_indices) + 2)

    path, total_cost, extract_message = _extract_path(
        model=model,
        graph=graph,
        start_idx=start_idx,
        end_idx=end_idx,
        parent=parent,
        blocked=dynamic_blocked,
        max_steps=max_path_steps,
    )

    if not path:
        return _failure_result(
            extract_message,
            start_idx,
            end_idx,
            expanded_nodes=expanded_nodes,
            visited_nodes=sum(1 for v in tag.values() if v != NEW),
            t0=t0,
            dynamic_blocked_count=len(dynamic_blocked),
        )

    if verbose:
        print("      D* summary:")
        print(f"        expanded nodes        : {expanded_nodes:,}")
        print(f"        reached states        : {sum(1 for v in tag.values() if v != NEW):,}")
        print(f"        dynamic blocked nodes : {len(dynamic_blocked):,}")
        print(f"        path nodes            : {len(path):,}")
        print(f"        path cost             : {total_cost:.8g}")

    result = {
        "success": True,
        "algorithm": "dstar",
        "message": extract_message,
        "path_indices": [int(i) for i in path],
        "total_cost": float(total_cost),
        "expanded_nodes": int(expanded_nodes),
        "expanded_states": int(expanded_nodes),
        "visited_nodes": int(sum(1 for v in tag.values() if v != NEW)),
        "runtime_seconds": float(time.time() - t0),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "goal_cost_to_go": float(h.get(start_idx, math.inf)),
        "open_remaining": int(sum(1 for v in tag.values() if v == OPEN)),
        "closed_nodes": int(sum(1 for v in tag.values() if v == CLOSED)),
        "dynamic_blocked_nodes": int(len(dynamic_blocked)),
        "dstar_stop_when_start_closed": bool(stop_when_start_closed),
        "dstar_last_open_key": float(last_key) if math.isfinite(last_key) else None,
    }

    if return_expand_list and expanded_order is not None:
        # Keep this disabled by default because it can be large on dense grids.
        result["expanded_order"] = [int(i) for i in expanded_order]

    return result
