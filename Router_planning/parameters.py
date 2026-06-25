#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter loading utilities for the node-based SPSO route planner.

The project uses a simple Python-like key/value parameter file:

    KEY = value

Values can be numbers, booleans, strings, lists, tuples, or dictionaries.
Lines beginning with # are comments.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    # ------------------------------------------------------------------
    # Input / output
    # ------------------------------------------------------------------
    "MODEL_FILE": "model/senario1/model_senario1_cost_for_pathfinding.xyz",
    "OUTPUT_DIR": "output/SPSO",
    "ROUTE_SUMMARY_CSV": "SPSO_route_summary.csv",
    "SAVE_ROUTE_NODE_CSV": True,
    "SAVE_ROUTE_POINT_CSV": True,
    "PLOT_ROUTES": True,
    "PLOT_FULL_REPORT": True,
    "PLOT_ZOOM_REPORT": True,
    "ZOOM_MARGIN_M": 250.0,
    "LEGEND_MAX_ROUTES": 12,
    "PLOT_DPI": 220,

    # ------------------------------------------------------------------
    # Route selection
    # ------------------------------------------------------------------
    # These can be exact labels such as ["DB01"] or prefixes such as ["DB"].
    "START_LABELS": ["DB02"],
    "END_LABELS": ["DK02"],
    # exact | prefix | auto. auto: exact if possible, otherwise prefix.
    "LABEL_MATCH_MODE": "auto",
    # Optional explicit pairs. Empty list means all START_LABELS x END_LABELS.
    # Example: [("DB01", "DK03"), ("DB02", "DK05")]
    "ROUTE_PAIRS": [],
    # 0 means no limit.
    "MAX_ROUTES": 0,

    # User-facing alias: number of route alternatives per enabled direction.
    # N_ROUTE=30 gives 30 forward routes, and also 30 backward routes if
    # RUN_BACKWARD_PATHS=True.
    "N_ROUTE": 10,

    # ------------------------------------------------------------------
    # Multiple route alternatives / bidirectional lanes
    # ------------------------------------------------------------------
    "N_PATHS_PER_DIRECTION": 2,
    "RUN_FORWARD_PATHS": True,
    "RUN_BACKWARD_PATHS": True,
    "MAX_OVERLAP_RATIO": 0.10,
    "W_OVERLAP": 5.0e5,
    "MULTI_PATH_ATTEMPTS_PER_RANK": 3,
    "ENDPOINT_OVERLAP_IGNORE_RADIUS_M": 200.0,
    "OVERLAP_COMPARE_FORWARD_BACKWARD": True,

    # ------------------------------------------------------------------
    # Node model / no-fly logic
    # ------------------------------------------------------------------
    "COORDINATE_MODE": "auto",   # auto | lonlat | xy
    "NOFLY_SLOWNESS_THRESHOLD": 10.0,
    "NOFLY_THRESHOLD_MODE": "greater_equal",  # greater_equal | greater | equal
    "FORCE_ENDPOINTS_FLYABLE": True,
    "ENDPOINT_FORCE_PREFIXES": ["DB", "DK", "BD", "BK", "FLZ"],
    "NEAREST_NODE_MAX_DIST_M": 80.0,
    "SAMPLE_STEP_M": 25.0,

    # ------------------------------------------------------------------
    # SPSO search controls
    # ------------------------------------------------------------------
    "PATH_DIMENSION": "2d",      # 2d | 3d
    "N_WAYPOINTS": 12,
    "N_POP": 160,
    "MAX_IT": 120,
    "SEED": 42,
    "VERBOSE": True,
    "PRINT_EVERY": 10,
    "EARLY_STOP_ITERS": 35,

    # Spherical vector bounds. The path is represented by N_WAYPOINTS
    # spherical movement vectors, similar to the original MATLAB code.
    "R_MAX_FACTOR": 2.0,
    "ANGLE_RANGE_DEG": 75.0,
    "ELEVATION_ANGLE_RANGE_DEG": 20.0,
    "Z_MIN_M": 0.0,
    "Z_MAX_M": 0.0,

    # Standard PSO constants.
    "INERTIA_WEIGHT": 1.0,
    "INERTIA_DAMPING": 0.98,
    "C1": 1.5,
    "C2": 1.5,
    "VELOCITY_ALPHA": 0.5,

    # ------------------------------------------------------------------
    # Cost function weights
    # ------------------------------------------------------------------
    # total = W_LENGTH*distance + W_TIME*travel_time + W_NOFLY*nofly_hits
    #       + W_OUTSIDE*outside_hits + W_SMOOTH*smooth_penalty
    #       + W_REPEAT_NODE*repeat_penalty
    "W_LENGTH": 1.0,
    "W_TIME": 1.0,
    "W_NOFLY": 1.0e7,
    "W_OUTSIDE": 1.0e6,
    "W_SMOOTH": 25.0,
    "W_REPEAT_NODE": 100.0,
    "TURNING_MAX_DEG": 45.0,
    "CLIMB_MAX_DEG": 45.0,

    # ------------------------------------------------------------------
    # Optional post-processing
    # ------------------------------------------------------------------
    "REMOVE_DUPLICATE_NODES": True,
    "SIMPLIFY_COLLINEAR_NODES": False,
}


def _strip_inline_comment(line: str) -> str:
    """Remove comments outside single/double quotes."""
    in_single = False
    in_double = False
    escaped = False
    out = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _parse_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        lowered = text.lower()
        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False
        if lowered in {"none", "null"}:
            return None
        # Preserve unquoted values as strings, so paths can be written simply.
        return text


def load_params(params_file: str | Path = "params/SPSO.params") -> Dict[str, Any]:
    """Load parameters and merge with DEFAULTS."""
    params = dict(DEFAULTS)
    path = Path(params_file)

    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_inline_comment(raw_line)
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid parameter line {lineno}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing parameter key at line {lineno}: {raw_line!r}")
        params[key] = _parse_value(value)


    # Normalize route-count aliases.  N_ROUTE is the clearer user-facing name;
    # N_PATHS_PER_DIRECTION is kept for backward compatibility.
    if "N_ROUTE" in params and params.get("N_ROUTE") is not None:
        params["N_PATHS_PER_DIRECTION"] = int(params["N_ROUTE"])
    else:
        params["N_ROUTE"] = int(params.get("N_PATHS_PER_DIRECTION", 1))

    return params


def save_params_template(path: str | Path = "params/SPSO.params") -> None:
    """Write a minimal template parameter file from DEFAULTS."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Node-based SPSO route-planning parameters"]
    for key, value in DEFAULTS.items():
        lines.append(f"{key} = {value!r}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
