#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main controller for the node-based SPSO route planner.

Run:
    python main.py --params params/SPSO.params

Typical project layout:
    main.py
    parameters.py
    params/SPSO.params
    src/SPSO.py
    model/senario1/model_senario1_cost_for_pathfinding.xyz
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import time
from pathlib import Path
from typing import Iterable

from parameters import load_params
from src import SPSO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node-based SPSO route planner")
    parser.add_argument(
        "--params",
        default="params/SPSO.params",
        help="Path to SPSO parameter file. Default: params/SPSO.params",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional override for MODEL_FILE in the parameter file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for OUTPUT_DIR in the parameter file.",
    )
    parser.add_argument(
        "--npaths",
        type=int,
        default=None,
        help="Optional override for N_PATHS_PER_DIRECTION / N_ROUTE.",
    )
    parser.add_argument(
        "--n-route", "--n_route",
        dest="n_route",
        type=int,
        default=None,
        help="Number of route alternatives per enabled direction, e.g. 30.",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=None,
        help="Optional override for MAX_OVERLAP_RATIO, e.g. 0.10 for 10%.",
    )
    return parser.parse_args()


def _candidate_paths(path_text: str, base_dir: Path) -> Iterable[Path]:
    """Yield sensible candidate paths for cwd-relative and project-relative files."""
    p = Path(path_text).expanduser()
    if p.is_absolute():
        yield p
    else:
        yield p
        yield base_dir / p


def resolve_path(path_text: str, base_dir: Path) -> Path:
    """Resolve relative paths first against current working directory, then project dir."""
    for p in _candidate_paths(path_text, base_dir):
        if p.exists():
            return p.resolve()

    # Nothing exists yet. Return a useful absolute path for error messages.
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_model_path(path_text: str, base_dir: Path) -> Path:
    """
    Resolve MODEL_FILE robustly.

    Common user mistake:
        MODEL_FILE = model_a.xyz model/senario1/model_a.xyz

    The normal parser treats that as one string. If the full string does not
    exist, split it into tokens and use the first token that resolves to an
    existing file. This keeps the run alive while still warning the user.
    """
    whole = resolve_path(path_text, base_dir)
    if whole.exists():
        return whole

    try:
        tokens = shlex.split(path_text)
    except ValueError:
        tokens = path_text.split()

    if len(tokens) > 1:
        tried = []
        for token in tokens:
            candidate = resolve_path(token, base_dir)
            tried.append(str(candidate))
            if candidate.exists():
                print("      [WARNING] MODEL_FILE contains more than one path.")
                print(f"      [WARNING] Using existing model file: {candidate}")
                return candidate
        print("      [WARNING] MODEL_FILE appears to contain multiple paths, but none exist:")
        for item in tried:
            print(f"                - {item}")

    return whole


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    params_path = resolve_path(args.params, project_dir)
    params = load_params(params_path)

    if args.model is not None:
        params["MODEL_FILE"] = args.model
    if args.output is not None:
        params["OUTPUT_DIR"] = args.output
    if args.npaths is not None:
        params["N_PATHS_PER_DIRECTION"] = int(args.npaths)
        params["N_ROUTE"] = int(args.npaths)
    if args.n_route is not None:
        params["N_PATHS_PER_DIRECTION"] = int(args.n_route)
        params["N_ROUTE"] = int(args.n_route)
    if args.max_overlap is not None:
        params["MAX_OVERLAP_RATIO"] = float(args.max_overlap)

    # Keep N_ROUTE and N_PATHS_PER_DIRECTION synchronized even for older params files.
    if "N_ROUTE" in params and params.get("N_ROUTE") is not None:
        params["N_PATHS_PER_DIRECTION"] = int(params["N_ROUTE"])
    params["N_ROUTE"] = int(params.get("N_PATHS_PER_DIRECTION", 1))

    model_file = resolve_model_path(str(params["MODEL_FILE"]), project_dir)
    output_dir = resolve_path(str(params["OUTPUT_DIR"]), project_dir)
    params["OUTPUT_DIR"] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(params_path, output_dir / Path(params_path).name)
    except Exception:
        pass

    print("=" * 72)
    print("NODE-BASED SPSO ROUTE PLANNER")
    print("=" * 72)
    print(f"Project dir     : {project_dir}")
    print(f"Parameter file  : {params_path}")
    print(f"Model file      : {model_file}")
    print(f"Output dir      : {output_dir}")

    t0 = time.perf_counter()

    print("\n[1/4] Loading node model...")
    model = SPSO.load_node_model(model_file, params)
    xmin, xmax, ymin, ymax = model.bounds_xy
    print(f"      nodes          : {model.size:,}")
    print(f"      coordinate mode: {model.coord_mode}")
    print(f"      local bounds   : x=[{xmin:.1f}, {xmax:.1f}] m, y=[{ymin:.1f}, {ymax:.1f}] m")
    print(f"      flyable nodes  : {int(model.flyable.sum()):,}")
    print(f"      no-fly nodes   : {int((~model.flyable).sum()):,}")

    labeled = model.df[model.df["label"].astype(str) != "N"]
    if not labeled.empty:
        print("      labels         : " + ", ".join(labeled["label"].astype(str).tolist()))

    print("\n[2/4] Building route pairs...")
    pairs = SPSO.build_route_pairs(model, params)
    print(f"      route pairs    : {len(pairs)}")
    for i, (si, ei) in enumerate(pairs[:20], start=1):
        print(f"        {i:02d}. {model.labels[si]} -> {model.labels[ei]}")
    if len(pairs) > 20:
        print(f"        ... {len(pairs) - 20} more")

    n_paths = int(params.get("N_PATHS_PER_DIRECTION", 1))
    run_forward = bool(params.get("RUN_FORWARD_PATHS", True))
    run_backward = bool(params.get("RUN_BACKWARD_PATHS", False))
    per_pair = n_paths * int(run_forward) + n_paths * int(run_backward)
    print("\n[3/4] Running SPSO...")
    print(f"      N_ROUTE        : {n_paths} per enabled direction")
    print(f"      paths per pair : {per_pair} total alternatives")
    print(f"      forward        : {run_forward}")
    print(f"      backward       : {run_backward}")
    print(f"      max overlap    : {float(params.get('MAX_OVERLAP_RATIO', 0.10)):.3f}")
    results = SPSO.run_all(model, pairs, params)

    print("\n[4/4] Writing summary...")
    summary_path = SPSO.save_summary(results, output_dir, params)

    ok = sum(1 for r in results if r.success)
    failed = len(results) - ok
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("SPSO BATCH DONE")
    print("=" * 72)
    print(f"Successful routes : {ok}/{len(results)}")
    print(f"Failed routes     : {failed}/{len(results)}")
    print(f"Summary CSV       : {summary_path}")
    print(f"Total runtime     : {elapsed:.2f} s")


if __name__ == "__main__":
    main()
