#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main controller for Scenario 1 path finding.

Flow:
  1. Read parameters from parameters.py
  2. Load labelled model XYZ
  3. Select real start/end nodes, usually DB/DK
  4. Apply numeric flyability rule if requested
  5. Plot initiate model figure
  6. Build graph using only flyable nodes
  7. Snap real DB/DK to searchable grid nodes if needed
  8. Run selected algorithm
  9. Export path footprint files
 10. Plot model + path report
 11. Optional cleanup

Expected outputs:
  Single-path:
    output/dat/senario1/{algorithm}/path_senario1_{algorithm}.csv
    output/figures/senario1/{algorithm}/path_report_{algorithm}_from_DB01_to_DK01.png

  Multiple-path module named {algorithm}_multiple.py:
    output/dat/senario1/{algorithm}/multiple/{nvaluerun}/path_senario1_{algorithm}_multiple.csv
    output/dat/senario1/{algorithm}/multiple/{nvaluerun}/path_senario1_{algorithm}_multiple_rank_001.csv
    output/figures/senario1/{algorithm}/multiple/{nvaluerun}/path_report_{algorithm}_multiple_from_DB01_to_DK01.png
"""

from pathlib import Path
import importlib
import inspect
import json
import sys
import math
import time

import numpy as np
import pandas as pd

try:
    import parameters as parameter
except ModuleNotFoundError:
    import parameter

import parameters as prm
from src.paths import create_possible_paths

from src.model_io import *

from src.output_io import export_path_outputs

from src.plotting import (
    plot_path_report,
    plot_initiate_model,
    plot_multiple_paths_report,
    plot_multiple_path_time_histogram,
    plot_costmap_surface_outputs,
    plot_model_slowness_side_by_side,
    plot_path_zoom_diagnostic,
)

from src.costmap import build_predefined_costmap, save_costmap_outputs
from src.cleanup import cleanup_intermediate_files, print_cleanup_summary


def get_param(name, default=None):
    return getattr(parameter, name, default)


def format_elapsed_time(seconds):
    """Return a compact human-readable elapsed-time string."""
    seconds = float(seconds or 0.0)
    if seconds < 60.0:
        return f"{seconds:.2f} s"

    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {sec:.2f} s"

    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {sec:.2f} s"


def maybe_write_processing_time_json(output_file: Path, data: dict) -> None:
    """Write timing metadata without interrupting the pathfinding workflow."""
    try:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Could not write processing-time JSON: {exc}")


# Internal label prefix used only while building the graph.
# It lets old build_grid_graph() implementations, which only understand
# label-based blocking, obey the new numeric rule:
#     slowness < 10   -> flyable
#     slowness >= 10  -> no-fly
INTERNAL_NO_FLY_LABEL_PREFIX_DEFAULT = "__NOFLY_BY_SLOWNESS__"


def _label_startswith_any(label, prefixes) -> bool:
    """Return True if a label starts with one of the requested prefixes."""
    if prefixes is None:
        return True

    prefixes = tuple(prefixes)
    if not prefixes:
        return True

    text = str(label)
    return any(text.startswith(str(prefix)) for prefix in prefixes)


def _forced_flyable_mask_from_prefixes(model, prefixes):
    """Return True for DB/DK/FLZ or other prefixes forced flyable.

    This checks both label_prefix and label because some graph builders use
    label_prefix while others use label text.
    """
    prefixes = tuple(str(p) for p in (prefixes or ()) if str(p))
    if not prefixes:
        return np.zeros(len(model), dtype=bool)

    mask = np.zeros(len(model), dtype=bool)

    if "label_prefix" in model.columns:
        prefix_text = model["label_prefix"].astype(str)
        for prefix in prefixes:
            mask |= prefix_text.str.startswith(prefix).to_numpy(bool)

    if "label" in model.columns:
        label_text = model["label"].astype(str)
        for prefix in prefixes:
            mask |= label_text.str.startswith(prefix).to_numpy(bool)

    return np.asarray(mask, dtype=bool)


def _prefix_mask_from_label(model, prefixes):
    """Return True where label or label_prefix starts with one of prefixes."""
    return _forced_flyable_mask_from_prefixes(model, prefixes)


def _compute_no_fly_mask_from_slowness(
    model,
    threshold: float,
    mode: str = "greater_equal",
    tolerance: float = 0.0,
):
    """Compute the no-fly mask from the slowness threshold.

    Preferred rule for the new model:
        slowness >= threshold  -> no-fly
        slowness <  threshold  -> flyable
    """
    if "slowness" not in model.columns:
        raise ValueError(
            "BLOCK_BY_SLOWNESS_THRESHOLD=True requires a 'slowness' column "
            "in the loaded model."
        )

    slow = pd.to_numeric(model["slowness"], errors="coerce")
    mode = str(mode or "greater_equal").strip().lower()
    tolerance = float(tolerance or 0.0)

    invalid = ~np.isfinite(slow.to_numpy(float))

    # For the new model, keep the boundary exact:
    #     slowness < threshold   -> flyable
    #     slowness >= threshold  -> no-fly
    # The tolerance parameter is kept for compatibility but is not used by
    # the threshold modes below.
    if mode in ("greater_equal", "ge", ">=", "threshold"):
        no_fly = slow >= float(threshold)
    elif mode in ("greater", "gt", ">"):
        no_fly = slow > float(threshold)
    elif mode in ("less_equal", "le", "<="):
        no_fly = slow <= float(threshold)
    elif mode in ("less", "lt", "<"):
        no_fly = slow < float(threshold)
    else:
        raise ValueError(
            f"Unsupported NO_FLY_THRESHOLD_MODE={mode!r}. Use 'greater_equal' "
            "for the new model."
        )

    # Non-finite slowness is unsafe for graph traversal.
    no_fly = np.asarray(no_fly, dtype=bool) | invalid
    return no_fly


def add_flyability_columns_from_slowness(
    model,
    block_by_slowness_threshold: bool,
    threshold: float,
    mode: str,
    tolerance: float,
    always_flyable_prefixes=(),
):
    """Add is_no_fly / is_flyable columns to the model.

    These columns are useful for diagnostics, snapping checks, and plotting.
    The model labels are not changed by this function.
    """
    model = model.copy()

    if block_by_slowness_threshold:
        no_fly_mask = _compute_no_fly_mask_from_slowness(
            model=model,
            threshold=threshold,
            mode=mode,
            tolerance=tolerance,
        )
    else:
        no_fly_mask = np.zeros(len(model), dtype=bool)

    # Operational facilities can be explicitly forced flyable. This is useful
    # when DB/DK/FLZ points are located on a no-fly background cell but must
    # still be usable as search endpoints / service nodes.
    forced_flyable_mask = _forced_flyable_mask_from_prefixes(
        model=model,
        prefixes=always_flyable_prefixes,
    )
    no_fly_mask = np.asarray(no_fly_mask, dtype=bool) & ~forced_flyable_mask

    model["is_no_fly"] = no_fly_mask
    model["is_flyable"] = ~no_fly_mask
    model["is_no_fly_by_slowness"] = no_fly_mask
    model["is_flyable_by_slowness"] = ~no_fly_mask

    return model


def print_flyability_summary(
    model,
    block_by_slowness_threshold: bool,
    threshold: float,
    mode: str,
):
    """Print a compact summary of the new flyability rule."""
    print("[FLYABILITY] Numeric slowness rule:")
    print(f"      enabled      : {block_by_slowness_threshold}")

    if "slowness" not in model.columns:
        print("      slowness     : missing")
        return

    slow = pd.to_numeric(model["slowness"], errors="coerce")
    finite = slow[np.isfinite(slow)]

    print(f"      rule         : slowness {mode} {threshold:g} => no-fly")
    if len(finite) > 0:
        print(f"      min slowness : {finite.min():.6g}")
        print(f"      max slowness : {finite.max():.6g}")

    if "is_flyable" in model.columns and "is_no_fly" in model.columns:
        n_flyable = int(model["is_flyable"].sum())
        n_no_fly = int(model["is_no_fly"].sum())
        print(f"      flyable nodes: {n_flyable:,}")
        print(f"      no-fly nodes : {n_no_fly:,}")

        if "label" in model.columns:
            try:
                label_counts = (
                    model.loc[model["is_no_fly"], "label"]
                    .astype(str)
                    .str.extract(r"^([A-Za-z_]+)", expand=False)
                    .fillna("UNKNOWN")
                    .value_counts()
                    .head(10)
                )
                if not label_counts.empty:
                    print("      no-fly by label prefix, top entries:")
                    for key, value in label_counts.items():
                        print(f"        {key:12s}: {int(value):,}")
            except Exception:
                pass


def make_graph_model_with_slowness_blocking(
    model,
    block_by_slowness_threshold: bool,
    threshold: float,
    mode: str,
    tolerance: float,
    block_prefixes,
    internal_no_fly_prefix: str = INTERNAL_NO_FLY_LABEL_PREFIX_DEFAULT,
    always_flyable_prefixes=(),
):
    """Create the model used only for build_grid_graph().

    Many older graph builders only block nodes by label prefix. To support the
    new numeric model without editing src/model_io.py, this function creates a
    temporary copy in which nodes with slowness >= threshold get an internal
    no-fly label prefix. The original model used for plotting/export is kept
    unchanged.
    """
    block_prefixes = tuple(block_prefixes or ())

    if not block_by_slowness_threshold:
        return model, block_prefixes

    if "label" not in model.columns:
        raise ValueError("The model must contain a 'label' column.")

    no_fly_mask = _compute_no_fly_mask_from_slowness(
        model=model,
        threshold=threshold,
        mode=mode,
        tolerance=tolerance,
    )

    forced_flyable_mask = _forced_flyable_mask_from_prefixes(
        model=model,
        prefixes=always_flyable_prefixes,
    )
    no_fly_mask = np.asarray(no_fly_mask, dtype=bool) & ~forced_flyable_mask

    graph_model = model.copy()
    graph_model["label"] = graph_model["label"].astype(str)

    # Important: some build_grid_graph() versions block using `label`, while
    # others block using `label_prefix`.  Update both so the numeric slowness
    # rule really reaches the graph builder.  This should make graph
    # traversable nodes close to the flyable-node count, not all model nodes.
    graph_model.loc[no_fly_mask, "label"] = (
        str(internal_no_fly_prefix) + graph_model.loc[no_fly_mask, "label"]
    )
    if "label_prefix" in graph_model.columns:
        graph_model["label_prefix"] = graph_model["label_prefix"].astype(str)
        graph_model.loc[no_fly_mask, "label_prefix"] = str(internal_no_fly_prefix)

    effective_block_prefixes = tuple(
        dict.fromkeys(block_prefixes + (str(internal_no_fly_prefix),))
    )

    return graph_model, effective_block_prefixes


def _nearest_valid_index_for_endpoint(
    model,
    source_idx: int,
    valid_indices,
    target_prefixes=None,
):
    """Find the nearest valid graph node to source_idx.

    This is a safety fallback when a snapping helper returns a node that is not
    in graph['valid_indices'] or violates the numeric flyability rule.
    """
    valid = [int(i) for i in valid_indices]
    if not valid:
        raise ValueError("Graph has no valid/traversable nodes.")

    if target_prefixes:
        target_prefixes = tuple(target_prefixes)
        valid = [
            i for i in valid
            if _label_startswith_any(model.loc[i, "label"], target_prefixes)
        ]

    if not valid:
        raise ValueError(
            "No valid graph nodes match SNAP_TARGET_PREFIXES. "
            "Try adding 'N' to SNAP_TARGET_PREFIXES or check the model."
        )

    coord_cols = ["x", "y"]
    if "z" in model.columns:
        coord_cols.append("z")

    source = model.loc[int(source_idx), coord_cols].to_numpy(dtype=float)
    candidates = model.loc[valid, coord_cols].to_numpy(dtype=float)

    # If lon/lat degrees are used, scale x/y approximately into meters.
    if coord_cols[:2] == ["x", "y"] and _is_lonlat_xy(model):
        lat0 = math.radians(float(model.loc[int(source_idx), "y"]))
        scale = np.ones(len(coord_cols), dtype=float)
        scale[0] = 111_320.0 * math.cos(lat0)
        scale[1] = 110_540.0
        source = source * scale
        candidates = candidates * scale

    d2 = np.sum((candidates - source) ** 2, axis=1)
    return int(valid[int(np.argmin(d2))])


def ensure_endpoint_indices_are_traversable(
    model,
    graph,
    start_idx: int,
    end_idx: int,
    search_start_idx: int,
    search_end_idx: int,
    snap: bool,
    snap_only_to_flyable: bool,
    target_prefixes,
):
    """Make sure search endpoints are inside graph['valid_indices'].

    This protects the new numeric rule from older snapping helpers that might
    return a labelled node without checking slowness < 10.
    """
    if not snap_only_to_flyable:
        return int(search_start_idx), int(search_end_idx)

    valid_indices = graph.get("valid_indices", set())
    valid_indices = {int(i) for i in valid_indices}

    fixed_start = int(search_start_idx)
    fixed_end = int(search_end_idx)

    if fixed_start not in valid_indices:
        if not snap:
            raise ValueError(
                f"Search start index {fixed_start} is not flyable/traversable. "
                "Enable SNAP_START_END_TO_GRID or choose a flyable START node."
            )
        fixed_start = _nearest_valid_index_for_endpoint(
            model=model,
            source_idx=int(start_idx),
            valid_indices=valid_indices,
            target_prefixes=target_prefixes,
        )
        print("      Start re-snapped to flyable node:")
        print(f"        old search start: {search_start_idx} | {model.loc[int(search_start_idx), 'label']}")
        print(f"        new search start: {fixed_start} | {model.loc[fixed_start, 'label']}")

    if fixed_end not in valid_indices:
        if not snap:
            raise ValueError(
                f"Search end index {fixed_end} is not flyable/traversable. "
                "Enable SNAP_START_END_TO_GRID or choose a flyable END node."
            )
        fixed_end = _nearest_valid_index_for_endpoint(
            model=model,
            source_idx=int(end_idx),
            valid_indices=valid_indices,
            target_prefixes=target_prefixes,
        )
        print("      End re-snapped to flyable node:")
        print(f"        old search end: {search_end_idx} | {model.loc[int(search_end_idx), 'label']}")
        print(f"        new search end: {fixed_end} | {model.loc[fixed_end, 'label']}")

    return fixed_start, fixed_end


def validate_path_uses_only_flyable_nodes(
    model,
    path_indices,
    threshold: float,
    mode: str,
    tolerance: float,
    context: str = "path",
    always_flyable_prefixes=(),
):
    """Raise an error if a path contains no-fly nodes by numeric rule."""
    if not path_indices:
        return

    no_fly_mask = _compute_no_fly_mask_from_slowness(
        model=model,
        threshold=threshold,
        mode=mode,
        tolerance=tolerance,
    )
    forced_flyable_mask = _forced_flyable_mask_from_prefixes(
        model=model,
        prefixes=always_flyable_prefixes,
    )
    no_fly_mask = np.asarray(no_fly_mask, dtype=bool) & ~forced_flyable_mask

    bad = [int(i) for i in path_indices if bool(no_fly_mask[int(i)])]
    if bad:
        example = bad[:10]
        raise RuntimeError(
            f"{context} contains {len(bad)} no-fly nodes by slowness threshold. "
            f"Example indices: {example}. Check graph blocking logic."
        )


def _is_lonlat_xy(model) -> bool:
    """Detect whether model x/y are lon/lat degrees."""
    try:
        x = pd.to_numeric(model["x"], errors="coerce")
        y = pd.to_numeric(model["y"], errors="coerce")
        return (
            x.dropna().between(-180.0, 180.0).all()
            and y.dropna().between(-90.0, 90.0).all()
        )
    except Exception:
        return False


def _segment_distance_m(model, idx_a: int, idx_b: int, use_z: bool = True) -> float:
    """Distance between two model nodes in meters.

    If x/y look like lon/lat, use a local equirectangular approximation.
    Otherwise use Euclidean x/y directly as meters.
    """
    x1 = float(model.loc[idx_a, "x"])
    y1 = float(model.loc[idx_a, "y"])
    x2 = float(model.loc[idx_b, "x"])
    y2 = float(model.loc[idx_b, "y"])

    if _is_lonlat_xy(model):
        lat0 = math.radians(0.5 * (y1 + y2))
        dx = (x2 - x1) * 111_320.0 * math.cos(lat0)
        dy = (y2 - y1) * 110_540.0
    else:
        dx = x2 - x1
        dy = y2 - y1

    if use_z and "z" in model.columns:
        try:
            dz = float(model.loc[idx_b, "z"] - model.loc[idx_a, "z"])
        except Exception:
            dz = 0.0
    else:
        dz = 0.0

    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def build_path_step_distance_table(model, path_indices, path_rank=None):
    """Build per-step path table with cumulative distance from start.

    Columns added for every path step:
      - segment_distance_m
      - distance_from_start_m
      - distance_from_start_km
      - segment_traveltime_s
      - traveltime_from_start_s
      - traveltime_from_start_min

    The first step always has segment distance = 0.
    """
    indices = [int(i) for i in path_indices]
    rows = []
    cumulative_distance_m = 0.0
    cumulative_traveltime_s = 0.0

    for step, idx in enumerate(indices):
        if step == 0:
            segment_distance_m = 0.0
            segment_traveltime_s = 0.0
        else:
            prev_idx = indices[step - 1]
            segment_distance_m = _segment_distance_m(model, prev_idx, idx, use_z=True)

            if "slowness" in model.columns:
                slow_prev = float(model.loc[prev_idx, "slowness"])
                slow_now = float(model.loc[idx, "slowness"])
                segment_traveltime_s = segment_distance_m * 0.5 * (slow_prev + slow_now)
            else:
                segment_traveltime_s = 0.0

        cumulative_distance_m += segment_distance_m
        cumulative_traveltime_s += segment_traveltime_s

        row = {
            "path_step": int(step),
            "node_index": int(idx),
            "segment_distance_m": float(segment_distance_m),
            "distance_from_start_m": float(cumulative_distance_m),
            "distance_from_start_km": float(cumulative_distance_m / 1000.0),
            "segment_traveltime_s": float(segment_traveltime_s),
            "traveltime_from_start_s": float(cumulative_traveltime_s),
            "traveltime_from_start_min": float(cumulative_traveltime_s / 60.0),
        }

        if path_rank is not None:
            row["path_rank"] = int(path_rank)

        for col in ("x", "y", "z", "slowness", "label"):
            if col in model.columns:
                value = model.loc[idx, col]
                try:
                    if col != "label":
                        value = float(value)
                except Exception:
                    pass
                row[col] = value

        rows.append(row)

    # Put rank first when available.
    df = pd.DataFrame(rows)
    if "path_rank" in df.columns:
        ordered = ["path_rank", "path_step", "node_index"]
        rest = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]

    return df


def add_path_step_distance_to_exported_files(
    model,
    path_indices,
    exported,
    path_rank=None,
    write_extra_step_files=True,
    overwrite_csv_xyz=True,
):
    """Add per-step cumulative distance table to exported CSV/XYZ files.

    This keeps the existing output naming style, but enriches the CSV and XYZ
    with distance_from_start_m for every path_step.  Extra *_path_steps files
    are also written for quick checking.
    """
    step_df = build_path_step_distance_table(
        model=model,
        path_indices=path_indices,
        path_rank=path_rank,
    )

    if not isinstance(exported, dict):
        exported = {}

    csv_file = exported.get("csv", None)
    xyz_file = exported.get("xyz", None)

    if overwrite_csv_xyz:
        if csv_file:
            csv_path = Path(csv_file)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            step_df.to_csv(csv_path, index=False, float_format="%.8f")
            exported["csv"] = str(csv_path)

        if xyz_file:
            xyz_path = Path(xyz_file)
            xyz_path.parent.mkdir(parents=True, exist_ok=True)
            step_df.to_csv(
                xyz_path,
                sep=" ",
                index=False,
                header=True,
                float_format="%.8f",
            )
            exported["xyz"] = str(xyz_path)

    if write_extra_step_files:
        base_path = None
        if csv_file:
            base_path = Path(csv_file)
        elif xyz_file:
            base_path = Path(xyz_file)

        if base_path is not None:
            step_csv = base_path.with_name(base_path.stem + "_path_steps.csv")
            step_xyz = base_path.with_name(base_path.stem + "_path_steps.xyz")

            step_df.to_csv(step_csv, index=False, float_format="%.8f")
            step_df.to_csv(
                step_xyz,
                sep=" ",
                index=False,
                header=True,
                float_format="%.8f",
            )

            exported["path_steps_csv"] = str(step_csv)
            exported["path_steps_xyz"] = str(step_xyz)

    return exported




def parse_rank_selection(value):
    """Parse ranked-path plot selection.

    Accepted examples:
      PLOT_MULTIPLE_RANKS = "all"
      PLOT_MULTIPLE_RANKS = 10
      PLOT_MULTIPLE_RANKS = [1, 2, 5, 10]
    """
    if value is None:
        return "all"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("all", "*"):
            return "all"
        if "," in v:
            return sorted({int(x.strip()) for x in v.split(",") if x.strip()})
        return [int(v)]
    if isinstance(value, int):
        return [int(value)]
    try:
        return sorted({int(x) for x in value})
    except TypeError:
        return [int(value)]


def read_path_indices_from_saved_table(model, path_file: Path):
    """Read path indices from an exported CSV/XYZ path file.

    Preferred column is node_index. If the file has no node_index column,
    the function falls back to nearest x/y/z matching against the model.
    """
    if not path_file.exists():
        raise FileNotFoundError(f"Path file not found: {path_file}")

    df = pd.read_csv(path_file, sep=None, engine="python", comment="#")

    if "node_index" in df.columns:
        return [int(i) for i in df["node_index"].dropna().astype(int).tolist()]

    # Fallback for old files without node_index.
    if not {"x", "y"}.issubset(set(df.columns)):
        raise ValueError(
            f"Cannot read node indices from {path_file}. "
            "The file must contain node_index or x/y columns."
        )

    coords_model = model[["x", "y"]].to_numpy(float)
    coords_path = df[["x", "y"]].to_numpy(float)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords_model)
        _, idx = tree.query(coords_path, k=1)
        return [int(i) for i in idx]
    except Exception:
        indices = []
        for x, y in coords_path:
            d2 = (coords_model[:, 0] - x) ** 2 + (coords_model[:, 1] - y) ** 2
            indices.append(int(np.argmin(d2)))
        return indices


def find_saved_path_files_for_plot(
    output_dir: Path,
    path_name: str,
    algorithm_name: str,
    is_multiple_algorithm: bool,
    rank_selection="all",
):
    """Find existing exported path files for plot-only mode."""
    output_dir = Path(output_dir)

    if is_multiple_algorithm:
        pattern = f"{path_name}_{algorithm_name}_rank_*.csv"
        files = sorted(output_dir.glob(pattern))

        # If no ranked files exist, fall back to the best-path file.
        if not files:
            best_file = output_dir / f"{path_name}_{algorithm_name}.csv"
            return [(1, best_file)] if best_file.exists() else []

        if rank_selection == "all":
            selected_files = files
        else:
            wanted = {int(r) for r in rank_selection}
            selected_files = []
            for f in files:
                stem = f.stem
                try:
                    rank = int(stem.split("_rank_")[-1])
                except Exception:
                    continue
                if rank in wanted:
                    selected_files.append(f)

        out = []
        for f in selected_files:
            try:
                rank = int(f.stem.split("_rank_")[-1])
            except Exception:
                rank = len(out) + 1
            out.append((rank, f))
        return out

    single_file = output_dir / f"{path_name}_{algorithm_name}.csv"
    return [(None, single_file)] if single_file.exists() else []


def plot_saved_path_reports(
    model,
    output_dir: Path,
    algorithm_figure_dir: Path,
    path_name: str,
    algorithm_name: str,
    is_multiple_algorithm: bool,
    safe_start_label: str,
    safe_end_label: str,
    max_model_points: int,
    dpi: int,
    model_alpha: float,
    model_marker_size: float,
    path_line_width: float,
    plot_model_as_flyable_nofly: bool,
    plot_no_fly_prefixes,
    plot_no_fly_slowness_threshold: float,
    plot_show_flz_overlay: bool,
    always_flyable_prefixes,
    result=None,
    rank_selection="all",
):
    """Plot existing saved path files without rerunning the algorithm."""
    files = find_saved_path_files_for_plot(
        output_dir=output_dir,
        path_name=path_name,
        algorithm_name=algorithm_name,
        is_multiple_algorithm=is_multiple_algorithm,
        rank_selection=rank_selection,
    )

    if not files:
        print(f"[WARNING] No saved path CSV files found in: {output_dir}")
        return []

    algorithm_figure_dir.mkdir(parents=True, exist_ok=True)
    plotted = []

    for rank, path_file in files:
        path_indices = read_path_indices_from_saved_table(model, path_file)

        if is_multiple_algorithm and rank is not None:
            figure_file = (
                algorithm_figure_dir
                / f"path_report_{algorithm_name}_rank_{rank:03d}_from_{safe_start_label}_to_{safe_end_label}.png"
            )
            plot_result = dict(result or {})
            plot_result["rank"] = int(rank)
        else:
            figure_file = (
                algorithm_figure_dir
                / f"path_report_{algorithm_name}_from_{safe_start_label}_to_{safe_end_label}.png"
            )
            plot_result = result

        print(f"      Plot path report: {figure_file}")
        plot_path_report(
            model=model,
            path_indices=path_indices,
            figure_file=figure_file,
            algorithm_name=algorithm_name,
            max_model_points=max_model_points,
            dpi=dpi,
            model_alpha=model_alpha,
            model_marker_size=model_marker_size,
            path_line_width=path_line_width,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=always_flyable_prefixes,
            result=plot_result,
        )
        plotted.append(str(figure_file))

    return plotted


def normalize_algorithm_names(value):
    """Return a clean list of algorithm module names.

    Accepts either:
        ALGORITHM = "dijkstra"
        ALGORITHM = ["dijkstra", "astar"]
        ALGORITHM = ("dijkstra", "astar")
    """
    if isinstance(value, str):
        names = [value]
    else:
        try:
            names = list(value)
        except TypeError:
            names = [value]

    clean = []
    for item in names:
        name = str(item).strip()
        if name and name not in clean:
            clean.append(name)

    if not clean:
        clean = ["astar"]

    return clean


def _param_is_none_like(value):
    """Return True for None-like START/END settings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null", "all", "auto")
    return False


def _is_fmm2d_all_facility_request(algorithm_name):
    """True when FMM2D should compute/return all DB/DK facility-pair paths."""
    if str(algorithm_name).strip().lower() != "fmm2d":
        return False

    pair_mode = str(get_param("FMM2D_PAIR_MODE", "selected")).strip().lower()
    return_mode = str(get_param("FMM2D_PAIR_RETURN_MODE", "requested")).strip().lower()

    facility_modes = (
        "facility", "facility_pairs", "facility_library",
        "all_facility_pairs", "all_pairs",
    )
    all_modes = ("all", "library", "all_pairs", "auto")

    no_specific_start_end = (
        _param_is_none_like(get_param("START_LABEL", None))
        and _param_is_none_like(get_param("END_LABEL", None))
        and _param_is_none_like(get_param("START_COORD", None))
        and _param_is_none_like(get_param("END_COORD", None))
    )

    return pair_mode in facility_modes and return_mode in all_modes and no_specific_start_end


def _first_index_with_label_prefix(model, prefixes):
    """Return the first row index whose label or label_prefix starts with prefixes."""
    prefixes = tuple(str(p).upper() for p in (prefixes or ()) if str(p))
    if not prefixes:
        raise ValueError("No prefixes supplied for facility dummy node selection.")

    mask = np.zeros(len(model), dtype=bool)

    if "label" in model.columns:
        labels = model["label"].fillna("").astype(str).str.upper()
        for prefix in prefixes:
            mask |= labels.str.startswith(prefix).to_numpy(bool)

    if "label_prefix" in model.columns:
        label_prefix = model["label_prefix"].fillna("").astype(str).str.upper()
        for prefix in prefixes:
            mask |= label_prefix.str.startswith(prefix).to_numpy(bool)

    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"Could not find any facility node with prefixes={prefixes}")
    return int(indices[0])


def _plot_all_facility_paths_report(
    model,
    ranked_paths,
    figure_file: Path,
    algorithm_name: str,
    max_model_points: int,
    dpi: int,
    model_alpha: float,
    model_marker_size: float,
    path_line_width: float,
    plot_model_as_flyable_nofly: bool,
    plot_no_fly_slowness_threshold: float,
):
    """Plot all FMM2D facility-pair fastest paths with all source/target nodes.

    This avoids the normal single-pair plot title/legend, which is misleading
    when START_LABEL=None and END_LABEL=None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if "lon" in model.columns and "lat" in model.columns:
        xcol, ycol = "lon", "lat"
    elif "x" in model.columns and "y" in model.columns:
        xcol, ycol = "x", "y"
    else:
        raise ValueError("Cannot plot all facility paths: model must contain lon/lat or x/y columns.")

    x = pd.to_numeric(model[xcol], errors="coerce").to_numpy(dtype=float, copy=True)
    y = pd.to_numeric(model[ycol], errors="coerce").to_numpy(dtype=float, copy=True)
    n_model = len(model)

    finite = np.isfinite(x) & np.isfinite(y)
    if n_model > int(max_model_points) > 0:
        keep_idx = np.flatnonzero(finite)
        step = max(1, int(np.ceil(len(keep_idx) / float(max_model_points))))
        plot_idx = keep_idx[::step]
    else:
        plot_idx = np.flatnonzero(finite)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=int(dpi))

    if plot_model_as_flyable_nofly and "slowness" in model.columns:
        slow = pd.to_numeric(model["slowness"], errors="coerce").to_numpy(dtype=float, copy=True)
        fly = np.isfinite(slow) & (slow < float(plot_no_fly_slowness_threshold))
        nofly = np.isfinite(slow) & (slow >= float(plot_no_fly_slowness_threshold))
        pidx = plot_idx
        ax.scatter(
            x[pidx[fly[pidx]]], y[pidx[fly[pidx]]],
            s=float(model_marker_size), marker="o", alpha=float(model_alpha),
            color="#79c79a", edgecolors="none", label=f"Flyable: s < {plot_no_fly_slowness_threshold:g}", zorder=1,
        )
        ax.scatter(
            x[pidx[nofly[pidx]]], y[pidx[nofly[pidx]]],
            s=float(model_marker_size), marker="s", alpha=float(model_alpha),
            color="red", edgecolors="none", label=f"No-fly: s >= {plot_no_fly_slowness_threshold:g}", zorder=1,
        )
    else:
        ax.scatter(x[plot_idx], y[plot_idx], s=float(model_marker_size), alpha=float(model_alpha), color="0.7", edgecolors="none", label="Model nodes", zorder=1)

    valid_paths = []
    for item in ranked_paths or []:
        path = item.get("path_indices", [])
        if path and len(path) >= 2:
            valid_paths.append(item)

    if not valid_paths:
        raise ValueError("No valid ranked_paths available for all-facility plot.")

    max_rank = max(int(item.get("rank", i + 1)) for i, item in enumerate(valid_paths))
    norm = Normalize(vmin=1, vmax=max(1, max_rank))
    cmap = plt.get_cmap("turbo") if max_rank > 1 else plt.get_cmap("viridis")

    source_indices = set()
    target_indices = set()
    labels_by_idx = {}

    for i, item in enumerate(valid_paths):
        rank = int(item.get("rank", i + 1))
        path = [int(v) for v in item.get("path_indices", []) if 0 <= int(v) < n_model]
        if len(path) < 2:
            continue

        color = cmap(norm(rank))
        alpha = 0.78 if len(valid_paths) <= 80 else 0.45
        ax.plot(
            x[path], y[path],
            color=color,
            linewidth=max(0.8, float(path_line_width) * (0.65 if len(valid_paths) > 80 else 1.0)),
            alpha=alpha,
            zorder=4,
        )

        src = int(item.get("source_idx", path[0]))
        dst = int(item.get("target_idx", path[-1]))
        if src != dst:
            source_indices.add(src)
            target_indices.add(dst)
            labels_by_idx[src] = str(item.get("source_label", model.loc[src, "label"] if "label" in model.columns else src))
            labels_by_idx[dst] = str(item.get("target_label", model.loc[dst, "label"] if "label" in model.columns else dst))

    both_indices = source_indices & target_indices
    source_only = sorted(source_indices - both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))
    target_only = sorted(target_indices - both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))
    both_sorted = sorted(both_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))

    def _scatter_indices(indices, marker, label, face, edge, size, z):
        if not indices:
            return
        idx = np.asarray(indices, dtype=int)
        ax.scatter(
            x[idx], y[idx],
            s=size, marker=marker,
            facecolors=face, edgecolors=edge,
            linewidths=1.7,
            label=label,
            zorder=z,
        )

    _scatter_indices(source_only, "*", "Start facilities", "yellow", "black", 180, 8)
    _scatter_indices(target_only, "s", "End facilities", "none", "blue", 120, 8)
    if both_sorted:
        _scatter_indices(both_sorted, "D", "Start + end facilities", "white", "black", 105, 9)
        _scatter_indices(both_sorted, "*", None, "yellow", "black", 155, 10)

    # Label all source/end facilities once.
    for pos, idx in enumerate(sorted(source_indices | target_indices, key=lambda ii: labels_by_idx.get(ii, str(ii)))):
        if idx < 0 or idx >= n_model or not np.isfinite(x[idx]) or not np.isfinite(y[idx]):
            continue
        dx = 5 if pos % 2 == 0 else -5
        dy = 5 if (pos // 2) % 2 == 0 else -7
        ax.annotate(
            labels_by_idx.get(idx, str(idx)),
            (x[idx], y[idx]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            zorder=12,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.83, pad=0.02)
    cbar.set_label("Rank index")

    unique_sources = len(source_indices)
    unique_targets = len(target_indices)
    pair_count = len(valid_paths)
    ax.set_title(f"Scenario 1 all fastest facility paths - {algorithm_name}", fontsize=18, fontweight="bold")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(False)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black", fontsize=9)

    text = (
        f"Paths plotted: {pair_count}\n"
        f"Unique starts: {unique_sources}\n"
        f"Unique ends: {unique_targets}\n"
        f"Self/same-label pairs: skipped"
    )
    ax.text(
        0.99, 0.99, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.82),
        zorder=20,
    )

    fig.tight_layout()
    fig.savefig(figure_file, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return str(figure_file)



def run_one_algorithm(algorithm_name):
    run_total_start_time = time.perf_counter()
    algorithm_search_elapsed_s = 0.0
    print_processing_time = bool(get_param("PRINT_PROCESSING_TIME", True))
    save_processing_time_json = bool(get_param("SAVE_PROCESSING_TIME_JSON", True))

    algorithm_name = str(algorithm_name).strip()
    if not algorithm_name:
        raise ValueError("Empty algorithm name in ALGORITHM list.")

    # ============================================================
    # Basic paths and algorithm
    # ============================================================
    model_file = Path(
        get_param(
            "MODEL_FILE",
            Path("model") / "senario1" / "model_senario1_with_label.xyz",
        )
    )

    dat_root_dir = Path(
        get_param(
            "DAT_ROOT_DIR",
            Path("output") / "dat" / "senario1",
        )
    )

    figure_root_dir = Path(
        get_param(
            "FIGURE_ROOT_DIR",
            Path("output") / "figures" / "senario1",
        )
    )

    # ============================================================
    # Output folder rule
    # ============================================================
    # Single-path algorithms:
    #   output/dat/senario1/dijkstra/
    #   output/figures/senario1/dijkstra/
    #
    # Multiple-path algorithms named {algorithm}_multiple.py:
    #   output/dat/senario1/{algorithm}/multiple/{nvaluerun}/
    #   output/figures/senario1/{algorithm}/multiple/{nvaluerun}/
    #
    # Example:
    #   ALGORITHM = ["dijkstra", "astar_multiple"]
    #   MULTI_PATH_K_PATHS = 100
    # gives:
    #   output/dat/senario1/dijkstra/
    #   output/dat/senario1/astar/multiple/100/
    is_multiple_algorithm = algorithm_name.endswith("_multiple")
    algorithm_base_name = (
        algorithm_name[: -len("_multiple")]
        if is_multiple_algorithm
        else algorithm_name
    )

    multiple_run_value = int(
        get_param(
            "MULTIPLE_OUTPUT_VALUE",
            get_param("MULTI_PATH_K_PATHS", 100),
        )
    )

    if is_multiple_algorithm:
        output_dir = dat_root_dir / algorithm_base_name / "multiple" / str(multiple_run_value)
        algorithm_figure_dir = figure_root_dir / algorithm_base_name / "multiple" / str(multiple_run_value)
    else:
        output_dir = dat_root_dir / algorithm_name
        algorithm_figure_dir = figure_root_dir / algorithm_name



    # ============================================================
    # Start/end settings
    # ============================================================
    start_label = get_param("START_LABEL", None)
    end_label = get_param("END_LABEL", None)
    start_coord = get_param("START_COORD", None)
    end_coord = get_param("END_COORD", None)

    # FMM2D can compute and plot all fastest DB/DK facility-pair paths.
    # In that mode START_LABEL/END_LABEL/START_COORD/END_COORD may all be None.
    # main.py will choose a temporary DB/DK dummy only for legacy plotting/graph
    # bookkeeping, while FMM2D returns the real all-pair path library.
    fmm2d_all_facility_mode = _is_fmm2d_all_facility_request(algorithm_name)

    # Figure files:
    # output/figures/senario1/{algorithm}/
    if fmm2d_all_facility_mode:
        safe_start_label = "ALL"
        safe_end_label = "FACILITY_PAIRS"
    else:
        safe_start_label = str(start_label).replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_end_label = str(end_label).replace("/", "_").replace("\\", "_").replace(" ", "_")

    report_figure_file = (
        algorithm_figure_dir
        / f"path_report_{algorithm_name}_from_{safe_start_label}_to_{safe_end_label}.png"
    )
    
    snap_start_end_to_grid = bool(get_param("SNAP_START_END_TO_GRID", True))
    snap_target_prefixes = tuple(get_param("SNAP_TARGET_PREFIXES", ("N", "FLZ", "DB", "DK")))
    snap_only_to_flyable = bool(get_param("SNAP_ONLY_TO_FLYABLE", True))
    include_real_start_end = bool(get_param("INCLUDE_REAL_START_END_IN_OUTPUT", True))
    if fmm2d_all_facility_mode:
        # All facility-pair paths already contain their own source/target nodes.
        # Do not add the temporary dummy DB/DK to every exported path.
        include_real_start_end = False

    # Operational facilities can be forced flyable even if they sit on a
    # no-fly/slowness background cell.  For the current rule this should be
    # ("DB", "DK", "FLZ").
    always_flyable_prefixes = tuple(get_param("ALWAYS_FLYABLE_PREFIXES", ()))
    force_search_start_end_flyable = bool(
        get_param("FORCE_SEARCH_START_END_FLYABLE", False)
    )

    endpoint_flyable_buffer_radius_m = float(
        get_param("ENDPOINT_FLYABLE_BUFFER_RADIUS_M", 0.0)
    )
    endpoint_flyable_buffer_mode = str(
        get_param("ENDPOINT_FLYABLE_BUFFER_MODE", "both")
    ).lower()

    # ============================================================
    # Graph rules
    # ============================================================
    # New model rule:
    #     slowness < 10.0   -> flyable
    #     slowness >= 10.0  -> no-fly
    #
    # Label-based blocking is still accepted for backward compatibility,
    # but for the new model it should usually be empty: BLOCK_LABEL_PREFIXES = ().
    block_prefixes = tuple(get_param("BLOCK_LABEL_PREFIXES", ()))
    high_cost_prefixes = tuple(get_param("HIGH_COST_LABEL_PREFIXES", ()))
    flz_cost_factor = float(get_param("FLZ_COST_FACTOR", 1.0))

    block_by_slowness_threshold = bool(
        get_param("BLOCK_BY_SLOWNESS_THRESHOLD", True)
    )
    no_fly_slowness_threshold = float(
        get_param("NO_FLY_SLOWNESS_THRESHOLD", 10.0)
    )
    no_fly_threshold_mode = str(
        get_param("NO_FLY_THRESHOLD_MODE", "greater_equal")
    ).strip().lower()
    no_fly_slowness_tolerance = float(
        get_param("NO_FLY_SLOWNESS_TOLERANCE", 0.0)
    )
    internal_no_fly_label_prefix = str(
        get_param(
            "INTERNAL_NO_FLY_LABEL_PREFIX",
            INTERNAL_NO_FLY_LABEL_PREFIX_DEFAULT,
        )
    )

    connectivity_2d = int(get_param("CONNECTIVITY_2D", 8))
    connectivity_3d = int(get_param("CONNECTIVITY_3D", 26))

    graph_neighbor_mode = str(get_param("GRAPH_NEIGHBOR_MODE", "kdtree")).lower()
    kdtree_radius_factor = float(get_param("KDTREE_RADIUS_FACTOR", 1.60))
    kdtree_max_neighbors_2d = int(get_param("KDTREE_MAX_NEIGHBORS_2D", 8))
    kdtree_max_neighbors_3d = int(get_param("KDTREE_MAX_NEIGHBORS_3D", 26))

    # ============================================================
    # Output naming
    # ============================================================
    path_name = str(get_param("PATH_NAME", "path_senario1"))

    save_path_step_distance = bool(get_param("SAVE_PATH_STEP_DISTANCE", True))
    write_extra_path_step_files = bool(get_param("WRITE_EXTRA_PATH_STEP_FILES", True))
    overwrite_path_csv_xyz_with_steps = bool(get_param("OVERWRITE_PATH_CSV_XYZ_WITH_STEPS", True))

    # ============================================================
    # Plot settings
    # ============================================================
    plot_max_model_points = int(get_param("PLOT_MAX_MODEL_POINTS", 300000))
    plot_dpi = int(get_param("PLOT_DPI", 300))
    plot_model_alpha = float(get_param("PLOT_MODEL_ALPHA", 0.45))
    plot_model_marker_size = float(get_param("PLOT_MODEL_MARKER_SIZE", 2.0))
    plot_path_line_width = float(get_param("PLOT_PATH_LINE_WIDTH", 2.0))

    plot_initiate_figure = bool(get_param("PLOT_INITIATE_FIGURE", True))
    initiate_figure_name = str(get_param("INITIATE_FIGURE_NAME", "00_initiate.png"))
    initiate_figure_file = algorithm_figure_dir / initiate_figure_name

    plot_model_as_flyable_nofly = bool(
        get_param("PLOT_MODEL_AS_FLYABLE_NOFLY", True)
    )
    plot_no_fly_prefixes = tuple(get_param("PLOT_NO_FLY_PREFIXES", ()))
    plot_no_fly_slowness_threshold = float(
        get_param("PLOT_NO_FLY_SLOWNESS_THRESHOLD", no_fly_slowness_threshold)
    )
    cap_slowness_after_load = bool(get_param("CAP_SLOWNESS_AFTER_LOAD", False))
    slowness_cap_value = float(
        get_param("SLOWNESS_CAP_VALUE", plot_no_fly_slowness_threshold)
    )
    plot_show_flz_overlay = bool(get_param("PLOT_SHOW_FLZ_OVERLAY", True))
    plot_always_flyable_prefixes = tuple(
        get_param("PLOT_ALWAYS_FLYABLE_PREFIXES", ())
    )
    plot_report_text_box = bool(get_param("PLOT_REPORT_TEXT_BOX", True))

    # ------------------------------------------------------------
    # Extra diagnostic plotting
    # ------------------------------------------------------------
    plot_input_slowness_side_by_side = bool(
        get_param("PLOT_INPUT_SLOWNESS_SIDE_BY_SIDE", True)
    )
    input_slowness_side_by_side_name = str(
        get_param(
            "INPUT_SLOWNESS_SIDE_BY_SIDE_NAME",
            f"00_input_vs_slowness_from_{safe_start_label}_to_{safe_end_label}.png",
        )
    )
    input_slowness_side_by_side_file = (
        algorithm_figure_dir / input_slowness_side_by_side_name
    )

    plot_slowness_discrete_bounds = get_param(
        "PLOT_SLOWNESS_DISCRETE_BOUNDS",
        [0.0, 0.02, 0.05, 0.085, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0],
    )

    plot_path_zoom_diagnostic_flag = bool(
        get_param("PLOT_PATH_ZOOM_DIAGNOSTIC", True)
    )
    path_zoom_buffer_m = float(get_param("PATH_ZOOM_BUFFER_M", 250.0))
    path_zoom_max_model_points = int(
        get_param("PATH_ZOOM_MAX_MODEL_POINTS", plot_max_model_points)
    )
    path_zoom_show_neighbor_edges = bool(
        get_param("PATH_ZOOM_SHOW_NEIGHBOR_EDGES", True)
    )
    path_zoom_show_adjacent_nodes = bool(
        get_param("PATH_ZOOM_SHOW_ADJACENT_NODES", True)
    )
    path_zoom_coordinate_mode = str(
        get_param("PATH_ZOOM_COORDINATE_MODE", "relative_m")
    )
    path_zoom_label_steps = bool(
        get_param("PATH_ZOOM_LABEL_STEPS", True)
    )
    path_zoom_label_step_every = int(
        get_param("PATH_ZOOM_LABEL_STEP_EVERY", 1)
    )
    path_zoom_arrow_every = int(
        get_param("PATH_ZOOM_ARROW_EVERY", 1)
    )

    # ============================================================
    # Run mode settings
    # ============================================================
    # RUN_MODE = "full"      : build graph, run algorithm, export, plot
    # RUN_MODE = "plot_only" : skip algorithm and replot existing CSV/XYZ paths
    run_mode = str(get_param("RUN_MODE", "full")).strip().lower()
    if run_mode in ("plot", "plot-only", "plotonly", "only_plot"):
        run_mode = "plot_only"

    plot_multiple_ranked_paths = bool(get_param("PLOT_MULTIPLE_RANKED_PATHS", True))
    plot_multiple_ranks = parse_rank_selection(get_param("PLOT_MULTIPLE_RANKS", "all"))
    plot_multiple_max_rank = get_param("PLOT_MULTIPLE_MAX_RANK", None)
    if plot_multiple_max_rank is not None:
        plot_multiple_max_rank = int(plot_multiple_max_rank)
    # ============================================================
    # Cleanup settings
    # ============================================================
    run_cleanup = bool(get_param("RUN_CLEANUP", False))
    cleanup_dry_run = bool(get_param("CLEANUP_DRY_RUN", True))
    cleanup_empty_dirs = bool(get_param("CLEANUP_EMPTY_DIRS", True))
    cleanup_target_dirs = get_param(
        "CLEANUP_TARGET_DIRS",
        [dat_root_dir, figure_root_dir],
    )
    cleanup_patterns = get_param(
        "CLEANUP_PATTERNS",
        [
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.backup",
            "*.log",
            "*.cache",
            "*.gmt",
            "*.cpt",
            "*.grd",
            "*.nc",
            "*.vrt",
            "*.aux.xml",
            "*_tmp.*",
            "*_temp.*",
            "tmp_*",
            "temp_*",
            ".gmt*",
        ],
    )

    # ============================================================
    # Make folders
    # ============================================================
    dat_root_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_root_dir.mkdir(parents=True, exist_ok=True)
    algorithm_figure_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Header
    # ============================================================
    print("=" * 70)
    print("PATH-FINDING MAIN CONTROLLER")
    print("=" * 70)
    print(f"Model file       : {model_file}")
    print(f"Algorithm module : {algorithm_name}")
    print(f"Algorithm output : {algorithm_base_name}")
    if is_multiple_algorithm:
        print(f"Multiple run n   : {multiple_run_value}")
    print(f"Data output dir  : {output_dir}")
    print(f"Figure dir       : {algorithm_figure_dir}")
    print(f"Report figure    : {report_figure_file}")
    print(f"Initiate figure  : {initiate_figure_file}")
    print(f"Side-by-side fig : {input_slowness_side_by_side_file}")
    print(f"Zoom diagnostic  : {plot_path_zoom_diagnostic_flag}")
    print(f"Run cleanup      : {run_cleanup}")
    print(f"Run mode         : {run_mode}")
    print(f"Flyable rule     : slowness < {no_fly_slowness_threshold:g}")
    print(f"No-fly rule      : slowness >= {no_fly_slowness_threshold:g}")
    print("=" * 70)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # ============================================================
    # 1. Load model
    # ============================================================
    print("[1/6] Loading labelled model...")
    model = load_labelled_model(model_file)

    print(f"      Nodes loaded: {len(model):,}")
    print(f"      Columns     : {list(model.columns)}")

    if cap_slowness_after_load:
        model, cap_summary = cap_slowness_values(
            model=model,
            cap_value=slowness_cap_value,
            inplace=False,
        )

        print("      Slowness cap:")
        print(f"        cap value       : {cap_summary['cap_value']:.6g}")
        print(f"        capped nodes    : {cap_summary['n_capped']:,}")
        print(f"        old max slowness: {cap_summary['old_max_slowness']:.6g}")
        print(f"        new max slowness: {cap_summary['new_max_slowness']:.6g}")

    # ============================================================
    # COSTMAP. Predefine risk map + emergency map before planning
    # ============================================================
    use_predefined_costmap = bool(get_param("USE_PREDEFINED_COSTMAP", False))
    use_effective_slowness_for_planning = bool(
        get_param("USE_EFFECTIVE_SLOWNESS_FOR_PLANNING", True)
    )
    save_costmap_outputs_flag = bool(get_param("SAVE_COSTMAP_OUTPUTS", True))
    plot_costmap_outputs_flag = bool(get_param("PLOT_COSTMAP_OUTPUTS", True))

    if use_predefined_costmap:
        print("[COSTMAP] Building predefined risk/emergency/final cost maps...")

        model = build_predefined_costmap(
            model=model,
            use_risk_map=bool(get_param("USE_RISK_MAP", True)),
            use_emergency_map=bool(get_param("USE_EMERGENCY_MAP", True)),
            base_risk=float(get_param("BASE_RISK", 0.05)),
            prefix_risk=get_param("PREFIX_RISK", None),
            risk_columns=get_param("RISK_COLUMNS", None),
            emergency_prefixes=tuple(get_param("EMERGENCY_PREFIXES", ("DB", "DK", "FLZ"))),
            emergency_distance_decay_m=float(get_param("EMERGENCY_DISTANCE_DECAY_M", 1000.0)),
            emergency_score_columns=get_param("EMERGENCY_SCORE_COLUMNS", None),
            restricted_prefixes=tuple(get_param("RESTRICTED_PREFIXES_FOR_EMERGENCY", ("RA",))),
            no_fly_slowness_threshold=float(get_param("PLOT_NO_FLY_SLOWNESS_THRESHOLD", 10.0)),
            no_fly_risk=float(get_param("NO_FLY_RISK", 1.0)),
            travel_weight=float(get_param("TRAVEL_WEIGHT", 1.0)),
            risk_weight=float(get_param("RISK_WEIGHT", 3.0)),
            emergency_weight=float(get_param("EMERGENCY_WEIGHT", 1.0)),
            min_effective_slowness=float(get_param("MIN_EFFECTIVE_SLOWNESS", 1e-9)),
            max_effective_slowness=get_param("MAX_EFFECTIVE_SLOWNESS", None),
            overwrite_slowness=use_effective_slowness_for_planning,
        )

        print("      Costmap columns added:")
        for col in [
            "base_slowness",
            "risk_map",
            "emergency_score",
            "emergency_distance_m",
            "emergency_penalty",
            "cost_multiplier",
            "effective_slowness",
        ]:
            if col in model.columns:
                print(
                    f"        {col:22s}: "
                    f"min={model[col].min():.6g}, "
                    f"max={model[col].max():.6g}"
                )

        if use_effective_slowness_for_planning:
            print("      Planning slowness: effective_slowness")
        else:
            print("      Planning slowness: original slowness")

        if save_costmap_outputs_flag:
            costmap_output_name = str(get_param("COSTMAP_OUTPUT_NAME", "costmap_senario1"))
            costmap_files = save_costmap_outputs(
                model=model,
                output_dir=output_dir,
                name=costmap_output_name,
            )
            print("      Costmap saved:")
            for key, value in costmap_files.items():
                print(f"        {key:14s}: {value}")

        if plot_costmap_outputs_flag:
            costmap_figure_dir = algorithm_figure_dir / "costmap"

            costmap_surface_spacing_m = float(
                get_param("COSTMAP_SURFACE_SPACING_M", 20.0)
            )

            costmap_figures = plot_costmap_surface_outputs(
                model=model,
                figure_dir=costmap_figure_dir,
                spacing_m=costmap_surface_spacing_m,
                dpi=plot_dpi,
                no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            )

            print("      Costmap surface figures:")
            for key, value in costmap_figures.items():
                print(f"        {key:22s}: {value}")

    # ============================================================
    # Apply numeric flyability rule from the loaded slowness/cost model
    # ============================================================
    model = add_flyability_columns_from_slowness(
        model=model,
        block_by_slowness_threshold=block_by_slowness_threshold,
        threshold=no_fly_slowness_threshold,
        mode=no_fly_threshold_mode,
        tolerance=no_fly_slowness_tolerance,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    print_flyability_summary(
        model=model,
        block_by_slowness_threshold=block_by_slowness_threshold,
        threshold=no_fly_slowness_threshold,
        mode=no_fly_threshold_mode,
    )

    # ============================================================
    # 2. Select real start/end
    # ============================================================
    print("[2/6] Selecting start/end nodes...")

    if fmm2d_all_facility_mode:
        db_prefixes = tuple(get_param("FMM2D_PAIR_DB_PREFIXES", ("DB",)))
        dk_prefixes = tuple(get_param("FMM2D_PAIR_DK_PREFIXES", ("DK",)))

        # Temporary dummy nodes only. FMM2D will compute all real DB/DK pairs.
        start_idx = _first_index_with_label_prefix(model, db_prefixes)
        try:
            end_idx = _first_index_with_label_prefix(model, dk_prefixes)
        except Exception:
            # If the file has no DK, still keep main.py alive for DB-DB mode.
            end_idx = start_idx

        print("      FMM2D all-facility-pair mode detected.")
        print("      START_LABEL/END_LABEL are None; using temporary dummy nodes only for main.py bookkeeping.")
        print(f"      Dummy start index: {start_idx} | {model.loc[start_idx, 'label']}")
        print(f"      Dummy end index  : {end_idx} | {model.loc[end_idx, 'label']}")
    else:
        start_idx, end_idx = find_start_end_indices(
            model,
            start_label=start_label,
            end_label=end_label,
            start_coord=start_coord,
            end_coord=end_coord,
        )

    print(f"      Start index: {start_idx}")
    print(f"      End index  : {end_idx}")
    print(f"      Start node : {model.loc[start_idx].to_dict()}")
    print(f"      End node   : {model.loc[end_idx].to_dict()}")

    # ============================================================
    # INIT. Plot initiate figure
    # ============================================================
    if plot_initiate_figure:
        print("[INIT] Plotting initiate model figure...")
        plot_initiate_model(
            model=model,
            start_idx=start_idx,
            end_idx=end_idx,
            figure_file=initiate_figure_file,
            max_model_points=plot_max_model_points,
            dpi=plot_dpi,
            model_alpha=plot_model_alpha,
            model_marker_size=plot_model_marker_size,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=plot_always_flyable_prefixes,
        )
        print(f"      Initiate figure: {initiate_figure_file}")

    # ============================================================
    # SIDE-BY-SIDE. Plot input model and slowness model
    # ============================================================
    if plot_input_slowness_side_by_side:
        print("[DIAG] Plotting input model side by side with slowness model...")
        try:
            plot_model_slowness_side_by_side(
                model=model,
                figure_file=input_slowness_side_by_side_file,
                start_idx=start_idx,
                end_idx=end_idx,
                max_model_points=plot_max_model_points,
                dpi=plot_dpi,
                model_alpha=plot_model_alpha,
                model_marker_size=plot_model_marker_size,
                no_fly_prefixes=plot_no_fly_prefixes,
                no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
                always_flyable_prefixes=plot_always_flyable_prefixes,
                show_flz_overlay=plot_show_flz_overlay,
                slowness_discrete_bounds=plot_slowness_discrete_bounds,
                cleanup_temp=bool(get_param("RUN_CLEANUP", False)),
            )
            print(f"      Side-by-side figure: {input_slowness_side_by_side_file}")
        except Exception as exc:
            print(f"[WARNING] Could not plot side-by-side diagnostic: {exc}")

    # ============================================================
    # 3. Build graph
    # ============================================================
    print("[3/6] Building graph...")

    # build_grid_graph() may only support label-prefix blocking.
    # For the new numeric model, create a temporary graph_model where
    # slowness >= 10 nodes receive an internal no-fly label prefix.
    graph_model, effective_block_prefixes = make_graph_model_with_slowness_blocking(
        model=model,
        block_by_slowness_threshold=block_by_slowness_threshold,
        threshold=no_fly_slowness_threshold,
        mode=no_fly_threshold_mode,
        tolerance=no_fly_slowness_tolerance,
        block_prefixes=block_prefixes,
        internal_no_fly_prefix=internal_no_fly_label_prefix,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    print("      Blocking rule:")
    print(f"        by slowness threshold : {block_by_slowness_threshold}")
    print(f"        no-fly threshold      : {no_fly_slowness_threshold:g}")
    print(f"        threshold mode        : {no_fly_threshold_mode}")
    print(f"        label block prefixes  : {effective_block_prefixes}")

    graph = build_grid_graph(
        graph_model,
        block_label_prefixes=effective_block_prefixes,
        high_cost_label_prefixes=high_cost_prefixes,
        high_cost_factor=flz_cost_factor,
        connectivity_2d=connectivity_2d,
        connectivity_3d=connectivity_3d,
        always_flyable_prefixes=always_flyable_prefixes,
        graph_neighbor_mode=graph_neighbor_mode,
        kdtree_radius_factor=kdtree_radius_factor,
        kdtree_max_neighbors_2d=kdtree_max_neighbors_2d,
        kdtree_max_neighbors_3d=kdtree_max_neighbors_3d,
    )

    print(f"      Traversable nodes: {len(graph['valid_indices']):,}")
    if "is_flyable" in model.columns:
        expected_flyable = int(model["is_flyable"].sum())
        actual_valid = int(len(graph["valid_indices"]))
        print(f"      Expected flyable : {expected_flyable:,}")
        if block_by_slowness_threshold and actual_valid != expected_flyable:
            print("[WARNING] Graph valid count differs from flyable count.")
            print("          Check build_grid_graph() blocking against label/label_prefix.")
    print(f"      Grid dimension   : {graph['dimension']}D")
    print(f"      Connectivity     : {graph['connectivity']}")

    if "graph_neighbor_mode" in graph:
        print(f"      Neighbor mode    : {graph['graph_neighbor_mode']}")
    if "grid_spacing_m" in graph:
        print(f"      Grid spacing     : {graph['grid_spacing_m']:.2f} m")
    if "neighbor_radius_m" in graph:
        print(f"      Neighbor radius  : {graph['neighbor_radius_m']:.2f} m")
    if "max_neighbors" in graph:
        print(f"      Max neighbors    : {graph['max_neighbors']}")

    # ============================================================
    # Snap DB/DK or special points to grid/search nodes
    # ============================================================
    search_start_idx, search_end_idx = snap_start_end_to_grid_if_needed(
        model=model,
        graph=graph,
        start_idx=start_idx,
        end_idx=end_idx,
        snap=snap_start_end_to_grid,
        target_prefixes=snap_target_prefixes,
    )

    search_start_idx, search_end_idx = ensure_endpoint_indices_are_traversable(
        model=model,
        graph=graph,
        start_idx=start_idx,
        end_idx=end_idx,
        search_start_idx=search_start_idx,
        search_end_idx=search_end_idx,
        snap=snap_start_end_to_grid,
        snap_only_to_flyable=snap_only_to_flyable,
        target_prefixes=snap_target_prefixes,
    )

    # ============================================================
    # Force search endpoints to be flyable
    # ============================================================
    if force_search_start_end_flyable:
        if block_by_slowness_threshold:
            print("      FORCE_SEARCH_START_END_FLYABLE=True:")
            print("        DB/DK/FLZ or selected endpoints are operationally allowed.")
            print("        Normal grid cells still obey slowness >= 10 as no-fly.")
        graph["valid_indices"].add(int(search_start_idx))
        graph["valid_indices"].add(int(search_end_idx))

        print("      Force search start/end as flyable:")
        print(
            f"        search start: {search_start_idx} | "
            f"{model.loc[search_start_idx, 'label']}"
        )
        print(
            f"        search end  : {search_end_idx} | "
            f"{model.loc[search_end_idx, 'label']}"
        )

    # ============================================================
    # Endpoint flyable buffer
    # ============================================================
    if endpoint_flyable_buffer_radius_m > 0:
        buffer_endpoint_indices = []

        if endpoint_flyable_buffer_mode in ("real", "both"):
            buffer_endpoint_indices.extend([start_idx, end_idx])

        if endpoint_flyable_buffer_mode in ("search", "both"):
            buffer_endpoint_indices.extend([search_start_idx, search_end_idx])

        # Remove duplicates but keep order
        buffer_endpoint_indices = list(
            dict.fromkeys([int(i) for i in buffer_endpoint_indices])
        )

        before_n_valid = len(graph["valid_indices"])

        graph = add_endpoint_flyable_buffer(
            model=model,
            graph=graph,
            endpoint_indices=buffer_endpoint_indices,
            radius_m=endpoint_flyable_buffer_radius_m,
        )

        after_n_valid = len(graph["valid_indices"])
        added_n = after_n_valid - before_n_valid

        print("      Endpoint flyable buffer:")
        print(f"        radius           : {endpoint_flyable_buffer_radius_m:.2f} m")
        print(f"        mode             : {endpoint_flyable_buffer_mode}")
        print(f"        endpoint indices : {buffer_endpoint_indices}")
        print(f"        added flyable    : {added_n:,} nodes")

    # ============================================================
    # Print snapping information
    # ============================================================
    if search_start_idx != start_idx:
        print("      Start snapped:")
        print(
            f"        real start index  : {start_idx} | "
            f"{model.loc[start_idx, 'label']}"
        )
        print(
            f"        search start index: {search_start_idx} | "
            f"{model.loc[search_start_idx, 'label']}"
        )

    if search_end_idx != end_idx:
        print("      End snapped:")
        print(
            f"        real end index  : {end_idx} | "
            f"{model.loc[end_idx, 'label']}"
        )
        print(
            f"        search end index: {search_end_idx} | "
            f"{model.loc[search_end_idx, 'label']}"
        )

    # ============================================================
    # Neighbor diagnostic
    # ============================================================
    print("      Neighbor diagnostic:")
    start_n_neighbors = count_valid_neighbors(model, graph, search_start_idx)
    end_n_neighbors = count_valid_neighbors(model, graph, search_end_idx)

    print(f"        search start neighbors: {start_n_neighbors}")
    print(f"        search end neighbors  : {end_n_neighbors}")

    if start_n_neighbors == 0 or end_n_neighbors == 0:
        print("[WARNING] Search start or end has zero graph neighbors.")
        print("          Try increasing KDTREE_RADIUS_FACTOR in parameters.py, for example:")
        print("          KDTREE_RADIUS_FACTOR = 2.0")

    print(f"      Traversable nodes: {len(graph['valid_indices']):,}")
    print(f"      Grid dimension   : {graph['dimension']}D")
    print(f"      Connectivity     : {graph['connectivity']}")

    # ============================================================
    # Plot-only mode: use saved path files and skip algorithm rerun
    # ============================================================
    if run_mode == "plot_only":
        print("[PLOT ONLY] Skip algorithm and replot existing path files...")
        plotted_files = plot_saved_path_reports(
            model=model,
            output_dir=output_dir,
            algorithm_figure_dir=algorithm_figure_dir,
            path_name=path_name,
            algorithm_name=algorithm_name,
            is_multiple_algorithm=(is_multiple_algorithm or fmm2d_all_facility_mode),
            safe_start_label=safe_start_label,
            safe_end_label=safe_end_label,
            max_model_points=plot_max_model_points,
            dpi=plot_dpi,
            model_alpha=plot_model_alpha,
            model_marker_size=plot_model_marker_size,
            path_line_width=plot_path_line_width,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=plot_always_flyable_prefixes,
            result=None if not plot_report_text_box else {
                "real_start_idx": int(start_idx),
                "real_end_idx": int(end_idx),
                "search_start_idx": int(search_start_idx),
                "search_end_idx": int(search_end_idx),
            },
            rank_selection=plot_multiple_ranks,
        )
        run_total_elapsed_s = time.perf_counter() - run_total_start_time
        timing_summary = {
            "algorithm": str(algorithm_name),
            "run_mode": "plot_only",
            "processing_time_total_s": float(run_total_elapsed_s),
            "processing_time_total_text": format_elapsed_time(run_total_elapsed_s),
            "plotted_files_count": int(len(plotted_files)),
        }
        if save_processing_time_json:
            timing_file = output_dir / f"{path_name}_{algorithm_name}_processing_time.json"
            maybe_write_processing_time_json(timing_file, timing_summary)

        print("=" * 70)
        print("PLOT ONLY DONE")
        print("=" * 70)
        for f in plotted_files:
            print(f"  figure: {f}")
        if print_processing_time:
            print("Processing time:")
            print(f"  total: {format_elapsed_time(run_total_elapsed_s)}")
            if save_processing_time_json:
                print(f"  timing file: {timing_file}")
        return timing_summary

    # ============================================================
    # 4. Run algorithm
    # ============================================================
    print("[4/6] Running algorithm...")
    try:
        alg_module = importlib.import_module(f"src.{algorithm_name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Algorithm src/{algorithm_name}.py not found. "
            f"Available examples: astar, flood_fill"
        ) from exc

    if not hasattr(alg_module, "run"):
        raise AttributeError(
            f"src/{algorithm_name}.py must contain a run(...) function."
        )

    algorithm_kwargs = {
        "k_paths": int(get_param("MULTI_PATH_K_PATHS", 1)),
        "turn_weight": float(get_param("MULTI_PATH_TURN_WEIGHT", 0.0)),
        "turn_angle_threshold_degree": float(
            get_param("MULTI_PATH_TURN_ANGLE_THRESHOLD_DEGREE", 1.0)
        ),
        "max_expansions": int(get_param("MULTI_PATH_MAX_EXPANSIONS", 5_000_000)),
        "max_states_per_node_direction": int(
            get_param("MULTI_PATH_MAX_STATES_PER_NODE_DIRECTION", 150)
        ),
        "heuristic_weight": float(get_param("MULTI_PATH_HEURISTIC_WEIGHT", 1.0)),
        "use_turn_penalty": bool(get_param("MULTI_PATH_USE_TURN_PENALTY", True)),
        "save_all_k_paths": bool(get_param("MULTI_PATH_SAVE_ALL_K_PATHS", True)),
        "verbose": bool(get_param("MULTI_PATH_VERBOSE", True)),

        # Multiple-path overlap control.
        # MULTI_PATH_OVERLAP_MODE = "allow"       -> old behavior; paths may share nodes/edges.
        # MULTI_PATH_OVERLAP_MODE = "non_overlap" -> later paths cannot reuse previous path
        #                                             nodes/edges except inside start/end/DB/DK/FLZ buffers.
        "path_overlap_mode": str(get_param("MULTI_PATH_OVERLAP_MODE", "allow")),
        "non_overlap_buffer_radius_m": float(
            get_param("MULTI_PATH_NON_OVERLAP_BUFFER_RADIUS_M", 150.0)
        ),
        "non_overlap_allowed_prefixes": tuple(
            get_param("MULTI_PATH_NON_OVERLAP_ALLOWED_PREFIXES", ("DB", "DK", "FLZ"))
        ),
        "non_overlap_block_edges": bool(
            get_param("MULTI_PATH_NON_OVERLAP_BLOCK_EDGES", True)
        ),

        "parallel": bool(get_param("MULTI_PATH_PARALLEL", True)),
        "n_cores": get_param("MULTI_PATH_N_CORES", None),
        "parallel_mode": str(get_param("MULTI_PATH_PARALLEL_MODE", "sequential")),
        "candidates_per_round": int(
            get_param("MULTI_PATH_CANDIDATES_PER_ROUND", 8)
        ),
        "max_rounds_per_path": int(
            get_param("MULTI_PATH_MAX_ROUNDS_PER_PATH", 3)
        ),
        "candidate_diversity_weight": float(
            get_param("MULTI_PATH_CANDIDATE_DIVERSITY_WEIGHT", 0.25)
        ),
        "candidate_seed": int(
            get_param("MULTI_PATH_CANDIDATE_SEED", 20260618)
        ),
        "max_expansions_per_candidate": get_param(
            "MULTI_PATH_MAX_EXPANSIONS_PER_CANDIDATE", None
        ),
    }

    # Only pass parameters accepted by the selected algorithm.
    # This keeps other algorithms such as flood_fill compatible.
    run_signature = inspect.signature(alg_module.run)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in run_signature.parameters.values()):
        accepted_algorithm_kwargs = algorithm_kwargs
    else:
        accepted_algorithm_kwargs = {
            key: value
            for key, value in algorithm_kwargs.items()
            if key in run_signature.parameters
        }

    algorithm_search_start_time = time.perf_counter()
    result = alg_module.run(
        model=model,
        graph=graph,
        start_idx=search_start_idx,
        end_idx=search_end_idx,
        **accepted_algorithm_kwargs,
    )
    algorithm_search_elapsed_s = time.perf_counter() - algorithm_search_start_time

    result["algorithm_search_time_s"] = float(algorithm_search_elapsed_s)
    result["algorithm_search_time_text"] = format_elapsed_time(algorithm_search_elapsed_s)

    if print_processing_time:
        print("      Algorithm search time:")
        print(f"        {format_elapsed_time(algorithm_search_elapsed_s)}")

    algorithm_path_indices = result.get("path_indices", [])

    if not algorithm_path_indices:
        print("[FAILED] No path found.")

        result["real_start_idx"] = int(start_idx)
        result["real_end_idx"] = int(end_idx)
        result["search_start_idx"] = int(search_start_idx)
        result["search_end_idx"] = int(search_end_idx)
        result["include_real_start_end_in_output"] = bool(include_real_start_end)
        result["endpoint_flyable_buffer_radius_m"] = float(
            endpoint_flyable_buffer_radius_m
        )
        result["endpoint_flyable_buffer_mode"] = str(endpoint_flyable_buffer_mode)

        # Add slowness-cap metadata
        result["cap_slowness_after_load"] = bool(cap_slowness_after_load)
        result["slowness_cap_value"] = float(slowness_cap_value)
        result["block_by_slowness_threshold"] = bool(block_by_slowness_threshold)
        result["no_fly_slowness_threshold"] = float(no_fly_slowness_threshold)
        result["no_fly_threshold_mode"] = str(no_fly_threshold_mode)

        result["start_neighbors"] = int(start_n_neighbors)
        result["end_neighbors"] = int(end_n_neighbors)

        run_total_elapsed_s = time.perf_counter() - run_total_start_time
        result["processing_time_total_s"] = float(run_total_elapsed_s)
        result["processing_time_total_text"] = format_elapsed_time(run_total_elapsed_s)
        result["algorithm_search_time_s"] = float(algorithm_search_elapsed_s)
        result["algorithm_search_time_text"] = format_elapsed_time(algorithm_search_elapsed_s)

        if print_processing_time:
            print("      Processing time before failure:")
            print(f"        total            : {format_elapsed_time(run_total_elapsed_s)}")
            print(f"        algorithm search : {format_elapsed_time(algorithm_search_elapsed_s)}")

        fail_file = output_dir / f"{path_name}_{algorithm_name}_FAILED.json"
        fail_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"Failure summary saved to: {fail_file}")
        sys.exit(1)

    if block_by_slowness_threshold:
        validate_path_uses_only_flyable_nodes(
            model=model,
            path_indices=algorithm_path_indices,
            threshold=no_fly_slowness_threshold,
            mode=no_fly_threshold_mode,
            tolerance=no_fly_slowness_tolerance,
            context="algorithm path",
            always_flyable_prefixes=always_flyable_prefixes,
        )

    # Add real DB/DK to output footprint if needed
    path_indices = add_real_start_end_to_path(
        path_indices=algorithm_path_indices,
        real_start_idx=start_idx,
        real_end_idx=end_idx,
        include=include_real_start_end,
    )
    # Compute metrics for algorithm path only.
    # This avoids counting artificial DB/DK connector segments if DB/DK are off-grid.
    algorithm_path_metrics = compute_path_metrics(
        model=model,
        graph=graph,
        path_indices=algorithm_path_indices,
    )

    # Compute metrics for exported path including DB/DK if included.
    # This gives the full footprint distance from real DB to real DK.
    output_path_metrics = compute_path_metrics(
        model=model,
        graph=graph,
        path_indices=path_indices,
    )

    # Add metadata
    result["real_start_idx"] = int(start_idx)
    result["real_end_idx"] = int(end_idx)
    result["search_start_idx"] = int(search_start_idx)
    result["search_end_idx"] = int(search_end_idx)
    result["include_real_start_end_in_output"] = bool(include_real_start_end)
    result["endpoint_flyable_buffer_radius_m"] = float(
        endpoint_flyable_buffer_radius_m
    )
    result["endpoint_flyable_buffer_mode"] = str(endpoint_flyable_buffer_mode)

    # Add slowness-cap metadata
    result["cap_slowness_after_load"] = bool(cap_slowness_after_load)
    result["slowness_cap_value"] = float(slowness_cap_value)
    result["block_by_slowness_threshold"] = bool(block_by_slowness_threshold)
    result["no_fly_slowness_threshold"] = float(no_fly_slowness_threshold)
    result["no_fly_threshold_mode"] = str(no_fly_threshold_mode)
    result["snap_only_to_flyable"] = bool(snap_only_to_flyable)

    result["start_neighbors"] = int(start_n_neighbors)
    result["end_neighbors"] = int(end_n_neighbors)

    result["graph_neighbor_mode"] = str(graph.get("graph_neighbor_mode", "unknown"))
    result["grid_spacing_m"] = float(graph.get("grid_spacing_m", 0.0))
    result["neighbor_radius_m"] = float(graph.get("neighbor_radius_m", 0.0))
    result["max_neighbors"] = int(graph.get("max_neighbors", 0))

    result["algorithm_path_distance_m"] = algorithm_path_metrics["distance_traveled_m"]
    result["algorithm_path_distance_km"] = algorithm_path_metrics["distance_traveled_km"]
    result["algorithm_estimated_traveltime_s"] = algorithm_path_metrics["estimated_traveltime_s"]
    result["algorithm_estimated_traveltime_min"] = algorithm_path_metrics["estimated_traveltime_min"]

    result["output_path_distance_m"] = output_path_metrics["distance_traveled_m"]
    result["output_path_distance_km"] = output_path_metrics["distance_traveled_km"]
    result["output_estimated_traveltime_s"] = output_path_metrics["estimated_traveltime_s"]
    result["output_estimated_traveltime_min"] = output_path_metrics["estimated_traveltime_min"]

    print(f"      Algorithm path nodes       : {len(algorithm_path_indices):,}")
    print(f"      Output path nodes          : {len(path_indices):,}")
    print(f"      Total cost                 : {result.get('total_cost', None)}")

    print("      Path metrics:")
    print(f"        algorithm distance       : {algorithm_path_metrics['distance_traveled_m']:.2f} m")
    print(f"        algorithm distance       : {algorithm_path_metrics['distance_traveled_km']:.4f} km")
    print(f"        algorithm traveltime     : {algorithm_path_metrics['estimated_traveltime_s']:.2f} s")
    print(f"        algorithm traveltime     : {algorithm_path_metrics['estimated_traveltime_min']:.2f} min")

    print(f"        output distance          : {output_path_metrics['distance_traveled_m']:.2f} m")
    print(f"        output distance          : {output_path_metrics['distance_traveled_km']:.4f} km")
    print(f"        output traveltime        : {output_path_metrics['estimated_traveltime_s']:.2f} s")
    print(f"        output traveltime        : {output_path_metrics['estimated_traveltime_min']:.2f} min")

    processing_time_until_export_s = time.perf_counter() - run_total_start_time
    result["processing_time_until_export_s"] = float(processing_time_until_export_s)
    result["processing_time_until_export_text"] = format_elapsed_time(processing_time_until_export_s)

    # ============================================================
    # 5. Export path footprint files
    # ============================================================
    print("[5/6] Exporting path footprint files...")
    exported = export_path_outputs(
        model=model,
        path_indices=path_indices,
        output_dir=output_dir,
        path_name=f"{path_name}_{algorithm_name}",
        algorithm_name=algorithm_name,
        result=result,
    )

    if save_path_step_distance:
        exported = add_path_step_distance_to_exported_files(
            model=model,
            path_indices=path_indices,
            exported=exported,
            path_rank=result.get("rank", None),
            write_extra_step_files=write_extra_path_step_files,
            overwrite_csv_xyz=overwrite_path_csv_xyz_with_steps,
        )


    # Optional: export all K multiple paths when a multiple-path algorithm returns path_results.
    # The first path is still exported above using the original name for backward compatibility.
    all_path_results = result.get("path_results", [])
    save_all_k_paths = bool(get_param("MULTI_PATH_SAVE_ALL_K_PATHS", True))
    exported_k_paths = []

    if save_all_k_paths and len(all_path_results) > 1:
        print(f"      Exporting all ranked paths: {len(all_path_results)}")

        for path_item in all_path_results:
            rank = int(path_item.get("rank", len(exported_k_paths) + 1))
            ranked_algorithm_path_indices = path_item.get("path_indices", [])

            if not ranked_algorithm_path_indices:
                continue

            ranked_path_indices = add_real_start_end_to_path(
                path_indices=ranked_algorithm_path_indices,
                real_start_idx=start_idx,
                real_end_idx=end_idx,
                include=(False if fmm2d_all_facility_mode else include_real_start_end),
            )

            ranked_algorithm_metrics = compute_path_metrics(
                model=model,
                graph=graph,
                path_indices=ranked_algorithm_path_indices,
            )
            ranked_output_metrics = compute_path_metrics(
                model=model,
                graph=graph,
                path_indices=ranked_path_indices,
            )

            ranked_result = dict(result)
            ranked_result.pop("path_results", None)
            ranked_result["path_indices"] = [int(i) for i in ranked_algorithm_path_indices]
            ranked_result["rank"] = rank
            ranked_result["total_cost"] = float(path_item.get("total_cost", path_item.get("cost", 0.0)))
            ranked_result["travel_cost"] = float(path_item.get("travel_cost", 0.0))
            ranked_result["turn_cost"] = float(path_item.get("turn_cost", 0.0))
            ranked_result["turn_count"] = int(path_item.get("turn_count", 0))
            ranked_result["total_turn_angle_degree"] = float(path_item.get("total_turn_angle_degree", 0.0))
            for meta_key in (
                "source_idx", "target_idx", "source_label", "target_label",
                "pair_type", "pair_key", "pair_undirected_key", "direct_distance_m",
            ):
                if meta_key in path_item:
                    ranked_result[meta_key] = path_item.get(meta_key)

            ranked_result["algorithm_path_distance_m"] = ranked_algorithm_metrics["distance_traveled_m"]
            ranked_result["algorithm_path_distance_km"] = ranked_algorithm_metrics["distance_traveled_km"]
            ranked_result["algorithm_estimated_traveltime_s"] = ranked_algorithm_metrics["estimated_traveltime_s"]
            ranked_result["algorithm_estimated_traveltime_min"] = ranked_algorithm_metrics["estimated_traveltime_min"]
            ranked_result["output_path_distance_m"] = ranked_output_metrics["distance_traveled_m"]
            ranked_result["output_path_distance_km"] = ranked_output_metrics["distance_traveled_km"]
            ranked_result["output_estimated_traveltime_s"] = ranked_output_metrics["estimated_traveltime_s"]
            ranked_result["output_estimated_traveltime_min"] = ranked_output_metrics["estimated_traveltime_min"]

            ranked_export = export_path_outputs(
                model=model,
                path_indices=ranked_path_indices,
                output_dir=output_dir,
                path_name=f"{path_name}_{algorithm_name}_rank_{rank:03d}",
                algorithm_name=algorithm_name,
                result=ranked_result,
            )

            if save_path_step_distance:
                ranked_export = add_path_step_distance_to_exported_files(
                    model=model,
                    path_indices=ranked_path_indices,
                    exported=ranked_export,
                    path_rank=rank,
                    write_extra_step_files=write_extra_path_step_files,
                    overwrite_csv_xyz=overwrite_path_csv_xyz_with_steps,
                )

            exported_k_paths.append(ranked_export)

        k_summary_file = output_dir / f"{path_name}_{algorithm_name}_top_{len(exported_k_paths):03d}_summary.csv"
        try:
            import pandas as pd

            summary_rows = []
            for path_item in all_path_results:
                summary_path_indices = path_item.get("path_indices", [])
                if summary_path_indices:
                    summary_output_indices = add_real_start_end_to_path(
                        path_indices=summary_path_indices,
                        real_start_idx=start_idx,
                        real_end_idx=end_idx,
                        include=(False if fmm2d_all_facility_mode else include_real_start_end),
                    )
                    summary_step_df = build_path_step_distance_table(
                        model=model,
                        path_indices=summary_output_indices,
                        path_rank=int(path_item.get("rank", 0)),
                    )
                    total_distance_m = float(summary_step_df["distance_from_start_m"].iloc[-1])
                    total_distance_km = float(summary_step_df["distance_from_start_km"].iloc[-1])
                else:
                    total_distance_m = 0.0
                    total_distance_km = 0.0

                summary_rows.append({
                    "rank": int(path_item.get("rank", 0)),
                    "source_label": path_item.get("source_label", ""),
                    "target_label": path_item.get("target_label", ""),
                    "pair_key": path_item.get("pair_key", ""),
                    "pair_undirected_key": path_item.get("pair_undirected_key", ""),
                    "pair_type": path_item.get("pair_type", ""),
                    "source_idx": path_item.get("source_idx", ""),
                    "target_idx": path_item.get("target_idx", ""),
                    "direct_distance_m": path_item.get("direct_distance_m", ""),
                    "total_cost": float(path_item.get("total_cost", path_item.get("cost", 0.0))),
                    "travel_cost": float(path_item.get("travel_cost", 0.0)),
                    "turn_cost": float(path_item.get("turn_cost", 0.0)),
                    "nodes": int(path_item.get("nodes", len(path_item.get("path_indices", [])))),
                    "distance_from_start_m_final": total_distance_m,
                    "distance_from_start_km_final": total_distance_km,
                    "turn_count": int(path_item.get("turn_count", 0)),
                    "total_turn_angle_degree": float(path_item.get("total_turn_angle_degree", 0.0)),
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(k_summary_file, index=False)
            exported["top_k_summary"] = str(k_summary_file)
            exported["top_k_count"] = len(exported_k_paths)
            print(f"      Top-K summary: {k_summary_file}")

            # A clearer all-path report name for facility-library mode and
            # multiple-path runs. This contains one row per returned path.
            all_paths_report_file = output_dir / f"{path_name}_{algorithm_name}_all_paths_report.csv"
            summary_df.to_csv(all_paths_report_file, index=False)
            exported["all_paths_report_csv"] = str(all_paths_report_file)
            print(f"      All-path report CSV: {all_paths_report_file}")

            # One combined file with every node step for every returned path.
            # This is useful for checking/plotting all fastest facility paths
            # without opening hundreds of rank_XXX files.
            all_step_frames = []
            for path_item in all_path_results:
                step_path_indices = path_item.get("path_indices", [])
                if not step_path_indices:
                    continue

                step_output_indices = add_real_start_end_to_path(
                    path_indices=step_path_indices,
                    real_start_idx=start_idx,
                    real_end_idx=end_idx,
                    include=(False if fmm2d_all_facility_mode else include_real_start_end),
                )
                step_df = build_path_step_distance_table(
                    model=model,
                    path_indices=step_output_indices,
                    path_rank=int(path_item.get("rank", 0)),
                )

                # Insert path-level metadata at the front of every step row.
                meta_cols = {
                    "source_label": path_item.get("source_label", ""),
                    "target_label": path_item.get("target_label", ""),
                    "pair_key": path_item.get("pair_key", ""),
                    "pair_undirected_key": path_item.get("pair_undirected_key", ""),
                    "pair_type": path_item.get("pair_type", ""),
                    "source_idx": path_item.get("source_idx", ""),
                    "target_idx": path_item.get("target_idx", ""),
                    "total_cost": float(path_item.get("total_cost", path_item.get("cost", 0.0))),
                    "travel_cost": float(path_item.get("travel_cost", 0.0)),
                    "path_distance_m": float(path_item.get("distance_m", 0.0)),
                    "path_distance_km": float(path_item.get("distance_km", 0.0)),
                }
                for col, val in reversed(list(meta_cols.items())):
                    step_df.insert(0, col, val)
                all_step_frames.append(step_df)

            if all_step_frames:
                all_paths_steps_file = output_dir / f"{path_name}_{algorithm_name}_all_paths_steps.csv"
                pd.concat(all_step_frames, ignore_index=True).to_csv(
                    all_paths_steps_file,
                    index=False,
                    float_format="%.8f",
                )
                exported["all_paths_steps_csv"] = str(all_paths_steps_file)
                print(f"      All-path steps CSV : {all_paths_steps_file}")
        except Exception as exc:
            print(f"[WARNING] Could not write Top-K summary CSV: {exc}")

    # ============================================================
    # 6. Plot path report
    # ============================================================
    print("[6/6] Plotting path report...")

    # For FMM2D all-facility mode, the normal single-pair path report is
    # misleading because START_LABEL/END_LABEL are intentionally None and
    # main.py used only temporary dummy nodes.  Plot only the all-facility
    # combined report below.
    if not fmm2d_all_facility_mode:
        # Always write the best-path report.  For astar_multiple this keeps the
        # normal report filename while the combined ranked-path figure is written
        # as an extra diagnostic below.
        plot_path_report(
            model=model,
            path_indices=path_indices,
            figure_file=report_figure_file,
            algorithm_name=algorithm_name,
            max_model_points=plot_max_model_points,
            dpi=plot_dpi,
            model_alpha=plot_model_alpha,
            model_marker_size=plot_model_marker_size,
            path_line_width=plot_path_line_width,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=plot_always_flyable_prefixes,
            result=result if plot_report_text_box else None,
        )

    # astar_multiple returns paths in result["path_results"].  Older code
    # looked only for result["ranked_paths"], so the combined multiple-path
    # figure was skipped even though the ranked CSV files were exported.
    ranked_paths = result.get("ranked_paths", None) or result.get("path_results", None)

    if (
        plot_multiple_ranked_paths
        and ranked_paths
        and len(ranked_paths) > 1
    ):
        selected_ranked_paths = []
        wanted_ranks = None if plot_multiple_ranks == "all" else {int(r) for r in plot_multiple_ranks}

        for item in ranked_paths:
            rank = int(item.get("rank", len(selected_ranked_paths) + 1))
            if plot_multiple_max_rank is not None and rank > plot_multiple_max_rank:
                continue
            if wanted_ranks is not None and rank not in wanted_ranks:
                continue

            item_for_plot = dict(item)
            item_for_plot["path_indices"] = add_real_start_end_to_path(
                path_indices=item.get("path_indices", []),
                real_start_idx=start_idx,
                real_end_idx=end_idx,
                include=(False if fmm2d_all_facility_mode else include_real_start_end),
            )
            selected_ranked_paths.append(item_for_plot)

        if selected_ranked_paths:
            multiple_figure_file = (
                algorithm_figure_dir
                / f"path_report_{algorithm_name}_all_ranks_from_{safe_start_label}_to_{safe_end_label}.png"
            )

            if fmm2d_all_facility_mode:
                _plot_all_facility_paths_report(
                    model=model,
                    ranked_paths=selected_ranked_paths,
                    figure_file=multiple_figure_file,
                    algorithm_name=algorithm_name,
                    max_model_points=plot_max_model_points,
                    dpi=plot_dpi,
                    model_alpha=plot_model_alpha,
                    model_marker_size=plot_model_marker_size,
                    path_line_width=plot_path_line_width,
                    plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
                    plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
                )
            else:
                plot_multiple_paths_report(
                    model=model,
                    ranked_paths=selected_ranked_paths,
                    figure_file=multiple_figure_file,
                    algorithm_name=algorithm_name,
                    max_model_points=plot_max_model_points,
                    dpi=plot_dpi,
                    model_alpha=plot_model_alpha,
                    model_marker_size=plot_model_marker_size,
                    path_line_width=plot_path_line_width,
                    plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
                    plot_no_fly_prefixes=plot_no_fly_prefixes,
                    plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
                    plot_show_flz_overlay=plot_show_flz_overlay,
                    always_flyable_prefixes=plot_always_flyable_prefixes,
                    result=result if plot_report_text_box else None,
                )

            exported["multiple_path_figure"] = str(multiple_figure_file)
            print(f"      Multiple-path plot: {multiple_figure_file}")

    # ============================================================
    # ZOOM. Plot path-corridor diagnostic for adjacent-node checking
    # ============================================================
    if plot_path_zoom_diagnostic_flag:
        zoom_figure_file = (
            algorithm_figure_dir
            / f"path_zoom_{algorithm_name}_from_{safe_start_label}_to_{safe_end_label}.png"
        )
        print("[ZOOM] Plotting path-corridor diagnostic...")
        try:
            # Use the exported/output path so the plot shows the same footprint
            # that is written to CSV/XYZ. If DB/DK were off-grid and included,
            # they will also appear at the corridor ends.
            plot_path_zoom_diagnostic(
                model=model,
                graph=graph,
                path_indices=path_indices,
                figure_file=zoom_figure_file,
                algorithm_name=algorithm_name,
                buffer_m=path_zoom_buffer_m,
                max_model_points=path_zoom_max_model_points,
                dpi=plot_dpi,
                model_marker_size=plot_model_marker_size,
                path_line_width=max(plot_path_line_width, 1.2),
                no_fly_prefixes=plot_no_fly_prefixes,
                no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
                always_flyable_prefixes=plot_always_flyable_prefixes,
                show_neighbor_edges=path_zoom_show_neighbor_edges,
                show_adjacent_nodes=path_zoom_show_adjacent_nodes,
                slowness_discrete_bounds=plot_slowness_discrete_bounds,
                cleanup_temp=bool(get_param("RUN_CLEANUP", False)),
                coordinate_mode=path_zoom_coordinate_mode,
                label_path_steps=path_zoom_label_steps,
                label_step_every=path_zoom_label_step_every,
                arrow_every=path_zoom_arrow_every,
            )
            exported["path_zoom_figure"] = str(zoom_figure_file)
            print(f"      Zoom diagnostic figure: {zoom_figure_file}")
        except Exception as exc:
            print(f"[WARNING] Could not plot path zoom diagnostic: {exc}")

    # ============================================================
    # Optional cleanup
    # ============================================================
    if run_cleanup:
        print("[CLEANUP] Removing intermediate files...")
        cleanup_summary = cleanup_intermediate_files(
            target_dirs=cleanup_target_dirs,
            patterns=cleanup_patterns,
            dry_run=cleanup_dry_run,
            remove_empty_dirs=cleanup_empty_dirs,
        )
        print_cleanup_summary(cleanup_summary)

    # ============================================================
    # Final print
    # ============================================================
    run_total_elapsed_s = time.perf_counter() - run_total_start_time
    result["processing_time_total_s"] = float(run_total_elapsed_s)
    result["processing_time_total_text"] = format_elapsed_time(run_total_elapsed_s)
    result["algorithm_search_time_s"] = float(algorithm_search_elapsed_s)
    result["algorithm_search_time_text"] = format_elapsed_time(algorithm_search_elapsed_s)

    timing_summary = {
        "algorithm": str(algorithm_name),
        "run_mode": str(run_mode),
        "success": bool(result.get("success", True)),
        "k_paths_requested": int(result.get("k_paths_requested", 1)),
        "k_paths_found": int(result.get("k_paths_found", 1 if result.get("path_indices") else 0)),
        "expanded_states": int(result.get("expanded_states", 0)),
        "algorithm_search_time_s": float(algorithm_search_elapsed_s),
        "algorithm_search_time_text": format_elapsed_time(algorithm_search_elapsed_s),
        "processing_time_total_s": float(run_total_elapsed_s),
        "processing_time_total_text": format_elapsed_time(run_total_elapsed_s),
    }

    if save_processing_time_json:
        timing_file = output_dir / f"{path_name}_{algorithm_name}_processing_time.json"
        maybe_write_processing_time_json(timing_file, timing_summary)
        exported["processing_time_json"] = str(timing_file)

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print("Exported path files:")
    for key, value in exported.items():
        print(f"  {key:20s}: {value}")

    print(f"Initiate figure : {initiate_figure_file}")
    print(f"Report figure   : {report_figure_file}")

    if print_processing_time:
        print("Processing time:")
        print(f"  algorithm search : {format_elapsed_time(algorithm_search_elapsed_s)}")
        print(f"  total workflow   : {format_elapsed_time(run_total_elapsed_s)}")
        if save_processing_time_json:
            print(f"  timing file      : {timing_file}")

    return timing_summary


def main():
    batch_start_time = time.perf_counter()
    print_processing_time = bool(get_param("PRINT_PROCESSING_TIME", True))

    # ============================================================
    # Possible DB/DK path connection calculation
    # ============================================================
    paths_df = None
    nodes_df = None

    if getattr(prm, "RUN_POSSIBLE_PATH_CALCULATION", True):
        paths_df, nodes_df = create_possible_paths(prm)
    else:
        print("\n[SKIP] Possible DB/DK path connection calculation is disabled.")
        print("       Set RUN_POSSIBLE_PATH_CALCULATION = True in parameters.py to enable it.")
        paths_df, nodes_df = None, None
        
    # Check the algorithms name
    algorithms = normalize_algorithm_names(get_param("ALGORITHM", "astar"))

    print("=" * 70)
    print("PATH-FINDING BATCH CONTROLLER")
    print("=" * 70)
    print("Algorithms to run:", ", ".join(algorithms))
    print("Run mode         :", str(get_param("RUN_MODE", "full")))
    print("=" * 70)

    failed = []
    algorithm_timing_summaries = []
    stop_on_failure = bool(get_param("STOP_ON_ALGORITHM_FAILURE", False))

    for i, algorithm_name in enumerate(algorithms, start=1):
        print("\n" + "#" * 70)
        print(f"RUN {i}/{len(algorithms)}: {algorithm_name}")
        print("#" * 70)

        algorithm_loop_start_time = time.perf_counter()
        try:
            timing_summary = run_one_algorithm(algorithm_name)
            algorithm_loop_elapsed_s = time.perf_counter() - algorithm_loop_start_time
            if isinstance(timing_summary, dict):
                algorithm_timing_summaries.append(timing_summary)
            else:
                algorithm_timing_summaries.append({
                    "algorithm": str(algorithm_name),
                    "processing_time_total_s": float(algorithm_loop_elapsed_s),
                    "processing_time_total_text": format_elapsed_time(algorithm_loop_elapsed_s),
                })
        except SystemExit:
            algorithm_loop_elapsed_s = time.perf_counter() - algorithm_loop_start_time
            failed.append(algorithm_name)
            algorithm_timing_summaries.append({
                "algorithm": str(algorithm_name),
                "success": False,
                "processing_time_total_s": float(algorithm_loop_elapsed_s),
                "processing_time_total_text": format_elapsed_time(algorithm_loop_elapsed_s),
            })
            print(f"[FAILED] Algorithm failed: {algorithm_name}")
            if print_processing_time:
                print(f"Processing time before failure: {format_elapsed_time(algorithm_loop_elapsed_s)}")
            if stop_on_failure:
                raise
        except Exception as exc:
            algorithm_loop_elapsed_s = time.perf_counter() - algorithm_loop_start_time
            failed.append(algorithm_name)
            algorithm_timing_summaries.append({
                "algorithm": str(algorithm_name),
                "success": False,
                "processing_time_total_s": float(algorithm_loop_elapsed_s),
                "processing_time_total_text": format_elapsed_time(algorithm_loop_elapsed_s),
                "error": str(exc),
            })
            print(f"[FAILED] Algorithm failed: {algorithm_name}")
            print(f"Reason: {exc}")
            if print_processing_time:
                print(f"Processing time before failure: {format_elapsed_time(algorithm_loop_elapsed_s)}")
            if stop_on_failure:
                raise

    print("\n" + "=" * 70)
    print("BATCH DONE")
    print("=" * 70)
    print("Algorithms requested:", ", ".join(algorithms))
    if failed:
        print("Failed algorithms:", ", ".join(failed))
        if stop_on_failure:
            sys.exit(1)
    else:
        print("All algorithms completed successfully.")

    if print_processing_time:
        batch_elapsed_s = time.perf_counter() - batch_start_time
        print("Processing time summary:")
        for item in algorithm_timing_summaries:
            name = item.get("algorithm", "unknown")
            total_text = item.get("processing_time_total_text")
            if total_text is None:
                total_text = format_elapsed_time(item.get("processing_time_total_s", 0.0))
            search_text = item.get("algorithm_search_time_text", None)
            if search_text:
                print(f"  {name:20s}: total={total_text}, search={search_text}")
            else:
                print(f"  {name:20s}: total={total_text}")
        print(f"  {'batch total':20s}: {format_elapsed_time(batch_elapsed_s)}")


if __name__ == "__main__":
    main()
