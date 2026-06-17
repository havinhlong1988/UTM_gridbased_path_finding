#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predefined risk map + emergency map + final cost map builder.

Workflow:
    model = load_labelled_model(...)
    model = build_predefined_costmap(model, overwrite_slowness=True)
    graph = build_grid_graph(model, ...)

The planner then uses model['slowness'], which can be replaced by
model['effective_slowness'] before graph construction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# Coordinate helpers
# ============================================================

def detect_lonlat_xy(model: pd.DataFrame) -> bool:
    x = pd.to_numeric(model["x"], errors="coerce")
    y = pd.to_numeric(model["y"], errors="coerce")

    return (
        x.dropna().between(-180.0, 180.0).all()
        and y.dropna().between(-90.0, 90.0).all()
    )


def xy_to_local_meters(model: pd.DataFrame) -> np.ndarray:
    """Convert x/y to local metric coordinates for distance calculation."""
    x = pd.to_numeric(model["x"], errors="coerce").to_numpy(dtype=float, copy=True)
    y = pd.to_numeric(model["y"], errors="coerce").to_numpy(dtype=float, copy=True)

    if detect_lonlat_xy(model):
        lon0 = float(np.nanmean(x))
        lat0 = float(np.nanmean(y))
        lat0_rad = math.radians(lat0)

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(lat0_rad)

        xm = (x - lon0) * meters_per_deg_lon
        ym = (y - lat0) * meters_per_deg_lat
    else:
        xm = x
        ym = y

    return np.column_stack([xm, ym])


# ============================================================
# Normalization helpers
# ============================================================

def normalize_series_01(values, invert: bool = False) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
    valid = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)

    if valid.sum() == 0:
        return out

    vmin = float(np.nanmin(arr[valid]))
    vmax = float(np.nanmax(arr[valid]))

    if np.isclose(vmin, vmax):
        out[valid] = 0.0
    else:
        out[valid] = (arr[valid] - vmin) / (vmax - vmin)

    out[~valid] = 0.0

    if invert:
        out = 1.0 - out

    return np.clip(out, 0.0, 1.0)


def get_prefix(model: pd.DataFrame) -> pd.Series:
    if "label_prefix" in model.columns:
        return model["label_prefix"].astype(str).str.upper()

    if "label" in model.columns:
        return (
            model["label"]
            .astype(str)
            .str.extract(r"^([A-Za-z]+)", expand=False)
            .fillna("")
            .str.upper()
        )

    return pd.Series([""] * len(model), index=model.index)


# ============================================================
# Risk map
# ============================================================

def build_predefined_risk_map(
    model: pd.DataFrame,
    base_risk: float = 0.05,
    prefix_risk: dict | None = None,
    risk_columns: dict | None = None,
    no_fly_slowness_threshold: float = 10.0,
    no_fly_risk: float = 1.0,
) -> np.ndarray:
    """
    Build risk_map in range 0-1.

    Sources:
      1. label-prefix risk
      2. optional numeric risk layers in model
      3. high-slowness/no-fly override
    """
    n = len(model)
    risk = np.full(n, float(base_risk), dtype=float)

    prefix = get_prefix(model)

    if prefix_risk is None:
        prefix_risk = {
            "N": 0.05,
            "DB": 0.0,
            "DK": 0.0,
            "FLZ": 0.10,
            "RA": 1.0,
        }

    for p, value in prefix_risk.items():
        mask = prefix == str(p).upper()
        risk[mask.to_numpy()] = float(value)

    # Optional continuous layers.
    # Example: population_density, building_density, building_height.
    if risk_columns is not None:
        for col, weight in risk_columns.items():
            if col not in model.columns:
                continue
            layer = normalize_series_01(model[col], invert=False)
            risk += float(weight) * layer

    if "slowness" in model.columns:
        slow = pd.to_numeric(model["slowness"], errors="coerce").fillna(np.inf)
        slow_mask = slow >= float(no_fly_slowness_threshold)
        risk[slow_mask.to_numpy()] = np.maximum(
            risk[slow_mask.to_numpy()],
            float(no_fly_risk),
        )

    return np.clip(risk, 0.0, 1.0)


# ============================================================
# Emergency map
# ============================================================

def compute_distance_to_emergency_nodes(
    model: pd.DataFrame,
    emergency_prefixes=("DB", "DK", "FLZ"),
) -> np.ndarray:
    """Compute distance to nearest emergency/safe node."""
    prefix = get_prefix(model)
    emergency_prefixes = tuple(str(p).upper() for p in emergency_prefixes)
    emergency_mask = prefix.isin(emergency_prefixes).to_numpy()

    xy = xy_to_local_meters(model)

    if emergency_mask.sum() == 0:
        return np.full(len(model), np.inf, dtype=float)

    emergency_xy = xy[emergency_mask]
    tree = cKDTree(emergency_xy)
    dist, _ = tree.query(xy, k=1)
    return dist.astype(float)


def build_predefined_emergency_map(
    model: pd.DataFrame,
    emergency_prefixes=("DB", "DK", "FLZ"),
    emergency_distance_decay_m: float = 1000.0,
    emergency_score_columns: dict | None = None,
    restricted_prefixes=("RA",),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build emergency_score and emergency_distance_m.

    score = exp(-distance_to_nearest_emergency_node / decay)
    """
    emergency_distance_m = compute_distance_to_emergency_nodes(
        model=model,
        emergency_prefixes=emergency_prefixes,
    )

    decay = max(float(emergency_distance_decay_m), 1.0)
    emergency_score = np.exp(-emergency_distance_m / decay)
    emergency_score[~np.isfinite(emergency_score)] = 0.0

    if emergency_score_columns is not None:
        for col, weight in emergency_score_columns.items():
            if col not in model.columns:
                continue
            layer = normalize_series_01(model[col], invert=False)
            emergency_score += float(weight) * layer

    prefix = get_prefix(model)
    restricted_prefixes = tuple(str(p).upper() for p in restricted_prefixes)
    restricted_mask = prefix.isin(restricted_prefixes).to_numpy()
    emergency_score[restricted_mask] = 0.0

    emergency_score = np.clip(emergency_score, 0.0, 1.0)
    return emergency_score, emergency_distance_m


# ============================================================
# Final cost map
# ============================================================

def build_predefined_costmap(
    model: pd.DataFrame,
    use_risk_map: bool = True,
    use_emergency_map: bool = True,
    base_risk: float = 0.05,
    prefix_risk: dict | None = None,
    risk_columns: dict | None = None,
    emergency_prefixes=("DB", "DK", "FLZ"),
    emergency_distance_decay_m: float = 1000.0,
    emergency_score_columns: dict | None = None,
    restricted_prefixes=("RA",),
    no_fly_slowness_threshold: float = 10.0,
    no_fly_risk: float = 1.0,
    travel_weight: float = 1.0,
    risk_weight: float = 3.0,
    emergency_weight: float = 1.0,
    min_effective_slowness: float = 1e-9,
    max_effective_slowness: float | None = None,
    overwrite_slowness: bool = False,
) -> pd.DataFrame:
    """
    Add costmap columns to model.

    Formula:
        effective_slowness =
            base_slowness * travel_weight
            * (1 + risk_weight * risk_map)
            * (1 + emergency_weight * emergency_penalty)

    where:
        emergency_penalty = 1 - emergency_score
    """
    out = model.copy()

    if "slowness" not in out.columns:
        raise ValueError("model must contain a 'slowness' column.")

    base_slowness = pd.to_numeric(out["slowness"], errors="coerce").to_numpy(dtype=float, copy=True)
    base_slowness[~np.isfinite(base_slowness)] = 0.0

    if "base_slowness" not in out.columns:
        out["base_slowness"] = base_slowness

    if use_risk_map:
        risk_map = build_predefined_risk_map(
            model=out,
            base_risk=base_risk,
            prefix_risk=prefix_risk,
            risk_columns=risk_columns,
            no_fly_slowness_threshold=no_fly_slowness_threshold,
            no_fly_risk=no_fly_risk,
        )
    else:
        risk_map = np.zeros(len(out), dtype=float)

    if use_emergency_map:
        emergency_score, emergency_distance_m = build_predefined_emergency_map(
            model=out,
            emergency_prefixes=emergency_prefixes,
            emergency_distance_decay_m=emergency_distance_decay_m,
            emergency_score_columns=emergency_score_columns,
            restricted_prefixes=restricted_prefixes,
        )
    else:
        emergency_score = np.ones(len(out), dtype=float)
        emergency_distance_m = np.zeros(len(out), dtype=float)

    emergency_penalty = 1.0 - emergency_score

    cost_multiplier = (
        float(travel_weight)
        * (1.0 + float(risk_weight) * risk_map)
        * (1.0 + float(emergency_weight) * emergency_penalty)
    )

    effective_slowness = (base_slowness * cost_multiplier).astype(float, copy=True)

    slow_mask = base_slowness >= float(no_fly_slowness_threshold)
    effective_slowness[slow_mask] = np.maximum(
        effective_slowness[slow_mask],
        base_slowness[slow_mask],
    )

    effective_slowness = np.maximum(effective_slowness, float(min_effective_slowness))

    if max_effective_slowness is not None:
        effective_slowness = np.minimum(effective_slowness, float(max_effective_slowness))

    out["risk_map"] = risk_map
    out["emergency_score"] = emergency_score
    out["emergency_distance_m"] = emergency_distance_m
    out["emergency_penalty"] = emergency_penalty
    out["cost_multiplier"] = cost_multiplier
    out["effective_slowness"] = effective_slowness

    if overwrite_slowness:
        out["slowness"] = out["effective_slowness"]

    return out


def save_costmap_outputs(
    model: pd.DataFrame,
    output_dir: Path,
    name: str = "costmap_senario1",
) -> dict:
    """Save costmap table to CSV and XYZ-like file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / f"{name}.csv"
    xyz_file = output_dir / f"{name}.xyz"

    model.to_csv(csv_file, index=False)

    cols = [
        c for c in [
            "x",
            "y",
            "z",
            "base_slowness",
            "slowness",
            "risk_map",
            "emergency_score",
            "emergency_distance_m",
            "emergency_penalty",
            "cost_multiplier",
            "effective_slowness",
            "label",
        ]
        if c in model.columns
    ]

    model[cols].to_csv(
        xyz_file,
        sep=" ",
        index=False,
        header=True,
        float_format="%.10f",
    )

    return {
        "costmap_csv": str(csv_file),
        "costmap_xyz": str(xyz_file),
    }
