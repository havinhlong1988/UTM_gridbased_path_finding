#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check and report building height data from:
  1. OpenBuildingMap (OBM)
  2. GlobalBuildingAtlas LoD1 (GBA)

Run from:
  make_model/
or project root.

Outputs:
  output/01_HoaLac_studies_area/building_height_check/
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# Candidate input files
# ============================================================

OBM_CANDIDATE_FILES = [
    # If running from make_model/
    Path("../downloaddata/output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"),

    # If copied into make_model/output/
    Path("output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"),

    # If running from project root
    Path("downloaddata/output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"),
]

GBA_CANDIDATE_FILES = [
    # If running from make_model/
    Path("../downloaddata/output/01_HoaLac_studies_area/globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg"),

    # If copied into make_model/output/
    Path("output/01_HoaLac_studies_area/globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg"),

    # If running from project root
    Path("downloaddata/output/01_HoaLac_studies_area/globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg"),
]

OUT_DIR_CANDIDATES = [
    # If running from make_model/
    Path("output/01_HoaLac_studies_area/building_height_check"),

    # If running from project root
    Path("make_model/output/01_HoaLac_studies_area/building_height_check"),
]


# ============================================================
# Utilities
# ============================================================

def find_existing_file(candidates, label):
    print(f"\n========== CHECK {label} FILE ==========")
    for f in candidates:
        status = "FOUND" if f.exists() and f.stat().st_size > 0 else "missing"
        print(f"  {status:<8} {f}")

        if f.exists() and f.stat().st_size > 0:
            return f

    return None


def choose_output_dir():
    # Prefer make_model/output if running inside make_model.
    for d in OUT_DIR_CANDIDATES:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            pass

    d = Path("building_height_check")
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_obm_height_to_m(value, default_m=3.0):
    """
    Convert OBM GEM taxonomy height strings to approximate meters.

    Examples:
        HHT:10.0   -> 10.0 m
        H:2        -> 6.0 m, assuming 3 m/story
        HBET:1-3   -> 6.0 m, average stories * 3 m
        UNK / NaN  -> default_m
    """
    if value is None or pd.isna(value):
        return default_m

    txt = str(value).strip()
    if not txt or txt.upper() in ["UNK", "NULL", "NAN", "NONE"]:
        return default_m

    parts = txt.split("+")

    # Prefer explicit meter height.
    for part in parts:
        part = part.strip()
        if part.startswith("HHT:"):
            try:
                return float(part.split(":", 1)[1])
            except Exception:
                pass

    # Number of stories.
    for part in parts:
        part = part.strip()
        if part.startswith("H:"):
            try:
                return float(part.split(":", 1)[1]) * 3.0
            except Exception:
                pass

    # Range of stories.
    for part in parts:
        part = part.strip()
        if part.startswith("HBET:"):
            try:
                rng = part.split(":", 1)[1]
                if "-" in rng:
                    a, b = rng.split("-", 1)
                    return ((float(a) + float(b)) / 2.0) * 3.0
                return float(rng) * 3.0
            except Exception:
                pass

    return default_m


def get_numeric_height_series(gdf, source_name):
    """
    Return best numeric height series for OBM or GBA.
    """
    if source_name.upper() == "GBA":
        if "height_m" in gdf.columns:
            return pd.to_numeric(gdf["height_m"], errors="coerce"), "height_m"
        if "height" in gdf.columns:
            return pd.to_numeric(gdf["height"], errors="coerce"), "height"

    if source_name.upper() == "OBM":
        if "height_m" in gdf.columns:
            return pd.to_numeric(gdf["height_m"], errors="coerce"), "height_m"

        if "height" in gdf.columns:
            # OBM height often stores GEM taxonomy strings.
            converted = gdf["height"].apply(parse_obm_height_to_m)
            return pd.to_numeric(converted, errors="coerce"), "height -> parsed_height_m"

        # Fallback: search for possible numerical height/floor columns.
        for col in gdf.columns:
            name = col.lower()
            if any(k in name for k in ["height", "level", "floor", "elev"]):
                vals = pd.to_numeric(gdf[col], errors="coerce")
                if vals.notna().any():
                    return vals, col

    return pd.Series(dtype=float), "NOT_FOUND"


def calculate_area_if_missing(gdf):
    """
    Add footprint_area_m2 if not available.
    """
    gdf = gdf.copy()

    if "footprint_area_m2" in gdf.columns:
        gdf["footprint_area_m2"] = pd.to_numeric(gdf["footprint_area_m2"], errors="coerce")
        return gdf

    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf["footprint_area_m2"] = gdf_utm.geometry.area.to_numpy()
    except Exception as e:
        print(f"[WARN] Could not calculate footprint_area_m2: {e}")
        gdf["footprint_area_m2"] = pd.NA

    return gdf


def report_source(gdf, source_name, out_dir):
    """
    Print and save height report for one source.
    """
    print(f"\n========== {source_name} BUILDING HEIGHT REPORT ==========")

    print("\nColumns:")
    print(gdf.columns.tolist())

    print("\nFirst rows:")
    print(gdf.head())

    print("\nPossible height / level / floor / elevation columns:")
    found_cols = []
    for col in gdf.columns:
        if any(k in col.lower() for k in ["height", "level", "floor", "elev"]):
            found_cols.append(col)
            print(f"\n--- {col} ---")
            print(gdf[col].describe())
            print(gdf[col].dropna().head(20))

    if not found_cols:
        print("No obvious height-like column found.")

    height_m, height_col = get_numeric_height_series(gdf, source_name)
    gdf = calculate_area_if_missing(gdf)

    valid_height = height_m.replace([float("inf"), float("-inf")], pd.NA).dropna()
    area_m2 = pd.to_numeric(gdf["footprint_area_m2"], errors="coerce")
    valid_area = area_m2.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if len(valid_height) > 0:
        volume_m3 = area_m2 * height_m
        valid_volume = volume_m3.replace([float("inf"), float("-inf")], pd.NA).dropna()
    else:
        volume_m3 = pd.Series(dtype=float)
        valid_volume = pd.Series(dtype=float)

    summary = {
        "source": source_name,
        "n_buildings": int(len(gdf)),
        "height_column_used": height_col,
        "height_valid_count": int(valid_height.count()),
        "height_missing_count": int(len(gdf) - valid_height.count()),
        "height_min_m": float(valid_height.min()) if len(valid_height) else None,
        "height_mean_m": float(valid_height.mean()) if len(valid_height) else None,
        "height_median_m": float(valid_height.median()) if len(valid_height) else None,
        "height_max_m": float(valid_height.max()) if len(valid_height) else None,
        "footprint_area_valid_count": int(valid_area.count()),
        "footprint_area_total_m2": float(valid_area.sum()) if len(valid_area) else None,
        "footprint_area_mean_m2": float(valid_area.mean()) if len(valid_area) else None,
        "footprint_area_median_m2": float(valid_area.median()) if len(valid_area) else None,
        "footprint_area_max_m2": float(valid_area.max()) if len(valid_area) else None,
        "volume_total_m3": float(valid_volume.sum()) if len(valid_volume) else None,
        "volume_mean_m3": float(valid_volume.mean()) if len(valid_volume) else None,
        "volume_median_m3": float(valid_volume.median()) if len(valid_volume) else None,
        "volume_max_m3": float(valid_volume.max()) if len(valid_volume) else None,
    }

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save per-building useful table.
    out_table = pd.DataFrame({
        "source": source_name,
        "height_m": height_m,
        "footprint_area_m2": area_m2,
        "volume_m3": area_m2 * height_m,
    })

    if "centroid_lon" in gdf.columns and "centroid_lat" in gdf.columns:
        out_table["centroid_lon"] = gdf["centroid_lon"]
        out_table["centroid_lat"] = gdf["centroid_lat"]

    out_csv = out_dir / f"{source_name.lower()}_height_area_volume_values.csv"
    out_table.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved values: {out_csv}")

    return summary


def main():
    out_dir = choose_output_dir()

    obm_file = find_existing_file(OBM_CANDIDATE_FILES, "OBM")
    gba_file = find_existing_file(GBA_CANDIDATE_FILES, "GBA")

    summaries = []

    if obm_file is not None:
        print(f"\n[OK] Reading OBM: {obm_file}")
        obm = gpd.read_file(obm_file)
        summaries.append(report_source(obm, "OBM", out_dir))
    else:
        print("\n[WARN] OBM file not found. Skip OBM height report.")

    if gba_file is not None:
        print(f"\n[OK] Reading GBA: {gba_file}")
        gba = gpd.read_file(gba_file)
        summaries.append(report_source(gba, "GBA", out_dir))
    else:
        print("\n[WARN] GBA file not found. Skip GBA height report.")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = out_dir / "building_height_summary_OBM_vs_GBA.csv"
        summary_df.to_csv(summary_file, index=False)

        print("\n========== COMBINED SUMMARY ==========")
        print(summary_df.to_string(index=False))
        print(f"\n[OK] Saved combined summary: {summary_file}")

    print(f"\nOutput folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
