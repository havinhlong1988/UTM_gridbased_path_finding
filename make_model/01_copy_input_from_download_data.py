#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy only necessary downloaded GIS/model input files from:
    {PROJECT_ROOT}/downloaddata/output/01_HoaLac_studies_area

to:
    {PROJECT_ROOT}/make_model/output/01_HoaLac_studies_area

Purpose:
    Prepare a clean, lightweight input dataset for the make_model step.

This script intentionally does NOT copy large raw download files such as:
    - GlobalBuildingAtlas raw parquet tiles
    - bbox cache parquet
    - figures
    - temporary files

Recommended location:
    {PROJECT_ROOT}/make_model/01_copy_input_from_download_data.py

Run:
    cd {PROJECT_ROOT}
    python make_model/01_copy_input_from_download_data.py

or:
    cd {PROJECT_ROOT}/make_model
    python 01_copy_input_from_download_data.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from datetime import datetime


# ============================================================
# User parameters
# ============================================================

# If None, the script auto-detects the project root by searching for
# folders named "downloaddata" and "make_model".
PROJECT_DIR: Path | None = None

# Source and destination are created from PROJECT_DIR.
SOURCE_REL = Path("downloaddata/output/01_HoaLac_studies_area")
DEST_REL = Path("make_model/output/01_HoaLac_studies_area")

# Existing destination files will be replaced when True.
OVERWRITE_EXISTING = True

# If True, remove destination folder before copying.
# Keep False for safety.
CLEAN_DESTINATION_FIRST = False

# If True, only print what would be copied.
DRY_RUN = False

# Optional extras.
COPY_FIGURES = False
COPY_GBA_OBJ = True
COPY_GBA_FULL_ATTRIBUTES_CSV = True
COPY_GBA_BBOX_CACHE_PARQUET = False

# If True, raise an error when important files are missing.
# Keep False if some sources were intentionally skipped.
STRICT_CRITICAL_FILES = False


# ============================================================
# Necessary files for next make_model stage
# ============================================================

# These are copied while preserving their relative path below
# output/01_HoaLac_studies_area.
COPY_PATTERNS = [
    # ------------------------------
    # Study-area metadata / AOI
    # ------------------------------
    "metadata/study_area_aoi.gpkg",
    "metadata/study_area_aoi.geojson",
    "metadata/project_metadata.json",
    "metadata/*.csv",

    # ------------------------------
    # OpenStreetMap roads
    # ------------------------------
    "osm/roads/osm_roads_edges.gpkg",
    "osm/roads/osm_roads_edges.geojson",
    "osm/roads/osm_roads_nodes.gpkg",
    "osm/roads/osm_road_class_summary.csv",

    # ------------------------------
    # OpenStreetMap extra features
    # useful for water, landuse, amenity, aeroway, railway, etc.
    # ------------------------------
    "osm/extra_features/osm_extra_features.gpkg",
    "osm/extra_features/osm_extra_features.geojson",
    "osm/extra_features/osm_extra_tag_summary.csv",

    # ------------------------------
    # OpenTopography DEM and terrain products
    # ------------------------------
    "opentopography/*_dem_wgs84.tif",
    "opentopography/*_dem_utm.tif",
    "opentopography/terrain_products/*.tif",
    "opentopography/terrain_products/*.csv",

    # ------------------------------
    # OpenBuildingMap clipped buildings
    # ------------------------------
    "openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg",
    "openbuildingmap/clipped/obm_buildings_hoalac_clipped.geojson",
    "openbuildingmap/clipped/obm_summary.csv",

    # ------------------------------
    # KML/KMZ plan exported files, if available
    # DB, DK, FLZ, RA, setup_fly_control, rings, etc.
    # ------------------------------
    "kml_plan/*.xyz",
    "kml_plan/*.csv",
    "kml_plan/*.gpkg",
    "kml_plan/*.geojson",

    # ------------------------------
    # GlobalBuildingAtlas LoD1 metadata
    # ------------------------------
    "globalbuildingatlas_lod1/metadata/hoalac_polygon.gpkg",
    "globalbuildingatlas_lod1/metadata/selected_gba_5deg_tiles.csv",
    "globalbuildingatlas_lod1/metadata/selected_gba_5deg_tiles.gpkg",
    "globalbuildingatlas_lod1/metadata/gba_lod1_summary.csv",
    "globalbuildingatlas_lod1/metadata/selected_tiles_download_status.csv",

    # ------------------------------
    # GlobalBuildingAtlas LoD1 processed outputs
    # IMPORTANT for model building / density / obstacle extraction
    # ------------------------------
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac.xyz",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac_with_info.xyz",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_vertices_hoalac.xyz",
]

OPTIONAL_PATTERNS = []

if COPY_GBA_FULL_ATTRIBUTES_CSV:
    OPTIONAL_PATTERNS.append(
        "globalbuildingatlas_lod1/processed/gba_lod1_buildings_full_attributes.csv"
    )

if COPY_GBA_OBJ:
    OPTIONAL_PATTERNS.append(
        "globalbuildingatlas_lod1/processed/gba_lod1_buildings_lod1.obj"
    )

if COPY_GBA_BBOX_CACHE_PARQUET:
    OPTIONAL_PATTERNS.append(
        "globalbuildingatlas_lod1/processed/gba_lod1_buildings_bbox_filtered_lowram.parquet"
    )

if COPY_FIGURES:
    OPTIONAL_PATTERNS.extend([
        "globalbuildingatlas_lod1/figures/*.png",
        "figures/**/*.png",
    ])

# These are important for most next-stage model-building workflows.
# Missing files are reported clearly.
CRITICAL_PATTERNS = [
    "metadata/study_area_aoi.gpkg",
    "osm/roads/osm_roads_edges.gpkg",
    "opentopography/*_dem_utm.tif",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac.xyz",
    "globalbuildingatlas_lod1/processed/gba_lod1_buildings_vertices_hoalac.xyz",
]

# Never copy these by accident.
EXCLUDE_PARTS = {
    "raw_tiles",
    "__pycache__",
    ".ipynb_checkpoints",
}

EXCLUDE_SUFFIXES = {
    ".tmp",
    ".temp",
    ".lock",
}


# ============================================================
# Helpers
# ============================================================

def find_project_root() -> Path:
    """Find project root containing both downloaddata and make_model."""
    if PROJECT_DIR is not None:
        return Path(PROJECT_DIR).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent

    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "downloaddata").is_dir() and (candidate / "make_model").is_dir():
            return candidate

    # Safe fallback if the script is located directly in make_model or downloaddata.
    if script_dir.name in {"make_model", "downloaddata"}:
        return script_dir.parent

    # Last fallback: current working directory.
    return Path.cwd().resolve()


def is_excluded(path: Path) -> bool:
    """Return True if file should not be copied."""
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def human_size(nbytes: int) -> str:
    """Human-readable file size."""
    size = float(nbytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def copy_one_file(src_file: Path, src_root: Path, dst_root: Path, manifest_rows: list[dict]) -> None:
    """Copy one file while preserving relative path from src_root."""
    rel = src_file.relative_to(src_root)
    dst_file = dst_root / rel

    if is_excluded(rel):
        manifest_rows.append({
            "status": "excluded",
            "source": str(src_file),
            "destination": str(dst_file),
            "size_bytes": src_file.stat().st_size if src_file.exists() else 0,
            "size_human": human_size(src_file.stat().st_size) if src_file.exists() else "0 B",
        })
        return

    if dst_file.exists() and not OVERWRITE_EXISTING:
        manifest_rows.append({
            "status": "exists_skip",
            "source": str(src_file),
            "destination": str(dst_file),
            "size_bytes": src_file.stat().st_size,
            "size_human": human_size(src_file.stat().st_size),
        })
        return

    if not DRY_RUN:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    manifest_rows.append({
        "status": "copied" if not DRY_RUN else "dry_run_copy",
        "source": str(src_file),
        "destination": str(dst_file),
        "size_bytes": src_file.stat().st_size,
        "size_human": human_size(src_file.stat().st_size),
    })


def copy_patterns(src_root: Path, dst_root: Path, patterns: list[str]) -> tuple[list[dict], list[str]]:
    """Copy files matching patterns. Return manifest rows and missing patterns."""
    manifest_rows: list[dict] = []
    missing_patterns: list[str] = []
    copied_sources: set[Path] = set()

    for pattern in patterns:
        matches = sorted(src_root.glob(pattern))
        matches = [p for p in matches if p.is_file()]

        if not matches:
            missing_patterns.append(pattern)
            manifest_rows.append({
                "status": "missing_pattern",
                "source": str(src_root / pattern),
                "destination": "",
                "size_bytes": 0,
                "size_human": "0 B",
            })
            continue

        for src_file in matches:
            src_file = src_file.resolve()
            if src_file in copied_sources:
                continue
            copied_sources.add(src_file)
            copy_one_file(src_file, src_root, dst_root, manifest_rows)

    return manifest_rows, missing_patterns


def check_critical_files(src_root: Path) -> list[str]:
    """Check critical files/patterns in source folder."""
    missing = []
    for pattern in CRITICAL_PATTERNS:
        matches = [p for p in src_root.glob(pattern) if p.is_file()]
        if not matches:
            missing.append(pattern)
    return missing


def write_manifest(dst_root: Path, manifest_rows: list[dict]) -> Path:
    """Write copy manifest CSV in destination root."""
    manifest_file = dst_root / "copy_manifest_from_downloaddata.csv"
    if DRY_RUN:
        manifest_file = dst_root / "copy_manifest_from_downloaddata_DRY_RUN.csv"

    if not DRY_RUN:
        manifest_file.parent.mkdir(parents=True, exist_ok=True)

    # For dry run, still create destination folder so user can inspect manifest.
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["status", "source", "destination", "size_bytes", "size_human"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_file


def report_file(label: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"  [OK]      {label:<22} {path}")
    else:
        print(f"  [MISSING] {label:<22} {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    project_dir = find_project_root()
    src_root = project_dir / SOURCE_REL
    dst_root = project_dir / DEST_REL

    print("\n========== COPY INPUT FROM DOWNLOAD DATA ==========")
    print(f"Project dir: {project_dir}")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Dry run:     {DRY_RUN}")

    if not src_root.exists():
        raise FileNotFoundError(
            "\n[ERROR] Source folder does not exist:\n"
            f"  {src_root}\n\n"
            "Expected path:\n"
            "  {PROJECT_ROOT}/downloaddata/output/01_HoaLac_studies_area\n"
        )

    critical_missing = check_critical_files(src_root)
    if critical_missing:
        print("\n[WARN] Some critical source files are missing:")
        for p in critical_missing:
            print(f"  - {p}")
        if STRICT_CRITICAL_FILES:
            raise FileNotFoundError("Critical files are missing. Stop because STRICT_CRITICAL_FILES=True.")

    if CLEAN_DESTINATION_FIRST and dst_root.exists() and not DRY_RUN:
        print(f"\n[INFO] Removing old destination folder: {dst_root}")
        shutil.rmtree(dst_root)

    all_patterns = COPY_PATTERNS + OPTIONAL_PATTERNS
    manifest_rows, missing_patterns = copy_patterns(src_root, dst_root, all_patterns)

    # Add run metadata to destination.
    meta_file = dst_root / "copy_info.txt"
    if not DRY_RUN:
        dst_root.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write("Copy input from downloaddata\n")
            f.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Project dir: {project_dir}\n")
            f.write(f"Source: {src_root}\n")
            f.write(f"Destination: {dst_root}\n")
            f.write(f"Overwrite existing: {OVERWRITE_EXISTING}\n")
            f.write(f"Clean destination first: {CLEAN_DESTINATION_FIRST}\n")

    manifest_file = write_manifest(dst_root, manifest_rows)

    copied = [r for r in manifest_rows if r["status"] in {"copied", "dry_run_copy"}]
    copied_bytes = sum(int(r["size_bytes"]) for r in copied)
    excluded = [r for r in manifest_rows if r["status"] == "excluded"]

    print("\n========== COPY SUMMARY ==========")
    print(f"Copied files:       {len(copied)}")
    print(f"Copied size:        {human_size(copied_bytes)}")
    print(f"Excluded files:     {len(excluded)}")
    print(f"Missing patterns:   {len(missing_patterns)}")
    print(f"Manifest:           {manifest_file}")

    if missing_patterns:
        print("\n[INFO] Missing patterns. This is okay if those data sources were not downloaded:")
        for p in missing_patterns:
            print(f"  - {p}")

    print("\nImportant copied outputs:")
    report_file("AOI", dst_root / "metadata/study_area_aoi.gpkg")
    report_file("OSM roads", dst_root / "osm/roads/osm_roads_edges.gpkg")
    report_file("OSM features", dst_root / "osm/extra_features/osm_extra_features.gpkg")
    report_file("DEM UTM", next(dst_root.glob("opentopography/*_dem_utm.tif"), dst_root / "opentopography/NO_DEM_UTM_FOUND.tif"))
    report_file("Slope degree", dst_root / "opentopography/terrain_products/slope_degree.tif")
    report_file("OBM buildings", dst_root / "openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg")
    report_file("GBA buildings", dst_root / "globalbuildingatlas_lod1/processed/gba_lod1_buildings_hoalac_clipped.gpkg")
    report_file("GBA centroid XYZ", dst_root / "globalbuildingatlas_lod1/processed/gba_lod1_buildings_centroid_hoalac.xyz")
    report_file("GBA vertices XYZ", dst_root / "globalbuildingatlas_lod1/processed/gba_lod1_buildings_vertices_hoalac.xyz")
    report_file("KML plan dir", dst_root / "kml_plan")

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
