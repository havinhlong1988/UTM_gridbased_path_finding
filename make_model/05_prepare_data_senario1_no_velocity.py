#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare input data for scenario 1 without velocity.

Copies input data for:
  - roads from OSM
  - buildings from OpenBuildingMap or GlobalBuildingAtlas
  - DEM / topography from OpenTopography
  - OSM extra features
  - KML/KMZ plan exported points/rings XYZ

Default input source:
  output/01_HoaLac_studies_area/

Output:
  input/02_data_senario1_no_velocity/
"""

from pathlib import Path
import shutil


# ============================================================
# User settings
# ============================================================

PROJECT_DIR = Path(".").resolve()

INPUT_SOURCE_PATH = "output/01_HoaLac_studies_area"
INPUT_SOURCE_DIR = (PROJECT_DIR / INPUT_SOURCE_PATH).resolve()

OUT_DIR = PROJECT_DIR / "input" / "02_data_senario1_no_velocity"

# Choose building source:
#   "openbuildingmap"
#   "globalbuildingatlas"
#   "both"
BUILDING_DATA_SOURCE = "globalbuildingatlas"

# Low-memory options
LOW_MEMORY_MODE = True

# Copy in small chunks to avoid RAM/cache pressure on large DEM files
COPY_BUFFER_SIZE_MB = 8

# Recommended True.
# If the same filename already exists in the destination, skip it instead of
# creating _001, _002, _003 every time the script is rerun.
SKIP_EXISTING_FILES = True

# If True, overwrite existing copied files.
# Keep False unless you really want to refresh copied data.
OVERWRITE_EXISTING_FILES = False


# ============================================================
# Allowed file types
# ============================================================

ALLOWED_SUFFIXES = {
    ".xyz",
    ".gpkg",
    ".tif",
    ".tiff",
    ".csv",
    ".nc",
}


# ============================================================
# Simple helpers
# ============================================================

def is_inside(child: Path, parent: Path) -> bool:
    """Return True if child is inside parent."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def source_name() -> str:
    """Normalized building source name."""
    return BUILDING_DATA_SOURCE.lower().strip()


def is_openbuildingmap_file(f: Path) -> bool:
    """Return True if file looks like OpenBuildingMap / OBM building data."""
    p = str(f).lower()
    name = f.name.lower()

    if "openbuildingmap" in p:
        return True

    if "obm" in name:
        return True

    return False


def is_globalbuildingatlas_file(f: Path) -> bool:
    """Return True if file looks like GlobalBuildingAtlas / GBA building data."""
    p = str(f).lower()
    name = f.name.lower()

    if "globalbuildingatlas" in p and ("building" in name or "buildings" in name):
        return True

    if "gba" in name and ("building" in name or "buildings" in name):
        return True

    if "lod1" in name and ("building" in name or "buildings" in name):
        return True

    return False


def is_building_file(f: Path) -> bool:
    """Return True if file matches the selected building source."""
    source = source_name()

    if source not in {"openbuildingmap", "globalbuildingatlas", "both"}:
        raise ValueError(
            "Invalid BUILDING_DATA_SOURCE. "
            "Use 'openbuildingmap', 'globalbuildingatlas', or 'both'."
        )

    name = f.name.lower()

    # Do not copy metadata/inventory as building geometry
    if "selected_gba_5deg_tiles" in name:
        return False

    if "inventory" in name:
        return False

    if source == "openbuildingmap":
        return is_openbuildingmap_file(f)

    if source == "globalbuildingatlas":
        return is_globalbuildingatlas_file(f)

    if source == "both":
        return is_openbuildingmap_file(f) or is_globalbuildingatlas_file(f)

    return False


def is_road_file(f: Path) -> bool:
    """Return True if file looks like OSM road data."""
    p = str(f).lower()
    name = f.name.lower()

    if "/osm/" not in p:
        return False

    if f.suffix.lower() != ".xyz":
        return False

    if "road" in name or "roads" in name:
        return True

    return False


def is_dem_file(f: Path) -> bool:
    """Return True if file is inside OpenTopography folder."""
    p = str(f).lower()

    if "/opentopography/" not in p:
        return False

    if f.suffix.lower() in {".xyz", ".tif", ".tiff", ".csv", ".nc"}:
        return True

    return False


def is_osm_extra_file(f: Path) -> bool:
    """Return True if file is OSM extra feature but not roads."""
    p = str(f).lower()
    name = f.name.lower()

    if "/osm/" not in p:
        return False

    if f.suffix.lower() not in {".xyz", ".gpkg"}:
        return False

    # Keep road files only in roads category
    if "road" in name or "roads" in name:
        return False

    return True


def is_kml_plan_file(f: Path) -> bool:
    """Return True if file is KML plan exported XYZ."""
    p = str(f).lower()

    if "/kml_plan/" not in p:
        return False

    if f.suffix.lower() != ".xyz":
        return False

    return True


def classify_file(f: Path) -> str | None:
    """
    Classify file into one output category.

    Priority is important:
      roads before osm_extra_features
      buildings before generic rules
    """

    if f.suffix.lower() not in ALLOWED_SUFFIXES:
        return None

    if is_road_file(f):
        return "roads"

    if is_building_file(f):
        return "buildings"

    if is_dem_file(f):
        return "dem"

    if is_osm_extra_file(f):
        return "osm_extra_features"

    if is_kml_plan_file(f):
        return "kml_plan"

    return None


def iter_input_files_low_memory(root: Path):
    """
    Yield files one by one.

    This avoids collecting a large list of all files in RAM.
    """

    for f in root.rglob("*"):
        if not f.is_file():
            continue

        # Avoid copying from output target if user accidentally puts OUT_DIR inside source
        if is_inside(f, OUT_DIR):
            continue

        yield f.resolve()


def copy_file_chunked(src: Path, dst: Path):
    """
    Copy file in chunks.

    This is safer for large DEM rasters than reading/copying with a big buffer.
    """

    buffer_size = int(COPY_BUFFER_SIZE_MB * 1024 * 1024)

    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=buffer_size)

    shutil.copystat(src, dst)


def safe_copy(src: Path, dst_dir: Path) -> Path | None:
    """
    Copy src to dst_dir.

    If SKIP_EXISTING_FILES is True:
      - skip when same filename already exists.

    If OVERWRITE_EXISTING_FILES is True:
      - overwrite same filename.

    Otherwise:
      - append _001, _002, ...
    """

    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name

    if dst.exists():
        if OVERWRITE_EXISTING_FILES:
            copy_file_chunked(src, dst)
            return dst

        if SKIP_EXISTING_FILES:
            return None

        stem = src.stem
        suffix = src.suffix

        i = 1
        while True:
            new_dst = dst_dir / f"{stem}_{i:03d}{suffix}"
            if not new_dst.exists():
                dst = new_dst
                break
            i += 1

    copy_file_chunked(src, dst)
    return dst


def open_manifest():
    """Open manifest CSV and write header."""
    manifest = OUT_DIR / "manifest_copied_input_files.csv"

    f = open(manifest, "w", encoding="utf-8")
    f.write("category,file_type,source_file,copied_file,status\n")

    return manifest, f


def write_manifest_record(fh, category: str, src: Path, dst: Path | None, status: str):
    """Write one manifest row immediately."""
    file_type = src.suffix.lower().replace(".", "")

    if dst is None:
        copied_file = ""
    else:
        copied_file = str(dst)

    fh.write(f"{category},{file_type},{src},{copied_file},{status}\n")
    fh.flush()


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Input source folder does not exist: {INPUT_SOURCE_DIR}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    category_counts = {
        "roads": 0,
        "buildings": 0,
        "dem": 0,
        "osm_extra_features": 0,
        "kml_plan": 0,
    }

    skipped_existing = 0
    copied_count = 0

    print("=" * 70)
    print("Preparing data for scenario 1 without velocity")
    print(f"Project dir           : {PROJECT_DIR}")
    print(f"Input source dir      : {INPUT_SOURCE_DIR}")
    print(f"Output dir            : {OUT_DIR}")
    print(f"Building data source  : {BUILDING_DATA_SOURCE}")
    print(f"Low memory mode       : {LOW_MEMORY_MODE}")
    print(f"Copy buffer           : {COPY_BUFFER_SIZE_MB} MB")
    print(f"Skip existing files   : {SKIP_EXISTING_FILES}")
    print(f"Overwrite existing    : {OVERWRITE_EXISTING_FILES}")
    print("=" * 70)

    manifest, manifest_fh = open_manifest()

    try:
        for src in iter_input_files_low_memory(INPUT_SOURCE_DIR):
            category = classify_file(src)

            if category is None:
                continue

            dst_category_dir = OUT_DIR / category
            dst = safe_copy(src, dst_category_dir)

            if dst is None:
                skipped_existing += 1
                write_manifest_record(
                    manifest_fh,
                    category,
                    src,
                    None,
                    "skipped_existing",
                )
                continue

            copied_count += 1
            category_counts[category] += 1

            write_manifest_record(
                manifest_fh,
                category,
                src,
                dst,
                "copied",
            )

            print(f"[{category}] copied: {src.relative_to(PROJECT_DIR)}")
            print(f"       -> {dst.relative_to(PROJECT_DIR)}")

    finally:
        manifest_fh.close()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Copied files          : {copied_count}")
    print(f"Skipped existing      : {skipped_existing}")
    print(f"Output folder         : {OUT_DIR}")
    print(f"Manifest              : {manifest}")
    print("-" * 70)

    for category, count in category_counts.items():
        print(f"{category:20s}: {count}")

    print("=" * 70)


if __name__ == "__main__":
    main()