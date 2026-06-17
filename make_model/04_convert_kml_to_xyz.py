#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch export Google Earth KML/KMZ marked geometries to XYZ.

Input:
    kml/plan/*.kml
    kml/plan/*.kmz

Output:
    output/01_HoaLac_studies_area/kml_plan/*.xyz

Export format:
    name lon lat elevation

Supported geometries:
    - Point
    - LineString
    - Polygon outer boundary (LinearRing)

If elevation is missing/NaN, elevation = 0.
"""

from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = Path("kml/plan")
OUT_DIR = Path("output/01_HoaLac_studies_area/kml_plan")

WRITE_HEADER = True

# Export extra geometry types
EXPORT_POINTS = True
EXPORT_LINESTRINGS = True
EXPORT_POLYGON_RINGS = True

# Keep closing duplicate point for polygon rings
# Useful if you want to plot a closed circle/outline directly
KEEP_CLOSED_RING_LAST_POINT = True


# ============================================================
# HELPERS
# ============================================================

def extract_kml_from_kmz(kmz_file: Path) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="kmz_extract_"))

    with zipfile.ZipFile(kmz_file, "r") as z:
        kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]

        if not kml_files:
            raise FileNotFoundError(f"No .kml file found inside: {kmz_file}")

        kml_name = "doc.kml" if "doc.kml" in kml_files else kml_files[0]
        z.extract(kml_name, tmp_dir)

    return tmp_dir / kml_name


def clean_name(text: str | None) -> str:
    if text is None:
        return "Unnamed"

    text = str(text).strip()

    if text == "":
        return "Unnamed"

    text = text.replace(" ", "_")
    text = text.replace("\t", "_")
    text = text.replace(",", "_")

    return text


def safe_float(value: str | None, default: float | None = 0.0) -> float | None:
    if value is None:
        return default

    try:
        v = float(str(value).strip())
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def get_namespace(root: ET.Element) -> dict:
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].replace("{", "")
        return {"kml": uri}
    return {"kml": "http://www.opengis.net/kml/2.2"}


def parse_coordinates_text(coord_text: str) -> list[tuple[float, float, float]]:
    """
    Parse KML coordinates text:
        lon,lat,elev lon,lat,elev ...
    Returns list of tuples: (lon, lat, elev)
    """
    coords = []

    if coord_text is None:
        return coords

    for coord in coord_text.strip().split():
        values = coord.split(",")

        if len(values) < 2:
            continue

        lon = safe_float(values[0], default=None)
        lat = safe_float(values[1], default=None)

        if lon is None or lat is None:
            continue

        elev = 0.0
        if len(values) >= 3:
            elev = safe_float(values[2], default=0.0)

        coords.append((lon, lat, elev))

    return coords


def parse_kml_geometries(kml_file: Path) -> list[dict]:
    """
    Extract Point, LineString, and Polygon outer ring vertices from KML.
    Returns rows with:
        name, lon, lat, elevation
    """
    tree = ET.parse(kml_file)
    root = tree.getroot()
    ns = get_namespace(root)

    rows = []

    placemarks = root.findall(".//kml:Placemark", ns)

    for pm in placemarks:
        base_name_el = pm.find("kml:name", ns)
        base_name = clean_name(base_name_el.text if base_name_el is not None else None)

        # --------------------------------------------------------
        # 1. Point
        # --------------------------------------------------------
        if EXPORT_POINTS:
            point_els = pm.findall(".//kml:Point", ns)

            for point_el in point_els:
                coord_el = point_el.find("kml:coordinates", ns)
                if coord_el is None or coord_el.text is None:
                    continue

                coords = parse_coordinates_text(coord_el.text)

                for i, (lon, lat, elev) in enumerate(coords, start=1):
                    name = base_name if len(coords) == 1 else f"{base_name}_point_{i:03d}"
                    rows.append(
                        {
                            "name": name,
                            "lon": lon,
                            "lat": lat,
                            "elevation": elev,
                        }
                    )

        # --------------------------------------------------------
        # 2. LineString
        # --------------------------------------------------------
        if EXPORT_LINESTRINGS:
            line_els = pm.findall(".//kml:LineString", ns)

            for line_idx, line_el in enumerate(line_els, start=1):
                coord_el = line_el.find("kml:coordinates", ns)
                if coord_el is None or coord_el.text is None:
                    continue

                coords = parse_coordinates_text(coord_el.text)

                for i, (lon, lat, elev) in enumerate(coords, start=1):
                    name = f"{base_name}_line{line_idx:02d}_{i:03d}"
                    rows.append(
                        {
                            "name": name,
                            "lon": lon,
                            "lat": lat,
                            "elevation": elev,
                        }
                    )

        # --------------------------------------------------------
        # 3. Polygon outer boundary / ring
        # --------------------------------------------------------
        if EXPORT_POLYGON_RINGS:
            ring_els = pm.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", ns)

            for ring_idx, ring_el in enumerate(ring_els, start=1):
                coord_el = ring_el.find("kml:coordinates", ns)
                if coord_el is None or coord_el.text is None:
                    continue

                coords = parse_coordinates_text(coord_el.text)

                if not KEEP_CLOSED_RING_LAST_POINT and len(coords) >= 2:
                    if coords[0] == coords[-1]:
                        coords = coords[:-1]

                for i, (lon, lat, elev) in enumerate(coords, start=1):
                    name = f"ring{ring_idx:02d}_{i:03d}"

                    rows.append(
                        {
                            "name": name,
                            "lon": lon,
                            "lat": lat,
                            "elevation": elev,
                        }
                    )

    return rows


def convert_one_file(input_file: Path, out_dir: Path) -> Path:
    input_file = Path(input_file)

    if input_file.suffix.lower() == ".kmz":
        kml_file = extract_kml_from_kmz(input_file)
    elif input_file.suffix.lower() == ".kml":
        kml_file = input_file
    else:
        raise ValueError(f"Unsupported file type: {input_file}")

    rows = parse_kml_geometries(kml_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_xyz = out_dir / f"{input_file.stem}.xyz"

    with open(out_xyz, "w", encoding="utf-8") as f:
        if WRITE_HEADER:
            f.write("# name lon lat elevation\n")

        for r in rows:
            f.write(
                f"{r['name']} "
                f"{r['lon']:.8f} "
                f"{r['lat']:.8f} "
                f"{r['elevation']:.3f}\n"
            )

    print(f"[OK] {input_file} -> {out_xyz} | exported rows: {len(rows)}")
    return out_xyz


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    files = sorted(list(INPUT_DIR.glob("*.kml")) + list(INPUT_DIR.glob("*.kmz")))

    if len(files) == 0:
        raise FileNotFoundError(f"No .kml or .kmz files found in: {INPUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== EXPORT KML/KMZ PLAN GEOMETRIES TO XYZ ==========")
    print(f"[INFO] Input folder:  {INPUT_DIR}")
    print(f"[INFO] Output folder: {OUT_DIR}")
    print(f"[INFO] Number of files: {len(files)}")

    total_rows = 0

    for f in files:
        try:
            out_xyz = convert_one_file(f, OUT_DIR)

            with open(out_xyz, "r", encoding="utf-8") as fp:
                n = sum(1 for line in fp if line.strip() and not line.startswith("#"))

            total_rows += n

        except Exception as e:
            print(f"[WARN] Failed: {f}")
            print(f"       Reason: {e}")

    print("\n========== DONE ==========")
    print(f"[INFO] Total exported rows: {total_rows}")
    print(f"[INFO] XYZ files saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()