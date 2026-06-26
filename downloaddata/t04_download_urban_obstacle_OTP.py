#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download ONE OpenTopography DEM dataset for the Hoa Lac study area,
then check and plot all available/derived layers from that single raster.

IMPORTANT
---------
This script downloads only ONE OpenTopography data type, controlled by:

    OPENTOPOGRAPHY_DEM_TYPE = "COP30"

Typical options supported by OpenTopography Global DEM API include:
    COP30, COP90, AW3D30, NASADEM, SRTMGL1, SRTMGL3

For UAV AGL work, COP30 is a DSM-like surface product. If you need
bare-earth AGL, compare later with FABDEM or local LiDAR/DTM.

Outputs are saved to:
    output/01_HoaLac_studies_area/opentopography

Output structure:
    opentopography/
    ├── hoalac_polygon.gpkg
    ├── raw_bbox_tif/
    ├── clipped_tif/
    ├── derived_tif/
    ├── xyz/
    ├── figures/
    ├── api_errors/
    ├── opentopography_single_dataset_layer_summary.csv
    └── opentopography_single_dataset_layer_summary.txt

Derived layers plotted/checking:
    - all original raster bands
    - elevation band 1
    - slope degree
    - aspect degree
    - hillshade
    - TRI terrain ruggedness index
    - contour map
    - elevation histogram

XYZ format:
    lon lat value
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter


# ======================================================================
# USER INPUT PARAMETERS
# ======================================================================

# Paste your OpenTopography API key here.
# You can also leave it blank and set environment variable outside the script
# if you modify get_api_key().
OPENTOPOGRAPHY_API_KEY_IN_SCRIPT = "9b13849a6bd3486c4ed72960d230a366"

# Download ONLY this one DEM type.
# Recommended start: COP30
OPENTOPOGRAPHY_DEM_TYPE = "COP30"

# Output folder requested by user
OUTDIR = "output/01_HoaLac_studies_area/opentopography"

# Hoa Lac polygon, lon/lat
HOALAC_POLYGON = [
    (105.5035, 21.0145),
    (105.5125, 20.9935),
    (105.5310, 20.9815),
    (105.5565, 20.9845),
    (105.5735, 20.9985),
    (105.5705, 21.0190),
    (105.5480, 21.0285),
    (105.5205, 21.0270),
    (105.5035, 21.0145),
]

# Extra bbox padding around the polygon for API download.
# 0.002 degree ~ 200 m.
BBOX_PADDING_DEG = 0.002

# If True, re-download even when the raw file already exists.
FORCE_DOWNLOAD = False

# Terrain derivative settings
TRI_WINDOW_SIZE = 3
CONTOUR_LEVELS = 12

# Plot settings
DPI = 220
CMAP_ELEVATION = "terrain"
CMAP_DERIVED = "viridis"
CMAP_ASPECT = "twilight"
CMAP_HILLSHADE = "gray"


# ======================================================================
# GEOMETRY HELPERS
# ======================================================================

def make_hoalac_polygon_gdf() -> gpd.GeoDataFrame:
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_study_area"]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def get_bbox_from_polygon(poly_gdf: gpd.GeoDataFrame, padding_deg: float = 0.0):
    west, south, east, north = poly_gdf.total_bounds
    return (
        float(west - padding_deg),
        float(south - padding_deg),
        float(east + padding_deg),
        float(north + padding_deg),
    )


# ======================================================================
# OPENTOPOGRAPHY DOWNLOAD
# ======================================================================

def get_api_key() -> str:
    key = OPENTOPOGRAPHY_API_KEY_IN_SCRIPT.strip()
    if not key or key == "PASTE_YOUR_OPENTOPOGRAPHY_API_KEY_HERE":
        raise RuntimeError(
            "OpenTopography API key is missing.\n"
            "Paste it near the top of this script:\n"
            "    OPENTOPOGRAPHY_API_KEY_IN_SCRIPT = \"your_key_here\""
        )
    return key


def build_opentopography_url(dem_type: str, west: float, south: float, east: float, north: float, api_key: str) -> str:
    # OpenTopography Global DEM API endpoint.
    # Parameters use lon/lat bbox and GeoTIFF output.
    base_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": f"{south:.8f}",
        "north": f"{north:.8f}",
        "west": f"{west:.8f}",
        "east": f"{east:.8f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    req = requests.Request("GET", base_url, params=params).prepare()
    return req.url


def download_one_opentopography_dem(
    dem_type: str,
    west: float,
    south: float,
    east: float,
    north: float,
    out_tif: Path,
    error_dir: Path,
) -> Path:
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    if out_tif.exists() and out_tif.stat().st_size > 0 and not FORCE_DOWNLOAD:
        print(f"[SKIP] Existing raw DEM: {out_tif}")
        return out_tif

    api_key = get_api_key()
    url = build_opentopography_url(
        dem_type=dem_type,
        west=west,
        south=south,
        east=east,
        north=north,
        api_key=api_key,
    )

    safe_url = url.replace(api_key, "***API_KEY_HIDDEN***")
    print("\n[INFO] Downloading one OpenTopography DEM dataset")
    print(f"[INFO] DEM type: {dem_type}")
    print(f"[INFO] URL: {safe_url}")

    try:
        r = requests.get(url, timeout=300)
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenTopography request failed: {exc}") from exc

    content_type = r.headers.get("Content-Type", "")
    if r.status_code != 200:
        err_file = error_dir / f"{dem_type}_api_error_status_{r.status_code}.txt"
        err_file.write_text(r.text, encoding="utf-8", errors="ignore")
        raise RuntimeError(
            f"OpenTopography API returned status {r.status_code}.\n"
            f"Error text saved to: {err_file}"
        )

    # Sometimes API errors are returned as text/html or JSON, not GeoTIFF.
    if ("tif" not in content_type.lower()) and ("image" not in content_type.lower()) and ("octet-stream" not in content_type.lower()):
        head = r.content[:200].decode("utf-8", errors="ignore")
        if "error" in head.lower() or "invalid" in head.lower() or "api" in head.lower():
            err_file = error_dir / f"{dem_type}_api_error_content.txt"
            err_file.write_bytes(r.content)
            raise RuntimeError(
                f"OpenTopography returned non-raster content: {content_type}.\n"
                f"Error content saved to: {err_file}"
            )

    out_tif.write_bytes(r.content)

    if out_tif.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_tif}")

    # Validate readable raster
    try:
        with rasterio.open(out_tif) as src:
            _ = src.count
            _ = src.width
            _ = src.height
    except Exception as exc:
        err_file = error_dir / f"{dem_type}_unreadable_download.bin"
        err_file.write_bytes(out_tif.read_bytes())
        out_tif.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a readable raster. Saved copy: {err_file}"
        ) from exc

    print(f"[OK] Saved raw DEM: {out_tif}")
    return out_tif


# ======================================================================
# RASTER PROCESSING
# ======================================================================

def clip_raster_by_polygon(in_tif: Path, poly_gdf: gpd.GeoDataFrame, out_tif: Path) -> Path:
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_tif) as src:
        poly_for_raster = poly_gdf.to_crs(src.crs)
        geoms = [geom for geom in poly_for_raster.geometry]

        nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = -9999.0

        clipped, clipped_transform = mask(
            src,
            geoms,
            crop=True,
            nodata=nodata_value,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            nodata=nodata_value,
            compress="lzw",
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(clipped)

    print(f"[OK] Saved clipped DEM: {out_tif}")
    return out_tif


def read_band_as_float(src: rasterio.io.DatasetReader, band: int = 1) -> np.ndarray:
    arr = src.read(band).astype("float64")
    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def get_pixel_size_m(src: rasterio.io.DatasetReader):
    transform = src.transform
    if src.crs and src.crs.is_geographic:
        center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
        dy_m = abs(transform.e) * 111_320.0
        dx_m = abs(transform.a) * 111_320.0 * np.cos(np.deg2rad(center_lat))
    else:
        dx_m = abs(transform.a)
        dy_m = abs(transform.e)
    return float(dx_m), float(dy_m)


def calculate_derivative_layers(clipped_tif: Path, derived_dir: Path, dem_type: str) -> dict[str, Path]:
    derived_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "elevation": derived_dir / f"{dem_type}_elevation_m.tif",
        "slope_degree": derived_dir / f"{dem_type}_slope_degree.tif",
        "aspect_degree": derived_dir / f"{dem_type}_aspect_degree.tif",
        "hillshade": derived_dir / f"{dem_type}_hillshade.tif",
        "tri_m": derived_dir / f"{dem_type}_tri_m.tif",
    }

    with rasterio.open(clipped_tif) as src:
        dem = read_band_as_float(src, 1)
        profile = src.profile.copy()
        dx_m, dy_m = get_pixel_size_m(src)

        dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degree = np.rad2deg(slope_rad)

        # Aspect: 0 north, 90 east, 180 south, 270 west
        aspect_rad = np.arctan2(-dz_dx, dz_dy)
        aspect_degree = np.degrees(aspect_rad)
        aspect_degree = np.where(aspect_degree < 0, 360.0 + aspect_degree, aspect_degree)

        # Simple analytical hillshade
        azimuth = np.deg2rad(315.0)
        altitude = np.deg2rad(45.0)
        zenith = np.pi / 2.0 - altitude
        aspect_math = np.deg2rad(aspect_degree)
        hillshade = 255.0 * (
            np.cos(zenith) * np.cos(slope_rad)
            + np.sin(zenith) * np.sin(slope_rad) * np.cos(azimuth - aspect_math)
        )
        hillshade = np.clip(hillshade, 0, 255)

        def tri_func(window):
            center = window[len(window) // 2]
            if not np.isfinite(center):
                return np.nan
            return np.sqrt(np.nanmean((window - center) ** 2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tri = generic_filter(
                dem,
                tri_func,
                size=TRI_WINDOW_SIZE,
                mode="nearest",
            )

        layers = {
            "elevation": dem,
            "slope_degree": slope_degree,
            "aspect_degree": aspect_degree,
            "hillshade": hillshade,
            "tri_m": tri,
        }

        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")

        for layer_name, arr in layers.items():
            arr_write = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
            with rasterio.open(out_paths[layer_name], "w", **profile) as dst:
                dst.write(arr_write, 1)
            print(f"[OK] Saved derived raster: {out_paths[layer_name]}")

    return out_paths


def raster_to_xyz(in_tif: Path, out_xyz: Path, band: int = 1) -> None:
    out_xyz.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_tif) as src:
        arr = read_band_as_float(src, band)
        rows, cols = np.where(np.isfinite(arr))

        if len(rows) == 0:
            out_xyz.write_text("")
            print(f"[WARN] Empty XYZ saved: {out_xyz}")
            return

        xs, ys = xy(src.transform, rows, cols, offset="center")
        vals = arr[rows, cols]

        points = gpd.GeoDataFrame(
            {"value": vals},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": points.geometry.x.to_numpy(),
        "lat": points.geometry.y.to_numpy(),
        "value": points["value"].to_numpy(),
    })

    df.to_csv(out_xyz, sep=" ", index=False, header=False, float_format="%.8f")
    print(f"[OK] Saved XYZ: {out_xyz}")


# ======================================================================
# CHECK LAYER EXISTENCE AND STATS
# ======================================================================

def raster_stats(in_tif: Path, layer_name: str, band: int = 1, source_type: str = "raster") -> dict:
    if not in_tif.exists() or in_tif.stat().st_size == 0:
        return {
            "layer": layer_name,
            "source_type": source_type,
            "exists": False,
            "status": "missing",
            "band": band,
            "valid_pixels": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "note": "File missing or empty",
        }

    try:
        with rasterio.open(in_tif) as src:
            if band > src.count:
                return {
                    "layer": layer_name,
                    "source_type": source_type,
                    "exists": False,
                    "status": "band_missing",
                    "band": band,
                    "valid_pixels": 0,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "note": f"Raster has only {src.count} band(s)",
                }
            arr = read_band_as_float(src, band)
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                status = "empty"
            else:
                status = "usable"
            return {
                "layer": layer_name,
                "source_type": source_type,
                "exists": True,
                "status": status,
                "band": band,
                "valid_pixels": int(vals.size),
                "min": float(np.nanmin(vals)) if vals.size else np.nan,
                "max": float(np.nanmax(vals)) if vals.size else np.nan,
                "mean": float(np.nanmean(vals)) if vals.size else np.nan,
                "std": float(np.nanstd(vals)) if vals.size else np.nan,
                "note": "OK" if vals.size else "No valid raster pixels",
            }
    except Exception as exc:
        return {
            "layer": layer_name,
            "source_type": source_type,
            "exists": False,
            "status": "error",
            "band": band,
            "valid_pixels": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "note": str(exc),
        }


def build_layer_summary(clipped_tif: Path, derived_paths: dict[str, Path], out_csv: Path, out_txt: Path) -> pd.DataFrame:
    records = []

    with rasterio.open(clipped_tif) as src:
        band_count = src.count

    # Original bands inside the downloaded dataset
    for band in range(1, band_count + 1):
        records.append(
            raster_stats(
                clipped_tif,
                layer_name=f"original_band_{band}",
                band=band,
                source_type="original_downloaded_dataset",
            )
        )

    # Derived layers from the single downloaded raster
    for layer_name, path in derived_paths.items():
        records.append(
            raster_stats(
                path,
                layer_name=layer_name,
                band=1,
                source_type="derived_from_single_dataset",
            )
        )

    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("OpenTopography single-dataset layer availability summary")
    lines.append("=" * 70)
    for _, row in df.iterrows():
        lines.append(
            f"{row['layer']:24s} | {row['source_type']:32s} | "
            f"{row['status']:10s} | valid={int(row['valid_pixels']):8d} | "
            f"min={row['min']:.3f} max={row['max']:.3f} mean={row['mean']:.3f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n========== LAYER SUMMARY ==========")
    print(df[["layer", "source_type", "status", "valid_pixels", "min", "max", "mean"]].to_string(index=False))
    print(f"[OK] Saved summary CSV: {out_csv}")
    print(f"[OK] Saved summary TXT: {out_txt}")
    return df


# ======================================================================
# PLOTTING
# ======================================================================

def get_raster_extent(src: rasterio.io.DatasetReader):
    b = src.bounds
    return [b.left, b.right, b.bottom, b.top]


def plot_raster_map(
    in_tif: Path,
    out_png: Path,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    band: int = 1,
    poly_gdf: gpd.GeoDataFrame | None = None,
    contour: bool = False,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not in_tif.exists() or in_tif.stat().st_size == 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, f"Missing layer\n{title}", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return

    with rasterio.open(in_tif) as src:
        arr = read_band_as_float(src, band)
        extent = get_raster_extent(src)
        crs = src.crs

    fig, ax = plt.subplots(figsize=(8, 7))

    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
        vmin = float(np.nanpercentile(valid, 2))
        vmax = float(np.nanpercentile(valid, 98))
        if math.isclose(vmin, vmax):
            vmin = float(np.nanmin(valid))
            vmax = float(np.nanmax(valid))
    else:
        vmin, vmax = None, None

    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if contour and valid.size > 0:
        try:
            ax.contour(
                arr,
                levels=CONTOUR_LEVELS,
                extent=extent,
                origin="upper",
                linewidths=0.5,
                colors="black",
                alpha=0.55,
            )
        except Exception:
            pass

    if poly_gdf is not None:
        try:
            poly_plot = poly_gdf.to_crs(crs)
            poly_plot.boundary.plot(ax=ax, linewidth=1.2, color="black")
        except Exception:
            pass

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude / projected X")
    ax.set_ylabel("Latitude / projected Y")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_histogram(in_tif: Path, out_png: Path, title: str, x_label: str, band: int = 1) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_tif) as src:
        arr = read_band_as_float(src, band)

    vals = arr[np.isfinite(arr)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if vals.size > 0:
        ax.hist(vals, bins=40)
        ax.axvline(float(np.nanmean(vals)), linestyle="--", linewidth=1.2, label="mean")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid values", ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pixel count")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_availability_counts(summary_df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.copy()
    plot_df["valid_pixels"] = pd.to_numeric(plot_df["valid_pixels"], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["valid_pixels"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["layer"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Valid pixel count")
    ax.set_title("Layer availability from one OpenTopography dataset")
    for i, v in enumerate(plot_df["valid_pixels"].to_numpy()):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def plot_all_layers_gallery(derived_paths: dict[str, Path], clipped_tif: Path, out_png: Path, poly_gdf: gpd.GeoDataFrame) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    layer_specs = [
        ("Elevation", clipped_tif, CMAP_ELEVATION, 1),
        ("Slope degree", derived_paths["slope_degree"], CMAP_DERIVED, 1),
        ("Aspect degree", derived_paths["aspect_degree"], CMAP_ASPECT, 1),
        ("Hillshade", derived_paths["hillshade"], CMAP_HILLSHADE, 1),
        ("TRI", derived_paths["tri_m"], CMAP_DERIVED, 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for ax, (title, path, cmap, band) in zip(axes, layer_specs):
        with rasterio.open(path) as src:
            arr = read_band_as_float(src, band)
            extent = get_raster_extent(src)
            crs = src.crs
        valid = arr[np.isfinite(arr)]
        if valid.size:
            vmin = float(np.nanpercentile(valid, 2))
            vmax = float(np.nanpercentile(valid, 98))
        else:
            vmin, vmax = None, None
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        try:
            poly_gdf.to_crs(crs).boundary.plot(ax=ax, linewidth=1.0, color="black")
        except Exception:
            pass
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, shrink=0.70, pad=0.02)

    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.90,
        "Single OpenTopography dataset\nDerived layers for checking\nDEM type: " + OPENTOPOGRAPHY_DEM_TYPE,
        transform=axes[-1].transAxes,
        va="top",
        fontsize=11,
    )

    fig.suptitle("OpenTopography single dataset layer check", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")


def make_all_figures(
    clipped_tif: Path,
    derived_paths: dict[str, Path],
    poly_gdf: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
    dem_type: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Original bands inside the data
    with rasterio.open(clipped_tif) as src:
        band_count = src.count

    for band in range(1, band_count + 1):
        plot_raster_map(
            clipped_tif,
            figures_dir / f"01_original_band_{band}_elevation_m.png",
            title=f"{dem_type} original band {band}: elevation",
            cbar_label="Elevation (m)",
            cmap=CMAP_ELEVATION,
            band=band,
            poly_gdf=poly_gdf,
        )

    plot_raster_map(
        derived_paths["elevation"],
        figures_dir / "02_elevation_m.png",
        title=f"{dem_type} elevation clipped to Hoa Lac",
        cbar_label="Elevation (m)",
        cmap=CMAP_ELEVATION,
        poly_gdf=poly_gdf,
    )

    plot_raster_map(
        derived_paths["slope_degree"],
        figures_dir / "03_slope_degree.png",
        title=f"{dem_type} slope",
        cbar_label="Slope (degree)",
        cmap=CMAP_DERIVED,
        poly_gdf=poly_gdf,
    )

    plot_raster_map(
        derived_paths["aspect_degree"],
        figures_dir / "04_aspect_degree.png",
        title=f"{dem_type} aspect",
        cbar_label="Aspect (degree)",
        cmap=CMAP_ASPECT,
        poly_gdf=poly_gdf,
    )

    plot_raster_map(
        derived_paths["hillshade"],
        figures_dir / "05_hillshade.png",
        title=f"{dem_type} hillshade",
        cbar_label="Hillshade (0-255)",
        cmap=CMAP_HILLSHADE,
        poly_gdf=poly_gdf,
    )

    plot_raster_map(
        derived_paths["tri_m"],
        figures_dir / "06_tri_m.png",
        title=f"{dem_type} terrain ruggedness index",
        cbar_label="TRI (m)",
        cmap=CMAP_DERIVED,
        poly_gdf=poly_gdf,
    )

    plot_raster_map(
        derived_paths["elevation"],
        figures_dir / "07_elevation_contours.png",
        title=f"{dem_type} elevation contours",
        cbar_label="Elevation (m)",
        cmap=CMAP_ELEVATION,
        poly_gdf=poly_gdf,
        contour=True,
    )

    plot_histogram(
        derived_paths["elevation"],
        figures_dir / "08_elevation_histogram.png",
        title=f"{dem_type} elevation distribution",
        x_label="Elevation (m)",
    )

    plot_availability_counts(
        summary_df,
        figures_dir / "00a_layer_availability_counts.png",
    )

    plot_all_layers_gallery(
        derived_paths,
        clipped_tif,
        figures_dir / "00b_all_layers_gallery.png",
        poly_gdf,
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    outdir = Path(OUTDIR)
    raw_dir = outdir / "raw_bbox_tif"
    clipped_dir = outdir / "clipped_tif"
    derived_dir = outdir / "derived_tif"
    xyz_dir = outdir / "xyz"
    figures_dir = outdir / "figures"
    error_dir = outdir / "api_errors"

    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== OPEN TOPOGRAPHY SINGLE DATASET CHECK ==========")
    print(f"[INFO] DEM type selected: {OPENTOPOGRAPHY_DEM_TYPE}")
    print(f"[INFO] Output directory: {outdir.resolve()}")

    # 1. Study polygon
    poly_gdf = make_hoalac_polygon_gdf()
    poly_file = outdir / "hoalac_polygon.gpkg"
    poly_gdf.to_file(poly_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {poly_file}")

    # 2. Bbox
    west, south, east, north = get_bbox_from_polygon(poly_gdf, BBOX_PADDING_DEG)
    print("\n[INFO] Download bbox:")
    print(f"  WEST  = {west:.8f}")
    print(f"  SOUTH = {south:.8f}")
    print(f"  EAST  = {east:.8f}")
    print(f"  NORTH = {north:.8f}")

    # 3. Download only one raster product
    dem_type = OPENTOPOGRAPHY_DEM_TYPE.strip().upper()
    raw_tif = raw_dir / f"{dem_type}_bbox.tif"
    download_one_opentopography_dem(
        dem_type=dem_type,
        west=west,
        south=south,
        east=east,
        north=north,
        out_tif=raw_tif,
        error_dir=error_dir,
    )

    # 4. Clip to polygon
    clipped_tif = clipped_dir / f"{dem_type}_hoalac_clipped.tif"
    clip_raster_by_polygon(raw_tif, poly_gdf, clipped_tif)

    # 5. Derived layers from this one raster
    derived_paths = calculate_derivative_layers(clipped_tif, derived_dir, dem_type)

    # 6. XYZ exports
    # Original band(s)
    with rasterio.open(clipped_tif) as src:
        band_count = src.count

    for band in range(1, band_count + 1):
        raster_to_xyz(
            clipped_tif,
            xyz_dir / f"{dem_type}_original_band_{band}_hoalac.xyz",
            band=band,
        )

    for layer_name, path in derived_paths.items():
        raster_to_xyz(
            path,
            xyz_dir / f"{dem_type}_{layer_name}_hoalac.xyz",
        )

    # 7. Layer availability/stats summary
    summary_df = build_layer_summary(
        clipped_tif=clipped_tif,
        derived_paths=derived_paths,
        out_csv=outdir / "opentopography_single_dataset_layer_summary.csv",
        out_txt=outdir / "opentopography_single_dataset_layer_summary.txt",
    )

    # 8. Figures
    make_all_figures(
        clipped_tif=clipped_tif,
        derived_paths=derived_paths,
        poly_gdf=poly_gdf,
        summary_df=summary_df,
        figures_dir=figures_dir,
        dem_type=dem_type,
    )

    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")
    print("\nImportant files:")
    print(f"  Raw DEM:       {raw_tif}")
    print(f"  Clipped DEM:   {clipped_tif}")
    print(f"  Derived rasters: {derived_dir}")
    print(f"  XYZ files:     {xyz_dir}")
    print(f"  Figures:       {figures_dir}")
    print(f"  Summary CSV:   {outdir / 'opentopography_single_dataset_layer_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[FAILED]", exc)
        sys.exit(1)
