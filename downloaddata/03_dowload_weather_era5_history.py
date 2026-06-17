#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download latest available ERA5 hourly single-level weather data
for the Hoa Lac study area.

Run:
    python 03_download_weather_era5_latest.py

Before first use:
    1. Login to Copernicus Climate Data Store.
    2. Get your Personal Access Token.
    3. Paste it into CDS_API_KEY below.
    4. Accept the ERA5 dataset terms online once.

Output:
    output/02_senario1_no_velocity/03_weather/era5/
        era5_hoalac_latest.nc
        era5_hoalac_YYYYMMDD_2200_to_YYYYMMDD_2300_downloaded_YYYYMMDD_HHMMSSUTC.nc

Important:
    ERA5 is reanalysis, not forecast.
    ERA5/ERA5T is usually delayed by several days.
    For operational next-2-hour planning, use Open-Meteo or forecast model data.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import subprocess
import sys
import shutil


# ============================================================
# CDS API LOGIN - PUT EVERYTHING HERE
# ============================================================

CDS_API_URL = "https://cds.climate.copernicus.eu/api"

# Paste your Copernicus CDS Personal Access Token here.
# Example:
# CDS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxx"
CDS_API_KEY = "1fd55e0d-3151-432b-bf58-7bd91e3e618a"

# If cdsapi is missing, the script will try to install it automatically.
AUTO_INSTALL_CDSAPI_IF_MISSING = True


# ============================================================
# STUDY AREA
# ============================================================

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

OUT_DIR = Path("output/01_HoaLac_studies_area/era5")

DATASET = "reanalysis-era5-single-levels"

# ERA5 grid is coarse, so use padding around the Hoa Lac polygon.
BBOX_PADDING_DEG = 0.20

# Latest available ERA5 can vary.
# The script tries 5 days behind today first, then older dates.
TRY_LAG_DAYS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# User request: latest data, equivalent to latest available 2 hourly records.
# To avoid cross-day request problems, this script downloads 22:00 and 23:00 UTC
# from the latest available ERA5 date.
ERA5_HOURS_UTC = ["22:00", "23:00"]

# ERA5 variables useful for weather-risk / UTM validation.
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_precipitation",
    "visibility",
    "convective_available_potential_energy",
    "boundary_layer_height",
]


# ============================================================
# PACKAGE SETUP
# ============================================================

def import_or_install_cdsapi():
    """
    Import cdsapi. If missing, optionally install it automatically.
    """
    try:
        import cdsapi
        return cdsapi
    except ImportError:
        if not AUTO_INSTALL_CDSAPI_IF_MISSING:
            raise RuntimeError(
                "cdsapi is not installed. Install with:\n"
                "    pip install 'cdsapi>=0.7.7'"
            )

        print("[INFO] cdsapi not found. Try automatic install...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "cdsapi>=0.7.7",
        ])

        import cdsapi
        return cdsapi


# ============================================================
# FUNCTIONS
# ============================================================

def check_token():
    """
    Make sure user pasted the CDS API token.
    """
    if (
        not CDS_API_KEY
        or CDS_API_KEY.strip() == ""
        or CDS_API_KEY.strip() == "PASTE_YOUR_PERSONAL_ACCESS_TOKEN_HERE"
    ):
        raise RuntimeError(
            "\nCDS_API_KEY is not set.\n\n"
            "Open this script and paste your Copernicus Personal Access Token here:\n\n"
            '    CDS_API_KEY = "YOUR_PERSONAL_ACCESS_TOKEN"\n\n'
            "Then run again:\n\n"
            "    python 03_download_weather_era5_latest.py\n"
        )


def get_bbox_from_polygon(
    polygon_lonlat: list[tuple[float, float]],
    padding_deg: float,
) -> list[float]:
    """
    Return CDS area format:
        [north, west, south, east]
    """
    lons = [p[0] for p in polygon_lonlat]
    lats = [p[1] for p in polygon_lonlat]

    west = min(lons) - padding_deg
    east = max(lons) + padding_deg
    south = min(lats) - padding_deg
    north = max(lats) + padding_deg

    return [
        round(north, 4),
        round(west, 4),
        round(south, 4),
        round(east, 4),
    ]


def build_candidate_date(lag_days: int) -> datetime:
    """
    Build candidate ERA5 date using UTC today minus lag days.
    """
    now_utc = datetime.now(timezone.utc)
    candidate = now_utc - timedelta(days=lag_days)
    return candidate.replace(hour=0, minute=0, second=0, microsecond=0)


def build_request(
    candidate_date: datetime,
    area: list[float],
) -> dict:
    """
    Build CDS ERA5 request.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": ERA5_VARIABLES,
        "year": [f"{candidate_date.year:04d}"],
        "month": [f"{candidate_date.month:02d}"],
        "day": [f"{candidate_date.day:02d}"],
        "time": ERA5_HOURS_UTC,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    return request


def download_era5(
    client,
    request: dict,
    target_file: Path,
):
    """
    Download ERA5 file.

    Uses the newer style:
        client.retrieve(...).download(target)
    """
    result = client.retrieve(DATASET, request)
    result.download(str(target_file))


def main():
    check_token()

    cdsapi = import_or_install_cdsapi()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== DOWNLOAD ERA5 LATEST AVAILABLE ==========")
    print("[INFO] ERA5 is reanalysis, not forecast.")
    print("[INFO] This script downloads the latest available 2 hourly ERA5 records.")
    print("[INFO] No ~/.cdsapirc is required because the token is inside this script.")

    area = get_bbox_from_polygon(
        polygon_lonlat=HOALAC_POLYGON,
        padding_deg=BBOX_PADDING_DEG,
    )

    print(f"[INFO] Area [N, W, S, E]: {area}")
    print(f"[INFO] ERA5 hours UTC:     {ERA5_HOURS_UTC}")
    print(f"[INFO] Output dir:         {OUT_DIR}")

    client = cdsapi.Client(
        url=CDS_API_URL,
        key=CDS_API_KEY,
    )

    download_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")

    success = False
    final_file = None
    final_date = None
    final_lag = None

    for lag_days in TRY_LAG_DAYS:
        candidate_date = build_candidate_date(lag_days)

        ymd = candidate_date.strftime("%Y%m%d")
        start_hour = ERA5_HOURS_UTC[0].replace(":", "")
        end_hour = ERA5_HOURS_UTC[-1].replace(":", "")

        target_file = OUT_DIR / (
            f"era5_hoalac_{ymd}_{start_hour}_to_{ymd}_{end_hour}"
            f"_downloaded_{download_timestamp}.nc"
        )

        request = build_request(
            candidate_date=candidate_date,
            area=area,
        )

        print("\n----------------------------------------")
        print(f"[INFO] Try lag_days = {lag_days}")
        print(f"[INFO] Candidate date UTC: {candidate_date.strftime('%Y-%m-%d')}")
        print(f"[INFO] Target file: {target_file}")

        try:
            download_era5(
                client=client,
                request=request,
                target_file=target_file,
            )

            if not target_file.exists() or target_file.stat().st_size == 0:
                raise RuntimeError("Downloaded file does not exist or is empty.")

            success = True
            final_file = target_file
            final_date = candidate_date
            final_lag = lag_days
            break

        except Exception as exc:
            print(f"[WARN] Failed for lag_days = {lag_days}")
            print(f"[WARN] Reason: {exc}")

            if target_file.exists():
                try:
                    target_file.unlink()
                except Exception:
                    pass

    if not success:
        raise RuntimeError(
            "\nCould not download ERA5 for any tested lag day.\n\n"
            "Check these points:\n"
            "  1. CDS_API_KEY is correct.\n"
            "  2. You accepted the ERA5 dataset terms online.\n"
            "  3. Internet connection is OK.\n"
            "  4. cdsapi version is new enough.\n"
            "  5. Try increasing TRY_LAG_DAYS.\n"
        )

    latest_file = OUT_DIR / "era5_hoalac_latest.nc"
    shutil.copy2(final_file, latest_file)

    print("\n========== ERA5 DOWNLOAD COMPLETE ==========")
    print(f"[OK] Lag days used:     {final_lag}")
    print(f"[OK] ERA5 date UTC:     {final_date.strftime('%Y-%m-%d')}")
    print(f"[OK] ERA5 hours UTC:    {ERA5_HOURS_UTC}")
    print(f"[OK] Saved archive:     {final_file}")
    print(f"[OK] Saved latest copy: {latest_file}")


if __name__ == "__main__":
    main()