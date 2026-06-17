#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create KML polygon circle for a no-fly zone.

Input:
    centroid latitude / longitude
    radius in km

Output:
    no_fly_centroid_40km.kml

Open the output KML in Google Earth Pro.
"""

from pathlib import Path
import math


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# User centroid
# You gave:
#   21.032427° 105.496035°
# This means:
#   latitude  = 21.032427
#   longitude = 105.496035
CENTER_LAT = 21.032427
CENTER_LON = 105.496035

# Radius of no-fly zone
RADIUS_KM = 4.0

# Number of points around circle
# 360 means 1 point per degree; smooth enough for Google Earth.
N_POINTS = 360

# Output KML file
OUT_KML = Path("kml/no_fly_centroid_4km.kml")


# ============================================================
# GEODESIC CIRCLE FUNCTIONS
# ============================================================

def destination_point(lat_deg, lon_deg, bearing_deg, distance_km, earth_radius_km=6371.0088):
    """
    Calculate destination lon/lat from start point, bearing, and distance.

    Uses spherical Earth approximation, good enough for 40 km radius.

    Returns:
        lon_deg, lat_deg
    """
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular_distance = distance_km / earth_radius_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    # Normalize longitude to [-180, 180]
    lon2 = (lon2 + 540.0) % 360.0 - 180.0

    return lon2, lat2


def make_circle_points(center_lat, center_lon, radius_km, n_points=360):
    """
    Make closed polygon points around centroid.

    Output format:
        [(lon, lat), ...]
    """
    points = []

    for i in range(n_points):
        bearing = i * 360.0 / n_points
        lon, lat = destination_point(
            lat_deg=center_lat,
            lon_deg=center_lon,
            bearing_deg=bearing,
            distance_km=radius_km,
        )
        points.append((lon, lat))

    # Close polygon
    points.append(points[0])

    return points


def make_kml(center_lat, center_lon, radius_km, points):
    """
    Build KML text.
    """
    coord_text = "\n".join(
        f"              {lon:.8f},{lat:.8f},0"
        for lon, lat in points
    )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>No-fly centroid {radius_km:.0f} km</name>

    <Style id="no_fly_circle_style">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
      <PolyStyle>
        <color>330000ff</color>
      </PolyStyle>
    </Style>

    <Style id="centroid_style">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>
        </Icon>
      </IconStyle>
      <LabelStyle>
        <scale>1.0</scale>
      </LabelStyle>
    </Style>

    <Placemark>
      <name>No-fly centroid</name>
      <description>
        Center latitude: {center_lat:.6f}
        Center longitude: {center_lon:.6f}
        Radius: {radius_km:.1f} km
      </description>
      <styleUrl>#centroid_style</styleUrl>
      <Point>
        <coordinates>{center_lon:.8f},{center_lat:.8f},0</coordinates>
      </Point>
    </Placemark>

    <Placemark>
      <name>No-fly zone radius {radius_km:.0f} km</name>
      <description>
        Geodesic circle generated from centroid.
        Coordinate order in KML is longitude,latitude,altitude.
      </description>
      <styleUrl>#no_fly_circle_style</styleUrl>
      <Polygon>
        <tessellate>1</tessellate>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
{coord_text}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>

  </Document>
</kml>
"""
    return kml


# ============================================================
# MAIN
# ============================================================

def main():
    points = make_circle_points(
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        radius_km=RADIUS_KM,
        n_points=N_POINTS,
    )

    kml_text = make_kml(
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        radius_km=RADIUS_KM,
        points=points,
    )

    OUT_KML.write_text(kml_text, encoding="utf-8")

    print("[OK] KML saved:")
    print(OUT_KML.resolve())

    print("\nCenter:")
    print(f"  Latitude:  {CENTER_LAT}")
    print(f"  Longitude: {CENTER_LON}")
    print(f"  Radius:    {RADIUS_KM} km")

    print("\nOpen in Google Earth Pro:")
    print(f"  File -> Open -> {OUT_KML}")


if __name__ == "__main__":
    main()