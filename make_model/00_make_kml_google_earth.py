from pathlib import Path

# ============================================================
# USER INPUT
# ============================================================

OUT_KML = Path("kml/HoaLac_Hitech_park_polygon.kml")

# Format: lon, lat
POLYGON_POINTS = [
    (105.5035, 21.0145),
    (105.5125, 20.9935),
    (105.5310, 20.9815),
    (105.5565, 20.9845),
    (105.5735, 20.9985),
    (105.5705, 21.0190),
    (105.5480, 21.0285),
    (105.5205, 21.0270),
]

POLYGON_NAME = "Hoa Lac research interest region"


# ============================================================
# MAKE KML
# ============================================================

def close_polygon(points):
    """
    Make sure first and last point are the same.
    """
    if points[0] != points[-1]:
        points = points + [points[0]]
    return points


def make_kml(points, polygon_name):
    """
    Create KML text from lon/lat polygon points.
    """
    points = close_polygon(points)

    coord_text = "\n".join(
        [f"              {lon:.8f},{lat:.8f},0" for lon, lat in points]
    )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{polygon_name}</name>

    <Style id="research_polygon_style">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
      <PolyStyle>
        <color>330000ff</color>
      </PolyStyle>
    </Style>

    <Placemark>
      <name>{polygon_name}</name>
      <styleUrl>#research_polygon_style</styleUrl>
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


def main():
    kml_text = make_kml(POLYGON_POINTS, POLYGON_NAME)
    OUT_KML.write_text(kml_text, encoding="utf-8")
    print(f"[OK] Saved: {OUT_KML.resolve()}")


if __name__ == "__main__":
    main()