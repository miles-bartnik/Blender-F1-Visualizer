"""
streets.py
----------
OSM street/road fetching for the F1 circuit pipeline.
"""

import math
import requests

from config import OVERPASS_URL, OVERPASS_DELAY, TERRAIN_RESOLUTION_M

R_EARTH         = 6_371_000.0
DEFAULT_WIDTH_M = 0.01
LANE_WIDTH_M    = 3.5   # assumed width per lane when only lanes tag present


def fetch_osm_streets(lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass API for all highway ways within the bounding box.

    Returns a list of street dicts, each with:
      osm_id      : OSM way ID
      highway_tag : highway type string ('motorway', 'residential', etc.)
      width_m     : resolved width in metres
      nodes       : list of [lon, lat] coordinate pairs
    """
    bbox  = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    query = f"""
[out:json][timeout:90];
way["highway"]({bbox});
out geom;
"""
    try:
        resp = requests.post(
            OVERPASS_URL, data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "F1CircuitsPipeline/1.0"},
            timeout=95)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [streets] OSM query failed: {e}")
        return []

    elements = data.get("elements", [])
    streets  = []

    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue

        tags        = el.get("tags", {})
        highway_tag = tags.get("highway", "")
        if not highway_tag:
            continue

        # Resolve width
        width_m = DEFAULT_WIDTH_M
        if "width" in tags:
            try:
                width_m = float(tags["width"].split()[0])
            except (ValueError, IndexError):
                pass
        elif "lanes" in tags:
            try:
                width_m = int(tags["lanes"]) * LANE_WIDTH_M
            except ValueError:
                pass

        nodes = [[g["lon"], g["lat"]] for g in geom]

        streets.append({
            "osm_id":      el["id"],
            "highway_tag": highway_tag,
            "width_m":     width_m,
            "nodes":       nodes,
        })

    # Clip nodes to bbox
    try:
        from shapely.geometry import LineString as ShapelyLine, box
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        clipped   = []
        for street in streets:
            nodes = street["nodes"]
            if len(nodes) < 2:
                continue
            try:
                line      = ShapelyLine([(n[0], n[1]) for n in nodes])
                intersect = line.intersection(bbox_poly)
                if intersect.is_empty:
                    continue
                if intersect.geom_type == 'LineString':
                    street["nodes"] = [[c[0], c[1]] for c in intersect.coords]
                    clipped.append(street)
                elif intersect.geom_type == 'MultiLineString':
                    import copy
                    for part in intersect.geoms:
                        s = copy.copy(street)
                        s["nodes"] = [[c[0], c[1]] for c in part.coords]
                        clipped.append(s)
            except Exception:
                clipped.append(street)
        streets = clipped
    except ImportError:
        pass

    print(f"  [streets] {len(streets)} ways fetched")
    return streets