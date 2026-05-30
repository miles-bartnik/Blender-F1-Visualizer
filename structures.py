"""
structures.py
-------------
OSM structure (building) fetching for the F1 circuit pipeline.
"""

import json as _json
import requests
from config import OVERPASS_URL, RAW_DATA_DIR, USE_RAW_CACHE


def fetch_osm_structures(circuit_id, lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass API for structure (building) footprints within a bounding box.

    Returns a list of structure dicts, each with:
      footprint    : list of [lon, lat] pairs (closed polygon)
      height_m     : height in metres (from tag, levels*3, or default 10m)
      min_height_m : base height (for elevated sections)
      roof_shape   : roof shape string ('flat', 'gabled', etc.)
      osm_id       : OSM way ID
    """
    _cp = RAW_DATA_DIR / "osm" / f"{circuit_id}_structures.json"
    if USE_RAW_CACHE and _cp.exists():
        with open(_cp) as _f: return _json.load(_f)

    DEFAULT_HEIGHT_M = 10.0
    METRES_PER_LEVEL = 3.0

    query = f"""
[out:json][timeout:60];
(
  way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
    try:
        for attempt in range(3):
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent":   "F1CircuitsPipeline/1.0"},
                timeout=65)
            if resp.status_code in (429, 504):
                import time as _t; _t.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            print(f"  [structures] failed after 3 attempts")
            return []
    except Exception as e:
        print(f"skip(OSM structures failed: {e})")
        return []

    elements = data.get("elements", [])
    print(f"\n    OSM returned {len(elements)} elements", end=" ")

    structures     = []
    skipped_no_geom = 0
    skipped_short   = 0

    for el in elements:
        if "geometry" not in el:
            skipped_no_geom += 1
            continue

        tags = el.get("tags", {})
        geom = el["geometry"]
        if not isinstance(geom, list) or len(geom) < 3:
            skipped_short += 1
            continue

        footprint = [[g["lon"], g["lat"]] for g in geom]

        # Height
        height_m = DEFAULT_HEIGHT_M
        if "height" in tags:
            try:
                h   = tags["height"].replace("m", "").replace("ft", "").strip()
                val = float(h)
                if "ft" in tags["height"]:
                    val *= 0.3048
                height_m = val
            except ValueError:
                pass
        elif "building:levels" in tags:
            try:
                height_m = float(tags["building:levels"]) * METRES_PER_LEVEL
            except ValueError:
                pass

        # Min height
        min_height_m = 0.0
        if "building:min_height" in tags:
            try:
                min_height_m = float(
                    tags["building:min_height"].replace("m", "").strip())
            except ValueError:
                pass
        elif "min_level" in tags:
            try:
                min_height_m = float(tags["min_level"]) * METRES_PER_LEVEL
            except ValueError:
                pass

        roof_shape = tags.get("roof:shape", "flat")

        structures.append({
            "osm_id":       el.get("id"),
            "footprint":    footprint,
            "height_m":     round(height_m, 2),
            "min_height_m": round(min_height_m, 2),
            "roof_shape":   roof_shape,
        })

    print(f"(skipped: {skipped_no_geom} no-geom, {skipped_short} short)")

    # ── Clip all footprints to bbox ───────────────────────────────────────
    try:
        from shapely.geometry import Polygon as ShapelyPoly, box
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        clipped   = []
        for s in structures:
            fp = s.get("footprint", [])
            if len(fp) < 3:
                continue
            try:
                poly      = ShapelyPoly([(p[0], p[1]) for p in fp])
                intersect = poly.intersection(bbox_poly)
                if intersect.is_empty:
                    continue
                if intersect.geom_type == 'Polygon':
                    s["footprint"] = [[c[0], c[1]] for c in intersect.exterior.coords]
                    clipped.append(s)
                elif intersect.geom_type == 'MultiPolygon':
                    for part in intersect.geoms:
                        import copy
                        b = copy.copy(s)
                        b["footprint"] = [[c[0], c[1]] for c in part.exterior.coords]
                        clipped.append(b)
            except Exception:
                clipped.append(s)
        structures = clipped
    except ImportError:
        pass

    _cp.parent.mkdir(parents=True, exist_ok=True)
    with open(_cp, "w") as _f: _json.dump(structures, _f)
    return structures