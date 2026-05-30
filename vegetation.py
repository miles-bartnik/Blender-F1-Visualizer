"""
vegetation.py
-------------
OSM vegetation polygon fetching for the F1 circuit pipeline.
"""

import math

import requests

import json as _json
from config import OVERPASS_URL, OVERPASS_DELAY, TERRAIN_RESOLUTION_M, RAW_DATA_DIR, USE_RAW_CACHE

R_EARTH = 6_371_000.0

VEGETATION_TAGS = [
    '["natural"="wood"]',
    '["natural"="scrub"]',
    '["natural"="heath"]',
    '["landuse"="forest"]',
    '["landuse"="grass"]',
    '["landuse"="meadow"]',
    '["leisure"="park"]',
    '["leisure"="garden"]',
]


def fetch_osm_vegetation(circuit_id, lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass API for vegetation polygons within the bounding box.

    Returns a list of vegetation dicts, each with:
      footprint : list of [lon, lat] pairs (closed polygon)
      veg_tag   : vegetation type string ('wood', 'forest', 'grass', etc.)
      osm_id    : OSM way/relation ID
      osm_type  : 'way' or 'relation'
    """
    _cp = RAW_DATA_DIR / "osm" / f"{circuit_id}_vegetation.json"
    if USE_RAW_CACHE and _cp.exists():
        with open(_cp) as _f: return _json.load(_f)

    bbox      = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    way_lines = "\n  ".join(f'way{t}({bbox});'      for t in VEGETATION_TAGS)
    rel_lines = "\n  ".join(f'relation{t}({bbox});' for t in VEGETATION_TAGS)

    query = f"""
[out:json][timeout:90];
(
  {way_lines}
  {rel_lines}
);
out geom;
"""
    try:
        for attempt in range(3):
            resp = requests.post(
                OVERPASS_URL, data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "F1CircuitsPipeline/1.0"},
                timeout=95)
            if resp.status_code in (429, 504):
                import time as _t; _t.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            print(f"  [vegetation] failed after 3 attempts")
            return []
    except Exception as e:
        print(f"  skip(OSM vegetation failed: {e})")
        return []

    elements  = data.get("elements", [])
    way_geom  = {}
    relations = []
    direct    = []

    for el in elements:
        t = el.get("type")
        if t == "way":
            geom = el.get("geometry", [])
            if geom:
                way_geom[el["id"]] = [(g["lon"], g["lat"]) for g in geom]
            if el.get("tags"):
                direct.append(el)
        elif t == "relation":
            relations.append(el)

    # Resolve missing member way geometry
    if relations:
        missing = {m["ref"] for rel in relations
                   for m in rel.get("members", [])
                   if m.get("type") == "way" and m["ref"] not in way_geom}
        if missing:
            q2 = (f"[out:json][timeout:60];\n"
                  f"way(id:{','.join(map(str, missing))});\nout geom;\n")
            try:
                r2 = requests.post(
                    OVERPASS_URL, data={"data": q2},
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "User-Agent": "F1CircuitsPipeline/1.0"},
                    timeout=65)
                r2.raise_for_status()
                for el in r2.json().get("elements", []):
                    if el.get("type") == "way":
                        geom = el.get("geometry", [])
                        if geom:
                            way_geom[el["id"]] = [(g["lon"], g["lat"]) for g in geom]
            except Exception as e:
                print(f"  [vegetation] member way fetch failed: {e}")

    def primary_tag(el):
        tags = el.get("tags", {})
        for key in ("natural", "landuse", "leisure"):
            if key in tags:
                return tags[key]
        return "vegetation"

    def too_small(fp):
        if not fp:
            return True
        lons = [p[0] for p in fp]; lats = [p[1] for p in fp]
        mid_lat  = (min(lats) + max(lats)) / 2.0
        cos_lat  = math.cos(math.radians(mid_lat))
        deg_to_m = R_EARTH * math.pi / 180.0
        return ((max(lons)-min(lons)) * cos_lat * deg_to_m < TERRAIN_RESOLUTION_M or
                (max(lats)-min(lats))             * deg_to_m < TERRAIN_RESOLUTION_M)

    TOL = 1e-5

    def endpoints_match(a, b):
        return abs(a[0]-b[0]) < TOL and abs(a[1]-b[1]) < TOL

    def stitch_ways(way_ids):
        chains = [list(way_geom[wid])
                  for wid in way_ids if wid in way_geom and way_geom[wid]]
        if not chains:
            return []
        merged = True
        while merged:
            merged = False
            for i in range(len(chains)):
                for j in range(len(chains)):
                    if i == j: continue
                    if endpoints_match(chains[i][-1], chains[j][0]):
                        chains[i].extend(chains[j][1:]); chains.pop(j)
                        merged = True; break
                    if endpoints_match(chains[i][-1], chains[j][-1]):
                        chains[i].extend(reversed(chains[j][:-1])); chains.pop(j)
                        merged = True; break
                if merged: break
        rings = []
        for chain in chains:
            if len(chain) < 3: continue
            if not endpoints_match(chain[0], chain[-1]):
                chain.append(chain[0])
            rings.append(chain)
        return rings

    vegetation = []
    seen       = set()

    # Direct ways
    for el in direct:
        eid = el["id"]
        if eid in seen: continue
        seen.add(eid)
        geom = way_geom.get(eid, [])
        if len(geom) < 3: continue
        fp  = [[p[0], p[1]] for p in geom]
        if too_small(fp): continue
        vegetation.append({
            "osm_id":   eid,
            "osm_type": "way",
            "veg_tag":  primary_tag(el),
            "footprint": fp,
        })

    # Relations
    for el in relations:
        eid = el["id"]
        if eid in seen: continue
        seen.add(eid)
        outer_ids = [m["ref"] for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role") == "outer"]
        if not outer_ids: continue
        rings = stitch_ways(outer_ids)
        if not rings: continue
        outer = max(rings, key=len)
        fp    = [[p[0], p[1]] for p in outer]
        if len(fp) < 3 or too_small(fp): continue
        for mid in outer_ids: seen.add(mid)
        vegetation.append({
            "osm_id":   eid,
            "osm_type": "relation",
            "veg_tag":  primary_tag(el),
            "footprint": fp,
        })

    # Clip to bbox using Shapely
    try:
        from shapely.geometry import Polygon as ShapelyPoly, box
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        clipped   = []
        for body in vegetation:
            fp = body.get("footprint", [])
            if len(fp) < 3:
                continue
            try:
                poly      = ShapelyPoly([(p[0], p[1]) for p in fp])
                if bbox_poly.contains(poly):
                    clipped.append(body)
                    continue
                intersect = poly.intersection(bbox_poly)
                if intersect.is_empty:
                    continue
                if intersect.geom_type == 'Polygon':
                    body["footprint"] = [[c[0], c[1]]
                                         for c in intersect.exterior.coords]
                    clipped.append(body)
                elif intersect.geom_type == 'MultiPolygon':
                    import copy
                    for part in intersect.geoms:
                        b = copy.copy(body)
                        b["footprint"] = [[c[0], c[1]]
                                          for c in part.exterior.coords]
                        clipped.append(b)
            except Exception:
                clipped.append(body)
        vegetation = clipped
    except ImportError:
        pass

    _cp.parent.mkdir(parents=True, exist_ok=True)
    with open(_cp, "w") as _f: _json.dump(vegetation, _f)
    return vegetation