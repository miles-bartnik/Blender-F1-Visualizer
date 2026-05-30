"""
water.py
--------
OSM water body and coastline fetching, polygon assembly, and sea normal
determination for the F1 circuit pipeline.
"""

import math
import time

import requests

from config import (
    OVERPASS_URL, OVERPASS_DELAY,
    COASTLINE_EXTEND_KM, TERRAIN_RESOLUTION_M,
    WATER_COVERAGE_THRESHOLD,
    RAW_DATA_DIR, USE_RAW_CACHE,
)

import json as _json

R_EARTH = 6_371_000.0


def _osm_cache_path(circuit_id, data_type):
    return RAW_DATA_DIR / "osm" / f"{circuit_id}_{data_type}.json"


def _load_osm_cache(circuit_id, data_type):
    p = _osm_cache_path(circuit_id, data_type)
    if USE_RAW_CACHE and p.exists():
        with open(p) as f:
            return _json.load(f)
    return None


def _save_osm_cache(circuit_id, data_type, data):
    p = _osm_cache_path(circuit_id, data_type)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        _json.dump(data, f)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def point_in_polygon(lon, lat, ring):
    """Ray casting point-in-polygon test. Ring need not be closed."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat) and
                lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside



def _extended_bbox(lat_min, lon_min, lat_max, lon_max):
    """Compute extended bbox corners CCW (BL→BR→TR→TL) at +COASTLINE_EXTEND_KM."""
    ext_m      = COASTLINE_EXTEND_KM * 1000.0
    lat_centre = (lat_min + lat_max) / 2.0
    cos_lat    = math.cos(math.radians(lat_centre))
    d_lat      = math.degrees(ext_m / R_EARTH)
    d_lon      = math.degrees(ext_m / (R_EARTH * cos_lat))
    return (
        lon_min - d_lon, lat_min - d_lat,
        lon_max + d_lon, lat_max + d_lat,
    )



# =============================================================================
# OSM WATER BODIES
# =============================================================================

def fetch_osm_water(circuit_id, lat_min, lon_min, lat_max, lon_max, elevations, n_lat, n_lon):
    """
    Query Overpass for all significant water body polygons within the bbox.
    Results are cached to data/raw/osm/{circuit_id}_water.json.

    Tags queried:
      natural=water/wetland, landuse=reservoir/basin,
      waterway=riverbank/dock/canal

    Ways with inline geometry and multipolygon relations (resolved via a
    second targeted query for member ways) are both handled.

    Returns a list of water dicts:
      osm_id, osm_type, water_tag, footprint, ele_m
    """
    cached = _load_osm_cache(circuit_id, "water")
    if cached is not None:
        return cached

    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    tags = [
        '["natural"="water"]',   '["natural"="wetland"]',
        '["landuse"="reservoir"]', '["landuse"="basin"]',
        '["waterway"="riverbank"]', '["waterway"="dock"]',
        '["waterway"="canal"]',
    ]
    way_lines      = "\n  ".join(f'way{t}({bbox});'      for t in tags)
    relation_lines = "\n  ".join(f'relation{t}({bbox});' for t in tags)

    query = f"""
[out:json][timeout:90];
(
  {way_lines}
  {relation_lines}
);
out geom;
"""
    try:
        data = None
        for attempt in range(3):
            resp = requests.post(
                OVERPASS_URL, data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "F1CircuitsPipeline/1.0"},
                timeout=95)
            if resp.status_code in (429, 504):
                import time as _t
                wait = 30 * (attempt + 1)
                print(f"\n  [water] server error {resp.status_code}, "
                      f"waiting {wait}s ...", end=" ", flush=True)
                _t.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        if data is None:
            print(f"skip(OSM water failed after 3 attempts)")
            return []
    except Exception as e:
        print(f"skip(OSM water failed: {e})")
        return []

    elements    = data.get("elements", [])
    way_geom    = {}
    relations   = []
    direct_ways = []

    for el in elements:
        t = el.get("type")
        if t == "way":
            geom = el.get("geometry", [])
            if geom:
                way_geom[el["id"]] = [(g["lon"], g["lat"]) for g in geom]
            if el.get("tags"):
                direct_ways.append(el)
        elif t == "relation":
            relations.append(el)

    # Resolve missing member way geometry
    if relations:
        missing = {m["ref"] for rel in relations
                   for m in rel.get("members", [])
                   if m.get("type") == "way" and m["ref"] not in way_geom}
        if missing:
            q2 = f"[out:json][timeout:60];\nway(id:{','.join(map(str,missing))});\nout geom;\n"
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
                print(f"  [water] member way fetch failed: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────────
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    def sample_elevation(footprint):
        if not footprint:
            return 0.0
        c_lon = sum(p[0] for p in footprint) / len(footprint)
        c_lat = sum(p[1] for p in footprint) / len(footprint)
        col_f = (c_lon - lon_min) / lon_range * (n_lon - 1) if lon_range else 0.0
        row_f = (c_lat - lat_min) / lat_range * (n_lat - 1) if lat_range else 0.0
        col_i = max(0, min(n_lon - 1, int(round(col_f))))
        row_i = max(0, min(n_lat - 1, int(round(row_f))))
        idx   = row_i * n_lon + col_i
        ele   = elevations[idx] if idx < len(elevations) and elevations[idx] is not None else 0.0
        return round(float(ele), 2)

    def primary_tag(el):
        tags = el.get("tags", {})
        for key in ("water", "natural", "landuse", "waterway"):
            if key in tags:
                return tags[key]
        return "water"

    def footprint_too_small(fp):
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
                        chains[i].extend(chains[j][1:]); chains.pop(j); merged = True; break
                    if endpoints_match(chains[i][-1], chains[j][-1]):
                        chains[i].extend(reversed(chains[j][:-1])); chains.pop(j); merged = True; break
                if merged: break
        rings = []
        for chain in chains:
            if len(chain) < 3: continue
            if not endpoints_match(chain[0], chain[-1]): chain.append(chain[0])
            rings.append(chain)
        return rings

    # ── Process ways ──────────────────────────────────────────────────────────
    water = []
    seen  = set()

    for el in direct_ways:
        eid = el["id"]
        if eid in seen: continue
        seen.add(eid)
        geom = way_geom.get(eid, [])
        if len(geom) < 3: continue
        fp  = [[p[0], p[1]] for p in geom]
        if footprint_too_small(fp): continue
        tag = primary_tag(el)
        if tag == "bay": continue
        water.append({"osm_id": eid, "osm_type": "way", "water_tag": tag,
                      "footprint": fp, "holes": [],
                      "ele_m": sample_elevation(fp)})

    # ── Process relations ─────────────────────────────────────────────────────
    for el in relations:
        eid = el["id"]
        if eid in seen: continue
        seen.add(eid)
        outer_ids = [m["ref"] for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role") == "outer"]
        inner_ids = [m["ref"] for m in el.get("members", [])
                     if m.get("type") == "way" and m.get("role") == "inner"]
        if not outer_ids: continue
        rings = stitch_ways(outer_ids)
        if not rings: continue
        outer = max(rings, key=len)
        fp    = [[p[0], p[1]] for p in outer]
        if len(fp) < 3 or footprint_too_small(fp): continue
        tag = primary_tag(el)
        if tag == "bay": continue
        for mid in outer_ids: seen.add(mid)

        # Collect island holes - inner rings that aren't too small
        holes = []
        if inner_ids:
            inner_rings = stitch_ways(inner_ids)
            for ir in inner_rings:
                hole = [[p[0], p[1]] for p in ir]
                if len(hole) >= 3 and not footprint_too_small(hole):
                    holes.append(hole)
            for mid in inner_ids: seen.add(mid)

        water.append({"osm_id": eid, "osm_type": "relation", "water_tag": tag,
                      "footprint": fp, "holes": holes,
                      "ele_m": sample_elevation(fp)})

    # ── Clip all footprints to terrain bbox ──────────────────────────────────
    try:
        from shapely.geometry import Polygon as ShapelyPoly, box
        bbox_poly  = box(lon_min, lat_min, lon_max, lat_max)
        clipped    = []
        n_trimmed  = 0
        n_dropped  = 0
        for body in water:
            fp = body.get("footprint", [])
            if len(fp) < 3:
                continue
            try:
                poly      = ShapelyPoly([(p[0], p[1]) for p in fp])
                # Check if already fully inside
                if bbox_poly.contains(poly):
                    clipped.append(body)
                    continue
                intersect = poly.intersection(bbox_poly)
                if intersect.is_empty:
                    n_dropped += 1
                    continue
                if intersect.geom_type == 'Polygon':
                    body["footprint"] = [[c[0], c[1]] for c in intersect.exterior.coords]
                    # Clip holes too
                    clipped_holes = []
                    for hole in body.get("holes", []):
                        try:
                            hp = ShapelyPoly([(p[0], p[1]) for p in hole])
                            hi = hp.intersection(bbox_poly)
                            if not hi.is_empty and hi.geom_type == 'Polygon':
                                clipped_holes.append([[c[0], c[1]] for c in hi.exterior.coords])
                        except Exception:
                            clipped_holes.append(hole)
                    body["holes"] = clipped_holes
                    clipped.append(body)
                    n_trimmed += 1
                elif intersect.geom_type == 'MultiPolygon':
                    import copy
                    for part in intersect.geoms:
                        b = copy.copy(body)
                        b["footprint"] = [[c[0], c[1]] for c in part.exterior.coords]
                        b["holes"]     = []  # holes not preserved for multipolygon splits
                        clipped.append(b)
                    n_trimmed += 1
            except Exception as e:
                clipped.append(body)
        water = clipped
        print(f"  [water clip] {len(water)} kept, {n_trimmed} trimmed, {n_dropped} dropped")

        # ── Drop bodies ≥95% covered by a larger water body ──────────────────
        water_polys = []
        for body in water:
            fp = body.get("footprint", [])
            try:
                poly = ShapelyPoly([(p[0], p[1]) for p in fp])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                water_polys.append((poly, poly.area))
            except Exception:
                water_polys.append((None, 0.0))

        keep = [True] * len(water)
        for i in range(len(water)):
            if not keep[i]:
                continue
            small_poly, small_area = water_polys[i]
            if small_poly is None or small_area == 0:
                continue
            for j in range(len(water)):
                if i == j or not keep[j]:
                    continue
                large_poly, large_area = water_polys[j]
                if large_poly is None or large_area <= small_area:
                    continue
                try:
                    overlap = small_poly.intersection(large_poly).area
                    if overlap / small_area >= WATER_COVERAGE_THRESHOLD:
                        keep[i] = False
                        break
                except Exception:
                    pass

        n_covered = sum(1 for k in keep if not k)
        if n_covered:
            water = [w for w, k in zip(water, keep) if k]
            print(f"  [water coverage] dropped {n_covered} "
                  f"body/bodies ≥95% covered by a larger water body")

    except ImportError:
        print("  [water clip] shapely not available - skipping clip")

    _save_osm_cache(circuit_id, "water", water)
    return water


# =============================================================================
# OSM COASTLINE
# =============================================================================

def fetch_osm_coastline(circuit_id, lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass for natural=coastline ways within an extended bounding box
    (terrain bbox + COASTLINE_EXTEND_KM). Returns the longest stitched chain
    as an open list of [lon, lat] pairs, or None if none found.
    """
    cached = _load_osm_cache(circuit_id, "coastline")
    if cached is not None:
        return cached[0], cached[1]

    ext_lon_min, ext_lat_min, ext_lon_max, ext_lat_max = _extended_bbox(
        lat_min, lon_min, lat_max, lon_max)

    query = f"""
[out:json][timeout:90];
(
  way["natural"="coastline"]({ext_lat_min},{ext_lon_min},{ext_lat_max},{ext_lon_max});
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
                wait = 30 * (attempt + 1)
                print(f"  [coastline] server error {resp.status_code}, "
                      f"waiting {wait}s ...", end=" ", flush=True)
                import time as _time
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            print(f"skip(OSM coastline failed after 3 attempts)")
            return None, False
    except Exception as e:
        print(f"skip(OSM coastline failed: {e})")
        return None, False

    elements = data.get("elements", [])
    if not elements:
        return None, False

    segments = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        nodes = [(g["lon"], g["lat"]) for g in el["geometry"]]
        if len(nodes) >= 2:
            segments.append(nodes)

    if not segments:
        return None, False

    TOL = 1e-5

    def endpoints_match(a, b):
        return abs(a[0]-b[0]) < TOL and abs(a[1]-b[1]) < TOL

    # Stitch segments into the longest continuous chain.
    # Track whether any segment was appended in reverse - if so the combined
    # chain no longer reliably follows the OSM land-left convention.
    chains        = [list(seg) for seg in segments]
    had_reversals = False
    merged        = True
    while merged:
        merged = False
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i == j:
                    continue
                if endpoints_match(chains[i][-1], chains[j][0]):
                    chains[i].extend(chains[j][1:])
                    chains.pop(j)
                    merged = True
                    break
                if endpoints_match(chains[i][-1], chains[j][-1]):
                    chains[i].extend(reversed(chains[j][:-1]))
                    chains.pop(j)
                    had_reversals = True
                    merged        = True
                    break
            if merged:
                break

    chain  = max(chains, key=len)
    result = [[p[0], p[1]] for p in chain], had_reversals
    _save_osm_cache(circuit_id, "coastline", list(result))
    return result


# =============================================================================
# SEA POLYGON ASSEMBLY
# =============================================================================

def _sea_side_score(chain_pts, cx, cy):
    """
    Signed cross-product score of centroid (cx, cy) against a directed chain.

    For each segment A->B and centroid P:
        cross = (B.x-A.x)*(P.y-A.y) - (B.y-A.y)*(P.x-A.x)
    Summed over the chain this gives:
        negative => centroid is to the RIGHT of the chain (sea, OSM land-left)
        positive => centroid is to the LEFT (land)
    """
    score = 0.0
    for i in range(len(chain_pts) - 1):
        ax, ay = chain_pts[i]
        bx, by = chain_pts[i + 1]
        dx = bx - ax
        dy = by - ay
        score += dx * (cy - ay) - dy * (cx - ax)
    return score


def build_sea_polygons(sea_chain, lat_min, lon_min, lat_max, lon_max,
                       had_reversals=False, structures=None):
    """
    Split the terrain bbox with the coastline and return sea-side polygon rings.

    Returns a list of [[lon, lat], ...] closed rings (one per sea region),
    or [] on failure.

    Uses OSM land-left convention: land is to the LEFT of the chain direction,
    sea is to the RIGHT. The cross-product score is negative for centroids on
    the right (sea) side.

    had_reversals and structures are accepted for API compatibility but ignored -
    the cross-product approach is robust regardless of chain assembly details.
    """
    try:
        from shapely.geometry import box, LineString
        from shapely.ops import split
    except ImportError:
        print("  [sea] shapely not installed - cannot build sea polygon")
        return []

    if not sea_chain or len(sea_chain) < 2:
        return []

    pts    = [(p[0], p[1]) for p in sea_chain]
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.02

    def extend_endpoint(inner, outer):
        dx = outer[0] - inner[0]
        dy = outer[1] - inner[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return outer
        scale = (length + margin) / length
        return (inner[0] + dx * scale, inner[1] + dy * scale)

    extended = (
        [extend_endpoint(pts[1],  pts[0])]
        + pts
        + [extend_endpoint(pts[-2], pts[-1])]
    )

    bbox_poly  = box(lon_min, lat_min, lon_max, lat_max)
    coast_line = LineString(extended)

    try:
        pieces = split(bbox_poly, coast_line)
        polys  = list(pieces.geoms)
    except Exception as e:
        print(f"  [sea] shapely split failed: {e}")
        return []

    if not polys:
        return []

    if len(polys) == 1:
        print("  [sea] coastline produced 1 polygon")
        return [[[c[0], c[1]] for c in polys[0].exterior.coords]]

    print(f"  [sea] split bbox into {len(polys)} polygon(s)")

    # Use only the chain portion inside the bbox for scoring.
    # The chain is fetched from an extended bbox so most of it lies outside.
    bbox_chain  = [p for p in sea_chain
                   if lon_min <= p[0] <= lon_max and lat_min <= p[1] <= lat_max]
    score_chain = bbox_chain if len(bbox_chain) >= 2 else sea_chain

    # Score each polygon. Sea is to the RIGHT of the chain -> negative score.
    scored = []
    for poly in polys:
        cx, cy = poly.centroid.x, poly.centroid.y
        score  = _sea_side_score(score_chain, cx, cy)
        scored.append((score, poly))
        print(f"  [sea]   area={poly.area:.6f}  score={score:+.4f}")

    sea_polys = [poly for score, poly in scored if score < 0]

    if not sea_polys:
        # No clearly negative polygon - take the most negative one as fallback
        best_score, best_poly = min(scored, key=lambda x: x[0])
        sea_polys = [best_poly]
        print(f"  [sea] no negative-score polygon - using most negative ({best_score:+.4f})")
    else:
        print(f"  [sea] {len(sea_polys)} sea polygon(s) identified")

    return [[[c[0], c[1]] for c in poly.exterior.coords] for poly in sea_polys]


# =============================================================================
# H3 HEXAGONAL MESH GENERATION
# =============================================================================

def build_water_hexagons(ring, resolution=9, holes=None):
    """
    Convert a closed polygon ring [[lon, lat], ...] into H3 triangles.
    Holes (islands) are excluded via h3.LatLngPoly hole rings.

    ring   : outer boundary [[lon, lat], ...]
    holes  : list of inner rings [[lon, lat], ...] (islands)
    """
    try:
        import h3
    except ImportError:
        print("  [h3] h3 not installed -- skipping hexagon mesh generation")
        return []

    if not ring or len(ring) < 3:
        return []

    latlng_ring = [(p[1], p[0]) for p in ring]
    if latlng_ring[0] == latlng_ring[-1]:
        latlng_ring = latlng_ring[:-1]

    latlng_holes = []
    for hole in (holes or []):
        if not hole or len(hole) < 3:
            continue
        lr = [(p[1], p[0]) for p in hole]
        if lr[0] == lr[-1]:
            lr = lr[:-1]
        if len(lr) >= 3:
            latlng_holes.append(lr)

    try:
        poly  = h3.LatLngPoly(latlng_ring, *latlng_holes)
        cells = h3.h3shape_to_cells(poly, resolution)
    except Exception as e:
        print(f"  [h3] polyfill failed: {e}")
        return []

    if not cells:
        return []

    triangles = []
    for cell in cells:
        clat, clon = h3.cell_to_latlng(cell)
        centre     = [clon, clat]
        boundary   = h3.cell_to_boundary(cell)
        n          = len(boundary)
        for i in range(n):
            v0 = [boundary[i][1],         boundary[i][0]]
            v1 = [boundary[(i+1)%n][1],   boundary[(i+1)%n][0]]
            triangles.append([centre, v0, v1])

    return triangles