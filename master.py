"""
master.py
---------
Main entry point for the F1 circuit pipeline.
Orchestrates circuit processing, terrain fetching, and GeoJSON assembly.

Usage:
    python -m new.master
    # or
    python master.py  (from within the new/ directory)
"""

import json
import math
import warnings

warnings.filterwarnings("ignore")

import fastf1
fastf1.set_log_level("WARNING")

from config import (
    CIRCUITS_INDEX_URL, CIRCUITS_DIR_URL,
    OUTPUT_DIR, CIRCUITS_OUT_DIR, COMBINED_OUT,
    CACHE_DIR, INPUT_DIRECTORY, PIT_DIRECTORY, SECTORS_FILE, START_FINISH_FILE,
    CIRCUITS_TO_PROCESS, OVERWRITE_TERRAIN,
    OVERWRITE_STRUCTURES, OVERWRITE_WATER, OVERWRITE_VEGETATION,
    H3_RESOLUTION, BOUNDARY_DEPTH,
    CIRCUITS_EXCLUDE,
    WATER_DEPTH_K, WATER_DEPTH_MAX_M,
)
from circuits import (
    fetch_json, extract_linestring, extract_all_coordinates, centroid_lonlat,
    process_circuit,
)
from terrain    import fetch_terrain_grid, build_terrain_hexagons, build_terrain_boundary_hexagons
from water      import build_sea_polygons, build_water_hexagons
from vegetation import fetch_osm_vegetation

R_EARTH = 6_371_000.0


def _polygon_area_m2(ring):
    """Shoelace area of a [[lon, lat], ...] ring in square metres."""
    n = len(ring)
    if n < 3:
        return 0.0
    lat_ref       = sum(p[1] for p in ring) / n
    cos_lat       = math.cos(math.radians(lat_ref))
    m_per_deg_lat = R_EARTH * math.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * cos_lat
    area = 0.0
    for i in range(n):
        j    = (i + 1) % n
        xi   = ring[i][0] * m_per_deg_lon
        yi   = ring[i][1] * m_per_deg_lat
        xj   = ring[j][0] * m_per_deg_lon
        yj   = ring[j][1] * m_per_deg_lat
        area += xi * yj - xj * yi
    return abs(area) / 2.0


def _bake_water_depressions(terrain_data, water_bodies, k, max_depth_m):
    """
    Bake Håkanson + cosine-falloff depression for every water body (lake /
    river / canal / reservoir / …) into terrain_data["terrain_points"].

    depth  = min(max_depth_m, k * sqrt(area_m²))
    factor = 0.5 * (1 − cos(π * t))   where t = dist_to_boundary / max_interior_dist

    Holes (islands) are excluded: points inside a hole are not depressed.
    The boundary used for distance includes both the outer shore AND any island
    shores (both act as "coasts" → shallower nearby).

    Returns the set of terrain-point indices that were modified (used by
    _bake_sea_depression to avoid double-depressing the same vertices).
    """
    try:
        from shapely.geometry import Polygon as _Poly, Point as _Pt
        from shapely.prepared import prep as _prep
    except ImportError:
        return set()

    points   = terrain_data["terrain_points"]
    modified = set()

    for body in water_bodies:
        fp    = body.get("footprint", [])
        holes = body.get("holes", [])
        if len(fp) < 3:
            continue

        try:
            outer      = [(p[0], p[1]) for p in fp]
            hole_rings = [[(p[0], p[1]) for p in h] for h in holes if len(h) >= 3]
            poly       = _Poly(outer, hole_rings)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        prepared = _prep(poly)
        boundary = poly.boundary   # includes outer shore + island shores

        area_m2 = _polygon_area_m2(fp)
        depth_m = min(max_depth_m, k * math.sqrt(area_m2))

        # First pass: collect interior indices + boundary distances
        body_idx   = []
        body_dists = []
        for i, pt in enumerate(points):
            if pt[2] is None:
                continue
            p = _Pt(pt[0], pt[1])
            if prepared.covers(p):
                body_idx.append(i)
                body_dists.append(boundary.distance(p))

        if not body_idx:
            continue

        max_dist = max(body_dists) or 1.0

        # Second pass: apply cosine falloff
        for i, dist in zip(body_idx, body_dists):
            t            = dist / max_dist
            factor       = 0.5 * (1.0 - math.cos(math.pi * t))
            points[i][2] = round(points[i][2] - depth_m * factor, 2)
            modified.add(i)

    if modified:
        print(f"  [water depression] {len(water_bodies)} bodies, "
              f"{len(modified)} terrain pts depressed")
    return modified


def _bake_sea_depression(terrain_data, sea_ring, sea_chain, k, max_depth_m,
                         skip_indices=None):
    """
    Bake Håkanson + cosine-falloff sea depression into terrain_data["terrain_points"].

    depth  = min(max_depth_m, k * sqrt(area_m²))
    factor = 0.5 * (1 − cos(π * t))   where t = dist_from_coast / max_dist

    sea_chain is the raw OSM coastline (used for distance; avoids the artificial
    bbox-edge distances that the full polygon boundary would introduce).

    skip_indices: set of point indices already modified by _bake_water_depressions
    (water bodies inside the sea polygon are NOT double-depressed).

    Modifies terrain_data["terrain_points"] in-place.
    """
    try:
        from shapely.geometry import Polygon as _Poly, LineString as _LS, Point as _Pt
        from shapely.prepared import prep as _prep
    except ImportError:
        return

    if not sea_ring or len(sea_ring) < 3:
        return

    sea_poly = _Poly([(p[0], p[1]) for p in sea_ring])
    if not sea_poly.is_valid:
        sea_poly = sea_poly.buffer(0)
    sea_prep = _prep(sea_poly)

    # Use the raw OSM coastline chain for distance so we measure how far each
    # point is from the actual shore, not from the artificial bbox edges.
    if sea_chain and len(sea_chain) >= 2:
        coast_ls = _LS([(p[0], p[1]) for p in sea_chain])
    else:
        coast_ls = _LS([(p[0], p[1]) for p in sea_ring])

    area_m2 = _polygon_area_m2(sea_ring)
    depth_m = min(max_depth_m, k * math.sqrt(area_m2))

    points     = terrain_data["terrain_points"]
    skip       = skip_indices or set()

    # First pass: collect sea point indices and distances, skipping water-
    # body-modified points so inland water bodies are not double-depressed.
    sea_indices = []
    sea_dists   = []
    for i, pt in enumerate(points):
        if pt[2] is None or i in skip:
            continue
        p = _Pt(pt[0], pt[1])
        if sea_prep.covers(p):
            sea_indices.append(i)
            sea_dists.append(coast_ls.distance(p))

    if not sea_indices:
        return

    max_dist = max(sea_dists) or 1.0

    # Second pass: apply cosine falloff depression
    for i, dist in zip(sea_indices, sea_dists):
        t            = dist / max_dist
        factor       = 0.5 * (1.0 - math.cos(math.pi * t))
        points[i][2] = round(points[i][2] - depth_m * factor, 2)

    print(f"  [sea depression] {len(sea_indices)} pts  "
          f"depth={depth_m:.1f}m  max_dist={max_dist:.5f}°")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CIRCUITS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # ── Load sectors config ───────────────────────────────────────────────────
    sectors_lookup = {}
    if SECTORS_FILE.exists():
        try:
            with open(SECTORS_FILE, encoding="utf-8") as f:
                sectors_data = json.load(f)
            sectors_lookup = {r["circuit_id"]: r for r in sectors_data
                              if "circuit_id" in r}
            print(f"Loaded sectors config for "
                  f"{len(sectors_lookup)} circuits from {SECTORS_FILE}\n")
        except Exception as e:
            print(f"Warning: could not load {SECTORS_FILE}: {e}\n")
    else:
        print(f"Warning: {SECTORS_FILE} not found\n")

    # ── Load start/finish line positions ─────────────────────────────────────
    sf_lookup = {}
    if START_FINISH_FILE.exists():
        try:
            with open(START_FINISH_FILE, encoding="utf-8") as f:
                sf_data = json.load(f)
            for feat in sf_data.get("features", []):
                props  = feat.get("properties", {})
                cid    = props.get("circuit_id")
                coords = (feat.get("geometry") or {}).get("coordinates")
                if cid and coords:
                    sf_lookup[cid] = coords
            print(f"Loaded start/finish lines for "
                  f"{len(sf_lookup)} circuits from {START_FINISH_FILE}\n")
        except Exception as e:
            print(f"Warning: could not load {START_FINISH_FILE}: {e}\n")
    else:
        print(f"Warning: {START_FINISH_FILE} not found — "
              f"circuit rotation will fail\n")

    # ── Fetch circuit index ───────────────────────────────────────────────────
    print("Fetching circuit index from GitHub ...")
    locations   = fetch_json(CIRCUITS_INDEX_URL)
    circuit_ids = [loc["id"] for loc in locations if "id" in loc]
    if CIRCUITS_TO_PROCESS:
        circuit_ids = [c for c in circuit_ids if c in CIRCUITS_TO_PROCESS]
    if CIRCUITS_EXCLUDE:
        circuit_ids = [c for c in circuit_ids if c not in CIRCUITS_EXCLUDE]
    print(f"Found {len(circuit_ids)} circuits.\n")

    all_features = []
    telemetry_ok = []
    srtm_ok      = []
    failed       = []

    for circuit_id in circuit_ids:
        url = f"{CIRCUITS_DIR_URL}/{circuit_id}.geojson"
        print(f"{'='*60}")
        print(f"[{circuit_id}]")

        out_path = CIRCUITS_OUT_DIR / f"{circuit_id}.geojson"

        # ── Fetch source GeoJSON ──────────────────────────────────────────────
        try:
            geojson = fetch_json(url)
        except Exception as e:
            print(f"  SKIP — could not fetch GeoJSON: {e}")
            failed.append(circuit_id)
            continue

        coords = extract_linestring(geojson)
        if not coords:
            coords = extract_all_coordinates(geojson)
        if not coords:
            print(f"  SKIP — no coordinates in GeoJSON")
            failed.append(circuit_id)
            continue

        lon0, lat0 = centroid_lonlat(coords)

        # ── Process circuit ───────────────────────────────────────────────────
        sector_config = dict(sectors_lookup.get(circuit_id) or {})
        sf = sf_lookup.get(circuit_id)
        if sf:
            sector_config["start_finish_line"] = sf
        try:
            pipeline_result = process_circuit(circuit_id, geojson, lon0, lat0, sector_config)
        except Exception as e:
            print(f"  SKIP — {e}")
            failed.append(circuit_id)
            continue

        if pipeline_result is None:
            failed.append(circuit_id); continue

        result, sectors = pipeline_result
        if result is None:
            failed.append(circuit_id); continue

        # ── Terrain ───────────────────────────────────────────────────────────
        terrain_data = fetch_terrain_grid(circuit_id, coords, CIRCUITS_OUT_DIR)

        # ── Terrain depression baking (must precede terrain hex generation) ────
        # Water bodies first → their modified indices are skipped by sea pass
        # so inland lakes inside a sea polygon are not double-depressed.
        sea_ring  = None
        sea_chain = None
        if terrain_data:
            water_modified = _bake_water_depressions(
                terrain_data,
                terrain_data.get("water", []),
                WATER_DEPTH_K, WATER_DEPTH_MAX_M,
            )

            sea_chain  = terrain_data.get("sea")
            sea_rings  = []   # list of [[lon,lat],...] rings from build_sea_polygons
            if sea_chain and len(sea_chain) >= 2:
                had_rev  = terrain_data.get("sea_had_reversals")
                sea_rings = build_sea_polygons(
                    sea_chain,
                    terrain_data["lat_min"], terrain_data["lon_min"],
                    terrain_data["lat_max"], terrain_data["lon_max"],
                    had_reversals=(had_rev if had_rev is not None else True),
                    structures=terrain_data.get("structures", []),
                )
                for sea_ring in sea_rings:
                    if sea_ring:
                        _bake_sea_depression(terrain_data, sea_ring, sea_chain,
                                             WATER_DEPTH_K, WATER_DEPTH_MAX_M,
                                             skip_indices=water_modified)

        # ── Circuit display name ──────────────────────────────────────────────
        circuit_name_display = circuit_id

        def find_name(obj):
            nonlocal circuit_name_display
            t = obj.get("type", "")
            if t == "FeatureCollection":
                for feat in obj.get("features", []): find_name(feat)
            elif t == "Feature":
                n = (obj.get("properties") or {}).get("Name", "")
                if n: circuit_name_display = n
        find_name(result)

        # ── Build consolidated FeatureCollection ──────────────────────────────
        features = []

        # 1. Circuit track
        if result.get("type") == "FeatureCollection":
            for feat in result.get("features", []):
                props = feat.get("properties") or {}
                props["feature_type"] = "circuit"
                feat["properties"] = props
                features.append(feat)
        elif result.get("type") == "Feature":
            props = result.get("properties") or {}
            props["feature_type"] = "circuit"
            result["properties"] = props
            features.append(result)

        # 2. Sectors
        if sectors:
            s1_coords, s2_coords, s3_coords = sectors
            for sector_num, sector_coords in enumerate([s1_coords, s2_coords, s3_coords], 1):
                if not sector_coords or len(sector_coords) < 2:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":         f"{circuit_name_display} - S{sector_num}",
                        "circuit_id":   circuit_id,
                        "feature_type": f"sector{sector_num}",
                    },
                    "geometry": {
                        "type":        "LineString",
                        "coordinates": sector_coords,
                    }
                })
            print(f"  Sectors: S1({len(s1_coords)}) S2({len(s2_coords)}) S3({len(s3_coords)}) pts")

        # 3. Pit lane
        pit_src = PIT_DIRECTORY / f"{circuit_id}_pit.geojson"
        if pit_src.exists():
            try:
                with open(pit_src, encoding="utf-8") as f:
                    pit_geojson = json.load(f)
                pit_feats = (pit_geojson.get("features", [])
                             if pit_geojson.get("type") == "FeatureCollection"
                             else [pit_geojson])
                for feat in pit_feats:
                    props = feat.get("properties") or {}
                    props.setdefault("Name", f"{circuit_name_display} - Pit")
                    props.setdefault("circuit_id", circuit_id)
                    props["feature_type"] = "pitlane"
                    feat["properties"] = props
                    features.append(feat)
                print(f"  Pit lane folded in from {pit_src.name}")
            except Exception as e:
                print(f"  Pit lane SKIP — {e}")

        # 4. Terrain
        if terrain_data:
            terrain_tris  = build_terrain_hexagons(terrain_data, H3_RESOLUTION)
            boundary_tris = build_terrain_boundary_hexagons(terrain_data, H3_RESOLUTION, BOUNDARY_DEPTH)
            features.append({
                "type": "Feature",
                "properties": {
                    "Name":         f"{circuit_name_display} - Terrain",
                    "circuit_id":   circuit_id,
                    "feature_type": "terrain",
                    "extend_km":    terrain_data["extend_km"],
                    "resolution_m": terrain_data["resolution_m"],
                    "n_lat":        terrain_data["n_lat"],
                    "n_lon":        terrain_data["n_lon"],
                    "lon_centre":   terrain_data["lon_centre"],
                    "lat_centre":   terrain_data["lat_centre"],
                    "lon_min":      terrain_data["lon_min"],
                    "lat_min":      terrain_data["lat_min"],
                    "lon_max":      terrain_data["lon_max"],
                    "lat_max":      terrain_data["lat_max"],
                    "ele_min_m":    terrain_data["ele_min_m"],
                    "ele_max_m":    terrain_data["ele_max_m"],
                    "points":       terrain_data["terrain_points"],
                    "triangles":    terrain_tris,
                },
                "geometry": None,
            })
            if boundary_tris:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":         f"{circuit_name_display} - Terrain Boundary",
                        "circuit_id":   circuit_id,
                        "feature_type": "terrain_boundary",
                        "ele_min_m":    terrain_data["ele_min_m"],
                        "triangles":    boundary_tris,
                    },
                    "geometry": None,
                })

        # 5. Structures
        if terrain_data and terrain_data.get("structures"):
            features.append({
                "type": "Feature",
                "properties": {
                    "Name":         f"{circuit_name_display} - Structures",
                    "circuit_id":   circuit_id,
                    "feature_type": "structures",
                    "structures":   terrain_data["structures"],
                },
                "geometry": None,
            })

        # 6. Water bodies
        if terrain_data:
            for body in terrain_data.get("water", []):
                fp = body.get("footprint", [])
                if len(fp) < 3:
                    continue
                ring  = fp if fp[0] == fp[-1] else fp + [fp[0]]
                holes = body.get("holes", [])
                tag   = body.get("water_tag", "water")
                tris  = build_water_hexagons(ring, H3_RESOLUTION, holes=holes)

                # GeoJSON Polygon: first ring = outer, subsequent = holes
                geom_rings = [ring] + [
                    (h if h[0] == h[-1] else h + [h[0]]) for h in holes
                ]
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":         f"{circuit_name_display} - {tag.title()}",
                        "circuit_id":   circuit_id,
                        "feature_type": "water_body",
                        "water_tag":    tag,
                        "ele_m":        body.get("ele_m", 0.0),
                        "osm_id":       body.get("osm_id"),
                        "osm_type":     body.get("osm_type"),
                        "triangles":    tris,
                    },
                    "geometry": {
                        "type":        "Polygon",
                        "coordinates": geom_rings,
                    },
                })

        # 7. Sea — H3 hexagons from coastline polygon(s).
        # sea_rings is the list returned by build_sea_polygons above.
        for sea_idx, sea_ring in enumerate(sea_rings):
            if not sea_ring or len(sea_ring) < 3:
                continue
            tris = build_water_hexagons(sea_ring, H3_RESOLUTION)
            if not tris:
                continue
            geo_ring = sea_ring if sea_ring[0] == sea_ring[-1] else sea_ring + [sea_ring[0]]
            features.append({
                "type": "Feature",
                "properties": {
                    "Name":         f"{circuit_name_display} - Sea",
                    "circuit_id":   circuit_id,
                    "feature_type": "sea",
                    "sea_index":    sea_idx,
                    "triangles":    tris,
                },
                "geometry": {
                    "type":        "Polygon",
                    "coordinates": [geo_ring],
                },
            })
            print(f"  Sea [{sea_idx}]: {len(tris)//6} H3 cells  {len(tris)} tris (coastline-based)")

        n_water = sum(1 for f in features
                      if (f.get("properties") or {}).get("feature_type") == "water_body")
        n_sea   = sum(1 for f in features
                      if (f.get("properties") or {}).get("feature_type") == "sea")
        if n_water or n_sea:
            print(f"  Water: {n_water} bodies, {n_sea} sea polygon(s)")

        # 8. Vegetation
        if terrain_data:
            for veg in terrain_data.get("vegetation", []):
                fp = veg.get("footprint", [])
                if len(fp) < 3:
                    continue
                ring    = fp if fp[0] == fp[-1] else fp + [fp[0]]
                veg_tag = veg.get("veg_tag", "vegetation")
                tris    = build_water_hexagons(ring, H3_RESOLUTION)
                if not tris:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":         f"{circuit_name_display} - {veg_tag.title()}",
                        "circuit_id":   circuit_id,
                        "feature_type": "vegetation",
                        "veg_tag":      veg_tag,
                        "osm_id":       veg.get("osm_id"),
                        "osm_type":     veg.get("osm_type"),
                        "triangles":    tris,
                    },
                    "geometry": {
                        "type":        "Polygon",
                        "coordinates": [ring],
                    },
                })

        # 9. Streets
        if terrain_data:
            for street in terrain_data.get("streets", []):
                nodes = street.get("nodes", [])
                if len(nodes) < 2:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":         f"{circuit_name_display} - {street.get('highway_tag', 'road')}",
                        "circuit_id":   circuit_id,
                        "feature_type": "street",
                        "highway_tag":  street.get("highway_tag", ""),
                        "width_m":      street.get("width_m", 0.01),
                        "osm_id":       street.get("osm_id"),
                    },
                    "geometry": {
                        "type":        "LineString",
                        "coordinates": nodes,
                    },
                })
        consolidated = {"type": "FeatureCollection", "features": features}
        with open(out_path, "w") as f:
            json.dump(consolidated, f, separators=(",", ":"))
        print(f"  Saved → {out_path}  ({len(features)} features)")

        # ── Track elevation source for summary ────────────────────────────────
        ele_src = "unknown"
        def find_src(obj):
            nonlocal ele_src
            t = obj.get("type", "")
            if t == "FeatureCollection":
                for feat in obj.get("features", []): find_src(feat)
            elif t == "Feature":
                ele_src = (obj.get("properties") or {}).get("elevation_source", "unknown")
        find_src(result)

        if "fastf1" in ele_src:
            telemetry_ok.append(circuit_id)
        else:
            srtm_ok.append(circuit_id)

        # Collect circuit feature only for combined file
        for feat in features:
            ft = (feat.get("properties") or {}).get("feature_type", "")
            if ft not in ("terrain", "structures", "water_body", "sea"):
                all_features.append(feat)

        print()

    # ── Write combined file ───────────────────────────────────────────────────
    combined = {"type": "FeatureCollection", "features": all_features}
    with open(COMBINED_OUT, "w") as f:
        json.dump(combined, f, separators=(",", ":"))

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Telemetry aligned : {len(telemetry_ok)}")
    print(f"  SRTM fallback     : {len(srtm_ok)}")
    print(f"  Failed            : {len(failed)}")
    if failed:
        print(f"  Failed circuits   : {', '.join(failed)}")
    print(f"\n  Individual files  → {CIRCUITS_OUT_DIR}/")
    print(f"  Combined file     → {COMBINED_OUT}")


if __name__ == "__main__":
    main()