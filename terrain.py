"""
terrain.py
----------
Terrain grid fetching and caching for the F1 circuit pipeline.
Fetches elevation data from Copernicus DEM 30m (COP30) Cloud Optimised
GeoTIFFs hosted publicly on AWS S3, with Open-Topo-Data as fallback.
No API key required.  Only the pixels covering each circuit bbox are
transferred via HTTP range requests.
"""

import json
import math
import time

import numpy as np
import requests

from config import (
    TERRAIN_EXTEND_KM, TERRAIN_RESOLUTION_M,
    TERRAIN_API, TERRAIN_BATCH, TERRAIN_DELAY,
    OVERPASS_DELAY,
    OVERWRITE_TERRAIN, OVERWRITE_STRUCTURES, OVERWRITE_WATER,
    OVERWRITE_VEGETATION, OVERWRITE_STREETS,
    RAW_DATA_DIR, USE_RAW_CACHE,
)

# ── Copernicus DEM 30m (COP30) COG helpers ───────────────────────────────────

_COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _cop30_tile_url(tile_lat, tile_lon):
    """
    Build the HTTPS URL for a 1°×1° COP30 COG tile given its SW-corner
    integer coordinates.

    tile_lat : int  floor of latitude  (e.g. -38 for the tile covering
                    latitudes -38° to -37°, i.e. 38°S to 37°S)
    tile_lon : int  floor of longitude (e.g. 144 for 144°E–145°E)
    """
    lat_pfx = "N" if tile_lat >= 0 else "S"
    lon_pfx = "E" if tile_lon >= 0 else "W"
    name    = (f"Copernicus_DSM_COG_10_{lat_pfx}{abs(tile_lat):02d}_00_"
               f"{lon_pfx}{abs(tile_lon):03d}_00_DEM")
    return f"{_COP30_BASE}/{name}/{name}.tif"


def _fetch_cop30_cog(circuit_id, lat_min, lon_min, lat_max, lon_max):
    """
    Read Copernicus DEM 30m elevation for the given bbox from AWS-hosted
    Cloud Optimised GeoTIFFs using HTTP range requests.

    No API key required.  No rate limits.  Only the pixels covering the
    bbox are transferred (~50–150 KB per circuit, regardless of tile size).

    A circuit bbox typically falls within a single 1°×1° tile but can
    straddle up to four tiles; all cases are handled via rasterio merge.

    Returns
    -------
    elevations : list[float | None]  flat, row-major, south→north
    n_lat      : int
    n_lon      : int
    """
    import json as _json
    import numpy as _np

    cache_path = RAW_DATA_DIR / "cop30" / f"{circuit_id}.json"
    if USE_RAW_CACHE and cache_path.exists():
        print(f"  [terrain] COP30 (cached) ...", end=" ", flush=True)
        with open(cache_path) as _f:
            _d = _json.load(_f)
        print("OK")
        return _d["elevations"], _d["n_lat"], _d["n_lon"]

    try:
        import rasterio
        from rasterio.merge import merge as _rmerge
    except ImportError:
        raise ImportError("rasterio required. Install: pip install rasterio")

    # Collect the SW-corner integer coordinates of every tile touched by bbox
    tile_lats = range(math.floor(lat_min), math.ceil(lat_max))
    tile_lons = range(math.floor(lon_min), math.ceil(lon_max))
    urls = [_cop30_tile_url(tlat, tlon)
            for tlat in tile_lats
            for tlon in tile_lons]

    n_tiles = len(urls)
    print(f"  [terrain] COP30 ({n_tiles} tile{'s' if n_tiles > 1 else ''}) ...",
          end=" ", flush=True)

    env_opts = dict(
        GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS = "tif",
        GDAL_HTTP_TIMEOUT = "30",
    )

    with rasterio.Env(**env_opts):
        datasets = []
        nodata   = None
        for url in urls:
            try:
                ds     = rasterio.open(url)
                nodata = nodata if nodata is not None else ds.nodata
                datasets.append(ds)
            except Exception:
                pass   # ocean / missing tiles have no file → skip

        if not datasets:
            raise RuntimeError(
                "No COP30 tiles found for this bbox — "
                "check coordinates or try the OpenTopoData fallback")

        try:
            # merge() crops to bounds and mosaics if multiple tiles
            band, _ = _rmerge(
                datasets,
                bounds=(lon_min, lat_min, lon_max, lat_max),
                nodata=nodata,
            )
        finally:
            for ds in datasets:
                ds.close()

    data = band[0].astype(_np.float64)
    if nodata is not None:
        data[data == nodata] = _np.nan

    # GeoTIFF / rasterio convention: row 0 = north.  Flip to south→north.
    data = _np.flipud(data)

    n_lat, n_lon = data.shape
    elevations   = [None if _np.isnan(v) else round(float(v), 2)
                    for v in data.ravel()]

    valid = [e for e in elevations if e is not None]
    print(f"OK  ({n_lat}×{n_lon}, "
          f"{min(valid):.0f}m – {max(valid):.0f}m)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as _f:
        _json.dump({"elevations": elevations, "n_lat": n_lat, "n_lon": n_lon}, _f)

    return elevations, n_lat, n_lon

from structures import fetch_osm_structures
from water      import fetch_osm_water, fetch_osm_coastline
from vegetation import fetch_osm_vegetation
from streets    import fetch_osm_streets

R_EARTH = 6_371_000.0


def _make_bilinear_z(terrain_data):
    """
    Build a vectorised bilinear Z interpolation function from terrain_data.

    Returns bilinear_z(lons, lats) where lons/lats are numpy arrays.
    Also returns a scalar version bilinear_z_scalar(lon, lat) for convenience.
    """
    lon_min   = terrain_data["lon_min"]
    lat_min   = terrain_data["lat_min"]
    lon_max   = terrain_data["lon_max"]
    lat_max   = terrain_data["lat_max"]
    n_lat     = terrain_data["n_lat"]
    n_lon     = terrain_data["n_lon"]
    points    = terrain_data["terrain_points"]
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Build Z grid as numpy array once
    z_grid = np.zeros(n_lat * n_lon, dtype=float)
    for i, pt in enumerate(points):
        v = pt[2] if len(pt) > 2 and pt[2] is not None else 0.0
        z_grid[i] = float(v)
    z_grid = z_grid.reshape(n_lat, n_lon)

    def bilinear_z_vec(lons, lats):
        """Vectorised bilinear interpolation. lons/lats are numpy arrays."""
        if lon_range == 0 or lat_range == 0:
            return np.zeros_like(lons)
        col_f = np.clip((lons - lon_min) / lon_range * (n_lon - 1), 0, n_lon - 1)
        row_f = np.clip((lats - lat_min) / lat_range * (n_lat - 1), 0, n_lat - 1)
        c0 = col_f.astype(int);  c1 = np.minimum(c0 + 1, n_lon - 1)
        r0 = row_f.astype(int);  r1 = np.minimum(r0 + 1, n_lat - 1)
        fc = col_f - c0;          fr = row_f - r0
        return (z_grid[r0, c0] * (1 - fc) * (1 - fr) +
                z_grid[r0, c1] * fc       * (1 - fr) +
                z_grid[r1, c0] * (1 - fc) * fr       +
                z_grid[r1, c1] * fc       * fr)

    def bilinear_z_scalar(lon, lat):
        return float(bilinear_z_vec(np.array([lon]), np.array([lat]))[0])

    return bilinear_z_vec, bilinear_z_scalar


def _cells_to_triangles(cells, bilinear_z_vec, bilinear_z_scalar, h3,
                         clamp_lon_min=None, clamp_lon_max=None,
                         clamp_lat_min=None, clamp_lat_max=None,
                         include_cell_id=False):
    """
    Convert a set of H3 cells to triangles with bilinear Z interpolation.

    If clamp_* bounds are provided, cell centroids are clamped to those
    bounds before sampling Z (used for boundary ring cells outside the terrain).

    If include_cell_id is True each triangle vertex gains a 4th element: the
    sequential 0-based index of its H3 cell within this call.  All three
    vertices of a triangle carry the same index (they all belong to the same
    cell).  Used downstream to group vertices by cell for cliff detection.

    Returns list of [[lon,lat,z],[lon,lat,z],[lon,lat,z]]
            or      [[lon,lat,z,cell_idx], ...] when include_cell_id=True.
    """
    if not cells:
        return []

    # Collect all points in bulk for vectorised Z lookup
    cell_list    = list(cells)
    centres      = [h3.cell_to_latlng(c)    for c in cell_list]   # (lat, lon)
    boundaries   = [h3.cell_to_boundary(c)  for c in cell_list]   # list of (lat,lon)

    # Build flat arrays of all lons/lats to interpolate
    all_lons, all_lats, point_map = [], [], []
    for i, (clat, clon) in enumerate(centres):
        c_lon = clon; c_lat = clat
        if clamp_lon_min is not None:
            c_lon = max(clamp_lon_min, min(clamp_lon_max, c_lon))
            c_lat = max(clamp_lat_min, min(clamp_lat_max, c_lat))
        all_lons.append(c_lon); all_lats.append(c_lat)
        point_map.append(('c', i))
        for blat, blon in boundaries[i]:
            bc_lon = blon; bc_lat = blat
            if clamp_lon_min is not None:
                bc_lon = max(clamp_lon_min, min(clamp_lon_max, bc_lon))
                bc_lat = max(clamp_lat_min, min(clamp_lat_max, bc_lat))
            all_lons.append(bc_lon); all_lats.append(bc_lat)
            point_map.append(('b', i))

    z_vals = bilinear_z_vec(np.array(all_lons), np.array(all_lats))

    # Reconstruct per-cell
    idx        = 0
    triangles  = []
    for i, cell in enumerate(cell_list):
        # Centre
        clat, clon = centres[i]
        cz         = float(z_vals[idx]); idx += 1
        centre     = [round(clon, 8), round(clat, 8), round(cz, 4)]
        if include_cell_id:
            centre = centre + [i]

        bverts = []
        for blat, blon in boundaries[i]:
            bz = float(z_vals[idx]); idx += 1
            bv = [round(blon, 8), round(blat, 8), round(bz, 4)]
            if include_cell_id:
                bv = bv + [i]
            bverts.append(bv)

        n = len(bverts)
        for j in range(n):
            triangles.append([centre, bverts[j], bverts[(j + 1) % n]])

    return triangles




def fetch_terrain_grid(circuit_id, circuit_coords, out_dir):
    """
    Fetch a terrain elevation grid for a circuit.

    The bbox is computed from the Shapely bounding box of the circuit
    linestring, extended by TERRAIN_EXTEND_KM (1km) in each direction.

    circuit_coords : list of [lon, lat, ...] coordinate pairs
    """
    out_path = out_dir / f"{circuit_id}_terrain.json"

    file_exists       = out_path.exists()
    need_terrain      = not file_exists or OVERWRITE_TERRAIN
    need_structures   = not file_exists or OVERWRITE_STRUCTURES
    need_water        = not file_exists or OVERWRITE_WATER
    need_vegetation   = not file_exists or OVERWRITE_VEGETATION
    need_streets      = not file_exists or OVERWRITE_STREETS

    if not need_terrain and not need_structures and not need_water and not need_vegetation and not need_streets:
        print(f"  [terrain] already exists — loading from cache")
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [terrain] cache load failed: {e}")
            return None

    existing_structures       = []
    existing_water            = []
    existing_sea              = None
    existing_sea_had_reversals = None
    existing_vegetation       = []
    existing_streets          = []
    if file_exists and (not need_structures or not need_water or not need_vegetation or not need_streets):
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if not need_structures:
                existing_structures = cached.get("structures", [])
                print(f"  [structures] keeping existing "
                      f"({len(existing_structures)} structures)")
            if not need_water:
                existing_water             = cached.get("water", [])
                existing_sea               = cached.get("sea")
                existing_sea_had_reversals = cached.get("sea_had_reversals")
                print(f"  [water] keeping existing "
                      f"({len(existing_water)} bodies)")
            if not need_vegetation:
                existing_vegetation = cached.get("vegetation", [])
                print(f"  [vegetation] keeping existing "
                      f"({len(existing_vegetation)} polygons)")
            if not need_streets:
                existing_streets = cached.get("streets", [])
                print(f"  [streets] keeping existing "
                      f"({len(existing_streets)} ways)")
        except Exception as e:
            print(f"  [terrain] could not load existing sidecar: {e}")

    # ── Compute bbox from circuit coords extended by TERRAIN_EXTEND_KM ───
    lons       = [c[0] for c in circuit_coords]
    lats       = [c[1] for c in circuit_coords]
    lat_centre = (min(lats) + max(lats)) / 2.0
    lon_centre = (min(lons) + max(lons)) / 2.0

    ext_m   = TERRAIN_EXTEND_KM * 1000.0
    lat_rad = math.radians(lat_centre)
    d_lat   = math.degrees(ext_m / R_EARTH)
    d_lon   = math.degrees(ext_m / (R_EARTH * math.cos(lat_rad)))

    lat_min = min(lats) - d_lat;  lat_max = max(lats) + d_lat
    lon_min = min(lons) - d_lon;  lon_max = max(lons) + d_lon

    lat_extent_m = (lat_max - lat_min) * R_EARTH * math.pi / 180.0
    lon_extent_m = (lon_max - lon_min) * R_EARTH * math.pi / 180.0 * math.cos(lat_rad)
    n_lat = max(2, int(round(lat_extent_m / TERRAIN_RESOLUTION_M)) + 1)
    n_lon = max(2, int(round(lon_extent_m / TERRAIN_RESOLUTION_M)) + 1)

    # ── Elevation grid ────────────────────────────────────────────────────────
    if need_terrain:
        # ── Primary: Copernicus DEM 30m COG (AWS S3, HTTP range requests) ────
        cop30_ok = False
        try:
            elevations, n_lat, n_lon = _fetch_cop30_cog(
                circuit_id, lat_min, lon_min, lat_max, lon_max)
            valid    = [e for e in elevations if e is not None]
            ele_min  = min(valid) if valid else 0.0
            ele_max  = max(valid) if valid else 0.0
            cop30_ok = True
        except Exception as _e:
            print(f"\n  [terrain] COP30 failed ({_e}) — falling back to OpenTopoData")

        # ── Fallback: OpenTopoData batch queries ─────────────────────────────
        if not cop30_ok:
            grid_points = []
            for row in range(n_lat):
                lat = lat_min + row * (lat_max - lat_min) / (n_lat - 1)
                for col in range(n_lon):
                    lon = lon_min + col * (lon_max - lon_min) / (n_lon - 1)
                    grid_points.append((lat, lon))

            total     = len(grid_points)
            n_batches = math.ceil(total / TERRAIN_BATCH)
            elevations = []

            print(f"  [terrain] {n_lat}×{n_lon} grid = {total} pts, "
                  f"{n_batches} batches (OpenTopoData) ...")

            for b in range(n_batches):
                start   = b * TERRAIN_BATCH
                end     = min(start + TERRAIN_BATCH, total)
                batch   = grid_points[start:end]
                loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
                try:
                    resp = requests.get(
                        TERRAIN_API, params={"locations": loc_str}, timeout=30)
                    resp.raise_for_status()
                    elevations.extend(
                        r.get("elevation")
                        for r in resp.json().get("results", []))
                except Exception as e:
                    print(f"\n    batch {b+1}/{n_batches} failed: {e}")
                    elevations.extend([None] * len(batch))

                if b % 50 == 49 or b == n_batches - 1:
                    print(f"    {b+1}/{n_batches} batches done ...")
                if b < n_batches - 1:
                    time.sleep(TERRAIN_DELAY)

            while len(elevations) < total:
                elevations.append(None)

            valid   = [e for e in elevations if e is not None]
            ele_min = min(valid) if valid else 0.0
            ele_max = max(valid) if valid else 0.0

    else:
        print(f"  [terrain] elevation exists — loading for update")
        with open(out_path) as f:
            existing   = json.load(f)
        elevations = existing["elevations"]
        n_lat      = existing["n_lat"]    # use actual stored dimensions — COP30
        n_lon      = existing["n_lon"]    # may differ from the formula estimate
        ele_min    = existing["ele_min_m"]
        ele_max    = existing["ele_max_m"]

    # ── Structures ────────────────────────────────────────────────────────────
    if need_structures:
        print(f"  [structures] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        structures = fetch_osm_structures(circuit_id, lat_min, lon_min, lat_max, lon_max)
        print(f"OK ({len(structures)} structures)")
    else:
        structures = existing_structures

    # ── Water bodies ──────────────────────────────────────────────────────────
    if need_water:
        print(f"  [water] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        water = fetch_osm_water(circuit_id, lat_min, lon_min, lat_max, lon_max,
                                elevations, n_lat, n_lon)
        print(f"OK ({len(water)} bodies)")
    else:
        water = existing_water

    # ── Vegetation ────────────────────────────────────────────────────────────
    if need_vegetation:
        print(f"  [vegetation] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        vegetation = fetch_osm_vegetation(circuit_id, lat_min, lon_min, lat_max, lon_max)
        print(f"OK ({len(vegetation)} polygons)")
    else:
        vegetation = existing_vegetation

    # ── Streets ───────────────────────────────────────────────────────────────
    if need_streets:
        print(f"  [streets] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY * 4)   # longer delay — streets is a large query
        streets = fetch_osm_streets(circuit_id, lat_min, lon_min, lat_max, lon_max)
    else:
        streets = existing_streets

    # ── Coastline ─────────────────────────────────────────────────────────────
    need_sea = not file_exists or OVERWRITE_WATER

    if need_sea:
        print(f"  [sea] querying OSM coastline ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        sea_footprint, sea_had_reversals = fetch_osm_coastline(
            circuit_id, lat_min, lon_min, lat_max, lon_max)
        print(f"OK ({len(sea_footprint)} pts)" if sea_footprint else "none (landlocked)")
    else:
        sea_footprint     = existing_sea
        sea_had_reversals = existing_sea_had_reversals
        if sea_footprint:
            print(f"  [sea] kept existing ({len(sea_footprint)} pts)")

    # ── Terrain points (no sea depression here — baked by master.py) ─────────
    terrain_points = []
    for row in range(n_lat):
        lat = lat_min + row * (lat_max - lat_min) / max(n_lat - 1, 1)
        for col in range(n_lon):
            lon = lon_min + col * (lon_max - lon_min) / max(n_lon - 1, 1)
            idx = row * n_lon + col
            ele = elevations[idx] if idx < len(elevations) else None
            terrain_points.append([
                round(lon, 6), round(lat, 6),
                round(ele, 2) if ele is not None else None,
            ])

    terrain_data = {
        "circuit_id":     circuit_id,
        "extend_km":      TERRAIN_EXTEND_KM,
        "resolution_m":   TERRAIN_RESOLUTION_M,
        "n_lat":          n_lat,
        "n_lon":          n_lon,
        "lon_centre":     lon_centre,
        "lat_centre":     lat_centre,
        "lon_min":        lon_min,
        "lat_min":        lat_min,
        "lon_max":        lon_max,
        "lat_max":        lat_max,
        "ele_min_m":      round(ele_min, 2),
        "ele_max_m":      round(ele_max, 2),
        "elevations":     elevations,
        "structures":     structures,
        "water":          water,
        "vegetation":     vegetation,
        "streets":        streets,
        "sea":                sea_footprint,
        "sea_had_reversals":  sea_had_reversals,
        "terrain_points":     terrain_points,
    }

    with open(out_path, "w") as f:
        json.dump(terrain_data, f, separators=(",", ":"))

    print(f"  [terrain] saved → {out_path}  "
          f"({ele_min:.0f}m–{ele_max:.0f}m, {len(structures)} structures)")
    return terrain_data


def build_terrain_hexagons(terrain_data, resolution):
    """
    Generate H3 hexagonal triangles covering the terrain bbox.
    Z values are bilinearly interpolated from the terrain grid.
    Returns list of [[lon,lat,z],[lon,lat,z],[lon,lat,z]].
    """
    try:
        import h3
    except ImportError:
        print("  [h3] not installed — skipping terrain hexagon generation")
        return []

    bilinear_z_vec, _ = _make_bilinear_z(terrain_data)

    lon_min = terrain_data["lon_min"]; lon_max = terrain_data["lon_max"]
    lat_min = terrain_data["lat_min"]; lat_max = terrain_data["lat_max"]

    bbox_poly = h3.LatLngPoly([
        (lat_min, lon_min), (lat_min, lon_max),
        (lat_max, lon_max), (lat_max, lon_min),
    ])
    try:
        cells = h3.h3shape_to_cells(bbox_poly, resolution)
    except Exception as e:
        print(f"  [h3 terrain] polyfill failed: {e}")
        return []

    triangles = _cells_to_triangles(cells, bilinear_z_vec, None, h3,
                                     include_cell_id=True)
    print(f"  [h3 terrain] {len(cells)} cells → {len(triangles)} triangles "
          f"at resolution {resolution}")
    return triangles


def build_terrain_boundary_hexagons(terrain_data, resolution, depth=2):
    """
    Generate a ring of H3 hexagonal triangles outside the terrain bbox.
    Z values are sampled from the nearest terrain edge point.
    Returns list of [[lon,lat,z],[lon,lat,z],[lon,lat,z]].
    """
    try:
        import h3
    except ImportError:
        print("  [h3 boundary] not installed -- skipping")
        return []

    bilinear_z_vec, _ = _make_bilinear_z(terrain_data)

    lon_min = terrain_data["lon_min"]; lon_max = terrain_data["lon_max"]
    lat_min = terrain_data["lat_min"]; lat_max = terrain_data["lat_max"]

    avg_edge_m = 6 * 110_000 / (2 ** (resolution + 1))
    inset_deg  = math.degrees(avg_edge_m * depth / R_EARTH)

    outer_poly = h3.LatLngPoly([
        (lat_min - inset_deg, lon_min - inset_deg),
        (lat_min - inset_deg, lon_max + inset_deg),
        (lat_max + inset_deg, lon_max + inset_deg),
        (lat_max + inset_deg, lon_min - inset_deg),
    ])
    inner_poly = h3.LatLngPoly([
        (lat_min, lon_min), (lat_min, lon_max),
        (lat_max, lon_max), (lat_max, lon_min),
    ])

    try:
        outer_cells = h3.h3shape_to_cells(outer_poly, resolution)
        inner_cells = h3.h3shape_to_cells(inner_poly, resolution)
    except Exception as e:
        print(f"  [h3 boundary] polyfill failed: {e}")
        return []

    ring_cells = set(outer_cells) - set(inner_cells)
    if not ring_cells:
        print(f"  [h3 boundary] no ring cells found")
        return []

    triangles = _cells_to_triangles(
        ring_cells, bilinear_z_vec, None, h3,
        clamp_lon_min=lon_min, clamp_lon_max=lon_max,
        clamp_lat_min=lat_min, clamp_lat_max=lat_max)

    print(f"  [h3 boundary] {len(ring_cells)} ring cells -> "
          f"{len(triangles)} triangles at resolution {resolution}")
    return triangles