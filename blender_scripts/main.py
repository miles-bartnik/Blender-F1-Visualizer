"""
import_circuits.py
--------------------
Blender Python script — run from the Text Editor or Scripting workspace.

Reads all F1 circuit GeoJSON files produced by f1_circuits.py and creates
one Curve object per circuit in the current scene.

Elevation is read directly from the third coordinate element [lon, lat, ele]
of each GeoJSON point — no sidecar files needed. Both telemetry-aligned and
SRTM circuits use the same path.

For SRTM circuits (elevation_source == "srtm" in GeoJSON properties),
optional spike rejection and smoothing are applied. Telemetry-aligned
circuits are already clean and bypass smoothing.

End-gap blending closes the elevation discontinuity at the start/finish
join for all circuits.

Usage:
  1. Run setup_environment.py once and restart Blender if packages were installed
  2. Run f1_circuits.py to generate GeoJSON files
  3. Open Blender → Scripting workspace
  4. Open this file, set GEOJSON_DIR, run
"""

import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    print("[import] numpy not available — run setup_environment.py first")

try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import CubicSpline
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY    = False
    savgol_filter = None
    CubicSpline   = None
    print("[import] scipy not available — run setup_environment.py first")

try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY   = False
    Point          = None
    ShapelyPolygon = None
    print("[import] shapely not available — run setup_environment.py first")



# =============================================================================
# CONFIGURATION
# =============================================================================

GEOJSON_DIR = r"C:\Users\Miles\PycharmProjects\F1\output\circuits"

# Restrict import to a specific list of circuit IDs.
# Set to None or [] to import all circuits in GEOJSON_DIR.
#
# Available circuit IDs:
#   Current calendar (2025):
#     au-1953   Australia — Albert Park
#     cn-2004   China — Shanghai
#     jp-1962   Japan — Suzuka
#     bh-2002   Bahrain — Sakhir
#     sa-2021   Saudi Arabia — Jeddah
#     us-2022   Miami
#     it-1953   Emilia-Romagna — Imola
#     mc-1929   Monaco
#     es-1991   Spain — Barcelona
#     ca-1978   Canada — Montreal
#     at-1969   Austria — Red Bull Ring
#     gb-1948   Great Britain — Silverstone
#     be-1925   Belgium — Spa
#     hu-1986   Hungary — Hungaroring
#     nl-1948   Netherlands — Zandvoort
#     it-1922   Italy — Monza
#     az-2016   Azerbaijan — Baku
#     sg-2008   Singapore — Marina Bay
#     us-2012   USA — COTA
#     mx-1962   Mexico City
#     br-1940   Brazil — Interlagos
#     us-2023   Las Vegas
#     qa-2004   Qatar — Lusail
#     ae-2009   Abu Dhabi — Yas Marina
#   Retired circuits:
#     de-1927   Nürburgring
#     de-1932   Hockenheim
#     fr-1969   Paul Ricard
#     pt-2008   Portimão
#     tr-2005   Istanbul
#     ru-2014   Sochi
#     it-1914   Mugello
#     my-1999   Malaysia — Sepang
#     fr-1960   Magny-Cours
#     pt-1972   Estoril
#     ar-1952   Buenos Aires
#     za-1961   Kyalami
#     us-1909   Indianapolis
#     us-1956   Watkins Glen
#     br-1977   Jacarepaguá
CIRCUITS_TO_IMPORT = ["mc-1929"]   # None = all circuits

CIRCUITS_EXCLUDE = ["es-2008"]

# To import specific circuits only, set CIRCUITS_TO_IMPORT = ["au-1953", ...]

# Scale: degrees -> Blender units (~1 unit per km at mid-latitudes)
SCALE = 100.0

# Metres per degree of latitude (equatorial approximation).
METRES_PER_DEGREE = 111_320.0

# Z scale: converts elevation in metres to Blender units.
# Set equal to SCALE / METRES_PER_DEGREE so vertical distances are
# proportional to horizontal distances — 1m climb = same visual size
# as 1m of track length.
Z_SCALE = SCALE / METRES_PER_DEGREE  # ~0.000898

# Resampling resolution in metres. Curves are resampled to one point
# every POINT_RESOLUTION_M metres before NURBS smoothing is applied.
# With NURBS, 5m gives smooth results without excessive point counts.
POINT_RESOLUTION_M = 10.0
WATER_RESOLUTION   = 50.0   # subdivision resolution for water surfaces (metres)
MIN_WATER_HEX      = 3      # minimum H3 cells a water body must have to be imported
SEA_STD_THRESHOLD  = 0.001  # Z std dev below which a vertex group is considered sea

# ── Vegetation ────────────────────────────────────────────────────────────────
# Minimum triangle count for a vegetation polygon to be imported.
# 18 triangles = 3 H3 hexes ≈ 60 m² — filters single-hex noise patches.
VEG_MIN_TRIS = 18

# Per-tag density triplets fed into the 'Vegetation Scatter' GN modifier.
# Tuple order: (tree_density, bush_density, grass_density)
# Values are normalised 0–1; the GN controls actual point counts.
VEG_DENSITIES = {
    "wood":   (1.0, 0.0, 0.0),
    "forest": (1.0, 0.0, 0.0),
    "scrub":  (0.0, 1.0, 0.0),
    "heath":  (0.0, 1.0, 0.0),
    "grass":  (0.0, 0.0, 1.0),
    "meadow": (0.0, 0.0, 1.0),
    "park":   (0.0, 0.0, 1.0),
    "garden": (0.0, 1.0, 0.0),
}
VEG_DENSITY_DEFAULT = (0.0, 0.0, 1.0)  # fallback for unmapped tags

# Water depth model — proportional to max inscribed radius (pole of inaccessibility).
# depth_m = min(tag_cap, WATER_DEPTH_PROPORTIONAL_K * max_dist_m)
# where max_dist_m is the maximum distance from any interior point to the shoreline.
# This is geometrically equivalent to Håkanson for circular bodies, but correctly
# gives shallower depths for elongated or irregular polygons.
WATER_DEPTH_PROPORTIONAL_K = 0.1   # 10 % of inscribed radius → max depth

# Maximum depression depth (metres) per OSM water_tag.
# Lookup order matches primary_tag() in water.py: water > natural > landuse > waterway.
WATER_DEPTH_BY_TAG = {
    # ── Open sea / coast ───────────────────────────────────────────────────────
    "sea":          50.0,
    "ocean":        50.0,
    # ── Lakes and generic open water ──────────────────────────────────────────
    "lake":          8.0,
    "water":         5.0,   # natural=water without a sub-type
    "pond":          1.5,
    "oxbow":         2.0,
    "lagoon":        6.0,
    "moat":          1.5,
    # ── Managed / artificial ──────────────────────────────────────────────────
    "reservoir":     8.0,
    "basin":         2.0,   # retention / drainage basin — intentionally shallow
    # ── Wetland ───────────────────────────────────────────────────────────────
    "wetland":       1.0,
    # ── Rivers and waterways ──────────────────────────────────────────────────
    "river":         4.0,
    "riverbank":     3.0,
    "canal":         2.0,
    "dock":          8.0,
    "harbour":      10.0,   # landuse=harbour — open harbour basins (e.g. Port Hercule)
}
WATER_DEPTH_DEFAULT_M = 5.0   # fallback for unmapped tags

# Fraction of the water body's half-width that is the cosine edge-blend zone.
# 0.0 = sharp cliff (no taper).  1.0 = full S-curve taper (old behaviour).
# 0.25 → outer 25% tapers, inner 75% is at full depth (realistic bowl/shelf).
WATER_EDGE_BLEND = 0.25

# 'POLY' = exact straight segments  |  'NURBS' = smooth interpolation
CURVE_TYPE = 'NURBS'

# Bevel depth for tube appearance (0 = plain curve)
BEVEL_DEPTH = 0

# True = circuits at real-world projected positions
# False = all stacked at origin for per-circuit inspection
USE_REAL_WORLD_POSITION = False

# ---------------------------------------------------------------------------
# SRTM smoothing — only applied to circuits where elevation_source == "srtm"
# Telemetry-aligned circuits are already clean and skip this entirely
# ---------------------------------------------------------------------------

SPIKE_THRESHOLD = 15.0   # metres — 0 to disable
SMOOTH_WINDOW   = 11     # points — 0 to disable

# Smoothing window applied to ALL circuits after elevation is read.
# Uses Savitzky-Golay filtering (scipy) if available — preserves peaks
# and valleys far better than a moving average at the same window size.
# Falls back to moving average if scipy is not installed in Blender.
# Window must be odd and larger than SAVGOL_POLY_ORDER.
ELEVATION_SMOOTH_WINDOW = 31   # increase for smoother, decrease for more detail
SAVGOL_POLY_ORDER       = 3    # polynomial order — 3 is a good default

# ---------------------------------------------------------------------------
# End-gap blending — applied to all circuits
# ---------------------------------------------------------------------------

BLEND_FRACTION = 0.05    # fraction of points at each end — 0 to disable

# Track width — used when TUMFTM data is unavailable for a circuit
DEFAULT_WIDTH_M = 15.0

# Track Profile geometry nodes — profile ribbon dimensions
# X is driven per-point by the curve radius (track width from TUMFTM)
# Y is fixed — gives the ribbon a small physical thickness
PROFILE_Y_M = 0.1

# If True, delete any existing parent empty matching this circuit and all
# its children before importing. Prevents duplicate objects on re-runs.
# Looks up by blender_name (circuit display name) and circuit_id as fallback.
CLEAR_EXISTING = True


# =============================================================================
# PROJECTION
# =============================================================================

# Cached cos(lat_origin) per circuit — recomputed once per circuit in main
# and stored here so equirectangular doesn't call trig on every point.
_cos_lat_cache = {}

def equirectangular(lon, lat, lon_origin, lat_origin):
    cos_lat = _cos_lat_cache.get(lat_origin)
    if cos_lat is None:
        cos_lat = math.cos(math.radians(lat_origin))
        _cos_lat_cache[lat_origin] = cos_lat
    x = (lon - lon_origin) * cos_lat * SCALE
    y = (lat - lat_origin) * SCALE
    return x, y


def water_edge_falloff(t):
    """
    Displacement factor for water body depth at normalised distance t,
    where t=0 is the shoreline and t=1 is the deepest interior point.

    The outer WATER_EDGE_BLEND fraction of the width uses a cosine taper
    (0 → 1).  Everything inward of that zone is at full depth (factor = 1).

    Shape for WATER_EDGE_BLEND=0.25:
      t  0.00 → factor 0.00  (shoreline — no displacement)
      t  0.12 → factor 0.50  (mid-taper)
      t  0.25 → factor 1.00  (inner edge of taper — full depth)
      t  1.00 → factor 1.00  (centre — full depth, unchanged)
    """
    if WATER_EDGE_BLEND <= 0.0 or t >= WATER_EDGE_BLEND:
        return 1.0
    tt = t / WATER_EDGE_BLEND          # 0→1 across the edge zone only
    return 0.5 * (1.0 - math.cos(math.pi * tt))


def _make_water_depth_offset_fn(ring, water_tag, lon_origin, lat_origin, bbox=None):
    """
    Build a depth-offset function from the water polygon geometry.

    Maximum depth is computed from the polygon's maximum inscribed radius
    (pole of inaccessibility approximated by sampling the interior):

        depth_m = min(tag_cap, WATER_DEPTH_PROPORTIONAL_K * max_dist_m)

    where max_dist_m is the furthest distance from any sampled interior point
    to the natural shoreline.  This replaces the Hakanson area-based formula
    and correctly gives shallower depths for elongated or irregular polygons.

    Depth at each terrain vertex scales linearly with its distance from the
    natural shoreline, normalised by max_dist (cosine taper applied within
    the outer WATER_EDGE_BLEND fraction of the width).

    bbox : optional (lon_min, lat_min, lon_max, lat_max). Ring vertices on a
        bbox edge are excluded from the KD-tree so that water bodies clipped
        to the terrain boundary do not taper near the bbox edge — only the
        natural shoreline drives the falloff.
    """
    if not HAVE_SHAPELY or not ring or len(ring) < 3:
        return None

    try:
        import numpy as _np
        from scipy.spatial import cKDTree as _KD
    except ImportError:
        return None   # scipy required

    cos_lat = _cos_lat_cache.get(lat_origin) or math.cos(math.radians(lat_origin))

    try:
        poly = ShapelyPolygon([(p[0], p[1]) for p in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
    except Exception:
        return None

    # ── Ring vertices for containment (full ring, lon/lat) ───────────────────
    _ring_x = [p[0] for p in ring]
    _ring_y = [p[1] for p in ring]

    # ── Expanded containment polygon (smooth coastline boundary) ─────────────
    # Expand by half an H3 res-9 cell (~111 m = 0.001 deg) so that terrain
    # vertices just outside the jagged H3 hex boundary are treated as inside.
    # Their shore distance is tiny, so depth_fn still returns ≈ 0 there —
    # the depth gradient is now continuous across the coastline edge.
    _CONT_BUFFER = 0.001   # degrees, ~111 m
    try:
        _poly_cont = poly.buffer(_CONT_BUFFER)
        if hasattr(_poly_cont, 'exterior') and not _poly_cont.is_empty:
            _cc = list(_poly_cont.exterior.coords)
            _cont_ring_x = [c[0] for c in _cc]
            _cont_ring_y = [c[1] for c in _cc]
        else:
            _cont_ring_x = _ring_x
            _cont_ring_y = _ring_y
    except Exception:
        _cont_ring_x = _ring_x
        _cont_ring_y = _ring_y

    # ── Shoreline KD-tree (natural boundary only, Blender XY) ────────────────
    # Exclude vertices on the bbox edge so depth doesn't taper to 0 there.
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        _tol = 1e-6
        natural = [
            (p[0], p[1]) for p in ring
            if not (abs(p[0] - lon_min) < _tol or abs(p[0] - lon_max) < _tol or
                    abs(p[1] - lat_min) < _tol or abs(p[1] - lat_max) < _tol)
        ]
        shore_pts = natural if len(natural) >= 2 else [(p[0], p[1]) for p in ring]
    else:
        shore_pts = [(p[0], p[1]) for p in ring]

    shore_bl = _np.array([
        ((lon - lon_origin) * cos_lat * SCALE,
         (lat - lat_origin) * SCALE)
        for lon, lat in shore_pts
    ])
    if len(shore_bl) == 0:
        return None

    _shore_tree = _KD(shore_bl)

    # ── max_dist: maximum inscribed radius (Blender units) ───────────────────
    # Approximate the pole of inaccessibility by sampling a 15×15 grid of
    # points inside the polygon bounding box, keeping only those inside the
    # polygon, and taking the furthest distance to the natural shoreline.
    # This gives a far better estimate than centroid-to-boundary for concave
    # or elongated polygons like bays and river channels.
    minx, miny, maxx, maxy = poly.bounds
    _gx = _np.linspace(minx, maxx, 15)
    _gy = _np.linspace(miny, maxy, 15)
    _gxx, _gyy = _np.meshgrid(_gx, _gy)
    _grid_lonlat = _np.column_stack([_gxx.ravel(), _gyy.ravel()])

    # Ray-cast containment for the grid
    _inside_mask = _np.zeros(len(_grid_lonlat), dtype=bool)
    _n = len(_ring_x)
    for _i in range(_n):
        _j = (_i + 1) % _n
        _xi, _yi = _ring_x[_i], _ring_y[_i]
        _xj, _yj = _ring_x[_j], _ring_y[_j]
        _cond = ((_yi > _grid_lonlat[:, 1]) != (_yj > _grid_lonlat[:, 1])) & (
            _grid_lonlat[:, 0] < (_xj - _xi) * (_grid_lonlat[:, 1] - _yi)
            / ((_yj - _yi) + 1e-12) + _xi
        )
        _inside_mask ^= _cond

    _interior = _grid_lonlat[_inside_mask]
    if len(_interior) == 0:
        # Fallback to centroid
        _interior = _np.array([[poly.centroid.x, poly.centroid.y]])

    _interior_bl = _np.column_stack([
        (_interior[:, 0] - lon_origin) * cos_lat * SCALE,
        (_interior[:, 1] - lat_origin) * SCALE,
    ])
    _d_grid, _ = _shore_tree.query(_interior_bl)
    max_dist_bl = float(_d_grid.max())

    if max_dist_bl < 1e-12:
        return None

    # Convert max_dist to metres and compute depth_m
    max_dist_m   = max_dist_bl * METRES_PER_DEGREE / SCALE
    depth_cap    = WATER_DEPTH_BY_TAG.get(water_tag, WATER_DEPTH_DEFAULT_M)
    depth_m      = min(depth_cap, WATER_DEPTH_PROPORTIONAL_K * max_dist_m)
    print(f"    depth [{water_tag}]  "
          f"max_dist={max_dist_m:.0f} m  "
          f"raw={WATER_DEPTH_PROPORTIONAL_K * max_dist_m:.1f} m  "
          f"cap={depth_cap} m  -> {depth_m:.1f} m")

    if depth_m <= 0:
        return None

    # ── Vectorised ray-cast containment check (uses expanded polygon) ────────
    def _contains_batch(lons, lats):
        inside = _np.zeros(len(lons), dtype=bool)
        n = len(_cont_ring_x)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = _cont_ring_x[i], _cont_ring_y[i]
            xj, yj = _cont_ring_x[j], _cont_ring_y[j]
            cond = ((yi > lats) != (yj > lats)) & (
                lons < (xj - xi) * (lats - yi) / ((yj - yi) + 1e-12) + xi
            )
            inside ^= cond
        return inside

    _blend = WATER_EDGE_BLEND

    # ── Scalar depth function (uses expanded containment polygon) ────────────
    def depth_fn(px, py):
        lon = px / (cos_lat * SCALE) + lon_origin
        lat = py / SCALE + lat_origin
        inside = False
        n = len(_cont_ring_x)
        for k in range(n):
            j = (k + 1) % n
            xi, yi = _cont_ring_x[k], _cont_ring_y[k]
            xj, yj = _cont_ring_x[j], _cont_ring_y[j]
            if ((yi > lat) != (yj > lat) and
                    lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-12) + xi):
                inside = not inside
        if not inside:
            return 0.0
        d, _ = _shore_tree.query((px, py))
        t = min(d / max_dist_bl, 1.0)
        return depth_m * water_edge_falloff(t) * Z_SCALE

    # ── Batch (numpy) depth function ─────────────────────────────────────────
    def depth_fn_batch(xs, ys):
        d_approx, _ = _shore_tree.query(_np.column_stack([xs, ys]))
        t = _np.minimum(d_approx / max_dist_bl, 1.0)

        if _blend <= 0.0:
            factors = _np.ones_like(t)
        else:
            factors = _np.where(
                t >= _blend,
                1.0,
                0.5 * (1.0 - _np.cos(_np.pi * t / _blend))
            )

        result = depth_m * factors * Z_SCALE

        lons = xs / (cos_lat * SCALE) + lon_origin
        lats = ys / SCALE + lat_origin
        result[~_contains_batch(lons, lats)] = 0.0

        return result

    depth_fn.batch = depth_fn_batch
    return depth_fn


def project_water_mesh_to_terrain(water_obj, terrain_z_fn, target_z=None):
    """
    Project each water body mesh vertex to the terrain surface Z.

    If target_z is given, all vertices are set to that fixed value (e.g. 0.0
    for sea, which sits at sea level regardless of terrain).
    Otherwise each vertex gets its own Z from terrain_z_fn(x, y) — the same
    per-point lookup used by project_spline_to_terrain for circuits and roads.
    """
    if water_obj is None:
        return
    mesh = water_obj.data
    if not mesh.vertices:
        return

    if target_z is not None:
        for v in mesh.vertices:
            v.co.z = target_z
    elif terrain_z_fn is None:
        return
    elif HAVE_NUMPY:
        import numpy as _np
        xy    = _np.array([(v.co.x, v.co.y) for v in mesh.vertices])
        batch = getattr(terrain_z_fn, 'batch', None)
        zs    = batch(xy[:, 0], xy[:, 1]) if batch is not None else \
                _np.array([terrain_z_fn(x, y) for x, y in xy])
        for v, z in zip(mesh.vertices, zs.tolist()):
            v.co.z = z
    else:
        for v in mesh.vertices:
            v.co.z = terrain_z_fn(v.co.x, v.co.y)

    mesh.update()



# =============================================================================
# GEOJSON HELPERS
# =============================================================================

def extract_linestrings(geojson):
    results = []
    def walk(obj):
        t = obj.get("type", "")
        if t == "FeatureCollection":
            for f in obj.get("features", []): walk(f)
        elif t == "Feature":
            walk(obj.get("geometry") or {})
        elif t == "LineString":
            results.append(obj["coordinates"])
        elif t == "MultiLineString":
            for line in obj["coordinates"]: results.append(line)
    walk(geojson)
    return results


def centroid(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def find_feature_properties(geojson):
    name = ""
    location = ""
    props = {}
    elevation_source = "srtm"
    def walk(obj):
        nonlocal name, location, props, elevation_source
        t = obj.get("type", "")
        if t == "FeatureCollection":
            for f in obj.get("features", []): walk(f)
        elif t == "Feature":
            p = obj.get("properties") or {}
            if not props:
                props = p
            if not name and p.get("Name"):
                name = p["Name"]
            if not location and p.get("Location"):
                location = p["Location"]
            if p.get("elevation_source"):
                elevation_source = p["elevation_source"]
    walk(geojson)
    return props, name, location, elevation_source


# =============================================================================
# ELEVATION PROCESSING
# =============================================================================

def _median(values):
    s = sorted(v for v in values if v is not None)
    if not s: return 0.0
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid-1] + s[mid]) / 2.0


def reject_spikes(elevations, threshold):
    if threshold <= 0.0: return list(elevations)
    result = list(elevations)
    n = len(elevations)
    for i in range(n):
        start = max(0, i - 5)
        end   = min(n, i + 6)
        hood  = [elevations[j] for j in range(start, end)
                 if elevations[j] is not None]
        if not hood: continue
        med = _median(hood)
        val = elevations[i] if elevations[i] is not None else med
        if abs(val - med) > threshold:
            result[i] = med
    return result


def moving_average(elevations, window):
    if window <= 1: return list(elevations)
    if window % 2 == 0: window += 1
    half = window // 2
    n    = len(elevations)
    result = []
    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half + 1)
        vals  = [elevations[j] for j in range(start, end)
                 if elevations[j] is not None]
        result.append(sum(vals) / len(vals) if vals else 0.0)
    return result


def close_elevation_gap(elevations, blend_fraction):
    """
    Force the last point to exactly match the first point by subtracting
    a linear ramp across the full profile that removes the gap gradually.
    The correction is zero at the start and equals the full gap at the end,
    so no single point jumps — the adjustment is spread across every point.

    blend_fraction is retained as a parameter for API compatibility but the
    ramp approach guarantees exact endpoint equality without needing a
    blend zone — the entire profile absorbs the correction.
    """
    if len(elevations) < 2:
        return elevations

    result  = list(elevations)
    n       = len(result)
    gap     = result[-1] - result[0]   # how much the end overshoots the start

    # Subtract a ramp that goes from 0 at index 0 to `gap` at index n-1
    # This tilts the whole profile just enough to close the join exactly
    for i in range(n):
        t         = i / (n - 1)
        result[i] -= gap * t

    return result


def get_elevations(coords, is_srtm):
    """
    Read elevation from coord[2] for every point.
    SRTM-specific spike rejection runs first for SRTM circuits.
    A global smooth pass runs for ALL circuits to remove GPS/SRTM jitter.
    End-gap blending always applied last.
    """
    raw = [c[2] if len(c) > 2 and c[2] is not None else 0.0
           for c in coords]

    if is_srtm:
        if SPIKE_THRESHOLD > 0.0:
            raw = reject_spikes(raw, SPIKE_THRESHOLD)
        if SMOOTH_WINDOW > 1:
            raw = moving_average(raw, SMOOTH_WINDOW)

    # Global smooth — applied to telemetry and SRTM alike.
    # Savitzky-Golay fits a polynomial to each local window, preserving
    # peaks and valleys far better than a moving average at the same window.
    if ELEVATION_SMOOTH_WINDOW > 1:
        window = ELEVATION_SMOOTH_WINDOW
        if window % 2 == 0:
            window += 1
        if HAVE_SCIPY and savgol_filter is not None and len(raw) > window:
            # Clamp poly order to be less than window size
            poly = min(SAVGOL_POLY_ORDER, window - 1)
            raw  = savgol_filter(raw, window, poly).tolist()
        else:
            # Fallback: moving average
            raw = moving_average(raw, window)

    raw = close_elevation_gap(raw, BLEND_FRACTION)
    return raw


# =============================================================================
# RESAMPLING
# =============================================================================

# =============================================================================
# RESAMPLING
# =============================================================================

def resample_by_length(coords, elevations, resolution_m):
    """
    Fit a cubic spline through the coordinate positions and resample
    at uniform arc-length intervals of `resolution_m` metres.

    This produces genuine smooth geometry rather than just subdividing
    the existing straight segments. Both XY and elevation are interpolated
    along the fitted spline.

    Falls back to linear resampling if scipy is unavailable or the
    coordinate list is too short to fit a spline.
    """
    if len(coords) < 2:
        return coords, elevations

    # Convert coords to real-world metres for arc-length computation
    lon0 = coords[0][0]
    lat0 = coords[0][1]
    cos_lat = math.cos(math.radians(lat0))

    xs_m = [(c[0] - lon0) * cos_lat * METRES_PER_DEGREE for c in coords]
    ys_m = [(c[1] - lat0)             * METRES_PER_DEGREE for c in coords]

    # Cumulative arc length in metres
    dists = [0.0]
    for i in range(1, len(xs_m)):
        dx = xs_m[i] - xs_m[i-1]
        dy = ys_m[i] - ys_m[i-1]
        dists.append(dists[-1] + math.sqrt(dx*dx + dy*dy))

    total_m = dists[-1]
    if total_m == 0 or resolution_m <= 0:
        return coords, elevations

    n_out   = max(2, int(round(total_m / resolution_m)) + 1)
    t_out   = [i * total_m / (n_out - 1) for i in range(n_out)]

    if HAVE_SCIPY and CubicSpline is not None and len(coords) >= 4:
        # CubicSpline requires strictly increasing x (arc-length distances).
        # Deduplicate any consecutive points with zero distance between them.
        eles = elevations if len(elevations) == len(coords) \
               else [0.0] * len(coords)

        t_in_raw  = dists
        xs_m_raw  = xs_m
        ys_m_raw  = ys_m
        eles_raw  = eles
        extra_raw = [[c[k] if c[k] is not None else 0.0 for c in coords]
                     for k in range(3, len(coords[0]))]

        t_in, xs_f, ys_f, eles_f, extra_f = [], [], [], [], [[] for _ in extra_raw]
        for i, t in enumerate(t_in_raw):
            if t_in and t <= t_in[-1]:
                continue   # skip duplicate or non-increasing distance
            t_in.append(t)
            xs_f.append(xs_m_raw[i])
            ys_f.append(ys_m_raw[i])
            eles_f.append(eles_raw[i])
            for j, er in enumerate(extra_raw):
                extra_f[j].append(er[i])

        if len(t_in) < 4:
            # Not enough unique points for a spline — fall through to linear
            pass
        else:
            cs_x = CubicSpline(t_in, xs_f)
            cs_y = CubicSpline(t_in, ys_f)
            cs_e = CubicSpline(t_in, eles_f)
            extra_splines = [CubicSpline(t_in, ef) for ef in extra_f]

            new_coords     = []
            new_elevations = []

            for t in t_out:
                xm  = float(cs_x(t))
                ym  = float(cs_y(t))
                lon = lon0 + xm / (cos_lat * METRES_PER_DEGREE)
                lat = lat0 + ym / METRES_PER_DEGREE
                row = [round(lon, 8), round(lat, 8)]
                for es in extra_splines:
                    row.append(round(float(es(t)), 4))
                new_coords.append(row)
                new_elevations.append(float(cs_e(t)))

            return new_coords, new_elevations

    # Fallback: linear interpolation (scipy unavailable or < 4 unique points)
    new_coords     = []
    new_elevations = []
    j = 0
    for t in t_out:
        while j < len(dists) - 2 and dists[j+1] < t:
            j += 1
        if j >= len(dists) - 1:
            new_coords.append(list(coords[-1]))
            new_elevations.append(elevations[-1] if elevations else 0.0)
        else:
            seg  = dists[j+1] - dists[j]
            frac = (t - dists[j]) / seg if seg > 0 else 0.0
            lon  = coords[j][0] + frac * (coords[j+1][0] - coords[j][0])
            lat  = coords[j][1] + frac * (coords[j+1][1] - coords[j][1])
            row  = [round(lon, 8), round(lat, 8)]
            for k in range(3, len(coords[j])):
                v0 = coords[j][k]   if coords[j][k]   is not None else 0.0
                v1 = coords[j+1][k] if coords[j+1][k] is not None else 0.0
                row.append(v0 + frac * (v1 - v0))
            new_coords.append(row)
            e0 = elevations[j]   if j   < len(elevations) else 0.0
            e1 = elevations[j+1] if j+1 < len(elevations) else e0
            new_elevations.append(e0 + frac * (e1 - e0))

    return new_coords, new_elevations


# =============================================================================
# BLENDER CURVE CREATION
# =============================================================================

def attach_nodegroup(obj, group_name):
    """
    Attach a Geometry Nodes modifier to obj using an existing node group.
    Prints a warning if the node group doesn't exist — never creates one.
    """
    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        print(f"  WARNING: node group '{group_name}' not found — "
              f"skipping modifier on '{obj.name}'")
        return
    mod            = obj.modifiers.new(group_name, type='NODES')
    mod.node_group = ng


def get_or_create_collection(name, parent_col=None):
    """Get or create a Blender collection, linked to parent_col (or scene root)."""
    col    = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    target = parent_col if parent_col is not None else bpy.context.scene.collection
    if col.name not in [c.name for c in target.children]:
        target.children.link(col)
    # If placing inside a specific parent, remove any stale link at the scene root
    if target is not bpy.context.scene.collection:
        scene_root = bpy.context.scene.collection
        if col.name in [c.name for c in scene_root.children]:
            scene_root.children.unlink(col)
    return col


def _col(collection):
    """Resolve a collection, falling back to the scene root 'Collection'."""
    return collection or bpy.data.collections.get("Collection") or bpy.context.collection


def set_collection_visibility(col_name, visible):
    """Set viewport and render visibility for a top-level collection."""
    col = bpy.data.collections.get(col_name)
    if col is None:
        return
    col.hide_render = not visible

    def find_lc(layer_col, name):
        if layer_col.collection.name == name:
            return layer_col
        for child in layer_col.children:
            result = find_lc(child, name)
            if result:
                return result
        return None

    lc = find_lc(bpy.context.view_layer.layer_collection, col_name)
    if lc:
        lc.hide_viewport = not visible


def build_terrain_mesh(terrain_props, lon_origin, lat_origin,
                       parent_obj, blender_name, collection=None):
    """
    Build terrain mesh from a terrain feature's properties dict
    (feature_type == "terrain"). Sea is now built from OSM coastline
    via the separate "sea" feature — see build_sea_mesh.

    Creates one object parented to parent_obj:
      '<name> - Terrain' : quad mesh, Z from elevation points
    """
    points     = terrain_props.get("points", [])
    n_lat      = terrain_props["n_lat"]
    n_lon      = terrain_props["n_lon"]

    if not points or n_lat < 2 or n_lon < 2:
        print(f"  [{blender_name} - Terrain]  SKIP — no point data")
        return

    # Build vertex positions from [[lon, lat, z], ...] point list
    terrain_verts = []

    for idx, pt in enumerate(points):
        lon = pt[0]
        lat = pt[1]
        ele = pt[2] if pt[2] is not None else 0.0
        x, y = equirectangular(lon, lat, lon_origin, lat_origin)
        z    = max(ele, 0.0) * Z_SCALE
        terrain_verts.append((x, y, z))

    # Build quad faces
    terrain_faces = []
    for row in range(n_lat - 1):
        for col in range(n_lon - 1):
            i00 = row * n_lon + col
            i10 = i00 + 1
            i01 = i00 + n_lon
            i11 = i01 + 1
            terrain_faces.append((i00, i10, i11, i01))

    t_mesh = bpy.data.meshes.new(f"{blender_name} - Terrain")
    t_mesh.from_pydata(terrain_verts, [], terrain_faces)
    t_mesh.update()
    t_obj  = bpy.data.objects.new(f"{blender_name} - Terrain", t_mesh)
    _col(collection).objects.link(t_obj)
    t_obj.parent = parent_obj
    print(f"  [{blender_name} - Terrain]  "
          f"{len(terrain_verts)} verts  {len(terrain_faces)} faces")
    return t_obj


def build_triangle_mesh(triangles, obj_name, z, lon_origin, lat_origin,
                        feature_type, parent_obj, blender_name, extra_props=None,
                        terrain_z_fn=None, min_hexagons=0, collection=None):
    """
    Build a single merged mesh object from a list of triangles.
    Each triangle is [[lon,lat],[lon,lat],[lon,lat]].

    min_hexagons : if > 0, connected face islands with <= min_hexagons*6 faces
                  are deleted after merge. Use to remove stray coastline fragments.
    If terrain_z_fn is provided, Z is projected per vertex.
    Otherwise the flat z value is used for all vertices.
    """
    import bmesh as _bmesh

    if not triangles:
        return None

    n_tris  = len(triangles)
    cos_lat = _cos_lat_cache.get(lat_origin) or math.cos(math.radians(lat_origin))

    if HAVE_NUMPY:
        import numpy as _np
        # Flatten all triangle points in one shot: shape (n_tris*3, 2)
        raw = _np.array([[pt[0], pt[1]] for tri in triangles for pt in tri])
        xs  = (raw[:, 0] - lon_origin) * cos_lat * SCALE
        ys  = (raw[:, 1] - lat_origin) * SCALE

        batch = getattr(terrain_z_fn, 'batch', None)
        if batch is not None:
            zs = batch(xs, ys)
        elif terrain_z_fn is not None:
            zs = _np.array([terrain_z_fn(x, y) for x, y in zip(xs, ys)])
        else:
            zs = _np.full(len(xs), float(z))

        all_verts = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    else:
        all_verts = []
        for tri in triangles:
            for pt in tri:
                x  = (pt[0] - lon_origin) * cos_lat * SCALE
                y  = (pt[1] - lat_origin) * SCALE
                pz = terrain_z_fn(x, y) if terrain_z_fn is not None else z
                all_verts.append((x, y, pz))

    all_faces = [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(n_tris)]

    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(all_verts, [], all_faces)
    me.update()

    merge_dist = 0.01 * Z_SCALE
    bm = _bmesh.new()
    bm.from_mesh(me)
    _bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)

    # Remove disconnected face islands smaller than min_hexagons hexagons.
    # Each H3 hex produces 6 triangular faces, so the threshold is min_hexagons*6.
    if min_hexagons > 0:
        bm.faces.ensure_lookup_table()
        visited  = set()
        islands  = []
        for seed in bm.faces:
            if seed in visited:
                continue
            island = []
            stack  = [seed]
            while stack:
                f = stack.pop()
                if f in visited:
                    continue
                visited.add(f)
                island.append(f)
                for edge in f.edges:
                    for lf in edge.link_faces:
                        if lf not in visited:
                            stack.append(lf)
            islands.append(island)

        min_faces   = min_hexagons * 6
        to_delete   = [f for isl in islands if len(isl) <= min_faces for f in isl]
        n_removed   = len([isl for isl in islands if len(isl) <= min_faces])
        if to_delete:
            _bmesh.ops.delete(bm, geom=to_delete, context='FACES')
            # delete(FACES) leaves orphaned vertices — remove them too so their
            # positions cannot enter the depth-offset KDTree and cause phantom
            # terrain depression at the deleted island locations.
            orphaned = [v for v in bm.verts if not v.link_faces]
            if orphaned:
                _bmesh.ops.delete(bm, geom=orphaned, context='VERTS')
            print(f"  [island filter] '{obj_name}': removed {n_removed} island(s) "
                  f"(<= {min_hexagons} hexes, {len(to_delete)} faces)")

    bm.to_mesh(me)
    bm.free()
    me.update()

    # Return None if the mesh is now empty after island filtering
    if len(me.vertices) == 0:
        return None

    obj = bpy.data.objects.new(obj_name, me)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = feature_type
    if extra_props:
        for k, v in extra_props.items():
            obj[k] = v
    _col(collection).objects.link(obj)
    obj.parent = parent_obj

    return obj


def build_sea_curve(feat, lon_origin, lat_origin, parent_obj, blender_name,
                    depth_m=0.0, ring=None, collection=None):
    """
    Build sea H3 hex mesh at Z=0 (flat). Z is set to sea level (0) by
    project_water_mesh_flat after this returns. Depth displacement is
    handled separately by apply_water_depth_to_terrain.
    Returns (mesh_obj, depth_offset_fn) — either may be None on failure.
    """
    props     = feat.get("properties", {})
    sea_index = props.get("sea_index", 0)
    label     = chr(65 + sea_index)
    obj_base  = f"{blender_name} - Sea {label}"

    triangles = props.get("triangles", [])
    if not triangles or len(triangles) // 6 < MIN_WATER_HEX:
        return None, None

    mesh_name = f"{obj_base} Mesh"
    mesh_obj  = build_triangle_mesh(
        triangles, mesh_name, 0.0,
        lon_origin, lat_origin,
        "sea", parent_obj, blender_name,
        extra_props={"sea_index": sea_index},
        terrain_z_fn=None,
        min_hexagons=7,
        collection=collection)

    if mesh_obj is None:
        print(f"  [{mesh_name}]  mesh build failed or empty after island filter - skipped")
        return None, None

    # Build depth fn from actual surviving mesh vertices so deleted
    # islands cannot cause phantom terrain depressions.
    attach_nodegroup(mesh_obj, "Sea")
    print(f"  [{mesh_name}]  {len(triangles)} tris  "
          f"{len(mesh_obj.data.vertices)} verts (merged)  depth={depth_m:.1f}m")
    return mesh_obj


def build_water_curve(feat, lon_origin, lat_origin, parent_obj, blender_name,
                      tag_counts, depth_m=0.0, ring=None,
                      collection=None):
    """
    Build a water body hex mesh at Z=0 (flat). Z is projected to mean terrain
    height by project_water_mesh_flat after this returns. Depth displacement is
    computed from the projected mesh and applied by apply_water_depth_to_terrain.
    Returns mesh_obj or None on failure.
    """
    props     = feat.get("properties", {})
    water_tag = props.get("water_tag", "water")

    triangles = props.get("triangles", [])
    if not triangles or len(triangles) // 6 < MIN_WATER_HEX:
        return None

    tag_counts[water_tag] = tag_counts.get(water_tag, 0) + 1
    obj_name  = f"{blender_name} - Water - {water_tag} - {tag_counts[water_tag]}"
    mesh_name = f"{obj_name} Mesh"

    mesh_obj = build_triangle_mesh(
        triangles, mesh_name, 0.0,
        lon_origin, lat_origin,
        "water_mesh", parent_obj, blender_name,
        extra_props={"water_tag": water_tag},
        terrain_z_fn=None,
        min_hexagons=7,
        collection=collection)

    if mesh_obj is None:
        print(f"  [{mesh_name}]  mesh build failed or empty after island filter - skipped")
        return None

    attach_nodegroup(mesh_obj, "Water")
    print(f"  [{mesh_name}]  {len(triangles)} tris  "
          f"{len(mesh_obj.data.vertices)} verts (merged)  depth={depth_m:.1f}m")
    return mesh_obj


def apply_water_depth_to_terrain(terrain_obj, depth_fns, blender_name):
    """
    Depress terrain vertices by the combined depth offset from all water bodies.

    depth_fns : list of callables returned by _make_water_depth_offset_fn.
                Each returns a depression amount in Blender Z-units for a given
                (x, y) position. Supports a .batch(xs, ys) numpy path.

    For each terrain vertex the maximum offset across all water bodies is used
    (lakes and sea can overlap; the deeper one wins). Vertices with zero offset
    are untouched (land outside all water footprints).
    """
    if terrain_obj is None or not depth_fns:
        return

    mesh = terrain_obj.data
    n    = len(mesh.vertices)

    if HAVE_NUMPY:
        import numpy as _np
        xs = _np.array([v.co.x for v in mesh.vertices])
        ys = _np.array([v.co.y for v in mesh.vertices])

        total = _np.zeros(n)
        for fn in depth_fns:
            batch = getattr(fn, 'batch', None)
            offsets = batch(xs, ys) if batch is not None else \
                      _np.array([fn(x, y) for x, y in zip(xs, ys)])
            _np.maximum(total, offsets, out=total)

        moved = 0
        for i, v in enumerate(mesh.vertices):
            if total[i] > 0.0:
                v.co.z -= total[i]
                moved  += 1
    else:
        moved = 0
        for v in mesh.vertices:
            offset = max(fn(v.co.x, v.co.y) for fn in depth_fns)
            if offset > 0.0:
                v.co.z -= offset
                moved  += 1

    mesh.update()
    print(f"  [{blender_name}] water depth: {moved} terrain verts depressed")


def build_bounding_wall(terrain_props, lon_origin, lat_origin,
                        parent_obj, blender_name, collection=None):
    """
    Build a simple rectangular wall around the terrain bounding box.
    Wall height equals the Z range of the terrain mesh plus 0.1m offset
    on both top and bottom.
    """
    import bmesh as _bmesh

    if not terrain_props:
        return

    lon_min = terrain_props["lon_min"];  lon_max = terrain_props["lon_max"]
    lat_min = terrain_props["lat_min"];  lat_max = terrain_props["lat_max"]

    x0, y0 = equirectangular(lon_min, lat_min, lon_origin, lat_origin)
    x1, y1 = equirectangular(lon_max, lat_max, lon_origin, lat_origin)

    # Find the terrain mesh object — try hex first, fall back to quad
    terrain_obj = (bpy.data.objects.get(f"{blender_name} - Terrain") or
                   bpy.data.objects.get(f"{blender_name} - Terrain"))

    thick = 500.0 * Z_SCALE   # wall thickness in Blender units

    if terrain_obj is not None and "terrain_z_bottom" in terrain_obj:
        z_bottom = terrain_obj["terrain_z_bottom"]
        z_top    = z_bottom + 0.1 * Z_SCALE
    elif terrain_obj is not None:
        corners  = [terrain_obj.matrix_world @ Vector(c)
                    for c in terrain_obj.bound_box]
        z_bottom = min(c.z for c in corners)
        z_top    = z_bottom + 0.1 * Z_SCALE
    else:
        z_bottom = -0.1 * Z_SCALE
        z_top    = 0.0

    name = f"{blender_name} - Boundary Wall"
    bm   = _bmesh.new()

    # Outer and inner bbox corners for wall thickness
    outer = [(x0 - thick, y0 - thick),
             (x1 + thick, y0 - thick),
             (x1 + thick, y1 + thick),
             (x0 - thick, y1 + thick)]
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # Build 4 wall segments, each a quad with thickness
    for i in range(4):
        j    = (i + 1) % 4
        o_i  = bm.verts.new((outer[i][0], outer[i][1], z_bottom))
        o_j  = bm.verts.new((outer[j][0], outer[j][1], z_bottom))
        i_i  = bm.verts.new((inner[i][0], inner[i][1], z_bottom))
        i_j  = bm.verts.new((inner[j][0], inner[j][1], z_bottom))
        o_it = bm.verts.new((outer[i][0], outer[i][1], z_top))
        o_jt = bm.verts.new((outer[j][0], outer[j][1], z_top))
        i_it = bm.verts.new((inner[i][0], inner[i][1], z_top))
        i_jt = bm.verts.new((inner[j][0], inner[j][1], z_top))

        # Outer face, inner face, bottom cap, top cap
        bm.faces.new([o_i,  o_j,  o_jt, o_it])
        bm.faces.new([i_j,  i_i,  i_it, i_jt])
        bm.faces.new([o_i,  i_i,  i_j,  o_j ])
        bm.faces.new([o_it, o_jt, i_jt, i_it])

    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(name, me)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = "boundary_wall"
    _col(collection).objects.link(obj)
    obj.parent = parent_obj

    vg     = obj.vertex_groups.new(name="Bottom")
    bot_vi = [v.index for v in me.vertices if abs(v.co.z - z_bottom) < 1e-6]
    vg.add(bot_vi, 1.0, 'REPLACE')

    print(f"  [{name}]  Z {z_bottom:.4f} → {z_top:.4f}  bottom_verts={len(bot_vi)}")


def _polygon_area_m2(ring):
    """Shoelace area of a [[lon, lat], ...] ring in square metres."""
    n       = len(ring)
    if n < 3:
        return 0.0
    lat_ref     = sum(p[1] for p in ring) / n
    m_per_deg_lon = METRES_PER_DEGREE * math.cos(math.radians(lat_ref))
    m_per_deg_lat = METRES_PER_DEGREE
    area = 0.0
    for i in range(n):
        x0 = ring[i][0]         * m_per_deg_lon
        y0 = ring[i][1]         * m_per_deg_lat
        x1 = ring[(i+1) % n][0] * m_per_deg_lon
        y1 = ring[(i+1) % n][1] * m_per_deg_lat
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def extrude_flat_groups(terrain_obj, groups, blender_name, offsets=None):
    """
    Depress Z vertices for the given vertex groups on terrain_obj.

    offsets : dict {group_name: z_offset_blender_units} giving the maximum
              depression (negative value) for each group. Groups absent from
              the dict receive no displacement.

    Two falloff strategies are used:
    - Groups named "W..." (water bodies): use the per-vertex weight stored in
      the vertex group by assign_sea_vertex_groups. The weight is a cosine
      falloff derived from Shapely distance to the water polygon boundary
      (0 at the edge, 1 at the centre). Works correctly for narrow features
      (rivers, canals) where BFS hop-count would give every vertex factor=0.
    - All other groups (sea "A", "B", ...): BFS hop-count from the transition
      boundary, then cosine falloff normalised to max hop depth.
    """
    from collections import deque as _deque
    import bmesh as _bm

    if not groups or terrain_obj is None:
        return

    offsets = offsets or {}
    mesh    = terrain_obj.data

    # ── Build adjacency once ──────────────────────────────────────────────
    bm = _bm.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    adjacency = {v.index: set() for v in bm.verts}
    for edge in bm.edges:
        a, b = edge.verts[0].index, edge.verts[1].index
        adjacency[a].add(b)
        adjacency[b].add(a)
    bm.free()

    # ── Precompute group membership with per-vertex weights ──────────────
    vg_names     = {vg.name: vg.index for vg in terrain_obj.vertex_groups}
    # group_members: {vg_index: {vert_index: weight}}
    group_members = {vg_names[g]: {} for g in groups if g in vg_names}
    for v in mesh.vertices:
        for g in v.groups:
            if g.group in group_members:
                group_members[g.group][v.index] = g.weight

    # ── Helper: look up a group and return (z_offset, in_group) or None ────
    def _get_group(group_name):
        if group_name not in vg_names:
            print(f"  [{blender_name}] group '{group_name}' not found — skipping")
            return None, None
        z_off = offsets.get(group_name)
        if z_off is None:
            print(f"  [{blender_name} - depress] '{group_name}': no offset — skipping")
            return None, None
        ig = group_members.get(vg_names[group_name], {})
        if not ig:
            print(f"  [{blender_name} - depress] '{group_name}': EMPTY — no vertices in group")
            return None, None
        return z_off, ig

    # ── Pass 1 — water body groups (W*), processed FIRST ─────────────────
    # Water groups use pre-computed cosine weights (Shapely boundary
    # distance).  They must run before sea groups so that inland water
    # bodies (e.g. Albert Park Lake, Melbourne) are not clobbered by the
    # sea BFS falloff even when the sea polygon happens to contain them.
    water_modified = set()   # vertex indices already moved by a water group

    for group_name in groups:
        if not group_name.startswith("W"):
            continue
        z_offset, in_group = _get_group(group_name)
        if z_offset is None:
            continue
        depth_m = abs(z_offset) / Z_SCALE
        moved   = 0
        for v in mesh.vertices:
            if v.index not in in_group:
                continue
            factor = in_group[v.index]     # pre-computed cosine [0, 1]
            v.co.z += z_offset * factor
            water_modified.add(v.index)
            moved += 1
        max_w = max(in_group.values(), default=0.0)
        print(f"  [{blender_name} - depress] '{group_name}': {moved} verts  "
              f"max_weight={max_w:.3f}  depth={depth_m:.1f}m")

    # ── Pass 2 — sea / other groups, skip water-modified vertices ────────
    # BFS hop-count from the group's transition boundary, cosine falloff.
    # Vertices already moved by a water group are left untouched so inland
    # lakes / rivers are not double-depressed by the surrounding sea polygon.
    for group_name in groups:
        if group_name.startswith("W"):
            continue
        z_offset, in_group = _get_group(group_name)
        if z_offset is None:
            continue
        depth_m = abs(z_offset) / Z_SCALE

        in_group_set = set(in_group.keys())
        transition_boundary = {vi for vi in in_group_set
                                if any(nb not in in_group_set
                                       for nb in adjacency[vi])}

        hop_dist = {}
        queue    = _deque()
        for vi in transition_boundary:
            hop_dist[vi] = 0
            queue.append(vi)
        while queue:
            vi   = queue.popleft()
            dist = hop_dist[vi]
            for nb in adjacency[vi]:
                if nb in in_group_set and nb not in hop_dist:
                    hop_dist[nb] = dist + 1
                    queue.append(nb)

        max_hop = max(hop_dist.values(), default=1) or 1
        moved   = 0
        for v in mesh.vertices:
            if v.index not in in_group_set:
                continue
            if v.index in water_modified:
                continue    # water group already handled this vertex
            hops   = hop_dist.get(v.index, max_hop)
            t      = min(hops / max_hop, 1.0)
            factor = water_edge_falloff(t)
            v.co.z += z_offset * factor
            moved  += 1

        print(f"  [{blender_name} - depress] '{group_name}': {moved} verts  "
                  f"max_hop={max_hop}  depth={depth_m:.1f}m")

    mesh.update()


def identify_sea_vertex_group(terrain_obj, std_dev_threshold=SEA_STD_THRESHOLD):
    """
    Identify sea vertex groups by finding all groups whose Z standard
    deviation falls below std_dev_threshold. Water surfaces are flat
    (std dev ≈ 0), land has elevation variation.

    Stores the list of sea group names as a custom property 'sea_groups'
    on the terrain object.
    """
    if terrain_obj is None or terrain_obj.type != 'MESH':
        return []

    mesh      = terrain_obj.data
    results   = []
    sea_groups = []

    for vg in terrain_obj.vertex_groups:
        # W* groups are water bodies — never coastline/sea groups.
        # They are flat for the same reason (H3 cells over water) so they
        # would always pass the std_z threshold and get double-depressed.
        if vg.name.startswith("W"):
            continue
        vg_idx = vg.index
        zs     = [v.co.z for v in mesh.vertices
                  if any(g.group == vg_idx for g in v.groups)]
        if not zs:
            continue
        n      = len(zs)
        mean_z = sum(zs) / n
        std_z  = math.sqrt(sum((z - mean_z) ** 2 for z in zs) / n)
        results.append((vg.name, std_z, n))

    results.sort(key=lambda r: r[1])

    for name, std_z, n in results:
        is_sea = std_z < std_dev_threshold
        tag    = " <- sea" if is_sea else ""
        print(f"  [sea id] group '{name}':  n={n}  std_z={std_z:.6f}{tag}")
        if is_sea:
            sea_groups.append(name)

    if sea_groups:
        print(f"  [sea id] → sea groups: {sea_groups}")
    else:
        print(f"  [sea id] → no groups below threshold {std_dev_threshold} "
              f"(lowest was {results[0][1]:.6f} '{results[0][0]}')")

    terrain_obj["sea_groups"] = str(sea_groups)
    return sea_groups


STREET_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "residential", "service",
}


def build_street_curve(feat, lon_origin, lat_origin, terrain_z_fn,
                       parent_obj, blender_name, collection=None):
    """
    Build a single street as a 3D NURBS curve, projected onto terrain.
    Skips highway types not in STREET_HIGHWAY_TYPES.
    """
    props       = feat.get("properties", {})
    geom        = feat.get("geometry", {})
    highway_tag = props.get("highway_tag", "")

    if highway_tag not in STREET_HIGHWAY_TYPES:
        return
    if not geom or geom.get("type") != "LineString":
        return

    coords = geom.get("coordinates", [])
    if len(coords) < 2:
        return

    width_m  = props.get("width_m", 0.01)
    osm_id   = props.get("osm_id", "")
    obj_name = f"{blender_name} - Road - {highway_tag} - {osm_id}"

    # Resample at POINT_RESOLUTION_M
    coords = resample_fixed_spacing(coords, POINT_RESOLUTION_M)

    curve            = bpy.data.curves.new(obj_name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 0

    spline = curve.splines.new('NURBS')
    spline.points.add(len(coords) - 1)
    spline.use_endpoint_u = True
    spline.order_u        = min(4, len(coords))

    for i, coord in enumerate(coords):
        x, y = equirectangular(coord[0], coord[1], lon_origin, lat_origin)
        z    = terrain_z_fn(x, y) if terrain_z_fn else 0.0
        spline.points[i].co     = (x, y, z, 1.0)
        spline.points[i].radius = width_m

    obj = bpy.data.objects.new(obj_name, curve)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = "street"
    obj["highway_tag"]  = highway_tag
    _col(collection).objects.link(obj)
    obj.parent = parent_obj

    attach_nodegroup(obj, "Road Profile")


def _set_gn_input(mod, name, value):
    """
    Set a named input socket on a Geometry Nodes modifier.
    Works with both Blender 3.x (inputs[]) and 4.x (interface.items_tree).
    Silently skips if the socket is not found.
    """
    ng = mod.node_group
    if ng is None:
        return
    # Blender 4.x — interface.items_tree
    try:
        for item in ng.interface.items_tree:
            if (getattr(item, "item_type", None) == "SOCKET"
                    and getattr(item, "in_out",   None) == "INPUT"
                    and item.name == name):
                mod[item.identifier] = value
                return
    except AttributeError:
        pass
    # Blender 3.x fallback — direct key access by name
    try:
        mod[name] = value
    except Exception:
        pass


def build_vegetation(feat, lon_origin, lat_origin, terrain_z_fn,
                     parent_obj, blender_name, veg_counts=None, collection=None):
    """
    Build a vegetation object from a GeoJSON vegetation feature.
    Reads H3 triangles, builds a mesh, projects Z onto terrain,
    attaches the Vegetation Scatter GN modifier.
    """
    props   = feat.get("properties", {})
    veg_tag = props.get("veg_tag", "vegetation")
    tris    = props.get("triangles", [])

    if not tris or len(tris) < VEG_MIN_TRIS:
        return

    if veg_counts is not None:
        veg_counts[veg_tag] = veg_counts.get(veg_tag, 0) + 1
        obj_name = f"{blender_name} - Veg - {veg_tag} - {veg_counts[veg_tag]}"
    else:
        obj_name = f"{blender_name} - Veg - {veg_tag}"

    import bmesh as _bm
    cos_lat = _cos_lat_cache.get(lat_origin) or math.cos(math.radians(lat_origin))

    if HAVE_NUMPY:
        import numpy as _np
        raw   = _np.array([[pt[0], pt[1]] for tri in tris for pt in tri])
        xs    = (raw[:, 0] - lon_origin) * cos_lat * SCALE
        ys    = (raw[:, 1] - lat_origin) * SCALE
        batch = getattr(terrain_z_fn, 'batch', None)
        if batch is not None:
            zs = batch(xs, ys)
        elif terrain_z_fn is not None:
            zs = _np.array([terrain_z_fn(x, y) for x, y in zip(xs, ys)])
        else:
            zs = _np.zeros(len(xs))
        all_verts = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    else:
        all_verts = []
        for tri in tris:
            for pt in tri:
                px = (pt[0] - lon_origin) * cos_lat * SCALE
                py = (pt[1] - lat_origin) * SCALE
                pz = terrain_z_fn(px, py) if terrain_z_fn else 0.0
                all_verts.append((px, py, pz))

    all_faces = [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(len(tris))]

    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(all_verts, [], all_faces)
    me.update()

    bm = _bm.new()
    bm.from_mesh(me)
    _bm.ops.remove_doubles(bm, verts=bm.verts, dist=0.01 * Z_SCALE)
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(obj_name, me)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = "vegetation"
    obj["veg_tag"]      = veg_tag
    _col(collection).objects.link(obj)
    obj.parent = parent_obj

    # Attach Vegetation Scatter GN modifier and set density inputs
    tree_d, bush_d, grass_d = VEG_DENSITIES.get(veg_tag, VEG_DENSITY_DEFAULT)
    ng = bpy.data.node_groups.get("Vegetation Scatter")
    if ng is None:
        print(f"  WARNING: 'Vegetation Scatter' node group not found — "
              f"skipping modifier on '{obj.name}'")
    else:
        mod            = obj.modifiers.new("Vegetation Scatter", type='NODES')
        mod.node_group = ng
        _set_gn_input(mod, "tree_density",  tree_d)
        _set_gn_input(mod, "bush_density",  bush_d)
        _set_gn_input(mod, "grass_density", grass_d)

    print(f"  [{obj_name}]  {len(tris)} tris  {len(me.vertices)} verts  "
          f"tree={tree_d:.1f}  bush={bush_d:.1f}  grass={grass_d:.1f}")


def transfer_vertex_groups(src_obj, dst_obj):
    """
    Transfer vertex group membership from src_obj to dst_obj by proximity.
    For each vertex in dst_obj, finds the nearest vertex in src_obj and
    copies its group memberships.
    Uses cKDTree for fast nearest-neighbour lookup.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [transfer VG] scipy not available — skipping")
        return

    if not src_obj.vertex_groups:
        return

    src_mesh = src_obj.data
    dst_mesh = dst_obj.data

    # Build KDTree from src vertex positions
    src_coords = [(v.co.x, v.co.y, v.co.z) for v in src_mesh.vertices]
    tree       = cKDTree(src_coords)

    # Build src group membership: {vert_index: [(group_index, weight), ...]}
    src_groups = {}
    for v in src_mesh.vertices:
        if v.groups:
            src_groups[v.index] = [(g.group, g.weight) for g in v.groups]

    if not src_groups:
        return

    # Recreate vertex groups on dst_obj matching src_obj names
    vg_map = {}   # src group index -> dst group
    for vg in src_obj.vertex_groups:
        if vg.name not in [g.name for g in dst_obj.vertex_groups]:
            dst_vg = dst_obj.vertex_groups.new(name=vg.name)
        else:
            dst_vg = dst_obj.vertex_groups[vg.name]
        vg_map[vg.index] = dst_vg

    # Query nearest src vert for each dst vert
    dst_coords = [(v.co.x, v.co.y, v.co.z) for v in dst_mesh.vertices]
    _, indices  = tree.query(dst_coords)

    # Accumulate assignments per group
    group_assignments = {vg.index: [] for vg in dst_obj.vertex_groups}
    for dst_idx, src_idx in enumerate(indices):
        for src_gi, weight in src_groups.get(src_idx, []):
            if src_gi in vg_map:
                dst_gi = vg_map[src_gi].index
                group_assignments[dst_gi].append((dst_idx, weight))

    for dst_gi, assignments in group_assignments.items():
        if assignments:
            vg = dst_obj.vertex_groups[dst_gi]
            vert_ids = [a[0] for a in assignments]
            vg.add(vert_ids, 1.0, 'REPLACE')

    print(f"  [transfer VG] {len(src_obj.vertex_groups)} groups "
          f"transferred from '{src_obj.name}' → '{dst_obj.name}'")


def assign_sea_vertex_groups(terrain_obj, sea_feats, water_feats,
                             lon_origin, lat_origin):
    """
    Assign vertex groups on terrain_obj:
    - Sea polygons → groups named "A", "B", "C" ...
    - Water body polygons → groups named "W1", "W2", "W3" ...

    Sea containment uses the OSM polygon outer ring in lon/lat space.
    Water body containment is built from the same H3 triangles used to
    construct the water mesh object — this means the vertex group boundary
    exactly matches the water body's XY footprint in Blender.
    """
    if terrain_obj is None:
        return

    if not HAVE_SHAPELY:
        print("  [vertex groups] shapely not available — skipping")
        return

    cos_lat = _cos_lat_cache.get(lat_origin)
    if cos_lat is None:
        cos_lat = math.cos(math.radians(lat_origin))
        _cos_lat_cache[lat_origin] = cos_lat

    # ── Convert all terrain vertices to lon/lat once ──────────────────────
    t_verts    = terrain_obj.data.vertices
    t_lonlat   = [(v.co.x / (cos_lat * SCALE) + lon_origin,
                   v.co.y / SCALE             + lat_origin)
                  for v in t_verts]
    sea_claimed = set()   # indices claimed by sea — skipped for water bodies

    # ── Sea groups: elevation-based assignment ────────────────────────────
    # Using polygon containment on the convex hull geometry ring is unreliable
    # because the hull may include land vertices with positive Z, pushing the
    # group's Z std dev above SEA_STD_THRESHOLD and preventing displacement.
    # Instead we assign directly by terrain vertex elevation — consistent with
    # the elevation threshold used in master.py to generate sea H3 cells.
    SEA_Z_THRESHOLD = 1.0 * Z_SCALE   # 1 metre in Blender units
    sea_assigned = {}

    for feat in sorted(sea_feats or [],
                       key=lambda f: (f.get("properties") or {}).get("sea_index", 0)):
        idx   = (feat.get("properties") or {}).get("sea_index", len(sea_assigned))
        label = chr(65 + idx)

        indices = [v.index for v in terrain_obj.data.vertices
                   if v.co.z <= SEA_Z_THRESHOLD]
        if indices:
            vg = terrain_obj.vertex_groups.new(name=label)
            vg.add(indices, 1.0, 'REPLACE')
            sea_claimed.update(indices)
            sea_assigned[label] = len(indices)

    # ── Water body groups: nearest-neighbour from water mesh vertices ─────
    # For each unique triangle vertex in the water body mesh, find the
    # nearest terrain vertex. This directly maps "the same XY positions as
    # the water body" onto the terrain regardless of grid resolution, so
    # narrow features (rivers, canals) are fully covered.
    #
    # Threshold: half a typical terrain grid step (~30 m → 0.000270 deg).
    # A terrain vertex is claimed if any water mesh vertex is within
    # WATER_NN_THRESHOLD degrees of it.
    WATER_NN_THRESHOLD = 0.0003   # ≈ 33 m

    water_assigned = {}
    try:
        import numpy as _np
        from scipy.spatial import cKDTree as _cKDTree

        t_arr      = _np.array(t_lonlat)
        terrain_kd = _cKDTree(t_arr)

        for i, feat in enumerate(water_feats or []):
            label = f"W{i+1}"
            tris  = (feat.get("properties") or {}).get("triangles", [])
            if not tris:
                continue

            # Collect unique lon/lat positions from the triangle vertices
            w_set = set()
            for tri in tris:
                for pt in tri[:3]:
                    w_set.add((round(pt[0], 7), round(pt[1], 7)))
            if not w_set:
                continue

            w_arr          = _np.array(list(w_set))
            dists, indices = terrain_kd.query(w_arr, k=1)

            # Do NOT exclude sea_claimed here — a sea polygon may overlap
            # inland water bodies (e.g. Port Phillip Bay polygon covering
            # Albert Park Lake in Melbourne).  Water body groups are
            # processed BEFORE sea groups in extrude_flat_groups, which
            # then skips already-water-modified vertices in the sea pass.
            claimed = list({int(idx) for d, idx in zip(dists, indices)
                            if d < WATER_NN_THRESHOLD})
            if claimed:
                vg = terrain_obj.vertex_groups.new(name=label)
                # Assign per-vertex weights based on Shapely distance to polygon
                # boundary — ONLY for terrain vertices that lie INSIDE the polygon.
                #
                # Why inside-only?  The NN threshold (~33 m) may capture terrain
                # vertices that are just outside the polygon edge.  Those must get
                # weight = 0 so no terrain displacement appears outside the water
                # body.  Vertices exactly at the edge (distance ≈ 0) also get
                # weight ≈ 0 (border vertices don't move).  Only interior vertices
                # receive a cosine-falloff weight that peaks at 1.0 at the centre.
                #
                # This also works for narrow features (rivers, canals) where every
                # terrain vertex is a BFS grid-boundary vertex: the geometric
                # distance to the polygon edge still varies across the width.
                try:
                    from shapely.ops import unary_union as _uu

                    # ── Primary: build containment geometry from the H3
                    # water triangles themselves.  This is more reliable than
                    # reconstructing from the GeoJSON ring coordinates because:
                    #  • The H3 tris ARE the water mesh — exact match guaranteed
                    #  • No winding-order or polygon-validity surprises
                    #  • Islands inside the lake are naturally excluded (their
                    #    H3 cells were not tagged as water, so no triangles there)
                    tri_shapes = []
                    for tri in tris:
                        if len(tri) >= 3:
                            tri_shapes.append(
                                ShapelyPolygon([(tri[j][0], tri[j][1])
                                               for j in range(3)])
                            )

                    wpoly   = _uu(tri_shapes) if tri_shapes else None
                    wbnd    = wpoly.boundary  if wpoly   else None
                    prep_wp = _prep(wpoly)    if wpoly   else None

                    # ── Fallback: GeoJSON polygon ring (used when wpoly is
                    # None, or as cross-check when inside_dists is empty)
                    geom       = feat.get("geometry", {})
                    coords     = geom.get("coordinates") or []
                    outer_ring = coords[0] if coords else []
                    hole_rings = coords[1:] if len(coords) > 1 else []
                    if len(outer_ring) >= 3:
                        wpoly_ring = ShapelyPolygon(
                            [(p[0], p[1]) for p in outer_ring],
                            [[(p[0], p[1]) for p in h] for h in hole_rings],
                        )
                    else:
                        wpoly_ring = None

                    # If unary_union failed, fall back to the ring polygon
                    if wpoly is None and wpoly_ring is not None:
                        wpoly   = wpoly_ring
                        wbnd    = wpoly.boundary
                        prep_wp = _prep(wpoly)

                    if prep_wp is not None:
                        inside_dists = {}
                        for vi in claimed:
                            pt = Point(t_lonlat[vi][0], t_lonlat[vi][1])
                            if prep_wp.covers(pt):
                                inside_dists[vi] = wbnd.distance(pt)

                        # If triangle-union gave 0 inside verts, retry with
                        # the ring polygon — useful when Shapely's unary_union
                        # produces a geometry whose boundary doesn't quite
                        # cover the terrain vertex positions.
                        if not inside_dists and wpoly_ring is not None:
                            prep_ring = _prep(wpoly_ring)
                            ring_bnd  = wpoly_ring.boundary
                            for vi in claimed:
                                pt = Point(t_lonlat[vi][0], t_lonlat[vi][1])
                                if prep_ring.covers(pt):
                                    inside_dists[vi] = ring_bnd.distance(pt)

                        max_d    = max(inside_dists.values()) if inside_dists else 0.0
                        non_zero = sum(1 for d in inside_dists.values() if d > 1e-12)
                        print(f"  [{terrain_obj.name}] {label}: claimed={len(claimed)}"
                              f"  inside={len(inside_dists)}  max_d={max_d:.2e}"
                              f"  non_zero_w={non_zero}")
                        if max_d > 1e-12:
                            for vi, d in inside_dists.items():
                                t = min(d / max_d, 1.0)
                                w = water_edge_falloff(t)
                                vg.add([vi], w, 'REPLACE')
                        elif inside_dists:
                            # All inside vertices exactly on boundary (max_d≈0):
                            # assign uniform weight so at least some depression
                            # is visible.
                            vg.add(list(inside_dists.keys()), 1.0, 'REPLACE')
                        # else: no inside vertices — group remains empty
                    else:
                        # No geometry could be built at all
                        vg.add(list(claimed), 1.0, 'REPLACE')
                except Exception as _exc:
                    # Shapely failed — fall back to uniform weight
                    print(f"  [{terrain_obj.name}] {label}: Shapely exception — {_exc}")
                    vg.add(list(claimed), 1.0, 'REPLACE')
                water_assigned[label] = len(claimed)

    except ImportError:
        # scipy unavailable — fall back to polygon containment
        for i, feat in enumerate(water_feats or []):
            label = f"W{i+1}"
            geom  = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            rings = geom.get("coordinates", [])
            if not rings or len(rings[0]) < 3:
                continue
            try:
                poly     = ShapelyPolygon([(p[0], p[1]) for p in rings[0]])
                prepared = _prep(poly)
            except Exception:
                continue
            indices = [i for i, (lon, lat) in enumerate(t_lonlat)
                       if i not in sea_claimed and prepared.covers(Point(lon, lat))]
            if indices:
                vg = terrain_obj.vertex_groups.new(name=label)
                vg.add(indices, 1.0, 'REPLACE')
                water_assigned[label] = len(indices)

    if sea_assigned:
        totals = ", ".join(f"{l}:{n}" for l, n in sea_assigned.items())
        print(f"  [{terrain_obj.name}]  sea groups:   {totals}")
    if water_assigned:
        totals = ", ".join(f"{l}:{n}" for l, n in water_assigned.items())
        print(f"  [{terrain_obj.name}]  water groups: {totals}")


def build_structures_mesh(structures_props, terrain_z_fn, lon_origin, lat_origin,
                          parent_obj, blender_name, collection=None):
    """
    Build extruded structure (building) meshes from a structures feature's
    properties dict (feature_type == "structures"). Uses terrain_z_fn for
    ground elevation.
    """
    structures = structures_props.get("structures", [])
    if not structures:
        return

    b_verts = []
    b_faces = []
    n_built = 0

    for bld in structures:
        footprint    = bld.get("footprint", [])
        height_m     = bld.get("height_m", 10.0)
        min_height_m = bld.get("min_height_m", 0.0)

        if len(footprint) < 3:
            continue

        base_verts = []
        for pt in footprint:
            lon, lat = pt[0], pt[1]
            x, y     = equirectangular(lon, lat, lon_origin, lat_origin)
            ground_z = terrain_z_fn(x, y) if terrain_z_fn is not None else 0.0
            z_base   = ground_z + min_height_m * Z_SCALE
            z_top    = ground_z + height_m     * Z_SCALE
            base_verts.append((x, y, z_base, z_top))

        if not base_verts:
            continue

        n_pts    = len(base_verts)
        v_offset = len(b_verts)

        for x, y, z_base, _ in base_verts:
            b_verts.append((x, y, z_base))
        for x, y, _, z_top in base_verts:
            b_verts.append((x, y, z_top))

        for i in range(n_pts):
            j = (i + 1) % n_pts
            b_faces.append((
                v_offset + i,
                v_offset + j,
                v_offset + n_pts + j,
                v_offset + n_pts + i,
            ))
        b_faces.append(tuple(v_offset + n_pts + i for i in range(n_pts)))
        n_built += 1

    if b_verts:
        bld_mesh = bpy.data.meshes.new(f"{blender_name} - Structures")
        bld_mesh.from_pydata(b_verts, [], b_faces)
        bld_mesh.update()
        bld_obj  = bpy.data.objects.new(f"{blender_name} - Structures", bld_mesh)
        _col(collection).objects.link(bld_obj)
        bld_obj.parent = parent_obj
        attach_nodegroup(bld_obj, "Structures")
        print(f"  [{blender_name} - Structures]  "
              f"{n_built} structures  {len(b_verts)} verts")


def build_terrain_lookup(terrain_props, lon_origin, lat_origin):
    """
    Build a fast bilinear Z lookup from a terrain feature's properties dict.
    Returns terrain_z(x, y) → Z in Blender units, or None if invalid.
    """
    points = terrain_props.get("points", [])
    n_lat  = terrain_props.get("n_lat", 0)
    n_lon  = terrain_props.get("n_lon", 0)

    if not points or n_lat < 2 or n_lon < 2:
        return None

    lon_min = terrain_props["lon_min"]
    lon_max = terrain_props["lon_max"]
    lat_min = terrain_props["lat_min"]
    lat_max = terrain_props["lat_max"]

    x_min, y_min = equirectangular(lon_min, lat_min, lon_origin, lat_origin)
    x_max, y_max = equirectangular(lon_max, lat_max, lon_origin, lat_origin)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Flat elevation list avoids per-call list indexing overhead
    ele_flat = [pt[2] if len(pt) > 2 and pt[2] is not None else 0.0 for pt in points]

    def terrain_z(px, py):
        col_f = max(0.0, min(n_lon - 1,
                    (px - x_min) / x_range * (n_lon - 1) if x_range else 0.0))
        row_f = max(0.0, min(n_lat - 1,
                    (py - y_min) / y_range * (n_lat - 1) if y_range else 0.0))
        col0 = int(col_f); col1 = min(col0 + 1, n_lon - 1)
        row0 = int(row_f); row1 = min(row0 + 1, n_lat - 1)
        fc = col_f - col0; fr = row_f - row0
        z = (ele_flat[row0 * n_lon + col0] * (1 - fc) * (1 - fr) +
             ele_flat[row0 * n_lon + col1] *       fc  * (1 - fr) +
             ele_flat[row1 * n_lon + col0] * (1 - fc) *       fr  +
             ele_flat[row1 * n_lon + col1] *       fc  *       fr)
        return max(z, 0.0) * Z_SCALE

    if HAVE_NUMPY:
        import numpy as _np
        _ele = _np.array(ele_flat).reshape(n_lat, n_lon)
        _xr  = x_range or 1.0
        _yr  = y_range or 1.0

        def terrain_z_batch(xs, ys):
            col_f = _np.clip((xs - x_min) / _xr * (n_lon - 1), 0.0, n_lon - 1)
            row_f = _np.clip((ys - y_min) / _yr * (n_lat - 1), 0.0, n_lat - 1)
            col0  = col_f.astype(_np.int32)
            col1  = _np.minimum(col0 + 1, n_lon - 1)
            row0  = row_f.astype(_np.int32)
            row1  = _np.minimum(row0 + 1, n_lat - 1)
            fc = col_f - col0; fr = row_f - row0
            z = (_ele[row0, col0] * (1 - fc) * (1 - fr) +
                 _ele[row0, col1] *       fc  * (1 - fr) +
                 _ele[row1, col0] * (1 - fc) *       fr  +
                 _ele[row1, col1] *       fc  *       fr)
            return _np.maximum(z, 0.0) * Z_SCALE

        terrain_z.batch = terrain_z_batch

    return terrain_z


def project_spline_to_terrain(obj, terrain_z_fn):
    """
    For each point on every spline of a curve object, replace Z with
    the terrain surface Z sampled at that point's XY position.
    """
    if terrain_z_fn is None:
        return
    for spline in obj.data.splines:
        for pt in spline.points:
            x, y = pt.co.x, pt.co.y
            z    = terrain_z_fn(x, y)
            pt.co = (x, y, z, 1.0)
    obj.data.update_tag()




def add_sector_type_driver(obj, mod):
    """
    Add a driver to the 'Sector Type' input of a Track Profile modifier
    so it reads obj["sector_type"] automatically.
    """
    try:
        ng = mod.node_group
        if ng is None:
            return
        socket_id = None
        for item in ng.interface.items_tree:
            if item.name == "Sector Type" and item.in_out == 'INPUT':
                socket_id = item.identifier
                break
        if socket_id is None:
            return
        data_path = f'modifiers["Track Profile"]["{socket_id}"]'
        obj.driver_remove(data_path)
        drv              = obj.driver_add(data_path).driver
        drv.type         = 'SCRIPTED'
        drv.expression   = 'var'
        var              = drv.variables.new()
        var.name         = 'var'
        var.type         = 'SINGLE_PROP'
        target           = var.targets[0]
        target.id_type   = 'OBJECT'
        target.id        = obj
        target.data_path = '["sector_type"]'
    except Exception as e:
        print(f"    [driver] could not add sector_type driver: {e}")


def make_curve_object(name, linestrings, lon_origin, lat_origin,
                      properties, is_srtm):
    curve_data                  = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions       = '3D'
    curve_data.bevel_depth      = BEVEL_DEPTH
    curve_data.bevel_resolution = 3
    curve_data.fill_mode        = 'FULL'
    curve_data.twist_mode       = 'Z_UP'

    for coords in linestrings:
        if len(coords) < 2:
            continue

        coords    = resample_fixed_spacing(coords, POINT_RESOLUTION_M)
        has_width = len(coords[0]) > 3
        n         = len(coords)
        spline = curve_data.splines.new(CURVE_TYPE)
        spline.points.add(n - 1)

        for i, coord in enumerate(coords):
            lon = coord[0]
            lat = coord[1]
            x, y = equirectangular(lon, lat, lon_origin, lat_origin)
            z    = (coord[2] * Z_SCALE if len(coord) > 2
                    and coord[2] is not None else 0.0)
            spline.points[i].co     = (x, y, z, 1.0)
            spline.points[i].tilt   = 0.0
            spline.points[i].radius = (coord[3] if has_width and len(coord) > 3
                                       and coord[3] is not None
                                       else DEFAULT_WIDTH_M)

        if CURVE_TYPE == 'NURBS':
            spline.use_endpoint_u = True
            spline.use_cyclic_u   = True
            spline.order_u        = min(4, n)

    obj = bpy.data.objects.new(name=name, object_data=curve_data)

    for key, value in properties.items():
        obj[key] = value

    ng = bpy.data.node_groups.get("Track Profile")
    if ng is None:
        raise RuntimeError("'Track Profile' geometry node group not found — please create it before importing")
    mod      = obj.modifiers.new(name="Track Profile", type='NODES')
    if ng:
        mod.node_group = ng
        add_sector_type_driver(obj, mod)

    return obj


def resample_fixed_spacing(coords, spacing_m):
    """
    Resample a coordinate list at fixed spacing in metres along the curve,
    always preserving the exact start and end points.

    For a curve of length L with spacing S, the output points are at
    arc-length distances: 0, S, 2S, ... floor(L/S)*S, L

    coords : list of [lon, lat, z, ...] — any extra elements are
             linearly interpolated along with lon/lat
    spacing_m : target spacing in metres (e.g. 10.0)
    """
    if len(coords) < 2:
        return list(coords)

    DEG_TO_M = 111_320.0
    cos_lat  = math.cos(math.radians(coords[0][1]))

    if HAVE_NUMPY:
        arr  = np.array([[c[k] if c[k] is not None else 0.0
                          for k in range(len(c))] for c in coords],
                        dtype=float)
        dlon = np.diff(arr[:, 0]) * cos_lat * DEG_TO_M
        dlat = np.diff(arr[:, 1]) * DEG_TO_M
        segs = np.sqrt(dlon**2 + dlat**2)
        dists = np.concatenate([[0.0], np.cumsum(segs)])
    else:
        arr   = None
        dists_list = [0.0]
        for i in range(1, len(coords)):
            dlon = (coords[i][0] - coords[i-1][0]) * cos_lat * DEG_TO_M
            dlat = (coords[i][1] - coords[i-1][1]) * DEG_TO_M
            dists_list.append(dists_list[-1] + math.sqrt(dlon*dlon + dlat*dlat))
        dists = dists_list

    total = float(dists[-1])
    if total == 0:
        return list(coords)

    # Build target distances
    targets = list(np.arange(0.0, total, spacing_m)) if HAVE_NUMPY \
              else [i * spacing_m for i in range(int(total / spacing_m) + 1)
                   if i * spacing_m < total - 1e-6]
    targets.append(total)

    def interp(t_target):
        if HAVE_NUMPY:
            j    = max(0, np.searchsorted(dists, t_target, side='right') - 1)
            j    = min(j, len(coords) - 2)
        else:
            j = 0
            while j < len(dists) - 2 and dists[j+1] < t_target:
                j += 1
        seg  = float(dists[j+1]) - float(dists[j])
        frac = (t_target - float(dists[j])) / seg if seg > 0 else 0.0
        if arr is not None:
            return list(arr[j] + frac * (arr[j+1] - arr[j]))
        result = []
        for k in range(len(coords[j])):
            v0 = coords[j][k]   if coords[j][k]   is not None else 0.0
            v1 = coords[j+1][k] if coords[j+1][k] is not None else 0.0
            result.append(v0 + frac * (v1 - v0))
        return result

    out = [list(coords[0])]
    for t in targets[1:-1]:
        out.append(interp(t))
    out.append(list(coords[-1]))
    return out


def add_sector_type_driver(obj, mod):
    """
    Add a driver to the 'Sector Type' input of a Track Profile modifier
    so it reads obj["sector_type"] automatically.
    """
    try:
        ng = mod.node_group
        if ng is None:
            return

        socket_id = None
        for item in ng.interface.items_tree:
            if item.name == "Sector Type" and item.in_out == 'INPUT':
                socket_id = item.identifier
                break

        if socket_id is None:
            return

        data_path = f'modifiers["Track Profile"]["{socket_id}"]'
        obj.driver_remove(data_path)

        # Add new driver
        drv       = obj.driver_add(data_path).driver
        drv.type  = 'SCRIPTED'
        drv.expression = 'var'

        var             = drv.variables.new()
        var.name        = 'var'
        var.type        = 'SINGLE_PROP'
        target          = var.targets[0]
        target.id_type  = 'OBJECT'
        target.id       = obj
        target.data_path = '["sector_type"]'

    except Exception as e:
        print(f"    [driver] could not add sector_type driver: {e}")



def build_track_curve(name, feat, circuit_id, feature_type, sector_type,
                      lon_origin, lat_origin, terrain_z_fn, track_pts,
                      parent_obj, blender_name, nurbs_order=4, collection=None):
    """
    Build a curve object from a LineString feature.
    Coordinates are used as-is from the GeoJSON — no resampling,
    no transformation, no interpolation.
    Z is taken from terrain_z_fn if available, otherwise from the
    coordinate's own Z value.
    """
    linestrings = extract_linestrings(
        {"type": "FeatureCollection", "features": [feat]})
    if not linestrings:
        return

    curve             = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions  = '3D'
    curve.bevel_depth = BEVEL_DEPTH
    curve.twist_mode  = 'Z_UP'

    for ls in linestrings:
        if len(ls) < 2:
            continue

        ls     = resample_fixed_spacing(ls, POINT_RESOLUTION_M)
        n      = len(ls)
        spline = curve.splines.new(CURVE_TYPE)
        spline.points.add(n - 1)

        for i, coord in enumerate(ls):
            px, py = equirectangular(coord[0], coord[1], lon_origin, lat_origin)
            if terrain_z_fn is not None:
                best_z = terrain_z_fn(px, py)
            else:
                best_z = (coord[2] if len(coord) > 2 and coord[2] is not None
                          else 0.0) * Z_SCALE
            spline.points[i].co     = (px, py, best_z, 1.0)
            spline.points[i].tilt   = 0.0
            spline.points[i].radius = (coord[3] if len(coord) > 3
                                       and coord[3] is not None
                                       else DEFAULT_WIDTH_M)

        if CURVE_TYPE == 'NURBS':
            spline.use_endpoint_u = True
            spline.order_u        = min(nurbs_order, n)

    obj = bpy.data.objects.new(name, curve)
    obj["circuit_id"]   = circuit_id
    obj["feature_type"] = feature_type
    obj["sector_type"]  = sector_type
    _col(collection).objects.link(obj)
    obj.parent = parent_obj

    ng = bpy.data.node_groups.get("Track Profile")
    if ng is None:
        print(f"  WARNING: 'Track Profile' node group not found — skipping modifier on '{obj.name}'")
    else:
        mod            = obj.modifiers.new("Track Profile", type='NODES')
        mod.node_group = ng
        add_sector_type_driver(obj, mod)

    pts = sum(sp.points.__len__() for sp in curve.splines)
    print(f"  [{name}]  {pts} pts")


# =============================================================================
# MAIN
# =============================================================================

def main():
    geojson_dir = Path(GEOJSON_DIR)

    if not geojson_dir.exists():
        print(f"ERROR: GEOJSON_DIR not found: {geojson_dir}")
        print("Edit GEOJSON_DIR at the top of this script.")
        return

    geojson_files = sorted(
        f for f in geojson_dir.glob("*.geojson")
        if (not CIRCUITS_TO_IMPORT or f.stem in CIRCUITS_TO_IMPORT)
        and f.stem not in CIRCUITS_EXCLUDE
    )

    if not geojson_files:
        print(f"ERROR: No .geojson files found in {geojson_dir}")
        return

    print(f"Found {len(geojson_files)} circuit files.")
    print(f"Scipy available: {HAVE_SCIPY} — {'cubic spline' if HAVE_SCIPY else 'linear'} interpolation")
    print(f"Elevation read from coord[2] — telemetry circuits bypass smoothing")
    print(f"SRTM smoothing: spike={SPIKE_THRESHOLD}m  window={SMOOTH_WINDOW}pts")
    print(f"End blending: {BLEND_FRACTION*100:.0f}% of points at each end\n")

    # Delete existing empty and all children if requested.
    # Name is determined per circuit so we defer deletion to inside the loop.

    imported      = 0
    skipped       = 0
    telem_count   = 0
    srtm_count    = 0

    for filepath in geojson_files:
        circuit_id = filepath.stem

        try:
            with open(filepath, encoding="utf-8") as f:
                geojson = json.load(f)
        except Exception as e:
            print(f"  [{circuit_id}] SKIP — {e}")
            skipped += 1
            continue

        # Index all features by feature_type up front
        features_by_type = {}
        for feat in (geojson.get("features", [])
                     if geojson.get("type") == "FeatureCollection"
                     else []):
            ft = (feat.get("properties") or {}).get("feature_type", "")
            features_by_type.setdefault(ft, []).append(feat)

        # Extract linestrings from the circuit feature only
        circuit_geojson = {"type": "FeatureCollection",
                           "features": features_by_type.get("circuit", [])}
        linestrings = extract_linestrings(circuit_geojson)

        if not linestrings:
            print(f"  [{circuit_id}] SKIP — no LineString geometry in circuit feature")
            skipped += 1
            continue

        all_coords             = [c for ls in linestrings for c in ls]
        lon_origin, lat_origin = centroid(all_coords)

        _, circuit_name, location_name, elevation_source = \
            find_feature_properties(circuit_geojson)

        if not circuit_name:
            circuit_name = circuit_id

        blender_name = circuit_name if (circuit_name and circuit_name != circuit_id) else circuit_id

        # Delete existing empty with this circuit's name and all its children.
        # Try blender_name first, then circuit_id as fallback in case the
        # object was created under a previous naming scheme.
        if CLEAR_EXISTING:
            to_delete = set()

            # 1. Parent-child tree (catches properly parented objects)
            existing = (bpy.data.objects.get(blender_name) or
                        bpy.data.objects.get(circuit_id))
            if existing is not None:
                def collect_children(o):
                    to_delete.add(o)
                    for child in o.children:
                        collect_children(child)
                collect_children(existing)

            # 2. All objects in any Monaco collection (catches orphaned objects
            #    that are in the collection but not in the parent-child tree,
            #    e.g. curves left over from a failed or earlier import run).
            col_names = [
                blender_name,
                f"{blender_name} - Circuit",
                f"{blender_name} - Terrain",
                f"{blender_name} - Structures",
                f"{blender_name} - Water Bodies",
                f"{blender_name} - Roads",
                f"{blender_name} - Vegetation",
            ]
            for col_name in col_names:
                col = bpy.data.collections.get(col_name)
                if col:
                    for obj in list(col.objects):
                        to_delete.add(obj)

            if to_delete:
                for o in to_delete:
                    try:
                        bpy.data.objects.remove(o, do_unlink=True)
                    except Exception:
                        pass
                print(f"Cleared {len(to_delete)} existing objects.\n")

        # Create per-circuit collection hierarchy
        main_col    = get_or_create_collection(blender_name, _col(None))
        set_collection_visibility(blender_name, imported == 0)
        circuit_col = get_or_create_collection(f"{blender_name} - Circuit",      main_col)
        terrain_col = get_or_create_collection(f"{blender_name} - Terrain",      main_col)
        struct_col  = get_or_create_collection(f"{blender_name} - Structures",   main_col)
        water_col   = get_or_create_collection(f"{blender_name} - Water Bodies", main_col)
        roads_col   = get_or_create_collection(f"{blender_name} - Roads",        main_col)
        veg_col     = get_or_create_collection(f"{blender_name} - Vegetation",   main_col)

        # Create empty named after the circuit, placed in the main collection
        parent_empty = bpy.data.objects.new(blender_name, None)
        parent_empty.empty_display_type = 'PLAIN_AXES'
        parent_empty.empty_display_size = 2.0
        main_col.objects.link(parent_empty)

        is_srtm = (elevation_source == "srtm")

        raw_eles = [c[2] for c in all_coords
                    if len(c) > 2 and c[2] is not None]
        ele_tag  = (f"{elevation_source}  "
                    f"{min(raw_eles):.1f}m–{max(raw_eles):.1f}m"
                    if raw_eles else f"{elevation_source}  no-elev")

        if is_srtm:
            srtm_count += 1
        else:
            telem_count += 1

        properties = {
            "circuit_id":        circuit_id,
            "circuit_name":      circuit_name,
            "location":          location_name,
            "lon_origin":        lon_origin,
            "lat_origin":        lat_origin,
            "elevation_source":  elevation_source,
            "sector_type":       0,
        }

        # Track curve named "<circuit_name> - Circuit"
        obj = make_curve_object(
            name        = f"{blender_name} - Circuit",
            linestrings = linestrings,
            lon_origin  = lon_origin,
            lat_origin  = lat_origin,
            properties  = properties,
            is_srtm     = is_srtm,
        )

        circuit_col.objects.link(obj)
        obj.parent = parent_empty
        imported  += 1

        total_pts = sum(len(ls) for ls in linestrings)
        print(f"  [{blender_name} - Circuit]  {total_pts} pts  {ele_tag}")
        terrain_z_fn  = None
        terrain_props = None
        terrain_obj   = None
        for feat in features_by_type.get("terrain", []):
            try:
                terrain_props = feat.get("properties", {})
                tc_lon = terrain_props.get("lon_centre")
                tc_lat = terrain_props.get("lat_centre")
                if tc_lon is not None:
                    print(f"  [terrain] centre:    {tc_lon:.6f}, {tc_lat:.6f}")
                    print(f"  [track]  lon_origin: {lon_origin:.6f}, {lat_origin:.6f}")

                terrain_z_fn = build_terrain_lookup(
                    terrain_props, lon_origin, lat_origin)

                if terrain_z_fn is not None:
                    lmin = terrain_props["lon_min"]; laMin = terrain_props["lat_min"]
                    lmax = terrain_props["lon_max"]; laMax = terrain_props["lat_max"]
                    x0, y0 = equirectangular(lmin, laMin, lon_origin, lat_origin)
                    x1, y1 = equirectangular(lmax, laMax, lon_origin, lat_origin)
                    print(f"  [terrain] XY bounds: ({x0:.3f},{y0:.3f}) → ({x1:.3f},{y1:.3f})")

                # Build construction quad mesh — used for Z lookup only
                terrain_obj = build_terrain_mesh(terrain_props, lon_origin, lat_origin,
                                                 parent_empty, blender_name,
                                                 collection=terrain_col)

                # Mark construction quad mesh as _tmp before building hex
                # so the hex mesh can take the clean name
                if terrain_obj is not None:
                    terrain_obj.name = f"{blender_name} - Terrain_tmp"
                    if terrain_obj.data:
                        terrain_obj.data.name = f"{blender_name} - Terrain_tmp"

                # Build H3 hex terrain mesh if triangles are available
                terrain_tris = terrain_props.get("triangles", [])
                if terrain_tris and terrain_z_fn is not None:
                    hex_name = f"{blender_name} - Terrain"
                    n_tris   = len(terrain_tris)
                    print(f"  [{hex_name}]  building {n_tris} tris ...",
                          end=" ", flush=True)
                    import bmesh as _bm

                    cos_lat = _cos_lat_cache.get(lat_origin) or \
                              math.cos(math.radians(lat_origin))

                    if HAVE_NUMPY:
                        import numpy as _np
                        raw  = _np.array([[pt[0], pt[1]]
                                          for tri in terrain_tris for pt in tri])
                        xs   = (raw[:, 0] - lon_origin) * cos_lat * SCALE
                        ys   = (raw[:, 1] - lat_origin) * SCALE
                        batch = getattr(terrain_z_fn, 'batch', None)
                        zs    = batch(xs, ys) if batch is not None else \
                                _np.array([terrain_z_fn(x, y) for x, y in zip(xs, ys)])
                        all_verts = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
                    else:
                        all_verts = []
                        for tri in terrain_tris:
                            for pt in tri:
                                px = (pt[0] - lon_origin) * cos_lat * SCALE
                                py = (pt[1] - lat_origin) * SCALE
                                all_verts.append((px, py, terrain_z_fn(px, py)))

                    all_faces = [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(n_tris)]

                    print(f"{len(all_verts)} verts, creating mesh ...",
                          end=" ", flush=True)
                    me = bpy.data.meshes.new(hex_name)
                    me.from_pydata(all_verts, [], all_faces)
                    me.update()
                    print(f"merging doubles ...", end=" ", flush=True)

                    bm = _bm.new()
                    bm.from_mesh(me)
                    _bm.ops.remove_doubles(bm, verts=bm.verts,
                                           dist=0.01 * Z_SCALE)
                    bm.to_mesh(me)
                    bm.free()
                    me.update()

                    hex_obj = bpy.data.objects.new(hex_name, me)
                    hex_obj["circuit_id"]   = circuit_id
                    hex_obj["feature_type"] = "terrain"
                    terrain_col.objects.link(hex_obj)
                    hex_obj.parent = parent_empty
                    attach_nodegroup(hex_obj, "Terrain")
                    print(f"done  ({len(me.vertices)} verts after merge)")
                    print(f"  [{hex_name}]  {len(terrain_tris)} tris  "
                          f"{len(me.vertices)} verts (merged)")

                    terrain_obj = hex_obj

            except Exception as e:
                print(f"  [{blender_name} - Terrain]  SKIP — {e}")

        # Project circuit onto terrain now that terrain_z_fn is available
        if terrain_z_fn is not None:
            project_spline_to_terrain(obj, terrain_z_fn)

        # ── Structures ─────────────────────────────────────────────────
        for feat in features_by_type.get("structures", []):
            try:
                build_structures_mesh(
                    feat.get("properties", {}), terrain_z_fn,
                    lon_origin, lat_origin, parent_empty, blender_name,
                    collection=struct_col)
            except Exception as e:
                print(f"  [{blender_name} - Structures]  SKIP — {e}")

        # ── Step 2: Build water body meshes (flat at Z=0) ─────────────
        # Each entry: (mesh_obj, is_sea, ring, water_tag)
        # Depth is now computed inside _make_water_depth_offset_fn from
        # the polygon's maximum inscribed radius (step 4a).
        all_water_data = []
        water_tag_counts = {}

        for feat in features_by_type.get("sea", []):
            try:
                ring  = ((feat.get("geometry") or {}).get("coordinates") or [[]])[0]
                w_obj = build_sea_curve(feat, lon_origin, lat_origin,
                                        parent_empty, blender_name,
                                        ring=ring, collection=water_col)
                if w_obj is not None:
                    all_water_data.append((w_obj, True, ring, "sea"))
            except Exception as e:
                print(f"  [{blender_name} - Sea]  SKIP - {e}")

        for feat in features_by_type.get("water_body", []):
            try:
                water_tag = (feat.get("properties") or {}).get("water_tag", "water")
                ring      = ((feat.get("geometry") or {}).get("coordinates") or [[]])[0]
                w_obj     = build_water_curve(feat, lon_origin, lat_origin,
                                              parent_empty, blender_name,
                                              water_tag_counts,
                                              ring=ring, collection=water_col)
                if w_obj is not None:
                    all_water_data.append((w_obj, False, ring, water_tag))
            except Exception as e:
                print(f"  [{blender_name} - water]  SKIP - {e}")

        # ── Step 3: Project water bodies onto terrain ──────────────────
        # Use terrain_z_fn per vertex for ALL water bodies (sea and inland).
        # terrain_z_fn clamps to max(DEM, 0.0), so open-sea vertices land at
        # Z=0 naturally. Boundary H3 hexagons that straddle the polygon edge
        # will have land-side vertices at terrain elevation and water-side
        # vertices at Z=0 — the depth formula handles the boundary smoothly
        # via KD-tree proximity rather than a hard polygon containment cutoff.
        for w_obj, _is_sea, _ring, _depth in all_water_data:
            try:
                project_water_mesh_to_terrain(w_obj, terrain_z_fn)
            except Exception as e:
                print(f"  [{w_obj.name}]  projection SKIP - {e}")

        # ── Step 4a: Build depth functions from projected meshes ────────
        # Pass bbox so that bbox edges are excluded from the falloff boundary —
        # only natural shoreline/coastline segments drive the edge taper.
        _tp   = terrain_props or {}
        _bbox = (
            _tp.get("lon_min"), _tp.get("lat_min"),
            _tp.get("lon_max"), _tp.get("lat_max"),
        ) if all(_tp.get(k) is not None for k in
                 ("lon_min", "lat_min", "lon_max", "lat_max")) else None

        depth_fns = []
        for w_obj, _is_sea, ring, water_tag in all_water_data:
            try:
                depth_fn = _make_water_depth_offset_fn(
                    ring, water_tag, lon_origin, lat_origin,
                    bbox=_bbox)
                if depth_fn is not None:
                    depth_fns.append(depth_fn)
            except Exception as e:
                print(f"  [{w_obj.name}]  depth_fn SKIP - {e}")

        boundary_feats = features_by_type.get("terrain_boundary", [])
        if boundary_feats:
            for feat in boundary_feats:
                try:
                    props = feat.get("properties", {})
                    tris  = props.get("triangles", [])
                    if not tris:
                        continue
                    name  = f"{blender_name} - Terrain Boundary"

                    all_verts = []
                    all_faces = []
                    for tri in tris:
                        base = len(all_verts)
                        for pt in tri:
                            px, py = equirectangular(pt[0], pt[1],
                                                     lon_origin, lat_origin)
                            pz     = pt[2] * Z_SCALE if len(pt) > 2 else 0.0
                            all_verts.append((px, py, pz))
                        all_faces.append((base, base + 1, base + 2))

                    import bmesh as _bm2
                    me = bpy.data.meshes.new(name)
                    me.from_pydata(all_verts, [], all_faces)
                    me.update()

                    bm = _bm2.new()
                    bm.from_mesh(me)
                    _bm2.ops.remove_doubles(bm, verts=bm.verts, dist=0.01 * Z_SCALE)

                    # Step 1: set all ring verts to terrain absolute min Z
                    # (after sea depression has been applied)
                    if terrain_obj is not None:
                        z_min_terrain = min(v.co.z for v in terrain_obj.data.vertices)
                    else:
                        ele_min = (feat.get("properties", {}).get("ele_min_m")
                                   or (terrain_props or {}).get("ele_min_m", 0.0))
                        z_min_terrain = float(ele_min) * Z_SCALE

                    for v in bm.verts:
                        v.co.z = z_min_terrain

                    # Step 2: build KDTree for fast nearest terrain vertex lookup
                    if terrain_obj is not None:
                        t_data = terrain_obj.data.vertices
                        t_xy   = [(v.co.x, v.co.y) for v in t_data]
                        t_z    = [v.co.z             for v in t_data]
                        try:
                            from scipy.spatial import cKDTree
                            tree     = cKDTree(t_xy)
                            have_kd  = True
                        except ImportError:
                            have_kd = False
                    else:
                        t_xy = t_z = None
                        have_kd    = False

                    # Step 3: extrude — top vert Z = nearest terrain Z + 0.1m
                    bm.verts.ensure_lookup_table()
                    top_offset = 0.1 * Z_SCALE
                    new_top    = {}

                    if have_kd:
                        ring_xy   = [(v.co.x, v.co.y) for v in bm.verts]
                        _, idxs   = tree.query(ring_xy)
                        top_zs    = [t_z[i] + top_offset for i in idxs]
                    else:
                        top_zs = None

                    for i, v in enumerate(list(bm.verts)):
                        top_z = top_zs[i] if top_zs is not None \
                                else z_min_terrain + top_offset
                        tv = bm.verts.new((v.co.x, v.co.y, top_z))
                        new_top[v.index] = tv

                    # Build side quads connecting bottom ring to top ring
                    # along boundary edges (1 face neighbour)
                    bm.edges.ensure_lookup_table()
                    boundary_edges = [e for e in bm.edges
                                      if len(e.link_faces) == 1]

                    for edge in boundary_edges:
                        v0 = edge.verts[0]; v1 = edge.verts[1]
                        t0 = new_top[v0.index]; t1 = new_top[v1.index]
                        try:
                            bm.faces.new([v0, v1, t1, t0])
                        except Exception:
                            pass

                    # Add top faces by duplicating original faces with top verts
                    orig_faces = [f for f in bm.faces
                                  if all(v.index in new_top for v in f.verts)]
                    for f in orig_faces:
                        top_verts = [new_top[v.index] for v in f.verts]
                        try:
                            bm.faces.new(top_verts)
                        except Exception:
                            pass

                    bm.normal_update()
                    bm.to_mesh(me)
                    bm.free()
                    me.update()

                    bnd_obj = bpy.data.objects.new(name, me)
                    bnd_obj["circuit_id"]   = circuit_id
                    bnd_obj["feature_type"] = "terrain_boundary"
                    terrain_col.objects.link(bnd_obj)
                    bnd_obj.parent = parent_empty

                    vg_bot = bnd_obj.vertex_groups.new(name="Bottom")
                    bot_vi = [v.index for v in me.vertices
                              if abs(v.co.z - z_min_terrain) < 1e-6]
                    vg_bot.add(bot_vi, 1.0, 'REPLACE')

                    attach_nodegroup(bnd_obj, "Wall")
                    print(f"  [{name}]  {len(tris)} tris  "
                          f"{len(me.vertices)} verts (merged)")
                except Exception as e:
                    print(f"  [{blender_name} - Terrain Boundary]  SKIP — {e}")
        else:
            # Fallback: rectangular wall until GeoJSON is regenerated
            try:
                build_bounding_wall(terrain_props, lon_origin, lat_origin,
                                    parent_empty, blender_name,
                                    collection=terrain_col)
            except Exception as e:
                print(f"  [{blender_name} - Boundary Wall]  SKIP — {e}")


        # ── Apply water depth to terrain ───────────────────────────────────
        if terrain_obj is not None:
            try:
                apply_water_depth_to_terrain(terrain_obj, depth_fns, blender_name)

                # Pin wall bottom vertices to the terrain's new minimum Z.
                z_min_final = min(v.co.z for v in terrain_obj.data.vertices)
                wall_obj = (bpy.data.objects.get(f"{blender_name} - Terrain Boundary")
                            or bpy.data.objects.get(f"{blender_name} - Boundary Wall"))
                if wall_obj is not None:
                    vg = wall_obj.vertex_groups.get("Bottom")
                    if vg is not None:
                        for v in wall_obj.data.vertices:
                            if any(g.group == vg.index for g in v.groups):
                                v.co.z = z_min_final
                        wall_obj.data.update()
                        print(f"  [{wall_obj.name}]  bottom pinned to z={z_min_final:.5f}")
            except Exception as e:
                print(f"  [{blender_name} - water depth]  SKIP - {e}")

        # Collect track spline points for Z fallback when terrain unavailable.
        track_pts = []
        for spline in obj.data.splines:
            for pt in spline.points:
                track_pts.append((pt.co.x, pt.co.y, pt.co.z))

        # Ensure Track Profile node group exists once per import run

        # ── Pit lane ───────────────────────────────────────────────────
        for feat in features_by_type.get("pitlane", []):
            try:
                build_track_curve(
                    f"{blender_name} - Pit", feat, circuit_id,
                    "pitlane", 0,
                    lon_origin, lat_origin, terrain_z_fn, track_pts,
                    parent_empty, blender_name, nurbs_order=4,
                    collection=circuit_col)
            except Exception as e:
                print(f"  [{blender_name} - Pit]  SKIP — {e}")

        # ── Sectors ────────────────────────────────────────────────────
        for sector_num in (1, 2, 3):
            for feat in features_by_type.get(f"sector{sector_num}", []):
                try:
                    build_track_curve(
                        f"{blender_name} - S{sector_num}", feat, circuit_id,
                        f"sector{sector_num}", sector_num,
                        lon_origin, lat_origin, terrain_z_fn, track_pts,
                        parent_empty, blender_name, nurbs_order=6,
                        collection=circuit_col)
                except Exception as e:
                    print(f"  [{blender_name} - S{sector_num}]  SKIP — {e}")

        # ── Streets ────────────────────────────────────────────────────
        for feat in features_by_type.get("street", []):
            try:
                build_street_curve(feat, lon_origin, lat_origin, terrain_z_fn,
                                   parent_empty, blender_name,
                                   collection=roads_col)
            except Exception as e:
                print(f"  [{blender_name} - street]  SKIP — {e}")

        # ── Vegetation ────────────────────────────────────────────────
        veg_counts = {}
        for feat in features_by_type.get("vegetation", []):
            try:
                build_vegetation(feat, lon_origin, lat_origin, terrain_z_fn,
                                 parent_empty, blender_name, veg_counts,
                                 collection=veg_col)
            except Exception as e:
                print(f"  [{blender_name} - vegetation]  SKIP — {e}")

        # ── Purge unused data between circuits ────────────────────────
        if len(geojson_files) > 1:
            bpy.ops.outliner.orphans_purge(do_recursive=True)

    print(f"\nDone.  Imported {imported}  |  Skipped {skipped}")
    print(f"Sources: {telem_count} telemetry  |  {srtm_count} SRTM")
    print(f"SCALE={SCALE}  {CURVE_TYPE}  bevel={BEVEL_DEPTH}")

    # ── Clean up any leftover construction objects ─────────────────────────
    tmp_objs = [o for o in bpy.data.objects if "_tmp" in o.name]
    for o in tmp_objs:
        mesh = o.data if o.type == 'MESH' else None
        bpy.data.objects.remove(o, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if tmp_objs:
        print(f"Cleaned up {len(tmp_objs)} construction object(s)")


if __name__ == "__main__":
    main()
else:
    main()