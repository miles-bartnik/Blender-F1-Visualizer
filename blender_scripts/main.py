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
CIRCUITS_TO_IMPORT = ["nl-1948"]   # None = all circuits

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
                        terrain_z_fn=None, collection=None):
    """
    Build a single merged mesh object from a list of triangles.
    Each triangle is [[lon,lat],[lon,lat],[lon,lat]].
    If terrain_z_fn is provided, Z is projected per vertex.
    Otherwise the flat z value is used for all vertices.
    """
    import bmesh as _bmesh

    if not triangles:
        return None

    all_verts = []
    all_faces = []
    for tri in triangles:
        base = len(all_verts)
        for pt in tri:
            x, y = equirectangular(pt[0], pt[1], lon_origin, lat_origin)
            pz   = terrain_z_fn(x, y) if terrain_z_fn is not None else z
            all_verts.append((x, y, pz))
        all_faces.append((base, base + 1, base + 2))

    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(all_verts, [], all_faces)
    me.update()

    # Merge by distance — 0.01m in Blender units
    merge_dist = 0.01 * Z_SCALE
    bm = _bmesh.new()
    bm.from_mesh(me)
    _bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(obj_name, me)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = feature_type
    if extra_props:
        for k, v in extra_props.items():
            obj[k] = v
    _col(collection).objects.link(obj)
    obj.location.z = z

    world_mat        = obj.matrix_world.copy()
    obj.parent       = parent_obj
    obj.matrix_world = world_mat

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    return obj


def build_sea_curve(feat, lon_origin, lat_origin, parent_obj, blender_name, collection=None):
    """Build a sea polygon as a flat 2D NGON-filled curve at Z=0,
    plus a merged triangle mesh from H3 cells if available."""
    geom = feat.get("geometry", {})
    if not geom or geom.get("type") != "Polygon":
        return
    rings = geom.get("coordinates", [])
    if not rings or len(rings[0]) < 3:
        return
    ring      = rings[0]
    pts       = ring[:-1] if ring[0] == ring[-1] else ring
    props = feat.get("properties", {})
    name  = f"{blender_name} - Sea"

    curve            = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '2D'
    curve.fill_mode  = 'FRONT'
    spline           = curve.splines.new('POLY')
    spline.points.add(len(pts) - 1)
    spline.use_cyclic_u = True
    for i, pt in enumerate(pts):
        x, y = equirectangular(pt[0], pt[1], lon_origin, lat_origin)
        spline.points[i].co = (x, y, 0.0, 1.0)

    obj = bpy.data.objects.new(name, curve)
    obj["circuit_id"]   = blender_name
    obj["feature_type"] = "sea"
    _col(collection).objects.link(obj)
    world_mat        = obj.matrix_world.copy()
    obj.parent       = parent_obj
    obj.matrix_world = world_mat
    attach_nodegroup(obj, "Sea")

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    print(f"  [{name}]  {len(pts)} pts")

    # Build triangle mesh from H3 cells if present and above minimum hex count
    triangles = props.get("triangles", [])
    if triangles and len(triangles) // 6 >= MIN_WATER_HEX:
        mesh_name = f"{blender_name} - Sea Mesh"
        mesh_obj  = build_triangle_mesh(
            triangles, mesh_name, 0.0,
            lon_origin, lat_origin,
            "sea_mesh", parent_obj, blender_name,
            extra_props={},
            collection=collection)
        if mesh_obj:
            attach_nodegroup(mesh_obj, "Sea")
            print(f"  [{mesh_name}]  {len(triangles)} tris  "
                  f"{len(mesh_obj.data.vertices)} verts (merged)")
            # Plane was a construction object — clean up now the mesh exists
            curve_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.curves.remove(curve_data)
        else:
            print(f"  [{mesh_name}]  mesh build failed — keeping curve fallback")


def build_water_curve(feat, lon_origin, lat_origin, parent_obj, blender_name,
                      tag_counts, terrain_z_fn=None, collection=None):
    """Build a water body hex mesh. Skips entirely if no H3 triangles exist."""
    props     = feat.get("properties", {})
    water_tag = props.get("water_tag", "water")

    # Only hex meshes — skip bodies with insufficient triangle data
    triangles = props.get("triangles", [])
    if not triangles or len(triangles) // 6 < MIN_WATER_HEX:
        return

    tag_counts[water_tag] = tag_counts.get(water_tag, 0) + 1
    obj_name  = f"{blender_name} - Water - {water_tag} - {tag_counts[water_tag]}"
    mesh_name = f"{obj_name} Mesh"

    mesh_obj = build_triangle_mesh(
        triangles, mesh_name, 0.0,
        lon_origin, lat_origin,
        "water_mesh", parent_obj, blender_name,
        extra_props={"water_tag": water_tag},
        terrain_z_fn=terrain_z_fn,
        collection=collection)

    if mesh_obj:
        attach_nodegroup(mesh_obj, "Water")
        print(f"  [{mesh_name}]  {len(triangles)} tris  "
              f"{len(mesh_obj.data.vertices)} verts (merged)")
    else:
        print(f"  [{mesh_name}]  mesh build failed — skipped")


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

    # Build mesh from H3 triangles, project Z onto terrain
    import bmesh as _bm
    all_verts = []
    all_faces = []
    for tri in tris:
        base = len(all_verts)
        for pt in tri:
            px, py = equirectangular(pt[0], pt[1], lon_origin, lat_origin)
            pz     = terrain_z_fn(px, py) if terrain_z_fn else 0.0
            all_verts.append((px, py, pz))
        all_faces.append((base, base + 1, base + 2))

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

    # Precompute Blender XY for grid corners
    x_min, y_min = equirectangular(lon_min, lat_min, lon_origin, lat_origin)
    x_max, y_max = equirectangular(lon_max, lat_max, lon_origin, lat_origin)
    x_range = x_max - x_min
    y_range = y_max - y_min

    def terrain_z(px, py):
        col_f = (px - x_min) / x_range * (n_lon - 1) if x_range != 0 else 0.0
        row_f = (py - y_min) / y_range * (n_lat - 1) if y_range != 0 else 0.0

        col_f = max(0.0, min(n_lon - 1, col_f))
        row_f = max(0.0, min(n_lat - 1, row_f))

        col0 = int(col_f); col1 = min(col0 + 1, n_lon - 1)
        row0 = int(row_f); row1 = min(row0 + 1, n_lat - 1)

        fc = col_f - col0
        fr = row_f - row0

        def ele(r, c):
            idx = r * n_lon + c
            v   = points[idx][2] if idx < len(points) else None
            return v if v is not None else 0.0

        z00 = ele(row0, col0); z10 = ele(row0, col1)
        z01 = ele(row1, col0); z11 = ele(row1, col1)
        z   = (z00 * (1-fc) * (1-fr) +
               z10 *    fc  * (1-fr) +
               z01 * (1-fc) *    fr  +
               z11 *    fc  *    fr)
        return max(z, 0.0) * Z_SCALE

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
            existing = (bpy.data.objects.get(blender_name) or
                        bpy.data.objects.get(circuit_id))
            if existing is not None:
                to_delete = []
                def collect_children(o):
                    to_delete.append(o)
                    for child in o.children:
                        collect_children(child)
                collect_children(existing)
                # Use bpy.data.objects.remove rather than the operator so
                # hidden objects are deleted regardless of viewport visibility.
                for o in to_delete:
                    bpy.data.objects.remove(o, do_unlink=True)
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
                    print(f"  [{hex_name}]  building {len(terrain_tris)} tris ...",
                          end=" ", flush=True)
                    import bmesh as _bm
                    all_verts = []
                    all_faces = []
                    for tri in terrain_tris:
                        base = len(all_verts)
                        for pt in tri:
                            px, py = equirectangular(pt[0], pt[1],
                                                     lon_origin, lat_origin)
                            pz     = terrain_z_fn(px, py)
                            all_verts.append((px, py, pz))
                        all_faces.append((base, base + 1, base + 2))

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

        # ── Sea polygons ───────────────────────────────────────────────
        for feat in features_by_type.get("sea", []):
            try:
                build_sea_curve(feat, lon_origin, lat_origin,
                                parent_empty, blender_name,
                                collection=water_col)
            except Exception as e:
                print(f"  [{blender_name} - Sea]  SKIP — {e}")

        # ── Water bodies (before terrain depression so Z is unmodified) ─
        water_tag_counts = {}
        for feat in features_by_type.get("water_body", []):
            try:
                build_water_curve(feat, lon_origin, lat_origin,
                                  parent_empty, blender_name,
                                  water_tag_counts, terrain_z_fn,
                                  collection=water_col)
            except Exception as e:
                print(f"  [{blender_name} - water]  SKIP — {e}")

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


        # Sea and water body depth are baked into terrain_points by master.py;
        # no vertex group assignment or extrude_flat_groups call needed here.

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
