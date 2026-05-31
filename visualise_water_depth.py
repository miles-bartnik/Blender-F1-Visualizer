"""
visualise_water_depth.py  —  Albert Park H3 hex mesh, vertices coloured by depth
"""

import json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from shapely.geometry import Polygon as SPoly, Point, box as SBox

GEOJSON_PATH = Path(r"C:\Users\Miles\PycharmProjects\F1\output\circuits\au-1953.geojson")

# H3 resolution 9 cell diameter ~174 m.  Trim this much of the natural
# coastline back from the bbox corners so those endpoints don't anchor
# depth = 0 at the bbox intersection.
HEX_DIAMETER_DEG = 0.002   # ~222 m — just over one cell width

WATER_DEPTH_PROPORTIONAL_K = 0.1
WATER_EDGE_BLEND           = 0.25
METRES_PER_DEGREE          = 111_320.0
WATER_DEPTH_BY_TAG = {
    "sea": 50.0, "ocean": 50.0,
    "lake": 8.0, "water": 5.0, "pond": 1.5, "oxbow": 2.0,
    "lagoon": 6.0, "moat": 1.5, "reservoir": 8.0, "basin": 2.0,
    "wetland": 1.0, "river": 4.0, "riverbank": 3.0,
    "canal": 2.0, "dock": 8.0,
    "harbour": 10.0,
}
WATER_DEPTH_DEFAULT_M = 5.0


def falloff(t):
    b = WATER_EDGE_BLEND
    return np.where(t >= b, 1.0, 0.5 * (1.0 - np.cos(np.pi * t / b)))


def area_m2(ring):
    n = len(ring)
    lat_ref = sum(p[1] for p in ring) / n
    mpd = METRES_PER_DEGREE * math.cos(math.radians(lat_ref))
    a = 0.0
    for i in range(n):
        x0, y0 = ring[i][0] * mpd,         ring[i][1]        * METRES_PER_DEGREE
        x1, y1 = ring[(i+1)%n][0] * mpd,   ring[(i+1)%n][1]  * METRES_PER_DEGREE
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def depth_m_for(max_dist_m, tag):
    cap = WATER_DEPTH_BY_TAG.get(tag, WATER_DEPTH_DEFAULT_M)
    return min(cap, WATER_DEPTH_PROPORTIONAL_K * max_dist_m)


def build_natural_shore(ring, bbox):
    """
    Return a Shapely geometry representing only the natural shoreline —
    bbox edges removed, and the endpoints where the coastline meets the
    bbox trimmed back by one hex cell so they don't anchor depth = 0
    at the bbox intersection corners.
    """
    poly = SPoly([(p[0], p[1]) for p in ring])
    if not poly.is_valid:
        poly = poly.buffer(0)

    full_bnd = poly.boundary

    if bbox is None:
        return full_bnd

    lon_min, lat_min, lon_max, lat_max = bbox
    bbox_bnd = SBox(lon_min, lat_min, lon_max, lat_max).boundary

    # Remove the bbox edges from the boundary
    natural = full_bnd.difference(bbox_bnd)
    if natural.is_empty:
        return full_bnd

    # Trim the segment ends that terminate at the bbox so their endpoints
    # don't pull nearby H3 vertices to depth = 0
    trim_zone = bbox_bnd.buffer(HEX_DIAMETER_DEG)
    trimmed   = natural.difference(trim_zone)

    return trimmed if not trimmed.is_empty else natural


def vertex_depths(ring, tag, uverts, bbox=None):
    """
    Depth at each unique H3 vertex (lon, lat) using exact Shapely distance
    to the trimmed natural shoreline.

    max_dist_m (pole of inaccessibility) is derived from the H3 vertices
    themselves — the furthest any vertex is from the natural shore.  This
    is then used to compute depth_m = min(cap, k * max_dist_m), mirroring
    _make_water_depth_offset_fn exactly.

    - Natural shoreline vertices → distance ≈ 0 → depth ≈ 0
    - Interior vertices          → distance / max_dist → cosine taper
    - Bbox-intersection vertices → endpoints trimmed, so they get the same
                                   continuous depth as interior neighbours
    """
    shore = build_natural_shore(ring, bbox)

    distances = np.array([shore.distance(Point(lon, lat)) for lon, lat in uverts])

    max_dist_deg = distances.max()
    if max_dist_deg < 1e-12:
        return np.zeros(len(uverts)), 0.0, 0.0

    max_dist_m = max_dist_deg * METRES_PER_DEGREE
    depth      = depth_m_for(max_dist_m, tag)

    t = np.minimum(distances / max_dist_deg, 1.0)
    return depth * falloff(t), max_dist_m, depth


# ── load ──────────────────────────────────────────────────────────────────────
with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

by_type = {}
for feat in geojson.get("features", []):
    ft = (feat.get("properties") or {}).get("feature_type", "")
    by_type.setdefault(ft, []).append(feat)

bbox = None
for feat in by_type.get("terrain", []):
    p = feat.get("properties", {})
    if all(k in p for k in ("lon_min", "lat_min", "lon_max", "lat_max")):
        bbox = (p["lon_min"], p["lat_min"], p["lon_max"], p["lat_max"])
        print(f"bbox: {bbox}")
        break

feats = []
for feat in by_type.get("sea", []):
    p    = feat.get("properties", {})
    g    = feat.get("geometry", {})
    tris = p.get("triangles", [])
    ring = (g.get("coordinates") or [[]])[0]
    if tris and len(ring) >= 3:
        feats.append((f"Sea {chr(65+p.get('sea_index',0))}", "sea", ring, tris))

for feat in by_type.get("water_body", []):
    p    = feat.get("properties", {})
    g    = feat.get("geometry", {})
    tag  = p.get("water_tag", "water")
    tris = p.get("triangles", [])
    ring = (g.get("coordinates") or [[]])[0]
    if tris and len(ring) >= 3:
        feats.append((tag, tag, ring, tris))

# ── plot ──────────────────────────────────────────────────────────────────────
nc = min(3, len(feats))
nr = math.ceil(len(feats) / nc)
fig, axes = plt.subplots(nr, nc, figsize=(7*nc, 7*nr), squeeze=False)

for i, (label, tag, ring, tris) in enumerate(feats):
    ax = axes[i // nc][i % nc]
    d  = depth_m_for(ring, tag)

    raw = np.array([[pt[0], pt[1]] for tri in tris for pt in tri[:3]])
    uverts, inv = np.unique(np.round(raw, 7), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)

    print(f"  {label}: computing depths for {len(uverts)} verts ...", flush=True)
    depths, max_dist_m, d = vertex_depths(ring, tag, uverts, bbox=bbox)

    print(f"    max_dist={max_dist_m:.0f} m  depth={d:.2f} m  "
          f"range: {depths.min():.2f} - {depths.max():.2f} m")

    triang = mtri.Triangulation(uverts[:, 0], uverts[:, 1], faces)
    tc = ax.tripcolor(triang, depths, cmap="Blues_r",
                      vmin=0, vmax=d, shading="gouraud")
    ax.triplot(triang, color="grey", lw=0.3, alpha=0.4)
    plt.colorbar(tc, ax=ax, label="Depth (m)")
    ax.set_title(f"{label}  |  {tag}  |  max_dist={max_dist_m:.0f} m  depth={d:.1f} m", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

for j in range(len(feats), nr*nc):
    axes[j//nc][j%nc].set_visible(False)

fig.suptitle("Albert Park — H3 hex vertices coloured by depth", fontsize=11)
plt.tight_layout()
out = GEOJSON_PATH.parent / "au-1953_water_depth.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
