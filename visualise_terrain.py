"""
visualise_terrain.py
--------------------
Visualise a *_terrain.json file as an interactive 3-D surface.
Overlays water body footprints and the circuit line if present.

Usage:
    python visualise_terrain.py                       # defaults to it-1922
    python visualise_terrain.py nl-1948
    python visualise_terrain.py au-1953
"""

import json
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output/circuits")
circuit_id   = sys.argv[1] if len(sys.argv) > 1 else "it-1922"

terrain_path = OUTPUT_DIR / f"{circuit_id}_terrain.json"
geojson_path = OUTPUT_DIR / f"{circuit_id}.geojson"

DOWNSAMPLE   = 1        # set to 2 or 3 to speed up rendering on large grids
WATER_COLOR  = "#4a90d9"
RIVER_ALPHA  = 0.85
TRACK_COLOR  = "red"
VERT_EXAG    = 3.0      # vertical exaggeration factor

# ── Load terrain ──────────────────────────────────────────────────────────────
print(f"Loading {terrain_path} ...")
with open(terrain_path) as f:
    td = json.load(f)

n_lat   = td["n_lat"]
n_lon   = td["n_lon"]
lon_min = td["lon_min"];  lon_max = td["lon_max"]
lat_min = td["lat_min"];  lat_max = td["lat_max"]
ele_min = td["ele_min_m"]; ele_max = td["ele_max_m"]
points  = td["terrain_points"]   # [[lon, lat, ele], ...]
water   = td.get("water", [])
sea     = td.get("sea")

print(f"  Grid:      {n_lat} × {n_lon}  ({n_lat*n_lon} pts)")
print(f"  Elevation: {ele_min:.1f} m – {ele_max:.1f} m")
print(f"  Water:     {len(water)} body/bodies")

# ── Build 2-D elevation grid ──────────────────────────────────────────────────
# terrain_points are stored row-major, south→north (row 0 = lat_min)
Z = np.array(
    [pt[2] if pt[2] is not None else ele_min for pt in points],
    dtype=float
).reshape(n_lat, n_lon)

# Longitude / latitude 1-D arrays for the mesh
lons = np.linspace(lon_min, lon_max, n_lon)
lats = np.linspace(lat_min, lat_max, n_lat)
LON, LAT = np.meshgrid(lons, lats)

# Project to metres relative to grid centre for better axis labels
lat_ref = (lat_min + lat_max) / 2.0
cos_lat = math.cos(math.radians(lat_ref))
R       = 6_371_000.0
m_per_deg_lat = R * math.pi / 180.0
m_per_deg_lon = m_per_deg_lat * cos_lat

X = (LON - (lon_min + lon_max) / 2.0) * m_per_deg_lon   # metres east of centre
Y = (LAT - (lat_min + lat_max) / 2.0) * m_per_deg_lat   # metres north of centre

# Downsample
s = DOWNSAMPLE
Xd, Yd, Zd = X[::s, ::s], Y[::s, ::s], Z[::s, ::s]

# Apply vertical exaggeration
Z_plot = Zd * VERT_EXAG

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
ax  = fig.add_subplot(111, projection='3d')

# Terrain surface
surf = ax.plot_surface(
    Xd, Yd, Z_plot,
    cmap=cm.terrain,
    linewidth=0,
    antialiased=False,
    alpha=0.9,
    rcount=min(100, n_lat // s),
    ccount=min(100, n_lon // s),
)

# Colour bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Elevation (m)")
# Remap colour bar ticks to real elevation (undo vert exag)
tick_vals = np.linspace(ele_min * VERT_EXAG, ele_max * VERT_EXAG, 6)
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{v/VERT_EXAG:.0f}" for v in tick_vals])

# ── Water body footprints ─────────────────────────────────────────────────────
for body in water:
    fp  = body.get("footprint", [])
    tag = body.get("water_tag", "water")
    if len(fp) < 3:
        continue

    # Close the ring
    ring = fp if fp[0] == fp[-1] else fp + [fp[0]]

    fx = [(p[0] - (lon_min + lon_max) / 2.0) * m_per_deg_lon for p in ring]
    fy = [(p[1] - (lat_min + lat_max) / 2.0) * m_per_deg_lat for p in ring]

    # Sample elevation under the footprint centre
    c_lon = sum(p[0] for p in fp) / len(fp)
    c_lat = sum(p[1] for p in fp) / len(fp)
    col_i = int(round((c_lon - lon_min) / (lon_max - lon_min) * (n_lon - 1)))
    row_i = int(round((c_lat - lat_min) / (lat_max - lat_min) * (n_lat - 1)))
    col_i = max(0, min(n_lon - 1, col_i))
    row_i = max(0, min(n_lat - 1, row_i))
    fz_base = Z[row_i, col_i] * VERT_EXAG + 1.0   # sit 1 m above terrain

    fz = [fz_base] * len(ring)
    ax.plot(fx, fy, fz, color=WATER_COLOR, linewidth=2.0, zorder=5,
            label=f"Water: {tag}")

    # Filled polygon at water level
    poly_verts = [list(zip(fx, fy, fz))]
    poly = Poly3DCollection(poly_verts, alpha=RIVER_ALPHA,
                            facecolor=WATER_COLOR, edgecolor="none")
    ax.add_collection3d(poly)

# ── Sea footprint outline ─────────────────────────────────────────────────────
if sea and len(sea) >= 2:
    sx = [(p[0] - (lon_min + lon_max) / 2.0) * m_per_deg_lon for p in sea]
    sy = [(p[1] - (lat_min + lat_max) / 2.0) * m_per_deg_lat for p in sea]
    sz = [ele_min * VERT_EXAG] * len(sea)
    ax.plot(sx, sy, sz, color="cyan", linewidth=1.5, linestyle="--",
            label="Coastline")

# ── Circuit track ─────────────────────────────────────────────────────────────
if geojson_path.exists():
    with open(geojson_path) as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if props.get("feature_type") != "circuit":
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom["coordinates"]
        tx = [(c[0] - (lon_min + lon_max) / 2.0) * m_per_deg_lon for c in coords]
        ty = [(c[1] - (lat_min + lat_max) / 2.0) * m_per_deg_lat for c in coords]
        # Sample terrain Z under each track point
        tz = []
        for c in coords:
            ci = int(round((c[0] - lon_min) / (lon_max - lon_min) * (n_lon - 1)))
            ri = int(round((c[1] - lat_min) / (lat_max - lat_min) * (n_lat - 1)))
            ci = max(0, min(n_lon - 1, ci))
            ri = max(0, min(n_lat - 1, ri))
            tz.append(Z[ri, ci] * VERT_EXAG + 2.0)   # 2 m above terrain
        ax.plot(tx, ty, tz, color=TRACK_COLOR, linewidth=2.5, zorder=10,
                label="Circuit")

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel("East  (m)")
ax.set_ylabel("North  (m)")
ax.set_zlabel(f"Elevation  (m, ×{VERT_EXAG} vert. exag.)")
ax.set_title(f"{circuit_id}  —  terrain  "
             f"({ele_min:.0f} m – {ele_max:.0f} m,  "
             f"{n_lat}×{n_lon} grid)")

handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
print("Rendering … close the window to exit.")
plt.show()
