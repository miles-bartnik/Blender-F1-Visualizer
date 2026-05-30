"""
debug_sea_polygon.py
--------------------
Plot the sea polygon(s) and water bodies for a circuit GeoJSON to diagnose
coastline issues (e.g. stray sea fragments over Albert Park).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

GEOJSON_PATH    = Path(r"C:\Users\Miles\PycharmProjects\F1\output\circuits\au-1953.geojson")
COASTLINE_PATH  = Path(r"C:\Users\Miles\PycharmProjects\F1\data\raw\osm\au-1953_coastline.json")


def plot_ring(ax, ring, **kwargs):
    if not ring or len(ring) < 3:
        return
    xs = [p[0] for p in ring] + [ring[0][0]]
    ys = [p[1] for p in ring] + [ring[0][1]]
    ax.plot(xs, ys, **kwargs)


def main():
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Sea & water polygons — {GEOJSON_PATH.stem}", fontsize=11)

    terrain_bbox = None
    circuit_coords = []

    for feat in features:
        props = feat.get("properties") or {}
        ft    = props.get("feature_type", "")
        geom  = feat.get("geometry") or {}

        if ft == "terrain":
            terrain_bbox = (
                props.get("lon_min"), props.get("lat_min"),
                props.get("lon_max"), props.get("lat_max"),
            )

        elif ft == "circuit":
            if geom.get("type") == "LineString":
                circuit_coords = geom.get("coordinates", [])

        elif ft == "sea":
            coords = geom.get("coordinates") or []
            outer  = coords[0] if coords else []
            idx    = props.get("sea_index", 0)
            label  = chr(65 + idx)
            n_tris = len(props.get("triangles", []))
            plot_ring(ax, outer,
                      color="steelblue", linewidth=1.5,
                      label=f"Sea {label}  ({len(outer)} pts, {n_tris//6} H3 hexes)")
            ax.fill([p[0] for p in outer], [p[1] for p in outer],
                    color="steelblue", alpha=0.15)

            # Mark triangle centroids so we can see where H3 cells landed
            tris = props.get("triangles", [])
            if tris:
                cx = [sum(pt[0] for pt in tri) / 3 for tri in tris]
                cy = [sum(pt[1] for pt in tri) / 3 for tri in tris]
                ax.scatter(cx, cy, s=2, color="steelblue", alpha=0.3, zorder=3)

        elif ft == "water_body":
            coords    = geom.get("coordinates") or []
            outer     = coords[0] if coords else []
            water_tag = props.get("water_tag", "?")
            n_tris    = len(props.get("triangles", []))
            plot_ring(ax, outer,
                      color="cornflowerblue", linewidth=1.0, linestyle="--",
                      label=f"Water [{water_tag}]  ({len(outer)} pts, {n_tris//6} H3 hexes)")
            ax.fill([p[0] for p in outer], [p[1] for p in outer],
                    color="cornflowerblue", alpha=0.20)

    # Terrain bbox
    if terrain_bbox:
        lon_min, lat_min, lon_max, lat_max = terrain_bbox
        bx = [lon_min, lon_max, lon_max, lon_min, lon_min]
        by = [lat_min, lat_min, lat_max, lat_max, lat_min]
        ax.plot(bx, by, "k--", linewidth=1.0, label="Terrain bbox")

    # Circuit trace
    if circuit_coords:
        ax.plot([c[0] for c in circuit_coords],
                [c[1] for c in circuit_coords],
                color="red", linewidth=1.5, label="Circuit")

    # Raw OSM coastline chain
    if COASTLINE_PATH.exists():
        with open(COASTLINE_PATH, encoding="utf-8") as f:
            coast_data = json.load(f)
        chain         = coast_data[0]
        had_reversals = coast_data[1]
        if chain:
            ax.plot([p[0] for p in chain],
                    [p[1] for p in chain],
                    color="darkorange", linewidth=2.0, zorder=5,
                    label=f"Coastline ({len(chain)} pts"
                          f"{', had reversals' if had_reversals else ''})")
            # Mark start and end so direction is visible
            ax.scatter([chain[0][0]],  [chain[0][1]],
                       color="green", s=60, zorder=6, label="Coast start")
            ax.scatter([chain[-1][0]], [chain[-1][1]],
                       color="red",   s=60, zorder=6, label="Coast end")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
