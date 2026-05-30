"""
drop_terrain.py
----------------------
Offsets vertices in one or more named vertex groups by a fixed Z value,
with a smooth cosine falloff based on hop distance from the mesh boundary
(edges with exactly 1 face neighbour).
Vertices at the boundary get zero offset. Vertices further in get
progressively more offset, reaching the full Z_OFFSET at FALLOFF_HOPS
hops or more from the boundary.
Usage:
  1. Select the terrain object
  2. Set VERTEX_GROUPS, Z_OFFSET and FALLOFF_HOPS below
  3. Run Script
"""
import bpy
import bmesh
import math
from collections import deque

# =============================================================================
# CONFIGURATION
# =============================================================================

CIRCUIT       = "Zandvoort"
VERTEX_GROUPS = ["B"] + [rf"W{i+1}" for i in range(30)]    # vertex group names to offset
Z_OFFSET      = -0.05   # max offset in Blender units (negative = down)
FALLOFF_HOPS  = 10      # number of hops over which the falloff occurs

# =============================================================================


def cosine_falloff(t):
    """t=0 at boundary (no offset), t=1 at centre (full offset)."""
    return 0.5 * (1.0 - math.cos(math.pi * t))


obj = bpy.data.objects.get(f"{CIRCUIT} - Terrain")

if obj is None:
    print(f"ERROR: Object '{CIRCUIT} - Terrain' not found.")
elif obj.type != 'MESH':
    print(f"ERROR: Object '{obj.name}' is not a mesh.")
else:
    mesh      = obj.data
    available = [vg.name for vg in obj.vertex_groups]
    total     = 0

    # ── Build mesh topology once ──────────────────────────────────────────
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Build adjacency list
    adjacency = {v.index: set() for v in bm.verts}
    for edge in bm.edges:
        a, b = edge.verts[0].index, edge.verts[1].index
        adjacency[a].add(b)
        adjacency[b].add(a)

    bm.free()

    # ── Apply offset per group ────────────────────────────────────────────
    # Boundary is seeded per group — vertices in the group that share an
    # edge with a vertex NOT in the group (the transition edge).
    # This is computed per group since each group has its own boundary.
    for group_name in VERTEX_GROUPS:
        if group_name not in available:
            print(f"WARNING: '{group_name}' not found — skipping. "
                  f"Available: {available}")
            continue

        vg_idx   = obj.vertex_groups[group_name].index
        in_group = {v.index for v in mesh.vertices
                    if any(g.group == vg_idx for g in v.groups)}

        if not in_group:
            print(f"  '{group_name}': no vertices in group — skipping")
            continue

        # Find boundary: group vertices adjacent to a non-group vertex
        boundary_verts = {vi for vi in in_group
                          if any(nb not in in_group for nb in adjacency[vi])}

        # BFS within the group from the transition boundary
        hop_dist = {}
        queue    = deque()
        for vi in boundary_verts:
            hop_dist[vi] = 0
            queue.append(vi)

        while queue:
            vi   = queue.popleft()
            dist = hop_dist[vi]
            for nb in adjacency[vi]:
                if nb in in_group and nb not in hop_dist:
                    hop_dist[nb] = dist + 1
                    queue.append(nb)

        moved = 0
        for v in mesh.vertices:
            if v.index not in in_group:
                continue
            hops   = hop_dist.get(v.index, FALLOFF_HOPS)
            t      = min(hops / FALLOFF_HOPS, 1.0)
            v.co.z += Z_OFFSET * cosine_falloff(t)
            moved  += 1

        print(f"  '{group_name}': offset {moved} vertices "
              f"({len(boundary_verts)} boundary verts)")
        total += moved

    mesh.update()
    print(f"Done. {total} vertices total on '{obj.name}'.")

    # ── Offset bottom vertices of the boundary wall ───────────────────────
    wall_obj = bpy.data.objects.get(f"{CIRCUIT} - Terrain Boundary")
    if wall_obj is None:
        print(f"WARNING: '{CIRCUIT} - Terrain Boundary' not found — wall not moved.")
    else:
        wall_mesh = wall_obj.data
        vg        = wall_obj.vertex_groups.get("Bottom")
        if vg is None:
            print(f"WARNING: 'Bottom' vertex group not found on '{wall_obj.name}' — re-import to tag it.")
        else:
            vg_idx   = vg.index
            bot_verts = [v for v in wall_mesh.vertices
                         if any(g.group == vg_idx for g in v.groups)]
            for v in bot_verts:
                v.co.z += Z_OFFSET
            wall_mesh.update()
            print(f"  '{wall_obj.name}': offset {len(bot_verts)} bottom vertices by {Z_OFFSET}")