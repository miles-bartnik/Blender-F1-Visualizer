#!/usr/bin/env python3
"""
f1_circuits.py
--------------
Single-file pipeline that:

  1. Fetches every F1 circuit GeoJSON from bacinger/f1-circuits on GitHub
  2. For circuits with FastF1 telemetry coverage (2018-2025):
       - Loads the best available race session
       - Aligns telemetry X/Y (local metres) to GeoJSON lon/lat via
         Procrustes fitting (Kabsch/Umeyama algorithm)
       - Assumes the local receiver exists within ±1km of the circuit centroid
       - Uses aligned telemetry coordinates as the track geometry
       - Uses zero-baselined telemetry Z as relative elevation
  3. For circuits with no telemetry coverage:
       - Keeps the original GeoJSON geometry
       - Queries Open-Topo-Data (SRTM GL1 ~30m) for elevation per point
  4. Writes one annotated GeoJSON per circuit plus a combined file
     Format: [lon, lat, elevation_m] per coordinate — compatible with
     f1_import_curves.py

Requirements:
    pip install requests numpy scipy fastf1

Usage:
    python f1_circuits.py

Output:
    ./output/circuits/<circuit_id>.geojson   — one file per circuit
    ./output/f1-circuits-elevation.geojson   — all circuits combined
"""

import copy
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import requests

import fastf1
import fastf1.exceptions
fastf1.set_log_level("WARNING")


# =============================================================================
# CONFIGURATION
# =============================================================================

# GitHub source
RAW_BASE           = "https://raw.githubusercontent.com/bacinger/f1-circuits/master"
CIRCUITS_INDEX_URL = f"{RAW_BASE}/f1-locations.json"
CIRCUITS_DIR_URL   = f"{RAW_BASE}/circuits"

# Output
OUTPUT_DIR       = Path("./output")
CIRCUITS_OUT_DIR = OUTPUT_DIR / "circuits"
COMBINED_OUT     = OUTPUT_DIR / "f1-circuits-elevation.geojson"

# FastF1 cache
CACHE_DIR = Path("./fastf1_cache")

# SRTM fallback (Open-Topo-Data)
ELEVATION_API  = "https://api.opentopodata.org/v1/srtm30m"
SRTM_BATCH     = 100
SRTM_DELAY     = 1.1

# Terrain grid
TERRAIN_GRID_KM      = 10.0   # total grid size in km (circuit centred)
TERRAIN_RESOLUTION_M = 30.0   # one sample every N metres
TERRAIN_API          = "https://api.opentopodata.org/v1/srtm30m"
TERRAIN_BATCH        = 100    # points per API request
TERRAIN_DELAY        = 1.1    # seconds between requests

# Telemetry alignment
ALIGN_POINTS         = 300    # resample size for Procrustes fit
SMOOTH_WINDOW        = 5      # Z smoothing window
MAX_RECEIVER_OFFSET  = 1000.0 # metres — warn if exceeded
TELEM_DELAY          = 2.0    # seconds between FastF1 API calls

# Overwrite control — set independently for circuits, terrain and buildings.
# True  = re-download and overwrite existing files
# False = skip if the file already exists
OVERWRITE_CIRCUITS  = True
OVERWRITE_TERRAIN   = False
OVERWRITE_BUILDINGS = False

# Restrict processing to a specific list of circuit IDs.
# Set to None to process all circuits from the bacinger index.
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
CIRCUITS_TO_PROCESS = [
    "mc-1929",   # Monaco
]

# Bank angle physics approximations
CAR_MASS_KG        = 800.0   # approximate 2025 F1 car + driver (kg)
DOWNFORCE_COEFF    = 0.0175  # downforce per (m/s)² as fraction of car mass
                              # at 300 km/h (~83 m/s): F_down ≈ 0.0175 × 83² × 800 ≈ 3.5× weight
DRS_DOWNFORCE_LOSS = 0.12    # fractional downforce reduction when DRS open (~12%)

# TUMFTM racetrack-database — track width source
TUMFTM_RAW = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks"

# Mapping from our circuit_id to TUMFTM CSV filename (without .csv)
# Only circuits present in the TUMFTM database are listed.
# Format per file: [x_m, y_m, w_tr_right_m, w_tr_left_m], no header
TUMFTM_MAP = {
    "us-2012":  "Austin",
    "au-1953":  "Melbourne",
    "at-1969":  "RedBullRing",
    "hu-1986":  "Budapest",
    "be-1925":  "Spa",
    "it-1922":  "Monza",
    "sg-2008":  "Singapore",
    "jp-1962":  "Suzuka",
    "de-1932":  "Hockenheim",
    "gb-1948":  "Silverstone",
    "es-1991":  "Catalunya",
    "ca-1978":  "Montreal",
    "mc-1929":  "Monaco",
    "nl-1948":  "Zandvoort",
    "cn-2004":  "Shanghai",
    "bh-2002":  "Bahrain",
    "az-2016":  "Baku",
    "fr-1969":  "MagnyCours",
    "pt-2008":  "Portimao",
    "tr-2005":  "Istanbul",
    "de-1927":  "Nuerburgring",
}


# =============================================================================
# CIRCUIT -> SESSION MAP
# circuit_id -> [(year, round), ...] newest first
# Round numbers verified against official FIA calendars 2018-2025
# =============================================================================

CIRCUIT_CALENDAR = {
    "au-1953": [(2025,1),(2024,3),(2023,3),(2022,3),(2021,3),(2019,1),(2018,1)],
    "cn-2004": [(2025,2),(2024,5),(2019,4),(2018,3)],
    "jp-1962": [(2025,3),(2024,4),(2023,17),(2022,18),(2019,13),(2018,17)],
    "bh-2002": [(2025,4),(2024,1),(2023,1),(2022,1),(2021,1),(2020,15),(2019,2),(2018,2)],
    "sa-2021": [(2025,5),(2024,2),(2023,2),(2022,2),(2021,21)],
    "us-2022": [(2025,6),(2024,6),(2023,5),(2022,5)],
    "it-1953": [(2025,7),(2024,7),(2022,4),(2021,14),(2020,13)],
    "mc-1929": [(2025,8),(2024,8),(2023,8),(2022,8),(2021,5),(2019,6),(2018,6)],
    "es-1991": [(2025,9),(2024,9),(2023,7),(2022,7),(2021,4),(2020,6),(2019,5),(2018,5)],
    "ca-1978": [(2025,10),(2024,10),(2023,9),(2022,9),(2019,7),(2018,7)],
    "at-1969": [(2025,11),(2024,11),(2023,10),(2022,10),(2021,8),(2020,1),(2019,9),(2018,9)],
    "gb-1948": [(2025,12),(2024,12),(2023,11),(2022,11),(2021,10),(2020,5),(2019,10),(2018,10)],
    "be-1925": [(2025,13),(2024,13),(2023,13),(2022,14),(2021,11),(2020,8),(2019,12),(2018,12)],
    "hu-1986": [(2025,14),(2024,14),(2023,12),(2022,13),(2021,12),(2020,4),(2019,11),(2018,11)],
    "nl-1948": [(2025,15),(2024,15),(2023,14),(2022,15),(2021,13)],
    "it-1922": [(2025,16),(2024,16),(2023,15),(2022,16),(2021,14),(2020,9),(2019,14),(2018,14)],
    "az-2016": [(2025,17),(2024,17),(2023,4),(2022,8),(2021,6),(2019,4),(2018,4)],
    "sg-2008": [(2025,18),(2024,18),(2023,16),(2022,17),(2019,15),(2018,15)],
    "us-2012": [(2025,19),(2024,19),(2023,19),(2022,19),(2021,18),(2019,19),(2018,18)],
    "mx-1962": [(2025,20),(2024,20),(2023,20),(2022,20),(2019,18),(2018,19)],
    "br-1940": [(2025,21),(2024,21),(2023,21),(2022,21),(2021,19),(2019,20),(2018,20)],
    "us-2023": [(2025,22),(2024,22),(2023,22)],
    "qa-2004": [(2025,23),(2024,23),(2023,18),(2021,20)],
    "ae-2009": [(2025,24),(2024,24),(2023,23),(2022,22),(2021,22),(2020,17),(2019,21),(2018,21)],
    "de-1927": [(2020,11)],
    "de-1932": [(2019,11),(2018,11)],
    "fr-1969": [(2022,12),(2021,7),(2020,7),(2019,8),(2018,8)],
    "pt-2008": [(2021,3),(2020,12)],
    "tr-2005": [(2021,16),(2020,14)],
    "ru-2014": [(2021,15),(2020,10),(2019,16),(2018,16)],
    "it-1914": [(2020,9)],
}

# No FastF1 coverage — SRTM only
NO_TELEMETRY = {
    "my-1999","br-1977","ar-1952","za-1961",
    "us-1909","us-1956","fr-1960","pt-1972","es-2026",
}


# =============================================================================
# SHARED GEOJSON HELPERS
# =============================================================================

def fetch_json(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_linestring(geojson):
    """Return the first linestring coordinate list from a GeoJSON object."""
    def walk(obj):
        t = obj.get("type","")
        if t == "FeatureCollection":
            for f in obj.get("features",[]):
                r = walk(f)
                if r is not None: return r
        elif t == "Feature":
            return walk(obj.get("geometry") or {})
        elif t == "LineString":
            return obj["coordinates"]
        elif t == "MultiLineString":
            lines = obj["coordinates"]
            return lines[0] if lines else None
        return None
    return walk(geojson)


def extract_all_coordinates(geojson):
    """Return all coordinate pairs across all geometry types."""
    coords = []
    def walk(obj):
        t = obj.get("type","")
        if t == "FeatureCollection":
            for f in obj.get("features",[]): walk(f)
        elif t == "Feature":
            walk(obj.get("geometry") or {})
        elif t == "LineString":
            coords.extend(obj["coordinates"])
        elif t == "MultiLineString":
            for line in obj["coordinates"]: coords.extend(line)
        elif t == "Polygon":
            for ring in obj["coordinates"]: coords.extend(ring)
        elif t == "MultiPolygon":
            for poly in obj["coordinates"]:
                for ring in poly: coords.extend(ring)
        elif t == "Point":
            coords.append(obj["coordinates"])
    walk(geojson)
    return coords


def centroid_lonlat(coords):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lons)/len(lons), sum(lats)/len(lats)


def replace_linestring_coords(geojson, new_coords):
    """Replace the first linestring's coordinates in a GeoJSON object."""
    result = copy.deepcopy(geojson)
    replaced = [False]
    def walk(obj):
        if replaced[0]: return
        t = obj.get("type","")
        if t == "FeatureCollection":
            for f in obj.get("features",[]): walk(f)
        elif t == "Feature":
            walk(obj.get("geometry") or {})
        elif t == "LineString":
            obj["coordinates"] = new_coords
            replaced[0] = True
        elif t == "MultiLineString":
            if obj["coordinates"]:
                obj["coordinates"][0] = new_coords
                replaced[0] = True
    walk(result)
    return result


def set_feature_property(geojson, key, value):
    """Set a property on all Feature objects."""
    def walk(obj):
        t = obj.get("type","")
        if t == "FeatureCollection":
            for f in obj.get("features",[]): walk(f)
        elif t == "Feature":
            if obj.get("properties") is None:
                obj["properties"] = {}
            obj["properties"][key] = value
    walk(geojson)


# =============================================================================
# PROJECTION HELPERS
# =============================================================================

R_EARTH = 6_371_000.0

def lonlat_to_metres(lon, lat, lon0, lat0):
    lat_rad = math.radians(lat0)
    x = math.radians(lon - lon0) * R_EARTH * math.cos(lat_rad)
    y = math.radians(lat - lat0) * R_EARTH
    return x, y


def metres_to_lonlat(x, y, lon0, lat0):
    lat_rad = math.radians(lat0)
    lon = lon0 + math.degrees(x / (R_EARTH * math.cos(lat_rad)))
    lat = lat0 + math.degrees(y / R_EARTH)
    return lon, lat


# =============================================================================
# CURVE RESAMPLING
# =============================================================================

def resample_curve_2d(points, n):
    """Resample a 2D curve to n evenly-spaced points by arc length."""
    if len(points) < 2:
        return list(points)
    dists = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        dists.append(dists[-1] + math.sqrt(dx*dx + dy*dy))
    total = dists[-1]
    if total == 0:
        return [points[0]] * n
    result = []
    j = 0
    for i in range(n):
        t = i * total / (n - 1)
        while j < len(dists) - 2 and dists[j+1] < t:
            j += 1
        if j >= len(dists) - 1:
            result.append(points[-1])
        else:
            seg = dists[j+1] - dists[j]
            frac = (t - dists[j]) / seg if seg > 0 else 0.0
            x = points[j][0] + frac * (points[j+1][0] - points[j][0])
            y = points[j][1] + frac * (points[j+1][1] - points[j][1])
            result.append((x, y))
    return result


# =============================================================================
# PROCRUSTES ALIGNMENT
# =============================================================================

def normalise_pts(pts):
    pts   = np.array(pts, dtype=float)
    mean  = pts.mean(axis=0)
    pts  -= mean
    scale = math.sqrt((pts**2).sum(axis=1).mean())
    if scale > 0:
        pts /= scale
    return pts, mean, scale


def kabsch(source, target):
    """
    Kabsch/Umeyama algorithm: find rotation R, scale s, translation t
    that maps source onto target with minimum RMSE.
    Returns (R, s, t, rmse_in_target_units).
    """
    src_n, src_mean, src_scale = normalise_pts(np.array(source))
    tgt_n, tgt_mean, tgt_scale = normalise_pts(np.array(target))
    H   = src_n.T @ tgt_n
    U, S, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)
    D   = np.diag([1.0, 1.0 if d > 0 else -1.0])
    R   = Vt.T @ D @ U.T
    s   = tgt_scale / src_scale
    t   = tgt_mean - s * (src_mean @ R.T)
    aligned = s * (np.array(source) @ R.T) + t
    rmse = float(np.sqrt(((aligned - np.array(target))**2).sum(axis=1).mean()))
    return R, s, t, rmse


def find_best_alignment(tel_pts_2d, geo_pts_2d):
    """
    Try Procrustes from 4 starting rotations (0/90/180/270°).
    Returns (R, s, t, rmse) for the best fit.
    """
    best = None
    for deg in [0, 90, 180, 270]:
        rad = math.radians(deg)
        c, ss = math.cos(rad), math.sin(rad)
        rotated = [(x*c - y*ss, x*ss + y*c) for x, y in tel_pts_2d]
        R, s, t, rmse = kabsch(rotated, geo_pts_2d)
        if best is None or rmse < best[3]:
            best = (R, s, t, rmse)
    return best


# =============================================================================
# SMOOTHING
# =============================================================================

def moving_average(values, window):
    if window <= 1:
        return list(values)
    if window % 2 == 0:
        window += 1
    half = window // 2
    n    = len(values)
    result = []
    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


# =============================================================================
# TELEMETRY LOADING
# =============================================================================

def load_best_telemetry(circuit_id):
    """
    Try sessions newest-first.
    Loads ALL clean laps from the best available session.
    Returns (laps_data, z_ref) where:
      laps_data : list of (xs, ys, zs, spds, drss) per valid lap
      z_ref     : smoothed Z from the fastest lap for elevation baseline
    Returns None if no usable data found.
    """
    if circuit_id not in CIRCUIT_CALENDAR:
        return None

    for year, round_num in CIRCUIT_CALENDAR[circuit_id]:
        print(f"      {year} R{round_num} ... ", end="", flush=True)
        try:
            sess = fastf1.get_session(year, round_num, "R")
            sess.load(telemetry=True, laps=True,
                      weather=False, messages=False)
        except fastf1.exceptions.DataNotLoadedError:
            print("skip(no API)")
            time.sleep(TELEM_DELAY)
            continue
        except Exception as e:
            print(f"skip({type(e).__name__})")
            time.sleep(TELEM_DELAY)
            continue

        try:
            all_laps = sess.laps.pick_quicklaps()
        except Exception:
            try:
                all_laps = sess.laps
            except Exception:
                print("skip(no laps)")
                time.sleep(TELEM_DELAY)
                continue

        if all_laps is None or all_laps.empty:
            print("skip(no laps)")
            time.sleep(TELEM_DELAY)
            continue

        try:
            all_laps = all_laps.sort_values("LapTime")
        except Exception:
            pass

        laps_data = []
        z_ref     = None
        n_total   = len(all_laps)

        for _, lap_row in all_laps.iterrows():
            try:
                pos = lap_row.get_pos_data(pad=1)
            except Exception:
                continue

            if pos is None or pos.empty \
               or not {"X","Y","Z"}.issubset(pos.columns):
                continue

            speed_ms = None
            drs_open = None
            try:
                car    = lap_row.get_car_data(pad=1)
                merged = pos.merge_channels(car)
                if "Speed" in merged.columns:
                    speed_ms = [
                        v / 3.6 if v == v and v is not None else None
                        for v in merged["Speed"].tolist()
                    ]
                if "DRS" in merged.columns:
                    drs_open = [
                        1.0 if (v == v and v is not None
                                and float(v) >= 10) else 0.0
                        for v in merged["DRS"].tolist()
                    ]
            except Exception:
                pass

            xs, ys, zs, spds, drss = [], [], [], [], []
            for i in range(len(pos)):
                x = pos["X"].iloc[i]
                y = pos["Y"].iloc[i]
                z = pos["Z"].iloc[i]
                spd = speed_ms[i] if speed_ms and i < len(speed_ms) else None
                drs = drs_open[i]  if drs_open  and i < len(drs_open)  else 0.0
                if x == x and y == y and z == z \
                   and x is not None and y is not None and z is not None:
                    xs.append(float(x))
                    ys.append(float(y))
                    zs.append(float(z))
                    spds.append(float(spd) if spd is not None
                                and spd == spd else 0.0)
                    drss.append(float(drs))

            if len(xs) < 50:
                continue

            zs_smooth = moving_average(zs, SMOOTH_WINDOW)
            if z_ref is None:
                z_ref = zs_smooth

            laps_data.append((xs, ys, zs_smooth, spds, drss))

        if not laps_data:
            print("skip(no usable laps)")
            time.sleep(TELEM_DELAY)
            continue

        # Extract sector split XY coordinates from the fastest lap.
        sector_splits_xy = None
        finish_line_xy   = None
        try:
            fastest_lap = all_laps.iloc[0]
            s1_time = fastest_lap.get("Sector1SessionTime")
            s2_time = fastest_lap.get("Sector2SessionTime")

            pos_fast = fastest_lap.get_pos_data(pad=0)
            if pos_fast is not None and not pos_fast.empty \
               and "SessionTime" in pos_fast.columns:
                times = pos_fast["SessionTime"].values
                pxs   = pos_fast["X"].values
                pys   = pos_fast["Y"].values

                def nearest_xy(target_time):
                    diffs = [abs((t - target_time).total_seconds())
                             if hasattr(t - target_time, 'total_seconds')
                             else abs(float(t) - float(target_time))
                             for t in times]
                    idx = diffs.index(min(diffs))
                    return (float(pxs[idx]), float(pys[idx]))

                # Sector splits
                if s1_time is not None and s2_time is not None \
                   and s1_time == s1_time and s2_time == s2_time:
                    s1_xy = nearest_xy(s1_time)
                    s2_xy = nearest_xy(s2_time)
                    sector_splits_xy = (s1_xy, s2_xy)

                # Finish line — find where LapNumber increments in car data
                # Use the lap start time as a proxy for finish line crossing
                try:
                    lap_start = fastest_lap.get("LapStartTime")
                    if lap_start is not None and lap_start == lap_start:
                        finish_line_xy = nearest_xy(lap_start)
                except Exception:
                    pass

        except Exception:
            sector_splits_xy = None
            finish_line_xy   = None

        # Extract human-readable circuit name from session event
        circuit_name = None
        try:
            ev = sess.event
            circuit_name = (str(ev.get("Location", "") or "").strip() or
                            str(ev.get("EventName", "") or "").strip() or
                            None)
        except Exception:
            pass

        print(f"OK ({len(laps_data)}/{n_total} laps, "
              f"Z {min(z_ref):.0f}\u2013{max(z_ref):.0f}"
              f"{', ' + circuit_name if circuit_name else ''})")
        time.sleep(TELEM_DELAY)
        return laps_data, z_ref, sess, circuit_name, sector_splits_xy, finish_line_xy


def load_pit_lane_telemetry(sess):
    """
    Extract pit lane position data from a loaded FastF1 session.

    Scans all laps for PitInTime / PitOutTime entries. For each pit stop:
      - Pit-in lap: slices position data from PitInTime to lap end
      - Pit-out lap: slices position data from lap start to PitOutTime

    Accumulates all pit lane X/Y/Z points across all drivers and pit stops.
    Returns a list of (x, y, z) tuples covering the pit lane, or None.
    """
    pit_points = []

    try:
        all_laps = sess.laps
    except Exception:
        return None

    if all_laps is None or all_laps.empty:
        return None

    for _, lap_row in all_laps.iterrows():
        pit_in  = lap_row.get("PitInTime")
        pit_out = lap_row.get("PitOutTime")

        # Only process laps that have a pit event
        has_pit_in  = pit_in  is not None and pit_in  == pit_in   # NaN check
        has_pit_out = pit_out is not None and pit_out == pit_out

        if not has_pit_in and not has_pit_out:
            continue

        try:
            pos = lap_row.get_pos_data(pad=0)
        except Exception:
            continue

        if pos is None or pos.empty or not {"X","Y","Z"}.issubset(pos.columns):
            continue

        # Use SessionTime index for slicing
        if "SessionTime" not in pos.columns and pos.index.name != "SessionTime":
            try:
                pos = pos.reset_index()
            except Exception:
                continue

        try:
            session_times = pos["SessionTime"] if "SessionTime" in pos.columns                             else pos.index.to_series()

            lap_start = lap_row.get("LapStartTime", None)
            lap_end   = lap_row.get("Time", None)

            if has_pit_in and lap_end is not None:
                # Slice from PitInTime to end of lap
                mask = session_times >= pit_in
                segment = pos[mask]
            elif has_pit_out and lap_start is not None:
                # Slice from start of lap to PitOutTime
                mask = session_times <= pit_out
                segment = pos[mask]
            else:
                segment = pos

            for i in range(len(segment)):
                x = segment["X"].iloc[i]
                y = segment["Y"].iloc[i]
                z = segment["Z"].iloc[i]
                if x == x and y == y and z == z                    and x is not None and y is not None and z is not None:
                    pit_points.append((float(x), float(y), float(z)))

        except Exception:
            continue

    if len(pit_points) < 10:
        return None

    return pit_points


def average_pit_lane_coords(pit_points, n_bins=200, smooth_window=5):
    """
    Given a raw point cloud of pit lane positions from multiple pit stops
    and drivers, produce a smooth centreline by:
      1. Sorting points by their angle from the centroid (approximate ordering)
      2. Binning into n_bins along the pit lane arc
      3. Averaging X/Y/Z per bin
      4. Smoothing

    Returns list of (x, y, z) centreline points, or None if insufficient data.
    """
    if not pit_points or len(pit_points) < 10:
        return None

    xs = [p[0] for p in pit_points]
    ys = [p[1] for p in pit_points]
    zs = [p[2] for p in pit_points]

    # Compute cumulative arc length using nearest-neighbour ordering.
    # Start from the point with minimum Y (entry end of pit lane typically).
    pts  = list(zip(xs, ys, zs))
    start_idx = min(range(len(pts)), key=lambda i: pts[i][1])
    ordered  = [pts[start_idx]]
    remaining = pts[:start_idx] + pts[start_idx+1:]

    while remaining:
        last = ordered[-1]
        nearest_idx = min(range(len(remaining)),
                          key=lambda i: (remaining[i][0]-last[0])**2
                                       +(remaining[i][1]-last[1])**2)
        ordered.append(remaining.pop(nearest_idx))

    oxs = [p[0] for p in ordered]
    oys = [p[1] for p in ordered]
    ozs = [p[2] for p in ordered]

    dists = [0.0]
    for i in range(1, len(oxs)):
        dx = oxs[i]-oxs[i-1]; dy = oys[i]-oys[i-1]
        dists.append(dists[-1] + math.sqrt(dx*dx+dy*dy))
    total = dists[-1]
    if total == 0:
        return None

    norm = [d/total for d in dists]

    # Average into bins
    bin_xs = [[] for _ in range(n_bins)]
    bin_ys = [[] for _ in range(n_bins)]
    bin_zs = [[] for _ in range(n_bins)]

    for i, s in enumerate(norm):
        b = min(int(s * n_bins), n_bins - 1)
        bin_xs[b].append(oxs[i])
        bin_ys[b].append(oys[i])
        bin_zs[b].append(ozs[i])

    centreline = []
    for b in range(n_bins):
        if bin_xs[b]:
            centreline.append((
                sum(bin_xs[b]) / len(bin_xs[b]),
                sum(bin_ys[b]) / len(bin_ys[b]),
                sum(bin_zs[b]) / len(bin_zs[b]),
            ))

    if len(centreline) < 5:
        return None

    # Smooth X/Y/Z separately
    cxs = moving_average([p[0] for p in centreline], smooth_window)
    cys = moving_average([p[1] for p in centreline], smooth_window)
    czs = moving_average([p[2] for p in centreline], smooth_window)

    return list(zip(cxs, cys, czs))


# =============================================================================
# =============================================================================
# BANK ANGLE -- HARDCODED LOOKUP TABLE
# =============================================================================
#
# Banking angles for F1 circuits based on verified published data.
# Only circuits with significant, documented banking are listed.
# All other circuits default to 0.0 degrees everywhere.
#
# Format per circuit:
#   circuit_id -> list of (s_start, s_end, peak_angle_deg)
#   where s_start/s_end are normalised track distance (0.0-1.0)
#   and peak_angle_deg is the maximum banking at the centre of the section.
#
# Within each section a smooth raised-cosine envelope is applied so the
# banking ramps up and back down cleanly rather than switching abruptly.
#
# Sources:
#   nl-1948 (Zandvoort): F1.com, Wikipedia, Dromo/Zaffelli statements
#     Turn 3  (Hugenholtzbocht): 19 deg peak (4.5 deg inside, 19 deg outside)
#     Turn 14 (Luyendijkbocht):  18 deg peak
#   de-1932 (Hockenheim): Ocon quote ~4x less than Zandvoort -> ~5 deg
#     Sachs curve and Mercedes arena section have mild banking
#   tr-2005 (Istanbul): Turn 8 has pronounced off-camber/adverse camber
#     Not positive banking -- actually negative camber ~3-4 deg
#   All other circuits: no documented significant banking -> 0.0

BANK_LOOKUP = {
    # Zandvoort -- two heavily banked corners
    # Turn 3 (Hugenholtzbocht): ~0.17-0.24 of lap distance
    # Turn 14 (Luyendijkbocht): ~0.88-0.97 of lap distance
    "nl-1948": [
        (0.17, 0.24, 19.0),
        (0.88, 0.97, 18.0),
    ],
    # Hockenheim -- mild banking in stadium section
    # Sachs curve and Mercedes hairpin area
    "de-1932": [
        (0.55, 0.65,  5.0),   # Mercedes arena / Sachs entry
        (0.68, 0.76,  5.0),   # Sachs curve
    ],
    # Istanbul -- Turn 8 is notably OFF-camber (negative banking)
    # Drivers must fight the adverse camber through the 4-apex corner
    "tr-2005": [
        (0.52, 0.72, -4.0),   # Turn 8 adverse camber (negative = outward tilt)
    ],
}

# Endpoint blend fraction -- same as elevation endpoint matching
BANK_BLEND_FRACTION = 0.01


def compute_track_distances(xs, ys):
    dists = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
        dists.append(dists[-1] + math.sqrt(dx*dx + dy*dy))
    return dists


def lookup_bank_angles(circuit_id, n_points):
    """
    Generate a bank angle profile for `circuit_id` with `n_points` values.

    Each entry in BANK_LOOKUP defines a banked section as
    (s_start, s_end, peak_deg). A raised-cosine envelope shapes the
    transition so angles ramp smoothly to and from the peak.

    Returns a list of n_points values in radians.
    """
    sections = BANK_LOOKUP.get(circuit_id, [])
    angles   = [0.0] * n_points

    for s_start, s_end, peak_deg in sections:
        peak_rad = math.radians(peak_deg)
        for i in range(n_points):
            s = i / max(n_points - 1, 1)
            if s_start <= s <= s_end:
                # t goes 0->1 across the section
                # Symmetric bell: 0 at edges, 1.0 at centre (t=0.5)
                # Using: 0.5*(1 - cos(2*pi*t)) peaks at t=0.5
                t   = (s - s_start) / (s_end - s_start)
                env = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
                angles[i] += peak_rad * env

    # Endpoint matching over BANK_BLEND_FRACTION
    blend_pts  = max(2, int(n_points * BANK_BLEND_FRACTION))
    start_bank = angles[0]
    end_bank   = angles[-1]
    mid_bank   = (start_bank + end_bank) / 2.0

    for i in range(blend_pts):
        t = i / blend_pts
        angles[i]                 = start_bank * (1.0 - t) + mid_bank * t
        angles[n_points - 1 - i]  = end_bank   * (1.0 - t) + mid_bank * t

    angles[0]           = mid_bank
    angles[n_points - 1] = mid_bank

    return angles


def average_bank_angles(circuit_id, laps_data, n_points):
    """
    Return (bank_angles, coverage_pct) for a circuit.
    Uses the hardcoded lookup table. laps_data is accepted for API
    compatibility but not used.
    """
    angles   = lookup_bank_angles(circuit_id, n_points)
    has_data = circuit_id in BANK_LOOKUP
    coverage = 100.0 if has_data else 0.0
    return angles, coverage


# SRTM ELEVATION FALLBACK
# =============================================================================

def fetch_srtm_elevations(latlon_pairs):
    """Query Open-Topo-Data SRTM for a list of (lat, lon) pairs."""
    elevations = []
    total = len(latlon_pairs)
    for i in range(0, total, SRTM_BATCH):
        batch   = latlon_pairs[i:i+SRTM_BATCH]
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        print(f"    SRTM query {i}–{min(i+SRTM_BATCH,total)-1} of {total} ...",
              end=" ")
        try:
            resp = requests.get(ELEVATION_API,
                                params={"locations": loc_str},
                                timeout=30)
            resp.raise_for_status()
            data = resp.json()
            elevations.extend(r.get("elevation") for r in data.get("results", []))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            elevations.extend([None] * len(batch))
        if i + SRTM_BATCH < total:
            time.sleep(SRTM_DELAY)
    return elevations


def inject_elevations_inplace(geojson, elevations):
    """Inject elevation as third coordinate element into all geometry."""
    result    = copy.deepcopy(geojson)
    elev_iter = iter(elevations)
    def inject(obj):
        t = obj.get("type", "")
        if t == "FeatureCollection":
            for f in obj.get("features", []): inject(f)
        elif t == "Feature":
            inject(obj.get("geometry") or {})
        elif t == "LineString":
            for c in obj["coordinates"]:
                e = next(elev_iter, None)
                if len(c) >= 3: c[2] = e
                else: c.append(e)
        elif t == "MultiLineString":
            for line in obj["coordinates"]:
                for c in line:
                    e = next(elev_iter, None)
                    if len(c) >= 3: c[2] = e
                    else: c.append(e)
        elif t == "Polygon":
            for ring in obj["coordinates"]:
                for c in ring:
                    e = next(elev_iter, None)
                    if len(c) >= 3: c[2] = e
                    else: c.append(e)
        elif t == "MultiPolygon":
            for poly in obj["coordinates"]:
                for ring in poly:
                    for c in ring:
                        e = next(elev_iter, None)
                        if len(c) >= 3: c[2] = e
                        else: c.append(e)
        elif t == "Point":
            e = next(elev_iter, None)
            c = obj["coordinates"]
            if len(c) >= 3: c[2] = e
            else: c.append(e)
    inject(result)
    return result


def split_into_sectors(new_coords, new_lonlat, tel_aligned,
                       sector_splits_xy, R, s, t):
    """
    Split new_coords into three sector lists.

    If sector_splits_xy is available (s1_xy, s2_xy in local metre frame):
      - Transform both split points through the Procrustes transform
      - Find the nearest index in new_lonlat to each transformed split point
      - Split new_coords at those two indices

    Fallback: split into three equal thirds by point count.

    Returns (s1_coords, s2_coords, s3_coords).
    """
    n = len(new_coords)
    if n < 6:
        return None

    split1, split2 = None, None

    if sector_splits_xy is not None:
        try:
            s1_local = np.array([[sector_splits_xy[0][0],
                                   sector_splits_xy[0][1]]], dtype=float)
            s2_local = np.array([[sector_splits_xy[1][0],
                                   sector_splits_xy[1][1]]], dtype=float)

            # Apply same Procrustes transform as the main track
            s1_aligned = s * (s1_local @ R.T) + t
            s2_aligned = s * (s2_local @ R.T) + t

            # Convert to lon/lat for comparison with new_lonlat
            s1_ll = metres_to_lonlat(
                float(s1_aligned[0, 0]), float(s1_aligned[0, 1]),
                *lonlat_centroid_from_lonlat(new_lonlat))
            s2_ll = metres_to_lonlat(
                float(s2_aligned[0, 0]), float(s2_aligned[0, 1]),
                *lonlat_centroid_from_lonlat(new_lonlat))

            # Find nearest index in new_lonlat
            def nearest_idx(target_lon, target_lat):
                best_d = float('inf')
                best_i = 0
                for i, (lon, lat) in enumerate(new_lonlat):
                    d = (lon - target_lon)**2 + (lat - target_lat)**2
                    if d < best_d:
                        best_d = d
                        best_i = i
                return best_i

            split1 = nearest_idx(s1_ll[0], s1_ll[1])
            split2 = nearest_idx(s2_ll[0], s2_ll[1])

            # Ensure correct order
            if split1 > split2:
                split1, split2 = split2, split1
            # Sanity check — splits shouldn't be too close to edges
            if split1 < 3 or split2 > n - 3 or split2 - split1 < 3:
                split1, split2 = None, None
        except Exception:
            split1, split2 = None, None

    # Fallback: equal thirds
    if split1 is None or split2 is None:
        split1 = n // 3
        split2 = (2 * n) // 3

    s1 = new_coords[:split1 + 1]
    s2 = new_coords[split1:split2 + 1]
    s3 = new_coords[split2:]

    return s1, s2, s3


def lonlat_centroid_from_lonlat(lonlat_pairs):
    """Return (lon0, lat0) centroid of a list of (lon, lat) pairs."""
    lons = [p[0] for p in lonlat_pairs]
    lats = [p[1] for p in lonlat_pairs]
    return sum(lons) / len(lons), sum(lats) / len(lats)


# =============================================================================
# OSM PIT LANE FALLBACK
# =============================================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_DELAY = 1.5  # seconds between requests

def fetch_osm_pit_lane(circuit_id, geojson):
    """
    Query the Overpass API for pit lane ways within the circuit bounding box.
    Looks for:
      - highway=service ways within the bbox
      - filtered to those closest to the main track geometry
        (pit lanes are adjacent to the main straight, not random service roads)

    Returns a list of [lon, lat] coordinate pairs forming the pit lane,
    or None if nothing useful is found.
    """
    # Get bounding box from GeoJSON bbox property or compute from coords
    bbox = geojson.get("bbox")
    if not bbox:
        coords = extract_all_coordinates(geojson)
        if not coords:
            return None
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = [min(lons), min(lats), max(lons), max(lats)]

    # Add small buffer around bbox
    buf  = 0.002  # ~200m
    south, west = bbox[1] - buf, bbox[0] - buf
    north, east  = bbox[3] + buf, bbox[2] + buf

    query = f"""
[out:json][timeout:30];
(
  way["highway"="service"]({south},{west},{north},{east});
  way["highway"="raceway"]["service"="pit_lane"]({south},{west},{north},{east});
  way["service"="pit_lane"]({south},{west},{north},{east});
  way["raceway"="pit_lane"]({south},{west},{north},{east});
);
out geom;
"""

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent":   "F1CircuitsPipeline/1.0"},
            timeout=35
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"skip(OSM query failed: {e})")
        return None

    elements = data.get("elements", [])
    if not elements:
        print("skip(no OSM ways found)")
        return None

    # Extract all way geometries
    ways = []
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            pts = [[g["lon"], g["lat"]] for g in el["geometry"]]
            if len(pts) >= 2:
                ways.append(pts)

    if not ways:
        print("skip(no usable OSM geometries)")
        return None

    # Filter to ways closest to the main track centreline.
    # Compute centroid of each way and find distance to nearest track coord.
    track_coords = extract_all_coordinates(geojson)
    if not track_coords:
        return None

    track_lons = [c[0] for c in track_coords]
    track_lats = [c[1] for c in track_coords]

    def way_dist_to_track(way):
        wlon = sum(p[0] for p in way) / len(way)
        wlat = sum(p[1] for p in way) / len(way)
        return min(
            math.sqrt((wlon - tlon)**2 + (wlat - tlat)**2)
            for tlon, tlat in zip(track_lons, track_lats)
        )

    # Keep only ways within 0.005 degrees (~500m) of the track
    near_ways = [w for w in ways if way_dist_to_track(w) < 0.005]

    if not near_ways:
        # Relax threshold and take the single closest way
        near_ways = [min(ways, key=way_dist_to_track)]

    # If multiple ways remain, join them into a single linestring
    # by ordering them by their centroid longitude (pit lanes run
    # roughly parallel to the main straight)
    near_ways.sort(key=lambda w: sum(p[0] for p in w) / len(w))

    merged = []
    for way in near_ways:
        if merged and len(way) > 0:
            # Flip way if its start is farther from merged end than its end
            d_start = math.sqrt(
                (way[0][0] - merged[-1][0])**2 +
                (way[0][1] - merged[-1][1])**2)
            d_end   = math.sqrt(
                (way[-1][0] - merged[-1][0])**2 +
                (way[-1][1] - merged[-1][1])**2)
            if d_end < d_start:
                way = list(reversed(way))
        merged.extend(way)

    return merged if len(merged) >= 2 else None


# =============================================================================
# PER-CIRCUIT PIPELINE
# =============================================================================

# =============================================================================
# TRACK WIDTH — TUMFTM RACETRACK DATABASE
# =============================================================================

def fetch_tumftm_csv(circuit_id):
    """
    Fetch the TUMFTM track CSV for a circuit.
    Returns list of (x_m, y_m, w_right_m, w_left_m) tuples, or None.
    """
    name = TUMFTM_MAP.get(circuit_id)
    if not name:
        return None

    url = f"{TUMFTM_RAW}/{name}.csv"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [width] TUMFTM fetch failed ({e})")
        return None

    rows = []
    for line in resp.text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('x'):
            continue
        try:
            parts = [float(v) for v in line.split(',')]
            if len(parts) >= 4:
                rows.append((parts[0], parts[1], parts[2], parts[3]))
        except ValueError:
            continue

    return rows if len(rows) > 10 else None


def align_tumftm_widths(tumftm_rows, R, s_scale, t, n_points):
    """
    Align TUMFTM width data to the same coordinate frame as the main track.

    The TUMFTM CSV is in local metres (same type of frame as FastF1).
    We apply the Procrustes transform (R, s_scale, t) already computed
    for the main track to bring TUMFTM into the GeoJSON frame,
    then resample total width (w_right + w_left) to n_points by arc-length.

    Returns a flat list of n_points total width values in metres.
    """
    if not tumftm_rows:
        return None

    # Apply Procrustes transform to TUMFTM X/Y
    xy    = np.array([(r[0], r[1]) for r in tumftm_rows], dtype=float)
    xy_al = s_scale * (xy @ R.T) + t

    xs_al = xy_al[:, 0].tolist()
    ys_al = xy_al[:, 1].tolist()
    widths = [r[2] + r[3] for r in tumftm_rows]   # total width = right + left

    # Resample to n_points by arc-length
    dists = [0.0]
    for i in range(1, len(xs_al)):
        dx = xs_al[i] - xs_al[i-1]
        dy = ys_al[i] - ys_al[i-1]
        dists.append(dists[-1] + math.sqrt(dx*dx + dy*dy))
    total = dists[-1]
    if total == 0:
        return None

    result = []
    j = 0
    for k in range(n_points):
        target = k * total / max(n_points - 1, 1)
        while j < len(dists) - 2 and dists[j+1] < target:
            j += 1
        if j >= len(dists) - 1:
            result.append(widths[-1])
        else:
            seg  = dists[j+1] - dists[j]
            frac = (target - dists[j]) / seg if seg > 0 else 0.0
            w    = widths[j] + frac * (widths[min(j+1, len(widths)-1)] - widths[j])
            result.append(round(w, 2))

    return result


# =============================================================================
# TERRAIN GRID
# =============================================================================

def fetch_osm_buildings(lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass API for building footprints within a bounding box.
    Returns a list of building dicts, each with:
      footprint    : list of [lon, lat] pairs (closed polygon)
      height_m     : height in metres (from tag, levels*3, or default 10m)
      min_height_m : base height (for elevated sections)
      roof_shape   : roof shape string ('flat', 'gabled', etc.)
      osm_id       : OSM way ID
    """
    DEFAULT_HEIGHT_M      = 10.0
    METRES_PER_LEVEL      = 3.0

    query = f"""
[out:json][timeout:60];
(
  way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent":   "F1CircuitsPipeline/1.0"},
            timeout=65
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"skip(OSM buildings failed: {e})")
        return []

    elements = data.get("elements", [])
    print(f"\n    OSM returned {len(elements)} elements", end=" ")

    buildings = []
    skipped_no_geom = 0
    skipped_short   = 0

    for el in elements:
        if "geometry" not in el:
            skipped_no_geom += 1
            continue

        tags = el.get("tags", {})

        geom = el["geometry"]
        if not isinstance(geom, list) or len(geom) < 3:
            skipped_short += 1
            continue

        footprint = [[g["lon"], g["lat"]] for g in geom]

        # Height
        height_m = DEFAULT_HEIGHT_M
        if "height" in tags:
            try:
                # Strip units like "m" or "ft"
                h = tags["height"].replace("m", "").replace("ft", "").strip()
                val = float(h)
                if "ft" in tags["height"]:
                    val *= 0.3048
                height_m = val
            except ValueError:
                pass
        elif "building:levels" in tags:
            try:
                height_m = float(tags["building:levels"]) * METRES_PER_LEVEL
            except ValueError:
                pass

        # Min height (elevated sections)
        min_height_m = 0.0
        if "building:min_height" in tags:
            try:
                min_height_m = float(
                    tags["building:min_height"].replace("m", "").strip())
            except ValueError:
                pass
        elif "min_level" in tags:
            try:
                min_height_m = float(tags["min_level"]) * METRES_PER_LEVEL
            except ValueError:
                pass

        # Roof shape
        roof_shape = tags.get("roof:shape", "flat")

        buildings.append({
            "osm_id":       el.get("id"),
            "footprint":    footprint,
            "height_m":     round(height_m, 2),
            "min_height_m": round(min_height_m, 2),
            "roof_shape":   roof_shape,
        })

    print(f"(skipped: {skipped_no_geom} no-geom, {skipped_short} short)")
    return buildings


def fetch_terrain_grid(circuit_id, lon_centre, lat_centre, out_dir):
    """
    Fetch a TERRAIN_GRID_KM × TERRAIN_GRID_KM elevation grid centred on
    the circuit, at TERRAIN_RESOLUTION_M resolution, from Open-Topo-Data.
    Also fetches OSM building footprints for the same bounding box.

    Saves the result as <circuit_id>_terrain.json. Skips if the file
    already exists and OVERWRITE_TERRAIN is False.

    Returns True on success, False on failure.
    """
    out_path = out_dir / f"{circuit_id}_terrain.json"

    # Determine what needs fetching
    file_exists       = out_path.exists()
    need_terrain      = not file_exists or OVERWRITE_TERRAIN
    need_buildings    = not file_exists or OVERWRITE_BUILDINGS

    if not need_terrain and not need_buildings:
        print(f"  [terrain] already exists — skipping")
        return True

    R_EARTH = 6_371_000.0
    half_km = TERRAIN_GRID_KM * 500.0

    lat_rad  = math.radians(lat_centre)
    d_lat    = math.degrees(half_km / R_EARTH)
    d_lon    = math.degrees(half_km / (R_EARTH * math.cos(lat_rad)))

    lat_min  = lat_centre - d_lat
    lat_max  = lat_centre + d_lat
    lon_min  = lon_centre - d_lon
    lon_max  = lon_centre + d_lon

    n_lat = max(2, int(round(TERRAIN_GRID_KM * 1000 / TERRAIN_RESOLUTION_M)) + 1)
    n_lon = max(2, int(round(TERRAIN_GRID_KM * 1000 / TERRAIN_RESOLUTION_M)) + 1)

    # ── Elevation grid ────────────────────────────────────────────────────────
    if need_terrain:
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
              f"{n_batches} batches ...")

        for b in range(n_batches):
            start = b * TERRAIN_BATCH
            end   = min(start + TERRAIN_BATCH, total)
            batch = grid_points[start:end]

            loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
            try:
                resp = requests.get(
                    TERRAIN_API,
                    params={"locations": loc_str},
                    timeout=30)
                resp.raise_for_status()
                data   = resp.json()
                result = [r.get("elevation") for r in data.get("results", [])]
                elevations.extend(result)
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
        # Load existing terrain data to preserve elevation grid
        print(f"  [terrain] elevation exists — loading for buildings update")
        with open(out_path) as f:
            existing = json.load(f)
        elevations = existing["elevations"]
        ele_min    = existing["ele_min_m"]
        ele_max    = existing["ele_max_m"]

    # ── Buildings ─────────────────────────────────────────────────────────────
    if need_buildings:
        print(f"  [buildings] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        buildings = fetch_osm_buildings(lat_min, lon_min, lat_max, lon_max)
        print(f"OK ({len(buildings)} buildings)")
    else:
        # Preserve existing buildings if not overwriting
        buildings = []
        if file_exists:
            try:
                with open(out_path) as f:
                    buildings = json.load(f).get("buildings", [])
                print(f"  [buildings] kept existing ({len(buildings)} buildings)")
            except Exception:
                pass

    terrain_data = {
        "circuit_id":    circuit_id,
        "grid_km":       TERRAIN_GRID_KM,
        "resolution_m":  TERRAIN_RESOLUTION_M,
        "n_lat":         n_lat,
        "n_lon":         n_lon,
        "lon_centre":    lon_centre,
        "lat_centre":    lat_centre,
        "lon_min":       lon_min,
        "lat_min":       lat_min,
        "lon_max":       lon_max,
        "lat_max":       lat_max,
        "ele_min_m":     round(ele_min, 2),
        "ele_max_m":     round(ele_max, 2),
        "elevations":    elevations,
        "buildings":     buildings,
    }

    with open(out_path, "w") as f:
        json.dump(terrain_data, f, separators=(",", ":"))

    print(f"  [terrain] saved → {out_path}  "
          f"({ele_min:.0f}m–{ele_max:.0f}m, {len(buildings)} buildings)")
    return True


def process_circuit(circuit_id, geojson, lon0, lat0):
    """
    Process one circuit. Returns (annotated_geojson, pit_coords, sectors).

    XY coordinates are kept from the original bacinger GeoJSON — they are
    already in correct lon/lat and project correctly with equirectangular.
    Elevation (Z) is handled entirely by terrain projection in the import
    script — we no longer inject Z values here.

    Telemetry is used only for:
      - circuit_name (from FastF1 session event)
      - pit lane lon/lat coordinates (from PitInTime/PitOutTime)
      - sector split positions (from Sector1/2SessionTime)
      - finish line position (from LapStartTime)
    """
    circuit_name     = None
    pit_coords_local = None
    sectors          = None

    # ── TELEMETRY — metadata only ─────────────────────────────────────────────
    if circuit_id not in NO_TELEMETRY and circuit_id in CIRCUIT_CALENDAR:
        print(f"  [telemetry] loading sessions newest-first:")
        telem = load_best_telemetry(circuit_id)

        if telem is not None:
            laps_data, z_ref, sess, circuit_name, sector_splits_xy, finish_line_xy = telem
            tel_x, tel_y, tel_z, tel_spd, tel_drs = laps_data[0]

            # We still need the Procrustes alignment to:
            # 1. Convert pit lane telemetry XY -> lon/lat
            # 2. Find finish line and sector split positions on the GeoJSON
            coords = extract_linestring(geojson)
            if coords is not None:
                geo_m = [lonlat_to_metres(c[0], c[1], lon0, lat0)
                         for c in coords]
                geo_r = resample_curve_2d(geo_m, ALIGN_POINTS)
                tel_r = resample_curve_2d(list(zip(tel_x, tel_y)), ALIGN_POINTS)

                print(f"  [align] fitting {ALIGN_POINTS}-pt Procrustes ...",
                      end=" ", flush=True)
                R, s, t, rmse = find_best_alignment(tel_r, geo_r)
                print(f"RMSE {rmse:.2f}m")

                # ── Track width from TUMFTM ────────────────────────────────
                width_vals = None
                if circuit_id in TUMFTM_MAP:
                    print(f"  [width] fetching TUMFTM data ...", end=" ", flush=True)
                    tumftm_rows = fetch_tumftm_csv(circuit_id)
                    if tumftm_rows:
                        width_vals = align_tumftm_widths(
                            tumftm_rows, R, s, t, len(coords))
                        if width_vals:
                            print(f"OK  {min(width_vals):.1f}–{max(width_vals):.1f}m")
                        else:
                            print("skip(alignment failed)")
                    else:
                        print("skip(fetch failed)")

                # Inject width as fourth coord element into original GeoJSON coords
                # XY stays as original bacinger lon/lat — Z is 0.0 placeholder
                # (terrain projection in import script sets the real Z)
                new_coords = [
                    [round(c[0], 8),
                     round(c[1], 8),
                     0.0,
                     round(width_vals[i], 2) if width_vals and i < len(width_vals)
                     else None]
                    for i, c in enumerate(coords)
                ]

                # ── Finish line rotation ───────────────────────────────────
                print(f"  [finish] locating finish line ...", end=" ", flush=True)
                finish_idx = 0
                if finish_line_xy is not None:
                    try:
                        fl_arr     = np.array([[finish_line_xy[0],
                                                finish_line_xy[1]]], dtype=float)
                        fl_aligned = s * (fl_arr @ R.T) + t
                        fl_lon, fl_lat = metres_to_lonlat(
                            float(fl_aligned[0, 0]),
                            float(fl_aligned[0, 1]), lon0, lat0)

                        best_d = float('inf')
                        lonlat_list = [(c[0], c[1]) for c in coords]
                        for i, (lon, lat) in enumerate(lonlat_list):
                            d = (lon - fl_lon)**2 + (lat - fl_lat)**2
                            if d < best_d:
                                best_d = d
                                finish_idx = i

                        new_coords = new_coords[finish_idx:] + new_coords[:finish_idx]
                        print(f"OK (index {finish_idx})")
                    except Exception as e:
                        print(f"skip ({e})")
                else:
                    print("skip (no finish line data)")

                result = replace_linestring_coords(geojson, new_coords)
                if circuit_name:
                    set_feature_property(result, "Name", circuit_name)
                set_feature_property(result, "elevation_source", "terrain")

                # ── Pit lane ──────────────────────────────────────────────
                try:
                    print(f"  [pit] extracting pit lane ...", end=" ", flush=True)
                    raw_pit = load_pit_lane_telemetry(sess)
                    if raw_pit:
                        centreline = average_pit_lane_coords(raw_pit)
                        if centreline:
                            pit_arr     = np.array([(p[0], p[1]) for p in centreline])
                            pit_aligned = s * (pit_arr @ R.T) + t
                            pit_coords_local = [
                                [round(lon, 8), round(lat, 8), 0.0]
                                for lon, lat in (
                                    metres_to_lonlat(
                                        float(pa[0]), float(pa[1]), lon0, lat0)
                                    for pa in pit_aligned)
                            ]
                            print(f"OK ({len(pit_coords_local)} pts)")
                        else:
                            print("skip(could not average centreline)")
                    else:
                        print("skip(no pit stop data)")
                except Exception as e:
                    print(f"skip({e})")
                    pit_coords_local = None

                # OSM fallback for pit lane
                if pit_coords_local is None:
                    print(f"  [pit] trying OSM fallback ...", end=" ", flush=True)
                    time.sleep(OVERPASS_DELAY)
                    osm_coords = fetch_osm_pit_lane(circuit_id, geojson)
                    if osm_coords:
                        pit_coords_local = [
                            [round(c[0], 8), round(c[1], 8), 0.0]
                            for c in osm_coords
                        ]
                        print(f"OK ({len(pit_coords_local)} pts, OSM source)")
                    else:
                        print("no data")

                # ── Sectors ───────────────────────────────────────────────
                print(f"  [sectors] splitting ...", end=" ", flush=True)
                rotated_lonlat = [(c[0], c[1]) for c in new_coords]
                sectors = split_into_sectors(
                    new_coords, rotated_lonlat,
                    s * (np.array(list(zip(tel_x, tel_y)), dtype=float) @ R.T) + t,
                    sector_splits_xy, R, s, t)
                if sectors:
                    s1, s2, s3 = sectors
                    src = "telemetry" if sector_splits_xy else "equal thirds"
                    print(f"OK ({len(s1)}/{len(s2)}/{len(s3)} pts, {src})")
                else:
                    print("skip")

                return result, pit_coords_local, sectors

        print("  [telemetry] no usable data — falling back to original GeoJSON")

    # ── NO TELEMETRY — use original GeoJSON coordinates as-is ─────────────────
    coords = extract_linestring(geojson)
    if not coords:
        return None

    # Wrap in [lon, lat, 0.0] format — Z set by terrain in import script
    new_coords = [[round(c[0], 8), round(c[1], 8), 0.0] for c in coords]
    result     = replace_linestring_coords(geojson, new_coords)
    set_feature_property(result, "elevation_source", "terrain")

    # Equal thirds sector split
    n  = len(new_coords)
    if n >= 6:
        s1 = new_coords[:n // 3 + 1]
        s2 = new_coords[n // 3:2 * n // 3 + 1]
        s3 = new_coords[2 * n // 3:]
        sectors = (s1, s2, s3)

    return result, None, sectors


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CIRCUITS_OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    print("Fetching circuit index from GitHub ...")
    locations    = fetch_json(CIRCUITS_INDEX_URL)
    circuit_ids  = [loc["id"] for loc in locations if "id" in loc]
    if CIRCUITS_TO_PROCESS is not None:
        circuit_ids = [c for c in circuit_ids if c in CIRCUITS_TO_PROCESS]
    print(f"Found {len(circuit_ids)} circuits.\n")

    all_features = []
    telemetry_ok = []
    srtm_ok      = []
    failed       = []

    for circuit_id in circuit_ids:
        url = f"{CIRCUITS_DIR_URL}/{circuit_id}.geojson"
        print(f"{'='*60}")
        print(f"[{circuit_id}]")

        # Skip if already done and OVERWRITE_CIRCUITS is False
        out_path = CIRCUITS_OUT_DIR / f"{circuit_id}.geojson"
        if out_path.exists() and not OVERWRITE_CIRCUITS:
            print(f"  already exists — skipping (OVERWRITE_CIRCUITS=False)")
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if existing.get("type") == "FeatureCollection":
                    all_features.extend(existing.get("features",[]))
                elif existing.get("type") == "Feature":
                    all_features.append(existing)
            except Exception:
                pass
            telemetry_ok.append(circuit_id)
            # Still run terrain/buildings fetch in case those flags differ
            coords = extract_all_coordinates(existing) if existing else []
            if coords:
                lon0, lat0 = centroid_lonlat(coords)
                fetch_terrain_grid(circuit_id, lon0, lat0, CIRCUITS_OUT_DIR)
            continue

        # Fetch GeoJSON from GitHub
        try:
            geojson = fetch_json(url)
        except Exception as e:
            print(f"  SKIP — could not fetch GeoJSON: {e}")
            failed.append(circuit_id)
            continue

        # Compute centroid for projection origin
        coords = extract_all_coordinates(geojson)
        if not coords:
            print(f"  SKIP — no coordinates in GeoJSON")
            failed.append(circuit_id)
            continue

        lon0, lat0 = centroid_lonlat(coords)

        # Run pipeline
        pipeline_result = process_circuit(circuit_id, geojson, lon0, lat0)

        if pipeline_result is None:
            failed.append(circuit_id)
            continue

        result, pit_coords, sectors = pipeline_result

        if result is None:
            failed.append(circuit_id)
            continue

        # Save main track GeoJSON
        with open(out_path, "w") as f:
            json.dump(result, f, separators=(",", ":"))
        print(f"  Saved → {out_path}")

        # Fetch terrain grid for this circuit
        fetch_terrain_grid(circuit_id, lon0, lat0, CIRCUITS_OUT_DIR)

        # Save sector GeoJSON files
        if sectors:
            s1_coords, s2_coords, s3_coords = sectors
            # Get circuit name for sector naming
            pit_circuit_name = circuit_id
            def find_name_s(obj):
                nonlocal pit_circuit_name
                t = obj.get("type","")
                if t == "FeatureCollection":
                    for feat in obj.get("features",[]): find_name_s(feat)
                elif t == "Feature":
                    n = (obj.get("properties") or {}).get("Name","")
                    if n: pit_circuit_name = n
            find_name_s(result)

            for sector_num, sector_coords in enumerate(
                    [s1_coords, s2_coords, s3_coords], 1):
                if not sector_coords or len(sector_coords) < 2:
                    continue
                sector_geojson = {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "properties": {
                            "Name":         f"{pit_circuit_name} - S{sector_num}",
                            "circuit_id":   circuit_id,
                            "feature_type": f"sector{sector_num}",
                        },
                        "geometry": {
                            "type":        "LineString",
                            "coordinates": sector_coords,
                        }
                    }]
                }
                s_path = CIRCUITS_OUT_DIR / f"{circuit_id}_s{sector_num}.geojson"
                with open(s_path, "w") as f:
                    json.dump(sector_geojson, f, separators=(",", ":"))
            print(f"  Saved sectors → S1({len(s1_coords)}) "
                  f"S2({len(s2_coords)}) S3({len(s3_coords)}) pts")

        # Save pit lane as separate GeoJSON
        if pit_coords and len(pit_coords) >= 3:
            # Get circuit name from result properties for the pit lane name
            pit_circuit_name = circuit_id
            def find_name(obj):
                nonlocal pit_circuit_name
                t = obj.get("type","")
                if t == "FeatureCollection":
                    for feat in obj.get("features",[]): find_name(feat)
                elif t == "Feature":
                    n = (obj.get("properties") or {}).get("Name","")
                    if n: pit_circuit_name = n
            find_name(result)

            pit_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "Name":             f"{pit_circuit_name} - Pit",
                        "circuit_id":       circuit_id,
                        "feature_type":     "pitlane",
                        "elevation_source": "fastf1_aligned",
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": pit_coords,
                    }
                }]
            }
            pit_path = CIRCUITS_OUT_DIR / f"{circuit_id}_pit.geojson"
            with open(pit_path, "w") as f:
                json.dump(pit_geojson, f, separators=(",", ":"))
            print(f"  Saved pit → {pit_path}")

        # Track source
        ele_src = "unknown"
        def find_src(obj):
            nonlocal ele_src
            t = obj.get("type","")
            if t == "FeatureCollection":
                for feat in obj.get("features",[]): find_src(feat)
            elif t == "Feature":
                ele_src = (obj.get("properties") or {}).get(
                    "elevation_source", "unknown")
        find_src(result)

        if "fastf1" in ele_src:
            telemetry_ok.append(circuit_id)
        else:
            srtm_ok.append(circuit_id)

        if result.get("type") == "FeatureCollection":
            all_features.extend(result.get("features",[]))
        elif result.get("type") == "Feature":
            all_features.append(result)

        print()

    # Write combined file
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