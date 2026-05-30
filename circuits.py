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

# Input
INPUT_DIRECTORY  = Path("./data")
PIT_DIRECTORY    = INPUT_DIRECTORY / "pit"
SECTORS_FILE         = INPUT_DIRECTORY / "sectors.json"
START_FINISH_FILE    = INPUT_DIRECTORY / "start_finish_lines.geojson"

# Output
OUTPUT_DIR       = Path("./output")
CIRCUITS_OUT_DIR = OUTPUT_DIR / "circuits"
COMBINED_OUT     = OUTPUT_DIR / "f1-circuits-elevation.geojson"

# FastF1 cache — lives under data/raw/fastf1 (see config.py)
from config import CACHE_DIR as _CACHE_DIR
CACHE_DIR = _CACHE_DIR

# SRTM fallback (Open-Topo-Data)
ELEVATION_API  = "https://api.opentopodata.org/v1/srtm30m"
SRTM_BATCH     = 100
SRTM_DELAY     = 1.1

# Terrain grid
TERRAIN_GRID_KM      = 10.0   # total grid size in km (circuit centred)
COASTLINE_EXTEND_KM  = 50.0  # extra bbox margin for coastline query
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
OVERWRITE_WATER     = True

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
    "mc-1929",  # Monaco
    "be-1925",  # Spa
    "au-1953",  # Albert Park
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

        # Extract sector split XY coordinates and finish line from fastest lap.
        # Strategy: find the telemetry XY position at each timing event, then
        # later snap those XY positions to the nearest point on new_coords.
        # This is purely geographic — no time-to-distance approximation.
        sector_splits_xy   = None
        sector_split_fracs = None   # kept for API compat but no longer used
        finish_line_xy     = None
        try:
            fastest_lap = all_laps.iloc[0]
            s1_time  = fastest_lap.get("Sector1SessionTime")   # session-absolute
            s2_time  = fastest_lap.get("Sector2SessionTime")   # session-absolute
            lap_start = fastest_lap.get("LapStartTime")        # session-absolute

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

                # Sector boundary positions in telemetry XY (metres)
                if (s1_time is not None and s1_time == s1_time and
                    s2_time is not None and s2_time == s2_time):
                    try:
                        s1_xy = nearest_xy(s1_time)
                        s2_xy = nearest_xy(s2_time)
                        sector_splits_xy = (s1_xy, s2_xy)
                        print(f"  [sectors] S1 XY: ({s1_xy[0]:.1f}, {s1_xy[1]:.1f})  "
                              f"S2 XY: ({s2_xy[0]:.1f}, {s2_xy[1]:.1f})")
                    except Exception as e:
                        sector_splits_xy = None
                        print(f"  [sectors] could not extract split XY: {e}")

                # Finish line position in telemetry XY (metres)
                try:
                    if lap_start is not None and lap_start == lap_start:
                        finish_line_xy = nearest_xy(lap_start)
                except Exception:
                    pass

        except Exception:
            sector_splits_xy   = None
            sector_split_fracs = None
            finish_line_xy     = None

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


def split_into_sectors(new_coords, sector_fracs, start_finish_lonlat=None):
    """
    Split new_coords into N sectors defined by arc-length fractions.

    sector_fracs        : list of N+1 fractions from sectors.json, e.g.
                          [0, 0.33, 0.66, 1]. Must start at 0 and end at 1.
                          Supports any number of sectors.
    start_finish_lonlat : (lon, lat) of the start/finish line. The point on
                          new_coords nearest to this coordinate becomes index 0
                          of the rotated track before arc-length fractions are
                          applied. Falls back to new_coords[0] if None or [0,0].

    Each sector shares its boundary point with the adjacent sector so curves
    connect cleanly in Blender. The final point of the last sector is always
    the same coordinate as the first point of sector 1.

    Returns a list of coordinate lists, one per sector, or None on failure.
    """
    n = len(new_coords)
    if n < 6 or not sector_fracs or len(sector_fracs) < 3:
        return None

    # ── Find rotation index from start_finish_line ────────────────────────
    sf_lon, sf_lat = start_finish_lonlat
    start_idx = min(
        range(n),
        key=lambda i: (new_coords[i][0] - sf_lon) ** 2
                    + (new_coords[i][1] - sf_lat) ** 2
    )

    # Rotate new_coords so start_idx is first
    rotated = new_coords[start_idx:] + new_coords[:start_idx]

    # ── Compute cumulative arc-length along rotated coords ─────────────────
    dists = [0.0]
    for i in range(1, n):
        dlon = rotated[i][0] - rotated[i-1][0]
        dlat = rotated[i][1] - rotated[i-1][1]
        dists.append(dists[-1] + math.sqrt(dlon*dlon + dlat*dlat))
    total = dists[-1]

    if total == 0:
        return None

    norm = [d / total for d in dists]

    # ── Map each fraction to the nearest index ─────────────────────────────
    def frac_to_idx(frac):
        return min(range(n), key=lambda i: abs(norm[i] - frac))

    # Build split indices from all fractions except first (0) and last (1)
    split_indices = [frac_to_idx(f) for f in sector_fracs]
    # Force first=0 and last=n-1 so sectors always span the full track
    split_indices[0]  = 0
    split_indices[-1] = n - 1

    # Validate: each split must be strictly after the previous
    for i in range(1, len(split_indices)):
        if split_indices[i] <= split_indices[i-1]:
            split_indices[i] = split_indices[i-1] + 1
        if split_indices[i] >= n:
            print(f"  [sectors] degenerate split at fraction "
                  f"{sector_fracs[i]:.3f} — falling back to equal spacing")
            return None

    # ── Slice into sectors ─────────────────────────────────────────────────
    # Adjacent sectors share their boundary point.
    # The final point of the last sector wraps back to rotated[0] so it
    # always terminates exactly at the start/finish coordinate.
    sectors = []
    n_sectors = len(split_indices) - 1
    for i in range(n_sectors):
        s = split_indices[i]
        e = split_indices[i + 1]
        if i < n_sectors - 1:
            sectors.append(rotated[s:e + 1])
        else:
            # Last sector: end at the start/finish point
            sector = rotated[s:e + 1]
            sector[-1] = rotated[0]
            sectors.append(sector)

    return sectors



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


OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
OVERPASS_DELAY = 1.5  # seconds between requests


def fetch_osm_water(lat_min, lon_min, lat_max, lon_max, elevations,
                    n_lat, n_lon):
    """
    Query Overpass for all significant water body polygons within the bbox.

    Tags queried:
      natural=water          — lakes, ponds, rivers (incl. water=river)
      natural=wetland        — marshes, swamps, bogs
      landuse=reservoir      — reservoirs
      landuse=basin          — detention/retention basins
      waterway=riverbank     — legacy river polygons (deprecated but present)
      waterway=dock          — harbour/dock basins
      waterway=canal         — canals mapped as areas

    Ways are fetched with inline geometry (out geom). Relations are fetched
    separately and their member ways are resolved via a second targeted query
    using the way IDs returned in the relation members list — this guarantees
    inline geometry for all member ways regardless of whether they carry
    water tags themselves.

    Returns a list of water dicts:
      osm_id    : OSM element ID
      osm_type  : "way" or "relation"
      water_tag : the primary OSM tag value
      footprint : list of [lon, lat] pairs (closed ring)
      ele_m     : elevation in metres (from terrain grid)
    """
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    tags = [
        '["natural"="water"]',
        '["natural"="wetland"]',
        '["landuse"="reservoir"]',
        '["landuse"="basin"]',
        '["waterway"="riverbank"]',
        '["waterway"="dock"]',
        '["waterway"="canal"]',
    ]

    way_lines      = "\n  ".join(f'way{t}({bbox});'      for t in tags)
    relation_lines = "\n  ".join(f'relation{t}({bbox});' for t in tags)

    # Query 1: ways with inline geometry + relations with member lists
    query = f"""
[out:json][timeout:90];
(
  {way_lines}
  {relation_lines}
);
out geom;
"""
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent":   "F1CircuitsPipeline/1.0"},
            timeout=95
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"skip(OSM water failed: {e})")
        return []

    elements = data.get("elements", [])

    # Separate ways and relations, build way geometry index
    way_geom    = {}   # way_id -> [(lon, lat), ...]
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

    # Query 2: fetch inline geometry for all relation member ways
    # that aren't already in way_geom from the first query
    if relations:
        all_member_ids = set()
        for rel in relations:
            for m in rel.get("members", []):
                if m.get("type") == "way":
                    wid = m["ref"]
                    if wid not in way_geom:
                        all_member_ids.add(wid)

        if all_member_ids:
            id_list  = ",".join(str(i) for i in all_member_ids)
            q2 = f"""
[out:json][timeout:60];
way(id:{id_list});
out geom;
"""
            try:
                r2 = requests.post(
                    OVERPASS_URL,
                    data={"data": q2},
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "User-Agent":   "F1CircuitsPipeline/1.0"},
                    timeout=65
                )
                r2.raise_for_status()
                d2 = r2.json()
                for el in d2.get("elements", []):
                    if el.get("type") == "way":
                        geom = el.get("geometry", [])
                        if geom:
                            way_geom[el["id"]] = [
                                (g["lon"], g["lat"]) for g in geom]
            except Exception as e:
                print(f"  [water] member way fetch failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    def sample_elevation(footprint):
        if not footprint:
            return 0.0
        c_lon = sum(p[0] for p in footprint) / len(footprint)
        c_lat = sum(p[1] for p in footprint) / len(footprint)
        col_f = (c_lon - lon_min) / lon_range * (n_lon - 1) \
                if lon_range else 0.0
        row_f = (c_lat - lat_min) / lat_range * (n_lat - 1) \
                if lat_range else 0.0
        col_i = max(0, min(n_lon - 1, int(round(col_f))))
        row_i = max(0, min(n_lat - 1, int(round(row_f))))
        idx   = row_i * n_lon + col_i
        ele   = elevations[idx] if idx < len(elevations) \
                and elevations[idx] is not None else 0.0
        return round(float(ele), 2)

    def primary_tag(el):
        tags = el.get("tags", {})
        for key in ("water", "natural", "landuse", "waterway"):
            if key in tags:
                return tags[key]
        return "water"

    TOL = 1e-5

    def endpoints_match(a, b):
        return abs(a[0] - b[0]) < TOL and abs(a[1] - b[1]) < TOL

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
                        merged = True
                        break
                if merged:
                    break

        rings = []
        for chain in chains:
            if len(chain) < 3:
                continue
            if not endpoints_match(chain[0], chain[-1]):
                chain.append(chain[0])
            rings.append(chain)
        return rings

    def footprint_too_small(footprint):
        """
        Return True if the footprint's bounding box is smaller than
        TERRAIN_RESOLUTION_M in either dimension. Converts degrees to
        metres using an equirectangular approximation.
        """
        if not footprint:
            return True
        lons = [p[0] for p in footprint]
        lats = [p[1] for p in footprint]
        mid_lat    = (min(lats) + max(lats)) / 2.0
        cos_lat    = math.cos(math.radians(mid_lat))
        deg_to_m   = R_EARTH * math.pi / 180.0
        width_m    = (max(lons) - min(lons)) * cos_lat * deg_to_m
        height_m_v = (max(lats) - min(lats))            * deg_to_m
        return width_m < TERRAIN_RESOLUTION_M or height_m_v < TERRAIN_RESOLUTION_M

    # ── Process ways ──────────────────────────────────────────────────────
    water = []
    seen  = set()

    for el in direct_ways:
        eid = el["id"]
        if eid in seen:
            continue
        seen.add(eid)

        geom = way_geom.get(eid, [])
        if len(geom) < 3:
            continue

        footprint = [[p[0], p[1]] for p in geom]
        if footprint_too_small(footprint):
            continue
        tag = primary_tag(el)
        if tag == "bay":
            continue
        water.append({
            "osm_id":    eid,
            "osm_type":  "way",
            "water_tag": tag,
            "footprint": footprint,
            "ele_m":     sample_elevation(footprint),
        })

    # ── Process relations ─────────────────────────────────────────────────
    for el in relations:
        eid = el["id"]
        if eid in seen:
            continue
        seen.add(eid)

        members   = el.get("members", [])
        outer_ids = [m["ref"] for m in members
                     if m.get("type") == "way" and m.get("role") == "outer"]

        if not outer_ids:
            continue

        rings = stitch_ways(outer_ids)
        if not rings:
            continue

        outer     = max(rings, key=len)
        footprint = [[p[0], p[1]] for p in outer]

        if len(footprint) < 3:
            continue

        if footprint_too_small(footprint):
            continue

        tag = primary_tag(el)
        if tag == "bay":
            continue

        for mid in outer_ids:
            seen.add(mid)

        water.append({
            "osm_id":    eid,
            "osm_type":  "relation",
            "water_tag": primary_tag(el),
            "footprint": footprint,
            "ele_m":     sample_elevation(footprint),
        })

    return water


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


def fetch_osm_coastline(lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass for natural=coastline ways within an extended bounding box
    (terrain bbox + COASTLINE_EXTEND_KM on all sides). Using a wider query
    area ensures the coastline chain fully crosses the extended box as a single
    segment, avoiding re-entry problems within the terrain bbox.

    Returns the longest stitched chain as a list of [lon, lat] pairs (open,
    not closed), or None if no coastline ways are found.

    The chain is intentionally returned open — closing into sea_a / sea_b
    polygons and clipping to the terrain bbox is handled by build_water_polygons.
    """
    R_EARTH = 6_371_000.0
    ext_m   = COASTLINE_EXTEND_KM * 1000.0

    # Extend terrain bbox by COASTLINE_EXTEND_KM on all sides
    lat_centre = (lat_min + lat_max) / 2.0
    cos_lat    = math.cos(math.radians(lat_centre))
    d_lat      = math.degrees(ext_m / R_EARTH)
    d_lon      = math.degrees(ext_m / (R_EARTH * cos_lat))

    ext_lat_min = lat_min - d_lat
    ext_lat_max = lat_max + d_lat
    ext_lon_min = lon_min - d_lon
    ext_lon_max = lon_max + d_lon

    query = f"""
[out:json][timeout:90];
(
  way["natural"="coastline"]({ext_lat_min},{ext_lon_min},{ext_lat_max},{ext_lon_max});
);
out geom;
"""
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent":   "F1CircuitsPipeline/1.0"},
            timeout=95
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"skip(OSM coastline failed: {e})")
        return None

    elements = data.get("elements", [])
    if not elements:
        return None

    # Build directed segments from OSM ways
    segments = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        nodes = [(g["lon"], g["lat"]) for g in el["geometry"]]
        if len(nodes) >= 2:
            segments.append(nodes)

    if not segments:
        return None

    # Stitch segments into chains by matching endpoints
    TOL = 1e-5

    def endpoints_match(a, b):
        return abs(a[0] - b[0]) < TOL and abs(a[1] - b[1]) < TOL

    chains = [list(seg) for seg in segments]
    merged = True
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
                    merged = True
                    break
            if merged:
                break

    # Return the longest chain, clipped to the terrain bbox so no points
    # extend beyond the bbox boundary.
    chain = max(chains, key=len)
    raw   = [[p[0], p[1]] for p in chain]
    clipped = clip_linestring_to_bbox(raw, lon_min, lat_min, lon_max, lat_max)
    return clipped if len(clipped) >= 2 else None



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


def sutherland_hodgman(polygon, clip_rect):
    """
    Clip a polygon to an axis-aligned rectangle using Sutherland-Hodgman.

    polygon   : list of [lon, lat] pairs (need not be closed)
    clip_rect : (lon_min, lat_min, lon_max, lat_max)

    Returns a list of [lon, lat] pairs forming the clipped polygon,
    or an empty list if the polygon is entirely outside the clip rect.
    """
    lon_min, lat_min, lon_max, lat_max = clip_rect

    def inside(p, edge):
        # edge: 0=left, 1=right, 2=bottom, 3=top
        if edge == 0: return p[0] >= lon_min
        if edge == 1: return p[0] <= lon_max
        if edge == 2: return p[1] >= lat_min
        if edge == 3: return p[1] <= lat_max

    def intersect(p1, p2, edge):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if edge == 0:
            t = (lon_min - p1[0]) / dx if dx else 0
        elif edge == 1:
            t = (lon_max - p1[0]) / dx if dx else 0
        elif edge == 2:
            t = (lat_min - p1[1]) / dy if dy else 0
        else:
            t = (lat_max - p1[1]) / dy if dy else 0
        return [p1[0] + t * dx, p1[1] + t * dy]

    output = list(polygon)
    for edge in range(4):
        if not output:
            return []
        input_list = output
        output     = []
        prev       = input_list[-1]
        for curr in input_list:
            if inside(curr, edge):
                if not inside(prev, edge):
                    output.append(intersect(prev, curr, edge))
                output.append(curr)
            elif inside(prev, edge):
                output.append(intersect(prev, curr, edge))
            prev = curr

    return output


def clip_linestring_to_bbox(chain, lon_min, lat_min, lon_max, lat_max):
    """
    Clip an open linestring to a bounding box using Cohen-Sutherland
    line clipping on each segment. Returns a new chain with all points
    inside or on the bbox boundary, inserting intersection points where
    segments cross the bbox edges.
    """
    def clip_segment(p1, p2):
        """Clip segment p1-p2 to bbox. Returns (clipped_p1, clipped_p2) or None."""
        x1, y1 = p1;  x2, y2 = p2

        def code(x, y):
            c = 0
            if x < lon_min: c |= 1
            if x > lon_max: c |= 2
            if y < lat_min: c |= 4
            if y > lat_max: c |= 8
            return c

        c1, c2 = code(x1, y1), code(x2, y2)

        while True:
            if not (c1 | c2):      # both inside
                return (x1, y1), (x2, y2)
            if c1 & c2:            # both outside same edge
                return None
            c = c1 if c1 else c2
            if   c & 8: x = x1 + (x2-x1)*(lat_max-y1)/(y2-y1); y = lat_max
            elif c & 4: x = x1 + (x2-x1)*(lat_min-y1)/(y2-y1); y = lat_min
            elif c & 2: y = y1 + (y2-y1)*(lon_max-x1)/(x2-x1); x = lon_max
            else:       y = y1 + (y2-y1)*(lon_min-x1)/(x2-x1); x = lon_min
            if c == c1: x1, y1, c1 = x, y, code(x, y)
            else:       x2, y2, c2 = x, y, code(x, y)

    clipped = []
    for i in range(len(chain) - 1):
        seg = clip_segment(chain[i], chain[i+1])
        if seg is None:
            continue
        p1, p2 = seg
        if not clipped or (abs(clipped[-1][0]-p1[0]) > 1e-9 or
                           abs(clipped[-1][1]-p1[1]) > 1e-9):
            clipped.append(list(p1))
        clipped.append(list(p2))

    return clipped


def build_water_polygons(water, buildings, sea_chain,
                         lat_min, lon_min, lat_max, lon_max):
    """
    Convert raw OSM water data and a coastline chain into proper GeoJSON
    Polygon features.

    Water bodies: each footprint → one Polygon feature (feature_type: "water_body")

    Sea polygons: the coastline chain is closed into two complementary polygons
    using an EXTENDED bbox (terrain bbox + COASTLINE_EXTEND_KM). This ensures
    the chain fully crosses the wider box as a clean open segment. The two
    sides (sea_a / sea_b) are then clipped to the terrain bbox using
    Sutherland-Hodgman. Only the clipped polygons are stored.

    Returns a list of GeoJSON Feature dicts.
    """
    features = []

    # ── Water bodies ──────────────────────────────────────────────────────
    for body in water:
        fp = body.get("footprint", [])
        if len(fp) < 3:
            continue
        ring = fp if fp[0] == fp[-1] else fp + [fp[0]]
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "water_body",
                "water_tag":    body.get("water_tag", "water"),
                "ele_m":        body.get("ele_m", 0.0),
                "osm_id":       body.get("osm_id"),
                "osm_type":     body.get("osm_type"),
            },
            "geometry": {
                "type":        "Polygon",
                "coordinates": [ring],
            },
        })

    # ── Sea polygons ──────────────────────────────────────────────────────
    if not sea_chain or len(sea_chain) < 2:
        return features

    # Drop closing point if already closed
    chain = sea_chain[:-1] if sea_chain[0] == sea_chain[-1] else list(sea_chain)

    # ── Build extended bbox ───────────────────────────────────────────────
    R_EARTH    = 6_371_000.0
    ext_m      = COASTLINE_EXTEND_KM * 1000.0
    lat_centre = (lat_min + lat_max) / 2.0
    cos_lat    = math.cos(math.radians(lat_centre))
    d_lat      = math.degrees(ext_m / R_EARTH)
    d_lon      = math.degrees(ext_m / (R_EARTH * cos_lat))

    ext_lon_min = lon_min - d_lon
    ext_lon_max = lon_max + d_lon
    ext_lat_min = lat_min - d_lat
    ext_lat_max = lat_max + d_lat

    # Extended bbox corners CCW: BL → BR → TR → TL
    ext_ccw = [
        [ext_lon_min, ext_lat_min],
        [ext_lon_max, ext_lat_min],
        [ext_lon_max, ext_lat_max],
        [ext_lon_min, ext_lat_max],
    ]

    def nearest_edge_point_ext(pt):
        """Project pt onto nearest EXTENDED bbox edge."""
        lon, lat = pt[0], pt[1]
        candidates = [
            (abs(lon - ext_lon_min),
             [ext_lon_min, max(ext_lat_min, min(ext_lat_max, lat))]),
            (abs(lon - ext_lon_max),
             [ext_lon_max, max(ext_lat_min, min(ext_lat_max, lat))]),
            (abs(lat - ext_lat_min),
             [max(ext_lon_min, min(ext_lon_max, lon)), ext_lat_min]),
            (abs(lat - ext_lat_max),
             [max(ext_lon_min, min(ext_lon_max, lon)), ext_lat_max]),
        ]
        return min(candidates, key=lambda c: c[0])[1]

    def nearest_corner_idx_ext(pt):
        return min(range(4), key=lambda i:
                   (ext_ccw[i][0] - pt[0])**2 + (ext_ccw[i][1] - pt[1])**2)

    start_edge   = nearest_edge_point_ext(chain[0])
    end_edge     = nearest_edge_point_ext(chain[-1])
    start_corner = nearest_corner_idx_ext(start_edge)
    end_corner   = nearest_corner_idx_ext(end_edge)

    chain_lonlat = [[p[0], p[1]] for p in chain]

    def bbox_walk_ccw(from_idx, to_idx):
        pts = []
        idx = from_idx
        while idx != to_idx:
            pts.append(list(ext_ccw[idx]))
            idx = (idx + 1) % 4
        pts.append(list(ext_ccw[to_idx]))
        return pts

    def bbox_walk_cw(from_idx, to_idx):
        pts = []
        idx = from_idx
        while idx != to_idx:
            pts.append(list(ext_ccw[idx]))
            idx = (idx - 1) % 4
        pts.append(list(ext_ccw[to_idx]))
        return pts

    # Side A — chain + CCW walk from end back to start
    side_a_open = (chain_lonlat +
                   [end_edge] +
                   bbox_walk_ccw(end_corner, start_corner) +
                   [start_edge])

    # Side B — chain + CW walk from end back to start (opposite side)
    side_b_open = (chain_lonlat +
                   [end_edge] +
                   bbox_walk_cw(end_corner, start_corner) +
                   [start_edge])

    # ── Clip both sides to terrain bbox ──────────────────────────────────
    clip_rect = (lon_min, lat_min, lon_max, lat_max)

    sea_candidates = []
    for side_pts in (side_a_open, side_b_open):
        clipped = sutherland_hodgman(side_pts, clip_rect)
        if len(clipped) < 3:
            continue
        ring = clipped if clipped[0] == clipped[-1] else clipped + [clipped[0]]
        sea_candidates.append(ring)

    if not sea_candidates:
        return features

    # ── Select the sea side by minimum building count ─────────────────────
    # The land polygon almost always contains far more buildings than the
    # sea polygon. Count how many building centroids fall inside each
    # candidate and keep the one with the fewest.

    def count_buildings_in_ring(ring, buildings):
        count = 0
        for b in buildings:
            fp = b.get("footprint", [])
            if not fp:
                continue
            # Use building centroid
            c_lon = sum(p[0] for p in fp) / len(fp)
            c_lat = sum(p[1] for p in fp) / len(fp)
            if point_in_polygon(c_lon, c_lat, ring):
                count += 1
        return count

    counts = [count_buildings_in_ring(ring, water) for ring in sea_candidates]
    best   = sea_candidates[counts.index(min(counts))]

    print(f"  [sea] building counts per side: "
          f"{counts} → keeping side with {min(counts)}")

    features.append({
        "type": "Feature",
        "properties": {
            "feature_type": "sea",
            "ele_m":        0.0,
        },
        "geometry": {
            "type":        "Polygon",
            "coordinates": [best],
        },
    })

    return features


def determine_sea_normal(sea_chain, buildings, lat_min, lon_min, lat_max, lon_max):
    """
    Determine which side of the coastline is sea by building count.

    Builds both polygon sides using the extended bbox (same as build_water_polygons),
    clips them to the terrain bbox, counts how many building centroids fall inside
    each, and returns "positive" for the side with fewer buildings (sea side) or
    "negative" for the other.

    "positive" means the first boolean cut (original normals) produces the sea.
    "negative" means the second cut (flipped normals) produces the sea.

    Returns "positive" or "negative", or None if determination fails.
    """
    if not sea_chain or len(sea_chain) < 2:
        return None

    chain = sea_chain[:-1] if sea_chain[0] == sea_chain[-1] else list(sea_chain)

    # Extended bbox — same expansion as fetch_osm_coastline
    R_EARTH    = 6_371_000.0
    ext_m      = COASTLINE_EXTEND_KM * 1000.0
    lat_centre = (lat_min + lat_max) / 2.0
    cos_lat    = math.cos(math.radians(lat_centre))
    d_lat      = math.degrees(ext_m / R_EARTH)
    d_lon      = math.degrees(ext_m / (R_EARTH * cos_lat))

    ext_lon_min = lon_min - d_lon;  ext_lon_max = lon_max + d_lon
    ext_lat_min = lat_min - d_lat;  ext_lat_max = lat_max + d_lat

    ext_ccw = [
        [ext_lon_min, ext_lat_min],
        [ext_lon_max, ext_lat_min],
        [ext_lon_max, ext_lat_max],
        [ext_lon_min, ext_lat_max],
    ]

    def nearest_edge_point(pt):
        lon, lat = pt[0], pt[1]
        candidates = [
            (abs(lon - ext_lon_min),
             [ext_lon_min, max(ext_lat_min, min(ext_lat_max, lat))]),
            (abs(lon - ext_lon_max),
             [ext_lon_max, max(ext_lat_min, min(ext_lat_max, lat))]),
            (abs(lat - ext_lat_min),
             [max(ext_lon_min, min(ext_lon_max, lon)), ext_lat_min]),
            (abs(lat - ext_lat_max),
             [max(ext_lon_min, min(ext_lon_max, lon)), ext_lat_max]),
        ]
        return min(candidates, key=lambda c: c[0])[1]

    def nearest_corner_idx(pt):
        return min(range(4), key=lambda i:
                   (ext_ccw[i][0] - pt[0])**2 + (ext_ccw[i][1] - pt[1])**2)

    def bbox_walk_ccw(from_idx, to_idx):
        pts, idx = [], from_idx
        while idx != to_idx:
            pts.append(list(ext_ccw[idx])); idx = (idx + 1) % 4
        pts.append(list(ext_ccw[to_idx]))
        return pts

    def bbox_walk_cw(from_idx, to_idx):
        pts, idx = [], from_idx
        while idx != to_idx:
            pts.append(list(ext_ccw[idx])); idx = (idx - 1) % 4
        pts.append(list(ext_ccw[to_idx]))
        return pts

    chain_lonlat = [[p[0], p[1]] for p in chain]
    start_edge   = nearest_edge_point(chain[0])
    end_edge     = nearest_edge_point(chain[-1])
    start_corner = nearest_corner_idx(start_edge)
    end_corner   = nearest_corner_idx(end_edge)

    side_a_open = (chain_lonlat + [end_edge] +
                   bbox_walk_ccw(end_corner, start_corner) + [start_edge])
    side_b_open = (chain_lonlat + [end_edge] +
                   bbox_walk_cw(end_corner, start_corner)  + [start_edge])

    clip_rect = (lon_min, lat_min, lon_max, lat_max)
    clip_a    = sutherland_hodgman(side_a_open, clip_rect)
    clip_b    = sutherland_hodgman(side_b_open, clip_rect)

    def count_buildings(ring):
        if not ring or not buildings:
            return 0
        return sum(1 for b in buildings
                   if point_in_polygon(
                       sum(p[0] for p in b["footprint"]) / len(b["footprint"]),
                       sum(p[1] for p in b["footprint"]) / len(b["footprint"]),
                       ring)
                   if b.get("footprint"))

    count_a = count_buildings(clip_a)
    count_b = count_buildings(clip_b)

    print(f"  [sea normal] building counts: CCW={count_a}  CW={count_b} "
          f"→ sea is {'CCW (positive)' if count_a <= count_b else 'CW (negative)'}")

    # CCW walk = side_a = first boolean cut (positive normals)
    # CW walk  = side_b = second boolean cut (flipped/negative normals)
    return "positive" if count_a <= count_b else "negative"


def fetch_terrain_grid(circuit_id, lon_centre, lat_centre, out_dir):
    """
    Fetch a TERRAIN_GRID_KM × TERRAIN_GRID_KM elevation grid centred on
    the circuit, at TERRAIN_RESOLUTION_M resolution, from Open-Topo-Data.
    Also fetches OSM building footprints and water bodies for the same bbox.

    Returns the terrain_data dict on success, or None on failure.
    Saves a _terrain.json sidecar as a cache to avoid re-fetching.
    """
    out_path = out_dir / f"{circuit_id}_terrain.json"

    # Determine what needs fetching
    file_exists    = out_path.exists()
    need_terrain   = not file_exists or OVERWRITE_TERRAIN
    need_buildings = not file_exists or OVERWRITE_BUILDINGS
    need_water     = not file_exists or OVERWRITE_WATER

    if not need_terrain and not need_buildings and not need_water:
        print(f"  [terrain] already exists — loading from cache")
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [terrain] cache load failed: {e}")
            return None

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
        buildings = []
        if file_exists:
            try:
                with open(out_path) as f:
                    buildings = json.load(f).get("buildings", [])
                print(f"  [buildings] kept existing ({len(buildings)} buildings)")
            except Exception:
                pass

    # ── Water ─────────────────────────────────────────────────────────────────
    if need_water:
        print(f"  [water] querying OSM ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        water = fetch_osm_water(lat_min, lon_min, lat_max, lon_max,
                                elevations, n_lat, n_lon)
        print(f"OK ({len(water)} bodies)")
    else:
        water = []
        if file_exists:
            try:
                with open(out_path) as f:
                    water = json.load(f).get("water", [])
                print(f"  [water] kept existing ({len(water)} bodies)")
            except Exception:
                pass

    # ── Coastline / Sea ───────────────────────────────────────────────────────
    need_sea = not file_exists or OVERWRITE_WATER

    if need_sea:
        print(f"  [sea] querying OSM coastline ...", end=" ", flush=True)
        time.sleep(OVERPASS_DELAY)
        sea_footprint = fetch_osm_coastline(lat_min, lon_min, lat_max, lon_max)
        if sea_footprint:
            print(f"OK ({len(sea_footprint)} pts)")
        else:
            print("none (landlocked)")
    else:
        sea_footprint = None
        if file_exists:
            try:
                with open(out_path) as f:
                    sea_footprint = json.load(f).get("sea")
                if sea_footprint:
                    print(f"  [sea] kept existing ({len(sea_footprint)} pts)")
            except Exception:
                pass

    # Build flat [[lon, lat, z], ...] point list for the terrain feature.
    # Points inside the sea polygon have their elevation reduced by 1m
    # so the terrain surface sits below sea level in coastal areas.
    # Row-major order matches the existing n_lat × n_lon grid layout.
    sea_ring = None
    if sea_footprint and len(sea_footprint) >= 3:
        sea_ring = sea_footprint

    terrain_points = []
    for row in range(n_lat):
        lat = lat_min + row * (lat_max - lat_min) / max(n_lat - 1, 1)
        for col in range(n_lon):
            lon = lon_min + col * (lon_max - lon_min) / max(n_lon - 1, 1)
            idx = row * n_lon + col
            ele = elevations[idx] if idx < len(elevations) else None
            if (ele is not None and sea_ring is not None and
                    point_in_polygon(lon, lat, sea_ring)):
                ele = round(ele - 1.0, 2)
            terrain_points.append([
                round(lon, 6),
                round(lat, 6),
                round(ele, 2) if ele is not None else None,
            ])

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
        "elevations":     elevations,   # kept for cache compatibility
        "buildings":      buildings,
        "water":          water,
        "sea":            sea_footprint,
        "terrain_points": terrain_points,
    }

    with open(out_path, "w") as f:
        json.dump(terrain_data, f, separators=(",", ":"))

    print(f"  [terrain] saved → {out_path}  "
          f"({ele_min:.0f}m–{ele_max:.0f}m, {len(buildings)} buildings)")
    return terrain_data


def process_circuit(circuit_id, geojson, lon0, lat0, sector_config=None):
    """
    Process one circuit. Returns (annotated_geojson, sectors).

    XY coordinates are kept from the original bacinger GeoJSON — they are
    already in correct lon/lat and project correctly with equirectangular.
    Elevation (Z) is handled entirely by terrain projection in the import
    script — we no longer inject Z values here.

    Telemetry is used only for:
      - circuit_name (from FastF1 session event)
      - sector split positions (from Sector1/2SessionTime)
      - finish line position (from LapStartTime)
    """
    circuit_name     = None
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

                # ── Start/finish line and rotation ────────────────────────
                # start_finish_line from start_finish_lines.geojson is the
                # single source of truth for circuit rotation and sector splits.
                # Raise an error if it is not defined — downstream sector
                # and circuit ordering cannot be guaranteed without it.
                print(f"  [finish] locating start/finish line ...",
                      end=" ", flush=True)

                if not sector_config or "start_finish_line" not in sector_config:
                    raise ValueError(
                        f"start_finish_line not defined in start_finish_lines.geojson "
                        f"for {circuit_id} — cannot continue")

                start_finish_lonlat = sector_config["start_finish_line"]
                sf_lon, sf_lat      = start_finish_lonlat

                # Find the nearest point on new_coords — on tie, first wins
                lonlat_list = [(c[0], c[1]) for c in new_coords]
                finish_idx  = min(
                    range(len(lonlat_list)),
                    key=lambda i: (lonlat_list[i][0] - sf_lon) ** 2
                                + (lonlat_list[i][1] - sf_lat) ** 2
                )

                # Compute residual offset for any telemetry-derived positions
                # (no longer used for pit lane but kept for diagnostics)
                fl_arr     = np.array([[finish_line_xy[0], finish_line_xy[1]]],
                                       dtype=float) if finish_line_xy else None
                if fl_arr is not None:
                    fl_aligned = s * (fl_arr @ R.T) + t
                    fl_lon, fl_lat = metres_to_lonlat(
                        float(fl_aligned[0, 0]),
                        float(fl_aligned[0, 1]), lon0, lat0)
                    lon_offset = lonlat_list[finish_idx][0] - fl_lon
                    lat_offset = lonlat_list[finish_idx][1] - fl_lat
                else:
                    lon_offset = 0.0
                    lat_offset = 0.0

                new_coords = new_coords[finish_idx:] + new_coords[:finish_idx]
                print(f"OK (index {finish_idx}, "
                      f"[{sf_lon}, {sf_lat}])")

                result = replace_linestring_coords(geojson, new_coords)
                if circuit_name:
                    set_feature_property(result, "Name", circuit_name)
                set_feature_property(result, "elevation_source", "terrain")

                # ── Sectors ───────────────────────────────────────────────
                print(f"  [sectors] splitting ...", end=" ", flush=True)
                sector_fracs = sector_config.get("sectors")
                if not sector_fracs or len(sector_fracs) < 3:
                    raise ValueError(
                        f"sectors not defined or too short in sectors.json "
                        f"for {circuit_id}")

                sectors = split_into_sectors(
                    new_coords, sector_fracs, start_finish_lonlat)
                if sectors:
                    counts = "/".join(str(len(s)) for s in sectors)
                    print(f"OK ({counts} pts)")
                else:
                    print("skip")

                return result, sectors

        print("  [telemetry] no usable data — falling back to original GeoJSON")

    # ── NO TELEMETRY — use original GeoJSON coordinates as-is ─────────────────
    coords = extract_linestring(geojson)
    if not coords:
        return None

    if not sector_config or "start_finish_line" not in sector_config:
        raise ValueError(
            f"start_finish_line not defined in start_finish_lines.geojson "
            f"for {circuit_id} — cannot continue")

    start_finish_lonlat = sector_config["start_finish_line"]
    sf_lon, sf_lat      = start_finish_lonlat

    # Wrap in [lon, lat, 0.0] format — Z set by terrain in import script
    new_coords = [[round(c[0], 8), round(c[1], 8), 0.0] for c in coords]

    # Rotate to start at start_finish_line — on tie, first wins
    finish_idx = min(
        range(len(new_coords)),
        key=lambda i: (new_coords[i][0] - sf_lon) ** 2
                    + (new_coords[i][1] - sf_lat) ** 2
    )
    new_coords = new_coords[finish_idx:] + new_coords[:finish_idx]

    result = replace_linestring_coords(geojson, new_coords)
    set_feature_property(result, "elevation_source", "terrain")

    sector_fracs = sector_config.get("sectors")
    if not sector_fracs or len(sector_fracs) < 3:
        raise ValueError(
            f"sectors not defined or too short in sectors.json "
            f"for {circuit_id}")

    sectors = split_into_sectors(new_coords, sector_fracs, start_finish_lonlat)
    return result, sectors


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CIRCUITS_OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    INPUT_DIRECTORY.mkdir(exist_ok=True)
    PIT_DIRECTORY.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Load sectors config
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
        print(f"Warning: {SECTORS_FILE} not found — "
              f"sectors will use equal thirds\n")

    # Load start/finish line positions
    sf_lookup = {}
    if START_FINISH_FILE.exists():
        try:
            with open(START_FINISH_FILE, encoding="utf-8") as f:
                sf_data = json.load(f)
            for feat in sf_data.get("features", []):
                props = feat.get("properties", {})
                cid   = props.get("circuit_id")
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
            # Run terrain fetch only if any terrain-related flag is set
            if OVERWRITE_TERRAIN or OVERWRITE_BUILDINGS or OVERWRITE_WATER:
                coords = extract_linestring(existing) if existing else []
                if not coords:
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

        # Compute centroid for projection origin.
        # Use only the main linestring coordinates — the same set the Blender
        # importer uses when it calls centroid() on the loaded GeoJSON.
        # Using extract_all_coordinates (which includes Points, Polygons etc.)
        # produces a different centroid and causes a systematic offset on the
        # pit lane and any other telemetry-derived geometry.
        coords = extract_linestring(geojson)
        if not coords:
            # Fallback: if no linestring found try all coordinates
            coords = extract_all_coordinates(geojson)
        if not coords:
            print(f"  SKIP — no coordinates in GeoJSON")
            failed.append(circuit_id)
            continue

        lon0, lat0 = centroid_lonlat(coords)

        # Run pipeline
        sector_config = dict(sectors_lookup.get(circuit_id) or {})
        sf = sf_lookup.get(circuit_id)
        if sf:
            sector_config["start_finish_line"] = sf
        pipeline_result = process_circuit(
            circuit_id, geojson, lon0, lat0, sector_config)

        if pipeline_result is None:
            failed.append(circuit_id)
            continue

        result, sectors = pipeline_result

        if result is None:
            failed.append(circuit_id)
            continue

        # ── Fetch terrain ──────────────────────────────────────────────────
        terrain_data = fetch_terrain_grid(
            circuit_id, lon0, lat0, CIRCUITS_OUT_DIR)

        # ── Extract circuit name from result ───────────────────────────────
        circuit_name_display = circuit_id
        def find_name_s(obj):
            nonlocal circuit_name_display
            t = obj.get("type", "")
            if t == "FeatureCollection":
                for feat in obj.get("features", []): find_name_s(feat)
            elif t == "Feature":
                n = (obj.get("properties") or {}).get("Name", "")
                if n: circuit_name_display = n
        find_name_s(result)

        # ── Build consolidated FeatureCollection ───────────────────────────
        features = []

        # 1. Circuit track (from process_circuit result)
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
            for sector_num, sector_coords in enumerate(
                    [s1_coords, s2_coords, s3_coords], 1):
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
            print(f"  Sectors: S1({len(s1_coords)}) "
                  f"S2({len(s2_coords)}) S3({len(s3_coords)}) pts")

        # 3. Pit lane — fold in manually drawn file if it exists
        pit_src = PIT_DIRECTORY / f"{circuit_id}_pit.geojson"
        if pit_src.exists():
            try:
                with open(pit_src, encoding="utf-8") as f:
                    pit_geojson = json.load(f)
                # Extract all features and tag them
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

        # 4. Terrain — custom feature with flat point array
        if terrain_data:
            features.append({
                "type": "Feature",
                "properties": {
                    "Name":           f"{circuit_name_display} - Terrain",
                    "circuit_id":     circuit_id,
                    "feature_type":   "terrain",
                    "grid_km":        terrain_data["grid_km"],
                    "resolution_m":   terrain_data["resolution_m"],
                    "n_lat":          terrain_data["n_lat"],
                    "n_lon":          terrain_data["n_lon"],
                    "lon_centre":     terrain_data["lon_centre"],
                    "lat_centre":     terrain_data["lat_centre"],
                    "lon_min":        terrain_data["lon_min"],
                    "lat_min":        terrain_data["lat_min"],
                    "lon_max":        terrain_data["lon_max"],
                    "lat_max":        terrain_data["lat_max"],
                    "ele_min_m":      terrain_data["ele_min_m"],
                    "ele_max_m":      terrain_data["ele_max_m"],
                    "points":         terrain_data["terrain_points"],
                },
                "geometry": None,
            })

        # 5. Buildings — custom feature with array in properties
        if terrain_data and terrain_data.get("buildings"):
            features.append({
                "type": "Feature",
                "properties": {
                    "Name":         f"{circuit_name_display} - Buildings",
                    "circuit_id":   circuit_id,
                    "feature_type": "buildings",
                    "buildings":    terrain_data["buildings"],
                },
                "geometry": None,
            })

        # 6 & 7. Water bodies and sea coastline
        if terrain_data:
            # Water bodies — proper GeoJSON Polygon features
            for body in terrain_data.get("water", []):
                fp = body.get("footprint", [])
                if len(fp) < 3:
                    continue
                ring = fp if fp[0] == fp[-1] else fp + [fp[0]]
                tag  = body.get("water_tag", "water")
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
                    },
                    "geometry": {
                        "type":        "Polygon",
                        "coordinates": [ring],
                    },
                })

            # Sea — raw coastline chain as LineString.
            # The importer handles the boolean cut to produce sea_a / sea_b.
            sea_chain = terrain_data.get("sea")
            if sea_chain and len(sea_chain) >= 2:
                normal_dir = determine_sea_normal(
                    sea_chain,
                    terrain_data.get("buildings", []),
                    terrain_data["lat_min"],
                    terrain_data["lon_min"],
                    terrain_data["lat_max"],
                    terrain_data["lon_max"],
                )
                features.append({
                    "type": "Feature",
                    "properties": {
                        "Name":             f"{circuit_name_display} - Coastline",
                        "circuit_id":       circuit_id,
                        "feature_type":     "coastline",
                        "normal_direction": normal_dir,
                        "lon_min":          terrain_data["lon_min"],
                        "lat_min":          terrain_data["lat_min"],
                        "lon_max":          terrain_data["lon_max"],
                        "lat_max":          terrain_data["lat_max"],
                    },
                    "geometry": {
                        "type":        "LineString",
                        "coordinates": sea_chain,
                    },
                })
                n_water = sum(1 for f in features
                              if (f.get("properties") or {})
                              .get("feature_type") == "water_body")
                print(f"  Water: {n_water} bodies, coastline "
                      f"({len(sea_chain)} pts)")

        consolidated = {"type": "FeatureCollection", "features": features}

        with open(out_path, "w") as f:
            json.dump(consolidated, f, separators=(",", ":"))
        print(f"  Saved → {out_path}  ({len(features)} features)")

        # Track source for summary stats
        ele_src = "unknown"
        def find_src(obj):
            nonlocal ele_src
            t = obj.get("type", "")
            if t == "FeatureCollection":
                for feat in obj.get("features", []): find_src(feat)
            elif t == "Feature":
                ele_src = (obj.get("properties") or {}).get(
                    "elevation_source", "unknown")
        find_src(result)

        if "fastf1" in ele_src:
            telemetry_ok.append(circuit_id)
        else:
            srtm_ok.append(circuit_id)

        # Collect circuit feature only for combined file
        # (terrain/buildings/water excluded — too large for combined)
        for feat in features:
            ft = (feat.get("properties") or {}).get("feature_type", "")
            if ft not in ("terrain", "buildings", "water"):
                all_features.append(feat)

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