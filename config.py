"""
config.py
---------
Shared configuration constants for the F1 circuit pipeline.
Imported by all module files.
"""

from pathlib import Path

# ── GitHub source ─────────────────────────────────────────────────────────────
RAW_BASE           = "https://raw.githubusercontent.com/bacinger/f1-circuits/master"
CIRCUITS_INDEX_URL = f"{RAW_BASE}/f1-locations.json"
CIRCUITS_DIR_URL   = f"{RAW_BASE}/circuits"

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIRECTORY  = Path("./data")
PIT_DIRECTORY    = INPUT_DIRECTORY / "pit"
SECTORS_FILE         = INPUT_DIRECTORY / "sectors.json"
START_FINISH_FILE    = INPUT_DIRECTORY / "start_finish_lines.geojson"
OUTPUT_DIR       = Path("./output")
CIRCUITS_OUT_DIR = OUTPUT_DIR / "circuits"
COMBINED_OUT     = OUTPUT_DIR / "f1-circuits-elevation.geojson"

# ── Raw data cache ────────────────────────────────────────────────────────────
# All raw API responses are stored under RAW_DATA_DIR so they can be reused
# without hitting the network again.
#
# Directory layout:
#   data/raw/
#     cop30/          COP30 elevation grids, one JSON per circuit
#     osm/            Overpass API responses (water, coastline, vegetation,
#                     streets, structures), one JSON per circuit per type
#     fastf1/         FastF1 session cache (Parquet files etc.)
#
# USE_RAW_CACHE = True  → use cached files when present; download + save if not
# USE_RAW_CACHE = False → always re-download from source and overwrite cache
RAW_DATA_DIR  = INPUT_DIRECTORY / "raw"
USE_RAW_CACHE = False

CACHE_DIR     = RAW_DATA_DIR / "fastf1"

# ── SRTM fallback ─────────────────────────────────────────────────────────────
ELEVATION_API = "https://api.opentopodata.org/v1/srtm30m"
SRTM_BATCH    = 100
SRTM_DELAY    = 1.1

# ── Terrain grid ──────────────────────────────────────────────────────────────
# Primary source: Copernicus DEM 30m (COP30) Cloud Optimised GeoTIFFs hosted
# publicly on AWS S3. No API key required; only the pixels covering each
# circuit bbox are fetched via HTTP range requests (~50–150 KB per circuit).
# OpenTopoData is kept as a fallback in case AWS is unreachable.
TERRAIN_EXTEND_KM    = 1.0   # extend circuit bbox by this many km in each direction
COASTLINE_EXTEND_KM  = 10.0
TERRAIN_RESOLUTION_M = 30.0
TERRAIN_API          = "https://api.opentopodata.org/v1/srtm30m"  # fallback only
TERRAIN_BATCH        = 50     # batch size for OpenTopoData fallback
TERRAIN_DELAY        = 1.1

# ── Telemetry alignment ───────────────────────────────────────────────────────
ALIGN_POINTS        = 300
SMOOTH_WINDOW       = 5
MAX_RECEIVER_OFFSET = 1000.0
TELEM_DELAY         = 2.0

# ── Overwrite flags ───────────────────────────────────────────────────────────
OVERWRITE_TERRAIN     = True
OVERWRITE_STRUCTURES  = True
OVERWRITE_WATER       = True
OVERWRITE_VEGETATION  = True
OVERWRITE_STREETS     = True

# ── OSM ───────────────────────────────────────────────────────────────────────
OVERPASS_URL   = "https://overpass-api.de/api/interpreter"
OVERPASS_DELAY = 10.0  # seconds between Overpass queries

# ── TUMFTM track width database ───────────────────────────────────────────────
TUMFTM_RAW = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks"
TUMFTM_MAP  = {
    "us-2012": "Austin",      "au-1953": "Melbourne",
    "at-1969": "RedBullRing", "hu-1986": "Budapest",
    "be-1925": "Spa",         "it-1922": "Monza",
    "sg-2008": "Singapore",   "jp-1962": "Suzuka",
    "de-1932": "Hockenheim",  "gb-1948": "Silverstone",
    "es-1991": "Catalunya",   "ca-1978": "Montreal",
    "mc-1929": "Monaco",      "nl-1948": "Zandvoort",
    "cn-2004": "Shanghai",    "bh-2002": "Bahrain",
    "az-2016": "Baku",        "fr-1969": "MagnyCours",
    "pt-2008": "Portimao",    "tr-2005": "Istanbul",
    "de-1927": "Nuerburgring",
}

# ── Circuit calendar: circuit_id -> [(year, round), ...] newest-first ─────────
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

# ── No FastF1 coverage — terrain only ─────────────────────────────────────────
NO_TELEMETRY = {
    "my-1999", "br-1977", "ar-1952", "za-1961",
    "us-1909", "us-1956", "fr-1960", "pt-1972", "es-2026",
}

# ── Water / H3 ───────────────────────────────────────────────────────────────
H3_RESOLUTION  = 12   # H3 hexagonal grid resolution for water mesh generation
BOUNDARY_DEPTH = 2    # thickness of terrain boundary ring in H3 cells
WATER_COVERAGE_THRESHOLD = 0.95  # drop a water body if this fraction of its area is covered by a larger one

# ── Water / sea depth model (Håkanson power law: depth_m = k * sqrt(area_m²)) ─
WATER_DEPTH_K     = 0.3    # empirical coefficient
WATER_DEPTH_MAX_M = 50.0   # maximum depression depth in metres
# ── Circuits to process ([] = all available on bacinger repo) ────────────────
CIRCUITS_TO_PROCESS = [
    "mc-1929",
    "au-1953"
]   # empty = all circuits

# Circuits to skip even when processing all
CIRCUITS_EXCLUDE = [
    "es-2008",
]