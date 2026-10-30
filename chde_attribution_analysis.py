"""
CHDE Attribution Analysis  v4
=============================
Revision of v3.  Changes are grouped by what they affect.

REGIONS (changes numbers)
  - The six regions are now DISJOINT.  Each is a union of half-open lat/lon
    rectangles (REGIONS below, also written to region_definitions.csv).  v3 used
    overlapping continental boxes (Asia contained north-east Africa and eastern
    Europe, Africa contained Arabia and the Levant).
  - Africa and Asia are separated by a stepped Red Sea / Suez line, Europe and Africa
    by a stepped Mediterranean line, Europe and Asia at 40 E (south of 45 N), 50 E
    (45 to 50 N) and 60 E (north of 50 N, the Urals).  See the notes next to REGIONS.
  - check_regions_disjoint() stops the run if any two regions share a cell.
    region_map.png/pdf and region_table.csv show what each region contains and how
    much valid land belongs to no region (Antarctica, Greenland, small islands).

SEASONS (changes numbers)
  - DJF now uses December of the PRECEDING year with January and February of the
    same year (29 winters per 30-year period).  The permutation null for DJF pairs
    whole winters, so the three months of a winter always move together.

CHECKS (no change to numbers, stops silent errors)
  - The hot, dry and compound files must share one grid (checked).
  - The compound file must equal hot AND dry (checked, error above 0.1 % of region
    area, warning above 0.0001 %).
  - Every region/severity is screened for saturation of the HOT and DRY indicators
    as well as the CHDE indicator (threshold_screen.csv, diagnostics.csv, and a
    "Saturated" flag next to every conditional probability in the Excel tables).

ENSEMBLE RULE
  - An ensemble event needs ENSEMBLE_MIN_MODELS models (default 3 of 4, as in the
    Figure 5 caption).  Set CHDE_ENS_MIN=2 to reproduce the v3 behaviour.

NULL MODEL
  - Main null: year-shuffled (as in v3).  New second null: years are shuffled only
    inside 5-year blocks, which keeps the slow shared trend of hot and dry and so
    isolates co-occurrence from the common trend.  Both are reported (CSV, figures
    with suffix _block5, extra columns in the Excel tables).

LABELS
  - "Observed" is gone: this script never reads the reference data.  Bars and
    columns are "simulated" (the bias-adjusted model data) and compared with the
    permutation null.

OUTPUTS (OUTPUT_DIR)
  region_definitions.csv, region_table.csv, region_map.png/pdf
  delta_bars_{freq,area}_severityXX.*          change between periods
  null_comparison_{freq,area}_severityXX[_block5].*   simulated vs permutation null
  monthly_prob_{hot,dry,compound}_severityXX.*  calendar months, ensemble null band
  probability_tables.xlsx                       seasonal metrics, DJF = Dec(y-1), Jan, Feb
  delta_data_sevXX.csv, null_summary_sevXX[_block5].csv, region_corr_sevXX.csv
  sensitivity_thresholds.csv, threshold_screen.csv, analytic_reference.csv,
  diagnostics.csv

Environment variables (all optional)
  CHDE_BASE_PATH, CHDE_OUT, CHDE_SEV ("1.0,1.5,2.0"), CHDE_N_PERM (1000),
  CHDE_ENS_MIN (3), CHDE_CACHE (1)
"""

import os
import hashlib
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_PATH = Path(os.environ.get(
    "CHDE_BASE_PATH",
    "/work/ch0636/u301861/phd/HighRESMIP/results/compoundXTREMe_3"))
OUTPUT_DIR = Path(os.environ.get("CHDE_OUT", "attribution_outputs_v4"))
CACHE_DIR = OUTPUT_DIR / "cache"
USE_CACHE = os.environ.get("CHDE_CACHE", "1") == "1"
CACHE_VERSION = "v4.0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["CNRM-CERFACS", "EC-Earth-HR", "HadGEM3-HM", "MPI-XR"]
DISPLAY = {
    "CNRM-CERFACS": "CNRM",
    "EC-Earth-HR":  "EC-Earth3P",
    "HadGEM3-HM":   "HadGEM3",
    "MPI-XR":       "MPI-XR",
    "ENSEMBLE":     "Ensemble",
}
KEYS = MODELS + ["ENSEMBLE"]

SEVERITIES = [s.strip() for s in os.environ.get("CHDE_SEV", "1.0,1.5,2.0").split(",")]
SEV_KEYS = {
    "1.0": {"temp": "TEMP1.0std", "spi": "SPI1",   "compound": "TEMP1.0std_SPI1"},
    "1.5": {"temp": "TEMP1.5std", "spi": "SPI1.5", "compound": "TEMP1.5std_SPI1.5"},
    "2.0": {"temp": "TEMP2.0std", "spi": "SPI2",   "compound": "TEMP2.0std_SPI2"},
}
VAR_NAMES = {"temp": "tasmax", "compound": "CompoundXTREMe"}

# Periods (each exactly N_YEARS long, January to December)
PERIODS = {"hist": (1985, 2014), "fut": (2021, 2050)}
PERIOD_LABEL = {"hist": "Historical", "fut": "Future"}
N_YEARS = 30
SPAN_START = min(p[0] for p in PERIODS.values())
SPAN_END = max(p[1] for p in PERIODS.values())
N_MONTHS_SPAN = (SPAN_END - SPAN_START + 1) * 12
for _p, (_s, _e) in PERIODS.items():
    assert _e - _s + 1 == N_YEARS, f"Period {_p} is not {N_YEARS} years long"

# Detection criteria: name -> function(valid_region_area_km2) -> threshold in km2.
_ABS_MKM2 = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
CRITERIA = {f"abs_{a}Mkm2": (lambda v, a=a: a * 1e6) for a in _ABS_MKM2}
CRITERIA["legacy_5pct_area"] = lambda v: 0.05 * v

# Main criterion: one absolute area for ALL regions, severities, models and periods.
# 1.0 Mkm2 leaves the ensemble CHDE indicator unsaturated at 1.5 sigma.  The HOT and DRY
# indicators DO saturate in the large regions at this value (the expected hot area of a
# stationary climate already exceeds 1 Mkm2 there), so conditional probabilities are
# flagged where P(hot) or P(dry) >= 0.98.  See threshold_screen.csv and diagnostics.csv.
MAIN_CRITERION = "abs_1.0Mkm2"
MAIN_CRITERION_BY_SEV = {sev: MAIN_CRITERION for sev in ("1.0", "1.5", "2.0")}
SAT_MONTHS = 11.0      # ensemble months/yr at or above this = saturated
EMPTY_MONTHS = 0.5     # future ensemble months/yr below this = empty
SAT_PROB = 0.98        # P(hot) or P(dry) at or above this = conditional probabilities trivial

# Ensemble event: at least this many models flag it (3 of 4 = strict majority).
ENSEMBLE_MIN_MODELS = int(os.environ.get("CHDE_ENS_MIN", 3))

# Permutation nulls
N_PERM = int(os.environ.get("CHDE_N_PERM", 1000))
RNG_SEED = 12345
NULL_PCT = (2.5, 97.5)
BLOCK = 5               # block length (years) of the trend-preserving null

# File/consistency tolerances (fraction of valid region area)
CONS_WARN = 1e-6
CONS_FAIL = 1e-3

R_EARTH_KM = 6371.0
GRID_RES_DEFAULT = 0.5

# Seasons.  DJF takes December from the preceding year (see season_units).
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASONS = list(SEASON_MONTHS)
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# -----------------------------------------------------------------------------
# REGIONS: disjoint unions of half-open rectangles (lat_min, lat_max, lon_min, lon_max)
# A cell belongs to a region if its CENTRE lies in [lat_min, lat_max) x [lon_min, lon_max).
# Longitudes are -180..180 (the files may use 0..360).  All limits lie on 0.5 degree
# cell edges.  Checked against Natural Earth continents (0.5 degree): more than 99 % of
# each continent's land area falls in the matching region, apart from the conventions
# listed here.
#
#   North America  Canada, USA, Mexico, Central America (7 N and north, west of 77.5 W),
#                  Caribbean.  Greenland is NOT included, except a strip of north-west
#                  Greenland (lat >= 75, lon -73 to -60) that rectangles cannot separate
#                  from Ellesmere Island (about 0.1 Mkm2).  Hawaii is included.
#   South America  south of 7 N, and north to 13 N east of 77.5 W.
#   Europe         Iberia to 40 E (south of 45 N), 50 E (45 to 50 N) and 60 E (north of
#                  50 N, the Urals); Iceland; Svalbard.  Southern edge steps along the
#                  Mediterranean (36, 37.5, 34.5 N) so that North Africa is excluded.
#                  Cyprus and Anatolia (Turkey east of 26 E, south of 42 N) are Asia.
#   Africa         continent west of a stepped Red Sea / Suez line (Sinai is Asia),
#                  Madagascar included, Mauritius, Reunion, Seychelles not.
#   Asia           Middle East, Caucasus, Central, South, South-East and East Asia and
#                  Siberia east of 60 E, Indonesia west of 130 E (Papua is Oceania),
#                  Japan and northern Pacific islands north of 24 N.
#   Oceania        Australia, New Zealand, New Guinea (lon >= 130 E, south of 1 N) and
#                  Pacific islands west of 180 E and south of 10.5 S.
#   Not assigned   Antarctica, most of Greenland, Micronesia and Polynesia, the Aleutians,
#                  small Atlantic and Indian Ocean islands, Franz Josef Land.
# -----------------------------------------------------------------------------
REGIONS = {
    "North America": [
        (15.0, 60.0, -170.0, -50.0),
        (60.0, 84.0, -170.0, -60.0),
        (7.0, 15.0, -93.0, -77.5),
    ],
    "South America": [
        (-56.0, 7.0, -82.0, -34.0),
        (7.0, 13.0, -77.5, -34.0),
    ],
    "Europe": [
        (36.0, 72.0, -12.0, -2.0),
        (37.5, 72.0, -2.0, 12.0),
        (34.5, 72.0, 12.0, 26.0),
        (42.0, 72.0, 26.0, 40.0),
        (45.0, 50.0, 40.0, 50.0),
        (50.0, 72.0, 40.0, 60.0),
        (63.0, 67.0, -25.0, -13.0),     # Iceland
        (72.0, 82.0, 9.0, 35.0),        # Svalbard
    ],
    "Africa": [
        (-35.0, 12.5, -18.0, 52.0),
        (12.5, 14.5, -18.0, 43.5),
        (14.5, 18.0, -18.0, 41.0),
        (18.0, 22.0, -18.0, 38.5),
        (22.0, 25.0, -18.0, 37.5),
        (25.0, 27.5, -18.0, 35.5),
        (27.5, 29.5, -18.0, 33.5),
        (29.5, 32.0, -18.0, 32.5),
        (32.0, 36.0, -18.0, -2.0),
        (32.0, 37.5, -2.0, 12.0),
        (32.0, 34.5, 12.0, 26.0),
    ],
    "Asia": [
        (12.5, 14.5, 43.5, 60.0),
        (14.5, 18.0, 41.0, 60.0),
        (18.0, 22.0, 38.5, 60.0),
        (22.0, 25.0, 37.5, 60.0),
        (25.0, 27.5, 35.5, 60.0),
        (27.5, 29.5, 33.5, 60.0),
        (29.5, 32.0, 32.5, 60.0),
        (32.0, 42.0, 26.0, 60.0),
        (42.0, 45.0, 40.0, 60.0),
        (45.0, 50.0, 50.0, 60.0),
        (-10.5, 82.0, 60.0, 130.0),
        (24.0, 82.0, 130.0, 180.0),
    ],
    "Oceania": [
        (-48.0, -10.5, 112.0, 180.0),
        (-10.5, 1.0, 130.0, 180.0),
    ],
}
REGION_ORDER = list(REGIONS.keys())
REGION_COLORS = {"North America": "#4C78A8", "South America": "#54A24B", "Europe": "#B279A2",
                 "Africa": "#F58518", "Asia": "#E45756", "Oceania": "#72B7B2"}

MODEL_COLORS = {
    "CNRM-CERFACS": "#1f77b4",
    "EC-Earth-HR":  "#ff7f0e",
    "HadGEM3-HM":   "#2ca02c",
    "MPI-XR":       "#d62728",
    "ENSEMBLE":     "#8B008B",
}
BAR_COLORS = {"temp": "#e05c2a", "spi": "#3a7fc1", "compound": "#8B008B"}
EXT_TYPES = ["temp", "spi", "compound"]
EXT_TITLES = {"temp": "Hot Extremes", "spi": "Dry Extremes", "compound": "CHDE"}

LAT_NAMES = ("lat", "latitude", "rlat", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "rlon", "nav_lon", "x")


def thr(crit: str, valid_km2: float) -> float:
    return float(CRITERIA[crit](valid_km2))


def crit_label(crit: str) -> str:
    if crit.startswith("abs_"):
        return f"≥ {float(crit[4:-4]):g}×10⁶ km²"
    return "≥ 5 % of region area"


# -----------------------------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------------------------
def model_file(model: str, severity: str, ext_type: str) -> Path:
    sk = SEV_KEYS[severity][ext_type]
    if ext_type == "compound":
        return BASE_PATH / model / "6months" / f"{model}_compound_{sk}_corrected_merged.nc"
    if ext_type == "temp":
        return (BASE_PATH / model / "6months" / "phase1"
                / f"{sk}_hist-baseline_{model}_corrected_merged.nc")
    return (BASE_PATH / model / "6months" / "phase1"
            / f"Ind_{sk}_{model}_corrected_merged.nc")


def var_name_for(severity: str, ext_type: str) -> str:
    if ext_type == "temp":
        return VAR_NAMES["temp"]
    if ext_type == "compound":
        return VAR_NAMES["compound"]
    return SEV_KEYS[severity]["spi"]


def load_da(filepath: Path, var: str):
    if not filepath.exists():
        print(f"  [MISS] {filepath.name}")
        return None
    try:
        ds = xr.open_dataset(filepath)
        if var not in ds:
            dvars = [v for v in ds.data_vars if "bnds" not in v.lower()]
            if not dvars:
                return None
            var = dvars[0]
        return ds[var]
    except Exception as e:
        print(f"  [ERR ] {filepath.name}: {e}")
        return None


# -----------------------------------------------------------------------------
# GEOMETRY AND REGION MASKS
# -----------------------------------------------------------------------------
def _find_latlon(da: xr.DataArray):
    for lc in LAT_NAMES:
        for oc in LON_NAMES:
            if lc in da.coords and oc in da.coords:
                return lc, oc
    raise ValueError(f"No lat/lon coords found. Available: {list(da.coords)}")


def band_area_km2(lat_centers, dlat, dlon):
    """Exact area of lat/lon cells on a sphere, one value per latitude."""
    lat1 = np.deg2rad(np.clip(np.asarray(lat_centers) - dlat / 2.0, -90, 90))
    lat2 = np.deg2rad(np.clip(np.asarray(lat_centers) + dlat / 2.0, -90, 90))
    return (R_EARTH_KM ** 2) * np.deg2rad(dlon) * np.abs(np.sin(lat2) - np.sin(lat1))


def _spacing(v, default=GRID_RES_DEFAULT):
    v = np.asarray(v, float)
    if v.size < 2:
        return default
    d = np.abs(np.diff(v))
    d = d[d > 0]
    return float(np.median(d)) if d.size else default


def region_mask_xy(region, lat, lon):
    """Mask for broadcastable lat and lon arrays (cell centres, any longitude convention)."""
    lat = np.asarray(lat, float)
    lon = (np.asarray(lon, float) + 180.0) % 360.0 - 180.0
    m = np.zeros(np.broadcast(lat, lon).shape, bool)
    for la0, la1, lo0, lo1 in REGIONS[region]:
        m |= (lat >= la0) & (lat < la1) & (lon >= lo0) & (lon < lo1)
    return m


def region_mask(region, lat1d, lon1d):
    """Boolean (nlat, nlon) mask of a region on a regular grid."""
    return region_mask_xy(region, np.asarray(lat1d, float)[:, None],
                          np.asarray(lon1d, float)[None, :])


def check_regions_disjoint(step=0.25):
    """Geometry-only check on a fine global grid: no cell may belong to two regions."""
    lat = np.arange(-90 + step / 2, 90, step)
    lon = np.arange(-180 + step / 2, 180, step)
    cnt = sum(region_mask(r, lat, lon).astype(int) for r in REGION_ORDER)
    if cnt.max() > 1:
        bad = np.argwhere(cnt > 1)[0]
        raise ValueError(f"Regions overlap, e.g. at lat {lat[bad[0]]}, lon {lon[bad[1]]}")
    print(f"  [OK] regions are disjoint on a {step} degree grid "
          f"({sum(len(v) for v in REGIONS.values())} rectangles)")


def _to_tll(sub: xr.DataArray, latdim, londim):
    for d in [d for d in sub.dims if d not in ("time", latdim, londim)]:
        sub = sub.isel({d: 0})
    return sub.transpose("time", latdim, londim)


def extract_region(da: xr.DataArray, region: str):
    """
    Return (values, area):
      values : float32 (N_MONTHS_SPAN, n_cells), NaN where the file has no data
      area   : float64 (n_cells,) cell area in km2
    Months are placed by (year, month).  Returns (None, None) if the region has no cells.
    """
    lc, oc = _find_latlon(da)
    tidx = da.indexes["time"]
    yrs = np.asarray(tidx.year)
    mos = np.asarray(tidx.month)
    pos = (yrs - SPAN_START) * 12 + (mos - 1)
    keep = np.where((pos >= 0) & (pos < N_MONTHS_SPAN))[0]
    if keep.size == 0:
        return None, None

    if da[lc].ndim == 1 and da[oc].ndim == 1:
        latdim, londim = da[lc].dims[0], da[oc].dims[0]
        lat = da[lc].values.astype(float)
        lon = da[oc].values.astype(float)
        m2d = region_mask(region, lat, lon)
        if not m2d.any():
            return None, None
        ilat = np.where(m2d.any(axis=1))[0]
        ilon = np.where(m2d.any(axis=0))[0]
        sub = _to_tll(da.isel({"time": keep, latdim: ilat, londim: ilon}), latdim, londim)
        msub = m2d[np.ix_(ilat, ilon)]
        vals = sub.values.astype(np.float32).reshape(sub.shape[0], -1)[:, msub.ravel()]
        a_lat = band_area_km2(lat, _spacing(lat), _spacing(lon))[ilat]
        area = np.broadcast_to(a_lat[:, None], msub.shape)[msub]
    else:                                                 # curvilinear fallback
        ydim, xdim = da[lc].dims
        lat2 = da[lc].values.astype(float)
        lon2 = da[oc].values.astype(float)
        m2d = region_mask_xy(region, lat2, lon2)
        if not m2d.any():
            return None, None
        sub = _to_tll(da.isel(time=keep), ydim, xdim)
        vals = sub.values.astype(np.float32).reshape(sub.shape[0], -1)[:, m2d.ravel()]
        area = band_area_km2(lat2[m2d], GRID_RES_DEFAULT, GRID_RES_DEFAULT)

    out = np.full((N_MONTHS_SPAN, vals.shape[1]), np.nan, dtype=np.float32)
    out[pos[keep]] = vals
    return out, area.astype(np.float64)


def _check_same_grid(das, model):
    """The three files are combined cell by cell, so their grids must be identical."""
    ref = das["compound"]
    lc, oc = _find_latlon(ref)
    for et in ("temp", "spi"):
        l2, o2 = _find_latlon(das[et])
        same = (das[et][l2].shape == ref[lc].shape and das[et][o2].shape == ref[oc].shape
                and np.allclose(das[et][l2].values, ref[lc].values)
                and np.allclose(das[et][o2].values, ref[oc].values))
        if not same:
            raise ValueError(f"{model}: the {et} file is on a different grid than the compound file")


def write_region_outputs():
    """region_definitions.csv, region_table.csv, region_map.png/pdf."""
    rows = []
    for r in REGION_ORDER:
        for k, (la0, la1, lo0, lo1) in enumerate(REGIONS[r], 1):
            rows.append({"region": r, "rectangle": k, "lat_min": la0, "lat_max": la1,
                         "lon_min": lo0, "lon_max": lo1})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "region_definitions.csv", index=False)

    da = None
    for model in MODELS:
        da = load_da(model_file(model, SEVERITIES[0], "compound"), VAR_NAMES["compound"])
        if da is not None:
            break
    if da is None:
        print("  [WARN] no data file found for the region map")
        return
    lc, oc = _find_latlon(da)
    if da[lc].ndim != 1:
        print("  [WARN] region map skipped (curvilinear grid)")
        return
    latdim, londim = da[lc].dims[0], da[oc].dims[0]
    lat = da[lc].values.astype(float)
    lon = da[oc].values.astype(float)
    sub = _to_tll(da.isel(time=slice(0, 24)), latdim, londim)
    valid = np.isfinite(sub.values).any(axis=0)
    area2d = np.broadcast_to(band_area_km2(lat, _spacing(lat), _spacing(lon))[:, None], valid.shape)

    code = np.zeros(valid.shape, int)
    trows = []
    for i, r in enumerate(REGION_ORDER, 1):
        m = region_mask(r, lat, lon) & valid
        code[m] = i
        trows.append({"region": r, "n_rectangles": len(REGIONS[r]), "n_valid_cells": int(m.sum()),
                      "valid_area_Mkm2": area2d[m].sum() / 1e6})
    un = valid & (code == 0)
    code[un] = len(REGION_ORDER) + 1
    unassigned = area2d[un].sum() / 1e6
    trows.append({"region": "(not assigned)", "n_rectangles": 0, "n_valid_cells": int(un.sum()),
                  "valid_area_Mkm2": unassigned})
    pd.DataFrame(trows).to_csv(OUTPUT_DIR / "region_table.csv", index=False)
    print(f"  Valid land in no region: {unassigned:.2f} Mkm2 "
          f"(expected: Antarctica, Greenland, small islands)")

    lon_n = (lon + 180.0) % 360.0 - 180.0
    ox, oy = np.argsort(lon_n), np.argsort(lat)
    g = code[np.ix_(oy, ox)]
    cmap = ListedColormap(["white"] + [REGION_COLORS[r] for r in REGION_ORDER] + ["#bbbbbb"])
    norm = BoundaryNorm(np.arange(-0.5, len(REGION_ORDER) + 2.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.pcolormesh(lon_n[ox], lat[oy], g, cmap=cmap, norm=norm, shading="nearest")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.25, linestyle=":")
    handles = [Patch(facecolor=REGION_COLORS[r], label=f"{r} ({t['valid_area_Mkm2']:.1f} Mkm²)")
               for r, t in zip(REGION_ORDER, trows)]
    handles.append(Patch(facecolor="#bbbbbb", label=f"Not assigned ({unassigned:.1f} Mkm²)"))
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"region_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] region_map.png/.pdf, region_table.csv, region_definitions.csv")


# -----------------------------------------------------------------------------
# PER MODEL / SEVERITY PROCESSING  (the only step that reads NetCDF)
# -----------------------------------------------------------------------------
def _cache_path(severity: str, model: str) -> Path:
    key = hashlib.md5(repr((REGIONS, PERIODS, SPAN_START, SPAN_END,
                            CACHE_VERSION)).encode()).hexdigest()[:8]
    return CACHE_DIR / f"res_sev{severity}_{model}_{key}.pkl"


def build_model_results(model: str, severity: str):
    """
    For every region returns a dict with
      valid_km2, n_cells, mean_cell_km2
      obs[ext][period]   : (N_YEARS, 12) flagged area in km2 per year and month
      M[period]          : (12, N_YEARS, N_YEARS) km2 flagged hot in year i AND
                           dry in year j, for each calendar month.  diag(M) is the
                           real same-year compound area, off-diagonal entries
                           give the permutation null.
      consistency[period]: max |compound file - hot&dry| / valid area
    """
    cp = _cache_path(severity, model)
    if USE_CACHE and cp.exists():
        with open(cp, "rb") as f:
            print(f"  [CACHE] {DISPLAY[model]} severity {severity}")
            return pickle.load(f)

    das = {}
    for et in EXT_TYPES:
        da = load_da(model_file(model, severity, et), var_name_for(severity, et))
        if da is None:
            return None
        das[et] = da
    _check_same_grid(das, model)

    out = {}
    for region in REGION_ORDER:
        arrs, area, ok = {}, None, True
        for et in EXT_TYPES:
            v, a = extract_region(das[et], region)
            if v is None or (area is not None and a.size != area.size):
                print(f"    [WARN] {DISPLAY[model]} {region}: no usable cells for {et}")
                ok = False
                break
            arrs[et], area = v, a
        if not ok:
            continue

        # valid cells: finite somewhere in time in ALL three files
        valid = np.all([np.isfinite(arrs[et]).any(axis=0) for et in EXT_TYPES], axis=0)
        if not valid.any():
            continue
        a_v = area[valid]
        a32 = a_v.astype(np.float32)
        B = {et: (arrs[et][:, valid] == 1) for et in EXT_TYPES}
        finite = {et: np.isfinite(arrs[et][:, valid]).any(axis=1) for et in EXT_TYPES}
        del arrs

        obs = {et: {} for et in EXT_TYPES}
        for et in EXT_TYPES:
            aff = B[et].astype(np.float32) @ a32            # km2 per month
            for p, (ps, pe) in PERIODS.items():
                s0 = (ps - SPAN_START) * 12
                obs[et][p] = aff[s0:s0 + N_YEARS * 12].reshape(N_YEARS, 12).astype(np.float64)
                n_miss = int((~finite[et][s0:s0 + N_YEARS * 12]).sum())
                if n_miss:
                    print(f"    [WARN] {DISPLAY[model]} {region} {et} {p}: "
                          f"{n_miss} months without data (counted as no event)")

        Mdict, cons = {}, {}
        for p, (ps, pe) in PERIODS.items():
            s0 = (ps - SPAN_START) * 12
            H = B["temp"][s0:s0 + N_YEARS * 12].reshape(N_YEARS, 12, -1)
            D = B["spi"][s0:s0 + N_YEARS * 12].reshape(N_YEARS, 12, -1)
            M = np.empty((12, N_YEARS, N_YEARS), dtype=np.float64)
            for m in range(12):
                Hm = H[:, m, :].astype(np.float32) * a32
                Dm = D[:, m, :].astype(np.float32)
                M[m] = Hm @ Dm.T
            Mdict[p] = M
            hd = np.stack([np.diag(M[m]) for m in range(12)], axis=1)   # (years, 12)
            cons[p] = float(np.max(np.abs(hd - obs["compound"][p])) / a_v.sum())
            if cons[p] > CONS_FAIL:
                raise ValueError(
                    f"{DISPLAY[model]} {region} {p}: compound file differs from hot AND dry "
                    f"by {cons[p]:.2e} of the region area. Check the input files.")
            if cons[p] > CONS_WARN:
                print(f"    [WARN] {DISPLAY[model]} {region} {p}: compound vs hot&dry "
                      f"mismatch {cons[p]:.2e} of region area")

        out[region] = {
            "valid_km2": float(a_v.sum()),
            "n_cells": int(valid.sum()),
            "mean_cell_km2": float(a_v.mean()),
            "obs": obs, "M": Mdict, "consistency": cons,
        }
        print(f"    {DISPLAY[model]:<11} {region:<14} valid area "
              f"{a_v.sum() / 1e6:6.2f} Mkm2, {int(valid.sum()):6d} cells")

    if USE_CACHE:
        with open(cp, "wb") as f:
            pickle.dump(out, f)
    return out


def load_all(severity: str) -> dict:
    res = {}
    for model in MODELS:
        print(f"  Loading {DISPLAY[model]}  (severity {severity} sigma)")
        r = build_model_results(model, severity)
        if r:
            res[model] = r
    return res


# -----------------------------------------------------------------------------
# ACCESSORS  (single models and the ensemble share one code path)
# -----------------------------------------------------------------------------
def _members(res, key, region):
    if key == "ENSEMBLE":
        return [m for m in MODELS if m in res and region in res[m]]
    return [key] if (key in res and region in res[key]) else []


def _need(key, n_members):
    return 1 if key != "ENSEMBLE" else min(ENSEMBLE_MIN_MODELS, n_members)


def get_valid(res, key, region):
    ms = _members(res, key, region)
    return float(np.mean([res[m][region]["valid_km2"] for m in ms])) if ms else None


def get_area(res, key, region, ext, period):
    """Flagged area (N_YEARS, 12) in km2; ensemble = mean over models."""
    ms = _members(res, key, region)
    if not ms:
        return None
    return np.mean([res[m][region]["obs"][ext][period] for m in ms], axis=0)


def get_events(res, key, region, ext, period, crit):
    """Boolean regional events (N_YEARS, 12); ensemble = vote of models."""
    ms = _members(res, key, region)
    if not ms:
        return None
    votes = sum((res[m][region]["obs"][ext][period]
                 >= thr(crit, res[m][region]["valid_km2"])).astype(int) for m in ms)
    return votes >= _need(key, len(ms))


def season_units(a, season):
    """
    (years, 12) -> (units, 3) arrays for one season.  DJF takes December from the
    PRECEDING year, so it has N_YEARS - 1 winters (winter w = Dec of year w, Jan and Feb of
    year w + 1).  The other seasons keep all N_YEARS.
    """
    if season == "DJF":
        return np.stack([a[:-1, 11], a[1:, 0], a[1:, 1]], axis=1)
    idx = [m - 1 for m in SEASON_MONTHS[season]]
    return a[:, idx]


# ---- permutation nulls -------------------------------------------------------
def make_derangements(n_perm, n=N_YEARS, seed=RNG_SEED):
    """Random permutations without fixed points (no year keeps its own partner)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, n), dtype=int)
    ident = np.arange(n)
    k = 0
    while k < n_perm:
        p = rng.permutation(n)
        if not np.any(p == ident):
            out[k] = p
            k += 1
    return out


def make_block_derangements(n_perm, n=N_YEARS, block=BLOCK, seed=RNG_SEED):
    """Derange years inside consecutive blocks: partners are at most block - 1 years apart."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, n), dtype=int)
    for k in range(n_perm):
        for s in range(0, n, block):
            idx = np.arange(s, min(s + block, n))
            while True:
                p = rng.permutation(idx)
                if not np.any(p == idx):
                    break
            out[k, idx] = p
    return out


def make_null_sets():
    """Two nulls, each with permutations for calendar years ('all') and for winters ('djf')."""
    return {
        "year": {"name": "year",
                 "all": make_derangements(N_PERM, N_YEARS, RNG_SEED),
                 "djf": make_derangements(N_PERM, N_YEARS - 1, RNG_SEED + 1)},
        f"block{BLOCK}": {"name": f"block{BLOCK}",
                          "all": make_block_derangements(N_PERM, N_YEARS, BLOCK, RNG_SEED + 2),
                          "djf": make_block_derangements(N_PERM, N_YEARS - 1, BLOCK, RNG_SEED + 3)},
    }


def _null_area_model(M, perms):
    """M: (12, years, years) -> null flagged area (n_perm, years, 12)."""
    A = np.transpose(M, (1, 0, 2))                      # (hot year i, month, dry year j)
    ar = np.arange(N_YEARS)[None, :]
    return A[ar, :, perms]


def _null_area_model_djf(M, perms_w):
    """Null for winters: hot winter w paired with dry winter perms_w[k, w] -> (n_perm, years-1, 3)."""
    w = np.arange(N_YEARS - 1)[None, :]
    dec = M[11][w, perms_w]                              # Dec of year w
    jan = M[0][w + 1, perms_w + 1]                       # Jan of year w + 1
    feb = M[1][w + 1, perms_w + 1]
    return np.stack([dec, jan, feb], axis=-1)


_NULL_CACHE = {}


def _null_model(res, m, region, period, ns, kind):
    key = (m, region, period, ns["name"], kind)
    if key not in _NULL_CACHE:
        M = res[m][region]["M"][period]
        _NULL_CACHE[key] = (_null_area_model(M, ns["all"]) if kind == "all"
                            else _null_area_model_djf(M, ns["djf"]))
    return _NULL_CACHE[key]


def get_null_area(res, key, region, period, ns, kind="all"):
    ms = _members(res, key, region)
    if not ms:
        return None
    return np.mean([_null_model(res, m, region, period, ns, kind) for m in ms], axis=0)


def get_null_events(res, key, region, period, crit, ns, kind="all"):
    ms = _members(res, key, region)
    if not ms:
        return None
    votes = sum((_null_model(res, m, region, period, ns, kind)
                 >= thr(crit, res[m][region]["valid_km2"])).astype(int) for m in ms)
    return votes >= _need(key, len(ms))


def null_season(res, key, region, period, crit, ns, season):
    """(events, area) of the null for one season, shapes (n_perm, units, 3)."""
    if season == "DJF":
        return (get_null_events(res, key, region, period, crit, ns, "djf"),
                get_null_area(res, key, region, period, ns, "djf"))
    idx = [m - 1 for m in SEASON_MONTHS[season]]
    return (get_null_events(res, key, region, period, crit, ns, "all")[:, :, idx],
            get_null_area(res, key, region, period, ns, "all")[:, :, idx])


# -----------------------------------------------------------------------------
# SUMMARY TABLES
# -----------------------------------------------------------------------------
def compute_summary(res, crit):
    """Simulated frequency and continuous area per model/region/extreme/period."""
    rows = []
    for key in KEYS:
        for region in REGION_ORDER:
            valid = get_valid(res, key, region)
            if valid is None:
                continue
            for et in EXT_TYPES:
                vals = {}
                for p in PERIODS:
                    ev = get_events(res, key, region, et, p, crit)
                    ar = get_area(res, key, region, et, p)
                    vals[p] = (ev.sum() / N_YEARS, ar.mean() / 1e6, 100 * ar.mean() / valid)
                rows.append({
                    "model": key, "region": region, "ext_type": et,
                    "valid_area_Mkm2": valid / 1e6,
                    "freq_hist": vals["hist"][0], "freq_fut": vals["fut"][0],
                    "delta_freq": vals["fut"][0] - vals["hist"][0],
                    "area_hist_Mkm2": vals["hist"][1], "area_fut_Mkm2": vals["fut"][1],
                    "delta_area": vals["fut"][1] - vals["hist"][1],
                    "area_pct_hist": vals["hist"][2], "area_pct_fut": vals["fut"][2],
                })
    return pd.DataFrame(rows)


def compute_null_summary(res, crit, ns):
    """Simulated CHDE frequency/area against the permutation null."""
    rows = []
    for key in KEYS:
        for region in REGION_ORDER:
            if get_valid(res, key, region) is None:
                continue
            for p in PERIODS:
                ev = get_events(res, key, region, "compound", p, crit)
                ar = get_area(res, key, region, "compound", p)
                nev = get_null_events(res, key, region, p, crit, ns)
                nar = get_null_area(res, key, region, p, ns)
                nfreq = nev.sum(axis=(1, 2)) / N_YEARS
                narea = nar.mean(axis=(1, 2)) / 1e6
                sf, sa = ev.sum() / N_YEARS, ar.mean() / 1e6
                lo_f, hi_f = np.percentile(nfreq, NULL_PCT)
                lo_a, hi_a = np.percentile(narea, NULL_PCT)
                rows.append({
                    "model": key, "region": region, "period": p, "null": ns["name"],
                    "sim_freq": sf, "null_freq_mean": nfreq.mean(),
                    "null_freq_lo": lo_f, "null_freq_hi": hi_f,
                    "p_freq": (1 + (nfreq >= sf - 1e-12).sum()) / (1 + len(nfreq)),
                    "sim_area": sa, "null_area_mean": narea.mean(),
                    "null_area_lo": lo_a, "null_area_hi": hi_a,
                    "p_area": (1 + (narea >= sa - 1e-12).sum()) / (1 + len(narea)),
                })
    return pd.DataFrame(rows)


def sensitivity_table(res, severity, ns):
    rows = []
    for crit in CRITERIA:
        summ = compute_summary(res, crit)
        nul = compute_null_summary(res, crit, ns).set_index(["model", "region", "period"])
        for _, r in summ.iterrows():
            nf_h = nf_f = np.nan
            if r["ext_type"] == "compound":
                nf_h = nul.loc[(r["model"], r["region"], "hist"), "null_freq_mean"]
                nf_f = nul.loc[(r["model"], r["region"], "fut"), "null_freq_mean"]
            rows.append({
                "severity": severity, "criterion": crit,
                "model": DISPLAY[r["model"]], "region": r["region"],
                "ext_type": r["ext_type"],
                "freq_hist": r["freq_hist"], "freq_fut": r["freq_fut"],
                "delta_freq": r["delta_freq"],
                "null_freq_hist": nf_h, "null_freq_fut": nf_f,
            })
    return rows


def _deseason(a):
    """(years, 12) -> anomalies relative to each calendar-month mean, flattened."""
    return (a - a.mean(axis=0, keepdims=True)).ravel()


def _pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_region_corr(res):
    """
    Within-region monthly correlation of the flagged areas (continuous, no event
    threshold): r(hot, CHDE) and r(dry, CHDE) over the 360 months of each period,
    after removing the calendar-month means so the seasonal cycle does not drive r.
    """
    rows = []
    for key in KEYS:
        for region in REGION_ORDER:
            if get_valid(res, key, region) is None:
                continue
            for p in PERIODS:
                h = _deseason(get_area(res, key, region, "temp", p))
                d = _deseason(get_area(res, key, region, "spi", p))
                c = _deseason(get_area(res, key, region, "compound", p))
                rows.append({"model": key, "region": region, "period": p,
                             "r_hot_chde": _pearson(h, c), "r_dry_chde": _pearson(d, c)})
    return pd.DataFrame(rows)


def threshold_screen(sens_df):
    """
    Per severity and criterion: is the binary ensemble metric usable for each of the
    HOT, DRY and CHDE indicators?  Saturated = at or above SAT_MONTHS months/yr in either
    period; empty = below EMPTY_MONTHS months/yr in the future.
    """
    e = sens_df[sens_df["model"] == "Ensemble"]
    rows = []
    for (sev, crit), g in e.groupby(["severity", "criterion"]):
        row = {"severity": sev, "criterion": crit,
               "is_main": crit == MAIN_CRITERION_BY_SEV.get(str(sev))}
        for et in EXT_TYPES:
            ge = g[g["ext_type"] == et]
            sat = ge[(ge["freq_hist"] >= SAT_MONTHS) | (ge["freq_fut"] >= SAT_MONTHS)]["region"].tolist()
            emp = ge[ge["freq_fut"] < EMPTY_MONTHS]["region"].tolist()
            row[f"{et}_n_saturated"] = len(sat)
            row[f"{et}_saturated"] = ", ".join(sat)
            row[f"{et}_n_empty"] = len(emp)
            row[f"{et}_empty"] = ", ".join(emp)
        gc = g[g["ext_type"] == "compound"]
        row["chde_hist_max"] = round(gc["freq_hist"].max(), 1)
        row["chde_fut_min"] = round(gc["freq_fut"].min(), 1)
        row["chde_fut_max"] = round(gc["freq_fut"].max(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def analytic_reference(res, severity):
    """Independence and spatially independent cells (the binomial argument)."""
    sig = float(severity)
    p = stats.norm.sf(sig)
    pc = p * p
    rows = []
    for region in REGION_ORDER:
        ms = _members(res, "ENSEMBLE", region)
        if not ms:
            continue
        m0 = res[ms[0]][region]
        N, mean_a, valid = m0["n_cells"], m0["mean_cell_km2"], m0["valid_km2"]
        for crit in CRITERIA:
            th = thr(crit, valid)
            k = int(np.ceil(th / mean_a))
            rows.append({
                "severity": severity, "region": region, "criterion": crit,
                "p_hot_cell": p, "p_dry_cell": p, "p_chde_cell_indep": pc,
                "N_valid_cells": N, "threshold_Mkm2": th / 1e6,
                "threshold_pct_of_valid": 100 * th / valid,
                "cells_needed": k, "expected_cells_indep": N * pc,
                "P_region_event_binomial": stats.binom.sf(k - 1, N, pc),
            })
    return rows


def diagnostics(res, severity):
    p = stats.norm.sf(float(severity))
    th_crit = MAIN_CRITERION_BY_SEV[severity]
    rows = []
    for m in MODELS:
        if m not in res:
            continue
        for region, r in res[m].items():
            th = thr(th_crit, r["valid_km2"])
            hot_h, dry_h = r["obs"]["temp"]["hist"], r["obs"]["spi"]["hist"]
            rows.append({
                "severity": severity, "model": DISPLAY[m], "region": region,
                "valid_Mkm2": r["valid_km2"] / 1e6, "n_cells": r["n_cells"],
                "threshold_Mkm2": th / 1e6,
                "mean_hot_frac_hist": hot_h.mean() / r["valid_km2"],
                "mean_dry_frac_hist": dry_h.mean() / r["valid_km2"],
                "mean_chde_frac_hist": r["obs"]["compound"]["hist"].mean() / r["valid_km2"],
                "hot_area_over_threshold_hist": hot_h.mean() / th,
                "dry_area_over_threshold_hist": dry_h.mean() / th,
                "p_hot_event_hist": float((hot_h >= th).mean()),
                "p_dry_event_hist": float((dry_h >= th).mean()),
                "expected_cell_p": p, "expected_chde_indep": p * p,
                "compound_vs_hot&dry_mismatch_hist": r["consistency"]["hist"],
                "compound_vs_hot&dry_mismatch_fut": r["consistency"]["fut"],
            })
    return rows


# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------
def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {stem}.png/.pdf")


def plot_delta_bars(delta_df, severity, metric, corr_df):
    """
    metric: 'freq' (months/yr, binary) or 'area' (10^6 km2, continuous).
    Legend: correlation ACROSS regions of the ensemble changes (one value per figure).
    Box in each panel: correlation WITHIN the region between monthly flagged areas
    (ensemble, deseasonalised) for the historical (H) and future (F) period.
    """
    col = "delta_freq" if metric == "freq" else "delta_area"
    ylab = "Δ regional event months yr⁻¹" if metric == "freq" else "Δ mean affected area (10⁶ km²)"
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.32)
    x_base = np.arange(len(KEYS))
    bar_w = 0.22
    offsets = {"temp": -bar_w, "spi": 0.0, "compound": bar_w}

    sub = delta_df[delta_df["model"] == "ENSEMBLE"]
    piv = {et: sub[sub["ext_type"] == et].set_index("region")[col] for et in EXT_TYPES}
    common = piv["temp"].index.intersection(piv["spi"].index).intersection(piv["compound"].index)
    rh = rd = np.nan
    if len(common) >= 3:
        rh = stats.pearsonr(piv["temp"][common], piv["compound"][common])[0]
        rd = stats.pearsonr(piv["spi"][common], piv["compound"][common])[0]

    for pi, region in enumerate(REGION_ORDER):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        for et, off in offsets.items():
            for xi, key in enumerate(KEYS):
                s = delta_df[(delta_df["model"] == key) & (delta_df["region"] == region)
                             & (delta_df["ext_type"] == et)]
                v = s[col].values[0] if len(s) else 0.0
                ax.bar(x_base[xi] + off, v, bar_w, color=BAR_COLORS[et],
                       alpha=1.0 if key == "ENSEMBLE" else 0.55,
                       edgecolor=MODEL_COLORS[key], linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x_base)
        ax.set_xticklabels([DISPLAY[k] for k in KEYS], rotation=30, ha="right", fontsize=12)
        ax.set_title(region, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.tick_params(axis="y", labelsize=12)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.30 * (y1 - y0))       # headroom for the r box
        cr = corr_df[(corr_df["model"] == "ENSEMBLE") & (corr_df["region"] == region)]
        if not cr.empty:
            lines = ["monthly r with CHDE area", "(deseasonalised)", "        hot     dry"]
            for p, lab in (("hist", "H"), ("fut", "F")):
                q = cr[cr["period"] == p]
                if not q.empty:
                    a, b = q["r_hot_chde"].values[0], q["r_dry_chde"].values[0]
                    f = lambda v: "  n/a" if np.isnan(v) else f"{v:5.2f}"
                    lines.append(f"{lab}      {f(a)}   {f(b)}")
            ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                    va="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.85))

    handles = ([Patch(facecolor=BAR_COLORS[et], label=f"Δ {EXT_TITLES[et]}") for et in EXT_TYPES]
               + [Patch(facecolor=MODEL_COLORS[k], label=DISPLAY[k], edgecolor="black",
                        alpha=1.0 if k == "ENSEMBLE" else 0.55) for k in KEYS])
    ttl = ("Across regions (ensemble Δ, n = %d): r(Δhot, ΔCHDE) = %.2f, r(Δdry, ΔCHDE) = %.2f"
           % (len(common), rh, rd)) if not (np.isnan(rh) or np.isnan(rd)) else None
    fig.legend(handles=handles, loc="lower center", ncol=len(handles) // 2 + 1,
               fontsize=13, bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
               title=ttl, title_fontsize=13)
    _save(fig, f"delta_bars_{metric}_severity{severity.replace('.', '')}")


def plot_null_comparison(null_df, severity, metric, tag=""):
    """Simulated CHDE metric (bars) against the permutation null (black mean + range)."""
    o, nm, lo, hi = (("sim_freq", "null_freq_mean", "null_freq_lo", "null_freq_hi")
                     if metric == "freq" else
                     ("sim_area", "null_area_mean", "null_area_lo", "null_area_hi"))
    ylab = "Regional CHDE months yr⁻¹" if metric == "freq" else "Mean CHDE area (10⁶ km²)"
    nlab = "year-shuffled" if not tag else f"{BLOCK}-year-block-shuffled"
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.30)
    w = 0.36
    for pi, region in enumerate(REGION_ORDER):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        for xi, key in enumerate(KEYS):
            for p, off, alpha in (("hist", -w / 2, 0.45), ("fut", w / 2, 0.95)):
                r = null_df[(null_df["model"] == key) & (null_df["region"] == region)
                            & (null_df["period"] == p)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.bar(xi + off, r[o], w, color=MODEL_COLORS[key], alpha=alpha,
                       edgecolor="black", linewidth=0.6)
                ax.vlines(xi + off, r[lo], r[hi], color="black", linewidth=1.6, zorder=5)
                ax.plot(xi + off, r[nm], marker="_", color="black", markersize=11,
                        markeredgewidth=2, zorder=6)
        ax.set_xticks(np.arange(len(KEYS)))
        ax.set_xticklabels([DISPLAY[k] for k in KEYS], rotation=30, ha="right", fontsize=12)
        ax.set_title(region, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.tick_params(axis="y", labelsize=12)
    handles = [Patch(facecolor="gray", alpha=0.45, edgecolor="black", label="Simulated, historical"),
               Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="Simulated, future"),
               Line2D([0], [0], color="black", marker="_", markersize=11, linewidth=1.6,
                      label=f"Permutation null ({nlab}): mean and 2.5 to 97.5 % range")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    suffix = f"_{tag}" if tag else ""
    _save(fig, f"null_comparison_{metric}_severity{severity.replace('.', '')}{suffix}")


def plot_monthly_prob(res, severity, ext_type, ns, crit):
    """Probability of a regional event by CALENDAR month (December of the same calendar year)."""
    months = np.arange(1, 13)
    fig = plt.figure(figsize=(20, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.28)
    cache, bands, all_vals = {}, {}, []
    for region in REGION_ORDER:
        for key in KEYS:
            for p in PERIODS:
                ev = get_events(res, key, region, ext_type, p, crit)
                if ev is None:
                    continue
                cache[(key, region, p)] = ev.mean(axis=0)
                all_vals.extend(cache[(key, region, p)].tolist())
        if ext_type == "compound":
            for p in PERIODS:
                nev = get_null_events(res, "ENSEMBLE", region, p, crit, ns)
                if nev is None:
                    continue
                lo, hi = np.percentile(nev.mean(axis=1), NULL_PCT, axis=0)
                bands[(region, p)] = (lo, hi)
                all_vals.extend(hi.tolist())
    ymax = max(max(all_vals) * 1.12 if all_vals else 1.0, 0.05)

    for pi, region in enumerate(REGION_ORDER):
        row, col = pi // 3, pi % 3
        ax = fig.add_subplot(gs[row, col])
        for p, shade in (("hist", "0.65"), ("fut", "0.35")):
            if (region, p) in bands:
                lo, hi = bands[(region, p)]
                ax.fill_between(months, lo, hi, color=shade, alpha=0.25, linewidth=0)
        for key in KEYS:
            lw = 2.5 if key == "ENSEMBLE" else 1.5
            alpha = 1.0 if key == "ENSEMBLE" else 0.75
            for p, ls in (("hist", "-"), ("fut", "--")):
                if (key, region, p) in cache:
                    ax.plot(months, cache[(key, region, p)], color=MODEL_COLORS[key],
                            linewidth=lw, linestyle=ls, alpha=alpha)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0, ymax)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS, fontsize=12)
        ax.set_title(region, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="y", labelsize=12)
        if col == 0:
            ax.set_ylabel(f"Probability of regional event ({crit_label(crit)})", fontsize=11)
        if row == 1:
            ax.set_xlabel("Month", fontsize=12)

    model_handles = [Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.0, label=DISPLAY[m])
                     for m in MODELS]
    model_handles.append(Line2D([0], [0], color=MODEL_COLORS["ENSEMBLE"], linewidth=2.5,
                                label=f"Ensemble (≥ {ENSEMBLE_MIN_MODELS} of {len(MODELS)} models)"))
    style_handles = [Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Historical"),
                     Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Future")]
    if ext_type == "compound":
        style_handles += [Patch(facecolor="0.65", alpha=0.25, label="Ensemble null, historical"),
                          Patch(facecolor="0.35", alpha=0.25, label="Ensemble null, future")]
    leg1 = fig.legend(handles=model_handles, loc="lower center", ncol=len(model_handles),
                      fontsize=12, bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
                      title="Model", title_fontsize=13)
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="lower center", ncol=len(style_handles),
               fontsize=13, bbox_to_anchor=(0.5, -0.10), framealpha=0.9,
               title="Period", title_fontsize=12)
    _save(fig, f"monthly_prob_{ext_type}_severity{severity.replace('.', '')}")


# -----------------------------------------------------------------------------
# SEASONAL PROBABILITY TABLE (Excel)
# -----------------------------------------------------------------------------
METRIC_COLS = [
    "n months",
    "P(hot)", "P(dry)", "P(CHDE)", "P(hot)xP(dry)", "P(hot & dry, regional)",
    "P(CHDE|hot)", "P(CHDE|dry)", "P(hot or dry)",
    "Saturated (hot or dry >= 0.98)",
    "Null P(CHDE) mean", "Null P(CHDE) 2.5%", "Null P(CHDE) 97.5%",
    "Sim/Null ratio", "p (sim >= null)",
    "Block-null P(CHDE) mean", "Sim/Block-null ratio", "p (sim >= block null)",
    "CHDE area (1e6 km2)", "Null CHDE area (1e6 km2)", "CHDE area (% of region)",
]
RATIO_COLS = ("Sim/Null ratio", "Sim/Block-null ratio")


def season_metrics(h, d, c, ac, nev, nar, nev_b, valid_km2):
    """h, d, c, ac: (units, 3) observed-type arrays; nev, nar, nev_b: (n_perm, units, 3)."""
    ph, pdry, pc = h.mean(), d.mean(), c.mean()
    joint = (h & d).mean()
    p_c_h = (c & h).mean() / ph if ph > 0 else np.nan
    p_c_d = (c & d).mean() / pdry if pdry > 0 else np.nan
    n_p = nev.mean(axis=(1, 2))
    b_p = nev_b.mean(axis=(1, 2))
    n_mean, b_mean = n_p.mean(), b_p.mean()
    lo, hi = np.percentile(n_p, NULL_PCT)
    pval = (1 + (n_p >= pc - 1e-12).sum()) / (1 + len(n_p))
    pval_b = (1 + (b_p >= pc - 1e-12).sum()) / (1 + len(b_p))
    a_obs = ac.mean() / 1e6
    a_null = nar.mean() / 1e6

    def r(x, k=4):
        return round(float(x), k) if not np.isnan(x) else np.nan
    return {
        "n months": int(h.size),
        "P(hot)": r(ph), "P(dry)": r(pdry), "P(CHDE)": r(pc),
        "P(hot)xP(dry)": r(ph * pdry), "P(hot & dry, regional)": r(joint),
        "P(CHDE|hot)": r(p_c_h), "P(CHDE|dry)": r(p_c_d),
        "P(hot or dry)": r(ph + pdry - joint),
        "Saturated (hot or dry >= 0.98)": "yes" if (ph >= SAT_PROB or pdry >= SAT_PROB) else "no",
        "Null P(CHDE) mean": r(n_mean), "Null P(CHDE) 2.5%": r(lo),
        "Null P(CHDE) 97.5%": r(hi),
        "Sim/Null ratio": r(pc / n_mean if n_mean > 0 else np.nan, 3),
        "p (sim >= null)": r(pval, 4),
        "Block-null P(CHDE) mean": r(b_mean),
        "Sim/Block-null ratio": r(pc / b_mean if b_mean > 0 else np.nan, 3),
        "p (sim >= block null)": r(pval_b, 4),
        "CHDE area (1e6 km2)": r(a_obs, 4), "Null CHDE area (1e6 km2)": r(a_null, 4),
        "CHDE area (% of region)": r(100 * ac.mean() / valid_km2, 3),
    }


def compute_prob_table(res, crit, ns, ns_blk):
    rows = []
    for key in KEYS:
        for region in REGION_ORDER:
            valid = get_valid(res, key, region)
            if valid is None:
                continue
            for p in PERIODS:
                evh = get_events(res, key, region, "temp", p, crit)
                evd = get_events(res, key, region, "spi", p, crit)
                evc = get_events(res, key, region, "compound", p, crit)
                ac = get_area(res, key, region, "compound", p)
                for season in SEASONS:
                    nev, nar = null_season(res, key, region, p, crit, ns, season)
                    nev_b, _ = null_season(res, key, region, p, crit, ns_blk, season)
                    row = {"region": region, "season": season,
                           "period": PERIOD_LABEL[p], "model": DISPLAY[key]}
                    row.update(season_metrics(
                        season_units(evh, season), season_units(evd, season),
                        season_units(evc, season), season_units(ac, season),
                        nev, nar, nev_b, valid))
                    rows.append(row)
    return pd.DataFrame(rows)


_HDR_FILL = PatternFill("solid", start_color="2E4057", end_color="2E4057")
_REG_FILL = PatternFill("solid", start_color="C8D8E4", end_color="C8D8E4")
_DEP_FILL = PatternFill("solid", start_color="FFF2CC", end_color="FFF2CC")
_SAT_FILL = PatternFill("solid", start_color="F8CBAD", end_color="F8CBAD")
_THIN = Border(left=Side(style="thin"), right=Side(style="thin"),
               top=Side(style="thin"), bottom=Side(style="thin"))
_WHT_FNT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
_BLD_FNT = Font(name="Arial", bold=True, size=10)
_NRM_FNT = Font(name="Arial", size=9)
_CTR = Alignment(horizontal="center", vertical="center", wrap_text=True)
_LFT = Alignment(horizontal="left", vertical="center")


def readme_lines():
    return [
        "Probability tables, CHDE attribution v4",
        "",
        f"Regional event: flagged area >= threshold; main criterion by severity: "
        f"{MAIN_CRITERION_BY_SEV} ({crit_label(MAIN_CRITERION)}); area-weighted, valid cells only.",
        "Regions are disjoint unions of lat/lon rectangles (region_definitions.csv).",
        "Seasons: MAM, JJA, SON use months of the same calendar year (n = 90). "
        "DJF = December of the preceding year with January and February (29 winters, n = 87).",
        "P(CHDE): regional event in the compound field (hot and dry in the same cells).",
        "P(hot)xP(dry): independence benchmark built from the two regional hot and dry series.",
        "P(hot & dry, regional): months in which both regional hot and regional dry events occur.",
        "P(CHDE|hot) = P(CHDE and hot) / P(hot), same for dry.",
        f"Saturated: P(hot) or P(dry) >= {SAT_PROB}. The conditional probabilities are then close to "
        "P(CHDE) by construction and carry no information about coupling.",
        f"Null: dry indicator paired with the hot indicator of a different year (random derangements, "
        f"{N_PERM} permutations); for DJF whole winters are paired. Spatial structure and marginals "
        "are preserved; a shared trend between hot and dry is removed.",
        f"Block null: as above but years are shuffled only inside {BLOCK}-year blocks, so the slow "
        "shared trend is kept. Compare the two ratios to see how much of the excess is trend.",
        "Sim/Null ratio: simulated P(CHDE) divided by the null mean. p: one-sided permutation p-value.",
        f"Ensemble: event if at least {ENSEMBLE_MIN_MODELS} of {len(MODELS)} models flag it; ensemble null "
        "built the same way from aligned permutations.",
        "CHDE area (% of region): mean flagged compound area as a share of the region's valid area.",
    ]


def write_prob_excel(all_prob, filepath):
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "README"
    for line in readme_lines():
        ws0.append([line])
    ws0.column_dimensions["A"].width = 130
    ws0["A1"].font = _BLD_FNT

    sat_ci = 4 + METRIC_COLS.index("Saturated (hot or dry >= 0.98)") + 1
    for severity, df in all_prob.items():
        ws = wb.create_sheet(title=f"Sev {severity} sigma")
        headers = ["Region", "Season", "Period", "Model"] + METRIC_COLS
        ws.append(headers)
        hdr_row = ws.max_row
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=hdr_row, column=ci, value=h)
            c.fill, c.font, c.border, c.alignment = _HDR_FILL, _WHT_FNT, _THIN, _CTR
        for ri, region in enumerate(REGION_ORDER):
            reg_fill = (_REG_FILL if ri % 2 == 0 else
                        PatternFill("solid", start_color="FFFFFF", end_color="FFFFFF"))
            sub_r = df[df["region"] == region]
            for season in SEASONS:
                for period in ("Historical", "Future"):
                    for model in [DISPLAY[k] for k in KEYS]:
                        sub = sub_r[(sub_r["season"] == season) & (sub_r["period"] == period)
                                    & (sub_r["model"] == model)]
                        if sub.empty:
                            continue
                        vals = [region, season, period, model]
                        for m in METRIC_COLS:
                            v = sub[m].values[0]
                            vals.append(None if (isinstance(v, float) and np.isnan(v)) else v)
                        ws.append(vals)
                        r = ws.max_row
                        saturated = ws.cell(row=r, column=sat_ci).value == "yes"
                        for ci in range(1, len(vals) + 1):
                            cell = ws.cell(row=r, column=ci)
                            cell.border, cell.font = _THIN, _NRM_FNT
                            if ci <= 4:
                                cell.alignment = _LFT
                                if ci == 1:
                                    cell.fill, cell.font = reg_fill, _BLD_FNT
                            else:
                                cell.alignment = _CTR
                                name = METRIC_COLS[ci - 5]
                                if name in RATIO_COLS and cell.value is not None:
                                    cell.fill = _DEP_FILL
                                    cell.font = Font(name="Arial", size=9, bold=True)
                                if saturated and name in ("P(CHDE|hot)", "P(CHDE|dry)",
                                                          "Saturated (hot or dry >= 0.98)"):
                                    cell.fill = _SAT_FILL
        for ci, w in enumerate([18, 7, 12, 14] + [14] * len(METRIC_COLS), 1):
            ws.column_dimensions[get_column_letter(ci)].width = w
        ws.freeze_panes = ws.cell(row=hdr_row + 1, column=5)
        ws.sheet_properties.tabColor = "2E4057"
    wb.save(filepath)
    print(f"  [OK] {Path(filepath).name}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CHDE ATTRIBUTION ANALYSIS  v4")
    print(f"Historical : {PERIODS['hist'][0]} to {PERIODS['hist'][1]}")
    print(f"Future     : {PERIODS['fut'][0]} to {PERIODS['fut'][1]}")
    print(f"Criteria   : main by severity {MAIN_CRITERION_BY_SEV}")
    print(f"             sensitivity grid: {', '.join(CRITERIA)}")
    print(f"Nulls      : year-shuffled and {BLOCK}-year-block-shuffled, "
          f"{N_PERM} permutations each, seed {RNG_SEED}")
    print(f"Models     : {', '.join(MODELS)} + Ensemble (>= {ENSEMBLE_MIN_MODELS} of {len(MODELS)} models)")
    print(f"Severities : {', '.join(SEVERITIES)}")
    print("=" * 70)

    check_regions_disjoint()
    write_region_outputs()

    nulls = make_null_sets()
    ns, ns_blk = nulls["year"], nulls[f"block{BLOCK}"]
    all_prob, sens_rows, ana_rows, diag_rows = {}, [], [], []

    for severity in SEVERITIES:
        print(f"\n{'-' * 60}\n  Severity {severity} sigma\n{'-' * 60}")
        _NULL_CACHE.clear()
        res = load_all(severity)
        if not res:
            print("  no data, skipping")
            continue
        sev_lab = severity.replace(".", "")
        crit = MAIN_CRITERION_BY_SEV[severity]
        print(f"  Main criterion: {crit}")

        diag_rows += diagnostics(res, severity)
        ana_rows += analytic_reference(res, severity)

        delta_df = compute_summary(res, crit)
        delta_df.to_csv(OUTPUT_DIR / f"delta_data_sev{sev_lab}.csv", index=False)
        null_df = compute_null_summary(res, crit, ns)
        null_df.to_csv(OUTPUT_DIR / f"null_summary_sev{sev_lab}.csv", index=False)
        null_blk = compute_null_summary(res, crit, ns_blk)
        null_blk.to_csv(OUTPUT_DIR / f"null_summary_sev{sev_lab}_block{BLOCK}.csv", index=False)

        ens = delta_df[delta_df["model"] == "ENSEMBLE"]
        for et in EXT_TYPES:
            e = ens[ens["ext_type"] == et]
            bad = e[(e["freq_hist"] >= SAT_MONTHS) | (e["freq_fut"] >= SAT_MONTHS)]["region"].tolist()
            if bad:
                print(f"  [NOTE] {EXT_TITLES[et]} indicator at or above {SAT_MONTHS:.0f} months/yr "
                      f"in: {', '.join(bad)}. Conditional probabilities there are flagged.")

        corr_df = compute_region_corr(res)
        corr_df.to_csv(OUTPUT_DIR / f"region_corr_sev{sev_lab}.csv", index=False)

        for metric in ("freq", "area"):
            plot_delta_bars(delta_df, severity, metric, corr_df)
            plot_null_comparison(null_df, severity, metric)
            plot_null_comparison(null_blk, severity, metric, tag=f"block{BLOCK}")
        for et in EXT_TYPES:
            plot_monthly_prob(res, severity, et, ns, crit)

        sens_rows += sensitivity_table(res, severity, ns)
        all_prob[severity] = compute_prob_table(res, crit, ns, ns_blk)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUTPUT_DIR / "sensitivity_thresholds.csv", index=False)
    if not sens_df.empty:
        screen = threshold_screen(sens_df)
        screen.to_csv(OUTPUT_DIR / "threshold_screen.csv", index=False)
        print("\nThreshold screen (ensemble, binary metric):")
        print(screen.to_string(index=False))
    pd.DataFrame(ana_rows).to_csv(OUTPUT_DIR / "analytic_reference.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUTPUT_DIR / "diagnostics.csv", index=False)
    if all_prob:
        write_prob_excel(all_prob, OUTPUT_DIR / "probability_tables.xlsx")

    print("\n" + "=" * 70)
    print(f"DONE. Outputs in: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
