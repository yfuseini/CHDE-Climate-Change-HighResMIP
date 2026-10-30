"""
CHDE coverage and exceedance analysis  v2
=========================================
Replaces the CDO/shell coverage pipeline and the old plotting script.

Why a rewrite
  - REGIONS: the old script read regional files made with overlapping boxes.  Coverage is now
    computed here from the monthly compound flag files on the SAME disjoint regions as
    chde_attribution_v4.py (both import chde_regions.py), plus Global and a Global
    sensitivity that excludes Antarctica and Greenland.
  - Coverage definition (as in the manuscript): share of valid land area, weighted by exact
    spherical cell area, in which at least one CHDE month of the given intensity occurs in a
    calendar year.  The result is reconciled against the old CDO global series when those
    files exist (reconciliation.csv).

New checks and robustness (items 59 and 62 of the review, plus the rest)
  59  FALSE-POSITIVE TEST OF THE EXCEEDANCE RULE.  The rule (mean + k sigma of the baseline,
      sustained for 5 years) is applied to 5000 AR(1) surrogate series with no climate trend,
      whose variability and persistence come from the detrended 1980-2014 ensemble-mean
      series.  Reported: how often the rule fires by chance (the rule as practised, and the
      rule with the real threshold), the year such false alarms typically appear, and a
      CALIBRATED threshold that holds the false-positive rate at 5 %.  The emergence year at
      the calibrated threshold is reported next to the original one.
  62  DEPENDENCE EVALUATION.  For every cell the hot and dry monthly flags give the joint
      frequency P(hot and dry), the independence benchmark P(hot) P(dry), their ratio (lift)
      and the phi coefficient.  Models are compared with the reference for 1985-2014, and the
      same statistics are given for 2021-2050.  This tests whether the bias-adjusted data
      keep the temperature-precipitation dependence that defines a CHDE.
  Further robustness
  - Bias correction in three modes (additive = original, multiplicative, none); additive and
    multiplicative are clipped to 0-100 %.  Anomalies are relative to the reference 1980-2014 mean.
  - Threshold sigma from four sources: ensemble mean (original), pooled single models,
    reference, detrended ensemble.  The ensemble mean of four models has less noise than a
    single series, so the original sigma is optimistic.
  - Emergence years for every model (range) and the share of years above the threshold after
    emergence.
  - Reference series extended to OBS_END (2019) and compared with the ensemble after 2014 and
    with the threshold: a direct test of "already emerging" that needs no model.
  - Splice check at 2014/15: jump of the mean of 2015-2019 relative to the extrapolated
    1995-2014 trend, in pp and in residual standard deviations, and the percentile of the
    2014 five-year block difference among all other block differences.
  - Year-aligned ensembles (the old code truncated by length), NaN-safe statistics, sample
    standard deviation (ddof = 1), smoothing with a shrinking window at the edges instead of
    linear extrapolation (np.RankWarning no longer exists in NumPy 2).

Reference files
  The script needs monthly flags for the reference (W5E5).  It looks in BASE_PATH/OBS/6months
  with the same naming scheme as the models.  If your names differ, fill OBS_FILE_OVERRIDES
  below; the files it picked are printed.  Without a reference file for a severity, that
  severity is run without bias correction and without reference series (and says so).

Outputs (OUTPUT_DIR)
  coverage_annual.csv                 raw coverage, every source, region, year
  coverage_ensemble_main.csv          corrected ensemble, spread and reference as plotted
  bias_table.csv, reconciliation.csv
  emergence_sensitivity.csv           bias mode x sigma source x data type
  emergence_main.csv                  main settings with false-positive test, calibration,
                                      reference extension and per-model emergence
  splice_check.csv
  dependence_summary.csv, dependence_maps_sevXX.png, dependence_lift_sevXX.png
  coverage_regional_<type>_sevXX.png/pdf   (6 regions + Global, 4x2 grid)
  coverage_global_<type>.png/pdf, coverage_global_all_types.png/pdf
  emergence_sensitivity.png/pdf, region_hash.txt

Environment variables (optional)
  CHDE_BASE_PATH, CHDE_OUT, CHDE_SEV, CHDE_N_SURR (5000), CHDE_OBS_END (2019),
  CHDE_DEPENDENCE (1), CHDE_CACHE (1)
"""

import os
import pickle
import hashlib
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.lib.stride_tricks import sliding_window_view

from chde_regions import (REGIONS, REGION_ORDER, band_area_km2, spacing, find_latlon,
                          region_mask, check_regions_disjoint, regions_hash)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_PATH = Path(os.environ.get(
    "CHDE_BASE_PATH",
    "/work/ch0636/u301861/phd/HighRESMIP/results/compoundXTREMe_3"))
OUTPUT_DIR = Path(os.environ.get("CHDE_OUT", "coverage_outputs_v2"))
CACHE_DIR = OUTPUT_DIR / "cache"
USE_CACHE = os.environ.get("CHDE_CACHE", "1") == "1"
CACHE_VERSION = "cov2.0"

MODELS = ["CNRM-CERFACS", "EC-Earth-HR", "HadGEM3-HM", "MPI-XR"]
OBS_NAME = "OBS"
SEVERITIES = [s.strip() for s in os.environ.get("CHDE_SEV", "1.0,1.5,2.0").split(",")]
SEV_KEYS = {
    "1.0": {"temp": "TEMP1.0std", "spi": "SPI1",   "compound": "TEMP1.0std_SPI1"},
    "1.5": {"temp": "TEMP1.5std", "spi": "SPI1.5", "compound": "TEMP1.5std_SPI1.5"},
    "2.0": {"temp": "TEMP2.0std", "spi": "SPI2",   "compound": "TEMP2.0std_SPI2"},
}
LEGACY_LABEL = {"1.0": "T1.0_SPI1", "1.5": "T1.5_SPI1.5", "2.0": "T2.0_SPI2"}
SEV_NAME = {"1.0": "Low (1.0σ)", "1.5": "Moderate (1.5σ)", "2.0": "High (2.0σ)"}
SEV_COLOR = {"1.0": "#2E86AB", "1.5": "#A23B72", "2.0": "#F18F01"}
STD_MULT = {"1.0": 1.0, "1.5": 1.5, "2.0": 2.0}
VAR_NAMES = {"temp": "tasmax", "compound": "CompoundXTREMe"}

# Fill if the reference files do not follow the model naming scheme, e.g.
# OBS_FILE_OVERRIDES = {("compound", "1.0"): "/path/OBS_compound_....nc", ("temp", "1.0"): ..., ("spi", "1.0"): ...}
OBS_FILE_OVERRIDES = {}

YEARS = np.arange(1980, 2051)
HIST = (1980, 2014)                 # baseline for bias correction and thresholds (as before)
SPLICE_YEAR = 2014
OBS_END = int(os.environ.get("CHDE_OBS_END", 2019))
DEP_PERIODS = {"hist": (1985, 2014), "fut": (2021, 2050)}   # same windows as chde_attribution_v4.py

SMOOTH_WINDOW = 5
SMOOTH_MIN = 3                      # shrinking window at the edges
SUSTAIN_K = 5
BIAS_MODES = ("additive", "multiplicative", "none")
MAIN_BIAS = "additive"
SIGMA_SOURCES = ("ensemble", "models", "reference", "ensemble_detrended")
MAIN_SIGMA = "ensemble"
EMERGENCE_TYPES = ("smoothed", "timeseries")
ALL_TYPES = ("smoothed", "timeseries", "anomalies")
TYPE_LABEL = {"smoothed": "5-year smoothed", "timeseries": "Raw annual", "anomalies": "Anomalies"}

N_SURR = int(os.environ.get("CHDE_N_SURR", 5000))
FPR_ALPHA = 0.05
RNG_SEED = 20261012
RUN_DEPENDENCE = os.environ.get("CHDE_DEPENDENCE", "1") == "1"

GLOBAL = "Global"
GLOBAL_NOPOLAR = "Global excl. Antarctica and Greenland"
PANEL_REGIONS = REGION_ORDER + [GLOBAL]            # 6 regions + Global = 7 panels
ALL_COV = PANEL_REGIONS + [GLOBAL_NOPOLAR]
ENS_COLOR = "#8B008B"
MODEL_COLORS = {"CNRM-CERFACS": "#1f77b4", "EC-Earth-HR": "#ff7f0e",
                "HadGEM3-HM": "#2ca02c", "MPI-XR": "#d62728"}
DISPLAY = {"CNRM-CERFACS": "CNRM", "EC-Earth-HR": "EC-Earth3P",
           "HadGEM3-HM": "HadGEM3", "MPI-XR": "MPI-XR", "OBS": "Reference"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HIST_MASK = (YEARS >= HIST[0]) & (YEARS <= HIST[1])


# -----------------------------------------------------------------------------
# FILES
# -----------------------------------------------------------------------------
def model_file(model, sev, kind):
    sk = SEV_KEYS[sev][kind]
    d = BASE_PATH / model / "6months"
    if kind == "compound":
        return d / f"{model}_compound_{sk}_corrected_merged.nc"
    if kind == "temp":
        return d / "phase1" / f"{sk}_hist-baseline_{model}_corrected_merged.nc"
    return d / "phase1" / f"Ind_{sk}_{model}_corrected_merged.nc"


def obs_file(sev, kind):
    if (kind, sev) in OBS_FILE_OVERRIDES:
        return Path(OBS_FILE_OVERRIDES[(kind, sev)])
    sk = SEV_KEYS[sev][kind]
    d = BASE_PATH / OBS_NAME / "6months"
    pats = {"compound": [f"*compound_{sk}_*.nc"],
            "temp": [f"phase1/{sk}_*.nc", f"{sk}_*.nc"],
            "spi": [f"phase1/Ind_{sk}_*.nc", f"Ind_{sk}_*.nc"]}[kind]
    for pat in pats:
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def flag_file(source, sev, kind):
    return obs_file(sev, kind) if source == OBS_NAME else model_file(source, sev, kind)


def var_for(sev, kind):
    return VAR_NAMES[kind] if kind != "spi" else SEV_KEYS[sev]["spi"]


def open_flags(path, var):
    ds = xr.open_dataset(path)
    if var not in ds:
        dv = [v for v in ds.data_vars if "bnds" not in v.lower() and "bounds" not in v.lower()]
        var = dv[0]
    return ds[var]


def to_tll(da, latdim, londim):
    for d in [d for d in da.dims if d not in ("time", latdim, londim)]:
        da = da.isel({d: 0})
    return da.transpose("time", latdim, londim)


def _stamp(path):
    st = Path(path).stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


# -----------------------------------------------------------------------------
# COVERAGE FROM MONTHLY FLAGS
# -----------------------------------------------------------------------------
def coverage_masks(lat, lon):
    m = {r: region_mask(r, lat, lon) for r in REGION_ORDER}
    m[GLOBAL] = np.ones((lat.size, lon.size), bool)
    lonn = (np.asarray(lon, float) + 180.0) % 360.0 - 180.0
    green = ((lat[:, None] >= 59.5) & (lat[:, None] < 84.0)
             & (lonn[None, :] >= -74.0) & (lonn[None, :] < -11.0))
    m[GLOBAL_NOPOLAR] = (lat[:, None] >= -60.0) & ~green
    return m


def compute_coverage(source, sev):
    """Annual 'at least one CHDE' coverage (%) per region.  Returns dict or None."""
    path = flag_file(source, sev, "compound")
    if path is None or not Path(path).exists():
        return None
    key = hashlib.md5(f"{path}|{_stamp(path)}|{regions_hash()}|{CACHE_VERSION}|{YEARS[0]}-{YEARS[-1]}"
                      .encode()).hexdigest()[:10]
    cp = CACHE_DIR / f"cov_{source}_{sev}_{key}.pkl"
    if USE_CACHE and cp.exists():
        with open(cp, "rb") as f:
            print(f"  [CACHE] coverage {DISPLAY.get(source, source)} {sev}")
            return pickle.load(f)

    print(f"  Reading {DISPLAY.get(source, source)} {sev}: {Path(path).name}")
    da = open_flags(path, VAR_NAMES["compound"])
    lc, oc = find_latlon(da)
    if da[lc].ndim != 1:
        raise ValueError("coverage needs a regular lat/lon grid")
    latdim, londim = da[lc].dims[0], da[oc].dims[0]
    lat, lon = da[lc].values.astype(float), da[oc].values.astype(float)
    yrs = np.asarray(da.indexes["time"].year)
    Y, nlat, nlon = len(YEARS), lat.size, lon.size
    cube = np.zeros((Y, nlat, nlon), bool)
    valid = np.zeros((nlat, nlon), bool)
    nmon = np.zeros(Y, int)
    for yi, y in enumerate(YEARS):
        idx = np.where(yrs == y)[0]
        if idx.size == 0:
            continue
        ch = to_tll(da.isel(time=idx), latdim, londim).values
        valid |= np.isfinite(ch).any(axis=0)
        cube[yi] = (ch == 1).any(axis=0)
        nmon[yi] = idx.size
    short = [int(YEARS[i]) for i in range(Y) if 0 < nmon[i] < 12]
    if short:
        print(f"    [WARN] years with fewer than 12 months: {short[:5]}")
    w = band_area_km2(lat, spacing(lat), spacing(lon))[:, None] * np.ones((1, nlon))
    flat = cube.reshape(Y, -1).astype(np.float64)
    cov, vmk = {}, {}
    for r, m in coverage_masks(lat, lon).items():
        wm = (w * (m & valid)).ravel()
        if wm.sum() == 0:
            continue
        c = flat @ wm / wm.sum() * 100.0
        c[nmon == 0] = np.nan
        cov[r], vmk[r] = c, wm.sum() / 1e6
    out = {"cov": cov, "valid_Mkm2": vmk, "path": str(path)}
    if USE_CACHE:
        with open(cp, "wb") as f:
            pickle.dump(out, f)
    return out


def legacy_global(source, sev):
    p = BASE_PATH / source / "6months" / "annual_coverage" / f"{LEGACY_LABEL[sev]}_any_annual_coverage_timeseries.nc"
    if not p.exists():
        return None
    with xr.open_dataset(p) as ds:
        dv = [v for v in ds.data_vars if "bnds" not in v.lower() and "bounds" not in v.lower()]
        da = ds[dv[0]]
        return da.time.dt.year.values, da.values.flatten().astype(float) * 100.0


def reconcile(source, sev, cov):
    """Compare Global coverage computed here with the old CDO series."""
    lg = legacy_global(source, sev)
    if lg is None or GLOBAL not in cov["cov"]:
        return None
    yrs, vals = lg
    s = pd.Series(vals, index=yrs)
    mine = pd.Series(cov["cov"][GLOBAL], index=YEARS)
    common = [y for y in YEARS if y in s.index and np.isfinite(mine[y])]
    if not common:
        return None
    d = (mine[common] - s[common]).abs()
    return {"severity": sev, "source": source, "n_years": len(common),
            "max_abs_diff_pp": float(d.max()), "mean_abs_diff_pp": float(d.mean()),
            "year_of_max": int(d.idxmax())}


# -----------------------------------------------------------------------------
# SERIES TOOLS
# -----------------------------------------------------------------------------
def smooth1d(x):
    return (pd.Series(x).rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_MIN)
            .mean().to_numpy())


def smooth_rows(a):
    """Row-wise centred running mean with a shrinking window at the edges."""
    h = SMOOTH_WINDOW // 2
    pad = np.pad(a, ((0, 0), (h, h)), constant_values=np.nan)
    win = sliding_window_view(pad, SMOOTH_WINDOW, axis=1)
    cnt = np.isfinite(win).sum(axis=2)
    out = np.nansum(win, axis=2) / np.maximum(cnt, 1)
    out[cnt < SMOOTH_MIN] = np.nan
    return out


def transform(x, dtype, ref_mean):
    if x is None:
        return None
    if dtype == "timeseries":
        return x
    if dtype == "smoothed":
        return smooth1d(x)
    return x - ref_mean


def bias_correct(raw, obs, mode):
    """raw: model -> array(len YEARS).  Returns corrected dict (clipped to 0-100 %)."""
    if obs is None or mode == "none":
        return {m: v.copy() for m, v in raw.items()}
    om = np.nanmean(obs[HIST_MASK])
    out = {}
    for m, v in raw.items():
        mm = np.nanmean(v[HIST_MASK])
        if mode == "additive":
            c = v + (om - mm)
        else:
            c = v * (om / mm) if mm > 0 else v.copy()
        out[m] = np.clip(c, 0.0, 100.0)
    return out


def bias_rows(raw, obs, sev, region):
    if obs is None:
        return []
    om = np.nanmean(obs[HIST_MASK])
    rows = []
    for m, v in raw.items():
        mm = np.nanmean(v[HIST_MASK])
        add = v + (om - mm)
        rows.append({"severity": sev, "region": region, "model": m, "reference_mean_pct": om,
                     "model_mean_pct": mm, "additive_bias_pp": om - mm,
                     "multiplicative_factor": om / mm if mm > 0 else np.nan,
                     "years_clipped_below_0_if_additive": int((add < 0).sum())})
    return rows


def ensemble(series):
    arr = np.vstack([series[m] for m in MODELS if m in series])
    ok = np.isfinite(arr).all(axis=0)
    mean = np.where(ok, arr.mean(axis=0), np.nan)
    lo = np.where(ok, arr.min(axis=0), np.nan)
    hi = np.where(ok, arr.max(axis=0), np.nan)
    return mean, lo, hi


def baseline_stats(series_models, ens_mean, obs_t, source):
    e = ens_mean[HIST_MASK]
    mu = np.nanmean(e)
    if source == "ensemble":
        sd = np.nanstd(e, ddof=1)
    elif source == "models":
        sd = np.nanmean([np.nanstd(v[HIST_MASK], ddof=1) for v in series_models.values()])
    elif source == "reference":
        sd = np.nanstd(obs_t[HIST_MASK], ddof=1) if obs_t is not None else np.nan
    else:                                               # ensemble_detrended
        t = np.arange(e.size)
        ok = np.isfinite(e)
        c = np.polyfit(t[ok], e[ok], 1)
        sd = (e[ok] - np.polyval(c, t[ok])).std(ddof=2)
    return mu, sd


def first_sustained(series, thr, k=SUSTAIN_K):
    """First year of the first run of k consecutive years above thr (strictly), else None."""
    if series is None or thr is None or not np.isfinite(thr):
        return None
    above = np.isfinite(series) & (series > thr)
    if above.size < k:
        return None
    idx = np.flatnonzero(sliding_window_view(above, k).all(axis=1))
    return int(YEARS[idx[0]]) if idx.size else None


# -----------------------------------------------------------------------------
# FALSE-POSITIVE TEST AND CALIBRATION  (item 59)
# -----------------------------------------------------------------------------
def simulate_ar1(n, L, mu, sd, phi, rng, burn=50):
    e = rng.standard_normal((n, L + burn)) * sd * np.sqrt(1.0 - phi ** 2)
    x = np.empty_like(e)
    x[:, 0] = rng.standard_normal(n) * sd
    for t in range(1, L + burn):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return mu + x[:, burn:]


def surrogate_test(ens_raw, dtype, mult, thr_real, rng):
    """
    Stationary AR(1) surrogates whose mean is the 1980-2014 mean of the ensemble series and
    whose variance and lag-1 autocorrelation come from the linearly detrended baseline.
    """
    x = ens_raw[HIST_MASK]
    t = np.arange(x.size)
    res = x - np.polyval(np.polyfit(t, x, 1), t)
    sd = res.std(ddof=2)
    if not np.isfinite(sd) or sd <= 0:
        return None
    phi = float(np.clip(np.corrcoef(res[:-1], res[1:])[0, 1], -0.2, 0.9))
    sims = simulate_ar1(N_SURR, len(YEARS), x.mean(), sd, phi, rng)
    if dtype == "smoothed":
        sims = smooth_rows(sims)
    nh = int(HIST_MASK.sum())
    base = sims[:, :nh]
    thr_i = np.nanmean(base, axis=1) + mult * np.nanstd(base, axis=1, ddof=1)
    run = sliding_window_view(sims > thr_i[:, None], SUSTAIN_K, axis=1).all(axis=2)
    det = run.any(axis=1)
    first = run.argmax(axis=1)
    run_r = sliding_window_view(sims > thr_real, SUSTAIN_K, axis=1).all(axis=2)
    S = sliding_window_view(sims, SUSTAIN_K, axis=1).min(axis=2).max(axis=1)
    return {
        "ar1_phi": phi, "sd_detrended": sd,
        "fpr_procedure": float(det.mean()),
        "fpr_procedure_run_starts_by_2014": float((det & (first <= SPLICE_YEAR - YEARS[0])).mean()),
        "fp_median_start_year": float(np.median(YEARS[first[det]])) if det.any() else np.nan,
        "fpr_real_threshold": float(run_r.any(axis=1).mean()),
        "thr_calibrated": float(np.quantile(S, 1.0 - FPR_ALPHA)),
    }


# -----------------------------------------------------------------------------
# REFERENCE EXTENSION AND SPLICE CHECK
# -----------------------------------------------------------------------------
def reference_extension(obs_raw, ens_raw, dtype, thr, ref_mean):
    if obs_raw is None:
        return None
    ext = (YEARS > HIST[1]) & np.isfinite(obs_raw)
    if ext.sum() == 0:
        return None
    o = transform(obs_raw, dtype, ref_mean)
    e = transform(ens_raw, dtype, ref_mean)
    ext = (YEARS > HIST[1]) & np.isfinite(o)
    return {"ext_first_year": int(YEARS[ext][0]), "ext_last_year": int(YEARS[ext][-1]),
            "ext_n_years": int(ext.sum()), "ext_reference_mean": float(np.nanmean(o[ext])),
            "ext_ensemble_mean": float(np.nanmean(e[ext])),
            "ext_reference_minus_ensemble": float(np.nanmean(o[ext] - e[ext])),
            "ext_reference_years_above_thr": int((o[ext] > thr).sum()),
            "ext_ensemble_years_above_thr": int((e[ext] > thr).sum())}


def splice_check(x):
    fit = (YEARS >= 1995) & (YEARS <= SPLICE_YEAR)
    post = (YEARS > SPLICE_YEAR) & (YEARS <= SPLICE_YEAR + 5)
    if not (np.isfinite(x[fit]).all() and np.isfinite(x[post]).all()):
        return None
    c = np.polyfit(YEARS[fit], x[fit], 1)
    sd = (x[fit] - np.polyval(c, YEARS[fit])).std(ddof=2)
    jump = float(np.mean(x[post] - np.polyval(c, YEARS[post])))
    ts, d = [], []
    for t0 in range(YEARS[0] + 4, YEARS[-1] - 4):
        b = x[(YEARS >= t0 - 4) & (YEARS <= t0)].mean()
        a = x[(YEARS >= t0 + 1) & (YEARS <= t0 + 5)].mean()
        ts.append(t0)
        d.append(a - b)
    ts, d = np.array(ts), np.array(d)
    d14 = float(d[ts == SPLICE_YEAR][0])
    pool = d[np.abs(ts - SPLICE_YEAR) > 4]
    return {"jump_vs_1995_2014_trend_pp": jump,
            "jump_in_residual_sd": jump / sd if sd > 0 else np.nan,
            "block_diff_2014_pp": d14, "block_diff_pool_median_pp": float(np.median(pool)),
            "block_diff_percentile": float((pool < d14).mean() * 100.0)}


# -----------------------------------------------------------------------------
# ANALYSIS OF ONE SEVERITY x REGION
# -----------------------------------------------------------------------------
def analyse(sev, region, raw_models, raw_obs, rng, valid_Mkm2):
    mult = STD_MULT[sev]
    ref_mean = np.nanmean(raw_obs[HIST_MASK]) if raw_obs is not None else np.nanmean(
        np.mean([v[HIST_MASK] for v in raw_models.values()], axis=0))
    corr = {mode: bias_correct(raw_models, raw_obs, mode) for mode in BIAS_MODES}
    sens, main, panels, splice = [], [], {}, []

    for mode in BIAS_MODES:
        for dtype in EMERGENCE_TYPES:
            ser = {m: transform(v, dtype, ref_mean) for m, v in corr[mode].items()}
            mean, lo, hi = ensemble(ser)
            obs_t = transform(raw_obs, dtype, ref_mean)
            for src in SIGMA_SOURCES:
                mu, sd = baseline_stats(ser, mean, obs_t, src)
                thr = mu + mult * sd if np.isfinite(sd) else np.nan
                em = first_sustained(mean, thr)
                per = [first_sustained(v, thr) for v in ser.values()]
                det = [p for p in per if p is not None]
                sens.append({"severity": sev, "region": region, "data_type": dtype,
                             "bias_mode": mode, "sigma_source": src, "baseline_mean": mu,
                             "baseline_sd": sd, "threshold": thr, "emergence": em,
                             "model_emergence_min": min(det) if det else None,
                             "model_emergence_max": max(det) if det else None,
                             "n_models_detected": len(det)})

    # ---- main settings: panels for all three data types -------------------------------
    ens_raw_main = ensemble(corr[MAIN_BIAS])[0]
    for dtype in ALL_TYPES:
        ser = {m: transform(v, dtype, ref_mean) for m, v in corr[MAIN_BIAS].items()}
        mean, lo, hi = ensemble(ser)
        obs_t = transform(raw_obs, dtype, ref_mean)
        p = {"mean": mean, "lo": np.maximum(lo, 0) if dtype != "anomalies" else lo, "hi": hi,
             "obs": obs_t, "valid_Mkm2": valid_Mkm2, "thr": None, "thr_cal": None,
             "emerg": None, "emerg_lo": None, "emerg_hi": None, "emerg_cal": None, "fpr": None}
        if dtype in EMERGENCE_TYPES:
            mu, sd = baseline_stats(ser, mean, obs_t, MAIN_SIGMA)
            thr = mu + mult * sd
            em = first_sustained(mean, thr)
            per = [first_sustained(v, thr) for v in ser.values()]
            det = [q for q in per if q is not None]
            above_after = (float(np.mean(mean[YEARS >= em] > thr)) if em is not None else np.nan)
            row = {"severity": sev, "region": region, "data_type": dtype, "bias_mode": MAIN_BIAS,
                   "sigma_source": MAIN_SIGMA, "baseline_mean": mu, "baseline_sd": sd,
                   "threshold": thr, "emergence": em,
                   "model_emergence_min": min(det) if det else None,
                   "model_emergence_max": max(det) if det else None,
                   "n_models_detected": len(det), "share_of_years_above_after_emergence": above_after}
            st = surrogate_test(ens_raw_main, dtype, mult, thr, rng)
            if st is not None:
                row.update(st)
                row["threshold_calibrated_sigma_equiv"] = (st["thr_calibrated"] - mu) / sd
                p["thr_cal"] = st["thr_calibrated"]
                p["emerg_cal"] = first_sustained(mean, st["thr_calibrated"])
                row["emergence_calibrated"] = p["emerg_cal"]
                p["fpr"] = st["fpr_procedure"]
                p["fpr_real"] = st["fpr_real_threshold"]
            ext = reference_extension(raw_obs, ens_raw_main, dtype, thr, ref_mean)
            if ext:
                row.update(ext)
            main.append(row)
            p.update(thr=thr, emerg=em, emerg_lo=min(det) if det else None,
                     emerg_hi=max(det) if det else None)
        panels[dtype] = p

    # ---- splice check on raw corrected series ------------------------------------------
    for name, x in [("Ensemble", ens_raw_main)] + [(m, v) for m, v in corr[MAIN_BIAS].items()]:
        sc = splice_check(x)
        if sc:
            splice.append({"severity": sev, "region": region, "series": name, **sc})
    return sens, main, panels, splice


# -----------------------------------------------------------------------------
# DEPENDENCE EVALUATION  (item 62)
# -----------------------------------------------------------------------------
def dep_counts(source, sev):
    hot_p, dry_p = flag_file(source, sev, "temp"), flag_file(source, sev, "spi")
    if hot_p is None or dry_p is None or not Path(hot_p).exists() or not Path(dry_p).exists():
        return None
    key = hashlib.md5(f"{hot_p}|{dry_p}|{_stamp(hot_p)}|{_stamp(dry_p)}|{CACHE_VERSION}|{DEP_PERIODS}"
                      .encode()).hexdigest()[:10]
    cp = CACHE_DIR / f"dep_{source}_{sev}_{key}.pkl"
    if USE_CACHE and cp.exists():
        with open(cp, "rb") as f:
            print(f"  [CACHE] dependence {DISPLAY.get(source, source)} {sev}")
            return pickle.load(f)
    print(f"  Dependence {DISPLAY.get(source, source)} {sev}")
    dh, dd = open_flags(hot_p, VAR_NAMES["temp"]), open_flags(dry_p, SEV_KEYS[sev]["spi"])
    lc, oc = find_latlon(dh)
    lc2, oc2 = find_latlon(dd)
    if not (dh[lc].shape == dd[lc2].shape and np.allclose(dh[lc].values, dd[lc2].values)
            and np.allclose(dh[oc].values, dd[oc2].values)):
        raise ValueError(f"{source}: hot and dry files are on different grids")
    ld, od = dh[lc].dims[0], dh[oc].dims[0]
    ld2, od2 = dd[lc2].dims[0], dd[oc2].dims[0]
    yh, yd = np.asarray(dh.indexes["time"].year), np.asarray(dd.indexes["time"].year)
    shape = (dh[lc].size, dh[oc].size)
    out = {"lat": dh[lc].values.astype(float), "lon": dh[oc].values.astype(float)}
    for p, (y0, y1) in DEP_PERIODS.items():
        nh, nd, nhd, nv = (np.zeros(shape) for _ in range(4))
        for y in range(y0, y1 + 1):
            ih, idd = np.where(yh == y)[0], np.where(yd == y)[0]
            if ih.size == 0 or ih.size != idd.size:
                continue
            ch = to_tll(dh.isel(time=ih), ld, od).values
            cd = to_tll(dd.isel(time=idd), ld2, od2).values
            ok = np.isfinite(ch) & np.isfinite(cd)
            h, d = (ch == 1) & ok, (cd == 1) & ok
            nh += h.sum(0)
            nd += d.sum(0)
            nhd += (h & d).sum(0)
            nv += ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[p] = tuple(np.where(nv > 0, a / nv, np.nan) for a in (nh, nd, nhd))
    if USE_CACHE:
        with open(cp, "wb") as f:
            pickle.dump(out, f)
    return out


def phi_lift(P):
    ph, pdr, phd = P
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(ph * (1 - ph) * pdr * (1 - pdr))
        phi = np.where(den > 0, (phd - ph * pdr) / den, np.nan)
        lift = np.where(ph * pdr > 0, phd / (ph * pdr), np.nan)
    return phi, lift


def dep_metrics(source, sev, D, ref):
    lat, lon = D["lat"], D["lon"]
    w2 = band_area_km2(lat, spacing(lat), spacing(lon))[:, None] * np.ones((1, lon.size))
    masks = coverage_masks(lat, lon)
    rows = []
    for p in DEP_PERIODS:
        ph, pdr, phd = D[p]
        phi, _ = phi_lift(D[p])
        phi_ref = phi_lift(ref[p])[0] if (ref is not None and ref.get(p) is not None) else None
        for r, m in masks.items():
            ok = m & np.isfinite(ph) & np.isfinite(pdr) & np.isfinite(phd)
            if not ok.any():
                continue
            w = w2[ok]
            W = w.sum()
            joint = (w * phd[ok]).sum() / W
            indep = (w * (ph * pdr)[ok]).sum() / W
            okp = ok & np.isfinite(phi)
            row = {"severity": sev, "source": source, "region": r, "period": p,
                   "p_hot": (w * ph[ok]).sum() / W, "p_dry": (w * pdr[ok]).sum() / W,
                   "p_joint": joint, "p_independent": indep, "lift": joint / indep,
                   "phi_mean": (w2[okp] * phi[okp]).sum() / w2[okp].sum()}
            if phi_ref is not None and source != OBS_NAME:
                s = okp & np.isfinite(phi_ref)
                if s.sum() > 2:
                    ws = w2[s]
                    row["phi_pattern_r_vs_reference"] = float(np.corrcoef(phi[s], phi_ref[s])[0, 1])
                    row["phi_rmse_vs_reference"] = float(np.sqrt((ws * (phi[s] - phi_ref[s]) ** 2).sum() / ws.sum()))
            rows.append(row)
    return rows


def plot_dependence(sev, deps, dep_df):
    srcs = [s for s in [OBS_NAME] + MODELS if deps.get(s) is not None]
    if not srcs:
        return
    sl = sev.replace(".", "")
    # maps
    lat, lon = deps[srcs[0]]["lat"], deps[srcs[0]]["lon"]
    lonn = (lon + 180.0) % 360.0 - 180.0
    ox, oy = np.argsort(lonn), np.argsort(lat)
    ref_phi = phi_lift(deps[OBS_NAME]["hist"])[0] if deps.get(OBS_NAME) is not None else None
    fig, axes = plt.subplots(len(srcs), 2, figsize=(15, 3.1 * len(srcs)), squeeze=False)
    for i, s in enumerate(srcs):
        phi = phi_lift(deps[s]["hist"])[0]
        ax = axes[i, 0]
        im = ax.pcolormesh(lonn[ox], lat[oy], phi[np.ix_(oy, ox)], cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                           shading="nearest")
        ax.set_title(f"{DISPLAY[s]}: phi(hot, dry), {DEP_PERIODS['hist'][0]}-{DEP_PERIODS['hist'][1]}", fontsize=11)
        ax2 = axes[i, 1]
        if s != OBS_NAME and ref_phi is not None:
            im2 = ax2.pcolormesh(lonn[ox], lat[oy], (phi - ref_phi)[np.ix_(oy, ox)], cmap="PuOr",
                                 vmin=-0.3, vmax=0.3, shading="nearest")
            ax2.set_title(f"{DISPLAY[s]} minus reference", fontsize=11)
        else:
            im2 = ax2.pcolormesh(lonn[ox], lat[oy], deps[s]["hist"][2][np.ix_(oy, ox)], cmap="viridis",
                                 vmin=0, vmax=0.12, shading="nearest")
            ax2.set_title(f"{DISPLAY[s]}: P(hot and dry), months", fontsize=11)
        for a in (ax, ax2):
            a.set_xlim(-180, 180)
            a.set_ylim(-60, 85)
            a.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"dependence_maps_sev{sl}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # lift by region
    d = dep_df[dep_df["severity"] == sev]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), sharey=True)
    regs = PANEL_REGIONS
    wid = 0.8 / len(srcs)
    for ax, p in zip(axes, ("hist", "fut")):
        for i, s in enumerate(srcs):
            v = [d[(d["source"] == s) & (d["region"] == r) & (d["period"] == p)]["lift"].values for r in regs]
            v = [x[0] if len(x) else np.nan for x in v]
            col = "black" if s == OBS_NAME else MODEL_COLORS[s]
            ax.bar(np.arange(len(regs)) + i * wid - 0.4 + wid / 2, v, wid, color=col,
                   alpha=0.95 if s == OBS_NAME else 0.7, label=DISPLAY[s])
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xticks(np.arange(len(regs)))
        ax.set_xticklabels(regs, rotation=25, ha="right")
        ax.set_title(f"{'Historical' if p == 'hist' else 'Future'} {DEP_PERIODS[p][0]}-{DEP_PERIODS[p][1]}")
        ax.grid(axis="y", alpha=0.3, ls=":")
    axes[0].set_ylabel("Lift = P(hot and dry) / [P(hot) P(dry)]")
    if OBS_NAME in srcs:
        ref_h = [d[(d["source"] == OBS_NAME) & (d["region"] == r) & (d["period"] == "hist")]["lift"].values for r in regs]
        for k, v in enumerate(ref_h):
            if len(v):
                axes[1].plot([k - 0.42, k + 0.42], [v[0], v[0]], color="black", lw=1.6)
        axes[1].plot([], [], color="black", lw=1.6, label="Reference, historical")
        axes[1].legend(fontsize=9, loc="upper right")
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=len(srcs), fontsize=10, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Dependence of hot and dry months, {SEV_NAME[sev]} (1 = independent)", fontsize=12)
    fig.savefig(OUTPUT_DIR / f"dependence_lift_sev{sl}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURES: coverage, exceedance
# -----------------------------------------------------------------------------
def _fmt_range(lo, hi):
    if lo is None:
        return "n.d."
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def draw_panel(ax, p, dtype, show_em=True, ylim=None, small=False):
    fs = 9 if small else 11
    if p is None or p["mean"] is None:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        return
    ok = np.isfinite(p["lo"]) & np.isfinite(p["hi"])
    ax.fill_between(YEARS[ok], p["lo"][ok], p["hi"][ok], color=ENS_COLOR, alpha=0.2, zorder=1)
    ax.plot(YEARS, p["mean"], color=ENS_COLOR, lw=2.2 if small else 2.6, zorder=9)
    if p["obs"] is not None:
        h = YEARS <= HIST[1]
        ax.plot(YEARS[h], p["obs"][h], color="black", lw=2.0 if small else 2.4, zorder=10)
        e = YEARS >= HIST[1]
        if np.isfinite(p["obs"][e]).sum() > 1:
            ax.plot(YEARS[e], p["obs"][e], color="black", lw=1.3, ls="--", zorder=10)
    if dtype == "anomalies":
        ax.axhline(0, color="black", lw=0.8, alpha=0.3)
    elif show_em and p["thr"] is not None and np.isfinite(p["thr"]):
        ax.axhline(p["thr"], color="red", ls="--", lw=1.4, alpha=0.75, zorder=8)
        if p["thr_cal"] is not None and np.isfinite(p["thr_cal"]):
            ax.axhline(p["thr_cal"], color="#E69F00", ls="-.", lw=1.4, alpha=0.9, zorder=8)
        if p["emerg"] is not None:
            i = int(np.where(YEARS == p["emerg"])[0][0])
            ax.plot(p["emerg"], p["mean"][i], "o", ms=7, mfc=ENS_COLOR, mec="black", mew=1.4, zorder=11)
            ax.annotate(f"{p['emerg']}", xy=(p["emerg"], p["mean"][i]), xytext=(7, 8),
                        textcoords="offset points", fontsize=fs - 1, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc=ENS_COLOR, ec="black", alpha=0.85), zorder=12)
        lines = [f"Individual models: {_fmt_range(p['emerg_lo'], p['emerg_hi'])}"]
        if p["thr_cal"] is not None:
            lines.append(f"Emergence at 5 % FPR level: {p['emerg_cal'] if p['emerg_cal'] else 'n.d.'}")
        if p["fpr"] is not None:
            lines.append(f"Rule false-alarm rate (no trend): {p['fpr']:.0%}")
        if p.get("fpr_real") is not None:
            lines.append(f"Threshold exceeded by chance: {p['fpr_real']:.0%}")
        ax.text(0.98, 0.03, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", ###
                fontsize=fs - 2, color="0.25", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8))
    ax.axvline(SPLICE_YEAR, color="gray", ls=":", lw=1.4, alpha=0.6)
    ax.set_xlim(1980, 2050)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, ls="--", lw=0.5)
    ax.tick_params(labelsize=fs - 1)


def shared_ylim(panels, dtype):
    vals = []
    for p in panels:
        if p is None or p["mean"] is None:
            continue
        for k in ("mean", "lo", "hi", "obs"):
            if p[k] is not None:
                vals.append(np.asarray(p[k], float))
        for k in ("thr", "thr_cal"):
            if p[k] is not None and np.isfinite(p[k]):
                vals.append(np.array([p[k]]))
    if not vals:
        return None
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    pad = (v.max() - v.min()) * 0.06
    return v.min() - pad, v.max() + pad


def legend_handles(bias_mode):
    return [Line2D([0], [0], color="black", lw=2.4, label=f"Reference ({HIST[0]}-{HIST[1]})"),
          #  Line2D([0], [0], color="black", lw=1.3, ls="--", label="Reference after 2014 (not used for correction)"),
            Line2D([0], [0], color=ENS_COLOR, lw=2.6, label=f"Ensemble mean"),
            Patch(facecolor=ENS_COLOR, alpha=0.2, label="Model spread (min-max)"),
            Line2D([0], [0], color="red", ls="--", lw=1.4, label="Threshold: baseline mean + kσ"),
            Line2D([0], [0], color="#E69F00", ls="-.", lw=1.4, label="Threshold at 5 % false-positive rate"),
            Line2D([0], [0], color="gray", ls=":", lw=1.4, label="Historical | future boundary (2014)"),
            Line2D([0], [0], marker="o", color=ENS_COLOR, mec="black", ls="none", ms=7,
                   label="Emergence year (ensemble mean)")]


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {stem}.png/.pdf")


def plot_regional(results, sev, dtype):
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.12)
    panels = [results.get((sev, r, dtype)) for r in PANEL_REGIONS]
    ylim = shared_ylim(panels, dtype)
    for i, r in enumerate(PANEL_REGIONS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        draw_panel(ax, panels[i], dtype, show_em=(dtype != "anomalies"), ylim=ylim)
        vm = panels[i]["valid_Mkm2"] if panels[i] else np.nan
        ax.set_title(f"{r}  ({vm:.1f} Mkm²)" if r != GLOBAL else f"Global, all valid land ({vm:.1f} Mkm²)",
                     fontsize=12, fontweight="bold", pad=6)
        if i % 2 == 0:
            ax.set_ylabel("Coverage anomaly (pp)" if dtype == "anomalies" else "Area coverage (%)", fontsize=11)
        else:
            ax.set_yticklabels([])
        if i // 2 == 3 or i == len(PANEL_REGIONS) - 2:      # last panel of each column
            ax.set_xlabel("Year", fontsize=11)
        else:
            ax.tick_params(axis="x", labelbottom=False)
    axl = fig.add_subplot(gs[3, 1])
    axl.axis("off")
    axl.legend(handles=legend_handles(MAIN_BIAS), loc="upper left", fontsize=11, framealpha=0.95) ###########
    fig.suptitle(f"{SEV_NAME[sev]}", fontsize=14, fontweight="bold", y=0.905) ##: {TYPE_LABEL[dtype]} coverage
    save(fig, f"coverage_regional_{dtype}_sev{sev.replace('.', '')}")


def plot_global_stack(results, dtype):
    fig, axes = plt.subplots(len(SEVERITIES), 1, figsize=(14, 4.2 * len(SEVERITIES)))
    axes = np.atleast_1d(axes)
    for ax, sev in zip(axes, SEVERITIES):
        p = results.get((sev, GLOBAL, dtype))
        draw_panel(ax, p, dtype, show_em=(dtype != "anomalies"))
        ax.set_title(SEV_NAME[sev], fontsize=15, fontweight="bold", color=SEV_COLOR[sev])
        ax.set_ylabel("Coverage anomaly (pp)" if dtype == "anomalies" else "Area coverage (%)", fontsize=13)
    axes[-1].set_xlabel("Year", fontsize=13)
    axes[0].legend(handles=legend_handles(MAIN_BIAS), loc="upper left", fontsize=9, ncol=2, framealpha=0.95)
    fig.tight_layout()
    save(fig, f"coverage_global_{dtype}")


def plot_global_combined(results):
    fig, axes = plt.subplots(len(SEVERITIES), 3, figsize=(20, 4 * len(SEVERITIES)), squeeze=False)
    for r, sev in enumerate(SEVERITIES):
        for c, dtype in enumerate(ALL_TYPES):
            ax = axes[r, c]
            draw_panel(ax, results.get((sev, GLOBAL, dtype)), dtype, show_em=(dtype != "anomalies"), small=True)
            if r == 0:
                ax.set_title(TYPE_LABEL[dtype], fontsize=15, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{SEV_NAME[sev]}\n" + ("Anomaly (pp)" if False else "Coverage (%)"),
                              fontsize=12, color=SEV_COLOR[sev], fontweight="bold")
            if r < len(SEVERITIES) - 1:
                ax.tick_params(axis="x", labelbottom=False)
    fig.legend(handles=legend_handles(MAIN_BIAS), loc="upper left", bbox_to_anchor=(1.0, 0.5), fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    save(fig, "coverage_global_all_types")


def plot_emergence_sensitivity(sens_df):
    d = sens_df[sens_df["data_type"] == "smoothed"]
    sevs = [s for s in SEVERITIES if s in d["severity"].unique()]
    if not sevs:
        return
    fig, axes = plt.subplots(len(sevs), 1, figsize=(15, 4.2 * len(sevs)), sharex=True, squeeze=False)
    mk = {"ensemble": "o", "models": "s", "reference": "^", "ensemble_detrended": "D"}
    col = {"additive": "#1f77b4", "multiplicative": "#d62728", "none": "#7f7f7f"}
    regs = PANEL_REGIONS
    for ax, sev in zip(axes[:, 0], sevs):
        for bm in BIAS_MODES:
            for ss in SIGMA_SOURCES:
                s = d[(d["severity"] == sev) & (d["bias_mode"] == bm) & (d["sigma_source"] == ss)]
                xs = np.arange(len(regs)) + (BIAS_MODES.index(bm) - 1) * 0.22 + (SIGMA_SOURCES.index(ss) - 1.5) * 0.045
                ys = [(s[s["region"] == r]["emergence"].values[0] if (s["region"] == r).any() else None) for r in regs]
                ys = [np.nan if (y is None or pd.isna(y)) else y for y in ys]
                nd = [i for i, y in enumerate(ys) if np.isnan(y)]
                ax.scatter(xs, [2053 if np.isnan(y) else y for y in ys], marker=mk[ss], c=col[bm], s=34,
                           edgecolors="black", linewidths=0.4)
        ax.set_ylim(1978, 2056)
        ax.axhline(2052, color="gray", ls=":", lw=0.8)
        ax.text(len(regs) - 0.45, 2053, "not detected", fontsize=8, color="gray", ha="right")
        ax.set_ylabel("Emergence year")
        ax.set_title(f"{SEV_NAME[sev]} (5-year smoothed)", fontsize=12, color=SEV_COLOR[sev], fontweight="bold")
        ax.grid(axis="y", alpha=0.3, ls=":")
    axes[-1, 0].set_xticks(np.arange(len(regs)))
    axes[-1, 0].set_xticklabels(regs, rotation=20, ha="right")
    h = ([Line2D([0], [0], marker="o", color=col[b], ls="none", label=f"bias: {b}") for b in BIAS_MODES]
         + [Line2D([0], [0], marker=mk[s], color="white", mec="black", ls="none", label=f"sigma: {s}")
            for s in SIGMA_SOURCES])
    axes[0, 0].legend(handles=h, ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "emergence_sensitivity")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CHDE COVERAGE AND EXCEEDANCE ANALYSIS  v2")
    print(f"Regions    : chde_regions.py, hash {regions_hash()}")
    print(f"Baseline   : {HIST[0]}-{HIST[1]};  reference read to {OBS_END}")
    print(f"Rule       : mean + k sigma, {SUSTAIN_K} consecutive years; surrogates: {N_SURR}")
    print(f"Severities : {', '.join(SEVERITIES)}")
    print("=" * 70)
    check_regions_disjoint()
    (OUTPUT_DIR / "region_hash.txt").write_text(regions_hash() + "\n")

    rng = np.random.default_rng(RNG_SEED)
    results, cov_rows, bias_all, recon, sens_all, main_all, splice_all, ens_rows = {}, [], [], [], [], [], [], []

    for sev in SEVERITIES:
        print(f"\n{'-' * 60}\n  Severity {sev} sigma\n{'-' * 60}")
        covs = {}
        for src in [OBS_NAME] + MODELS:
            p = flag_file(src, sev, "compound")
            if src == OBS_NAME and (p is None or not Path(p).exists()):
                print("  [WARN] reference compound file not found; running without reference.\n"
                      "         Fill OBS_FILE_OVERRIDES or check BASE_PATH/OBS/6months.")
                covs[src] = None
                continue
            c = compute_coverage(src, sev)
            covs[src] = c
            if c is None:
                print(f"  [MISS] {src}")
                continue
            rc = reconcile(src, sev, c)
            if rc:
                recon.append(rc)
                flag = "" if rc["max_abs_diff_pp"] < 0.05 else "   <-- CHECK"
                print(f"    reconcile {DISPLAY.get(src, src):<10} max |diff| vs CDO global series: "
                      f"{rc['max_abs_diff_pp']:.4f} pp{flag}")
            if src == OBS_NAME:
                for r, v in c["cov"].items():
                    v = v.copy()
                    v[YEARS > OBS_END] = np.nan
                    c["cov"][r] = v
            for r, v in c["cov"].items():
                cov_rows.append(pd.DataFrame({"severity": sev, "source": src, "region": r,
                                              "year": YEARS, "coverage_pct": v}))
        models_ok = {m: covs[m] for m in MODELS if covs.get(m) is not None}
        if len(models_ok) < 2:
            print("  fewer than 2 models available; skipping severity")
            continue
        obs_c = covs.get(OBS_NAME)

        for region in ALL_COV:
            raw_models = {m: c["cov"][region] for m, c in models_ok.items() if region in c["cov"]}
            if len(raw_models) < 2:
                continue
            raw_obs = obs_c["cov"][region] if (obs_c and region in obs_c["cov"]) else None
            vk = float(np.mean([c["valid_Mkm2"][region] for c in models_ok.values() if region in c["valid_Mkm2"]]))
            bias_all += bias_rows(raw_models, raw_obs, sev, region)
            sens, main, panels, splice = analyse(sev, region, raw_models, raw_obs, rng, vk)
            sens_all += sens
            main_all += main
            splice_all += splice
            for dtype, p in panels.items():
                results[(sev, region, dtype)] = p
                ens_rows.append(pd.DataFrame({
                    "severity": sev, "region": region, "data_type": dtype, "year": YEARS,
                    "ensemble_mean": p["mean"], "spread_min": p["lo"], "spread_max": p["hi"],
                    "reference": p["obs"] if p["obs"] is not None else np.nan}))
        mm = [r for r in main_all if r["severity"] == sev and r["data_type"] == "smoothed"
              and r["region"] in (GLOBAL, "Africa")]
        for r in mm:
            print(f"    {r['region']:<8} smoothed: emergence {r['emergence']}  models "
                  f"{r['model_emergence_min']}-{r['model_emergence_max']}  false-alarm rate "
                  f"{r.get('fpr_procedure', float('nan')):.0%}  5%-FPR emergence {r.get('emergence_calibrated')}")

    # ---------------- tables -----------------------------------------------------------
    if cov_rows:
        pd.concat(cov_rows).to_csv(OUTPUT_DIR / "coverage_annual.csv", index=False)
    if ens_rows:
        pd.concat(ens_rows).to_csv(OUTPUT_DIR / "coverage_ensemble_main.csv", index=False)
    pd.DataFrame(bias_all).to_csv(OUTPUT_DIR / "bias_table.csv", index=False)
    pd.DataFrame(recon).to_csv(OUTPUT_DIR / "reconciliation.csv", index=False)
    sens_df = pd.DataFrame(sens_all)
    sens_df.to_csv(OUTPUT_DIR / "emergence_sensitivity.csv", index=False)
    pd.DataFrame(main_all).to_csv(OUTPUT_DIR / "emergence_main.csv", index=False)
    pd.DataFrame(splice_all).to_csv(OUTPUT_DIR / "splice_check.csv", index=False)

    # ---------------- figures ----------------------------------------------------------
    for sev in SEVERITIES:
        for dtype in ALL_TYPES:
            if any(k[0] == sev and k[2] == dtype for k in results):
                plot_regional(results, sev, dtype)
    for dtype in ALL_TYPES:
        if any(k[1] == GLOBAL and k[2] == dtype for k in results):
            plot_global_stack(results, dtype)
    if any(k[1] == GLOBAL for k in results):
        plot_global_combined(results)
    if not sens_df.empty:
        plot_emergence_sensitivity(sens_df)

    # ---------------- dependence evaluation (item 62) ----------------------------------
    if RUN_DEPENDENCE:
        print("\nDependence evaluation (hot and dry monthly flags)")
        dep_rows = []
        deps_by_sev = {}
        for sev in SEVERITIES:
            deps = {}
            for src in [OBS_NAME] + MODELS:
                try:
                    deps[src] = dep_counts(src, sev)
                except Exception as e:
                    print(f"  [ERR ] {src} {sev}: {e}")
                    deps[src] = None
            if deps.get(OBS_NAME) is None:
                print(f"  [WARN] no reference hot/dry files for {sev}: models are shown without comparison")
            for src, D in deps.items():
                if D is not None:
                    dep_rows += dep_metrics(src, sev, D, deps.get(OBS_NAME))
            deps_by_sev[sev] = deps
        dep_df = pd.DataFrame(dep_rows)
        dep_df.to_csv(OUTPUT_DIR / "dependence_summary.csv", index=False)
        for sev, deps in deps_by_sev.items():
            plot_dependence(sev, deps, dep_df)

    print("\n" + "=" * 70)
    print(f"DONE. Outputs in: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
