"""
chde_regions.py
===============
Single source of truth for the six CHDE regions and the grid helpers that go with them.
Imported by chde_attribution_v4.py and chde_coverage_v2.py so that probabilities, coverage
and exceedance are all computed on identical, disjoint regions.

Edit REGIONS here and nothing else; both scripts pick the change up (their caches are keyed
on regions_hash()).
"""
import hashlib

import numpy as np

R_EARTH_KM = 6371.0
GRID_RES_DEFAULT = 0.5
LAT_NAMES = ("lat", "latitude", "rlat", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "rlon", "nav_lon", "x")

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


def find_latlon(da):
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


def spacing(v, default=GRID_RES_DEFAULT):
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




def regions_hash():
    """Short fingerprint of the region definitions (printed and stored with every result)."""
    return hashlib.md5(repr(REGIONS).encode()).hexdigest()[:8]
