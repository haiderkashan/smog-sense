"""smogsense.features.static_covariates — Static station descriptors.

Extracts spatial descriptors for target and source stations:
- Latitude and Longitude.
- Elevation (meters above sea level).
- Distance to domain centroid (km) via WGS84 geodesic.
- Regulatory reference indicator (`is_reference`).

Specification:
- docs/ml-architecture.md §2 ('Feature engineering (physics-informed)')
- configs/features.yaml ('static')
- configs/domains.yaml
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyproj

# Default domain centroids: lat, lon
DOMAIN_CENTROIDS: dict[str, tuple[float, float]] = {
    "lahore": (31.5204, 74.3587),
    "delhi": (28.6139, 77.2090),
    "amritsar": (31.6340, 74.8723),
    "ludhiana": (30.9010, 75.8573),
}

# Default typical elevations in meters for Indo-Gangetic Plain
DEFAULT_ELEVATIONS_M: dict[str, float] = {
    "lahore": 217.0,
    "delhi": 216.0,
    "amritsar": 234.0,
    "ludhiana": 244.0,
}


def compute_distance_to_centroid(
    lat: float,
    lon: float,
    domain: str = "lahore",
    centroid: tuple[float, float] | None = None,
) -> float:
    """Compute WGS84 geodesic distance (km) from station coordinates to the domain centroid."""
    if centroid is None:
        dom_clean = domain.lower().strip()
        c_lat, c_lon = DOMAIN_CENTROIDS.get(dom_clean, DOMAIN_CENTROIDS["lahore"])
    else:
        c_lat, c_lon = centroid

    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon, lat, c_lon, c_lat)
    return float(dist_m / 1000.0)


def extract_station_static_covariates(
    station_row: pd.Series | dict[str, Any],
    domain: str = "lahore",
) -> dict[str, float]:
    """Extract model-ready static covariates for a single station.

    Returns dictionary with keys:
    - `f_stat_lat`
    - `f_stat_lon`
    - `f_stat_elevation_m`
    - `f_stat_dist_city_center_km`
    - `f_stat_is_reference`
    """
    dom_clean = domain.lower().strip()
    lat = float(station_row.get("lat", station_row.get("latitude", DOMAIN_CENTROIDS.get(dom_clean, (31.52, 74.35))[0])))
    lon = float(station_row.get("lon", station_row.get("longitude", DOMAIN_CENTROIDS.get(dom_clean, (31.52, 74.35))[1])))

    # Elevation
    raw_elev = station_row.get("elevation", station_row.get("elevation_m", None))
    if raw_elev is not None and not pd.isna(raw_elev):
        elev_m = float(raw_elev)
    else:
        elev_m = DEFAULT_ELEVATIONS_M.get(dom_clean, 217.0)

    # Reference flag
    is_ref_raw = station_row.get("is_reference", station_row.get("is_monitor", False))
    is_ref = 1.0 if bool(is_ref_raw) and not pd.isna(is_ref_raw) else 0.0

    dist_km = compute_distance_to_centroid(lat, lon, domain=dom_clean)

    return {
        "f_stat_lat": float(lat),
        "f_stat_lon": float(lon),
        "f_stat_elevation_m": float(elev_m),
        "f_stat_dist_city_center_km": float(dist_km),
        "f_stat_is_reference": float(is_ref),
    }
