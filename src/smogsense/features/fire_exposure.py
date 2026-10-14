"""smogsense.features.fire_exposure — Target-centric, wind-aligned fire exposure kernel.

Implements the single-angle physics-informed fire exposure model:
1. Target-centric geometry:
   - theta_i: azimuth Target -> Fire (degrees clockwise from North) via WGS84 geodesic.
   - beta_i = (theta_i + 180°) mod 360°: transport bearing Fire -> Target.
   - phi(t) = atan2(u(t), v(t)) mod 360°: direction air moves toward (clockwise from North).
   - a_i(t) = max(0.0, cos(phi(t) - beta_i)): wind alignment factor.
2. Advective decay kernel:
   - Target wind speed w(t) = max(sqrt(u^2 + v^2), w_min) with w_min = 0.5 m/s.
   - Advective travel time Theta_i(t) = d_i / (3.6 * w(t)) [hours].
   - Age decay with tau_decay = 24h, tau_age = 48h:
     K_i(t) = a_i(t) * exp(-Theta_i(t) / tau_decay) * exp(- (min(t, T) - tau_i)+ / tau_age).
3. Platform-normalised exposure:
   - omega_i = 1 / n_plat(tau_i).
   - E(t) = sum_{i: tau_i <= T} omega_i * frp_i * K_i(t).
   - Block horizon exposure: E_h = (1/24) * sum_{t in [T+h-24, T+h)} E(t) for h in {24, 48, 72}.
4. Geodesic rings (0-50, 50-150, 150-300, 300-600 km) and 8-sector compass breakdowns.

Specification:
- docs/ml-architecture.md §3 ('Fire transport exposure')
- docs/data-engineering.md §9 ('NASA FIRMS fire ingestion')
- configs/features.yaml ('fire')
- data/schemas/fire_exposure.schema.yaml
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pyproj

from smogsense.utils.io import validate_frame

# Configuration defaults
DEFAULT_TAU_DECAY_H = 24.0
DEFAULT_TAU_AGE_H = 48.0
DEFAULT_WIND_MIN_MS = 0.5
DEFAULT_MAX_RADIUS_KM = 600.0

SECTOR_NAMES: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
RING_NAMES: tuple[str, ...] = ("r0_50", "r50_150", "r150_300", "r300_600")
RING_BOUNDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 50.0, "r0_50"),
    (50.0, 150.0, "r50_150"),
    (150.0, 300.0, "r150_300"),
    (300.0, 600.0, "r300_600"),
)

DOMAIN_CENTROIDS: dict[str, tuple[float, float]] = {
    "lahore": (31.5204, 74.3587),
    "delhi": (28.6139, 77.2090),
    "amritsar": (31.6340, 74.8723),
    "ludhiana": (30.9010, 75.8573),
}


def _load_fire_exposure_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/fire_exposure.schema.yaml"),
        Path("/app/data/schemas/fire_exposure.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/fire_exposure.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(str(p))
    raise FileNotFoundError(f"fire_exposure.schema.yaml not found in candidates: {candidates}")


def compute_single_angle_geometry(
    target_lat: float,
    target_lon: float,
    fire_lats: Sequence[float] | np.ndarray | pd.Series | float,
    fire_lons: Sequence[float] | np.ndarray | pd.Series | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute target-centric single-angle geometry for fire detections.

    Returns:
    - theta_deg: Target -> Fire azimuth (degrees clockwise from North, in [0, 360)).
    - beta_deg: Transport bearing Fire -> Target ((theta + 180) mod 360).
    - dist_km: Geodesic distance in kilometers on WGS84 ellipsoid.
    - sector_indices: Compass sector index 0..7 (N, NE, E, SE, S, SW, W, NW).
    """
    lats = np.atleast_1d(np.asarray(fire_lats, dtype=np.float64))
    lons = np.atleast_1d(np.asarray(fire_lons, dtype=np.float64))

    if len(lats) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, np.array([], dtype=np.int32)

    geod = pyproj.Geod(ellps="WGS84")
    target_lons = np.full_like(lons, target_lon)
    target_lats = np.full_like(lats, target_lat)

    fwd_az, _, dist_m = geod.inv(target_lons, target_lats, lons, lats)
    theta_deg = np.mod(fwd_az, 360.0)
    beta_deg = np.mod(theta_deg + 180.0, 360.0)
    dist_km = dist_m / 1000.0

    # Sector indices: 45° sectors centered on 0, 45, 90, 135, 180, 225, 270, 315
    sector_indices = np.floor((theta_deg + 22.5) / 45.0).astype(int) % 8

    return theta_deg, beta_deg, dist_km, sector_indices


def compute_wind_direction_and_speed(
    u10: float | np.ndarray,
    v10: float | np.ndarray,
    w_min: float = DEFAULT_WIND_MIN_MS,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Compute wind transport direction phi (degrees clockwise from North) and clamped speed.

    phi(t) = atan2(u, v) mod 360: direction the air moves TOWARD (u=East, v=North).
    w(t) = max(sqrt(u^2 + v^2), w_min).
    """
    is_scalar = np.isscalar(u10) and np.isscalar(v10)
    u_arr = np.asarray(u10, dtype=np.float64)
    v_arr = np.asarray(v10, dtype=np.float64)

    phi_deg = np.mod(np.degrees(np.arctan2(u_arr, v_arr)), 360.0)
    speed = np.maximum(np.sqrt(u_arr**2 + v_arr**2), float(w_min))

    if is_scalar:
        return float(phi_deg), float(speed)
    return phi_deg, speed


def compute_wind_alignment_weight(
    phi_deg: float | np.ndarray,
    beta_deg: float | np.ndarray,
) -> np.ndarray | float:
    """Compute wind alignment factor a_i(t) = max(0.0, cos(phi(t) - beta_i)).

    1.0 when wind blows directly from fire toward target, 0.0 when blowing away or orthogonal.
    """
    is_scalar = np.isscalar(phi_deg) and np.isscalar(beta_deg)
    diff_rad = np.radians(
        np.asarray(phi_deg, dtype=np.float64) - np.asarray(beta_deg, dtype=np.float64)
    )
    alignment = np.maximum(0.0, np.cos(diff_rad))

    if is_scalar:
        return float(alignment)
    return alignment


def compute_advective_kernel(
    dist_km: float | np.ndarray,
    wind_speed_ms: float | np.ndarray,
    alignment_weight: float | np.ndarray,
    detection_utc: datetime | pd.Timestamp,
    valid_time_utc: datetime | pd.Timestamp,
    issuance_utc: datetime | pd.Timestamp,
    tau_decay_h: float = DEFAULT_TAU_DECAY_H,
    tau_age_h: float = DEFAULT_TAU_AGE_H,
) -> Any:
    """Compute advective-decay kernel K_i(t) for fire detection i at valid time t.

    K_i(t) = a_i(t) * exp(-Theta_i(t) / tau_decay) * exp(- (min(t, T) - tau_i)+ / tau_age)
    where Theta_i(t) = d_i / (3.6 * w(t)).
    """
    is_scalar = np.isscalar(dist_km) and np.isscalar(wind_speed_ms)
    d = np.asarray(dist_km, dtype=np.float64)
    w = np.asarray(wind_speed_ms, dtype=np.float64)
    a = np.asarray(alignment_weight, dtype=np.float64)

    theta_adv_h = d / (3.6 * np.maximum(w, 1e-4))
    travel_decay = np.exp(-theta_adv_h / float(tau_decay_h))

    t_val = pd.to_datetime(valid_time_utc, utc=True)
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    tau_det = pd.to_datetime(detection_utc, utc=True)

    # Effective observation time is capped at issuance T
    eff_time = min(t_val, t_iss)
    age_seconds = max(0.0, (eff_time - tau_det).total_seconds())
    age_hours = age_seconds / 3600.0
    age_decay = np.exp(-age_hours / float(tau_age_h))

    kernel = a * travel_decay * age_decay
    if is_scalar:
        return float(kernel)
    return kernel


def compute_hourly_fire_exposure(
    detections_df: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    wind_df: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp,
    valid_hours: Sequence[datetime | pd.Timestamp],
    platforms_active: int = 2,
    tau_decay_h: float = DEFAULT_TAU_DECAY_H,
    tau_age_h: float = DEFAULT_TAU_AGE_H,
    w_min: float = DEFAULT_WIND_MIN_MS,
) -> dict[pd.Timestamp, float]:
    """Compute instantaneous transport-weighted fire exposure E(t) for each valid hour.

    Only detections knowable at issuance T (acq_datetime_utc <= T) are included.
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    norm_plat = max(1.0, float(platforms_active))

    if detections_df.empty or "latitude" not in detections_df.columns:
        return {pd.to_datetime(h, utc=True): 0.0 for h in valid_hours}

    dets = detections_df.copy()
    if "acq_datetime_utc" in dets.columns:
        dets["acq_dt"] = pd.to_datetime(dets["acq_datetime_utc"], utc=True)
    elif "acq_date" in dets.columns and "acq_time" in dets.columns:
        time_strs = dets["acq_time"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        dets["acq_dt"] = pd.to_datetime(
            dets["acq_date"].astype(str) + " " + time_strs.str[:2] + ":" + time_strs.str[2:],
            utc=True,
            errors="coerce",
        )
    elif "ts_utc" in dets.columns:
        dets["acq_dt"] = pd.to_datetime(dets["ts_utc"], utc=True)
    else:
        return {pd.to_datetime(h, utc=True): 0.0 for h in valid_hours}

    # Point-in-time enforcement: only detections with acq_dt <= T
    active_dets = dets[dets["acq_dt"] <= t_iss].copy()
    if active_dets.empty:
        return {pd.to_datetime(h, utc=True): 0.0 for h in valid_hours}

    # Calculate target-centric geometry
    fire_lats = active_dets["latitude"].to_numpy(dtype=np.float64)
    fire_lons = active_dets["longitude"].to_numpy(dtype=np.float64)
    _, betas, dists_km, _ = compute_single_angle_geometry(
        target_lat, target_lon, fire_lats, fire_lons
    )
    active_dets["beta_deg"] = betas
    active_dets["dist_km"] = dists_km

    # Filter to radius <= 600 km
    active_dets = active_dets[active_dets["dist_km"] <= DEFAULT_MAX_RADIUS_KM]
    if active_dets.empty:
        return {pd.to_datetime(h, utc=True): 0.0 for h in valid_hours}

    # Prepare wind lookups
    wdf = wind_df.copy()
    if "ts_utc" in wdf.columns:
        wdf["ts_utc"] = pd.to_datetime(wdf["ts_utc"], utc=True)
        wdf = wdf.set_index("ts_utc")

    u_col = next(
        (c for c in ("u10_ms", "u10", "10m_u_component_of_wind") if c in wdf.columns), None
    )
    v_col = next(
        (c for c in ("v10_ms", "v10", "10m_v_component_of_wind") if c in wdf.columns), None
    )

    frp_vals = (
        active_dets["frp"].to_numpy(dtype=np.float64)
        if "frp" in active_dets.columns
        else np.ones(len(active_dets))
    )
    betas_arr = active_dets["beta_deg"].to_numpy(dtype=np.float64)
    dists_arr = active_dets["dist_km"].to_numpy(dtype=np.float64)
    acq_dts = active_dets["acq_dt"].tolist()

    exposure_by_hour: dict[pd.Timestamp, float] = {}

    for vh in valid_hours:
        vh_ts = pd.to_datetime(vh, utc=True)
        if wdf is not None and vh_ts in wdf.index and u_col and v_col:
            u_val = float(pd.Series(wdf.loc[vh_ts, u_col]).iloc[0])
            v_val = float(pd.Series(wdf.loc[vh_ts, v_col]).iloc[0])
        elif wdf is not None and not wdf.empty and u_col and v_col:
            # Fallback to nearest wind record
            idx = int(wdf.index.get_indexer(pd.DatetimeIndex([vh_ts]), method="nearest")[0])
            u_val = float(pd.Series(wdf.iloc[idx][u_col]).iloc[0])
            v_val = float(pd.Series(wdf.iloc[idx][v_col]).iloc[0])
        else:
            u_val = 0.0
            v_val = 0.0

        phi, w_speed = compute_wind_direction_and_speed(u_val, v_val, w_min=w_min)
        alignments = compute_wind_alignment_weight(phi, betas_arr)

        theta_adv = dists_arr / (3.6 * float(w_speed))
        travel_decays = np.exp(-theta_adv / float(tau_decay_h))

        eff_t = min(vh_ts, t_iss)
        age_hours = np.array(
            [max(0.0, (eff_t - tau_i).total_seconds() / 3600.0) for tau_i in acq_dts],
            dtype=np.float64,
        )
        age_decays = np.exp(-age_hours / float(tau_age_h))

        kernels = alignments * travel_decays * age_decays
        hourly_exp = float(np.sum((1.0 / norm_plat) * frp_vals * kernels))
        exposure_by_hour[vh_ts] = hourly_exp

    return exposure_by_hour


def extract_block_fire_features(
    detections_df: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    wind_df: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp,
    horizons: Sequence[int] = (24, 48, 72),
    platforms_active: int = 2,
    tau_decay_h: float = DEFAULT_TAU_DECAY_H,
    tau_age_h: float = DEFAULT_TAU_AGE_H,
) -> dict[str, float]:
    """Compute feature-store fire exposure features for horizons {24, 48, 72} and concentric rings.

    Returns dictionary containing:
    - Ring counts and FRP sums: `f_fire_count_r0_50`, `f_fire_frp_r0_50`, ...
    - Block transport exposures: `f_fire_exposure_e24`, `f_fire_exposure_e48`, `f_fire_exposure_e72`
    - Compass sector counts: `f_fire_count_sector_N`, `f_fire_count_sector_NE`, ...
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    norm_plat = max(1.0, float(platforms_active))
    features: dict[str, float] = {}

    # Initialize all features to 0.0
    for _, _, r_name in RING_BOUNDS:
        features[f"f_fire_count_{r_name}"] = 0.0
        features[f"f_fire_frp_{r_name}"] = 0.0

    for s_name in SECTOR_NAMES:
        features[f"f_fire_count_sector_{s_name}"] = 0.0

    for h in horizons:
        features[f"f_fire_exposure_e{h}"] = 0.0

    features["f_fire_detections_total"] = 0.0
    features["f_fire_frp_total"] = 0.0

    if detections_df.empty or "latitude" not in detections_df.columns:
        return features

    dets = detections_df.copy()
    if "acq_datetime_utc" in dets.columns:
        dets["acq_dt"] = pd.to_datetime(dets["acq_datetime_utc"], utc=True)
    elif "acq_date" in dets.columns and "acq_time" in dets.columns:
        time_strs = dets["acq_time"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        dets["acq_dt"] = pd.to_datetime(
            dets["acq_date"].astype(str) + " " + time_strs.str[:2] + ":" + time_strs.str[2:],
            utc=True,
            errors="coerce",
        )
    elif "ts_utc" in dets.columns:
        dets["acq_dt"] = pd.to_datetime(dets["ts_utc"], utc=True)
    else:
        return features

    # Filter detections to the trailing 24h before T (for ring counts) and knowable at T
    lookback_24h = t_iss - timedelta(hours=24)
    active_24h = dets[(dets["acq_dt"] <= t_iss) & (dets["acq_dt"] >= lookback_24h)].copy()

    if not active_24h.empty:
        lats = active_24h["latitude"].to_numpy(dtype=np.float64)
        lons = active_24h["longitude"].to_numpy(dtype=np.float64)
        _, _, dists, sector_idxs = compute_single_angle_geometry(target_lat, target_lon, lats, lons)
        active_24h["dist_km"] = dists
        active_24h["sector_idx"] = sector_idxs

        # Filter to 600 km
        in_radius = active_24h[active_24h["dist_km"] <= DEFAULT_MAX_RADIUS_KM]
        features["f_fire_detections_total"] = float(len(in_radius))
        features["f_fire_frp_total"] = (
            float(in_radius["frp"].sum()) if "frp" in in_radius.columns else 0.0
        )

        for r_min, r_max, r_name in RING_BOUNDS:
            r_slice = in_radius[(in_radius["dist_km"] >= r_min) & (in_radius["dist_km"] < r_max)]
            features[f"f_fire_count_{r_name}"] = float(len(r_slice)) / norm_plat
            features[f"f_fire_frp_{r_name}"] = (
                float(r_slice["frp"].sum()) / norm_plat
                if "frp" in r_slice.columns and not r_slice.empty
                else 0.0
            )

        for s_idx, s_name in enumerate(SECTOR_NAMES):
            s_slice = in_radius[in_radius["sector_idx"] == s_idx]
            features[f"f_fire_count_sector_{s_name}"] = float(len(s_slice)) / norm_plat

    # Compute hourly exposures across all horizon blocks
    max_h = max(horizons)
    forecast_hours = [t_iss + timedelta(hours=i) for i in range(1, int(max_h) + 1)]
    hourly_exposures = compute_hourly_fire_exposure(
        detections_df=dets,
        target_lat=target_lat,
        target_lon=target_lon,
        wind_df=wind_df,
        issuance_utc=t_iss,
        valid_hours=forecast_hours,
        platforms_active=platforms_active,
        tau_decay_h=tau_decay_h,
        tau_age_h=tau_age_h,
    )

    for h in horizons:
        block_hours = [t_iss + timedelta(hours=i) for i in range(int(h) - 23, int(h) + 1)]
        b_vals = [hourly_exposures.get(bh, 0.0) for bh in block_hours]
        features[f"f_fire_exposure_e{h}"] = float(np.mean(b_vals)) if b_vals else 0.0

    return features


def exposure(
    detections: pd.DataFrame,
    target: str | tuple[float, float],
    wind_series: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Public module contract: compute hourly fire exposure dataframe conforming to fire_exposure.schema.yaml.

    Parameters:
    - detections: DataFrame of active fire detections.
    - target: Domain name (e.g. 'lahore') or (latitude, longitude) coordinate tuple.
    - wind_series: DataFrame of hourly wind (must contain ts_utc, u10_ms, v10_ms).
    - params: Optional dict with keys (issuance_utc, platforms_active, coverage_complete).
    """
    if isinstance(target, str):
        domain = target.lower().strip()
        if domain not in DOMAIN_CENTROIDS:
            raise ValueError(
                f"Unknown target domain '{target}'. Expected one of {list(DOMAIN_CENTROIDS.keys())}"
            )
        target_lat, target_lon = DOMAIN_CENTROIDS[domain]
    else:
        target_lat, target_lon = target
        domain = "lahore"

    p = params or {}
    issuance_utc = p.get("issuance_utc")
    if issuance_utc is None:
        if "ts_utc" in wind_series.columns and not wind_series.empty:
            issuance_utc = pd.to_datetime(wind_series["ts_utc"], utc=True).min()
        else:
            issuance_utc = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    t_iss = pd.to_datetime(issuance_utc, utc=True)

    platforms_active = int(p.get("platforms_active", 2))
    coverage_complete = bool(p.get("coverage_complete", True))

    if "ts_utc" in wind_series.columns and not wind_series.empty:
        hours = pd.to_datetime(wind_series["ts_utc"], utc=True).tolist()
    else:
        hours = [t_iss + timedelta(hours=i) for i in range(24)]

    # Compute hourly exposures
    hourly_exps = compute_hourly_fire_exposure(
        detections_df=detections,
        target_lat=target_lat,
        target_lon=target_lon,
        wind_df=wind_series,
        issuance_utc=t_iss,
        valid_hours=hours,
        platforms_active=platforms_active,
    )

    norm_plat = max(1.0, float(platforms_active))
    rows: list[dict[str, Any]] = []

    # Filter detections to target
    if not detections.empty and "latitude" in detections.columns:
        dets = detections.copy()
        if "acq_datetime_utc" in dets.columns:
            dets["ts_utc"] = pd.to_datetime(dets["acq_datetime_utc"], utc=True).dt.floor("h")
        elif "ts_utc" in dets.columns:
            dets["ts_utc"] = pd.to_datetime(dets["ts_utc"], utc=True).dt.floor("h")

        _, _, dists, _ = compute_single_angle_geometry(
            target_lat, target_lon, dets["latitude"].to_numpy(), dets["longitude"].to_numpy()
        )
        dets["dist_km"] = dists
    else:
        dets = pd.DataFrame()

    for h in hours:
        h_ts = pd.to_datetime(h, utc=True)
        if not dets.empty and "ts_utc" in dets.columns:
            h_dets = dets[(dets["ts_utc"] == h_ts) & (dets["dist_km"] <= DEFAULT_MAX_RADIUS_KM)]
        else:
            h_dets = pd.DataFrame()

        n_total = len(h_dets)
        frp_total = float(h_dets["frp"].sum()) if n_total > 0 and "frp" in h_dets.columns else 0.0

        ring_counts: dict[str, float] = {}
        ring_frps: dict[str, float] = {}
        for r_min, r_max, r_name in RING_BOUNDS:
            if n_total > 0:
                r_sub = h_dets[(h_dets["dist_km"] >= r_min) & (h_dets["dist_km"] < r_max)]
                ring_counts[r_name] = float(len(r_sub)) / norm_plat
                ring_frps[r_name] = (
                    float(r_sub["frp"].sum()) / norm_plat if "frp" in r_sub.columns else 0.0
                )
            else:
                ring_counts[r_name] = 0.0
                ring_frps[r_name] = 0.0

        exp_val = hourly_exps.get(h_ts, 0.0)

        rows.append(
            {
                "domain": domain,
                "ts_utc": h_ts,
                "platforms_active": np.int32(platforms_active),
                "n_detections_total": np.int32(n_total),
                "frp_sum_mw_total": float(frp_total),
                "fire_count_r0_50": ring_counts["r0_50"],
                "frp_mw_r0_50": ring_frps["r0_50"],
                "fire_count_r50_150": ring_counts["r50_150"],
                "frp_mw_r50_150": ring_frps["r50_150"],
                "fire_count_r150_300": ring_counts["r150_300"],
                "frp_mw_r150_300": ring_frps["r150_300"],
                "fire_count_r300_600": ring_counts["r300_600"],
                "frp_mw_r300_600": ring_frps["r300_600"],
                "exposure_e24": exp_val,
                "exposure_e48": exp_val,
                "exposure_e72": exp_val,
                "coverage_complete": bool(coverage_complete),
            }
        )

    out_df = pd.DataFrame(rows)
    schema = _load_fire_exposure_schema()
    return validate_frame(out_df, schema)
