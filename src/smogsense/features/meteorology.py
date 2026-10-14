"""smogsense.features.meteorology — RH (Magnus), wind, ventilation, dilution, stability.

Implements physics-informed meteorological features:
- Ventilation coefficient: VC(t) = max(PBLH, 50.0) * sqrt(u10^2 + v10^2), floored at VC >= 10.0 m^2/s.
- Log-ventilation: ln VC(t) aggregates (min, mean, night-mean where night is 18:00-06:00 local time).
- Dilution Index: DI = (1/24) * sum(1 / VC(t)) [harmonic box-model quantity].
- Stagnation run-lengths: max consecutive hours where VC(t) < theta for theta in {1000, 2000, 4000} m^2/s.
- Lower tropospheric stability: Delta theta = theta_925 - theta_2m with theta = T * (1000/p)^0.286.
- Inversion flag: (Delta theta > 0.0).
- Relative humidity via Magnus formula (Alduchov & Eskridge 1996):
    e_s(T) = 6.1094 * exp(17.625 * T / (T + 243.04)) hPa, RH = 100 * e_s(T_d) / e_s(T) in [0, 100]%.
- Fog flag: (RH > 90.0%).
- Scavenging proxy (block precipitation sum) and radiation proxy (mean surface solar radiation).

Specification:
- docs/ml-architecture.md §2 ('Feature engineering (physics-informed)')
- docs/data-engineering.md §7 ('Copernicus ADS: CAMS global forecasts')
- configs/features.yaml ('meteorology')
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Default floors from configs/features.yaml
DEFAULT_BLH_FLOOR_M = 50.0
DEFAULT_VC_FLOOR_M2S = 10.0
STAGNATION_THRESHOLDS_M2S: tuple[float, ...] = (1000.0, 2000.0, 4000.0)
POISSON_EXPONENT = 0.286  # R / c_p for dry air


def ventilation_coefficient(
    blh_m: float | np.ndarray | pd.Series,
    u10_ms: float | np.ndarray | pd.Series,
    v10_ms: float | np.ndarray | pd.Series,
    blh_floor_m: float = DEFAULT_BLH_FLOOR_M,
    vc_floor_m2s: float = DEFAULT_VC_FLOOR_M2S,
) -> Any:
    """Compute atmospheric ventilation coefficient VC = max(PBLH, blh_floor) * wind_speed.

    Strictly floored at `vc_floor_m2s` (default 10.0 m^2/s).
    Supports float scalars, numpy arrays, or pandas Series.
    """
    is_series = isinstance(blh_m, pd.Series) or isinstance(u10_ms, pd.Series)
    is_arr = isinstance(blh_m, np.ndarray) or isinstance(u10_ms, np.ndarray)

    blh_arr = np.asarray(blh_m, dtype=np.float64)
    u_arr = np.asarray(u10_ms, dtype=np.float64)
    v_arr = np.asarray(v10_ms, dtype=np.float64)
    eff_blh = np.maximum(blh_arr, float(blh_floor_m))
    ws = np.sqrt(u_arr**2 + v_arr**2)
    raw_vc = eff_blh * ws
    result = np.maximum(raw_vc, float(vc_floor_m2s))

    if is_series:
        idx = (
            blh_m.index
            if isinstance(blh_m, pd.Series)
            else (u10_ms.index if isinstance(u10_ms, pd.Series) else None)
        )
        return pd.Series(result, index=idx)
    if is_arr:
        return result
    return float(result)


def dilution_index(vc_series: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Compute the box-model Dilution Index DI = mean(1 / VC) over a block.

    This represents the harmonic mean behavior (mean of inverse VC, NOT inverse of mean VC),
    as concentration is inversely proportional to ventilation: C* ~ QL / VC.
    """
    arr = np.asarray(vc_series, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    # Guard against zero/negative, although VC >= 10.0 by contract
    arr_safe = np.maximum(arr, DEFAULT_VC_FLOOR_M2S)
    return float(np.mean(1.0 / arr_safe))


def stagnation_run_length(
    vc_series: Sequence[float] | np.ndarray | pd.Series,
    threshold_m2s: float = 1000.0,
) -> int:
    """Compute the longest run of consecutive hours where VC(t) < threshold_m2s."""
    arr = np.asarray(vc_series, dtype=np.float64)
    if len(arr) == 0:
        return 0

    is_stagnant = arr < float(threshold_m2s)
    max_run = 0
    current_run = 0

    for stagnant in is_stagnant:
        if stagnant:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return max_run


def potential_temperature(
    temp_k: float | np.ndarray | pd.Series,
    pressure_hpa: float | np.ndarray | pd.Series = 1000.0,
) -> Any:
    """Compute potential temperature theta = T * (1000.0 / p)^0.286 with T in Kelvin."""
    is_series = isinstance(temp_k, pd.Series) or isinstance(pressure_hpa, pd.Series)
    is_arr = isinstance(temp_k, np.ndarray) or isinstance(pressure_hpa, np.ndarray)

    t_arr = np.asarray(temp_k, dtype=np.float64)
    p_arr = np.asarray(pressure_hpa, dtype=np.float64)
    res = t_arr * (1000.0 / p_arr) ** POISSON_EXPONENT

    if is_series:
        idx = (
            temp_k.index
            if isinstance(temp_k, pd.Series)
            else (pressure_hpa.index if isinstance(pressure_hpa, pd.Series) else None)
        )
        return pd.Series(res, index=idx)
    if is_arr:
        return res
    return float(res)


def lower_tropospheric_stability(
    t925_k: float | np.ndarray | pd.Series,
    t2m_k: float | np.ndarray | pd.Series,
    surface_pressure_hpa: float | np.ndarray | pd.Series = 1013.25,
) -> tuple[Any, Any]:
    """Compute Lower Tropospheric Stability (LTS) Delta theta = theta_925 - theta_2m.

    Positive values indicate thermal inversion / atmospheric trapping.
    Returns (delta_theta_k, inversion_flag).
    """
    theta_925 = potential_temperature(t925_k, 925.0)
    theta_2m = potential_temperature(t2m_k, surface_pressure_hpa)
    delta_theta = theta_925 - theta_2m
    inversion = delta_theta > 0.0

    return delta_theta, inversion


def relative_humidity_magnus(
    t2m_c: float | np.ndarray | pd.Series,
    d2m_c: float | np.ndarray | pd.Series,
) -> Any:
    """Compute Relative Humidity (%) from 2m dry-bulb and dewpoint temperatures in Celsius.

    Uses the Alduchov & Eskridge (1996) Magnus formula:
        e_s(T) = 6.1094 * exp(17.625 * T / (T + 243.04)) hPa
        RH = 100.0 * e_s(T_d) / e_s(T) bounded to [0.0, 100.0]%.
    """
    is_series = isinstance(t2m_c, pd.Series) or isinstance(d2m_c, pd.Series)
    is_arr = isinstance(t2m_c, np.ndarray) or isinstance(d2m_c, np.ndarray)

    t_arr = np.asarray(t2m_c, dtype=np.float64)
    td_arr = np.asarray(d2m_c, dtype=np.float64)
    es_t = 6.1094 * np.exp(17.625 * t_arr / (t_arr + 243.04))
    es_td = 6.1094 * np.exp(17.625 * td_arr / (td_arr + 243.04))
    rh_raw = 100.0 * (es_td / np.maximum(es_t, 1e-6))
    rh_clipped = np.clip(rh_raw, 0.0, 100.0)

    if is_series:
        idx = (
            t2m_c.index
            if isinstance(t2m_c, pd.Series)
            else (d2m_c.index if isinstance(d2m_c, pd.Series) else None)
        )
        return pd.Series(rh_clipped, index=idx)
    if is_arr:
        return rh_clipped
    return float(rh_clipped)


def extract_block_meteorology_features(
    cams_df: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp,
    horizons: Sequence[int] = (24, 48, 72),
    timezone: str = "Asia/Karachi",
) -> dict[str, float]:
    """Compute block-level physics-informed meteorological features for forecast horizons.

    Input DataFrame `cams_df` must contain:
    - `ts_utc`: timestamp in UTC
    - `blh_m` (or `boundary_layer_height`): boundary layer height in meters
    - `u10_ms` (or `u10`): 10m eastward wind in m/s
    - `v10_ms` (or `v10`): 10m northward wind in m/s
    - `t2m_c` (or `t2m`): 2m temperature in Celsius (or Kelvin >= 200, auto-converted)
    - `d2m_c` (or `d2m`): 2m dewpoint in Celsius (or Kelvin >= 200, auto-converted)
    Optional columns:
    - `t925_k` (or `t925`): 925 hPa temperature in Kelvin
    - `sp_hpa` (or `surface_pressure`): surface pressure in hPa
    - `tp_m` (or `total_precipitation`): precipitation in meters or mm
    - `ssrd_wm2` (or `surface_solar_radiation_downwards`): solar radiation

    Returns dictionary of features keyed by `f_met_<feature_name>_h<horizon>`.
    """
    t0 = pd.to_datetime(issuance_utc, utc=True)
    features: dict[str, float] = {}

    if cams_df.empty or "ts_utc" not in cams_df.columns:
        for h in horizons:
            for suffix in (
                "vc_min",
                "vc_mean",
                "ln_vc_min",
                "ln_vc_mean",
                "ln_vc_night_mean",
                "dilution_index",
                "stag_run_1000",
                "stag_run_2000",
                "stag_run_4000",
                "rh_mean",
                "fog_hours",
                "delta_theta_mean",
                "inversion_hours",
                "precip_sum",
                "ssrd_mean",
            ):
                features[f"f_met_{suffix}_h{h}"] = float("nan")
        return features

    df = cams_df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    # Standardize column names
    blh_col = next((c for c in ("blh_m", "boundary_layer_height", "blh") if c in df.columns), None)
    u_col = next((c for c in ("u10_ms", "10m_u_component_of_wind", "u10") if c in df.columns), None)
    v_col = next((c for c in ("v10_ms", "10m_v_component_of_wind", "v10") if c in df.columns), None)
    t2m_col = next((c for c in ("t2m_c", "2m_temperature", "t2m") if c in df.columns), None)
    d2m_col = next((c for c in ("d2m_c", "2m_dewpoint_temperature", "d2m") if c in df.columns), None)
    t925_col = next((c for c in ("t925_k", "temperature_925", "t925") if c in df.columns), None)
    sp_col = next((c for c in ("sp_hpa", "surface_pressure", "sp") if c in df.columns), None)
    tp_col = next((c for c in ("tp_m", "total_precipitation", "tp") if c in df.columns), None)
    ssrd_col = next(
        (c for c in ("ssrd_wm2", "surface_solar_radiation_downwards", "ssrd") if c in df.columns),
        None,
    )

    tz = ZoneInfo(timezone)

    for h in horizons:
        block_start = t0 + timedelta(hours=int(h) - 24)
        block_end = t0 + timedelta(hours=int(h))
        blk = df[(df["ts_utc"] >= block_start) & (df["ts_utc"] < block_end)].copy()

        if blk.empty:
            for suffix in (
                "vc_min",
                "vc_mean",
                "ln_vc_min",
                "ln_vc_mean",
                "ln_vc_night_mean",
                "dilution_index",
                "stag_run_1000",
                "stag_run_2000",
                "stag_run_4000",
                "rh_mean",
                "fog_hours",
                "delta_theta_mean",
                "inversion_hours",
                "precip_sum",
                "ssrd_mean",
            ):
                features[f"f_met_{suffix}_h{h}"] = float("nan")
            continue

        # 1. Ventilation & Dilution
        if blh_col and u_col and v_col:
            blh_vals = blk[blh_col].to_numpy(dtype=np.float64)
            u_vals = blk[u_col].to_numpy(dtype=np.float64)
            v_vals = blk[v_col].to_numpy(dtype=np.float64)
            vc = ventilation_coefficient(blh_vals, u_vals, v_vals)
            ln_vc = np.log(vc)

            features[f"f_met_vc_min_h{h}"] = float(np.min(vc))
            features[f"f_met_vc_mean_h{h}"] = float(np.mean(vc))
            features[f"f_met_ln_vc_min_h{h}"] = float(np.min(ln_vc))
            features[f"f_met_ln_vc_mean_h{h}"] = float(np.mean(ln_vc))
            features[f"f_met_dilution_index_h{h}"] = dilution_index(vc)

            # Stagnation run-lengths
            for thresh in STAGNATION_THRESHOLDS_M2S:
                th_int = int(thresh)
                features[f"f_met_stag_run_{th_int}_h{h}"] = float(
                    stagnation_run_length(vc, threshold_m2s=thresh)
                )

            # Night-mean log ventilation: local time 18:00 - 06:00
            local_hours = blk["ts_utc"].dt.tz_convert(tz).dt.hour.to_numpy()
            night_mask = (local_hours >= 18) | (local_hours < 6)
            if np.any(night_mask):
                features[f"f_met_ln_vc_night_mean_h{h}"] = float(np.mean(ln_vc[night_mask]))
            else:
                features[f"f_met_ln_vc_night_mean_h{h}"] = float(np.mean(ln_vc))
        else:
            features[f"f_met_vc_min_h{h}"] = float("nan")
            features[f"f_met_vc_mean_h{h}"] = float("nan")
            features[f"f_met_ln_vc_min_h{h}"] = float("nan")
            features[f"f_met_ln_vc_mean_h{h}"] = float("nan")
            features[f"f_met_ln_vc_night_mean_h{h}"] = float("nan")
            features[f"f_met_dilution_index_h{h}"] = float("nan")
            for thresh in STAGNATION_THRESHOLDS_M2S:
                features[f"f_met_stag_run_{int(thresh)}_h{h}"] = float("nan")

        # 2. Moisture, Fog & Magnus RH
        if t2m_col and d2m_col:
            t2m_vals = blk[t2m_col].to_numpy(dtype=np.float64)
            d2m_vals = blk[d2m_col].to_numpy(dtype=np.float64)
            # Temperature unit check: if > 150, assume Kelvin and convert to Celsius
            t2m_c = t2m_vals - 273.15 if np.nanmean(t2m_vals) > 150.0 else t2m_vals
            d2m_c = d2m_vals - 273.15 if np.nanmean(d2m_vals) > 150.0 else d2m_vals

            rh = relative_humidity_magnus(t2m_c, d2m_c)
            features[f"f_met_rh_mean_h{h}"] = float(np.mean(rh))
            features[f"f_met_fog_hours_h{h}"] = float(np.sum(rh > 90.0))
        else:
            features[f"f_met_rh_mean_h{h}"] = float("nan")
            features[f"f_met_fog_hours_h{h}"] = float("nan")

        # 3. Atmospheric Stability & Inversion
        if t925_col and t2m_col:
            t925_vals = blk[t925_col].to_numpy(dtype=np.float64)
            t2m_vals = blk[t2m_col].to_numpy(dtype=np.float64)
            t925_k = t925_vals if np.nanmean(t925_vals) > 150.0 else t925_vals + 273.15
            t2m_k = t2m_vals if np.nanmean(t2m_vals) > 150.0 else t2m_vals + 273.15

            if sp_col:
                sp_vals = blk[sp_col].to_numpy(dtype=np.float64)
                # If Pa (> 50000), convert to hPa
                sp_hpa = sp_vals / 100.0 if np.nanmean(sp_vals) > 5000.0 else sp_vals
            else:
                sp_hpa = np.full_like(t2m_k, 1013.25)

            delta_th, inv_flags = lower_tropospheric_stability(t925_k, t2m_k, sp_hpa)
            features[f"f_met_delta_theta_mean_h{h}"] = float(np.mean(delta_th))
            features[f"f_met_inversion_hours_h{h}"] = float(np.sum(inv_flags))
        else:
            features[f"f_met_delta_theta_mean_h{h}"] = float("nan")
            features[f"f_met_inversion_hours_h{h}"] = float("nan")

        # 4. Precipitation & Radiation
        if tp_col:
            tp_vals = blk[tp_col].to_numpy(dtype=np.float64)
            features[f"f_met_precip_sum_h{h}"] = float(np.sum(np.nan_to_num(tp_vals, nan=0.0)))
        else:
            features[f"f_met_precip_sum_h{h}"] = 0.0

        if ssrd_col:
            ssrd_vals = blk[ssrd_col].to_numpy(dtype=np.float64)
            features[f"f_met_ssrd_mean_h{h}"] = float(
                np.nanmean(ssrd_vals) if len(ssrd_vals) > 0 else 0.0
            )
        else:
            features[f"f_met_ssrd_mean_h{h}"] = 0.0

    return features
