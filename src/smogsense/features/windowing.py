"""smogsense.features.windowing — Lookback sequence tensors with masks and elapsed time channels.

Builds L = 72 h hourly historical sequence matrices X in R^{72 x 10} per (station, issuance):
Channels defined in configs/features.yaml:
1. `pm25_log1p`: ln(1 + max(0, pm25_ugm3))
2. `cams_pm25_log1p`: ln(1 + max(0, cams_pm25))
3. `aod550`: Total CAMS AOD at 550 nm
4. `blh_log`: ln(max(PBLH, 50.0))
5. `u10_ms`: 10m eastward wind component
6. `v10_ms`: 10m northward wind component
7. `vc10_log`: ln(VC) with VC >= 10.0 m^2/s
8. `rh_pct`: Relative humidity in [0, 100]%
9. `t2m_c`: 2m temperature in Celsius
10. `fire_exposure_e24`: Target-centric wind-aligned fire exposure

Missing observation handling for PM2.5 follows Che et al. (2018) GRU-D representation:
Tuple [x * m, m, delta_t], where:
- m in {0, 1} is observation validity mask
- x * m is the zero-masked observation
- delta_t is elapsed time in hours since the last valid observation

Strict Point-in-Time Contract:
No record with ts_utc >= T or available_at_utc > T may be included.

Specification:
- docs/ml-architecture.md §2.4 ('Tensors, masks and scaling')
- configs/features.yaml ('sequence_channels')
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from smogsense.features.meteorology import (
    DEFAULT_BLH_FLOOR_M,
    DEFAULT_VC_FLOOR_M2S,
    relative_humidity_magnus,
    ventilation_coefficient,
)

SEQUENCE_CHANNELS: tuple[str, ...] = (
    "pm25_log1p",
    "cams_pm25_log1p",
    "aod550",
    "blh_log",
    "u10_ms",
    "v10_ms",
    "vc10_log",
    "rh_pct",
    "t2m_c",
    "fire_exposure_e24",
)


@dataclass
class SequenceTensor:
    """72-hour sequence tensor representation with missingness masks and elapsed time."""

    location_id: int
    issuance_utc: pd.Timestamp
    timestamps: pd.DatetimeIndex
    channels: list[str]
    values: np.ndarray  # Shape: (72, 10)
    mask: np.ndarray  # Shape: (72, 10)
    delta_t: np.ndarray  # Shape: (72,) elapsed hours since last valid PM2.5 observation

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def to_combined_array(self) -> np.ndarray:
        """Return combined tensor array including values, PM2.5 mask, and delta_t.

        Output shape: (72, 12): 10 value channels + pm25_mask + delta_t.
        """
        pm25_mask = self.mask[:, 0:1]
        delta_col = self.delta_t[:, np.newaxis]
        return np.hstack([self.values, pm25_mask, delta_col]).astype(np.float32)


def compute_elapsed_time_since_observation(mask: np.ndarray) -> np.ndarray:
    """Compute elapsed time delta_t (in hours) since the last valid observation (mask == 1).

    If observed at step t: delta_t = 0.
    If missing: delta_t = elapsed steps since previous valid observation.
    """
    n = len(mask)
    delta_t = np.zeros(n, dtype=np.float32)
    last_seen_step: int | None = None

    for i in range(n):
        if mask[i] > 0.5:
            delta_t[i] = 0.0
            last_seen_step = i
        else:
            if last_seen_step is None:
                delta_t[i] = float(i + 1)
            else:
                delta_t[i] = float(i - last_seen_step)

    return delta_t


def build_sequence_tensor(
    location_id: int,
    issuance_utc: datetime | pd.Timestamp,
    obs_df: pd.DataFrame,
    cams_df: pd.DataFrame,
    fire_exposure_df: pd.DataFrame | None = None,
    lookback_hours: int = 72,
) -> SequenceTensor:
    """Build the 72-hour sequence matrix X in R^{72 x 10} for a single station.

    Enforces strict point-in-time correctness:
    Raises ValueError if any data has available_at_utc > T or ts_utc >= T.
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    window_start = t_iss - timedelta(hours=lookback_hours)

    # 1. Point-in-time verification on inputs
    for name, df_in in [("observations", obs_df), ("cams", cams_df), ("fire", fire_exposure_df)]:
        if df_in is not None and not df_in.empty:
            if "available_at_utc" in df_in.columns:
                leaks = df_in[pd.to_datetime(df_in["available_at_utc"], utc=True) > t_iss]
                if not leaks.empty:
                    raise ValueError(
                        f"Point-in-time violation in {name}: found {len(leaks)} records with "
                        f"available_at_utc > issuance {t_iss}"
                    )
            if "available_at" in df_in.columns:
                leaks = df_in[pd.to_datetime(df_in["available_at"], utc=True) > t_iss]
                if not leaks.empty:
                    raise ValueError(
                        f"Point-in-time violation in {name}: found {len(leaks)} records with "
                        f"available_at > issuance {t_iss}"
                    )

    # 2. Construct 72-hour hourly lattice
    grid_hours = pd.date_range(window_start, t_iss, freq="1h", inclusive="left", tz=UTC)
    if len(grid_hours) != lookback_hours:
        grid_hours = grid_hours[:lookback_hours]

    n_steps = len(grid_hours)
    values = np.zeros((n_steps, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    mask = np.zeros((n_steps, len(SEQUENCE_CHANNELS)), dtype=np.float32)

    # 3. Align Observations (PM2.5)
    stn_obs = pd.DataFrame()
    if obs_df is not None and not obs_df.empty:
        obs_copy = obs_df.copy()
        obs_copy["ts_utc"] = pd.to_datetime(obs_copy["ts_utc"], utc=True)
        if "location_id" in obs_copy.columns:
            obs_copy = obs_copy[obs_copy["location_id"] == location_id]
        # Filter strictly within [window_start, t_iss)
        stn_obs = obs_copy[
            (obs_copy["ts_utc"] >= window_start) & (obs_copy["ts_utc"] < t_iss)
        ].set_index("ts_utc")

    pm25_obs_mask = np.zeros(n_steps, dtype=np.float32)
    for idx, h_ts in enumerate(grid_hours):
        if h_ts in stn_obs.index:
            row = stn_obs.loc[h_ts]
            val = float(pd.Series(row["pm25_ugm3"]).iloc[0]) if "pm25_ugm3" in row else float("nan")
            is_imputed = (
                bool(pd.Series(row["imputed"]).fillna(False).iloc[0]) if "imputed" in row else False
            )
            if pd.notna(val) and val >= 0.0 and not is_imputed:
                log_val = float(np.log1p(max(0.0, val)))
                values[idx, 0] = log_val
                mask[idx, 0] = 1.0
                pm25_obs_mask[idx] = 1.0
            else:
                values[idx, 0] = 0.0
                mask[idx, 0] = 0.0
        else:
            values[idx, 0] = 0.0
            mask[idx, 0] = 0.0

    # 4. Align CAMS gridded series (CAMS PM2.5, AOD, BLH, Winds, T2m, RH)
    cams_stn = pd.DataFrame()
    if cams_df is not None and not cams_df.empty:
        c_copy = cams_df.copy()
        c_copy["ts_utc"] = pd.to_datetime(c_copy["ts_utc"], utc=True)
        if "location_id" in c_copy.columns:
            c_sub = c_copy[c_copy["location_id"] == location_id]
            if not c_sub.empty:
                c_copy = c_sub
        cams_stn = c_copy[
            (c_copy["ts_utc"] >= window_start) & (c_copy["ts_utc"] < t_iss)
        ].set_index("ts_utc")

    cams_pm25_col = next(
        (c for c in ("cams_pm25", "pm25", "particulate_matter_2.5um") if c in cams_stn.columns),
        None,
    )
    aod_col = next(
        (c for c in ("aod550", "total_aerosol_optical_depth_550nm") if c in cams_stn.columns), None
    )
    blh_col = next(
        (c for c in ("blh_m", "boundary_layer_height", "blh") if c in cams_stn.columns), None
    )
    u_col = next(
        (c for c in ("u10_ms", "u10", "10m_u_component_of_wind") if c in cams_stn.columns), None
    )
    v_col = next(
        (c for c in ("v10_ms", "v10", "10m_v_component_of_wind") if c in cams_stn.columns), None
    )
    t2m_col = next((c for c in ("t2m_c", "t2m", "2m_temperature") if c in cams_stn.columns), None)
    d2m_col = next(
        (c for c in ("d2m_c", "d2m", "2m_dewpoint_temperature") if c in cams_stn.columns), None
    )
    rh_col = next((c for c in ("rh_pct", "rh", "relativehumidity") if c in cams_stn.columns), None)

    for idx, h_ts in enumerate(grid_hours):
        if h_ts in cams_stn.index:
            c_row = cams_stn.loc[h_ts]

            # CAMS PM2.5
            if cams_pm25_col and pd.notna(pd.Series(c_row[cams_pm25_col]).iloc[0]):
                c_val = float(pd.Series(c_row[cams_pm25_col]).iloc[0])
                values[idx, 1] = float(np.log1p(max(0.0, c_val)))
                mask[idx, 1] = 1.0

            # AOD 550
            if aod_col and pd.notna(pd.Series(c_row[aod_col]).iloc[0]):
                values[idx, 2] = float(pd.Series(c_row[aod_col]).iloc[0])
                mask[idx, 2] = 1.0

            # BLH log
            blh_val = float(DEFAULT_BLH_FLOOR_M)
            if blh_col and pd.notna(pd.Series(c_row[blh_col]).iloc[0]):
                blh_val = max(DEFAULT_BLH_FLOOR_M, float(pd.Series(c_row[blh_col]).iloc[0]))
                values[idx, 3] = float(np.log(blh_val))
                mask[idx, 3] = 1.0

            # Winds u10, v10
            u_val = 0.0
            v_val = 0.0
            if u_col and pd.notna(pd.Series(c_row[u_col]).iloc[0]):
                u_val = float(pd.Series(c_row[u_col]).iloc[0])
                values[idx, 4] = u_val
                mask[idx, 4] = 1.0
            if v_col and pd.notna(pd.Series(c_row[v_col]).iloc[0]):
                v_val = float(pd.Series(c_row[v_col]).iloc[0])
                values[idx, 5] = v_val
                mask[idx, 5] = 1.0

            # VC log
            vc_val = ventilation_coefficient(blh_val, u_val, v_val)
            values[idx, 6] = float(np.log(max(DEFAULT_VC_FLOOR_M2S, float(vc_val))))
            mask[idx, 6] = 1.0

            # T2m and RH
            t_val_c = 20.0
            if t2m_col and pd.notna(pd.Series(c_row[t2m_col]).iloc[0]):
                t_raw = float(pd.Series(c_row[t2m_col]).iloc[0])
                t_val_c = t_raw - 273.15 if t_raw > 150.0 else t_raw
                values[idx, 8] = t_val_c
                mask[idx, 8] = 1.0

            if rh_col and pd.notna(pd.Series(c_row[rh_col]).iloc[0]):
                values[idx, 7] = float(pd.Series(c_row[rh_col]).iloc[0])
                mask[idx, 7] = 1.0
            elif d2m_col and pd.notna(pd.Series(c_row[d2m_col]).iloc[0]):
                d_raw = float(pd.Series(c_row[d2m_col]).iloc[0])
                d_val_c = d_raw - 273.15 if d_raw > 150.0 else d_raw
                rh_val = float(relative_humidity_magnus(t_val_c, d_val_c))
                values[idx, 7] = rh_val
                mask[idx, 7] = 1.0

    # 5. Align Fire Exposure
    if fire_exposure_df is not None and not fire_exposure_df.empty:
        f_copy = fire_exposure_df.copy()
        f_copy["ts_utc"] = pd.to_datetime(f_copy["ts_utc"], utc=True)
        f_stn = f_copy[(f_copy["ts_utc"] >= window_start) & (f_copy["ts_utc"] < t_iss)].set_index(
            "ts_utc"
        )
        exp_col = next(
            (c for c in ("exposure_e24", "fire_exposure", "exposure") if c in f_stn.columns), None
        )
        if exp_col:
            for idx, h_ts in enumerate(grid_hours):
                if h_ts in f_stn.index:
                    e_val = float(pd.Series(f_stn.loc[h_ts, exp_col]).iloc[0])
                    values[idx, 9] = max(0.0, e_val)
                    mask[idx, 9] = 1.0

    # 6. Compute elapsed time delta_t for PM2.5 observation channel
    delta_t = compute_elapsed_time_since_observation(pm25_obs_mask)

    return SequenceTensor(
        location_id=location_id,
        issuance_utc=t_iss,
        timestamps=grid_hours,
        channels=list(SEQUENCE_CHANNELS),
        values=values,
        mask=mask,
        delta_t=delta_t,
    )


def build_batch_sequence_tensors(
    station_ids: Sequence[int],
    issuance_utc: datetime | pd.Timestamp,
    obs_df: pd.DataFrame,
    cams_df: pd.DataFrame,
    fire_exposure_df: pd.DataFrame | None = None,
    lookback_hours: int = 72,
) -> dict[int, SequenceTensor]:
    """Build SequenceTensor objects for multiple stations at issuance reference time T."""
    tensors: dict[int, SequenceTensor] = {}
    for sid in station_ids:
        tensors[sid] = build_sequence_tensor(
            location_id=sid,
            issuance_utc=issuance_utc,
            obs_df=obs_df,
            cams_df=cams_df,
            fire_exposure_df=fire_exposure_df,
            lookback_hours=lookback_hours,
        )
    return tensors
