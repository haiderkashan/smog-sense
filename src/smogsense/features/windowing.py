"""smogsense.features.windowing — Lookback sequence tensors with masks and elapsed time channels.

Builds L = 72 h hourly historical sequence matrices X in R^{72 x 6} per (station, issuance):
Six local observation channels defined in docs/ml-architecture.md §2.4 & configs/features.yaml:
1. `pm25_log1p_masked`: ln(1 + max(0, pm25_ugm3)) * m
2. `obs_minus_cams_pm25`: (pm25_ugm3 - cams_pm25) * m
3. `rh_pct`: Relative humidity in [0, 100]% (sensor's own where present, otherwise CAMS 2m)
4. `t2m_c`: 2m temperature in Celsius (sensor's own where present, otherwise CAMS 2m)
5. `pm25_mask`: Validity mask m in {0, 1}
6. `delta_t`: Elapsed time in hours since the last valid observation

Missing observation handling for PM2.5 follows Che et al. (2018) GRU-D representation.
CAMS physics and fire reach the model through the base only.

Strict Point-in-Time Contract:
No record with ts_utc >= T or available_at_utc > T may be included.

Specification:
- docs/ml-architecture.md §2.4 ('Tensors, masks and scaling') & §4 ('Sequence encoder')
- configs/features.yaml ('sequence_channels')
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from smogsense.features.meteorology import relative_humidity_magnus

SEQUENCE_CHANNELS: tuple[str, ...] = (
    "pm25_log1p_masked",
    "obs_minus_cams_pm25",
    "rh_pct",
    "t2m_c",
    "pm25_mask",
    "delta_t",
)


@dataclass
class SequenceTensor:
    """72-hour sequence tensor representation with missingness masks and elapsed time."""

    location_id: int
    issuance_utc: pd.Timestamp
    timestamps: pd.DatetimeIndex
    channels: list[str]
    values: np.ndarray  # Shape: (72, 6)
    mask: np.ndarray  # Shape: (72, 6)
    delta_t: np.ndarray  # Shape: (72,) elapsed hours since last valid PM2.5 observation

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def to_combined_array(self) -> np.ndarray:
        """Return combined tensor array adhering to C = 6 channels.

        Output shape: (72, 6).
        """
        return self.values.astype(np.float32)


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
    """Build the 72-hour sequence matrix X in R^{72 x 6} for a single station.

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

    # 3. Align Observations (PM2.5, optional sensor RH and T2m)
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

    # 4. Align CAMS gridded series (CAMS PM2.5, T2m, RH)
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
    t2m_col = next((c for c in ("t2m_c", "t2m", "2m_temperature") if c in cams_stn.columns), None)
    d2m_col = next(
        (c for c in ("d2m_c", "d2m", "2m_dewpoint_temperature") if c in cams_stn.columns), None
    )
    rh_col = next((c for c in ("rh_pct", "rh", "relativehumidity") if c in cams_stn.columns), None)

    pm25_obs_mask = np.zeros(n_steps, dtype=np.float32)

    for idx, h_ts in enumerate(grid_hours):
        # --- Channel 0: pm25_log1p_masked ---
        obs_pm25: float | None = None
        if h_ts in stn_obs.index:
            row = stn_obs.loc[h_ts]
            val = float(pd.Series(row["pm25_ugm3"]).iloc[0]) if "pm25_ugm3" in row else float("nan")
            is_imputed = (
                bool(pd.Series(row["imputed"]).fillna(False).iloc[0]) if "imputed" in row else False
            )
            if pd.notna(val) and val >= 0.0 and not is_imputed:
                obs_pm25 = val
                values[idx, 0] = float(np.log1p(max(0.0, val)))
                mask[idx, 0] = 1.0
                pm25_obs_mask[idx] = 1.0
            else:
                values[idx, 0] = 0.0
                mask[idx, 0] = 0.0
        else:
            values[idx, 0] = 0.0
            mask[idx, 0] = 0.0

        # --- Channel 1: obs_minus_cams_pm25 ---
        cams_val: float | None = None
        if h_ts in cams_stn.index:
            c_row = cams_stn.loc[h_ts]
            if cams_pm25_col and pd.notna(pd.Series(c_row[cams_pm25_col]).iloc[0]):
                cams_val = float(pd.Series(c_row[cams_pm25_col]).iloc[0])

        if obs_pm25 is not None and cams_val is not None:
            values[idx, 1] = float(obs_pm25 - cams_val)
            mask[idx, 1] = 1.0
        else:
            values[idx, 1] = 0.0
            mask[idx, 1] = 0.0

        # --- Channel 2: rh_pct (sensor's own where present, otherwise CAMS 2m) ---
        rh_assigned = False
        if h_ts in stn_obs.index:
            row = stn_obs.loc[h_ts]
            for col in ("rh_pct", "rh", "relativehumidity"):
                if col in row and pd.notna(pd.Series(row[col]).iloc[0]):
                    values[idx, 2] = float(pd.Series(row[col]).iloc[0])
                    mask[idx, 2] = 1.0
                    rh_assigned = True
                    break

        if not rh_assigned and h_ts in cams_stn.index:
            c_row = cams_stn.loc[h_ts]
            if rh_col and pd.notna(pd.Series(c_row[rh_col]).iloc[0]):
                values[idx, 2] = float(pd.Series(c_row[rh_col]).iloc[0])
                mask[idx, 2] = 1.0
                rh_assigned = True
            elif (
                d2m_col
                and t2m_col
                and pd.notna(pd.Series(c_row[d2m_col]).iloc[0])
                and pd.notna(pd.Series(c_row[t2m_col]).iloc[0])
            ):
                t_raw = float(pd.Series(c_row[t2m_col]).iloc[0])
                t_val_c = t_raw - 273.15 if t_raw > 150.0 else t_raw
                d_raw = float(pd.Series(c_row[d2m_col]).iloc[0])
                d_val_c = d_raw - 273.15 if d_raw > 150.0 else d_raw
                values[idx, 2] = float(relative_humidity_magnus(t_val_c, d_val_c))
                mask[idx, 2] = 1.0
                rh_assigned = True

        if not rh_assigned:
            values[idx, 2] = 50.0
            mask[idx, 2] = 0.0

        # --- Channel 3: t2m_c (sensor's own where present, otherwise CAMS 2m) ---
        t_assigned = False
        if h_ts in stn_obs.index:
            row = stn_obs.loc[h_ts]
            for col in ("t2m_c", "t2m", "temperature"):
                if col in row and pd.notna(pd.Series(row[col]).iloc[0]):
                    t_val = float(pd.Series(row[col]).iloc[0])
                    values[idx, 3] = t_val - 273.15 if t_val > 150.0 else t_val
                    mask[idx, 3] = 1.0
                    t_assigned = True
                    break

        if not t_assigned and h_ts in cams_stn.index:
            c_row = cams_stn.loc[h_ts]
            if t2m_col and pd.notna(pd.Series(c_row[t2m_col]).iloc[0]):
                t_raw = float(pd.Series(c_row[t2m_col]).iloc[0])
                values[idx, 3] = t_raw - 273.15 if t_raw > 150.0 else t_raw
                mask[idx, 3] = 1.0
                t_assigned = True

        if not t_assigned:
            values[idx, 3] = 20.0
            mask[idx, 3] = 0.0

        # --- Channel 4: pm25_mask ---
        values[idx, 4] = pm25_obs_mask[idx]
        mask[idx, 4] = 1.0

    # --- Channel 5: delta_t (elapsed hours since last valid PM2.5 observation) ---
    delta_t = compute_elapsed_time_since_observation(pm25_obs_mask)
    values[:, 5] = delta_t
    mask[:, 5] = 1.0

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
