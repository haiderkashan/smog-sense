"""smogsense.features.aerosol — AOD-derived features from CAMS (operational) and MAIAC (research).

Extracts total aerosol optical depth at 550 nm (AOD550) from CAMS station series:
- Horizon block statistics: mean and max AOD550 over [T+h-24, T+h).
- Trailing lookback statistics: mean AOD550 over [T-24, T).
- MAIAC research status: MAIAC is research-only and disabled in operations (ADR-006, docs/data-engineering.md §10).

Specification:
- docs/data-engineering.md §10 ('Aerosol optical depth strategy')
- configs/features.yaml ('aerosol')
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd


def extract_block_aerosol_features(
    cams_df: pd.DataFrame,
    issuance_utc: pd.Timestamp | str,
    horizons: Sequence[int] = (24, 48, 72),
) -> dict[str, float]:
    """Extract block-level CAMS AOD550 features for forecast horizons.

    Returns dictionary with keys:
    - `f_aod_mean_h<h>`
    - `f_aod_max_h<h>`
    - `f_aod_trailing_24h_mean`
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    features: dict[str, float] = {}

    for h in horizons:
        features[f"f_aod_mean_h{h}"] = float("nan")
        features[f"f_aod_max_h{h}"] = float("nan")
    features["f_aod_trailing_24h_mean"] = float("nan")

    if cams_df.empty or "ts_utc" not in cams_df.columns:
        return features

    df = cams_df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    aod_col = next(
        (
            c
            for c in (
                "aod550",
                "total_aerosol_optical_depth_550nm",
                "aerosol_optical_depth_550nm",
                "aod",
            )
            if c in df.columns
        ),
        None,
    )

    if not aod_col:
        return features

    # Trailing 24h before issuance
    lookback_start = t_iss - timedelta(hours=24)
    trailing_slice = df[(df["ts_utc"] >= lookback_start) & (df["ts_utc"] < t_iss)]
    if not trailing_slice.empty:
        vals = trailing_slice[aod_col].to_numpy(dtype=np.float64)
        features["f_aod_trailing_24h_mean"] = (
            float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
        )

    # Horizon blocks
    for h in horizons:
        block_start = t_iss + timedelta(hours=int(h) - 24)
        block_end = t_iss + timedelta(hours=int(h))
        blk = df[(df["ts_utc"] >= block_start) & (df["ts_utc"] < block_end)]
        if not blk.empty:
            vals = blk[aod_col].to_numpy(dtype=np.float64)
            features[f"f_aod_mean_h{h}"] = (
                float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
            )
            features[f"f_aod_max_h{h}"] = float(np.nanmax(vals)) if len(vals) > 0 else float("nan")

    return features
