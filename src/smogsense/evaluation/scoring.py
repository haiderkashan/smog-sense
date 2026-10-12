"""smogsense.evaluation.scoring — Pinball loss, interval score, CRPS, and verification block evaluation.

Specification: docs/evaluation-strategy.md → 'CRPS and quantile scoring'
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from smogsense.evaluation.crps import crps_from_quantiles
from smogsense.publishing.bulletin import determine_aqi_category

# Standard 19 quantiles: q05..q95 in steps of 0.05
Q_NAMES = [f"q{round((0.05 * (i + 1)) * 100):02d}" for i in range(19)]


def pinball_loss(y: float, q: float, tau: float) -> float:
    """Compute pinball (quantile) loss rho_tau(y - q) = max(tau * (y - q), (tau - 1) * (y - q))."""
    diff = y - q
    return float(max(tau * diff, (tau - 1.0) * diff))


def interval_score(y: float, lower: float, upper: float, alpha: float = 0.20) -> float:
    """Compute interval score IS_alpha(lower, upper, y) for (1 - alpha) prediction interval.

    Formula: IS = (upper - lower) + (2 / alpha) * max(0, lower - y) + (2 / alpha) * max(0, y - upper).
    """
    width = upper - lower
    penalty = (2.0 / alpha) * (max(0.0, lower - y) + max(0.0, y - upper))
    return float(width + penalty)


def compute_crps(quantiles: np.ndarray | list[float], y: float) -> float:
    """Compute Continuous Ranked Probability Score (CRPS) from 19 quantile levels."""
    q_arr = np.asarray(quantiles, dtype=float)
    return float(crps_from_quantiles(q_arr, y))


def score_forecast_block(
    forecast_row: pd.Series | dict[str, Any],
    valid_obs_series: pd.Series,
    stage: str,
    scored_at_utc: datetime | None = None,
    scoring_basis: str = "station",
    truth_basis: str | None = None,
) -> dict[str, Any]:
    """Score a single forecast horizon block against observed values.

    Enforces the 18-of-24 valid hours completeness rule (ADR-005):
    - If valid hours >= 18: computes block mean y_obs, CRPS, pinballs, interval score, hits, and AQI category.
    - If valid hours < 18: flags valid=False and leaves all metrics as None.
    """
    if scored_at_utc is None:
        scored_at_utc = datetime.now(UTC)

    # Clean valid observations (non-null)
    clean_obs = valid_obs_series.dropna()
    n_valid_hours = len(clean_obs)
    is_valid = n_valid_hours >= 18

    # Extract quantiles
    q10 = (
        float(forecast_row["q10"])
        if "q10" in forecast_row and pd.notna(forecast_row["q10"])
        else None
    )
    q50 = (
        float(forecast_row["q50"])
        if "q50" in forecast_row and pd.notna(forecast_row["q50"])
        else None
    )
    q90 = (
        float(forecast_row["q90"])
        if "q90" in forecast_row and pd.notna(forecast_row["q90"])
        else None
    )

    # Quantile array for CRPS
    q_vals = [
        float(forecast_row[q_name])
        for q_name in Q_NAMES
        if q_name in forecast_row and pd.notna(forecast_row[q_name])
    ]

    y_obs: float | None = None
    crps_val: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    is80: float | None = None
    hit_q10: bool | None = None
    hit_q50: bool | None = None
    hit_q90: bool | None = None
    obs_cat: str | None = None

    if is_valid:
        y_obs = float(clean_obs.mean())
        if q10 is not None:
            p10 = pinball_loss(y_obs, q10, 0.10)
            hit_q10 = bool(y_obs <= q10)
        if q50 is not None:
            p50 = pinball_loss(y_obs, q50, 0.50)
            hit_q50 = bool(y_obs <= q50)
        if q90 is not None:
            p90 = pinball_loss(y_obs, q90, 0.90)
            hit_q90 = bool(y_obs <= q90)
        if q10 is not None and q90 is not None:
            is80 = interval_score(y_obs, q10, q90, alpha=0.20)
        if len(q_vals) == 19:
            crps_val = compute_crps(q_vals, y_obs)

        obs_cat = determine_aqi_category(y_obs)

    issuance = forecast_row["issuance_utc"]
    if isinstance(issuance, str):
        issuance_dt = pd.to_datetime(issuance, utc=True).to_pydatetime()
    else:
        issuance_dt = issuance

    run_id = str(forecast_row.get("run_id") or forecast_row.get("forecast_run_id", "unknown"))

    return {
        "issuance_utc": issuance_dt,
        "scored_at_utc": scored_at_utc,
        "stage": stage,
        "domain": str(forecast_row["domain"]),
        "point_id": str(forecast_row["point_id"]),
        "horizon_h": np.int32(forecast_row["horizon_h"]),
        "method": str(forecast_row["method"]),
        "y_obs": y_obs,
        "n_valid_hours": np.int32(n_valid_hours),
        "valid": is_valid,
        "crps": crps_val,
        "pinball_q10": p10,
        "pinball_q50": p50,
        "pinball_q90": p90,
        "interval_score_80": is80,
        "hit_below_q10": hit_q10,
        "hit_below_q50": hit_q50,
        "hit_below_q90": hit_q90,
        "obs_category": obs_cat,
        "forecast_run_id": run_id,
        "scoring_basis": scoring_basis,
        "truth_basis": truth_basis
        or ("openaq_api" if stage == "provisional" else "openaq_archive"),
    }
