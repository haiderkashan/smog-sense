"""Unit tests for Task P1-17: Baselines M2, M3, M3b and City-Panel Target.

Covers:
1. M2CamsBiasCorrected (m2_cams_bc): systematic bias removal, 19-quantile output, M1 fallback (< 7 days), PIT check.
2. M3Climatology (m3_climatology): +/- 15-day DOY window (including Dec/Jan wrap-around), non-crossing non-negative quantiles, N=0 error.
3. M3bLinearQuantileRegression (m3b_linear_qr): compact point-in-time predictors, monotone rearrangement, edge cases, PIT check.
4. compute_city_panel_target & block_targets: 50m co-location hierarchy (reference over low-cost median), >= 18 non-imputed hours rule, >= 3 valid stations rule.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from smogsense.features.targets import block_targets, compute_city_panel_target
from smogsense.models.baselines import (
    M2CamsBiasCorrected,
    M3bLinearQuantileRegression,
    M3Climatology,
)
from smogsense.models.distribution import QuantileFunction

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. M2CamsBiasCorrected
# ---------------------------------------------------------------------------


def test_m2_cams_bc_removes_systematic_bias_and_emits_19_quantiles() -> None:
    rng = np.random.default_rng(42)
    n_days = 25
    cams_hist = np.linspace(40.0, 120.0, n_days)
    # True relationship: y = 1.5 * cams + 25.0 + small zero-mean noise
    noise = rng.normal(0.0, 3.0, size=n_days)
    obs_hist = 1.5 * cams_hist + 25.0 + noise

    m2 = M2CamsBiasCorrected(cams_history=cams_hist, obs_history=obs_hist, min_samples=7)
    assert not m2.is_fallback_
    assert m2.n_samples_ == n_days

    # Check point prediction on CAMS = 100.0 -> expected ~ 175.0 (whereas raw CAMS is 100.0)
    pred_qf = m2.predict(100.0)
    assert isinstance(pred_qf, QuantileFunction)
    assert len(pred_qf.q) == 19
    assert np.all(np.diff(pred_qf.q) >= 0.0), "Quantiles must be non-crossing"
    assert np.all(pred_qf.q >= 0.0), "Quantiles must be non-negative"

    median_pred = float(pred_qf.ppf(0.50))
    assert abs(median_pred - 175.0) < 5.0, (
        f"Expected ~175.0 after bias correction, got {median_pred}"
    )


def test_m2_cams_bc_log_space_mode() -> None:
    cams_hist = np.linspace(30.0, 150.0, 20)
    obs_hist = np.expm1(0.2 + 1.05 * np.log1p(cams_hist))
    m2 = M2CamsBiasCorrected(
        cams_history=cams_hist,
        obs_history=obs_hist,
        log_space=True,
    )
    assert not m2.is_fallback_
    qf = m2.predict(80.0)
    assert isinstance(qf, QuantileFunction)
    expected = np.expm1(0.2 + 1.05 * np.log1p(80.0))
    assert np.isclose(float(qf.ppf(0.50)), expected, atol=1.0)


def test_m2_cams_bc_fallback_to_m1_when_samples_insufficient() -> None:
    # Only 4 days of history (< min_samples=7)
    cams_short = [50.0, 60.0, 70.0, 80.0]
    obs_short = [90.0, 100.0, 110.0, 120.0]
    fallback_res = np.linspace(-10.0, 10.0, 19)

    m2 = M2CamsBiasCorrected(
        cams_history=cams_short,
        obs_history=obs_short,
        min_samples=7,
        fallback_residual_quantiles=fallback_res,
    )
    assert m2.is_fallback_ is True
    qf = m2.predict(65.0)
    assert isinstance(qf, QuantileFunction)
    assert np.isclose(float(qf.ppf(0.50)), 65.0)

    # Also test N=0 deterministic fallback when requested
    m2_empty = M2CamsBiasCorrected()
    assert m2_empty.is_fallback_ is True
    assert m2_empty.predict(55.0, deterministic_at_n0=True) == 55.0
    qf_step = m2_empty.predict(55.0, deterministic_at_n0=False)
    assert isinstance(qf_step, QuantileFunction)
    assert np.allclose(qf_step.q, 55.0)


def test_m2_cams_bc_dataframe_and_pit_guard() -> None:
    t0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    dates = [t0 - timedelta(days=i) for i in range(1, 15)]
    df = pd.DataFrame(
        {
            "ts_utc": dates,
            "available_at_utc": [d + timedelta(hours=3) for d in dates],
            "cams_pm25": np.linspace(50.0, 100.0, 14),
            "obs_pm25": np.linspace(70.0, 120.0, 14),
        }
    )
    m2 = M2CamsBiasCorrected(history_df=df, issuance_utc=t0)
    assert not m2.is_fallback_
    qf = m2.predict(80.0)
    assert isinstance(qf, QuantileFunction)
    assert np.isclose(float(qf.ppf(0.50)), 100.0, atol=1.0)

    # Inject a future observation (available_at_utc > t0) -> must raise ValueError
    df_leak = df.copy()
    df_leak.loc[0, "available_at_utc"] = t0 + timedelta(hours=2)
    with pytest.raises(ValueError, match="Point-in-time leakage"):
        M2CamsBiasCorrected(history_df=df_leak, issuance_utc=t0)


# ---------------------------------------------------------------------------
# 2. M3Climatology
# ---------------------------------------------------------------------------


def test_m3_climatology_seasonal_window_and_wrap_around() -> None:
    # Build synthetic multi-month history:
    # Winter around Jan 1 (Dec 25 - Jan 10): high PM2.5 (~200 ug/m3)
    # Summer around Jul 1 (Jun 20 - Jul 10): low PM2.5 (~35 ug/m3)
    records = []
    for day_offset in range(-10, 11):
        dt_winter = datetime(2025, 1, 1, 0, 0, tzinfo=UTC) + timedelta(days=day_offset)
        records.append(
            {
                "ts_utc": dt_winter,
                "available_at_utc": dt_winter + timedelta(hours=3),
                "pm25_ugm3": 180.0 + float(day_offset + 10) * 2.0,
            }
        )
        dt_summer = datetime(2025, 7, 1, 0, 0, tzinfo=UTC) + timedelta(days=day_offset)
        records.append(
            {
                "ts_utc": dt_summer,
                "available_at_utc": dt_summer + timedelta(hours=3),
                "pm25_ugm3": 25.0 + float(day_offset + 10) * 1.0,
            }
        )

    hist_df = pd.DataFrame(records)
    issuance = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    m3 = M3Climatology(history_df=hist_df, window_half_days=15, issuance_utc=issuance)

    qf_winter = m3.predict(target_date=issuance, horizon_h=24)
    assert isinstance(qf_winter, QuantileFunction)
    assert len(qf_winter.q) == 19
    assert np.all(np.diff(qf_winter.q) >= 0.0)
    assert np.all(qf_winter.q >= 0.0)
    # Winter median should be ~200, not polluted by summer (~35)
    assert 185.0 <= float(qf_winter.ppf(0.50)) <= 215.0

    # Summer target date should select summer window (~35)
    qf_summer = m3.predict(target_date=datetime(2026, 7, 1, 0, 0, tzinfo=UTC), horizon_h=24)
    assert 25.0 <= float(qf_summer.ppf(0.50)) <= 45.0

    # Predict all 3 horizons {24, 48, 72}
    h_map = m3.predict_horizons(issuance)
    assert set(h_map.keys()) == {24, 48, 72}
    for h, qf in h_map.items():
        assert isinstance(qf, QuantileFunction), f"Horizon {h} did not return QuantileFunction"


def test_m3_climatology_undefined_at_n0_and_pit_guard() -> None:
    with pytest.raises(ValueError, match="undefined at N=0"):
        M3Climatology(values=[])

    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    df_leak = pd.DataFrame(
        {
            "ts_utc": [t0 - timedelta(days=1)],
            "available_at_utc": [t0 + timedelta(hours=1)],
            "pm25_ugm3": [100.0],
        }
    )
    with pytest.raises(ValueError, match="Point-in-time leakage"):
        M3Climatology(history_df=df_leak, issuance_utc=t0)


# ---------------------------------------------------------------------------
# 3. M3bLinearQuantileRegression
# ---------------------------------------------------------------------------


def test_m3b_linear_quantile_regression_fit_and_monotone_predict() -> None:
    rng = np.random.default_rng(123)
    n = 60
    t0 = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)

    lag_pm25 = rng.uniform(30.0, 200.0, size=n)
    cams_pm25 = rng.uniform(25.0, 180.0, size=n)
    blh_m = rng.uniform(150.0, 1200.0, size=n)
    wind_speed = rng.uniform(0.5, 8.0, size=n)

    # True data-generating process with heteroskedastic noise
    y_true = (
        0.45 * lag_pm25
        + 0.55 * cams_pm25
        - 0.02 * blh_m
        - 2.0 * wind_speed
        + 20.0
        + rng.normal(0.0, 10.0, size=n)
    )
    y_true = np.maximum(5.0, y_true)

    train_df = pd.DataFrame(
        {
            "ts_utc": [t0 - timedelta(days=i) for i in range(1, n + 1)],
            "available_at_utc": [
                t0 - timedelta(days=i) + timedelta(hours=2) for i in range(1, n + 1)
            ],
            "lag_24h_pm25": lag_pm25,
            "cams_pm25": cams_pm25,
            "blh_m": blh_m,
            "wind_speed": wind_speed,
            "y_obs": y_true,
        }
    )

    m3b = M3bLinearQuantileRegression()
    m3b.fit(train_df, issuance_utc=t0)
    assert m3b.is_fitted_
    assert len(m3b.models_) == 19

    # Single-row prediction via dict
    sample = {
        "lag_24h_pm25": 150.0,
        "cams_pm25": 140.0,
        "blh_m": 300.0,
        "wind_speed": 1.5,
    }
    qf = m3b.predict(sample)
    assert isinstance(qf, QuantileFunction)
    assert len(qf.q) == 19
    assert np.all(np.diff(qf.q) >= 0.0), "M3b quantiles must be monotone non-decreasing"
    assert np.all(qf.q >= 0.0), "M3b quantiles must be floored at 0.0"

    # Expected mean at sample: 0.45*150 + 0.55*140 - 0.02*300 - 2*1.5 + 20 = 67.5 + 77 - 6 - 3 + 20 = 155.5
    assert abs(float(qf.ppf(0.50)) - 155.5) < 15.0
    assert float(qf.ppf(0.90)) > float(qf.ppf(0.10))

    # Multi-row prediction with NaN feature handled via median imputation
    test_df = pd.DataFrame(
        [
            sample,
            {"lag_24h_pm25": np.nan, "cams_pm25": 60.0, "blh_m": 900.0, "wind_speed": 5.0},
        ]
    )
    qf_list = m3b.predict(test_df)
    assert isinstance(qf_list, list) and len(qf_list) == 2
    assert float(qf_list[0].ppf(0.50)) > float(qf_list[1].ppf(0.50))


def test_m3b_edge_cases_and_pit_guard() -> None:
    m3b = M3bLinearQuantileRegression(min_samples=5)

    # Unfitted predict raises ValueError
    with pytest.raises(ValueError, match="must be fitted"):
        m3b.predict([1.0, 2.0, 3.0, 4.0])

    # Too few samples raises ValueError
    with pytest.raises(ValueError, match="requires at least 5 valid training samples"):
        m3b.fit(np.ones((3, 4)), np.array([10.0, 20.0, 30.0]))

    # Lookahead leakage raises ValueError
    t0 = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    df_leak = pd.DataFrame(
        {
            "available_at_utc": [t0 + timedelta(hours=1)] * 6,
            "lag_24h_pm25": [50.0] * 6,
            "cams_pm25": [50.0] * 6,
            "blh_m": [500.0] * 6,
            "wind_speed": [2.0] * 6,
            "y_obs": [55.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="Point-in-time leakage"):
        m3b.fit(df_leak, issuance_utc=t0)


# ---------------------------------------------------------------------------
# 4. City-Panel Target & Co-location Hierarchy
# ---------------------------------------------------------------------------


def test_city_panel_target_colocation_hierarchy_and_completeness() -> None:
    t0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    hours = [t0 + timedelta(hours=h) for h in range(24)]

    # Registry with 5 stations forming 3 spatial groups:
    # Group A (<= 50m): station 1001 (Reference=True) + station 1002 (Low-cost=False, 20m away)
    # Group B (<= 50m): station 2001 (Low-cost) + station 2002 (Low-cost, 15m away) + station 2003 (Low-cost, 25m away)
    # Group C (isolated): station 3001 (Low-cost, 5 km away)
    # Station 4001 (isolated): has only 15 non-imputed hours (< 18 required) -> must be excluded!
    registry_df = pd.DataFrame(
        [
            {
                "location_id": 1001,
                "lat": 31.55000,
                "lon": 74.33000,
                "is_reference": True,
                "eligible": True,
            },
            {
                "location_id": 1002,
                "lat": 31.55010,
                "lon": 74.33010,
                "is_reference": False,
                "eligible": True,
            },
            {
                "location_id": 2001,
                "lat": 31.52000,
                "lon": 74.35000,
                "is_reference": False,
                "eligible": True,
            },
            {
                "location_id": 2002,
                "lat": 31.52008,
                "lon": 74.35008,
                "is_reference": False,
                "eligible": True,
            },
            {
                "location_id": 2003,
                "lat": 31.52012,
                "lon": 74.35012,
                "is_reference": False,
                "eligible": True,
            },
            {
                "location_id": 3001,
                "lat": 31.47000,
                "lon": 74.40000,
                "is_reference": False,
                "eligible": True,
            },
            {
                "location_id": 4001,
                "lat": 31.42000,
                "lon": 74.28000,
                "is_reference": False,
                "eligible": True,
            },
        ]
    )

    obs_rows = []
    for h_idx, ts in enumerate(hours):
        # Group A: 1001 (ref) = 100.0, 1002 (low-cost) = 160.0 -> hierarchy MUST pick 100.0 (reference)!
        obs_rows.append({"location_id": 1001, "ts_utc": ts, "pm25_ugm3": 100.0, "imputed": False})
        obs_rows.append({"location_id": 1002, "ts_utc": ts, "pm25_ugm3": 160.0, "imputed": False})
        # Group B: 2001 = 70.0, 2002 = 80.0, 2003 = 150.0 -> low-cost median MUST pick 80.0!
        obs_rows.append({"location_id": 2001, "ts_utc": ts, "pm25_ugm3": 70.0, "imputed": False})
        obs_rows.append({"location_id": 2002, "ts_utc": ts, "pm25_ugm3": 80.0, "imputed": False})
        obs_rows.append({"location_id": 2003, "ts_utc": ts, "pm25_ugm3": 150.0, "imputed": False})
        # Group C: 3001 = 120.0 (20 valid hours, 4 imputed hours -> >= 18 non-imputed, so VALID)
        obs_rows.append(
            {"location_id": 3001, "ts_utc": ts, "pm25_ugm3": 120.0, "imputed": h_idx >= 20}
        )
        # Station 4001: 15 non-imputed hours, 9 imputed hours -> < 18 non-imputed, so INVALID
        obs_rows.append(
            {"location_id": 4001, "ts_utc": ts, "pm25_ugm3": 500.0, "imputed": h_idx >= 15}
        )

    obs_df = pd.DataFrame(obs_rows)
    res = compute_city_panel_target(
        obs_df, registry_df=registry_df, min_valid_stations=3, min_valid_hours=18
    )

    assert res.valid is True
    assert res["valid"] is True
    assert res.n_valid_stations == 3
    assert res.n_raw_valid_stations == 6  # 1001, 1002, 2001, 2002, 2003, 3001 (4001 excluded)
    assert res.n_reference == 1
    assert res.n_low_cost == 2
    assert res.truth_basis == "mixed"

    # Group means should be: Group A = 100.0, Group B = 80.0, Group C = 120.0 -> City mean = 100.0
    assert res.y_city is not None
    assert np.isclose(res.y_city, 100.0)


def test_city_panel_target_requires_at_least_3_valid_groups() -> None:
    t0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    hours = [t0 + timedelta(hours=h) for h in range(24)]

    # Only 2 valid stations
    obs_df = pd.DataFrame(
        [
            {"location_id": loc, "ts_utc": ts, "pm25_ugm3": 90.0, "imputed": False}
            for loc in (1001, 2001)
            for ts in hours
        ]
    )
    res = compute_city_panel_target(obs_df, min_valid_stations=3, min_valid_hours=18)
    assert res.valid is False
    assert res.y_city is None
    assert res.n_valid_stations == 2
    assert res.truth_basis == "insufficient"


def test_block_targets_24_48_72() -> None:
    t0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    # Station 1001: valid for h=24 (24 hrs) and h=48 (18 hrs), invalid for h=72 (10 hrs)
    rows = [
        {
            "location_id": 1001,
            "ts_utc": t0 + timedelta(hours=h),
            "pm25_ugm3": 60.0,
            "imputed": False,
        }
        for h in range(24)
    ]
    rows.extend(
        {
            "location_id": 1001,
            "ts_utc": t0 + timedelta(hours=h),
            "pm25_ugm3": 90.0,
            "imputed": (h - 24) >= 18,
        }
        for h in range(24, 48)
    )
    rows.extend(
        {
            "location_id": 1001,
            "ts_utc": t0 + timedelta(hours=h),
            "pm25_ugm3": 120.0,
            "imputed": False,
        }
        for h in range(48, 58)
    )

    obs_df = pd.DataFrame(rows)
    out = block_targets(obs_df, issuance_utc=t0)
    assert len(out) == 1
    row = out.iloc[0]
    assert bool(row["valid_24"]) is True and np.isclose(row["y_24"], 60.0)
    assert bool(row["valid_48"]) is True and np.isclose(row["y_48"], 90.0)
    assert bool(row["valid_72"]) is False and np.isnan(row["y_72"])
