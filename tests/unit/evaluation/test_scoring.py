"""Unit tests for probabilistic proper scoring rules (scoring.py and crps.py)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from smogsense.evaluation.crps import (
    crps_from_quantiles,
    threshold_weighted_crps,
    tw_crps,
)
from smogsense.evaluation.scoring import (
    CANONICAL_TAUS,
    compute_crps,
    compute_tw_crps,
    crpss,
    interval_score,
    pinball_loss,
    pinball_loss_19,
    pinball_loss_vectorized,
    point_crps,
    score_forecast_block,
)


def test_pinball_loss_scalar_and_properties() -> None:
    # Under-prediction: y > q
    # rho_0.9(100 - 80) = 0.9 * 20 = 18.0
    loss_under = pinball_loss(100.0, 80.0, tau=0.90)
    assert np.isclose(loss_under, 18.0)

    # Over-prediction: y < q
    # rho_0.9(80 - 100) = (0.9 - 1) * (-20) = 2.0
    loss_over = pinball_loss(80.0, 100.0, tau=0.90)
    assert np.isclose(loss_over, 2.0)

    # Ratio of penalties at tau=0.9 is 9:1
    assert np.isclose(loss_under / loss_over, 9.0)

    # Exact prediction: y == q -> 0.0
    assert pinball_loss(100.0, 100.0, tau=0.50) == 0.0


def test_pinball_loss_vectorized() -> None:
    y = np.array([50.0, 100.0, 150.0])
    q = np.array([60.0, 100.0, 130.0])
    taus = np.array([0.10, 0.50, 0.90])

    losses = pinball_loss_vectorized(y, q, taus)
    assert isinstance(losses, np.ndarray)
    assert losses.shape == (3,)
    assert (losses >= 0.0).all()

    # Scalar check via vectorized
    s_loss = pinball_loss_vectorized(100.0, 80.0, 0.90)
    assert isinstance(s_loss, float)
    assert np.isclose(s_loss, 18.0)


def test_pinball_loss_19() -> None:
    # 19 quantiles
    taus = CANONICAL_TAUS
    q19 = norm.ppf(taus, loc=100.0, scale=20.0)

    # Scalar observation
    mean_pb = pinball_loss_19(100.0, q19)
    assert isinstance(mean_pb, float)
    assert mean_pb > 0.0

    # Batch observations
    y_batch = np.array([80.0, 100.0, 120.0])
    q_batch = np.tile(q19, (3, 1))
    batch_pb = pinball_loss_19(y_batch, q_batch)
    assert isinstance(batch_pb, np.ndarray)
    assert batch_pb.shape == (3,)
    assert (batch_pb > 0.0).all()

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError, match="exactly 19 columns"):
        pinball_loss_19(100.0, np.ones(10))


def test_interval_score_central_80_and_50() -> None:
    # y inside interval [80, 120], alpha = 0.20 (80% interval)
    # IS = width = 120 - 80 = 40.0
    is_inside = interval_score(100.0, lower=80.0, upper=120.0, alpha=0.20)
    assert np.isclose(is_inside, 40.0)

    # y below lower: y = 70.0
    # penalty = (2 / 0.20) * (80 - 70) = 10 * 10 = 100.0
    # IS = 40.0 + 100.0 = 140.0
    is_below = interval_score(70.0, lower=80.0, upper=120.0, alpha=0.20)
    assert np.isclose(is_below, 140.0)

    # y above upper: y = 135.0
    # penalty = (2 / 0.20) * (135 - 120) = 10 * 15 = 150.0
    # IS = 40.0 + 150.0 = 190.0
    is_above = interval_score(135.0, lower=80.0, upper=120.0, alpha=0.20)
    assert np.isclose(is_above, 190.0)

    # Vectorized array input
    y_arr = np.array([100.0, 70.0, 135.0])
    l_arr = np.array([80.0, 80.0, 80.0])
    u_arr = np.array([120.0, 120.0, 120.0])
    is_vec = interval_score(y_arr, l_arr, u_arr, alpha=0.20)
    assert isinstance(is_vec, np.ndarray)
    assert np.allclose(is_vec, [40.0, 140.0, 190.0])


def test_point_crps_and_crpss() -> None:
    # MAE point crps
    assert point_crps(100.0, 95.0) == 5.0
    assert np.allclose(
        point_crps(np.array([100.0, 50.0]), np.array([90.0, 55.0])),
        np.array([10.0, 5.0]),
    )

    # CRPSS
    assert np.isclose(crpss(crps_model=15.0, crps_ref=20.0), 0.25)
    assert np.isclose(crpss(crps_model=20.0, crps_ref=20.0), 0.0)
    assert np.isclose(crpss(crps_model=25.0, crps_ref=20.0), -0.25)


def test_threshold_weighted_crps_properties() -> None:
    taus = CANONICAL_TAUS
    # Non-crossing quantiles
    q19 = norm.ppf(taus, loc=100.0, scale=20.0)

    # For threshold = 125.5
    tw125 = threshold_weighted_crps(q19, observation=140.0, threshold=125.5)
    assert isinstance(tw125, float)
    assert tw125 >= 0.0

    # Aliasing check
    assert threshold_weighted_crps == tw_crps

    # When both observation and distribution are far below threshold (e.g. threshold = 300),
    # twCRPS should be 0.0 because v(x) = max(x, 300) = 300 for all x and y
    tw_zero = threshold_weighted_crps(q19, observation=50.0, threshold=300.0)
    assert np.isclose(tw_zero, 0.0, atol=1e-5)

    # Batch evaluation
    q_batch = np.array([q19, q19 + 50.0])
    obs_batch = np.array([130.0, 180.0])
    tw_batch = threshold_weighted_crps(q_batch, obs_batch, threshold=125.5)
    assert isinstance(tw_batch, np.ndarray)
    assert tw_batch.shape == (2,)
    assert (tw_batch >= 0.0).all()

    # compute_tw_crps helper
    val = compute_tw_crps(q19, 140.0, threshold=125.5)
    assert np.isclose(val, tw125)

    # crps_from_quantiles vs compute_crps
    c1 = crps_from_quantiles(q19, 140.0)
    c2 = compute_crps(q19, 140.0)
    assert np.isclose(c1, c2)
    assert c1 > tw125


def test_score_forecast_block_completeness_rule() -> None:
    forecast_row = {
        "issuance_utc": "2026-11-06T00:00:00Z",
        "domain": "lahore",
        "point_id": "8664",
        "horizon_h": 24,
        "method": "m5_lgbm_source_zero_shot",
        "q10": 80.0,
        "q50": 110.0,
        "q90": 160.0,
        "run_id": "test_run_01",
    }
    # Add all 19 quantiles
    for idx, tau in enumerate(CANONICAL_TAUS):
        col = f"q{round(tau * 100):02d}"
        forecast_row[col] = 60.0 + idx * 6.0

    # Case 1: valid hours < 18 (e.g. 15 valid hours)
    obs_incomplete = pd.Series([100.0] * 15 + [np.nan] * 9)
    res_incomplete = score_forecast_block(forecast_row, obs_incomplete, stage="settled")
    assert res_incomplete["valid"] is False
    assert res_incomplete["n_valid_hours"] == 15
    assert res_incomplete["y_obs"] is None
    assert res_incomplete["crps"] is None
    assert res_incomplete["pinball_q10"] is None

    # Case 2: valid hours >= 18 (e.g. 20 valid hours)
    obs_complete = pd.Series([110.0] * 20 + [np.nan] * 4)
    res_complete = score_forecast_block(forecast_row, obs_complete, stage="settled")
    assert res_complete["valid"] is True
    assert res_complete["n_valid_hours"] == 20
    assert np.isclose(res_complete["y_obs"], 110.0)
    assert res_complete["crps"] is not None
    assert res_complete["crps"] > 0.0
    assert res_complete["tw_crps_125_5"] is not None
    assert res_complete["pinball_q50"] is not None
    assert res_complete["interval_score_80"] is not None
    assert res_complete["obs_category"] is not None
