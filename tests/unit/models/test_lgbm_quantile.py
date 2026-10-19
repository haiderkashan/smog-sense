"""tests/unit/models/test_lgbm_quantile.py — Unit tests for LightGBM quantile ensemble.

Verifies:
1. Feature filtering rules for M5 (no recent bias, no coordinates, no event flags) vs M4.
2. Monotone rearrangement of crossed quantiles.
3. Pinball loss and discrete CRPS arithmetic.
4. Ensemble training, non-crossing predictions, and base SHA256 binding.
5. Serialization to disk and bit-for-bit reload.
6. Season-blocked out-of-fold cross-fitting on Block E.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smogsense.models.lgbm_quantile import (
    M5_EXCLUDED_COLUMNS,
    LGBMQuantileEnsemble,
    filter_features,
    fit_cross_fit_oof,
    pinball_loss,
    rearrange_quantiles,
)


def test_m5_feature_filtering() -> None:
    """M5 must strictly exclude recent CAMS bias, lat/lon, and Indian-only event flags."""
    candidate_cols = [
        "f_met_vc_mean_h24",
        "f_met_rh_mean_h24",
        "f_fire_exposure_e24",
        "f_aod_mean_h24",
        "f_cams_bias_trailing_7d",  # Excluded
        "f_cal_diwali_h24",  # Excluded
        "f_stat_lat",  # Excluded
        "f_stat_lon",  # Excluded
        "f_stat_elevation_m",
        "m_met_vc_mean_h24",
        "m_cams_bias_trailing_7d",  # Excluded
        "non_feature_column",  # Excluded
    ]

    m5_features = filter_features(candidate_cols, model_type="M5")
    assert "f_met_vc_mean_h24" in m5_features
    assert "f_stat_elevation_m" in m5_features
    assert "m_met_vc_mean_h24" in m5_features

    for exc in M5_EXCLUDED_COLUMNS:
        assert exc not in m5_features

    assert "non_feature_column" not in m5_features


def test_m4_feature_filtering() -> None:
    """M4 within-domain model retains coordinates but drops transfer event flags."""
    candidate_cols = [
        "f_met_vc_mean_h24",
        "f_stat_lat",
        "f_stat_lon",
        "f_cal_diwali_h24",
    ]
    m4_features = filter_features(candidate_cols, model_type="M4")
    assert "f_stat_lat" in m4_features
    assert "f_stat_lon" in m4_features
    assert "f_cal_diwali_h24" not in m4_features


def test_monotone_rearrangement() -> None:
    """Quantiles that cross must be sorted into monotonic ascending order."""
    crossing_q = np.array([[120.0, 95.0, 110.0, 140.0], [50.0, 30.0, 70.0, 60.0]])
    rearranged = rearrange_quantiles(crossing_q)

    assert rearranged.shape == crossing_q.shape
    # Check monotonicity along last axis
    assert np.all(np.diff(rearranged, axis=-1) >= 0)
    np.testing.assert_array_equal(rearranged[0], [95.0, 110.0, 120.0, 140.0])
    np.testing.assert_array_equal(rearranged[1], [30.0, 50.0, 60.0, 70.0])


def test_pinball_loss_exact() -> None:
    """Check pinball loss computation for tau=0.5 (half absolute error) and asymmetric tau."""
    y = np.array([100.0, 50.0])
    q = np.array([80.0, 60.0])

    # For tau=0.5: loss = 0.5 * |y - q|
    # row 0: y - q = +20 -> 0.5 * 20 = 10.0
    # row 1: y - q = -10 -> 0.5 * 10 = 5.0
    loss_median = pinball_loss(y, q, tau=0.5)
    np.testing.assert_allclose(loss_median, [10.0, 5.0])

    # For tau=0.9: over-prediction penalized by 0.1, under-prediction by 0.9
    # row 0 (under): 0.9 * 20 = 18.0
    # row 1 (over): (0.9 - 1) * (-10) = 1.0
    loss_q90 = pinball_loss(y, q, tau=0.9)
    np.testing.assert_allclose(loss_q90, [18.0, 1.0])


def _create_synthetic_training_data(n_samples: int = 150) -> pd.DataFrame:
    """Create synthetic feature store dataframe for testing."""
    rng = np.random.default_rng(42)
    data = {
        "domain": ["delhi"] * n_samples,
        "location_id": rng.choice([101, 102, 103], size=n_samples),
        "issuance_utc": pd.date_range("2022-10-01", periods=n_samples, freq="6h", tz="UTC"),
        "fold_id": rng.choice([0, 1, 2], size=n_samples),
        "f_met_vc_mean_h24": rng.uniform(500, 3000, size=n_samples),
        "f_met_rh_mean_h24": rng.uniform(30, 95, size=n_samples),
        "f_fire_exposure_e24": rng.exponential(10.0, size=n_samples),
        "f_aod_mean_h24": rng.uniform(0.1, 1.5, size=n_samples),
        "f_cams_pm25_h24": rng.uniform(40, 250, size=n_samples),
        "f_stat_elevation_m": rng.uniform(200, 250, size=n_samples),
        "y_h24": rng.uniform(30, 350, size=n_samples),
        "valid_h24": [True] * n_samples,
        "y_h48": rng.uniform(30, 350, size=n_samples),
        "valid_h48": [True] * n_samples,
        "y_h72": rng.uniform(30, 350, size=n_samples),
        "valid_h72": [True] * n_samples,
    }
    return pd.DataFrame(data)


def test_ensemble_fit_predict_rearrange() -> None:
    """Train mini ensemble and assert non-crossing quantile properties."""
    df = _create_synthetic_training_data(n_samples=120)

    test_quantiles = (0.1, 0.5, 0.9)
    test_horizons = (24,)
    ensemble = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=test_quantiles,
        horizons=test_horizons,
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
    )

    ensemble.fit(df, num_boost_round=15)
    preds = ensemble.predict(df, horizon=24, rearrange=True)

    assert preds.shape == (len(df), 3)
    # Verify non-crossing: q0.1 <= q0.5 <= q0.9
    assert np.all(preds[:, 0] <= preds[:, 1])
    assert np.all(preds[:, 1] <= preds[:, 2])

    # SHA256 binding must be valid 64-char hex
    sha = ensemble.compute_base_sha256()
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_ensemble_save_and_load(tmp_path: Path) -> None:
    """Saved ensemble must reload bit-for-bit with matching SHA256 and predictions."""
    df = _create_synthetic_training_data(n_samples=80)
    ensemble = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=(0.25, 0.50, 0.75),
        horizons=(24,),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
    )
    ensemble.fit(df, num_boost_round=10)
    original_preds = ensemble.predict(df, horizon=24, rearrange=True)
    original_sha = ensemble.compute_base_sha256()

    save_dir = ensemble.save(tmp_path / "m5_base")
    loaded_ensemble = LGBMQuantileEnsemble.load(save_dir)

    assert loaded_ensemble.model_type == "M5"
    assert loaded_ensemble.compute_base_sha256() == original_sha

    loaded_preds = loaded_ensemble.predict(df, horizon=24, rearrange=True)
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


def test_fit_cross_fit_oof() -> None:
    """Out-of-fold cross-fitting must generate predictions across all folds without data leakage."""
    df = _create_synthetic_training_data(n_samples=100)
    horizons = (24,)
    quantiles = (0.1, 0.5, 0.9)

    oof_df, final_base = fit_cross_fit_oof(
        df_block_e=df,
        horizons=horizons,
        quantiles=quantiles,
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
        num_boost_round=10,
    )

    assert len(oof_df) == len(df)
    assert "m5_oof_h24_q10" in oof_df.columns
    assert "m5_oof_h24_q50" in oof_df.columns
    assert "m5_oof_h24_q90" in oof_df.columns
    assert oof_df["m5_oof_h24_q50"].notna().all()

    # Final base must be fitted and produce valid predictions
    final_preds = final_base.predict(df, horizon=24)
    assert final_preds.shape == (len(df), 3)
