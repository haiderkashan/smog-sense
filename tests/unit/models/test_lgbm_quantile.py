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
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.lgbm_quantile import (
    DEFAULT_HORIZONS,
    DEFAULT_QUANTILES,
    M5_EXCLUDED_COLUMNS,
    LGBMQuantileEnsemble,
    apply_tabular_feature_group_dropout,
    filter_features,
    fit_cross_fit_oof,
    is_aod_feature,
    is_fire_feature,
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


def test_rearrangement_submodularity_and_monotonicity_property() -> None:
    """Property test for monotone rearrangement (Chernozhukov et al., 2010; T1-06).

    Verifies:
    1. Submodularity: Sorting never increases total pinball loss across all quantile levels.
    2. Monotonicity: Sorted quantiles are strictly non-decreasing along the quantile axis.
    3. Physical bounding: All values are bounded within [0.0, 2000.0] ug/m3.
    4. Exact analytical loss reduction for crossing inversions where y lies in the crossing interval.
    """
    rng = np.random.default_rng(2026)
    taus = DEFAULT_QUANTILES  # 19 quantile levels
    n_levels = len(taus)

    # 1. Randomized submodularity property test over 50 varied crossing profiles
    for _ in range(50):
        # Generate non-monotonic (crossing) predictions
        base_val = rng.uniform(20.0, 400.0)
        # Random scrambled offsets
        raw_offsets = rng.normal(loc=0.0, scale=30.0, size=n_levels)
        raw_q = base_val + np.cumsum(raw_offsets)  # Can easily cross and violate monotonicity

        # Add occasional negative values or extreme outliers to test clipping
        if rng.random() < 0.2:
            raw_q[0] = -25.0
        if rng.random() < 0.2:
            raw_q[-1] = 2500.0

        # True observation
        y_obs = rng.uniform(0.0, 500.0)

        # Loss before rearrangement
        loss_before = sum(pinball_loss(y_obs, raw_q[i], taus[i]) for i in range(n_levels))

        # Apply monotone rearrangement with physical bounding [0, 2000]
        rearranged_q = rearrange_quantiles(raw_q, clip_min=0.0, clip_max=2000.0)

        # Loss after rearrangement (must be <= loss_before for in-bound targets)
        loss_after = sum(pinball_loss(y_obs, rearranged_q[i], taus[i]) for i in range(n_levels))

        # Check submodularity: Sorting never increases pinball loss
        assert loss_after <= loss_before + 1e-9

        # Check monotonicity
        assert np.all(np.diff(rearranged_q) >= 0.0)

        # Check physical bounds
        assert np.all(rearranged_q >= 0.0)
        assert np.all(rearranged_q <= 2000.0)

    # 2. Exact analytical test of the submodular gain on crossing inversion:
    # Let tau_1 = 0.20, tau_2 = 0.80.
    # Suppose q_1 = 150.0 > q_2 = 50.0 (crossed).
    # For any y in [50, 150] (e.g. y = 100):
    # Loss before:
    # rho_0.20(100 - 150) = (1 - 0.20) * 50 = 40.0
    # rho_0.80(100 - 50)  = 0.80 * 50 = 40.0
    # Total before = 80.0
    # Rearranged: q_1* = 50.0, q_2* = 150.0
    # Loss after:
    # rho_0.20(100 - 50)  = 0.20 * 50 = 10.0
    # rho_0.80(100 - 150) = (1 - 0.80) * 50 = 10.0
    # Total after = 20.0
    # Exact reduction = (tau_2 - tau_1) * (q_1 - q_2) = (0.80 - 0.20) * (150 - 50) = 0.60 * 100 = 60.0
    tau_pair = (0.20, 0.80)
    q_crossed = np.array([150.0, 50.0])
    y_test = 100.0

    pb_before = pinball_loss(y_test, q_crossed[0], tau_pair[0]) + pinball_loss(
        y_test, q_crossed[1], tau_pair[1]
    )
    q_fixed = rearrange_quantiles(q_crossed)
    pb_after = pinball_loss(y_test, q_fixed[0], tau_pair[0]) + pinball_loss(
        y_test, q_fixed[1], tau_pair[1]
    )

    np.testing.assert_allclose(pb_before, 80.0)
    np.testing.assert_allclose(pb_after, 20.0)
    np.testing.assert_allclose(pb_before - pb_after, 60.0)


def test_validation_early_stopping_on_plateau() -> None:
    """Early stopping on validation set must stop boosting rounds before max iterations."""
    rng = np.random.default_rng(42)
    n_train = 200
    n_val = 100

    # Training data with random noise targets
    x_train_vals = rng.normal(size=(n_train, 5))
    train_df = pd.DataFrame(
        {
            "domain": ["delhi"] * n_train,
            "location_id": [101] * n_train,
            "issuance_utc": pd.date_range("2023-10-01", periods=n_train, freq="6h", tz="UTC"),
            "f_met_vc_mean_h24": x_train_vals[:, 0],
            "f_met_rh_mean_h24": x_train_vals[:, 1],
            "f_cams_pm25_h24": x_train_vals[:, 2],
            "f_fire_exposure_e24": x_train_vals[:, 3],
            "f_aod_mean_h24": x_train_vals[:, 4],
            "y_h24": rng.normal(size=n_train),
            "valid_h24": [True] * n_train,
        }
    )

    # Validation data with independent noise targets (will plateau / degrade early)
    x_val_vals = rng.normal(size=(n_val, 5))
    val_df = pd.DataFrame(
        {
            "domain": ["delhi"] * n_val,
            "location_id": [101] * n_val,
            "issuance_utc": pd.date_range("2023-12-01", periods=n_val, freq="6h", tz="UTC"),
            "f_met_vc_mean_h24": x_val_vals[:, 0],
            "f_met_rh_mean_h24": x_val_vals[:, 1],
            "f_cams_pm25_h24": x_val_vals[:, 2],
            "f_fire_exposure_e24": x_val_vals[:, 3],
            "f_aod_mean_h24": x_val_vals[:, 4],
            "y_h24": rng.normal(size=n_val),
            "valid_h24": [True] * n_val,
        }
    )

    ensemble = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=(0.10, 0.50, 0.90),
        horizons=(24,),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5, "learning_rate": 0.1},
        fire_dropout_p=0.0,
        aod_dropout_p=0.0,
    )

    max_rounds = 100
    patience = 5
    ensemble.fit(
        train_df,
        val_df=val_df,
        num_boost_round=max_rounds,
        early_stopping_rounds=patience,
    )

    # Every booster must have stopped well before max_rounds
    for q in (0.10, 0.50, 0.90):
        best_it = ensemble.best_iterations[24][q]
        assert best_it < max_rounds, (
            f"Quantile {q} failed to early stop: best_iteration={best_it} >= {max_rounds}"
        )


def test_tabular_feature_group_dropout_unit() -> None:
    """Tabular feature group dropout must mask fire and AOD columns with target probability."""
    rng = np.random.default_rng(1234)
    n_rows = 1000

    df = pd.DataFrame(
        {
            "f_met_vc_mean_h24": rng.uniform(100, 2000, size=n_rows),
            "f_fire_exposure_e24": rng.uniform(0, 50, size=n_rows),
            "m_fire_exposure_e24": [False] * n_rows,
            "f_aod_mean_h24": rng.uniform(0.1, 1.5, size=n_rows),
            "m_aod_mean_h24": [False] * n_rows,
            "f_stat_elevation_m": rng.uniform(100, 300, size=n_rows),
        }
    )

    features = list(df.columns)
    assert is_fire_feature("f_fire_exposure_e24")
    assert is_fire_feature("m_fire_exposure_e24")
    assert not is_fire_feature("f_met_vc_mean_h24")
    assert is_aod_feature("f_aod_mean_h24")
    assert not is_aod_feature("f_stat_elevation_m")

    # With p=0.0: strictly unchanged
    df_no_drop = apply_tabular_feature_group_dropout(
        df, features, fire_dropout_p=0.0, aod_dropout_p=0.0
    )
    pd.testing.assert_frame_equal(df, df_no_drop)

    # With p_fire=0.2, p_aod=0.3
    df_dropped = apply_tabular_feature_group_dropout(
        df, features, fire_dropout_p=0.20, aod_dropout_p=0.30, rng=rng
    )

    # Met and stat features must NEVER be dropped
    assert df_dropped["f_met_vc_mean_h24"].isna().sum() == 0
    assert df_dropped["f_stat_elevation_m"].isna().sum() == 0

    # Fire features should be dropped in approx 20% of rows (binomial 99.9% CI: [15%, 26%])
    fire_nan_rate = df_dropped["f_fire_exposure_e24"].isna().mean()
    assert 0.15 <= fire_nan_rate <= 0.26

    # Mask column must match dropped rows
    assert np.all(df_dropped.loc[df_dropped["f_fire_exposure_e24"].isna(), "m_fire_exposure_e24"])

    # AOD features should be dropped in approx 30% of rows (binomial 99.9% CI: [24%, 36%])
    aod_nan_rate = df_dropped["f_aod_mean_h24"].isna().mean()
    assert 0.24 <= aod_nan_rate <= 0.36


def test_feature_masking_operational_degradation_stability() -> None:
    """Ensemble trained with feature dropout must predict stably even if satellite feeds fail."""
    df_train = _create_synthetic_training_data(n_samples=150)
    df_test = _create_synthetic_training_data(n_samples=50)

    ensemble = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=DEFAULT_QUANTILES,
        horizons=DEFAULT_HORIZONS,
        lgbm_params={"num_leaves": 15, "min_data_in_leaf": 5},
        fire_dropout_p=0.1,
        aod_dropout_p=0.1,
    )
    ensemble.fit(df_train, num_boost_round=20)

    # 1. Clean test predictions
    clean_preds = ensemble.predict(df_test, horizon=24, rearrange=True)
    assert clean_preds.shape == (len(df_test), 19)
    assert np.all(np.isfinite(clean_preds))
    assert np.all(np.diff(clean_preds, axis=-1) >= 0.0)

    # 2. FIRMS fire feed unavailable (all fire features NaN)
    df_no_fire = df_test.copy()
    fire_cols = [c for c in ensemble.feature_names if is_fire_feature(c)]
    df_no_fire[fire_cols] = np.nan
    no_fire_preds = ensemble.predict(df_no_fire, horizon=24, rearrange=True)
    assert no_fire_preds.shape == (len(df_test), 19)
    assert np.all(np.isfinite(no_fire_preds))
    assert np.all(np.diff(no_fire_preds, axis=-1) >= 0.0)
    assert np.all(no_fire_preds >= 0.0)
    assert np.all(no_fire_preds <= 2000.0)

    # 3. CAMS AOD feed unavailable (all AOD features NaN)
    df_no_aod = df_test.copy()
    aod_cols = [c for c in ensemble.feature_names if is_aod_feature(c)]
    df_no_aod[aod_cols] = np.nan
    no_aod_preds = ensemble.predict(df_no_aod, horizon=24, rearrange=True)
    assert no_aod_preds.shape == (len(df_test), 19)
    assert np.all(np.isfinite(no_aod_preds))
    assert np.all(np.diff(no_aod_preds, axis=-1) >= 0.0)

    # 4. Both FIRMS and CAMS AOD feeds unavailable (combined degradation)
    df_degraded = df_test.copy()
    df_degraded[fire_cols + aod_cols] = np.nan
    degraded_preds = ensemble.predict(df_degraded, horizon=24, rearrange=True)
    assert degraded_preds.shape == (len(df_test), 19)
    assert np.all(np.isfinite(degraded_preds))
    assert np.all(np.diff(degraded_preds, axis=-1) >= 0.0)
    assert np.all(degraded_preds >= 0.0)


def test_deterministic_reproducibility_and_bundle_tamper_detection(tmp_path: Path) -> None:
    """Ensemble must reproduce exact byte-for-byte SHA256 and detect file tampering on disk."""
    df = _create_synthetic_training_data(n_samples=100)

    ens1 = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=(0.10, 0.50, 0.90),
        horizons=(24,),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5, "random_state": 42},
    )
    ens1.fit(df, num_boost_round=10)
    sha1 = ens1.compute_base_sha256()

    ens2 = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=(0.10, 0.50, 0.90),
        horizons=(24,),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5, "random_state": 42},
    )
    ens2.fit(df, num_boost_round=10)
    sha2 = ens2.compute_base_sha256()

    # Exact deterministic match
    assert sha1 == sha2
    np.testing.assert_array_equal(
        ens1.predict(df, horizon=24, rearrange=True),
        ens2.predict(df, horizon=24, rearrange=True),
    )

    # Different random seed produces different model SHA256
    ens3 = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=(0.10, 0.50, 0.90),
        horizons=(24,),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5, "random_state": 999},
    )
    ens3.fit(df, num_boost_round=10)
    sha3 = ens3.compute_base_sha256()
    assert sha1 != sha3

    # Tamper detection test (docs/test-specifications.md §T1-15)
    save_dir = ens1.save(tmp_path / "tamper_test_bundle")
    booster_file = save_dir / "booster_h24_q50.txt"
    assert booster_file.exists()

    # Intentionally corrupt the booster file on disk
    with booster_file.open("a") as f:
        f.write("# TAMPERED CONTENT\n")

    # Loading with verify_sha256=True must raise ValueError
    with pytest.raises(ValueError, match="Model bundle integrity verification failed"):
        LGBMQuantileEnsemble.load(save_dir, verify_sha256=True)

    # Loading with verify_sha256=False permits loading
    loaded_unverified = LGBMQuantileEnsemble.load(save_dir, verify_sha256=False)
    assert loaded_unverified.model_type == "M5"


def test_cli_train_lgbm_options() -> None:
    """CLI train lgbm command must expose all required flags for standard and OOF cross-fit."""
    runner = CliRunner()
    result = runner.invoke(app, ["train", "lgbm", "--help"])
    assert result.exit_code == 0
    assert "--domain" in result.stdout
    assert "--split" in result.stdout
    assert "--oof" in result.stdout
    assert "--out-dir" in result.stdout
    assert "--num-boost-round" in result.stdout
    assert "--feature-store-dir" in result.stdout
