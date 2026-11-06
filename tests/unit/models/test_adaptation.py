"""tests/unit/models/test_adaptation.py — Unit tests for deployment adaptation engine.

Specification:
- docs/ml-architecture.md §6, §8 (Deployment adaptation, Condition C5, C6, C7, C8)
- docs/test-specifications.md §T1-16 (Adaptation Outcomes) & §T1-17 (Peak RSS Memory Guard)
- decisions.md (ADR-002, ADR-003, ADR-015, ADR-024)

Verifies:
1. Canonical adaptation outcomes across all 5 branches (T1-16):
   - skipped_n0
   - skipped_insufficient_support
   - rejected_no_gain
   - rejected_residual_bound
   - adapted
2. Fallback invariant: All skipped or rejected outcomes yield Delta z = 0,
   reproducing Model M5 predictions bit-for-bit with degradation level 0.
3. Chronological support window embargo integrity:
   72-hour temporal separation between fit and check windows.
4. Memory RSS and graph hygiene (T1-17):
   Bounded memory execution, no retained autograd graphs or leaking tensors.
5. Adaptation step size modes:
   Scalar float (Model M6) vs MetaSGDStepSizes (Model M7).
6. Bundle loading and cryptographic base_sha256 tamper verification.
7. CLI command surface integration for 'smogsense evaluate adapt'.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.adaptation import (
    DEFAULT_RHO_99,
    DeploymentAdaptationEngine,
    get_process_rss_mb,
)
from smogsense.models.encoders.gru import LocalGRUEncoder
from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
    ResidualQuantileHead,
)
from smogsense.models.lgbm_quantile import LGBMQuantileEnsemble
from smogsense.models.maml import MetaSGDStepSizes
from smogsense.models.residual_bound import split_support_window


def _create_synthetic_dataset(
    n_days: int = 14,
    freq: str = "6h",
    n_stations: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic feature store dataframe for testing deployment adaptation."""
    rng = np.random.default_rng(seed)
    station_ids = [8664 + i for i in range(n_stations)]
    n_steps = n_days * 4  # 4 steps per day at 6h freq
    dates = pd.date_range("2024-11-01", periods=n_steps, freq=freq, tz="UTC")

    dfs: list[pd.DataFrame] = []
    for sid in station_ids:
        n_obs = len(dates)
        df_stn = pd.DataFrame(
            {
                "domain": ["lahore"] * n_obs,
                "location_id": [sid] * n_obs,
                "issuance_utc": dates,
                "source_split": ["V"] * n_obs,
                "season": ["2024-2025"] * n_obs,
                "f_stat_latitude": [31.52] * n_obs,
                "f_stat_longitude": [74.35] * n_obs,
                "f_met_vc_mean_h24": rng.uniform(500, 2500, size=n_obs),
                "f_met_rh_mean_h24": rng.uniform(30, 90, size=n_obs),
                "f_fire_exposure_e24": rng.exponential(5.0, size=n_obs),
                "f_aod_mean_h24": rng.uniform(0.1, 1.2, size=n_obs),
                "f_cams_pm25_h24": rng.uniform(40, 250, size=n_obs),
                "f_stat_elevation_m": [217.0] * n_obs,
                "f_cams_bias_trailing_7d": rng.uniform(-10, 10, size=n_obs),
                "f_met_ln_vc_mean_h24": rng.uniform(6.0, 7.5, size=n_obs),
                "f_met_ln_vc_mean_h48": rng.uniform(6.0, 7.5, size=n_obs),
                "f_met_ln_vc_mean_h72": rng.uniform(6.0, 7.5, size=n_obs),
                "y_h24": rng.uniform(50, 200, size=n_obs),
                "valid_h24": [True] * n_obs,
                "y_h48": rng.uniform(50, 200, size=n_obs),
                "valid_h48": [True] * n_obs,
                "y_h72": rng.uniform(50, 200, size=n_obs),
                "valid_h72": [True] * n_obs,
            }
        )
        dfs.append(df_stn)

    return pd.concat(dfs, ignore_index=True).sort_values("issuance_utc").reset_index(drop=True)


@pytest.fixture(scope="module")
def shared_base_model() -> LGBMQuantileEnsemble:
    """Train a fast mini LightGBM ensemble."""
    df_train = _create_synthetic_dataset(n_days=10, seed=123)
    base_model = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=CANONICAL_QUANTILES,
        horizons=CANONICAL_HORIZONS,
        lgbm_params={
            "n_estimators": 3,
            "min_child_samples": 5,
            "learning_rate": 0.1,
            "verbose": -1,
        },
    )
    base_model.fit(df_train)
    return base_model


@pytest.fixture
def fresh_calibrator() -> ResidualCalibrator:
    """Initialize a fresh ResidualCalibrator per test."""
    encoder = LocalGRUEncoder(input_size=6, hidden_size=64, num_layers=2, latent_dim=32)
    head = ResidualQuantileHead(input_dim=46, hidden_dim=64, output_dim=57)
    return ResidualCalibrator(encoder, head)


def test_support_window_embargo_integrity() -> None:
    """Verify chronological split enforces 72-hour embargo and non-overlap (Condition C6)."""
    df = _create_synthetic_dataset(n_days=14, seed=42)
    df_fit, df_check = split_support_window(df, embargo_hours=72, check_fraction=0.35)

    assert not df_fit.empty
    assert not df_check.empty

    t_fit_max = pd.to_datetime(df_fit["issuance_utc"].max(), utc=True)
    t_check_min = pd.to_datetime(df_check["issuance_utc"].min(), utc=True)

    # Strictly disjoint
    assert set(df_fit["issuance_utc"]).isdisjoint(set(df_check["issuance_utc"]))

    # Embargo buffer strictly >= 72 hours
    embargo_gap_h = (t_check_min - t_fit_max).total_seconds() / 3600.0
    assert embargo_gap_h >= 72.0


def test_adaptation_status_skipped_n0(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-16 Branch 1: Sensor-free mode (N=0) yields skipped_n0 and exact M5 fallback."""
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        rho_99=DEFAULT_RHO_99,
    )

    empty_df = pd.DataFrame()
    res = engine.adapt(support_df=empty_df, n_available=0)

    assert res.status == "skipped_n0"
    assert engine.is_adapted is False
    assert engine.adaptation_status == "skipped_n0"

    # Predictions match M5 bit-for-bit
    test_df = _create_synthetic_dataset(n_days=2, seed=999)
    res_m5 = shared_base_model.predict_all_horizons(test_df, rearrange=True)
    res_engine = engine.predict(test_df, rearrange=True)

    for h in CANONICAL_HORIZONS:
        np.testing.assert_allclose(res_engine[h], res_m5[h], atol=1e-5)

    delta_z = engine.predict_residual(test_df)
    np.testing.assert_array_equal(delta_z, 0.0)


def test_adaptation_status_skipped_insufficient_support(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-16 Branch 2: Support duration < 7 days yields skipped_insufficient_support."""
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        min_support_days=7.0,
    )

    short_df = _create_synthetic_dataset(n_days=3, seed=42)
    res = engine.adapt(support_df=short_df)

    assert res.status == "skipped_insufficient_support"
    assert engine.is_adapted is False
    assert engine.adaptation_status == "skipped_insufficient_support"

    delta_z = engine.predict_residual(short_df)
    np.testing.assert_array_equal(delta_z, 0.0)


def test_adaptation_status_rejected_no_gain(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-16 Branch 3: Lack of check window improvement yields rejected_no_gain."""
    # Setting min_improvement unreasonably high (e.g. 99%) forces rejection
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        min_improvement=0.99,
        rho_99=DEFAULT_RHO_99,
    )

    df_support = _create_synthetic_dataset(n_days=14, seed=42)
    res = engine.adapt(support_df=df_support)

    assert res.status == "rejected_no_gain"
    assert engine.is_adapted is False
    assert engine.adaptation_status == "rejected_no_gain"

    # Delta z is strictly 0 and M5 preserved
    delta_z = engine.predict_residual(df_support)
    np.testing.assert_array_equal(delta_z, 0.0)


def test_adaptation_status_rejected_residual_bound(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-16 Branch 4: Non-trivial residual exceeding rho_99 yields rejected_residual_bound."""
    # Set rho_99 near zero to trigger residual bound rejection when gain is accepted
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        min_improvement=-1e6,  # Any loss change is accepted on improvement
        min_days_better_fraction=0.0,
        rho_99=1e-8,  # Tiny threshold ensures R > rho_99
    )

    df_support = _create_synthetic_dataset(n_days=14, seed=42)
    res = engine.adapt(support_df=df_support)

    assert res.status == "rejected_residual_bound"
    assert engine.is_adapted is False
    assert engine.adaptation_status == "rejected_residual_bound"

    # Invariant: rejected outcome enforces Delta z = 0 identically
    delta_z = engine.predict_residual(df_support)
    np.testing.assert_array_equal(delta_z, 0.0)


def test_adaptation_status_adapted_pass(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-16 Branch 5: Meets improvement, daily win, and residual bound -> adapted."""
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        min_improvement=-1e6,  # Relaxed acceptance thresholds for synthetic data
        min_days_better_fraction=0.0,
        rho_99=10.0,  # Generous bound
    )

    df_support = _create_synthetic_dataset(n_days=14, seed=42)
    res = engine.adapt(support_df=df_support, is_reference_sensor=True)

    assert res.status == "adapted"
    assert engine.is_adapted is True
    assert engine.adaptation_status == "adapted"

    # Residuals can now be evaluated
    test_df = _create_synthetic_dataset(n_days=2, seed=888)
    delta_z = engine.predict_residual(test_df)
    assert delta_z.shape == (len(test_df), 3, 19)

    preds = engine.predict(test_df, rearrange=True)
    assert len(preds) == 3
    for h in CANONICAL_HORIZONS:
        assert preds[h].shape == (len(test_df), 19)


def test_memory_rss_hygiene_loop(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """T1-17: Repeated adaptations in a loop do not accumulate memory or autograd graphs."""
    engine = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        inner_steps=3,
        rho_99=DEFAULT_RHO_99,
    )

    df_support = _create_synthetic_dataset(n_days=14, seed=42)

    # Warmup
    for _ in range(2):
        engine.adapt(df_support)
    gc.collect()

    initial_objects = len(gc.get_objects())
    initial_rss = get_process_rss_mb()

    # Run 10 adaptation cycles
    for i in range(10):
        engine.adapt(df_support, seed=42 + i)

    gc.collect()
    final_objects = len(gc.get_objects())
    final_rss = get_process_rss_mb()

    # Bounded object delta (no unbounded graph accumulation)
    object_delta = abs(final_objects - initial_objects)
    assert object_delta < 1500, f"Object count grew excessively by {object_delta}"

    # If RSS tracking is available on the OS, RSS delta must be bounded (< 100 MB)
    if initial_rss > 0.0 and final_rss > 0.0:
        rss_delta_mb = final_rss - initial_rss
        assert rss_delta_mb < 100.0, f"Process RSS grew excessively by {rss_delta_mb:.2f} MB"


def test_step_sizes_modes_m6_vs_m7(
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """Verify scalar float (M6) vs MetaSGDStepSizes (M7) vs invalid step sizes."""
    # 1. Model M6: scalar float alpha
    engine_m6 = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        step_sizes=0.01,
    )
    resolved_m6 = engine_m6._resolve_step_sizes()
    for name in [n for n, _ in fresh_calibrator.named_parameters()]:
        assert name in resolved_m6
        assert resolved_m6[name].numel() == 1
        np.testing.assert_allclose(resolved_m6[name].cpu().numpy(), 0.01)

    # 2. Model M7: MetaSGDStepSizes
    meta_sgd = MetaSGDStepSizes(init_alpha=0.01)
    engine_m7 = DeploymentAdaptationEngine(
        base_model=shared_base_model,
        calibrator=fresh_calibrator,
        step_sizes=meta_sgd,
    )
    resolved_m7 = engine_m7._resolve_step_sizes()
    for name in [n for n, _ in fresh_calibrator.named_parameters()]:
        assert name in resolved_m7
        assert resolved_m7[name].numel() == 1

    # 3. Invalid step size type raises TypeError
    with pytest.raises(TypeError, match="Unsupported step_sizes type"):
        DeploymentAdaptationEngine(
            base_model=shared_base_model,
            calibrator=fresh_calibrator,
            step_sizes="invalid_type",  # type: ignore[arg-type]
        )


def test_from_paths_binding_verification(
    tmp_path: Path,
    shared_base_model: LGBMQuantileEnsemble,
    fresh_calibrator: ResidualCalibrator,
) -> None:
    """Verify cryptographic base_sha256 binding in from_paths (Condition C3 / T1-15)."""
    # Save base model
    base_dir = tmp_path / "base_m5"
    shared_base_model.save(base_dir)
    real_sha = shared_base_model.compute_base_sha256()

    # Save calibrator checkpoint with matching base_sha256
    cal_path_ok = tmp_path / "calibrator_ok.pt"
    torch.save(
        {
            "model_state_dict": fresh_calibrator.state_dict(),
            "base_sha256": real_sha,
        },
        cal_path_ok,
    )

    # Load with matching sha succeeds
    engine = DeploymentAdaptationEngine.from_paths(
        base_model_dir=base_dir,
        calibrator_path=cal_path_ok,
        verify_binding=True,
    )
    assert engine is not None

    # Save calibrator checkpoint with corrupted base_sha256
    cal_path_corrupted = tmp_path / "calibrator_corrupted.pt"
    torch.save(
        {
            "model_state_dict": fresh_calibrator.state_dict(),
            "base_sha256": "tampered_" + real_sha[9:],
        },
        cal_path_corrupted,
    )

    # Load with mismatched sha raises ValueError when verify_binding=True
    with pytest.raises(ValueError, match="Base model binding verification failed"):
        DeploymentAdaptationEngine.from_paths(
            base_model_dir=base_dir,
            calibrator_path=cal_path_corrupted,
            verify_binding=True,
        )

    # Load proceeds when verify_binding=False
    engine_unverified = DeploymentAdaptationEngine.from_paths(
        base_model_dir=base_dir,
        calibrator_path=cal_path_corrupted,
        verify_binding=False,
    )
    assert engine_unverified is not None


def test_cli_adapt_help() -> None:
    """Verify Typer CLI 'evaluate adapt --help' command works and documents options."""
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "adapt", "--help"])
    assert result.exit_code == 0
    assert "Run deployment adaptation lifecycle and acceptance check" in result.stdout
    assert "--domain" in result.stdout
    assert "--location-id" in result.stdout
    assert "--rho-99" in result.stdout
    assert "--support-days" in result.stdout
