"""tests/unit/models/test_residual_bound.py — Unit tests for residual bound and acceptance.

Specification:
- docs/ml-architecture.md §8 (Deployment adaptation, Condition C6, residual bound)
- configs/model.yaml (acceptance & residual_bound blocks)
- decisions.md (ADR-024 §6, §7)
- roadmap.md Task P2-12

Verifies:
1. Mathematical invariants & percentile ordering: rho_99 >= rho_95 >= rho_50 >= 0.
2. Acceptance decision gate:
   - Passing condition (improvement >= 2%, days >= 60%, R <= rho_99) -> 'adapted'
   - No-gain rejection (improvement < 2% or days < 60%) -> 'rejected_no_gain'
   - Residual bound rejection (improvement >= 2%, days >= 60%, R > rho_99) -> 'rejected_residual_bound'
   - Sensor-free mode (N = 0) -> 'skipped_n0'
   - Insufficient support (duration < 7 days or empty check) -> 'skipped_insufficient_support'
3. Split support window chronological separation and 72-hour embargo integrity.
4. Deterministic simulation reproducibility and JSON roundtrip serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from smogsense.models.heads import (
    ResidualCalibrator,
)
from smogsense.models.maml import MetaSGDStepSizes
from smogsense.models.residual_bound import (
    AdaptationAcceptanceResult,
    compute_residual_statistic,
    evaluate_adaptation_acceptance,
    simulate_deployments_on_block_v,
    split_support_window,
)


def _create_synthetic_block_v_df(
    n_days: int = 40,
    n_stations: int = 4,
    seed: int = 101,
) -> pd.DataFrame:
    """Create synthetic Block V dataframe with valid feature columns."""
    rng = np.random.default_rng(seed)
    station_ids = [7001 + i for i in range(n_stations)]
    dates = pd.date_range("2024-10-01", periods=n_days * 4, freq="6h", tz="UTC")

    dfs: list[pd.DataFrame] = []
    for sid in station_ids:
        n_obs = len(dates)
        df_stn = pd.DataFrame(
            {
                "domain": ["delhi"] * n_obs,
                "location_id": [sid] * n_obs,
                "issuance_utc": dates,
                "source_split": ["V"] * n_obs,
                "season": ["2024-2025"] * n_obs,
                "f_stat_latitude": [28.60 + 0.01 * (sid - 7000)] * n_obs,
                "f_stat_longitude": [77.20 + 0.01 * (sid - 7000)] * n_obs,
                "f_met_vc_mean_h24": rng.uniform(500, 2500, size=n_obs),
                "f_met_rh_mean_h24": rng.uniform(30, 90, size=n_obs),
                "f_fire_exposure_e24": rng.exponential(5.0, size=n_obs),
                "f_aod_mean_h24": rng.uniform(0.1, 1.2, size=n_obs),
                "f_cams_pm25_h24": rng.uniform(40, 250, size=n_obs),
                "f_stat_elevation_m": [215.0] * n_obs,
                "y_h24": rng.uniform(30, 300, size=n_obs),
                "valid_h24": [True] * n_obs,
                "y_h48": rng.uniform(30, 300, size=n_obs),
                "valid_h48": [True] * n_obs,
                "y_h72": rng.uniform(30, 300, size=n_obs),
                "valid_h72": [True] * n_obs,
            }
        )
        dfs.append(df_stn)

    return pd.concat(dfs, ignore_index=True).sort_values("issuance_utc").reset_index(drop=True)


def test_residual_statistic_computation() -> None:
    """Verify residual statistic R = mean(|Delta z_hat|) under zero, constant, and random values."""
    # 1. Zero residuals -> R = 0.0
    zero_tensor = torch.zeros(10, 3, 19)
    assert compute_residual_statistic(zero_tensor) == 0.0

    zero_arr = np.zeros((10, 3, 19))
    assert compute_residual_statistic(zero_arr) == 0.0

    # 2. Constant residuals -> R = |c|
    c_tensor = torch.full((5, 3, 19), -0.42)
    assert abs(compute_residual_statistic(c_tensor) - 0.42) < 1e-6

    c_arr = np.full((5, 3, 19), 0.75)
    assert abs(compute_residual_statistic(c_arr) - 0.75) < 1e-6

    # 3. Empty input safety -> R = 0.0
    assert compute_residual_statistic(torch.zeros(0, 3, 19)) == 0.0
    assert compute_residual_statistic(np.array([])) == 0.0


def test_split_support_window_embargo_and_proportions() -> None:
    """Verify fit/embargo/check window split satisfies 72h embargo and non-overlap."""
    dates = pd.date_range("2024-10-01", periods=14 * 24, freq="1h", tz="UTC")
    df = pd.DataFrame({"issuance_utc": dates, "location_id": [7001] * len(dates)})

    df_fit, df_check = split_support_window(df, embargo_hours=72, check_fraction=0.35)

    assert not df_fit.empty
    assert not df_check.empty

    t_fit_max = pd.to_datetime(df_fit["issuance_utc"].max(), utc=True)
    t_check_min = pd.to_datetime(df_check["issuance_utc"].min(), utc=True)

    # Embargo separation must be strictly at least 72 hours
    embargo_gap_h = (t_check_min - t_fit_max).total_seconds() / 3600.0
    assert embargo_gap_h >= 72.0

    # Check window too short (< embargo + min_fit + min_check) returns empty DataFrames
    short_dates = pd.date_range("2024-10-01", periods=48, freq="1h", tz="UTC")
    df_short = pd.DataFrame({"issuance_utc": short_dates})
    f_empty, c_empty = split_support_window(df_short, embargo_hours=72)
    assert f_empty.empty and c_empty.empty


def test_acceptance_logic_adapted_pass() -> None:
    """Passing condition: improvement >= 2%, days >= 60%, R <= rho_99 -> 'adapted'."""
    n_days = 10
    samples_per_day = 4
    total_samples = n_days * samples_per_day

    dates = pd.date_range("2024-10-10", periods=total_samples, freq="6h", tz="UTC")
    y_true = np.full((total_samples, 3), 4.5)  # log1p space
    y_mask = np.ones((total_samples, 3), dtype=bool)

    # M5 baseline predictions (higher loss)
    m5_pred = np.full((total_samples, 3, 19), 5.0)

    # Adapted predictions (closer to y_true -> ~10% improvement, consistent across all days)
    adapted_pred = np.full((total_samples, 3, 19), 4.9)

    rho_99 = 0.20
    res: AdaptationAcceptanceResult = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_pred,
        dates=dates,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=3,
        support_days=14.0,
    )

    assert res.accepted is True
    assert res.adaptation_status == "adapted"
    assert res.relative_improvement >= 0.02
    assert res.days_better_fraction >= 0.60
    assert res.residual_r <= rho_99
    assert abs(res.residual_r - 0.1) < 1e-5


def test_acceptance_logic_rejected_no_gain() -> None:
    """No-gain rejection: improvement < 2% or days < 60% -> 'rejected_no_gain'."""
    n_days = 10
    total_samples = n_days * 4
    dates = pd.date_range("2024-10-10", periods=total_samples, freq="6h", tz="UTC")
    y_true = np.full((total_samples, 3), 4.5)
    y_mask = np.ones((total_samples, 3), dtype=bool)

    # Case A: Adapted is worse than M5 (negative improvement)
    m5_pred = np.full((total_samples, 3, 19), 4.6)
    adapted_pred_worse = np.full((total_samples, 3, 19), 5.0)

    res_worse = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_pred_worse,
        dates=dates,
        rho_99=0.50,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_worse.accepted is False
    assert res_worse.adaptation_status == "rejected_no_gain"

    # Case B: Marginal improvement (1% < 2%)
    adapted_pred_marginal = np.full((total_samples, 3, 19), 4.599)
    res_marginal = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_pred_marginal,
        dates=dates,
        rho_99=0.50,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_marginal.accepted is False
    assert res_marginal.adaptation_status == "rejected_no_gain"


def test_acceptance_logic_rejected_residual_bound() -> None:
    """Residual bound rejection: improvement >= 2% and days >= 60%, but R > rho_99 -> 'rejected_residual_bound'."""
    n_days = 8
    total_samples = n_days * 4
    dates = pd.date_range("2024-10-10", periods=total_samples, freq="6h", tz="UTC")
    y_true = np.full((total_samples, 3), 4.0)
    y_mask = np.ones((total_samples, 3), dtype=bool)

    # M5 is far off at 5.5
    m5_pred = np.full((total_samples, 3, 19), 5.5)

    # Adapted makes a giant jump to 4.2 (big improvement of ~80%, but Delta z is 1.3)
    adapted_pred = np.full((total_samples, 3, 19), 4.2)

    # Strict rho_99 bound at 0.25 (exceeded by R = 1.3)
    strict_rho_99 = 0.25

    res = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_pred,
        dates=dates,
        rho_99=strict_rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )

    assert res.accepted is False
    assert res.adaptation_status == "rejected_residual_bound"
    assert res.relative_improvement > 0.02
    assert res.days_better_fraction > 0.60
    assert res.residual_r > strict_rho_99


def test_acceptance_logic_edge_cases() -> None:
    """Edge cases: N=0 -> 'skipped_n0'; support_days < 7 -> 'skipped_insufficient_support'."""
    dates = pd.date_range("2024-10-10", periods=12, freq="6h", tz="UTC")
    y_dummy = np.zeros((12, 3))
    m_dummy = np.ones((12, 3), dtype=bool)
    p_dummy = np.zeros((12, 3, 19))

    # 1. N = 0 (sensor-free mode)
    res_n0 = evaluate_adaptation_acceptance(
        y_true=y_dummy,
        y_mask=m_dummy,
        m5_pred=p_dummy,
        adapted_pred=p_dummy,
        dates=dates,
        rho_99=0.30,
        n_available=0,
    )
    assert res_n0.accepted is False
    assert res_n0.adaptation_status == "skipped_n0"

    # 2. Insufficient support duration (< 7 days)
    res_short = evaluate_adaptation_acceptance(
        y_true=y_dummy,
        y_mask=m_dummy,
        m5_pred=p_dummy,
        adapted_pred=p_dummy,
        dates=dates,
        rho_99=0.30,
        n_available=2,
        support_days=3.0,
    )
    assert res_short.accepted is False
    assert res_short.adaptation_status == "skipped_insufficient_support"

    # 3. Zero check samples
    res_empty = evaluate_adaptation_acceptance(
        y_true=np.zeros((0, 3)),
        y_mask=np.zeros((0, 3), dtype=bool),
        m5_pred=np.zeros((0, 3, 19)),
        adapted_pred=np.zeros((0, 3, 19)),
        dates=[],
        rho_99=0.30,
        n_available=2,
        support_days=14.0,
    )
    assert res_empty.accepted is False
    assert res_empty.adaptation_status == "skipped_insufficient_support"


def test_simulate_deployments_percentile_invariants_and_determinism(tmp_path: Path) -> None:
    """Verify percentile monotonic ordering, seeded determinism, and JSON persistence."""
    df_v = _create_synthetic_block_v_df(n_days=35, n_stations=3, seed=77)

    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    out_json = tmp_path / "residual_bound.json"

    # Run 15 simulations
    summary1 = simulate_deployments_on_block_v(
        df_block_v=df_v,
        calibrator=calibrator,
        step_sizes=step_sizes,
        n_simulations=15,
        support_window_days=(7, 14),
        embargo_hours=72,
        inner_steps=2,
        seed=42,
        output_json=out_json,
    )

    # 1. Percentile monotonic invariants
    p = summary1["percentiles"]
    assert p["max"] >= p["p99"] >= p["p95"] >= p["p90"] >= p["p50"] >= 0.0
    assert summary1["rho_99"] == p["p99"]
    assert summary1["n_simulations"] == 15
    assert summary1["statistic"] == "mean_abs_residual"

    # 2. JSON persistence & roundtrip verification
    assert out_json.exists()
    with out_json.open("r") as f:
        loaded = json.load(f)
    assert loaded["rho_99"] == summary1["rho_99"]
    assert loaded["percentiles"] == summary1["percentiles"]

    # 3. Seeded reproducibility
    summary2 = simulate_deployments_on_block_v(
        df_block_v=df_v,
        calibrator=calibrator,
        step_sizes=step_sizes,
        n_simulations=15,
        support_window_days=(7, 14),
        embargo_hours=72,
        inner_steps=2,
        seed=42,
    )
    np.testing.assert_allclose(summary2["rho_99"], summary1["rho_99"], rtol=1e-4)
    for k in ("p50", "p90", "p95", "p99", "max"):
        np.testing.assert_allclose(
            summary2["percentiles"][k], summary1["percentiles"][k], rtol=1e-4
        )
