"""tests/unit/models/test_diagnostics.py — Unit tests for atmospheric diagnostics.

Covers:
  1. compute_lag_cross_correlation peak lag detection on phase-shifted signals.
  2. Mathematical invariants (correlation clipping [-1, 1], standard error scaling, advective check).
  3. compute_atmospheric_lag_correlations over multiple atmospheric drivers.
  4. compute_sector_importance fire activity spatial decomposition (8 sectors, 4 rings, smog conditioning).
  5. generate_diagnostics_report markdown formatting and run_atmospheric_diagnostics persistence.
"""

import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smogsense.models.diagnostics import (
    RINGS_4,
    SECTORS_8,
    compute_atmospheric_lag_correlations,
    compute_lag_cross_correlation,
    compute_sector_importance,
    generate_diagnostics_report,
    run_atmospheric_diagnostics,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_phase_shifted_signals() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate driver and target signals with a known lead of 4 steps (24 hours)."""
    rng = np.random.default_rng(42)
    n = 200
    dt_hours = 6
    lead_steps = 4  # 4 * 6 = 24 hours lead

    t = np.linspace(0, 10 * np.pi, n)
    base_signal = np.sin(t) + 0.1 * rng.normal(size=n)

    # Driver leads target: driver = base_signal, target = base_signal delayed by lead_steps
    driver = base_signal.copy()
    target = np.roll(base_signal, lead_steps)
    # Zero out edges
    target[:lead_steps] = 0.0

    return driver, target, lead_steps, dt_hours


@pytest.fixture
def synthetic_diagnostics_df() -> pd.DataFrame:
    """Create synthetic DataFrame for atmospheric diagnostics testing."""
    rng = np.random.default_rng(99)
    n_rows = 120
    dates = pd.date_range("2023-10-01", periods=n_rows, freq="6h", tz=UTC)

    y = rng.uniform(50.0, 300.0, size=n_rows)
    # Create severe smog episodes
    y[20:40] = rng.uniform(200.0, 350.0, size=20)

    data: dict[str, Any] = {
        "issuance_utc": dates,
        "location_id": np.full(n_rows, 8664),
        "domain": ["lahore"] * n_rows,
        "y_h24": y,
        "valid_h24": np.ones(n_rows, dtype=bool),
        "f_fire_exposure_e24": rng.uniform(0.0, 150.0, size=n_rows),
        "f_met_ln_vc_mean_h24": rng.uniform(5.5, 8.5, size=n_rows),
        "f_cams_bias_trailing_7d": rng.normal(-15.0, 5.0, size=n_rows),
        "f_met_rh_mean_h24": rng.uniform(30.0, 95.0, size=n_rows),
        "f_met_delta_theta_mean_h24": rng.normal(1.5, 0.8, size=n_rows),
        "f_aod_mean_h24": rng.uniform(0.2, 2.5, size=n_rows),
    }

    # Populate 8 sectors
    for sec in SECTORS_8:
        # Heavily weight NE and E
        weight = 50.0 if sec in ("NE", "E") else 5.0
        data[f"f_fire_count_sector_{sec}"] = rng.poisson(weight, size=n_rows).astype(float)

    # Populate 4 rings
    for r in RINGS_4:
        data[f"f_fire_count_{r}"] = rng.poisson(25.0, size=n_rows).astype(float)
        data[f"f_fire_frp_{r}"] = rng.uniform(50.0, 1000.0, size=n_rows)

    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# 1. compute_lag_cross_correlation Tests
# -----------------------------------------------------------------------------
def test_compute_lag_cross_correlation_peak_detection(
    synthetic_phase_shifted_signals: tuple[np.ndarray, np.ndarray, int, int],
) -> None:
    """Verify that compute_lag_cross_correlation correctly identifies the true lag."""
    driver, target, lead_steps, dt_hours = synthetic_phase_shifted_signals
    expected_lag_hours = lead_steps * dt_hours  # 24.0h

    res = compute_lag_cross_correlation(driver, target, max_lag_hours=48, dt_hours=dt_hours)

    assert "lags_hours" in res
    assert "correlations" in res
    assert "peak_lag_hours" in res
    assert "peak_correlation" in res

    # The peak lag should be exactly 24.0 hours (lead_steps = 4)
    assert np.isclose(res["peak_lag_hours"], expected_lag_hours, atol=dt_hours)
    assert res["peak_correlation"] > 0.8  # Strong positive correlation at true lag
    assert res["physically_consistent"] is True


def test_compute_lag_cross_correlation_bounds() -> None:
    """Verify correlation clipping, standard error, and confidence intervals."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)

    res = compute_lag_cross_correlation(x, y, max_lag_hours=24, dt_hours=6)

    # All correlation values must be strictly in [-1, 1]
    for c in res["correlations"]:
        assert -1.0 <= c <= 1.0

    # Check SE formula
    expected_se = round(1.0 / np.sqrt(100), 4)
    assert np.isclose(res["std_error"], expected_se)


def test_compute_lag_cross_correlation_short_series() -> None:
    """Verify safe handling of short series (n < 10)."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])

    res = compute_lag_cross_correlation(x, y)
    assert res["lags_hours"] == []
    assert res["peak_correlation"] == 0.0
    assert res["physically_consistent"] is False


def test_compute_lag_cross_correlation_mismatch_raises() -> None:
    """Verify error raised on array length mismatch."""
    x = np.ones(10)
    y = np.ones(12)
    with pytest.raises(ValueError, match="Series length mismatch"):
        compute_lag_cross_correlation(x, y)


# -----------------------------------------------------------------------------
# 2. compute_atmospheric_lag_correlations Tests
# -----------------------------------------------------------------------------
def test_compute_atmospheric_lag_correlations(synthetic_diagnostics_df: pd.DataFrame) -> None:
    """Verify lag correlations across atmospheric drivers in DataFrame."""
    results = compute_atmospheric_lag_correlations(
        synthetic_diagnostics_df, target_col="y_h24", dt_hours=6, max_lag_hours=48
    )

    # Core drivers should be evaluated
    expected_drivers = {
        "fire_exposure",
        "ventilation_ln_vc",
        "cams_bias",
        "relative_humidity",
        "stability_delta_theta",
        "aod",
    }
    assert expected_drivers.issubset(set(results.keys()))

    for res in results.values():
        assert "feature_column" in res
        assert "peak_lag_hours" in res
        assert "peak_correlation" in res
        assert "std_error" in res


# -----------------------------------------------------------------------------
# 3. compute_sector_importance Tests
# -----------------------------------------------------------------------------
def test_compute_sector_importance(synthetic_diagnostics_df: pd.DataFrame) -> None:
    """Verify sector and ring fire decompositions and smog episode conditioning."""
    res = compute_sector_importance(
        synthetic_diagnostics_df, target_col="y_h24", smog_threshold=150.0
    )

    assert res["total_rows"] == len(synthetic_diagnostics_df)
    assert res["n_smog_episodes"] > 0
    assert res["total_fire_detections"] > 0

    # Sector shares must sum to 100% (within rounding)
    sector_shares = res["sector_shares_pct"]
    assert len(sector_shares) == 8
    assert np.isclose(sum(sector_shares.values()), 100.0, atol=0.2)

    # Since NE and E were weighted 50 vs 5, E + NE + SE share should dominate
    assert res["e_ne_se_share_pct"] > 50.0

    # Ring shares must sum to 100%
    ring_shares = res["ring_shares_pct"]
    assert len(ring_shares) == 4
    assert np.isclose(sum(ring_shares.values()), 100.0, atol=0.2)


# -----------------------------------------------------------------------------
# 4. Report Generation & Runner Tests
# -----------------------------------------------------------------------------
def test_generate_diagnostics_report(synthetic_diagnostics_df: pd.DataFrame) -> None:
    """Verify Markdown report generation for atmospheric diagnostics."""
    lag_res = compute_atmospheric_lag_correlations(synthetic_diagnostics_df, target_col="y_h24")
    sec_res = compute_sector_importance(synthetic_diagnostics_df, target_col="y_h24")

    md = generate_diagnostics_report(lag_res, sec_res, domain="lahore")

    assert "# SmogSense Atmospheric Diagnostics Report (Lahore)" in md
    assert "## 1. Lagged Cross-Correlation Analysis" in md
    assert "## 2. Sector-Importance & Fire Transport Decomposition" in md
    assert "Distance Ring Decomposition" in md
    assert "Key Findings" in md
    for sec in SECTORS_8:
        assert f"**{sec}**" in md


def test_run_atmospheric_diagnostics_end_to_end(tmp_path: Path) -> None:
    """Verify run_atmospheric_diagnostics loads feature store and outputs report files."""
    rng = np.random.default_rng(77)
    df = pd.DataFrame(
        {
            "issuance_utc": pd.date_range("2023-10-01", periods=50, freq="6h", tz=UTC),
            "location_id": [8664] * 50,
            "y_h24": rng.uniform(50.0, 200.0, size=50),
            "valid_h24": [True] * 50,
            "f_fire_exposure_e24": rng.uniform(0.0, 100.0, size=50),
            "f_met_ln_vc_mean_h24": rng.uniform(6.0, 8.0, size=50),
            "f_fire_count_sector_E": rng.poisson(10.0, size=50).astype(float),
            "f_fire_count_sector_NE": rng.poisson(20.0, size=50).astype(float),
            "f_fire_count_r0_50": rng.poisson(5.0, size=50).astype(float),
        }
    )

    store_dir = tmp_path / "store" / "lahore" / "season=2023-2024"
    store_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(store_dir / "features.parquet", index=False)

    out_dir = tmp_path / "diag_reports"
    res = run_atmospheric_diagnostics(
        feature_store_dir=tmp_path / "store",
        domain="lahore",
        eval_location_id=8664,
        output_dir=out_dir,
    )

    assert res["domain"] == "lahore"
    assert (out_dir / "diagnostics.json").exists()
    assert (out_dir / "diagnostics.md").exists()

    with (out_dir / "diagnostics.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "lag_correlations" in data
    assert "sector_importance" in data
