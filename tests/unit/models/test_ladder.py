"""tests/unit/models/test_ladder.py — Unit tests for Lahore Model Ladder evaluation.

Covers:
  1. evaluate_predictions metric computations (MAE, pinball loss, discrete CRPS, CRPSS, coverage/sharpness).
  2. Mathematical invariant assertions (non-negativity of CRPS/MAE/pinball, coverage bounds [0, 1]).
  3. ModelLadderEvaluator execution across all rungs (M0-M7) on synthetic datasets.
  4. Markdown scorecard generation and JSON serialization round-trip.
  5. run_model_ladder runner writing artifacts to disk.
"""

import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smogsense.models.heads import CANONICAL_HORIZONS, CANONICAL_QUANTILES
from smogsense.models.ladder import (
    LADDER_RUNGS,
    ModelLadderEvaluator,
    evaluate_predictions,
    run_model_ladder,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_quantile_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ground truth and ordered quantile predictions."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(20.0, 200.0, size=n)

    # Generate 19 quantiles around y_true with noise
    taus = np.array(CANONICAL_QUANTILES)
    offsets = np.quantile(rng.normal(0, 15.0, size=(n, 1000)), taus, axis=1).T
    q_preds = y_true[:, None] + offsets
    q_preds = np.sort(np.maximum(0.0, q_preds), axis=-1)
    return y_true, q_preds


@pytest.fixture
def synthetic_lahore_feature_store(tmp_path: Path) -> Path:
    """Create a temporary directory structure mimicking feature store for Lahore."""
    rng = np.random.default_rng(123)
    n_rows = 80
    dates = pd.date_range("2023-10-01", periods=n_rows, freq="6h", tz=UTC)

    data: dict[str, Any] = {
        "issuance_utc": dates,
        "location_id": np.full(n_rows, 8664),
        "domain": ["lahore"] * n_rows,
        "feature_set_version": ["v3"] * n_rows,
        "config_hash": ["test_hash"] * n_rows,
        "n_available_stations": [1] * n_rows,
        "source_split": ["T"] * n_rows,
        "fold_id": [0] * n_rows,
        "f_stat_lat": np.full(n_rows, 31.52),
        "f_stat_lon": np.full(n_rows, 74.35),
        "f_stat_elevation_m": np.full(n_rows, 217.0),
        "f_stat_dist_city_center_km": np.full(n_rows, 5.0),
        "f_stat_is_reference": np.full(n_rows, 1.0),
        "f_cams_bias_trailing_7d": rng.normal(-10.0, 5.0, size=n_rows),
        "f_cams_lead_offset_h": np.full(n_rows, 12.0),
    }

    # Add targets and valid masks
    for h in CANONICAL_HORIZONS:
        y_vals = rng.uniform(50.0, 250.0, size=n_rows)
        data[f"y_h{h}"] = y_vals
        data[f"valid_h{h}"] = np.ones(n_rows, dtype=bool)
        data[f"f_cams_pm25_h{h}"] = y_vals * 0.85
        data[f"f_met_ln_vc_mean_h{h}"] = rng.uniform(6.0, 8.5, size=n_rows)
        data[f"f_met_rh_mean_h{h}"] = rng.uniform(40.0, 85.0, size=n_rows)
        data[f"f_fire_exposure_e{h}"] = rng.uniform(0.0, 100.0, size=n_rows)

    # Cardinal sectors
    for sec in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        data[f"f_fire_count_sector_{sec}"] = rng.poisson(10.0, size=n_rows).astype(float)

    # Rings
    for r in ("r0_50", "r50_150", "r150_300", "r300_600"):
        data[f"f_fire_count_{r}"] = rng.poisson(15.0, size=n_rows).astype(float)
        data[f"f_fire_frp_{r}"] = rng.uniform(10.0, 500.0, size=n_rows)

    df = pd.DataFrame(data)

    domain_dir = tmp_path / "lahore" / "season=2023-2024"
    domain_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(domain_dir / "features.parquet", index=False)
    return tmp_path


# -----------------------------------------------------------------------------
# 1. evaluate_predictions Tests
# -----------------------------------------------------------------------------
def test_evaluate_predictions_metrics(
    synthetic_quantile_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Verify that evaluate_predictions produces correct metric keys and valid numerical bounds."""
    y_true, q_preds = synthetic_quantile_predictions

    res = evaluate_predictions(y_true, q_preds, ref_crps_m1=25.0, ref_crps_m2=20.0)

    # Required keys
    expected_keys = {
        "median_mae",
        "pinball_loss",
        "crps",
        "crpss_vs_m1",
        "crpss_vs_m2",
        "sharpness_50",
        "coverage_50",
        "sharpness_80",
        "coverage_80",
    }
    assert expected_keys.issubset(set(res.keys()))

    # Invariants
    assert res["median_mae"] >= 0.0
    assert res["pinball_loss"] >= 0.0
    assert res["crps"] >= 0.0
    assert res["sharpness_50"] >= 0.0
    assert res["sharpness_80"] >= res["sharpness_50"]  # 80% interval must be >= 50% interval
    assert 0.0 <= res["coverage_50"] <= 1.0
    assert 0.0 <= res["coverage_80"] <= 1.0

    # CRPSS formula: 1.0 - crps / ref_crps
    expected_crpss_m1 = round(1.0 - res["crps"] / 25.0, 4)
    expected_crpss_m2 = round(1.0 - res["crps"] / 20.0, 4)
    assert np.isclose(res["crpss_vs_m1"], expected_crpss_m1, atol=1e-3)
    assert np.isclose(res["crpss_vs_m2"], expected_crpss_m2, atol=1e-3)


def test_evaluate_predictions_perfect_fit() -> None:
    """Test evaluate_predictions when all quantiles equal true observations."""
    n = 50
    y_true = np.ones(n) * 100.0
    q_preds = np.ones((n, 19)) * 100.0

    res = evaluate_predictions(y_true, q_preds)
    assert np.isclose(res["median_mae"], 0.0)
    assert np.isclose(res["pinball_loss"], 0.0)
    assert np.isclose(res["crps"], 0.0)
    assert np.isclose(res["sharpness_50"], 0.0)
    assert np.isclose(res["sharpness_80"], 0.0)


def test_evaluate_predictions_empty() -> None:
    """Test evaluate_predictions with empty arrays."""
    y_true = np.array([])
    q_preds = np.zeros((0, 19))

    res = evaluate_predictions(y_true, q_preds)
    assert res["crps"] == 0.0
    assert res["median_mae"] == 0.0


# -----------------------------------------------------------------------------
# 2. ModelLadderEvaluator Execution Tests
# -----------------------------------------------------------------------------
def test_model_ladder_evaluator_runs_all_rungs(
    synthetic_lahore_feature_store: Path, tmp_path: Path
) -> None:
    """Verify ModelLadderEvaluator runs across all 9 rungs on synthetic dataset."""
    evaluator = ModelLadderEvaluator(
        feature_store_dir=synthetic_lahore_feature_store,
        models_dir=tmp_path / "models",  # Models don't exist, will use fallbacks
        domain="lahore",
        eval_location_id=8664,
        device="cpu",
    )

    results = evaluator.run_ladder()

    assert results["domain"] == "lahore"
    assert results["eval_location_id"] == 8664
    assert results["n_rows"] == 80

    summary = results["ladder_summary"]
    # All 9 rungs must be present in the summary
    for rung in LADDER_RUNGS:
        assert rung in summary
        assert "mean_crps" in summary[rung]
        assert "crpss_vs_m1" in summary[rung]
        assert "crpss_vs_m2" in summary[rung]

    # Check horizon breakdowns
    by_h = results["by_horizon"]
    for h in CANONICAL_HORIZONS:
        assert h in by_h
        for rung in LADDER_RUNGS:
            assert rung in by_h[h]
            assert "crps" in by_h[h][rung]
            assert "median_mae" in by_h[h][rung]


def test_generate_markdown_report(synthetic_lahore_feature_store: Path) -> None:
    """Verify Markdown scorecard report formatting."""
    evaluator = ModelLadderEvaluator(
        feature_store_dir=synthetic_lahore_feature_store,
        domain="lahore",
        eval_location_id=8664,
    )
    results = evaluator.run_ladder()
    md_report = evaluator.generate_markdown_report(results)

    assert "# SmogSense Model Ladder Evaluation Scorecard (Lahore)" in md_report
    assert "## 1. Overall Model Ladder Summary" in md_report
    assert "## 2. Performance Breakdown by Horizon" in md_report
    assert "Horizon h = 24 Hours" in md_report
    assert "Horizon h = 48 Hours" in md_report
    assert "Horizon h = 72 Hours" in md_report
    for rung in LADDER_RUNGS:
        assert f"**{rung}**" in md_report


def test_run_model_ladder_end_to_end(synthetic_lahore_feature_store: Path, tmp_path: Path) -> None:
    """Verify run_model_ladder persists JSON and Markdown reports to specified directory."""
    out_dir = tmp_path / "reports_out"
    results = run_model_ladder(
        feature_store_dir=synthetic_lahore_feature_store,
        models_dir=tmp_path / "empty_models",
        domain="lahore",
        eval_location_id=8664,
        output_dir=out_dir,
    )

    assert results is not None
    json_path = out_dir / "ladder.json"
    md_path = out_dir / "ladder.md"

    assert json_path.exists()
    assert md_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["domain"] == "lahore"
    assert "ladder_summary" in loaded_json
