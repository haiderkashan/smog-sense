"""tests/unit/test_low_cost_correction.py — Unit tests for low-cost sensor correction engine.

Covers:
- Robust Huber regression fitting on synthetic paired sensor data with known gain and RH bias.
- Non-negativity constraint enforcement on predict (pm25_corrected >= 0.0).
- 5-fold cross-validation computation and metric consistency.
- Inverted corruption prior extraction and comparison against model augmentation bounds.
- Serialization and deserialization to JSON and Parquet.
- Candidate station filtering by Haversine distance.
- Minimum pair sample threshold validation.
- Decision rule updating configs/preprocessing.yaml.
- Typer CLI command invocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.preprocessing.low_cost_correction import (
    LowCostCorrectionModel,
    audit_corruption_priors,
    build_paired_observations,
    find_candidate_stations,
    haversine_distance_km,
    update_preprocessing_config_decision,
)


@pytest.fixture
def synthetic_paired_data() -> pd.DataFrame:
    """Generate synthetic paired reference and optical low-cost sensor observations."""
    rng = np.random.default_rng(20261019)
    n = 600

    # True reference PM2.5 and atmospheric relative humidity
    pm25_ref = rng.uniform(20.0, 350.0, size=n)
    rh = rng.uniform(25.0, 95.0, size=n)

    # Low-cost sensor corrupted with gain 1.4, RH interference 0.35, offset 15.0
    # Inverse: pm25_ref = (pm25_raw - 0.35*rh - 15) / 1.4 = 0.714*pm25_raw - 0.25*rh - 10.71
    pm25_raw = 1.4 * pm25_ref + 0.35 * rh + 15.0 + rng.normal(0.0, 5.0, size=n)

    # Introduce extreme outlier spikes (smoke plume / local emission spike) on 3% of points
    spike_idx = rng.choice(n, size=int(0.03 * n), replace=False)
    pm25_raw[spike_idx] += rng.uniform(150.0, 400.0, size=len(spike_idx))

    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts_utc": ts,
            "location_id": [4527035] * n,
            "sensor_model": ["Optical Low-Cost"] * n,
            "distance_km": [3.5] * n,
            "pm25_raw": pm25_raw,
            "rh": rh,
            "pm25_ref": pm25_ref,
        }
    )


def test_huber_fit_synthetic_known_coefficients(synthetic_paired_data: pd.DataFrame) -> None:
    """Test that robust Huber fitting recovers expected linear coefficients despite outliers."""
    model = LowCostCorrectionModel(epsilon=1.35)
    model.fit(synthetic_paired_data)

    assert model.is_fitted
    # Raw slope ~ 1.4 -> expected a ~ 1/1.4 = 0.714 (within tolerance)
    assert 0.60 <= model.a <= 0.85
    # RH interference should be negative in the correction: ~ -0.25
    assert -0.45 <= model.b <= -0.10


def test_predict_non_negativity_constraint() -> None:
    """Assert predict() strictly clamps negative predictions to 0.0."""
    model = LowCostCorrectionModel(a=0.5, b=0.1, c=-50.0, is_fitted=True)

    pm25_raw = np.array([10.0, 20.0, 200.0])
    rh = np.array([30.0, 40.0, 60.0])

    preds = model.predict(pm25_raw, rh)

    # 0.5 * 10 + 0.1 * 30 - 50 = 5 + 3 - 50 = -42 -> clamped to 0.0
    assert preds[0] == 0.0
    # 0.5 * 20 + 0.1 * 40 - 50 = 10 + 4 - 50 = -36 -> clamped to 0.0
    assert preds[1] == 0.0
    # 0.5 * 200 + 0.1 * 60 - 50 = 100 + 6 - 50 = 56.0
    assert np.isclose(preds[2], 56.0)


def test_predict_unfitted_raises() -> None:
    """Predicting on unfitted model must raise RuntimeError."""
    model = LowCostCorrectionModel(is_fitted=False)
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict(np.array([50.0]), np.array([50.0]))


def test_cross_validation_metrics(synthetic_paired_data: pd.DataFrame) -> None:
    """Assert cross_validate computes complete metrics and shows clear error improvement."""
    model = LowCostCorrectionModel()
    metrics = model.cross_validate(synthetic_paired_data, n_splits=5)

    assert metrics["n_samples"] == len(synthetic_paired_data)
    assert metrics["n_splits"] == 5.0
    assert metrics["raw_rmse"] > metrics["corrected_rmse"]
    assert metrics["rmse_improvement"] > 0.05
    assert metrics["raw_mae"] > metrics["corrected_mae"]
    assert metrics["beneficial"] == 1.0


def test_inverted_corruption_priors() -> None:
    """Verify exact formula inversion for synthetic corruption priors."""
    # Given pm25_ref = 0.8 * pm25_raw + 0.24 * RH - 8.0
    # Inverted: gain = 1/0.8 = 1.25, rh_slope = 0.24/0.8 = 0.3, offset = -(-8.0)/0.8 = 10.0
    model = LowCostCorrectionModel(a=0.8, b=0.24, c=-8.0, is_fitted=True)
    priors = model.get_inverted_corruption_priors()

    assert np.isclose(priors["gain_prior"], 1.25)
    assert np.isclose(priors["rh_slope_prior"], 0.3)
    assert np.isclose(priors["offset_prior"], 10.0)


def test_model_serialization_json(tmp_path: Path) -> None:
    """Test saving and loading model via JSON."""
    model = LowCostCorrectionModel(
        a=0.75,
        b=0.15,
        c=5.5,
        is_fitted=True,
        metrics_={"rmse_improvement": 0.12},
        sensor_model="AirGradient",
    )
    json_path = tmp_path / "model.json"
    model.save(json_path)

    loaded = LowCostCorrectionModel.load(json_path)
    assert loaded.is_fitted
    assert np.isclose(loaded.a, 0.75)
    assert np.isclose(loaded.b, 0.15)
    assert np.isclose(loaded.c, 5.5)
    assert loaded.sensor_model == "AirGradient"
    assert loaded.metrics_["rmse_improvement"] == 0.12


def test_model_serialization_parquet(tmp_path: Path) -> None:
    """Test saving and loading model via Parquet."""
    model = LowCostCorrectionModel(
        a=0.82,
        b=-0.12,
        c=4.2,
        is_fitted=True,
        sensor_model="Plantower",
    )
    parquet_path = tmp_path / "model.parquet"
    model.save(parquet_path)

    loaded = LowCostCorrectionModel.load(parquet_path)
    assert loaded.is_fitted
    assert np.isclose(loaded.a, 0.82)
    assert np.isclose(loaded.b, -0.12)
    assert np.isclose(loaded.c, 4.2)
    assert loaded.sensor_model == "Plantower"


def test_find_candidate_stations() -> None:
    """Test candidate station identification and Haversine distance filtering."""
    reg_df = pd.DataFrame(
        {
            "location_id": [8664, 1001, 1002, 1003],
            "name": ["Ref Monitor", "Near 1", "Near 2", "Far 1"],
            "lat": [31.5601, 31.5300, 31.5000, 31.2000],
            "lon": [74.3359, 74.3300, 74.3400, 74.1000],
            "is_reference": [True, False, False, False],
            "sensor_model": ["BAM", "LowCost", "LowCost", "LowCost"],
        }
    )

    candidates = find_candidate_stations(reg_df, reference_location_id=8664, max_distance_km=8.0)

    assert len(candidates) == 2
    assert set(candidates["location_id"]) == {1001, 1002}
    assert candidates.iloc[0]["distance_km"] < candidates.iloc[1]["distance_km"]


def test_haversine_distance() -> None:
    """Test Haversine distance formula against known geographic points."""
    # Lahore (31.56, 74.34) to Delhi (28.61, 77.21) ~ 400-450 km
    dist = haversine_distance_km(31.56, 74.34, 28.61, 77.21)
    assert 400.0 < dist < 450.0


def test_build_paired_observations_insufficient_pairs() -> None:
    """Assert ValueError when paired records are below min_pairs."""
    reg_df = pd.DataFrame(
        {
            "location_id": [8664, 1001],
            "name": ["Ref", "Cand"],
            "lat": [31.5601, 31.5500],
            "lon": [74.3359, 74.3350],
            "is_reference": [True, False],
            "sensor_model": ["Ref", "LowCost"],
        }
    )

    obs_df = pd.DataFrame(
        {
            "location_id": [8664] * 20 + [1001] * 20,
            "ts_utc": list(pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")) * 2,
            "pm25_ugm3": [100.0] * 40,
            "pm25_raw_ugm3": [110.0] * 40,
            "rh_pct": [60.0] * 40,
        }
    )

    with pytest.raises(ValueError, match="Insufficient paired observations"):
        build_paired_observations(
            obs_df=obs_df,
            registry_df=reg_df,
            reference_location_id=8664,
            min_pairs=50,
        )


def test_audit_corruption_priors(tmp_path: Path) -> None:
    """Test auditing inverted empirical priors against model augmentation ranges."""
    cfg = {
        "corruption": {
            "gain_range": [0.8, 1.3],
            "rh_slope_range": [0.0, 1.5],
            "offset_range": [-10.0, 25.0],
        }
    }
    cfg_file = tmp_path / "model.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    priors_inside = {
        "gain_prior": 1.1,
        "rh_slope_prior": 0.5,
        "offset_prior": 5.0,
    }
    audit_ok = audit_corruption_priors(priors_inside, model_config_path=cfg_file)
    assert audit_ok["all_within_priors"] is True

    priors_outside = {
        "gain_prior": 1.55,  # Exceeds 1.3
        "rh_slope_prior": 0.5,
        "offset_prior": 5.0,
    }
    audit_fail = audit_corruption_priors(priors_outside, model_config_path=cfg_file)
    assert audit_fail["all_within_priors"] is False
    assert audit_fail["gain"]["within_prior"] is False
    assert audit_fail["rh_slope"]["within_prior"] is True


def test_update_preprocessing_config_decision(tmp_path: Path) -> None:
    """Test updating preprocessing.yaml based on the 5% threshold decision rule."""
    cfg = {
        "lowcost_correction": {
            "enabled": False,
            "form": "pm25_corrected = a * pm25_raw + b * rh_pct + c",
        }
    }
    cfg_file = tmp_path / "preprocessing.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # 1. Improvement >= 5% -> enabled = True
    enabled = update_preprocessing_config_decision(
        rmse_improvement=0.08, config_path=cfg_file, threshold=0.05
    )
    assert enabled is True
    with cfg_file.open("r", encoding="utf-8") as f:
        saved_cfg = yaml.safe_load(f)
    assert saved_cfg["lowcost_correction"]["enabled"] is True

    # 2. Improvement < 5% -> enabled = False
    enabled_low = update_preprocessing_config_decision(
        rmse_improvement=0.03, config_path=cfg_file, threshold=0.05
    )
    assert enabled_low is False
    with cfg_file.open("r", encoding="utf-8") as f:
        saved_cfg_2 = yaml.safe_load(f)
    assert saved_cfg_2["lowcost_correction"]["enabled"] is False


def test_cli_low_cost_correction(tmp_path: Path) -> None:
    """Test CLI subcommand 'preprocess low-cost-correction' end-to-end with json output."""
    runner = CliRunner()
    json_out = tmp_path / "cv_report.json"

    res = runner.invoke(
        app,
        [
            "preprocess",
            "low-cost-correction",
            "--domain",
            "lahore",
            "--output-json",
            str(json_out),
        ],
    )

    assert res.exit_code == 0
    assert "LOW-COST SENSOR CORRECTION FIT & VALIDATION" in res.output
    assert "RMSE Improvement:" in res.output
    assert json_out.exists()

    with json_out.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["domain"] == "lahore"
    assert "coefficients" in report
    assert "cross_validation" in report
    assert "inverted_priors" in report
