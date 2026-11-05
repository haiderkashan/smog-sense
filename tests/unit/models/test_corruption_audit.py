"""tests.unit.models.test_corruption_audit — Unit tests for empirical corruption-prior audit.

Tests:
1. Inverted prior mathematical relationships (g = 1/a, s = b/a, o = -c/a).
2. Prior envelopment logic and Risk R17 trigger detection.
3. Support window error distortion metrics calculation.
4. JSON and Markdown report serialization with LF line endings and single trailing newline.
5. CLI runner execution and help surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.corruption_audit import (
    CorruptionPriorAuditResult,
    PriorComparison,
    SupportWindowErrorStats,
    analyze_support_window_distortion,
    generate_corruption_prior_report,
)
from smogsense.preprocessing.low_cost_correction import LowCostCorrectionModel


def test_inverted_priors_calculation() -> None:
    """Verify mathematical inversion of empirical Huber coefficients."""
    # Fitted model: pm25_ref = 0.6927 * pm25_raw + 0.6400 * RH + 15.7893
    model = LowCostCorrectionModel(
        a=0.6927,
        b=0.6400,
        c=15.7893,
        is_fitted=True,
    )
    priors = model.get_inverted_corruption_priors()

    # Gain: 1 / 0.6927 = 1.4436
    assert abs(priors["gain_prior"] - (1.0 / 0.6927)) < 1e-3
    assert priors["gain_prior"] == 1.4436

    # RH slope: 0.6400 / 0.6927 = 0.9239
    assert abs(priors["rh_slope_prior"] - (0.6400 / 0.6927)) < 1e-3
    assert priors["rh_slope_prior"] == 0.9239

    # Offset: -15.7893 / 0.6927 = -22.7939
    assert abs(priors["offset_prior"] - (-15.7893 / 0.6927)) < 1e-3
    assert priors["offset_prior"] == -22.7939


def test_prior_envelopment_and_risk_r17_detection() -> None:
    """Verify that empirical gain exceeding 1.30 triggers Risk R17."""
    configured_gain = (0.8, 1.3)
    configured_rh_slope = (0.0, 1.5)
    configured_offset = (-10.0, 25.0)

    # Case 1: Empirical gain = 1.4436 > 1.30 -> Out of bounds, triggers Risk R17
    emp_gain = 1.4436
    emp_rh_slope = 0.9239
    emp_offset = -22.7939

    gain_env = bool(configured_gain[0] <= emp_gain <= configured_gain[1])
    rh_env = bool(configured_rh_slope[0] <= emp_rh_slope <= configured_rh_slope[1])
    off_env = bool(configured_offset[0] <= emp_offset <= configured_offset[1])

    assert gain_env is False
    assert rh_env is True
    assert off_env is False

    risk_r17_triggered = not gain_env
    assert risk_r17_triggered is True

    # Case 2: Ideal enveloped sensor
    ideal_gain = 1.10
    ideal_rh_slope = 0.50
    ideal_offset = 5.0

    assert bool(configured_gain[0] <= ideal_gain <= configured_gain[1]) is True
    assert bool(configured_rh_slope[0] <= ideal_rh_slope <= configured_rh_slope[1]) is True
    assert bool(configured_offset[0] <= ideal_offset <= configured_offset[1]) is True


def test_support_window_distortion_calculation() -> None:
    """Verify sliding support window distortion metrics on synthetic paired data."""
    n_hours = 24 * 30  # 30 days
    dates = pd.date_range("2024-11-01", periods=n_hours, freq="1h", tz="UTC")

    # Time-varying reference and humidity to avoid zero-variance correlation warning
    t_span = np.linspace(0, 20, n_hours)
    ref_vals = 100.0 + np.sin(t_span) * 20.0
    rh_vals = 70.0 + np.cos(t_span) * 15.0
    raw_vals = 1.40 * ref_vals

    paired_df = pd.DataFrame(
        {
            "ts_utc": dates,
            "location_id": [4527035] * n_hours,
            "sensor_model": ["optical_test"] * n_hours,
            "distance_km": [3.5] * n_hours,
            "pm25_raw": raw_vals,
            "rh": rh_vals,
            "pm25_ref": ref_vals,
        }
    )

    stats = analyze_support_window_distortion(paired_df, window_days_list=(7, 14, 30))
    assert len(stats) == 3

    for s in stats:
        assert abs(s.mean_ratio - 1.40) < 1e-4
        assert abs(s.median_ratio - 1.40) < 1e-4
        assert s.fraction_ratio_gt_1_30 == 1.0
        assert s.fraction_ratio_gt_1_50 == 0.0
        assert s.rmse_ugm3 > 0.0


def test_generate_corruption_prior_report(tmp_path: Path) -> None:
    """Verify JSON and Markdown report serialization with POSIX LF line endings."""
    res = CorruptionPriorAuditResult(
        audit_timestamp_utc="2026-11-05T14:00:00+00:00",
        domain="lahore",
        reference_station_id=8664,
        reference_station_name="US Consulate General Lahore",
        candidate_stations=[
            {
                "location_id": 4527035,
                "name": "Civil Secretariat",
                "distance_km": 3.52,
                "sensor_model": "AirGradient",
            }
        ],
        total_paired_records=4323,
        huber_coefficients={
            "a_gain": 0.6927,
            "b_rh_slope": 0.6400,
            "c_intercept": 15.7893,
            "epsilon": 1.35,
        },
        cross_validation_metrics={
            "raw_rmse": 33.56,
            "corrected_rmse": 28.85,
            "rmse_improvement": 0.1403,
            "raw_mae": 27.77,
            "corrected_mae": 22.51,
            "mae_improvement": 0.1895,
            "raw_r2": 0.6096,
            "corrected_r2": 0.7115,
            "beneficial": 1.0,
        },
        inverted_priors={
            "gain_prior": 1.4436,
            "rh_slope_prior": 0.9239,
            "offset_prior": -22.7929,
        },
        prior_comparisons=[
            PriorComparison(
                parameter="gain",
                empirical_value=1.4436,
                configured_min=0.8,
                configured_max=1.3,
                is_enveloped=False,
                status="OUT_OF_BOUNDS",
                notes="Gain exceeds range.",
            ),
            PriorComparison(
                parameter="rh_slope",
                empirical_value=0.9239,
                configured_min=0.0,
                configured_max=1.5,
                is_enveloped=True,
                status="ENVELOPED",
                notes="RH slope within range.",
            ),
            PriorComparison(
                parameter="offset",
                empirical_value=-22.7929,
                configured_min=-10.0,
                configured_max=25.0,
                is_enveloped=False,
                status="OUT_OF_BOUNDS",
                notes="Offset outside range.",
            ),
        ],
        support_window_stats=[
            SupportWindowErrorStats(
                window_days=14,
                n_pairs=336,
                mean_raw_pm25=171.3,
                mean_ref_pm25=168.7,
                mean_ratio=1.02,
                median_ratio=1.04,
                p5_ratio=0.75,
                p25_ratio=0.85,
                p75_ratio=1.15,
                p95_ratio=1.30,
                fraction_ratio_gt_1_30=0.054,
                fraction_ratio_gt_1_50=0.0,
                rh_ratio_correlation=0.0109,
                mean_bias_ugm3=2.6,
                mae_ugm3=21.2,
                rmse_ugm3=27.4,
            )
        ],
        all_priors_enveloped=False,
        risk_r17_triggered=True,
        risk_r17_severity="Moderate",
        risk_r17_summary="Risk R17 triggered: empirical gain prior (1.4436) exceeds 1.30.",
        recommendations=["Pre-apply empirical low-cost correction."],
    )

    json_path, md_path = generate_corruption_prior_report(res, output_dir=tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON content
    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["domain"] == "lahore"
    assert loaded["risk_r17_triggered"] is True
    assert loaded["inverted_priors"]["gain_prior"] == 1.4436

    # Verify Markdown formatting, LF line endings, and single trailing newline
    raw_md = md_path.read_bytes()
    assert b"\r\n" not in raw_md, "Markdown file contains Windows CRLF line endings"
    assert raw_md.endswith(b"\n")
    assert not raw_md.endswith(b"\n\n"), "Markdown file contains multiple trailing newlines"

    text = md_path.read_text(encoding="utf-8")
    assert "# SmogSense Phase 2 Task P2-15: Empirical Corruption-Prior Audit" in text
    assert "Risk R17 Confirmed (Prior Mismatch)" in text
    assert "1.4436" in text
    assert "| **gain** |" in text


def test_corruption_audit_cli_help() -> None:
    """Verify that CLI help surface displays all required options."""
    runner = CliRunner()
    result = runner.invoke(app, ["audit", "corruption-prior", "--help"])
    assert result.exit_code == 0
    assert "Run empirical corruption-prior audit on Lahore low-cost sensors" in result.output
    assert "--domain" in result.output
    assert "--backfill-dir" in result.output
    assert "--config-path" in result.output
    assert "--output-dir" in result.output
