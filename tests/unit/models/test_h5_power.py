"""tests/unit/models/test_h5_power.py — Unit tests for H5 Statistical Power & Pilot Study.

Specification:
- docs/evaluation-strategy.md §1, §6, §8, §10
- docs/ml-architecture.md §8 & §10
- ADR-024
- roadmap.md Task P2-14

Verifies:
1. Effective sample size n_eff calculation and lag-1 autocorrelation rho_1 clamping.
2. Analytical statistical power computation under normal approximation and target delta.
3. Power sensitivity surface generation and monotonic properties.
4. Station eligibility audit logic for Delhi Season T (>= 20 station threshold).
5. Condition summary aggregation for H5 (clean) and H5-C (corrupted prior).
6. JSON and Markdown report serialization with strict LF line endings and single trailing newline.
7. Typer CLI runner integration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.h5_power import (
    H5_TARGET_DELTA,
    MIN_ELIGIBLE_STATIONS,
    TARGET_CRPS_RATIO,
    H5ConditionSummary,
    H5PowerReport,
    PowerGridCell,
    StationEligibilityAudit,
    audit_season_t_stations,
    calculate_normal_power,
    compute_effective_sample_size,
    generate_h5_power_report,
    generate_power_sensitivity_grid,
    summarize_condition_metrics,
)


# -----------------------------------------------------------------------------
# 1. Effective Sample Size & Autocorrelation Tests
# -----------------------------------------------------------------------------
def test_compute_effective_sample_size_zero_autocorr() -> None:
    """When rho_1 is zero, n_eff should equal sample size n."""
    # An alternating sequence with zero lag-1 covariance
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    rho_1, _ = compute_effective_sample_size(x)
    assert -1.0 <= rho_1 <= 1.0
    # For small n or independent draws
    rng = np.random.default_rng(123)
    white_noise = rng.normal(0.0, 1.0, size=500)
    rho_wn, n_eff_wn = compute_effective_sample_size(white_noise)
    assert abs(rho_wn) < 0.15
    assert abs(n_eff_wn - 500) < 150


def test_compute_effective_sample_size_known_positive() -> None:
    """For positive autocorrelation rho_1 = 0.5, n_eff should be approximately n / 3."""
    # Synthetic AR(1) series: x_t = 0.5 * x_{t-1} + e_t
    n = 2000
    rng = np.random.default_rng(42)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0.0, 1.0)

    rho_1, n_eff = compute_effective_sample_size(x)
    assert 0.45 <= rho_1 <= 0.55
    # (1 - 0.5) / (1 + 0.5) = 1/3 => n_eff approx n / 3
    expected_n_eff = n * (1.0 - rho_1) / (1.0 + rho_1)
    assert pytest.approx(n_eff, rel=1e-3) == expected_n_eff


def test_compute_effective_sample_size_corner_cases() -> None:
    """Handles empty, single-element, constant, and extreme values gracefully."""
    # Empty
    rho_0, n_0 = compute_effective_sample_size([])
    assert rho_0 == 0.0
    assert n_0 == 1.0

    # Single element
    rho_1, n_1 = compute_effective_sample_size([42.0])
    assert rho_1 == 0.0
    assert n_1 == 1.0

    # Constant sequence (variance = 0)
    rho_c, n_c = compute_effective_sample_size([5.0, 5.0, 5.0, 5.0])
    assert rho_c == 0.0
    assert n_c == 4.0

    # Clamping behavior on extreme persistence
    perfect_pers = np.linspace(1.0, 100.0, 100)
    rho_p, n_p = compute_effective_sample_size(perfect_pers, clamp_range=(-0.99, 0.99))
    assert rho_p <= 0.99
    assert n_p >= 1.0


# -----------------------------------------------------------------------------
# 2. Analytical Normal Power Tests
# -----------------------------------------------------------------------------
def test_calculate_normal_power_null_effect() -> None:
    """When delta = 0 (no true effect), power should equal alpha (0.05)."""
    p = calculate_normal_power(delta=0.0, sigma=0.30, n_eff=30.0, alpha=0.05)
    assert pytest.approx(p, abs=1e-3) == 0.05


def test_calculate_normal_power_asymptotics() -> None:
    """Very large n_eff or delta approaches power = 1.0; zero n_eff yields 0.0."""
    # Huge sample size
    p_large = calculate_normal_power(delta=H5_TARGET_DELTA, sigma=0.20, n_eff=10000.0)
    assert pytest.approx(p_large, abs=1e-4) == 1.0

    # Zero sample size
    p_zero = calculate_normal_power(delta=H5_TARGET_DELTA, sigma=0.20, n_eff=0.0)
    assert p_zero == 0.0

    # Zero sigma with positive delta
    p_sig0 = calculate_normal_power(delta=0.05, sigma=0.0, n_eff=25.0)
    assert p_sig0 == 1.0


def test_calculate_normal_power_known_hand_values() -> None:
    """Matches the hand-computed normal approximation in docs/evaluation-strategy.md §1."""
    # From docs/evaluation-strategy.md §1:
    # Effect delta = 0.1625 (H1), n_eff = 28, z_0.95 = 1.645
    # sigma = 0.3 => stat = 0.1625 * sqrt(28) / 0.3 - 1.645 = 2.865 - 1.645 = 1.22
    # Phi(1.22) approx 0.8888 -> round(0.89)
    p_h1 = calculate_normal_power(delta=0.1625, sigma=0.30, n_eff=28.0)
    assert pytest.approx(p_h1, abs=0.01) == 0.89

    # H2: delta = 0.1054, sigma = 0.30, n_eff = 28 => stat = 0.1054 * 5.2915 / 0.3 - 1.645 = 1.859 - 1.645 = 0.214
    # Phi(0.214) approx 0.5847 -> round(0.59)
    p_h2 = calculate_normal_power(delta=0.1054, sigma=0.30, n_eff=28.0)
    assert pytest.approx(p_h2, abs=0.01) == 0.59


# -----------------------------------------------------------------------------
# 3. Power Sensitivity Grid Monotonicity Tests
# -----------------------------------------------------------------------------
def test_power_sensitivity_grid_properties() -> None:
    """Sensitivity grid exhibits correct monotonic behavior across sigma and n_eff."""
    sigmas = (0.15, 0.25, 0.35, 0.50)
    n_eff_vals = (20.0, 30.0, 50.0)
    grid = generate_power_sensitivity_grid(
        sigmas=sigmas, n_eff_values=n_eff_vals, delta=H5_TARGET_DELTA
    )

    assert len(grid) == len(sigmas) * len(n_eff_vals)

    # Convert to lookup (sigma, n_eff) -> power
    lookup = {(c.sigma, c.n_eff): c.power for c in grid}

    # 1. Monotonic increasing in n_eff for fixed sigma
    for s in sigmas:
        p20 = lookup[(s, 20.0)]
        p30 = lookup[(s, 30.0)]
        p50 = lookup[(s, 50.0)]
        assert p20 <= p30 <= p50, f"Expected monotonic increase in n_eff for sigma={s}"

    # 2. Monotonic decreasing in sigma for fixed n_eff
    for n in n_eff_vals:
        p15 = lookup[(0.15, n)]
        p25 = lookup[(0.25, n)]
        p35 = lookup[(0.35, n)]
        p50 = lookup[(0.50, n)]
        assert p15 >= p25 >= p35 >= p50, f"Expected monotonic decrease in sigma for n_eff={n}"


# -----------------------------------------------------------------------------
# 4. Station Eligibility Audit Tests
# -----------------------------------------------------------------------------
def test_audit_season_t_stations_threshold_met() -> None:
    """Verifies threshold_met=True when >= 20 eligible stations are present."""
    n_stations = 25
    n_dates = 10
    dates = pd.date_range("2025-11-01", periods=n_dates, freq="D", tz="UTC")

    records = [
        {
            "location_id": stn_id,
            "issuance_utc": dt,
            "valid_h24": True,
            "valid_h48": True,
            "valid_h72": True,
        }
        for stn_id in range(101, 101 + n_stations)
        for dt in dates
    ]
    df_synthetic = pd.DataFrame(records)

    audit = audit_season_t_stations(
        df_season_t=df_synthetic,
        registry_path=None,
        min_eligible_threshold=MIN_ELIGIBLE_STATIONS,
    )

    assert audit.total_stations == 25
    assert audit.eligible_stations == 25
    assert audit.min_threshold == 20
    assert audit.threshold_met is True
    assert len(audit.station_details) == 25


def test_audit_season_t_stations_threshold_unmet() -> None:
    """Verifies threshold_met=False when fewer than 20 eligible stations are present."""
    n_stations = 12
    records = [
        {
            "location_id": stn_id,
            "issuance_utc": f"2025-11-0{d + 1}T00:00:00Z",
            "valid_h24": True,
        }
        for stn_id in range(101, 101 + n_stations)
        for d in range(5)
    ]
    df_synthetic = pd.DataFrame(records)

    audit = audit_season_t_stations(
        df_season_t=df_synthetic,
        registry_path=None,
        min_eligible_threshold=20,
    )

    assert audit.total_stations == 12
    assert audit.eligible_stations == 12
    assert audit.threshold_met is False


# -----------------------------------------------------------------------------
# 5. Condition Metrics Summarizer Tests
# -----------------------------------------------------------------------------
def test_summarize_condition_metrics_pass() -> None:
    """Condition metrics properly summarize passing H5 criteria."""
    # 25 evaluation days with slight improvement for M7: ratio approx 0.95
    c6_vals = [30.0 + i * 0.1 for i in range(25)]
    c7_vals = [c * 0.95 for c in c6_vals]
    d_vals = [math.log(c7 / c6) for c6, c7 in zip(c6_vals, c7_vals, strict=True)]

    summary = summarize_condition_metrics(
        condition_name="H5",
        description="Clean regulatory test",
        daily_crps_m6=c6_vals,
        daily_crps_m7=c7_vals,
        daily_d=d_vals,
        target_delta=H5_TARGET_DELTA,
    )

    assert summary.n_eval_days == 25
    assert summary.crps_ratio_point <= TARGET_CRPS_RATIO
    assert summary.verdict_point_met is True
    assert summary.ci_95_ratio_upper < 1.00
    assert summary.verdict_ci_met is True
    assert summary.verdict_passed is True
    assert summary.analytical_power > 0.50


def test_summarize_condition_metrics_fail() -> None:
    """Condition metrics properly identify failing H5 criteria."""
    # M7 worse than M6 (ratio 1.02)
    c6_vals = [30.0 for _ in range(20)]
    c7_vals = [30.6 for _ in range(20)]
    d_vals = [math.log(c7 / c6) for c6, c7 in zip(c6_vals, c7_vals, strict=True)]

    summary = summarize_condition_metrics(
        condition_name="H5",
        description="Failing test",
        daily_crps_m6=c6_vals,
        daily_crps_m7=c7_vals,
        daily_d=d_vals,
        target_delta=H5_TARGET_DELTA,
    )

    assert summary.crps_ratio_point > TARGET_CRPS_RATIO
    assert summary.verdict_point_met is False
    assert summary.verdict_passed is False


# -----------------------------------------------------------------------------
# 6. Report Serialization & Trailing Newline Hygiene Tests
# -----------------------------------------------------------------------------
def test_generate_and_save_reports_roundtrip(tmp_path: Path) -> None:
    """Verifies JSON and Markdown report serialization and line ending hygiene."""
    audit = StationEligibilityAudit(
        total_stations=25,
        eligible_stations=24,
        min_threshold=20,
        threshold_met=True,
        station_details={
            "8664": {
                "name": "StateAir Delhi",
                "valid_samples": 120,
                "uptime_fraction": 0.95,
                "is_reference": True,
                "eligible": True,
            }
        },
    )

    h5_cln = H5ConditionSummary(
        condition_name="H5",
        description="Clean regulatory data",
        n_eval_days=25,
        mean_crps_m6=28.5,
        mean_crps_m7=27.2,
        crps_ratio_point=0.9544,
        d_bar=-0.04667,
        sigma_d=0.0821,
        rho_1=0.215,
        n_eff=16.14,
        ci_95_ratio_lower=0.9168,
        ci_95_ratio_upper=0.9935,
        analytical_power=0.685,
        verdict_point_met=True,
        verdict_ci_met=True,
        verdict_passed=True,
    )

    h5_crp = H5ConditionSummary(
        condition_name="H5-C",
        description="Corrupted support inputs",
        n_eval_days=25,
        mean_crps_m6=30.1,
        mean_crps_m7=28.4,
        crps_ratio_point=0.9435,
        d_bar=-0.05814,
        sigma_d=0.0912,
        rho_1=0.180,
        n_eff=17.37,
        ci_95_ratio_lower=0.9038,
        ci_95_ratio_upper=0.9850,
        analytical_power=0.720,
        verdict_point_met=True,
        verdict_ci_met=True,
        verdict_passed=True,
    )

    grid = [
        PowerGridCell(sigma=0.15, n_eff=20.0, power=0.85),
        PowerGridCell(sigma=0.25, n_eff=20.0, power=0.62),
    ]

    report = H5PowerReport(
        generated_at_utc="2026-11-05T12:00:00Z",
        domain="delhi",
        season="2025-2026",
        h5_target_delta=round(H5_TARGET_DELTA, 5),
        target_crps_ratio=TARGET_CRPS_RATIO,
        station_audit=audit,
        h5_clean=h5_cln,
        h5_corrupted=h5_crp,
        power_sensitivity_grid=grid,
        recommendations={"retain_pre_registered_threshold": True},
    )

    json_path, md_path = generate_h5_power_report(report, output_dir=tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    # Read back JSON
    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["domain"] == "delhi"
    assert loaded["season"] == "2025-2026"
    assert loaded["station_audit"]["eligible_stations"] == 24

    # Check Unix LF and single trailing newline hygiene on JSON
    raw_json_bytes = json_path.read_bytes()
    assert b"\r" not in raw_json_bytes
    assert raw_json_bytes.endswith(b"\n")
    assert not raw_json_bytes.endswith(b"\n\n")

    # Check Unix LF and single trailing newline hygiene on Markdown
    raw_md_bytes = md_path.read_bytes()
    assert b"\r" not in raw_md_bytes
    assert raw_md_bytes.endswith(b"\n")
    assert not raw_md_bytes.endswith(b"\n\n")

    md_text = md_path.read_text(encoding="utf-8")
    assert (
        "# SmogSense Phase 2 Task P2-14: H5 Statistical Power & Pilot Study on Season T" in md_text
    )
    assert "Condition H5 (Clean Regulatory)" in md_text
    assert "Condition H5-C (Corrupted Inputs Prior)" in md_text


# -----------------------------------------------------------------------------
# 7. CLI Integration Test
# -----------------------------------------------------------------------------
def test_cli_h5_power_help() -> None:
    """Verifies Typer CLI help output for 'evaluate h5-power'."""
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "h5-power", "--help"])
    assert result.exit_code == 0
    assert "Evaluate Hypothesis H5 statistical power" in result.stdout
    assert "--domain" in result.stdout
    assert "--season" in result.stdout
    assert "--output-dir" in result.stdout
