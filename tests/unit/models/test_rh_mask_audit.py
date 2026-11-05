"""tests/unit/models/test_rh_mask_audit.py — Unit tests for RH-Mask Audit engine.

Specification:
- docs/ml-architecture.md §8 (Condition C9, Risk R14)
- docs/data-engineering.md §5
- configs/model.yaml (lines 134-139)
- roadmap.md Task P2-13

Verifies:
1. Mask filter logic: drops low-cost support readings where RH > 85% when >= 50% survives.
2. Survival threshold: skips mask when > 50% of support readings exceed 85% RH.
3. Reference monitor invariance: station 8664 / is_reference=True is never masked.
4. Mathematical exactness of RH bin distribution and extreme smog coincidence.
5. Upper-quantile pinball and CRPS evaluation under unmasked and masked conditions.
6. JSON and Markdown report serialization with Unix LF line endings.
7. CLI integration via Typer runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.heads import ResidualCalibrator
from smogsense.models.maml import MetaSGDStepSizes
from smogsense.models.rh_mask_audit import (
    REFERENCE_MONITOR_ID,
    UPPER_QUANTILES,
    ExtremeSmogCoincidenceSummary,
    RHBinSummary,
    RHConditionMetrics,
    RHDistributionSummary,
    RHMaskAuditReport,
    RHMaskImpactSummary,
    SmogThresholdCoincidence,
    analyze_rh_distribution_and_extremes,
    apply_rh_support_mask,
    evaluate_rh_mask_impact,
    generate_rh_mask_report,
)


def _create_synthetic_lahore_feature_df(
    n_days: int = 30,
    n_stations: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic Lahore feature dataframe with co-measured RH and target values."""
    rng = np.random.default_rng(seed)
    station_ids = [REFERENCE_MONITOR_ID] + [7100 + i for i in range(1, n_stations)]
    n_hours = n_days * 4  # 4 issuances per day (00, 06, 12, 18 UTC)
    base_time = pd.Timestamp("2025-10-01 00:00:00", tz="UTC")

    records = []
    for sid in station_ids:
        is_ref = sid == REFERENCE_MONITOR_ID
        for h in range(n_hours):
            t = base_time + pd.Timedelta(hours=h * 6)
            rh_val = float(rng.uniform(40.0, 95.0))
            pm_val = float(rng.uniform(50.0, 300.0))

            row: dict[str, object] = {
                "domain": "lahore",
                "location_id": sid,
                "issuance_utc": t,
                "is_reference": is_ref,
                "is_monitor": is_ref,
                "rh_pct": rh_val,
                "f_met_rh_mean_h24": rh_val,
                "pm25_ugm3": pm_val,
                "pm25_raw_ugm3": pm_val,
                "y_h24": pm_val,
                "valid_h24": True,
                "y_h48": pm_val * 0.95,
                "valid_h48": True,
                "y_h72": pm_val * 0.90,
                "valid_h72": True,
                "f_stat_latitude": 31.56,
                "f_stat_longitude": 74.34,
                "f_stat_elevation_m": 215.0,
                "f_stat_dist_city_center_km": 3.5,
                "f_stat_is_reference": 1.0 if is_ref else 0.0,
            }

            # Add context features
            for i in range(15):
                row[f"context_{i}"] = float(rng.standard_normal())
            for hi in (24, 48, 72):
                row[f"cams_pm25_h{hi}"] = pm_val * 0.8
                for k in range(19):
                    row[f"base_pred_h{hi}_q{k}"] = float(np.log1p(pm_val))

            records.append(row)

    df = pd.DataFrame(records)
    df["issuance_utc"] = pd.to_datetime(df["issuance_utc"], utc=True)
    return df.sort_values("issuance_utc").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 1. Mask Filter Logic & Survival Threshold Tests
# -----------------------------------------------------------------------------


def test_apply_rh_support_mask_drops_high_rh() -> None:
    """Verify that low-cost readings with RH > 85% are dropped when >= 50% survive."""
    df = pd.DataFrame(
        {
            "location_id": [1001] * 10,
            "is_reference": [False] * 10,
            "rh_pct": [60.0, 65.0, 70.0, 75.0, 80.0, 84.0, 85.0, 86.0, 90.0, 95.0],
            "pm25_ugm3": [100.0] * 10,
        }
    )

    # 7 out of 10 have RH <= 85.0 (70% survival >= 50%) -> mask applied
    filtered, mask_applied = apply_rh_support_mask(df, rh_threshold=85.0, min_support_fraction=0.5)

    assert mask_applied is True
    assert len(filtered) == 7
    assert (filtered["rh_pct"] <= 85.0).all()


def test_apply_rh_support_mask_skips_when_survival_under_50pct() -> None:
    """Verify that mask is skipped when < 50% of low-cost support readings survive."""
    df = pd.DataFrame(
        {
            "location_id": [1001] * 10,
            "is_reference": [False] * 10,
            "rh_pct": [60.0, 70.0, 80.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0],
            "pm25_ugm3": [100.0] * 10,
        }
    )

    # Only 3 out of 10 have RH <= 85.0 (30% survival < 50%) -> mask skipped
    filtered, mask_applied = apply_rh_support_mask(df, rh_threshold=85.0, min_support_fraction=0.5)

    assert mask_applied is False
    assert len(filtered) == 10  # All readings survive to avoid starved adaptation


def test_apply_rh_support_mask_reference_monitor_invariance() -> None:
    """Verify that reference monitor (8664 or is_reference=True) is NEVER masked."""
    df = pd.DataFrame(
        {
            "location_id": [REFERENCE_MONITOR_ID, REFERENCE_MONITOR_ID, 1001, 1001],
            "is_reference": [True, True, False, False],
            "rh_pct": [95.0, 98.0, 70.0, 95.0],
            "pm25_ugm3": [200.0, 250.0, 150.0, 220.0],
        }
    )

    # For low-cost (1001): 1 survives out of 2 (50% survival >= 50%) -> mask applied
    filtered, mask_applied = apply_rh_support_mask(
        df,
        rh_threshold=85.0,
        min_support_fraction=0.5,
        ref_station_ids=(REFERENCE_MONITOR_ID,),
    )

    assert mask_applied is True
    assert len(filtered) == 3

    # Reference monitor rows with RH 95% and 98% MUST BOTH BE KEPT
    ref_rows = filtered[filtered["location_id"] == REFERENCE_MONITOR_ID]
    assert len(ref_rows) == 2

    # Low-cost row with RH 95% was dropped; row with 70% was kept
    lc_rows = filtered[filtered["location_id"] == 1001]
    assert len(lc_rows) == 1
    assert lc_rows.iloc[0]["rh_pct"] == 70.0


def test_apply_rh_support_mask_edge_cases() -> None:
    """Verify edge cases: empty dataframe, missing RH column, and reference-only data."""
    # 1. Empty DataFrame
    empty_df = pd.DataFrame()
    f_empty, applied_empty = apply_rh_support_mask(empty_df)
    assert f_empty.empty
    assert applied_empty is False

    # 2. Reference-only DataFrame
    ref_df = pd.DataFrame(
        {
            "location_id": [REFERENCE_MONITOR_ID] * 5,
            "rh_pct": [90.0] * 5,
            "is_reference": [True] * 5,
        }
    )
    f_ref, applied_ref = apply_rh_support_mask(ref_df)
    assert len(f_ref) == 5
    assert applied_ref is False

    # 3. Missing RH column
    no_rh_df = pd.DataFrame({"location_id": [1001] * 5, "pm25": [100.0] * 5})
    f_no_rh, applied_no_rh = apply_rh_support_mask(no_rh_df)
    assert len(f_no_rh) == 5
    assert applied_no_rh is False


# -----------------------------------------------------------------------------
# 2. Distribution & Extreme Coincidence Math Tests
# -----------------------------------------------------------------------------


def test_analyze_rh_distribution_and_extremes_math() -> None:
    """Verify exact bin counts, percentages, and extreme coincidence calculation."""
    # Construct exact known distribution: 100 low-cost observations
    # 25 in < 50%, 25 in 50-70%, 25 in 70-85%, 25 in > 85%
    b1_rh = [40.0] * 25
    b2_rh = [60.0] * 25
    b3_rh = [75.0] * 25
    b4_rh = [90.0] * 25
    all_rh = b1_rh + b2_rh + b3_rh + b4_rh

    # PM2.5 values:
    # In b3 (75% RH): 10 rows with PM=180, 15 with PM=50
    # In b4 (90% RH): 10 rows with PM=180, 10 with PM=300, 5 with PM=50
    # In b1 & b2: all PM=50
    pm_vals = [50.0] * 50  # b1, b2
    pm_vals += [180.0] * 10 + [50.0] * 15  # b3
    pm_vals += [180.0] * 10 + [300.0] * 10 + [50.0] * 5  # b4

    df = pd.DataFrame(
        {
            "location_id": [1001] * 100,
            "is_reference": [False] * 100,
            "rh_pct": all_rh,
            "pm25_ugm3": pm_vals,
        }
    )

    dist_summary, coin_summary = analyze_rh_distribution_and_extremes(
        df,
        rh_col="rh_pct",
        pm_col="pm25_ugm3",
        ref_station_ids=(REFERENCE_MONITOR_ID,),
    )

    # 1. Distribution Checks
    assert dist_summary.total_hours == 100
    assert len(dist_summary.bins) == 4
    for b in dist_summary.bins:
        assert b.count == 25
        assert b.percentage == 25.0

    assert dist_summary.min_rh == 40.0
    assert dist_summary.max_rh == 90.0

    # 2. Extreme Smog Coincidence Checks
    # PM >= 150: 10 in b3 (RH=75) + 10 in b4 (RH=90) + 10 in b4 (RH=90) = 30 total
    # Of these 30, 20 are in b4 (RH=90 > 85%) -> fraction = 20 / 30 = 2/3 = 66.67%
    assert coin_summary.threshold_150.total_hours == 30
    assert coin_summary.threshold_150.hours_high_rh == 20
    assert pytest.approx(coin_summary.threshold_150.fraction_high_rh, abs=1e-3) == 0.6667
    assert pytest.approx(coin_summary.threshold_150.percentage_high_rh, abs=0.1) == 66.67

    # PM >= 250: 10 in b4 (RH=90 > 85%) = 10 total
    # All 10 are in b4 -> fraction = 10 / 10 = 100.0%
    assert coin_summary.threshold_250.total_hours == 10
    assert coin_summary.threshold_250.hours_high_rh == 10
    assert coin_summary.threshold_250.fraction_high_rh == 1.0
    assert coin_summary.threshold_250.percentage_high_rh == 100.0


# -----------------------------------------------------------------------------
# 3. Adaptation Impact Benchmark Tests
# -----------------------------------------------------------------------------


def test_evaluate_rh_mask_impact_synthetic() -> None:
    """Verify adaptation impact benchmarking on synthetic Lahore feature data."""
    df_lahore = _create_synthetic_lahore_feature_df(n_days=25, n_stations=2, seed=42)

    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    impact = evaluate_rh_mask_impact(
        df_lahore=df_lahore,
        calibrator=calibrator,
        step_sizes=step_sizes,
        rh_threshold=85.0,
        min_support_fraction=0.5,
        ref_station_ids=(REFERENCE_MONITOR_ID,),
        n_windows=3,
        support_days=10,
        eval_days=5,
        inner_steps=2,
        seed=42,
    )

    assert isinstance(impact, RHMaskImpactSummary)
    assert impact.n_windows == 3
    assert impact.unmasked.upper_quantile_pinball >= 0.0
    assert impact.masked.upper_quantile_pinball >= 0.0
    assert impact.unmasked.crps >= 0.0
    assert impact.masked.crps >= 0.0
    assert len(impact.unmasked.per_quantile_pinball) == len(UPPER_QUANTILES)
    assert len(impact.masked.per_quantile_pinball) == len(UPPER_QUANTILES)
    assert "Condition C9 Decision:" in impact.recommendation


# -----------------------------------------------------------------------------
# 4. Report Serialization & Invariant Tests
# -----------------------------------------------------------------------------


def test_generate_and_save_reports_roundtrip(tmp_path: Path) -> None:
    """Verify that JSON and Markdown reports serialize cleanly with Unix LF line endings."""
    dist = RHDistributionSummary(
        total_hours=1000,
        mean_rh=65.0,
        std_rh=15.0,
        min_rh=20.0,
        median_rh=66.0,
        p75_rh=78.0,
        p90_rh=86.0,
        p95_rh=92.0,
        max_rh=98.0,
        bins=[
            RHBinSummary("< 50%", 200, 20.0),
            RHBinSummary("50-70%", 400, 40.0),
            RHBinSummary("70-85%", 250, 25.0),
            RHBinSummary("> 85%", 150, 15.0),
        ],
        by_station={"8664": {"mean_rh": 64.0}},
    )

    coin = ExtremeSmogCoincidenceSummary(
        threshold_150=SmogThresholdCoincidence(150.0, 300, 90, 0.30, 30.0),
        threshold_250=SmogThresholdCoincidence(250.0, 100, 42, 0.42, 42.0),
        by_station={},
    )

    unmasked_m = RHConditionMetrics("no_mask", 8.50, 14.20, 11.30, {"tau_0.90": 9.10}, 200, 40)
    masked_m = RHConditionMetrics("with_rh_mask", 8.85, 14.65, 12.10, {"tau_0.90": 9.55}, 200, 40)

    imp = RHMaskImpactSummary(
        n_windows=5,
        unmasked=unmasked_m,
        masked=masked_m,
        delta_upper_pinball=0.35,
        rel_delta_upper_pinball_pct=4.12,
        delta_crps=0.45,
        rel_delta_crps_pct=3.17,
        delta_high_rh_pinball=0.80,
        rel_delta_high_rh_pinball_pct=7.08,
        mask_applied_count=4,
        mask_skipped_count=1,
        r14_truncation_detected=True,
        recommendation="Keep adaptation.rh_mask.enabled = false.",
    )

    report = RHMaskAuditReport(
        domain="lahore",
        generated_at_utc="2026-11-05T12:00:00Z",
        rh_distribution=dist,
        extreme_smog_coincidence=coin,
        adaptation_impact=imp,
        policy_decision={"condition": "C9", "rh_mask_enabled": False},
    )

    json_path, md_path = generate_rh_mask_report(report, output_dir=tmp_path)

    # 1. Assert Files Exist
    assert json_path.exists()
    assert md_path.exists()

    # 2. Assert Valid JSON Roundtrip
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["domain"] == "lahore"
    assert data["policy_decision"]["rh_mask_enabled"] is False
    assert len(data["rh_distribution"]["bins"]) == 4

    # 3. Assert Markdown Formatting & Unix Line Endings
    with md_path.open("rb") as f:
        raw_bytes = f.read()
    assert b"\r\n" not in raw_bytes  # Pure LF line endings check

    md_text = raw_bytes.decode("utf-8")
    assert (
        "# SmogSense Phase 2 Task P2-13: RH-Mask Audit on Lahore Low-Cost Support Hours" in md_text
    )
    assert "Risk R14" in md_text
    assert "Policy Finding" in md_text
    assert "adaptation.rh_mask.enabled: false" in md_text


# -----------------------------------------------------------------------------
# 5. CLI Integration Tests
# -----------------------------------------------------------------------------


def test_cli_rh_mask_audit_help() -> None:
    """Verify that CLI evaluate rh-mask-audit command exposes all required options."""
    runner = CliRunner()
    result = runner.invoke(app, ["evaluate", "rh-mask-audit", "--help"])
    assert result.exit_code == 0
    assert "--rh-threshold" in result.output
    assert "--min-support-fraction" in result.output
    assert "--feature-store-dir" in result.output
    assert "--output-dir" in result.output
