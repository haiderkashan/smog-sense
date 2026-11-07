"""Unit & pipeline integration tests for Challenger Mode (Task P3-07).

Specifications:
- docs/deployment-and-ops.md (§7 Release and model promotion, §8 Alerting)
- docs/evaluation-strategy.md (§9 Prospective shadow-run protocol)
- docs/ml-architecture.md (§6, §8, Condition C6, Condition C7)
- configs/model.yaml (challenger block, promoted_bundle)

Verifies:
1. test_challenger_shadow_logging_unpromoted: Challenger forecasts appended to forecast_log,
   while public bulletin (latest.json) isolates and renders baseline strictly.
2. test_challenger_n0_sensor_free_fallback: When observations are empty (N=0), challenger
   enforces Condition C7: adaptation_status = "skipped_n0" with base model predictions.
3. test_challenger_adaptation_status_recording: Valid adaptation statuses recorded in both
   forecast_log.parquet and dual-written manifest files (.state/manifests/*.json).
4. test_forecast_log_schema_conformance: Multi-method ledger conforms strictly to
   forecast_log.schema.yaml with compound uniqueness and valid enums.
5. test_promoted_model_bulletin_switch: When promoted_bundle is set, public bulletin switches
   to publishing the candidate model instead of the baseline.
6. test_challenger_error_isolation: Zero-crash guarantee - challenger execution errors never
   abort the pipeline; baseline completes and publishes safely.
7. test_cli_challenger_options: CLI runner supports --enable-challenger, --no-challenger,
   --challenger-method, and --challenger-bundle flags.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.pipeline.orchestrator import (
    load_pandera_schema,
    run_challenger_forecast,
    run_daily_pipeline,
)
from smogsense.utils.io import validate_frame


@pytest.fixture
def clean_pipeline_state() -> Generator[None, None, None]:
    """Clean state directory before and after tests."""
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
        Path(".state/inputs/lahore"),
        Path("site/forecast"),
    ]
    files = [
        Path("site/latest.json"),
        Path("site/index.html"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()

    yield

    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()


def test_challenger_shadow_logging_unpromoted(clean_pipeline_state: None) -> None:
    """Test challenger forecasts are logged in forecast_log while latest.json strictly displays baseline."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"

    # Run daily pipeline in fixtures mode with unpromoted default
    exit_code = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
    assert exit_code in (0, 10)

    # 1. Verify forecast log contains BOTH baseline and challenger methods
    forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    assert forecast_path.exists(), f"Missing forecast parquet at {forecast_path}"
    df = pd.read_parquet(forecast_path)

    methods = set(df["method"].unique())
    assert "m1_cams_raw" in methods, "Baseline method m1_cams_raw missing from forecast_log"
    assert "m7_hybrid_maml" in methods, "Challenger method m7_hybrid_maml missing from forecast_log"

    # Baseline rows have model_version = "baseline"
    baseline_rows = df[df["method"] == "m1_cams_raw"]
    assert (baseline_rows["model_version"] == "baseline").all()

    # Challenger rows have challenger model_version
    challenger_rows = df[df["method"] == "m7_hybrid_maml"]
    assert len(challenger_rows) > 0
    assert (challenger_rows["model_version"] != "baseline").all()

    # 2. Verify latest.json bulletin renders baseline strictly
    bulletin_path = Path("site/forecast/latest.json")
    assert bulletin_path.exists(), f"Missing bulletin at {bulletin_path}"
    with bulletin_path.open("r", encoding="utf-8") as f:
        bulletin = json.load(f)

    assert bulletin["model"]["id"] == "m1_cams_raw", (
        f"Bulletin must display baseline, got {bulletin['model']}"
    )
    assert bulletin["model"]["version"] == "baseline"


def test_challenger_n0_sensor_free_fallback() -> None:
    """Test Condition C7: N=0 sensor-free fallback sets adaptation_status = 'skipped_n0'."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    empty_obs = pd.DataFrame()
    stations_df = pd.DataFrame(
        [{"location_id": "1", "name": "Station 1", "lat": 31.5, "lon": 74.3}]
    )
    cams_df = pd.DataFrame(
        [
            {
                "location_id": "1",
                "target_hour_utc": issuance_dt + pd.Timedelta(hours=24),
                "pm25_ugm3": 120.0,
            },
            {
                "location_id": "centroid",
                "target_hour_utc": issuance_dt + pd.Timedelta(hours=24),
                "pm25_ugm3": 120.0,
            },
        ]
    )

    result = run_challenger_forecast(
        issuance_utc=issuance_dt,
        stations_df=stations_df,
        obs_df=empty_obs,
        cams_df=cams_df,
        cams_base=issuance_dt - pd.Timedelta(hours=12),
        mode="full",
    )

    assert result is not None
    assert result["adaptation_status"] == "skipped_n0"
    assert result["n_available_stations"] == 0
    assert result["method"] == "m7_hybrid_maml"
    assert len(result["city_forecasts"]) == 3


def test_challenger_adaptation_status_recording(clean_pipeline_state: None) -> None:
    """Test adaptation status is recorded in both forecast_log and manifest."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"
    canonical_date = issuance_dt.strftime("%Y-%m-%d")

    exit_code = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
    assert exit_code in (0, 10)

    # Check forecast log
    forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df = pd.read_parquet(forecast_path)
    challenger_rows = df[df["method"] == "m7_hybrid_maml"]
    assert len(challenger_rows) > 0
    status = challenger_rows["adaptation_status"].iloc[0]
    assert status in {
        "adapted",
        "skipped_n0",
        "skipped_insufficient_support",
        "rejected_no_gain",
        "rejected_residual_bound",
    }

    # Check manifests (both run-specific and canonical date)
    for m_path in [
        Path(f".state/manifests/{run_id}.json"),
        Path(f".state/manifests/{canonical_date}.json"),
    ]:
        assert m_path.exists(), f"Manifest missing at {m_path}"
        with m_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "challenger" in manifest
        c_meta = manifest["challenger"]
        assert c_meta["enabled"] is True
        assert c_meta["method"] == "m7_hybrid_maml"
        assert c_meta["adaptation_status"] == status


def test_forecast_log_schema_conformance(clean_pipeline_state: None) -> None:
    """Verify multi-method forecast log validates against Pandera forecast_log.schema.yaml."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"

    run_daily_pipeline(issuance_dt, mode="fixtures", force=True)

    forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df = pd.read_parquet(forecast_path)

    schema = load_pandera_schema("forecast_log.schema.yaml")
    validated = validate_frame(df, schema)
    assert len(validated) == len(df)
    assert set(df["method"].unique()) == {"m1_cams_raw", "m7_hybrid_maml"}


def test_promoted_model_bulletin_switch(clean_pipeline_state: None) -> None:
    """Verify that when promoted_bundle is configured, latest.json switches to candidate model."""
    from smogsense.config import Settings

    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

    real_settings = Settings.load("configs")
    fake_settings = real_settings.model_copy(deep=True)
    fake_settings.model["promoted_bundle"] = "bundle-v1.0.0"

    with patch("smogsense.config.Settings.load", return_value=fake_settings):
        exit_code = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
        assert exit_code in (0, 10)

        bulletin_path = Path("site/forecast/latest.json")
        with bulletin_path.open("r", encoding="utf-8") as f:
            bulletin = json.load(f)

        assert bulletin["model"]["id"] == "m7_hybrid_maml", (
            f"Expected promoted model, got {bulletin['model']}"
        )
        assert bulletin["model"]["version"] != "baseline"


def test_challenger_error_isolation(clean_pipeline_state: None) -> None:
    """Test Zero-Crash Guarantee: challenger exceptions do not crash pipeline or prevent baseline issuance."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"

    with patch("smogsense.pipeline.orchestrator.run_challenger_forecast") as mock_challenger:
        # Simulate fatal error or corruption in challenger
        mock_challenger.side_effect = RuntimeError(
            "Fatal weights corruption in challenger neural head"
        )

        exit_code = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
        # Pipeline must still complete successfully and issue baseline
        assert exit_code in (0, 10)

        # Baseline forecast log and bulletin must exist
        forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
        assert forecast_path.exists()
        df = pd.read_parquet(forecast_path)
        assert (df["method"] == "m1_cams_raw").all()

        bulletin_path = Path("site/forecast/latest.json")
        assert bulletin_path.exists()
        with bulletin_path.open("r", encoding="utf-8") as f:
            bulletin = json.load(f)
        assert bulletin["model"]["id"] == "m1_cams_raw"


def test_cli_challenger_options(clean_pipeline_state: None) -> None:
    """Test CLI commands accept challenger options and flags."""
    runner = CliRunner()
    issuance_str = "2026-10-15T00:00:00Z"
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"

    # 1. Run with --no-challenger
    result_disabled = runner.invoke(
        app,
        [
            "run",
            "daily",
            "--mode",
            "fixtures",
            "--issuance",
            issuance_str,
            "--no-challenger",
            "--force",
        ],
    )
    assert result_disabled.exit_code == 10
    forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df_no = pd.read_parquet(forecast_path)
    assert "m7_hybrid_maml" not in df_no["method"].values
    assert (df_no["method"] == "m1_cams_raw").all()

    # 2. Run with --enable-challenger and --challenger-method override
    result_enabled = runner.invoke(
        app,
        [
            "forecast",
            "daily",
            "--mode",
            "fixtures",
            "--issuance",
            issuance_str,
            "--enable-challenger",
            "--challenger-method",
            "m5_lgbm_source_zero_shot",
            "--force",
        ],
    )
    assert result_enabled.exit_code == 10
    forecast_files = sorted(Path(".state/forecasts").glob(f"forecast_{run_id}*.parquet"))
    assert len(forecast_files) >= 1
    df_custom = pd.read_parquet(forecast_files[-1])
    assert "m5_lgbm_source_zero_shot" in df_custom["method"].values
