"""smogsense.tests.unit.test_shadow_scoring — Unit tests for prospective shadow scoring pipeline.

Tests:
1. Pure metric functions: pinball loss, interval score, CRPS.
2. 18-hour completeness rule: >= 18 non-imputed hours required for validity.
3. Imputed hours rejection: imputed hours are not counted as valid.
4. First-issuance-only rule: is_rerun=True forecasts are never scored.
5. Pandera schema conformance with data/schemas/score_log.schema.yaml.
6. Persistence to date-partition and canonical append-only score_log ledger.
7. End-to-end run_shadow_scoring orchestration.
8. CLI command surface: smogsense score [provisional|settled|shadow].

Specification:
- docs/evaluation-strategy.md §2 ('Forecast targets and verification data')
- docs/system-architecture.md §4 ('Verification and scoring loop')
- data/schemas/score_log.schema.yaml
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.scoring import (
    Q_NAMES,
    compute_crps,
    interval_score,
    pinball_loss,
    score_forecast_block,
)
from smogsense.pipeline.shadow import (
    run_shadow_scoring,
    save_score_log,
    score_forecast_targets,
    score_provisional,
    score_settled,
)

runner = CliRunner()


# --- 1. Pure Metric Calculations ---


def test_pinball_loss_exact() -> None:
    # y = 100, q = 90, diff = 10 (underprediction)
    # tau = 0.90 -> 0.90 * 10 = 9.0
    # tau = 0.10 -> 0.10 * 10 = 1.0
    assert np.isclose(pinball_loss(100.0, 90.0, 0.90), 9.0)
    assert np.isclose(pinball_loss(100.0, 90.0, 0.10), 1.0)

    # y = 80, q = 90, diff = -10 (overprediction)
    # tau = 0.90 -> (0.90 - 1) * (-10) = 1.0
    # tau = 0.10 -> (0.10 - 1) * (-10) = 9.0
    assert np.isclose(pinball_loss(80.0, 90.0, 0.90), 1.0)
    assert np.isclose(pinball_loss(80.0, 90.0, 0.10), 9.0)


def test_interval_score_exact() -> None:
    # Coverage interval [80, 120], width = 40, alpha = 0.20
    # Inside interval: y = 100 -> score = 40
    assert np.isclose(interval_score(100.0, 80.0, 120.0, alpha=0.20), 40.0)

    # Below interval: y = 70 -> penalty = (2 / 0.20) * (80 - 70) = 10 * 10 = 100 -> total 140
    assert np.isclose(interval_score(70.0, 80.0, 120.0, alpha=0.20), 140.0)

    # Above interval: y = 130 -> penalty = (2 / 0.20) * (130 - 120) = 100 -> total 140
    assert np.isclose(interval_score(130.0, 80.0, 120.0, alpha=0.20), 140.0)


def test_compute_crps_monotonic_quantiles() -> None:
    # 19 quantiles evenly spaced 50..140
    q_vals = np.linspace(50.0, 140.0, 19)
    # Observation near median
    score = compute_crps(q_vals, 95.0)
    assert isinstance(score, float)
    assert score > 0.0


# --- 2. Block Scoring & Completeness Rule ---


def _dummy_forecast_row(
    issuance_utc: datetime, horizon_h: int = 24, point_id: str = "station:1001"
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "test_run_123",
        "issuance_utc": issuance_utc,
        "domain": "lahore",
        "point_id": point_id,
        "horizon_h": np.int32(horizon_h),
        "method": "m1_cams_raw",
    }
    for i, q in enumerate(Q_NAMES):
        row[q] = 70.0 + i * 3.0
    row["q10"] = 80.0
    row["q50"] = 100.0
    row["q90"] = 120.0
    return row


def test_score_forecast_block_valid_18_hours() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_row = _dummy_forecast_row(t0)

    # 18 valid hours
    obs_vals = pd.Series([100.0] * 18)
    res = score_forecast_block(f_row, obs_vals, stage="provisional")

    assert bool(res["valid"]) is True
    assert res["n_valid_hours"] == 18
    assert res["y_obs"] == 100.0
    assert res["crps"] is not None and res["crps"] >= 0.0
    assert res["pinball_q50"] == 0.0  # y = q50 = 100
    assert res["interval_score_80"] == 40.0  # 120 - 80
    assert bool(res["hit_below_q50"]) is True
    assert res["obs_category"] == "unhealthy"  # 100 ug/m3 is US EPA Unhealthy (55.5-125.4)


def test_score_forecast_block_invalid_17_hours() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_row = _dummy_forecast_row(t0)

    # 17 valid hours -> fails completeness rule (< 18)
    obs_vals = pd.Series([100.0] * 17)
    res = score_forecast_block(f_row, obs_vals, stage="provisional")

    assert res["valid"] is False
    assert res["n_valid_hours"] == 17
    assert res["y_obs"] is None
    assert res["crps"] is None
    assert res["pinball_q10"] is None
    assert res["pinball_q50"] is None
    assert res["pinball_q90"] is None
    assert res["interval_score_80"] is None
    assert res["hit_below_q10"] is None
    assert res["obs_category"] is None


# --- 3. Target Matching, Imputation Exclusion & First-Issuance Rule ---


def _make_forecast_df(t0: datetime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in [24, 48, 72]:
        for pt in ["station:1001", "city:lahore"]:
            r = _dummy_forecast_row(t0, horizon_h=h, point_id=pt)
            r["window_start_utc"] = t0 + timedelta(hours=h - 24)
            r["window_end_utc"] = t0 + timedelta(hours=h)
            r["is_rerun"] = False
            rows.append(r)
    return pd.DataFrame(rows)


def _make_obs_df(t0: datetime) -> pd.DataFrame:
    # 72 hours of observations for station 1001
    times = [t0 + timedelta(hours=i) for i in range(72)]
    df = pd.DataFrame(
        {
            "location_id": 1001,
            "ts_utc": times,
            "pm25_ugm3": [105.0] * 72,
            "imputed": [False] * 72,
            "domain": "lahore",
        }
    )
    return df


def test_score_provisional_matches_and_validates_schema() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    scored_df = score_provisional(
        issuance_date="2026-10-15",
        forecast_log_df=f_df,
        api_obs_df=obs_df,
    )

    assert not scored_df.empty
    assert len(scored_df) == len(f_df)
    assert set(scored_df["stage"]) == {"provisional"}
    assert set(scored_df["horizon_h"]) == {24, 48, 72}

    # Station 1001 rows should be valid
    st_rows = scored_df[scored_df["point_id"] == "station:1001"]
    assert all(st_rows["valid"])
    assert all(st_rows["n_valid_hours"] == 24)
    assert all(st_rows["y_obs"] == 105.0)


def test_imputed_hours_excluded_from_completeness() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    # For the first 24h block, mark 10 hours as imputed
    # 24 - 10 = 14 non-imputed hours -> < 18 -> invalid!
    obs_df.loc[0:9, "imputed"] = True

    scored_df = score_provisional(
        issuance_date="2026-10-15",
        forecast_log_df=f_df,
        api_obs_df=obs_df,
    )

    # 24h block for station 1001 must be invalid
    h24_row = scored_df[
        (scored_df["point_id"] == "station:1001") & (scored_df["horizon_h"] == 24)
    ].iloc[0]
    assert bool(h24_row["valid"]) is False
    assert h24_row["n_valid_hours"] == 14
    assert h24_row["y_obs"] is None or pd.isna(h24_row["y_obs"])

    # 48h block was not imputed -> valid
    h48_row = scored_df[
        (scored_df["point_id"] == "station:1001") & (scored_df["horizon_h"] == 48)
    ].iloc[0]
    assert bool(h48_row["valid"]) is True
    assert h48_row["n_valid_hours"] == 24


def test_first_issuance_only_rule_rejects_reruns() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    # Mark some rows as reruns
    f_df["is_rerun"] = [True, False, True, False, True, False]

    scored_df = score_provisional(
        issuance_date="2026-10-15",
        forecast_log_df=f_df,
        api_obs_df=obs_df,
    )

    # Only the 3 non-rerun rows should be scored
    assert len(scored_df) == 3


def test_score_settled_records_truth_stage() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    scored_df = score_settled(
        issuance_date="2026-10-15",
        forecast_log_df=f_df,
        archive_obs_df=obs_df,
    )

    assert set(scored_df["stage"]) == {"settled"}
    assert set(scored_df["truth_basis"]) == {"openaq_archive"}


def test_score_forecast_targets_invalid_stage() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    with pytest.raises(ValueError, match="stage must be 'provisional' or 'settled'"):
        score_forecast_targets("2026-10-15", f_df, obs_df, stage="invalid_stage")


# --- 4. Persistence & Deduplication ---


def test_save_score_log_partitions_and_canonical(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    scores_df = score_provisional("2026-10-15", f_df, obs_df)

    saved_path = save_score_log(scores_df, "2026-10-15", state_dir=tmp_path)

    # Verify partition file
    expected_partition = tmp_path / "scores" / "dt=2026-10-15" / "score_log.parquet"
    assert saved_path == expected_partition
    assert expected_partition.exists()
    part_df = pd.read_parquet(expected_partition)
    assert len(part_df) == len(scores_df)

    # Verify canonical file
    canonical_file = tmp_path / "score_log.parquet"
    assert canonical_file.exists()
    canon_df = pd.read_parquet(canonical_file)
    assert len(canon_df) == len(scores_df)

    # Re-saving (idempotent / append)
    save_score_log(scores_df, "2026-10-15", state_dir=tmp_path)
    canon_df_after = pd.read_parquet(canonical_file)
    # Deduplication prevents row doubling
    assert len(canon_df_after) == len(scores_df)


# --- 5. Full Orchestration & CLI ---


def test_run_shadow_scoring_orchestration(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    # Setup mock .state directory
    forecasts_dir = tmp_path / "forecasts"
    forecasts_dir.mkdir(parents=True)
    f_df.to_parquet(forecasts_dir / "forecast_run_20261015_0000.parquet", index=False)

    obs_file = tmp_path / "observations.parquet"
    obs_df.to_parquet(obs_file, index=False)

    res = run_shadow_scoring(
        stage="provisional",
        issuance_date="2026-10-15",
        state_dir=tmp_path,
        obs_file=obs_file,
    )

    assert res["status"] == "ok"
    assert res["stage"] == "provisional"
    assert res["new_scores"] == len(f_df)
    assert Path(res["saved_path"]).exists()


def test_cli_score_subcommands(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    f_df = _make_forecast_df(t0)
    obs_df = _make_obs_df(t0)

    forecasts_dir = tmp_path / "forecasts"
    forecasts_dir.mkdir(parents=True)
    f_df.to_parquet(forecasts_dir / "forecast_run_20261015_0000.parquet", index=False)

    obs_file = tmp_path / "observations.parquet"
    obs_df.to_parquet(obs_file, index=False)

    # Test smogsense score provisional
    res_prov = runner.invoke(
        app,
        [
            "score",
            "provisional",
            "--date",
            "2026-10-15",
            "--state-dir",
            str(tmp_path),
            "--obs-file",
            str(obs_file),
        ],
    )
    assert res_prov.exit_code == 0
    assert "Provisional scoring: ok" in res_prov.stdout

    # Test smogsense score settled
    res_settled = runner.invoke(
        app,
        [
            "score",
            "settled",
            "--date",
            "2026-10-15",
            "--state-dir",
            str(tmp_path),
            "--obs-file",
            str(obs_file),
        ],
    )
    assert res_settled.exit_code == 0
    assert "Settled scoring: ok" in res_settled.stdout

    # Test smogsense score shadow --stage provisional
    res_shadow = runner.invoke(
        app,
        [
            "score",
            "shadow",
            "--stage",
            "provisional",
            "--date",
            "2026-10-15",
            "--state-dir",
            str(tmp_path),
            "--obs-file",
            str(obs_file),
        ],
    )
    assert res_shadow.exit_code == 0
    assert "Shadow scoring (provisional): ok" in res_shadow.stdout
