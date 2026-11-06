"""Unit tests for evaluation CLI commands (smogsense evaluate metrics / significance)."""

from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.scoring import CANONICAL_TAUS

runner = CliRunner()


def test_cli_evaluate_metrics_command(tmp_path: Path) -> None:
    # Create synthetic predictions CSV with 19 quantiles and true pm25
    n = 20
    taus = CANONICAL_TAUS
    records = []
    for i in range(n):
        row = {"issuance_utc": f"2026-11-{i + 1:02d}T00:00:00Z", "pm25": 80.0 + i * 2.0}
        for tau in taus:
            col = f"q{round(tau * 100):02d}"
            row[col] = 60.0 + tau * 60.0 + i * 2.0
        records.append(row)
    df = pd.DataFrame(records)

    pred_csv = tmp_path / "preds.csv"
    df.to_csv(pred_csv, index=False)
    out_json = tmp_path / "metrics_out.json"

    res = runner.invoke(
        app,
        [
            "evaluate",
            "metrics",
            "--predictions",
            str(pred_csv),
            "--truth-col",
            "pm25",
            "--out-json",
            str(out_json),
        ],
    )
    assert res.exit_code == 0
    assert "PROBABILISTIC EVALUATION METRICS REPORT" in res.output
    assert "mean_crps" in res.output
    assert out_json.exists()


def test_cli_evaluate_significance_command(tmp_path: Path) -> None:
    # Create two loss series CSVs
    n = 30
    rng = np.random.default_rng(42)
    dates = [f"2026-11-{i + 1:02d}T00:00:00Z" for i in range(n)]

    df_a = pd.DataFrame({"issuance_utc": dates, "crps": 12.0 + rng.normal(0, 1, size=n)})
    df_b = pd.DataFrame({"issuance_utc": dates, "crps": 18.0 + rng.normal(0, 1, size=n)})

    path_a = tmp_path / "loss_a.csv"
    path_b = tmp_path / "loss_b.csv"
    out_json = tmp_path / "significance_out.json"

    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)

    res = runner.invoke(
        app,
        [
            "evaluate",
            "significance",
            "--loss-a",
            str(path_a),
            "--loss-b",
            str(path_b),
            "--loss-col",
            "crps",
            "--bootstrap-reps",
            "100",
            "--out-json",
            str(out_json),
        ],
    )
    assert res.exit_code == 0
    assert "STATISTICAL SIGNIFICANCE EVALUATION" in res.output
    assert "DM* Statistic (HLN)" in res.output
    assert "STATIONARY BLOCK BOOTSTRAP" in res.output
    assert out_json.exists()
