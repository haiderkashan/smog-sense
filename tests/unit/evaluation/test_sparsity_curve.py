"""Unit tests for sensor-sparsity curve evaluation engine (Task P3-06).

Tests:
1. Twin-aware station cluster sampling and preservation across draws.
2. N=0 sensor-free mode exact fallback and status logging (Condition C7).
3. Power-law non-linear saturation curve fitting and parameter recovery.
4. N* determination (smallest N achieving CRPSS vs M2 >= 0.10).
5. Canonical adaptation status recording and distribution aggregation.
6. Report serialization to JSON and Markdown format.
7. Distributed matrix shard merging across separate N runs.
8. Typer CLI evaluation commands (`evaluate sparsity` and `--merge-shards`).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.sparsity_curve import (
    SparsityDrawResult,
    SparsityExperimentResult,
    SparsityPointResult,
    aggregate_draw_results,
    compute_n_star,
    fit_power_law,
    get_canonical_lahore_stations,
    merge_sparsity_shards,
    run_sparsity_experiment,
    sample_twin_aware_draws,
    save_sparsity_report,
)


@pytest.fixture
def mock_stations() -> pd.DataFrame:
    """Station registry with twin pairs (co-located <= 50m) and singletons."""
    return pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "station_name": ["S1_ref", "S1_low", "S2", "S3", "S4_ref", "S4_low", "S5", "S6"],
            # 1 and 2 co-located: exact same coordinates
            # 5 and 6 co-located: distance < 50m
            "latitude": [
                31.55000,
                31.55000,
                31.56000,
                31.57000,
                31.58000,
                31.58020,
                31.59000,
                31.60000,
            ],
            "longitude": [
                74.33000,
                74.33000,
                74.34000,
                74.35000,
                74.36000,
                74.36010,
                74.37000,
                74.38000,
            ],
            "colocated_group_id": [1, 1, None, None, 2, 2, None, None],
            "is_reference": [True, False, False, False, True, False, False, False],
        }
    )


def test_twin_aware_sampling_atomicity(mock_stations: pd.DataFrame) -> None:
    """Co-located station twins (<= 50m) must never be split across available and held-out sets."""
    rng = np.random.default_rng(42)

    # Sample for N=2
    draws = sample_twin_aware_draws(
        mock_stations,
        n=2,
        r=15,
        min_heldout=2,
        rng=rng,
    )

    assert len(draws) == 15
    for a_set, b_set in draws:
        set_a = set(a_set)
        set_b = set(b_set)

        # Disjoint invariant
        assert set_a.isdisjoint(set_b)

        # Twin pair 1 and 2: must both be in A, or both in B, or omitted from evaluation
        pair_1 = {1, 2}
        assert not (pair_1.intersection(set_a) and pair_1.intersection(set_b))

        # Twin pair 5 and 6: must never be split across A and B
        pair_2 = {5, 6}
        assert not (pair_2.intersection(set_a) and pair_2.intersection(set_b))


def test_n0_sensor_free_fallback() -> None:
    """At N=0, sensor-free mode must fallback exactly to M5 and assign status 'skipped_n0'."""
    stations = get_canonical_lahore_stations()
    result = run_sparsity_experiment(
        stations_df=stations,
        n_grid=(0,),
        draws=1,
        mode="neighbour_history",
        seed=123,
    )

    assert result.n_grid == [0]
    draws = result.raw_draws
    assert len(draws) == 7  # 7 methods evaluated for N=0

    # Index draws by method
    draw_by_method = {d.method: d for d in draws}

    # Invariant: M1 and M2 identical at N=0 (CAMS-BC reduces to raw CAMS)
    assert np.isclose(
        draw_by_method["m1_cams_raw"].mean_crps, draw_by_method["m2_cams_bc"].mean_crps
    )
    assert draw_by_method["m2_cams_bc"].adaptation_status == "skipped_n0"

    # Invariant: M5 physics base is evaluated
    m5_crps = draw_by_method["m5_lgbm_source_zero_shot"].mean_crps
    assert m5_crps > 0.0

    # Invariant: M6 and M7 fallback to M5 bit-for-bit
    assert np.isclose(draw_by_method["m6_finetune_transfer"].mean_crps, m5_crps)
    assert draw_by_method["m6_finetune_transfer"].adaptation_status == "skipped_n0"

    assert np.isclose(draw_by_method["m7_hybrid_maml"].mean_crps, m5_crps)
    assert draw_by_method["m7_hybrid_maml"].adaptation_status == "skipped_n0"

    # Invariant: M4 and M3b are marked untrainable_n0
    assert draw_by_method["m4_lgbm_local"].adaptation_status == "untrainable_n0"
    assert draw_by_method["m3b_linear_qr"].adaptation_status == "untrainable_n0"
    assert draw_by_method["m4_lgbm_local"].mean_crps > m5_crps


def test_power_law_fit_convergence() -> None:
    """Non-linear power law saturation model recovers parameters on exponential data."""
    # True underlying parameters
    c_inf_true = 16.5
    c_0_true = 38.0
    nu_true = 2.4

    n_vals = [0, 1, 2, 3, 5, 8]
    crps_vals = [c_inf_true + (c_0_true - c_inf_true) * float(np.exp(-n / nu_true)) for n in n_vals]

    fit = fit_power_law(n_vals, crps_vals)

    assert np.isclose(fit.c_infinity, c_inf_true, atol=0.2)
    assert np.isclose(fit.c_0, c_0_true, atol=0.2)
    assert np.isclose(fit.nu, nu_true, atol=0.2)
    assert fit.r_squared > 0.999


def test_n_star_determination() -> None:
    """N* is the smallest N where mean CRPSS vs M2 >= 0.10."""
    points = [
        SparsityPointResult(
            n_available=0,
            method="m7_hybrid_maml",
            mean_crps=22.0,
            std_crps=0.5,
            ci_95_crps=(21.5, 22.5),
            mean_crpss_vs_m2=0.01,
            ci_95_crpss_vs_m2=(0.0, 0.02),
            adaptation_status_counts={"skipped_n0": 5},
            adaptation_status_pct={"skipped_n0": 1.0},
            n_draws=5,
        ),
        SparsityPointResult(
            n_available=1,
            method="m7_hybrid_maml",
            mean_crps=19.5,
            std_crps=0.4,
            ci_95_crps=(19.1, 19.9),
            mean_crpss_vs_m2=0.08,
            ci_95_crpss_vs_m2=(0.06, 0.10),
            adaptation_status_counts={"adapted": 5},
            adaptation_status_pct={"adapted": 1.0},
            n_draws=5,
        ),
        SparsityPointResult(
            n_available=2,
            method="m7_hybrid_maml",
            mean_crps=17.8,
            std_crps=0.3,
            ci_95_crps=(17.5, 18.1),
            mean_crpss_vs_m2=0.14,  # Reaches threshold >= 0.10
            ci_95_crpss_vs_m2=(0.12, 0.16),
            adaptation_status_counts={"adapted": 5},
            adaptation_status_pct={"adapted": 1.0},
            n_draws=5,
        ),
        SparsityPointResult(
            n_available=3,
            method="m7_hybrid_maml",
            mean_crps=16.9,
            std_crps=0.2,
            ci_95_crps=(16.7, 17.1),
            mean_crpss_vs_m2=0.18,
            ci_95_crpss_vs_m2=(0.16, 0.20),
            adaptation_status_counts={"adapted": 5},
            adaptation_status_pct={"adapted": 1.0},
            n_draws=5,
        ),
    ]

    n_star = compute_n_star(points, method="m7_hybrid_maml", threshold_crpss=0.10)
    assert n_star == 2

    # Never achieved case
    n_star_high = compute_n_star(points, method="m7_hybrid_maml", threshold_crpss=0.50)
    assert n_star_high is None


def test_adaptation_status_logging() -> None:
    """Status counts and percentages are correctly recorded across draws."""
    raw_draws = [
        SparsityDrawResult(
            n_available=2,
            draw_idx=0,
            available_stations=[1, 2],
            held_out_stations=[3, 4],
            method="m7_hybrid_maml",
            adaptation_status="adapted",
            metrics_by_horizon={24: {"crps": 18.0}},
            mean_crps=18.0,
            mean_crpss_vs_m2=0.10,
        ),
        SparsityDrawResult(
            n_available=2,
            draw_idx=1,
            available_stations=[2, 3],
            held_out_stations=[1, 4],
            method="m7_hybrid_maml",
            adaptation_status="rejected_residual_bound",
            metrics_by_horizon={24: {"crps": 20.0}},
            mean_crps=20.0,
            mean_crpss_vs_m2=0.00,
        ),
        SparsityDrawResult(
            n_available=2,
            draw_idx=2,
            available_stations=[1, 4],
            held_out_stations=[2, 3],
            method="m7_hybrid_maml",
            adaptation_status="adapted",
            metrics_by_horizon={24: {"crps": 17.5}},
            mean_crps=17.5,
            mean_crpss_vs_m2=0.12,
        ),
    ]

    points, _, _ = aggregate_draw_results(raw_draws, methods=["m7_hybrid_maml"])
    assert len(points) == 1
    p = points[0]
    assert p.adaptation_status_counts["adapted"] == 2
    assert p.adaptation_status_counts["rejected_residual_bound"] == 1
    assert np.isclose(p.adaptation_status_pct["adapted"], 2.0 / 3.0)
    assert np.isclose(p.adaptation_status_pct["rejected_residual_bound"], 1.0 / 3.0)


def test_report_serialization_json_and_md(tmp_path: Path) -> None:
    """Report serialization writes valid JSON with schema roundtrip and rich Markdown table."""
    result = run_sparsity_experiment(
        n_grid=(0, 1),
        draws=2,
        mode="neighbour_history",
        seed=42,
    )

    json_path, md_path = save_sparsity_report(result, tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    # JSON roundtrip
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    reloaded = SparsityExperimentResult.from_dict(payload)
    assert reloaded.domain == result.domain
    assert reloaded.mode == result.mode
    assert reloaded.n_grid == result.n_grid
    assert len(reloaded.points) == len(result.points)

    # Markdown format inspection
    md_content = md_path.read_text(encoding="utf-8")
    assert "# SmogSense Sensor-Sparsity Curve Report" in md_content
    assert "Probabilistic Skill (CRPS)" in md_content
    assert "M7 Meta-SGD Residual" in md_content
    assert "Deployment Adaptation Status Distribution" in md_content


def test_matrix_shard_merging(tmp_path: Path) -> None:
    """Multiple partial matrix shards are merged into a unified report."""
    shard1 = run_sparsity_experiment(
        n_grid=(0,),
        draws=2,
        mode="neighbour_history",
        seed=10,
    )
    shard2 = run_sparsity_experiment(
        n_grid=(1,),
        draws=2,
        mode="neighbour_history",
        seed=20,
    )

    p1 = tmp_path / "shard_0.json"
    p2 = tmp_path / "shard_1.json"
    p1.write_text(json.dumps(shard1.to_dict(), indent=2), encoding="utf-8")
    p2.write_text(json.dumps(shard2.to_dict(), indent=2), encoding="utf-8")

    merged = merge_sparsity_shards([p1, p2])
    assert sorted(merged.n_grid) == [0, 1]
    assert len(merged.raw_draws) == len(shard1.raw_draws) + len(shard2.raw_draws)

    # Points aggregated for both N=0 and N=1
    point_n_vals = {p.n_available for p in merged.points}
    assert point_n_vals == {0, 1}


def test_cli_sparsity_command(tmp_path: Path) -> None:
    """CLI evaluates sensor-sparsity curve cleanly with exit code 0."""
    runner = CliRunner()

    out_dir = tmp_path / "cli_reports"
    res = runner.invoke(
        app,
        [
            "evaluate",
            "sparsity",
            "--n-val",
            "0",
            "--draws",
            "1",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert "SENSOR-SPARSITY CURVE EVALUATION" in res.stdout
    assert "SENSOR-SPARSITY EXPERIMENT COMPLETED" in res.stdout
    assert (out_dir / "sparsity_curve.json").exists()
    assert (out_dir / "sparsity_curve.md").exists()

    # Test merge-shards branch
    shard_file = out_dir / "sparsity_curve.json"
    merged_dir = tmp_path / "cli_merged"
    res_merge = runner.invoke(
        app,
        [
            "eval",
            "sparsity",
            "--merge-shards",
            str(shard_file),
            "--output-dir",
            str(merged_dir),
        ],
    )
    assert res_merge.exit_code == 0
    assert "Successfully merged" in res_merge.stdout
    assert (merged_dir / "sparsity_curve.json").exists()
