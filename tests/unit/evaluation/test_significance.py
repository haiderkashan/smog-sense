"""Unit tests for statistical hypothesis testing and block bootstrap (significance.py)."""

import numpy as np
import pandas as pd
import pytest

from smogsense.evaluation.significance import (
    BootstrapResult,
    DieboldMarianoResult,
    diebold_mariano_test,
    holm_bonferroni_correction,
    stationary_block_bootstrap,
)


def test_diebold_mariano_identical_series() -> None:
    # If loss_a == loss_b, difference is identically 0
    loss = np.array([10.0, 15.0, 12.0, 18.0, 14.0, 16.0])
    res = diebold_mariano_test(loss, loss, horizon_h=24)
    assert res.mean_loss_diff == 0.0
    assert res.dm_stat == 0.0
    assert res.dm_star == 0.0
    assert res.p_value == 1.0


def test_diebold_mariano_synthetic_superior_model() -> None:
    rng = np.random.default_rng(42)
    n = 60
    # Model A is significantly better than Model B (mean difference -5.0)
    loss_b = rng.normal(loc=25.0, scale=3.0, size=n)
    loss_a = loss_b - 5.0 + rng.normal(loc=0.0, scale=1.0, size=n)

    res = diebold_mariano_test(loss_a, loss_b, horizon_h=24, alternative="two-sided")
    assert isinstance(res, DieboldMarianoResult)
    assert res.n_samples == n
    assert res.mean_loss_diff < -4.0
    assert res.dm_stat < -10.0
    assert res.dm_star < -10.0
    assert res.p_value < 1e-10
    assert res.hln_factor > 0.0
    assert res.n_eff > 10.0

    # Test alternative="less" (Model A loss < Model B loss)
    res_less = diebold_mariano_test(loss_a, loss_b, horizon_h=24, alternative="less")
    assert res_less.p_value < 1e-10

    # Test alternative="greater" should yield p ~ 1.0
    res_greater = diebold_mariano_test(loss_a, loss_b, horizon_h=24, alternative="greater")
    assert res_greater.p_value > 0.999


def test_diebold_mariano_sample_size_error() -> None:
    with pytest.raises(ValueError, match="at least 3 valid samples"):
        diebold_mariano_test([1.0, 2.0], [2.0, 3.0])


def test_stationary_block_bootstrap_1d_array() -> None:
    rng = np.random.default_rng(100)
    data = rng.normal(loc=10.0, scale=2.0, size=50)

    res = stationary_block_bootstrap(
        data,
        statistic_fn=lambda arr: float(np.mean(arr)),
        mean_block_days=7.0,
        replicates=500,
        ci=0.95,
        seed=42,
    )
    assert isinstance(res, BootstrapResult)
    assert res.replicates_count == 500
    assert np.isclose(res.point_estimate, float(np.mean(data)))
    assert np.isclose(res.bootstrap_mean, res.point_estimate, atol=0.5)
    # 95% CI should contain true mean (10.0)
    assert res.ci_lower < 10.0 < res.ci_upper
    assert res.ci_lower < res.ci_upper


def test_stationary_block_bootstrap_dataframe_with_dates() -> None:
    # 20 days with 3 stations per day = 60 rows
    rng = np.random.default_rng(777)
    dates = pd.date_range("2026-11-01", periods=20, freq="1D")
    records = [
        {"date": d, "station": s, "val": 50.0 + rng.normal()}
        for d in dates
        for s in ["8664", "8665", "8666"]
    ]
    df = pd.DataFrame(records)

    res = stationary_block_bootstrap(
        df,
        statistic_fn=lambda sub_df: float(sub_df["val"].mean()),
        date_col="date",
        mean_block_days=5.0,
        replicates=200,
        seed=123,
    )
    assert res.replicates_count == 200
    assert 48.0 < res.point_estimate < 52.0
    assert res.ci_lower < res.ci_upper


def test_holm_bonferroni_correction() -> None:
    # Raw p-values
    p_vals = [0.005, 0.04, 0.02, 0.15]
    res = holm_bonferroni_correction(p_vals, alpha=0.05)

    assert len(res) == 4
    # Check monotonicity in sorted order:
    # Sorted order of p: 0.005 (idx 0), 0.02 (idx 2), 0.04 (idx 1), 0.15 (idx 3)
    # rank 0: 0.005 * 4 = 0.02 -> sig (<= 0.05)
    # rank 1: 0.02 * 3 = 0.06 -> not sig
    # rank 2: max(0.06, 0.04 * 2) = 0.08 -> not sig
    # rank 3: max(0.08, 0.15 * 1) = 0.15 -> not sig
    raw0, adj0, sig0 = res[0]
    assert np.isclose(raw0, 0.005)
    assert np.isclose(adj0, 0.02)
    assert sig0 is True

    raw2, adj2, sig2 = res[2]
    assert np.isclose(raw2, 0.02)
    assert np.isclose(adj2, 0.06)
    assert sig2 is False
