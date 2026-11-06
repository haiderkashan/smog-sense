"""Unit tests for calibration diagnostics, PIT, and reliability diagrams."""

import numpy as np
from scipy.stats import norm

from smogsense.evaluation.calibration_diagnostics import (
    compute_calibration_report,
    compute_pit_histogram,
    compute_pit_values,
    compute_quantile_reliability,
    coverage_50,
    coverage_80,
    coverage_below_median,
    interval_coverage,
    interval_sharpness,
    pit_uniformity_test,
)
from smogsense.evaluation.scoring import CANONICAL_TAUS


def test_coverage_and_sharpness_metrics() -> None:
    y_true = np.array([50.0, 75.0, 100.0, 125.0])
    lower = np.array([40.0, 70.0, 90.0, 130.0])  # sample 4 (125) is below lower (130)
    upper = np.array([60.0, 80.0, 110.0, 150.0])

    cov = interval_coverage(y_true, lower, upper)
    # Samples 1, 2, 3 inside; sample 4 is 125 < 130 -> 3 / 4 = 0.75
    assert np.isclose(cov, 0.75)

    shp = interval_sharpness(lower, upper)
    # Widths: 20, 10, 20, 20 -> mean = 70 / 4 = 17.5
    assert np.isclose(shp, 17.5)

    # Empty inputs
    assert interval_coverage([], [], []) == 0.0
    assert interval_sharpness([], []) == 0.0


def test_coverage_80_and_50() -> None:
    y = np.linspace(10, 100, 10)
    q10 = np.full(10, 5.0)
    q90 = np.full(10, 105.0)
    # All inside
    assert coverage_80(y, q10, q90) == 1.0

    # 50% interval [q25, q75]
    q25 = np.full(10, 25.0)
    q75 = np.full(10, 75.0)
    # y: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 -> [30, 40, 50, 60, 70] are 5 samples -> 0.50
    assert np.isclose(coverage_50(y, q25, q75), 0.50)

    # Coverage below median
    q50 = np.full(10, 50.0)
    # y: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 -> exactly 5 <= 50 -> 0.50
    assert np.isclose(coverage_below_median(y, q50), 0.50)


def test_pit_values_uniformity_for_calibrated_model() -> None:
    rng = np.random.default_rng(12345)
    n = 2000
    mu, sigma = 100.0, 20.0

    # True observations from N(mu, sigma^2)
    y_true = rng.normal(loc=mu, scale=sigma, size=n)

    # Forecast quantiles exactly matching N(mu, sigma^2)
    taus = CANONICAL_TAUS
    q_single = norm.ppf(taus, loc=mu, scale=sigma)
    q_mat = np.tile(q_single, (n, 1))

    pit_vals = compute_pit_values(y_true, q_mat)
    assert len(pit_vals) == n
    assert (pit_vals >= 0.0).all() and (pit_vals <= 1.0).all()

    # Uniformity chi2 test
    res = pit_uniformity_test(pit_vals, n_bins=20)
    assert res["df"] == 19
    # Calibrated predictions should not reject uniformity at alpha = 0.01
    assert res["p_value"] > 0.01
    assert res["is_pit_uniform"] is True


def test_pit_histogram_and_uniformity_rejection() -> None:
    # Intentionally biased PIT values (all concentrated in lower tail [0, 0.2])
    rng = np.random.default_rng(999)
    biased_pit = rng.uniform(0.0, 0.15, size=500)
    counts, edges = compute_pit_histogram(biased_pit, n_bins=20)
    assert len(counts) == 20
    assert len(edges) == 21
    assert int(np.sum(counts)) == 500

    test_res = pit_uniformity_test(biased_pit, n_bins=20)
    # Biased forecast must reject uniformity (p-value near 0)
    assert test_res["p_value"] < 1e-10
    assert test_res["is_pit_uniform"] is False


def test_compute_quantile_reliability() -> None:
    y_true = np.array([30.0, 50.0, 70.0, 90.0, 110.0])
    # Quantiles where everyone is below q (q very high)
    q_high = np.ones((5, 19)) * 200.0
    rel_high = compute_quantile_reliability(y_true, q_high)
    assert np.allclose(rel_high["empirical_probs"], 1.0)
    assert np.allclose(rel_high["coverage_errors"], 1.0 - CANONICAL_TAUS)


def test_calibration_report_end_to_end() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.normal(80.0, 15.0, size=n)
    q_single = norm.ppf(CANONICAL_TAUS, loc=80.0, scale=15.0)
    q_mat = np.tile(q_single, (n, 1))

    report = compute_calibration_report(y_true, q_mat)
    assert report.n_samples == n
    assert 0.70 <= report.coverage_80 <= 0.90
    assert 0.40 <= report.coverage_50 <= 0.60
    assert 0.40 <= report.median_coverage <= 0.60
    assert report.sharpness_80 > 0.0

    d = report.to_dict()
    assert isinstance(d, dict)
    assert "coverage_80" in d
    assert len(d["pit_counts"]) == 20
