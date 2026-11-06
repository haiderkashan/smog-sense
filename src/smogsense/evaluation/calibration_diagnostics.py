"""smogsense.evaluation.calibration_diagnostics — PIT/reliability and coverage diagnostics.

Probability Integral Transform (PIT) histogram, uniformity tests, empirical coverage, sharpness,
and reliability diagrams across 19 canonical quantile levels.

Specification: docs/evaluation-strategy.md → 'Complementary metrics (calibration, events)'
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import stats

from smogsense.evaluation.scoring import CANONICAL_TAUS
from smogsense.models.distribution import QuantileFunction


@dataclass(frozen=True)
class CalibrationReport:
    """Comprehensive summary of probabilistic calibration diagnostics."""

    n_samples: int
    coverage_80: float
    sharpness_80: float
    coverage_50: float
    sharpness_50: float
    median_coverage: float
    pit_chi2: float
    pit_p_value: float
    is_pit_uniform: bool
    pit_counts: list[int]
    pit_bin_edges: list[float]
    nominal_taus: list[float]
    empirical_probs: list[float]
    coverage_errors: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "n_samples": self.n_samples,
            "coverage_80": self.coverage_80,
            "sharpness_80": self.sharpness_80,
            "coverage_50": self.coverage_50,
            "sharpness_50": self.sharpness_50,
            "median_coverage": self.median_coverage,
            "pit_chi2": self.pit_chi2,
            "pit_p_value": self.pit_p_value,
            "is_pit_uniform": self.is_pit_uniform,
            "pit_counts": self.pit_counts,
            "pit_bin_edges": self.pit_bin_edges,
            "nominal_taus": self.nominal_taus,
            "empirical_probs": self.empirical_probs,
            "coverage_errors": self.coverage_errors,
        }


def interval_coverage(
    y_true: float | Sequence[float] | np.ndarray,
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float:
    """Compute empirical coverage fraction of a prediction interval: fraction where lower <= y <= upper.

    Args:
        y_true: True observation(s).
        lower: Lower interval bound(s).
        upper: Upper interval bound(s).

    Returns:
        Fraction of observations falling inside [lower, upper].
    """
    y_arr = np.asarray(y_true, dtype=float)
    l_arr = np.asarray(lower, dtype=float)
    u_arr = np.asarray(upper, dtype=float)

    if len(y_arr) == 0:
        return 0.0

    in_interval = (y_arr >= l_arr) & (y_arr <= u_arr)
    return float(np.mean(in_interval))


def interval_sharpness(
    lower: float | Sequence[float] | np.ndarray,
    upper: float | Sequence[float] | np.ndarray,
) -> float:
    """Compute sharpness (mean interval width): mean(upper - lower).

    Args:
        lower: Lower interval bound(s).
        upper: Upper interval bound(s).

    Returns:
        Mean interval width in ug/m3.
    """
    l_arr = np.asarray(lower, dtype=float)
    u_arr = np.asarray(upper, dtype=float)

    if len(l_arr) == 0:
        return 0.0

    width = u_arr - l_arr
    return float(np.mean(width))


def coverage_80(
    y_true: Sequence[float] | np.ndarray,
    q10: Sequence[float] | np.ndarray,
    q90: Sequence[float] | np.ndarray,
) -> float:
    """Compute empirical coverage of the central 80% interval [q10, q90].

    Nominal target (Hypothesis H6): 0.80 +/- 0.05.
    """
    return interval_coverage(y_true, q10, q90)


def coverage_50(
    y_true: Sequence[float] | np.ndarray,
    q25: Sequence[float] | np.ndarray,
    q75: Sequence[float] | np.ndarray,
) -> float:
    """Compute empirical coverage of the central 50% interval [q25, q75].

    Nominal target: 0.50 +/- 0.05.
    """
    return interval_coverage(y_true, q25, q75)


def coverage_below_median(
    y_true: Sequence[float] | np.ndarray,
    q50: Sequence[float] | np.ndarray,
) -> float:
    """Compute unconditional frequency P(y <= q50).

    Nominal target (Hypothesis H6): 0.50 +/- 0.07.
    """
    y_arr = np.asarray(y_true, dtype=float)
    m_arr = np.asarray(q50, dtype=float)
    if len(y_arr) == 0:
        return 0.0
    return float(np.mean(y_arr <= m_arr))


def compute_pit_values(
    y_true: Sequence[float] | np.ndarray,
    quantiles_19: np.ndarray,
) -> np.ndarray:
    """Compute Probability Integral Transform (PIT) values u_i = F_i(y_i) in [0, 1].

    Reconstructs the continuous distribution F_i with the canonical tail model for each forecast
    and evaluates the cumulative probability at observation y_i.

    Args:
        y_true: (N,) array of observed values.
        quantiles_19: (N, 19) array of 19 quantile predictions.

    Returns:
        (N,) array of PIT values in [0, 1].
    """
    y_arr = np.asarray(y_true, dtype=float)
    q_mat = np.atleast_2d(quantiles_19)

    if q_mat.shape[1] != 19:
        raise ValueError("quantiles_19 must have exactly 19 columns.")
    if len(y_arr) != q_mat.shape[0]:
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match quantile rows ({q_mat.shape[0]})."
        )

    n_samples = len(y_arr)
    pit_values = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        qf = QuantileFunction(q_mat[i])
        pit_values[i] = float(cast(float, qf.cdf(float(y_arr[i]))))

    return np.clip(pit_values, 0.0, 1.0)


def compute_pit_histogram(
    pit_values: Sequence[float] | np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PIT histogram counts and bin edges across [0, 1].

    Args:
        pit_values: Array of PIT values in [0, 1].
        n_bins: Number of histogram bins (default 20, matching 19-quantile boundaries).

    Returns:
        Tuple of (counts (n_bins,), bin_edges (n_bins + 1,)).
    """
    vals = np.asarray(pit_values, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, edges = np.histogram(vals, bins=bins)
    return counts, edges


def pit_uniformity_test(
    pit_values: Sequence[float] | np.ndarray,
    n_bins: int = 20,
) -> dict[str, Any]:
    """Perform Pearson chi-squared goodness-of-fit test of PIT uniformity against U(0, 1).

    Args:
        pit_values: Array of PIT values.
        n_bins: Number of bins (default 20; df = 19).

    Returns:
        Dict with chi2_stat, p_value, df, and is_uniform (p_value >= 0.05).
    """
    counts, _ = compute_pit_histogram(pit_values, n_bins=n_bins)
    n_total = int(np.sum(counts))
    if n_total == 0:
        return {"chi2_stat": 0.0, "p_value": 1.0, "df": n_bins - 1, "is_uniform": True}

    expected = n_total / float(n_bins)
    chi2_stat = float(np.sum(((counts - expected) ** 2) / expected))
    df = n_bins - 1
    p_value = float(stats.chi2.sf(chi2_stat, df=df))

    return {
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "df": df,
        "is_uniform": bool(p_value >= 0.05),
    }


def compute_quantile_reliability(
    y_true: Sequence[float] | np.ndarray,
    quantiles_19: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute empirical exceedance / coverage rate for each of the 19 canonical quantile levels.

    For each tau_k, computes p_hat_k = (1/N) * sum(1{y <= q_{tau_k}}).

    Args:
        y_true: (N,) array of true observations.
        quantiles_19: (N, 19) array of predicted quantiles.

    Returns:
        Dict with nominal_taus (19,), empirical_probs (19,), and coverage_errors (19,).
    """
    y_arr = np.asarray(y_true, dtype=float)
    q_mat = np.atleast_2d(quantiles_19)

    if q_mat.shape[1] != 19:
        raise ValueError("quantiles_19 must have exactly 19 columns.")
    if len(y_arr) != q_mat.shape[0]:
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match quantile rows ({q_mat.shape[0]})."
        )

    # Broadcast: shape (N, 19)
    y_col = y_arr[:, np.newaxis]
    hits = y_col <= q_mat
    empirical_probs = np.mean(hits, axis=0)
    nominal_taus = CANONICAL_TAUS.copy()
    errors = empirical_probs - nominal_taus

    return {
        "nominal_taus": nominal_taus,
        "empirical_probs": empirical_probs,
        "coverage_errors": errors,
    }


def compute_calibration_report(
    y_true: Sequence[float] | np.ndarray,
    quantiles_19: np.ndarray,
) -> CalibrationReport:
    """Generate a comprehensive calibration and reliability diagnostic report.

    Args:
        y_true: (N,) array of true observations.
        quantiles_19: (N, 19) array of predicted quantiles.

    Returns:
        CalibrationReport instance.
    """
    y_arr = np.asarray(y_true, dtype=float)
    q_mat = np.atleast_2d(quantiles_19)

    if q_mat.shape[1] != 19:
        raise ValueError("quantiles_19 must have exactly 19 columns.")
    if len(y_arr) != q_mat.shape[0]:
        raise ValueError(
            f"Observation length ({len(y_arr)}) must match quantile rows ({q_mat.shape[0]})."
        )

    # Canonical indices:
    # tau: 0.05 (idx 0), 0.10 (idx 1), 0.25 (idx 4), 0.50 (idx 9), 0.75 (idx 14), 0.90 (idx 17), 0.95 (idx 18)
    q10 = q_mat[:, 1]
    q25 = q_mat[:, 4]
    q50 = q_mat[:, 9]
    q75 = q_mat[:, 14]
    q90 = q_mat[:, 17]

    cov_80 = coverage_80(y_arr, q10, q90)
    shp_80 = interval_sharpness(q10, q90)
    cov_50 = coverage_50(y_arr, q25, q75)
    shp_50 = interval_sharpness(q25, q75)
    med_cov = coverage_below_median(y_arr, q50)

    pit_vals = compute_pit_values(y_arr, q_mat)
    counts, edges = compute_pit_histogram(pit_vals, n_bins=20)
    test_res = pit_uniformity_test(pit_vals, n_bins=20)

    rel_diag = compute_quantile_reliability(y_arr, q_mat)

    return CalibrationReport(
        n_samples=len(y_arr),
        coverage_80=cov_80,
        sharpness_80=shp_80,
        coverage_50=cov_50,
        sharpness_50=shp_50,
        median_coverage=med_cov,
        pit_chi2=float(test_res["chi2_stat"]),
        pit_p_value=float(test_res["p_value"]),
        is_pit_uniform=bool(test_res["is_uniform"]),
        pit_counts=[int(c) for c in counts],
        pit_bin_edges=[float(e) for e in edges],
        nominal_taus=[float(t) for t in rel_diag["nominal_taus"]],
        empirical_probs=[float(p) for p in rel_diag["empirical_probs"]],
        coverage_errors=[float(e) for e in rel_diag["coverage_errors"]],
    )


__all__ = [
    "CalibrationReport",
    "compute_calibration_report",
    "compute_pit_histogram",
    "compute_pit_values",
    "compute_quantile_reliability",
    "coverage_50",
    "coverage_80",
    "coverage_below_median",
    "interval_coverage",
    "interval_sharpness",
    "pit_uniformity_test",
]
