"""smogsense.evaluation.significance — Diebold-Mariano test and block bootstrap.

HAC-corrected DM test on CRPS differences with Harvey-Leybourne-Newbold small-sample correction;
stationary block bootstrap confidence intervals on days; Holm-Bonferroni correction.

Specification: docs/evaluation-strategy.md → 'Statistical inference'
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Statistical outcome of the Diebold-Mariano hypothesis test with HLN correction."""

    dm_stat: float
    dm_star: float
    p_value: float
    mean_loss_diff: float
    hac_se: float
    n_samples: int
    n_eff: float
    df: int
    hln_factor: float
    bandwidth: int
    lag1_autocorr: float
    alternative: str

    def to_dict(self) -> dict[str, Any]:
        """Convert DM result to dictionary format."""
        return {
            "dm_stat": self.dm_stat,
            "dm_star": self.dm_star,
            "p_value": self.p_value,
            "mean_loss_diff": self.mean_loss_diff,
            "hac_se": self.hac_se,
            "n_samples": self.n_samples,
            "n_eff": self.n_eff,
            "df": self.df,
            "hln_factor": self.hln_factor,
            "bandwidth": self.bandwidth,
            "lag1_autocorr": self.lag1_autocorr,
            "alternative": self.alternative,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Empirical and percentile confidence intervals from stationary block bootstrap."""

    point_estimate: float
    bootstrap_mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    replicates_count: int
    mean_block_length: float

    def to_dict(self) -> dict[str, Any]:
        """Convert bootstrap result to dictionary format."""
        return {
            "point_estimate": self.point_estimate,
            "bootstrap_mean": self.bootstrap_mean,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "replicates_count": self.replicates_count,
            "mean_block_length": self.mean_block_length,
        }


def diebold_mariano_test(
    loss_a: Sequence[float] | np.ndarray,
    loss_b: Sequence[float] | np.ndarray | None = None,
    horizon_h: int = 24,
    bandwidth: int | None = None,
    h_lead: int | None = None,
    alternative: str = "two-sided",
) -> DieboldMarianoResult:
    """Perform Diebold-Mariano test on daily loss differentials with HLN small-sample correction.

    Tests H0: E[d_t] = 0 on daily loss differential series d_t = Loss^A_t - Loss^B_t.
    Variance is estimated via Heteroskedasticity and Autocorrelation Consistent (HAC / Newey-West)
    estimator with Bartlett kernel and bandwidth K = ceil(horizon_h / 24) + 2.
    Degrees of freedom and test statistic are corrected using Harvey, Leybourne & Newbold (1997).

    Args:
        loss_a: Daily loss series of Model A (or pre-computed differential series if loss_b is None).
        loss_b: Daily loss series of Reference Model B (optional).
        horizon_h: Forecast horizon in hours (24, 48, or 72).
        bandwidth: HAC Bartlett lag bandwidth (default: ceil(horizon_h / 24) + 2).
        h_lead: Forecast step ahead (default: ceil(horizon_h / 24)).
        alternative: One of "two-sided", "less" (Model A has lower loss), "greater".

    Returns:
        DieboldMarianoResult instance.
    """
    arr_a = np.asarray(loss_a, dtype=float)
    if loss_b is not None:
        arr_b = np.asarray(loss_b, dtype=float)
        if len(arr_a) != len(arr_b):
            raise ValueError(
                f"loss_a length ({len(arr_a)}) must match loss_b length ({len(arr_b)})."
            )
        d = arr_a - arr_b
    else:
        d = arr_a.copy()

    # Drop NaNs
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        raise ValueError(f"Diebold-Mariano test requires at least 3 valid samples, got {n}.")

    if h_lead is None:
        h_lead = max(1, math.ceil(horizon_h / 24.0))

    if bandwidth is None:
        bandwidth = h_lead + 2

    bandwidth = min(bandwidth, n - 1)

    mean_d = float(np.mean(d))
    e = d - mean_d
    gamma_0 = float(np.mean(e**2))

    gammas: list[float] = []
    for k in range(1, bandwidth + 1):
        gamma_k = float(np.sum(e[k:] * e[:-k]) / n)
        gammas.append(gamma_k)

    # Newey-West HAC variance with Bartlett kernel weights w_k = 1 - k / (K + 1)
    v_hac = gamma_0 + 2.0 * sum(
        (1.0 - k / (bandwidth + 1.0)) * g for k, g in enumerate(gammas, start=1)
    )
    v_hac = max(v_hac, 1e-12)
    hac_se = float(np.sqrt(v_hac / n))

    # Raw DM statistic
    dm_stat = float(mean_d / hac_se)

    # Harvey-Leybourne-Newbold (HLN 1997) small-sample correction factor
    term = (n + 1 - 2 * h_lead + (h_lead * (h_lead - 1.0)) / n) / n
    hln_factor = float(np.sqrt(max(0.0, term)))
    dm_star = float(dm_stat * hln_factor)

    df = n - 1
    if alternative == "two-sided":
        p_val = float(2.0 * stats.t.sf(abs(dm_star), df=df))
    elif alternative == "less":
        # H1: mean(d) < 0 (Model A is better than Model B)
        p_val = float(stats.t.cdf(dm_star, df=df))
    elif alternative == "greater":
        # H1: mean(d) > 0 (Model A is worse than Model B)
        p_val = float(stats.t.sf(dm_star, df=df))
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', got '{alternative}'."
        )

    # Effective sample size n_eff = n * (1 - rho_1) / (1 + rho_1)
    rho_1 = float(gammas[0] / gamma_0) if gamma_0 > 0.0 and len(gammas) > 0 else 0.0
    rho_1 = float(np.clip(rho_1, -0.999, 0.999))
    n_eff = float(np.clip(n * (1.0 - rho_1) / (1.0 + rho_1), 1.0, float(n)))

    return DieboldMarianoResult(
        dm_stat=dm_stat,
        dm_star=dm_star,
        p_value=p_val,
        mean_loss_diff=mean_d,
        hac_se=hac_se,
        n_samples=n,
        n_eff=n_eff,
        df=df,
        hln_factor=hln_factor,
        bandwidth=bandwidth,
        lag1_autocorr=rho_1,
        alternative=alternative,
    )


def stationary_block_bootstrap(
    data: np.ndarray | pd.DataFrame | Sequence[Any],
    statistic_fn: Callable[[Any], float],
    mean_block_days: float = 7.0,
    replicates: int = 2000,
    ci: float = 0.95,
    date_col: str | None = None,
    seed: int | None = 42,
) -> BootstrapResult:
    """Compute confidence intervals via Stationary Block Bootstrap (Politis & Romano, 1994).

    Resamples consecutive temporal blocks with random block lengths drawn from a Geometric
    distribution with mean mean_block_days (p = 1 / mean_block_days).
    If a DataFrame with date_col is supplied, calendar dates are resampled to keep all co-observed
    stations within a day intact (preventing artificial cross-sectional independence).

    Args:
        data: 1D array, 2D array, or pandas DataFrame of samples.
        statistic_fn: Callable mapping a resampled slice of data to a scalar metric.
        mean_block_days: Mean block length in days (default 7.0 days).
        replicates: Number of bootstrap resamples B (default 2000).
        ci: Confidence level in (0, 1) (default 0.95).
        date_col: Optional column name in DataFrame representing calendar date.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult instance.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / float(mean_block_days)

    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        df = cast(pd.DataFrame, data)
        if date_col is not None and date_col in df.columns:
            # Group rows by date
            unique_dates = df[date_col].drop_duplicates().to_numpy()
            n_units = len(unique_dates)
            date_to_indices = df.groupby(date_col).indices
        else:
            n_units = len(df)
            unique_dates = None
            date_to_indices = None
    else:
        arr = np.asarray(data)
        n_units = len(arr)
        unique_dates = None
        date_to_indices = None

    if n_units < 2:
        raise ValueError(f"Stationary bootstrap requires at least 2 temporal units, got {n_units}.")

    # Evaluate point estimate on full data
    point_est = float(statistic_fn(data))

    boot_estimates = np.zeros(replicates, dtype=float)

    for b in range(replicates):
        # Generate stationary bootstrap indices of length n_units
        sampled_indices = np.empty(n_units, dtype=int)
        sampled_indices[0] = int(rng.integers(0, n_units))

        u = rng.uniform(0.0, 1.0, size=n_units - 1)
        new_starts = rng.integers(0, n_units, size=n_units - 1)

        for t in range(1, n_units):
            if u[t - 1] < p:
                sampled_indices[t] = int(new_starts[t - 1])
            else:
                sampled_indices[t] = int((sampled_indices[t - 1] + 1) % n_units)

        # Slice data
        if is_df:
            df = cast(pd.DataFrame, data)
            if unique_dates is not None and date_to_indices is not None:
                # Gather row indices for selected dates
                selected_dates = unique_dates[sampled_indices]
                row_idx_list = [date_to_indices[dt] for dt in selected_dates]
                boot_row_indices = np.concatenate(row_idx_list)
                resampled_data: Any = df.iloc[boot_row_indices].reset_index(drop=True)
            else:
                resampled_data = df.iloc[sampled_indices].reset_index(drop=True)
        else:
            arr = np.asarray(data)
            resampled_data = arr[sampled_indices]

        boot_estimates[b] = float(statistic_fn(resampled_data))

    boot_mean = float(np.mean(boot_estimates))
    std_error = float(np.std(boot_estimates, ddof=1))

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_estimates, lower_pct))
    ci_upper = float(np.percentile(boot_estimates, upper_pct))

    return BootstrapResult(
        point_estimate=point_est,
        bootstrap_mean=boot_mean,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci,
        replicates_count=replicates,
        mean_block_length=mean_block_days,
    )


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[float, float, bool]]:
    """Apply Holm-Bonferroni step-down correction for family-wise error rate control (T1-14).

    Controls FWER over a family of hypothesis tests (e.g. {H1, H2, H3_24, H3_48, H3_72, H4}).

    Args:
        p_values: Sequence of raw p-values.
        alpha: Target overall significance level (default 0.05).

    Returns:
        List of tuples (raw_p, adjusted_p, is_significant) in original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    p_arr = np.asarray(p_values, dtype=float)
    sorted_indices = np.argsort(p_arr)

    adjusted = np.zeros(m, dtype=float)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, float(p_arr[idx] * (m - rank)))

    # Enforce monotonicity across sorted ranks
    for r in range(1, m):
        curr_idx = sorted_indices[r]
        prev_idx = sorted_indices[r - 1]
        adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx])

    result: list[tuple[float, float, bool]] = []
    for raw, adj in zip(p_arr, adjusted, strict=False):
        sig = bool(adj <= alpha)
        result.append((float(raw), float(adj), sig))

    return result


__all__ = [
    "BootstrapResult",
    "DieboldMarianoResult",
    "diebold_mariano_test",
    "holm_bonferroni_correction",
    "stationary_block_bootstrap",
]
