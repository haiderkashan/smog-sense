"""smogsense.evaluation.crps - CRPS from quantile functions and point forecasts.

Exact integration of the interpolated quantile function with the tail model; analytic and Monte-
Carlo cross-checks via scoringrules in tests. Includes threshold-weighted CRPS (twCRPS) for smog
emergency tails (Gneiting & Ranjan, 2011; Allen et al., 2023).

Specification: docs/evaluation-strategy.md -> 'CRPS and quantile scoring'
"""

from typing import cast

import numpy as np


def crps_from_quantiles(
    quantiles: np.ndarray, observation: float | np.ndarray
) -> float | np.ndarray:
    """Calculate CRPS from 19 quantile levels using numerical integration.

    Integration evaluates the pinball loss over 400,000 points on the interval [1e-6, 1-1e-6],
    matching the documentation requirement for a 19-quantile discrete distribution with tail model.

    Args:
        quantiles: (19,) or (N, 19) array of non-crossing quantiles.
        observation: scalar or (N,) array of true values.

    Returns:
        CRPS scalar or array.
    """
    from smogsense.models.distribution import QuantileFunction

    q_arr = np.atleast_2d(quantiles)
    obs_arr = np.atleast_1d(observation)

    if q_arr.shape[1] != 19:
        raise ValueError("crps_from_quantiles requires exactly 19 quantiles.")

    n_samples = q_arr.shape[0]
    res = np.zeros(n_samples)

    # 400,000 points on [1e-6, 1-1e-6]
    taus = np.linspace(1e-6, 1.0 - 1e-6, 400_000)

    # use np.trapezoid if available (NumPy 2.0+), else np.trapz (NumPy < 2.0)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is None:
        raise AttributeError("NumPy does not have trapezoid or trapz function.")

    for i in range(n_samples):
        qf = QuantileFunction(q_arr[i])
        q_tau = cast(np.ndarray, qf.ppf(taus))
        u = obs_arr[i] - q_tau
        rho = u * (taus - (u < 0))
        integral = trapz_func(2.0 * rho, taus)
        res[i] = integral

    # Unpack scalar if needed
    if np.isscalar(observation) and quantiles.ndim == 1:
        return float(res[0])
    return res


def threshold_weighted_crps(
    quantiles: np.ndarray,
    observation: float | np.ndarray,
    threshold: float = 125.5,
    n_points: int = 50_000,
) -> float | np.ndarray:
    """Calculate threshold-weighted CRPS (twCRPS) targeting values >= threshold.

    Following Gneiting & Ranjan (2011) Theorem 1 and Allen et al. (2023), with indicator weight
    w(x) = 1{x >= threshold}, twCRPS corresponds to transforming the predictive distribution and
    observation via the chaining function v(x) = max(x, threshold):
        twCRPS(F, y; r) = 2 int_0^1 rho_tau(v(y) - v(Q(tau))) d tau

    Args:
        quantiles: (19,) or (N, 19) array of 19 quantile levels.
        observation: scalar or (N,) array of true values.
        threshold: threshold in ug/m3 (default 125.5 for Very Unhealthy smog).
        n_points: integration resolution across [1e-6, 1-1e-6].

    Returns:
        twCRPS scalar or array.
    """
    from smogsense.models.distribution import QuantileFunction

    q_arr = np.atleast_2d(quantiles)
    obs_arr = np.atleast_1d(observation)

    if q_arr.shape[1] != 19:
        raise ValueError("threshold_weighted_crps requires exactly 19 quantiles.")

    n_samples = q_arr.shape[0]
    res = np.zeros(n_samples)

    taus = np.linspace(1e-6, 1.0 - 1e-6, n_points)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is None:
        raise AttributeError("NumPy does not have trapezoid or trapz function.")

    thresh_float = float(threshold)

    for i in range(n_samples):
        qf = QuantileFunction(q_arr[i])
        q_tau = cast(np.ndarray, qf.ppf(taus))
        v_q = np.maximum(q_tau, thresh_float)
        v_obs = max(float(obs_arr[i]), thresh_float)
        u = v_obs - v_q
        rho = u * (taus - (u < 0))
        integral = trapz_func(2.0 * rho, taus)
        res[i] = integral

    if np.isscalar(observation) and quantiles.ndim == 1:
        return float(res[0])
    return res


# Canonical alias
tw_crps = threshold_weighted_crps
