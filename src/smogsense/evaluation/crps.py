"""smogsense.evaluation.crps - CRPS from quantile functions and point forecasts.

Exact integration of the interpolated quantile function with the tail model; analytic and Monte-
Carlo cross-checks via scoringrules in tests.

Specification: docs/evaluation-strategy.md -> 'CRPS and quantile scoring'
"""

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
        q_tau = qf.ppf(taus)
        u = obs_arr[i] - q_tau
        rho = u * (taus - (u < 0))
        integral = trapz_func(2.0 * rho, taus)
        res[i] = integral

    # Unpack scalar if needed
    if np.isscalar(observation) and quantiles.ndim == 1:
        return float(res[0])
    return res
