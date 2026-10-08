"""smogsense.models.distribution - Quantile-function object with tail model.

Piecewise-linear interior, exponential upper tail (linear in -ln(1-tau)), log-linear lower tail
floored at 0; provides ppf, cdf, prob_exceed and exact-integral CRPS. Includes monotone
rearrangement of crossing quantiles.

Specification: docs/evaluation-strategy.md -> 'CRPS and quantile scoring'
"""

import math
from collections.abc import Sequence

import numpy as np


class QuantileFunction:
    def __init__(
        self,
        quantiles: Sequence[float] | np.ndarray,
        taus: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Initialize the quantile function and compute tail parameters.

        Args:
            quantiles: Sequence of 19 quantile values.
            taus: Optional probability levels for the quantiles.
        """
        self.q = np.array(quantiles, dtype=float)
        if taus is None:
            self.taus = np.arange(0.05, 0.951, 0.05)
        else:
            self.taus = np.array(taus, dtype=float)

        if len(self.q) != 19 or len(self.taus) != 19:
            raise ValueError("QuantileFunction requires exactly 19 quantiles.")

        # Monotone rearrangement (sorting)
        self.q = np.sort(self.q)

        # Calculate tail slopes
        denom_lo = math.log(self.taus[1]) - math.log(self.taus[0])
        self.s_lo = (self.q[1] - self.q[0]) / denom_lo if denom_lo > 0 else 0.0

        lambda_hi = -math.log(1.0 - self.taus[18])
        lambda_hi_minus_1 = -math.log(1.0 - self.taus[17])
        denom_hi = lambda_hi - lambda_hi_minus_1
        self.s_hi = (self.q[18] - self.q[17]) / denom_hi if denom_hi > 0 else 0.0

    def ppf(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Point Percentile Function (inverse CDF)."""
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        if np.any((tau_arr < 0.0) | (tau_arr > 1.0)):
            raise ValueError("tau must be between 0 and 1.")

        res = np.atleast_1d(np.interp(tau_arr, self.taus, self.q)).copy()

        # Lower tail
        mask_lo = tau_arr < self.taus[0]
        if np.any(mask_lo):
            with np.errstate(divide="ignore"):
                val = self.q[0] + self.s_lo * (np.log(tau_arr[mask_lo]) - math.log(self.taus[0]))
            val = np.maximum(0.0, val)
            res[mask_lo] = val

        # Upper tail
        mask_hi = tau_arr > self.taus[-1]
        if np.any(mask_hi):
            # Clip to avoid log(0)
            tau_safe = np.clip(tau_arr[mask_hi], None, 1.0 - 1e-12)
            val = self.q[-1] + self.s_hi * (
                -np.log(1.0 - tau_safe) - (-math.log(1.0 - self.taus[-1]))
            )
            res[mask_hi] = val

        return float(res[0]) if np.isscalar(tau) else res

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative Distribution Function."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        res = np.zeros_like(x_arr)

        mask_inter = (x_arr >= self.q[0]) & (x_arr <= self.q[-1])
        if np.any(mask_inter):
            res[mask_inter] = np.interp(x_arr[mask_inter], self.q, self.taus)

        mask_lo = x_arr < self.q[0]
        if np.any(mask_lo):
            if self.s_lo <= 0:
                res[mask_lo] = 0.0
            else:
                tau_lo = np.exp((x_arr[mask_lo] - self.q[0]) / self.s_lo + math.log(self.taus[0]))
                res[mask_lo] = tau_lo

        mask_hi = x_arr > self.q[-1]
        if np.any(mask_hi):
            if self.s_hi <= 0:
                res[mask_hi] = 1.0
            else:
                one_minus_tau = np.exp(
                    -(x_arr[mask_hi] - self.q[-1]) / self.s_hi + math.log(1.0 - self.taus[-1])
                )
                res[mask_hi] = 1.0 - one_minus_tau

        res = np.clip(res, 0.0, 1.0)
        # Ensure x < 0 yields 0 exactly due to floor
        res[x_arr < 0.0] = 0.0

        return float(res[0]) if np.isscalar(x) else res

    def prob_exceed(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Probability of exceedance P(X > threshold)."""
        x = np.asarray(threshold)
        return 1.0 - self.cdf(x)

    def crps(self, observation: float) -> float:
        """Calculate CRPS for this distribution against a single observation."""
        from smogsense.evaluation.crps import crps_from_quantiles

        return float(crps_from_quantiles(self.q, observation))
