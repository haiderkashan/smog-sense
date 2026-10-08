"""smogsense.models.baselines - Reference forecasters M0-M3.

Persistence, raw CAMS, CAMS with trailing-window bias correction, and climatological residual-
quantile baselines; all emit the same quantile object as learned models.

Specification: docs/evaluation-strategy.md -> 'Baselines'
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from smogsense.models.distribution import QuantileFunction


class M0Persistence:
    """Baseline M0: Persistence.

    Uses the latest observed 24-h block mean repeated at all horizons.
    Probabilistic via residual quantiles from the available set A.
    Undefined at N=0 (raises ValueError if residual_quantiles is None).
    """

    def __init__(self, residual_quantiles: Sequence[float] | np.ndarray | None) -> None:
        """Initialize M0.

        Args:
            residual_quantiles: 19 quantiles of historical residuals from set A. None if N=0.
        """
        self.residual_quantiles = None
        if residual_quantiles is not None:
            res_arr = np.array(residual_quantiles, dtype=float)
            if len(res_arr) != 19:
                raise ValueError("M0 requires exactly 19 residual quantiles.")
            self.residual_quantiles = res_arr

    def predict(self, recent_24h_obs: pd.Series | np.ndarray | list[float]) -> QuantileFunction:
        """Predict the distribution for any horizon based on the last 24h block.

        Args:
            recent_24h_obs: The latest available 24-h block of observations.

        Returns:
            QuantileFunction representing the probabilistic forecast.

        Raises:
            ValueError: if N=0 (no residual_quantiles provided) or if recent_24h_obs is completely empty/NaN.
        """
        if self.residual_quantiles is None:
            raise ValueError("M0 Persistence is undefined at N=0 (no residual quantiles).")

        obs = np.asarray(recent_24h_obs, dtype=float)
        if np.isnan(obs).any() or len(obs) == 0:
            raise ValueError("M0 Persistence requires a complete valid 24h block.")
            raise ValueError("No valid observations provided to M0.")

        mean_val = float(np.mean(obs))
        return QuantileFunction(mean_val + self.residual_quantiles)


class M1Cams:
    """Baseline M1: Raw CAMS.

    Uses the raw CAMS PM2.5 forecast at the correct lead time.
    Deterministic at N=0, probabilistic via residual quantiles from A at N>=1.
    """

    def __init__(self, residual_quantiles: Sequence[float] | np.ndarray | None) -> None:
        """Initialize M1.

        Args:
            residual_quantiles: 19 quantiles of historical residuals from set A. None if N=0.
        """
        self.residual_quantiles = None
        if residual_quantiles is not None:
            res_arr = np.array(residual_quantiles, dtype=float)
            if len(res_arr) != 19:
                raise ValueError("M1 requires exactly 19 residual quantiles.")
            self.residual_quantiles = res_arr

    def predict(self, cams_forecast: float) -> QuantileFunction | float:
        """Predict the distribution for a specific horizon given its CAMS forecast.

        Args:
            cams_forecast: The raw CAMS point forecast for the target horizon.

        Returns:
            QuantileFunction if N>=1, else float (deterministic) if N=0.
        """
        if self.residual_quantiles is None:
            return float(cams_forecast)

        return QuantileFunction(cams_forecast + self.residual_quantiles)
