import numpy as np
import pandas as pd
import pytest

from smogsense.models.baselines import M0Persistence, M1Cams


def test_m0_persistence_n_zero() -> None:
    # N=0 means residual_quantiles is None
    m0 = M0Persistence(None)
    obs = pd.Series([10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match="undefined at N=0"):
        m0.predict(obs)


def test_m0_persistence_probabilistic() -> None:
    # Create fake 19 residual quantiles (e.g. from -10 to +10)
    residuals = np.linspace(-10, 10, 19)
    m0 = M0Persistence(residuals)

    obs = pd.Series([10.0, 20.0, 30.0, 20.0])  # complete block, mean is 20.0

    qf = m0.predict(obs)

    # 20.0 + (-10) = 10
    assert np.isclose(qf.ppf(0.05), 10.0)
    # 20.0 + 10 = 30
    assert np.isclose(qf.ppf(0.95), 30.0)


def test_m1_cams_n_zero() -> None:
    m1 = M1Cams(None)
    cams_val = 50.0

    res = m1.predict(cams_val)
    # Deterministic at N=0
    assert isinstance(res, float)
    assert np.isclose(res, 50.0)


def test_m1_cams_probabilistic() -> None:
    residuals = np.linspace(-5, 5, 19)
    m1 = M1Cams(residuals)

    cams_val = 50.0
    qf = m1.predict(cams_val)
    from smogsense.models.distribution import QuantileFunction

    assert isinstance(qf, QuantileFunction)
    # Probabilistic output
    assert np.isclose(qf.ppf(0.05), 45.0)
    assert np.isclose(qf.ppf(0.50), 50.0)
    assert np.isclose(qf.ppf(0.95), 55.0)
