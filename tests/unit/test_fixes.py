import numpy as np

from smogsense.models.distribution import QuantileFunction


def test_lower_tail_zero_mass() -> None:
    # If a distribution is floored at 0, passing tau=0 to PPF should yield exactly 0.
    qf = QuantileFunction(np.linspace(10, 100, 19))
    assert qf.ppf(0.0) == 0.0

    # CDF of 0 should return a non-zero probability if there is a mass at 0
    prob_at_zero = qf.cdf(0.0)
    assert prob_at_zero > 0.0

    # The PPF of that specific probability should also be 0
    assert np.isclose(qf.ppf(prob_at_zero), 0.0)


def test_m0_nan_rejection() -> None:
    import pytest

    from smogsense.models.baselines import M0Persistence

    m0 = M0Persistence(np.linspace(-10, 10, 19))

    with pytest.raises(ValueError, match="complete valid 24h block"):
        m0.predict([10, 20, np.nan, 30])
