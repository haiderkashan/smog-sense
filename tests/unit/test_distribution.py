import numpy as np

from smogsense.models.distribution import QuantileFunction


def test_quantile_function_init() -> None:
    q = np.arange(1, 20)
    qf = QuantileFunction(q)
    assert len(qf.q) == 19
    assert np.allclose(qf.taus, np.arange(0.05, 0.951, 0.05))


def test_monotone_rearrangement() -> None:
    # 19 quantiles out of order
    q_unsorted = np.linspace(50, 10, 19)
    qf = QuantileFunction(q_unsorted)
    # Must be sorted ascending
    assert np.all(np.diff(qf.q) >= 0)


def test_ppf_cdf_inverse() -> None:
    # Use positive quantiles so zero floor doesn't truncate our test range
    q = np.linspace(10, 50, 19)
    qf = QuantileFunction(q)

    # Test interior and tails, but avoid exact 0.0 where floor flattens CDF
    taus = np.linspace(0.01, 0.99, 50)
    x = qf.ppf(taus)
    taus_back = qf.cdf(x)

    np.testing.assert_allclose(taus, taus_back, rtol=1e-5, atol=1e-5)


def test_zero_floor() -> None:
    q = np.linspace(1, 10, 19)
    qf = QuantileFunction(q)

    # tau=1e-6 will be negative before flooring
    x_tiny = qf.ppf(1e-10)
    assert x_tiny == 0.0

    # CDF of 0 should be exactly the tau where it hits 0
    # Actually, cdf(-5) should be 0.0
    assert qf.cdf(-5.0) == 0.0
    assert qf.cdf(0.0) >= 0.0


def test_prob_exceed() -> None:
    q = np.linspace(10, 50, 19)
    qf = QuantileFunction(q)

    # Median is 30 (index 9)
    # prob_exceed(30) = 1 - cdf(30) = 1 - 0.5 = 0.5
    assert np.isclose(qf.prob_exceed(30.0), 0.5)


def test_flat_quantiles() -> None:
    # All quantiles are the same (e.g. perfect certainty in interior)
    q = np.ones(19) * 20.0
    qf = QuantileFunction(q)

    # ppf should return 20 for everything inside
    assert qf.ppf(0.5) == 20.0
    assert qf.ppf(0.05) == 20.0
    assert qf.ppf(0.95) == 20.0

    # slopes should be 0
    assert qf.s_lo == 0.0
    assert qf.s_hi == 0.0

    # tail CDF should step cleanly
    assert qf.cdf(19.9) == 0.0
    assert qf.cdf(20.1) == 1.0
