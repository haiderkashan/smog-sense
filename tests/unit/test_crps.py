import numpy as np
import pytest
from scipy.stats import norm

from smogsense.evaluation.crps import crps_from_quantiles
from smogsense.models.distribution import QuantileFunction


def analytical_normal_crps(mu: float, sigma: float, y: float) -> float:
    """Exact CRPS for Normal distribution N(mu, sigma^2)."""
    omega = (y - mu) / sigma
    return float(
        sigma * (omega * (2 * norm.cdf(omega) - 1) + 2 * norm.pdf(omega) - 1 / np.sqrt(np.pi))
    )


def test_crps_non_negative() -> None:
    q = np.linspace(10, 50, 19)
    # Regardless of where the observation is, CRPS must be >= 0
    obs_vals = [-10.0, 0.0, 30.0, 100.0]
    for obs in obs_vals:
        crps_val = crps_from_quantiles(q, obs)
        assert crps_val >= 0.0


def test_crps_analytical_crosscheck() -> None:
    """Test K=19 approximation against analytical Normal CRPS.

    Roadmap requires absolute error <= 0.5%.
    """
    mu = 30.0
    sigma = 5.0

    # Generate 19 quantiles for N(mu, sigma^2)
    taus = np.arange(0.05, 0.951, 0.05)
    q = norm.ppf(taus, loc=mu, scale=sigma)

    # Test across a few observation values
    obs_vals = [20.0, 30.0, 40.0, 50.0]
    for obs in obs_vals:
        approx_crps = float(crps_from_quantiles(q, obs))
        exact_crps = float(analytical_normal_crps(mu, sigma, obs))

        # Relative error
        err = abs(approx_crps - exact_crps) / exact_crps
        assert err <= 0.005, (
            f"CRPS error {err * 100:.2f}% exceeds 0.5% tolerance (exact={exact_crps}, approx={approx_crps})"
        )


def test_crps_method_in_quantile_function() -> None:
    q = np.linspace(10, 50, 19)
    qf = QuantileFunction(q)

    obs = 30.0
    # Should match the standalone evaluator
    val1 = qf.crps(obs)
    val2 = crps_from_quantiles(q, obs)
    assert np.isclose(val1, float(val2))


def test_crps_invalid_quantiles_shape() -> None:
    import pytest

    # Must have 19 quantiles
    q_invalid = np.linspace(10, 50, 10)
    with pytest.raises(ValueError, match="requires exactly 19 quantiles"):
        crps_from_quantiles(q_invalid, 30.0)


def test_crps_vectorized_evaluation() -> None:
    q_matrix = np.array([np.linspace(10, 50, 19), np.linspace(20, 60, 19)])
    obs_arr = np.array([25.0, 35.0])
    res = crps_from_quantiles(q_matrix, obs_arr)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert (res >= 0.0).all()


def test_crps_missing_trapezoid_func(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytest

    # Monkeypatch trapezoid and trapz away on numpy
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.delattr(np, "trapz", raising=False)

    q = np.linspace(10, 50, 19)
    with pytest.raises(AttributeError, match="NumPy does not have trapezoid or trapz"):
        crps_from_quantiles(q, 30.0)
