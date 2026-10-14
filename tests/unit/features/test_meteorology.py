"""Unit and property-based tests for meteorology and atmospheric physics features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from smogsense.features.meteorology import (
    dilution_index,
    extract_block_meteorology_features,
    lower_tropospheric_stability,
    potential_temperature,
    relative_humidity_magnus,
    stagnation_run_length,
    ventilation_coefficient,
)


def test_ventilation_coefficient_flooring() -> None:
    """Test that VC floors at 10.0 m^2/s even with zero wind and zero PBLH."""
    # Zero wind, zero PBLH -> PBLH floored at 50, wind speed 0 -> max(50 * 0, 10.0) = 10.0
    vc_zero = ventilation_coefficient(blh_m=0.0, u10_ms=0.0, v10_ms=0.0)
    assert vc_zero == pytest.approx(10.0, rel=1e-5)

    # Negative PBLH and negative wind components
    vc_neg = ventilation_coefficient(blh_m=-100.0, u10_ms=0.0, v10_ms=0.0)
    assert vc_neg == pytest.approx(10.0, rel=1e-5)

    # Moderate wind and PBLH: PBLH=200, u=3, v=4 -> ws=5 -> 200 * 5 = 1000.0
    vc_normal = ventilation_coefficient(blh_m=200.0, u10_ms=3.0, v10_ms=4.0)
    assert vc_normal == pytest.approx(1000.0, rel=1e-5)


def test_ventilation_coefficient_array() -> None:
    """Test ventilation coefficient on numpy arrays."""
    pblh = np.array([0.0, 100.0, 500.0])
    u10 = np.array([0.0, 1.0, 3.0])
    v10 = np.array([0.0, 0.0, 4.0])
    vc = ventilation_coefficient(pblh, u10, v10)
    assert np.all(vc >= 10.0)
    assert vc[0] == pytest.approx(10.0)
    assert vc[1] == pytest.approx(100.0)
    assert vc[2] == pytest.approx(2500.0)


def test_dilution_index_monotonicity() -> None:
    """Dilution index = 1 / VC is monotone decreasing in PBLH and wind speed."""
    vc1 = ventilation_coefficient(blh_m=200.0, u10_ms=2.0, v10_ms=0.0)
    vc2 = ventilation_coefficient(blh_m=400.0, u10_ms=2.0, v10_ms=0.0)
    vc3 = ventilation_coefficient(blh_m=400.0, u10_ms=4.0, v10_ms=0.0)

    di1 = dilution_index(vc1)
    di2 = dilution_index(vc2)
    di3 = dilution_index(vc3)

    assert di1 > di2 > di3 > 0.0


def test_relative_humidity_magnus_known_values() -> None:
    """Magnus RH(10 C, 8 C dewpoint) within 0.5 percentage points of reference."""
    # At T=10 C, Td=8 C, standard psychrometric reference is approx 87.4%
    rh = relative_humidity_magnus(t2m_c=10.0, d2m_c=8.0)
    assert 86.9 <= rh <= 88.0

    # Equal temperature and dewpoint -> saturation (100%)
    rh_sat = relative_humidity_magnus(t2m_c=25.0, d2m_c=25.0)
    assert rh_sat == pytest.approx(100.0, rel=1e-3)

    # Dewpoint > Temperature is clipped to 100%
    rh_supersat = relative_humidity_magnus(t2m_c=15.0, d2m_c=18.0)
    assert rh_supersat == pytest.approx(100.0)


def test_potential_temperature_and_stability() -> None:
    """Test Poisson equation potential temperature and lower tropospheric stability."""
    # At 1000 hPa, theta == T
    t_1000 = 290.0
    theta_1000 = potential_temperature(temp_k=t_1000, pressure_hpa=1000.0)
    assert theta_1000 == pytest.approx(t_1000, rel=1e-5)

    # At 925 hPa, theta > T
    t_925 = 285.0
    theta_925 = potential_temperature(temp_k=t_925, pressure_hpa=925.0)
    assert theta_925 > t_925

    # Inversion: warm dry air aloft at 925 hPa, cold surface
    lts_inversion, is_inversion = lower_tropospheric_stability(
        t925_k=295.0,
        t2m_k=280.0,
        surface_pressure_hpa=1013.25,
    )
    assert is_inversion is True
    assert lts_inversion > 0.0

    # Strong lapse condition (surface very warm, 925 hPa cold)
    lts_lapse, is_inversion_lapse = lower_tropospheric_stability(
        t925_k=270.0,
        t2m_k=305.0,
        surface_pressure_hpa=1013.25,
    )
    assert is_inversion_lapse is False
    assert lts_lapse < 0.0


def test_stagnation_run_length() -> None:
    """Test stagnation run length calculation."""
    # 10 hours: first 4 hours windy (2500 m2/s), next 6 hours stagnant (500 m2/s)
    vc = np.array([2500.0, 2500.0, 2500.0, 2500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])

    max_run = stagnation_run_length(vc, threshold_m2s=1000.0)
    assert max_run == 6

    # Test with no stagnation
    assert stagnation_run_length(np.array([2000.0, 3000.0])) == 0


def test_extract_block_meteorology_features() -> None:
    """Test block-level feature aggregation and harmonic mean dilution index."""
    # Create 72 hours of hourly meteorological data starting at 00Z
    issuance_utc = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    timestamps = pd.date_range("2026-11-01 00:00:00", periods=72, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "blh_m": np.linspace(100.0, 800.0, 72),
            "u10_ms": np.full(72, 2.0),
            "v10_ms": np.full(72, 1.5),
            "t2m_c": np.full(72, 18.0),
            "d2m_c": np.full(72, 10.0),
            "surface_pressure_hpa": np.full(72, 1010.0),
        }
    )

    horizons = (24, 48, 72)
    feats = extract_block_meteorology_features(
        df, issuance_utc, horizons=horizons, timezone="Asia/Karachi"
    )

    for h in [24, 48, 72]:
        assert f"f_met_vc_min_h{h}" in feats
        assert f"f_met_vc_mean_h{h}" in feats
        assert f"f_met_ln_vc_min_h{h}" in feats
        assert f"f_met_ln_vc_mean_h{h}" in feats
        assert f"f_met_dilution_index_h{h}" in feats
        assert f"f_met_rh_mean_h{h}" in feats

        # Check harmonic mean relation: dilution_index must strictly match mean(1 / VC)
        block_pblh = df["blh_m"].iloc[h - 24 : h].to_numpy(dtype=np.float64)
        block_u = df["u10_ms"].iloc[h - 24 : h].to_numpy(dtype=np.float64)
        block_v = df["v10_ms"].iloc[h - 24 : h].to_numpy(dtype=np.float64)
        expected_vc = ventilation_coefficient(block_pblh, block_u, block_v)
        expected_di = float(np.mean(1.0 / expected_vc))
        assert feats[f"f_met_dilution_index_h{h}"] == pytest.approx(expected_di, rel=1e-5)


@given(
    blh=st.floats(min_value=-50.0, max_value=4000.0),
    u10=st.floats(min_value=-30.0, max_value=30.0),
    v10=st.floats(min_value=-30.0, max_value=30.0),
)
def test_hypothesis_ventilation_coefficient_bounds(blh: float, u10: float, v10: float) -> None:
    """Property test: VC is always >= 10.0 m^2/s and non-negative."""
    vc = ventilation_coefficient(blh, u10, v10)
    assert vc >= 10.0
    di = dilution_index(vc)
    assert 0.0 < di <= 0.1
