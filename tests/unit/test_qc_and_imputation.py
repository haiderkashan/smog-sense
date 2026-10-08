import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from smogsense.preprocessing.imputation import pchip_impute
from smogsense.preprocessing.qc import (
    FLATLINE,
    IMPUTED,
    NEGATIVE_CLIPPED,
    RANGE_REJECT,
    SPIKE,
    apply_qc,
)


def test_qc_range_reject():
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=4, freq="h"),
            "pm25_ugm3": [-10.0, -5.0, 1500.0, 1501.0],
            "other_col": [1, 2, 3, 4],
        }
    )
    out = apply_qc(df)

    # -10 -> RANGE_REJECT, NaN
    # -5 -> NEGATIVE_CLIPPED, 0.0
    # 1500 -> OK (no flag, 1500)
    # 1501 -> RANGE_REJECT, NaN

    assert pd.isna(out.loc[0, "pm25_ugm3"])
    assert out.loc[0, "qc_flags"] & RANGE_REJECT

    assert out.loc[1, "pm25_ugm3"] == 0.0
    assert out.loc[1, "qc_flags"] & NEGATIVE_CLIPPED

    assert out.loc[2, "pm25_ugm3"] == 1500.0
    assert out.loc[2, "qc_flags"] == 0

    assert pd.isna(out.loc[3, "pm25_ugm3"])
    assert out.loc[3, "qc_flags"] & RANGE_REJECT

    assert len(out) == 4
    assert list(out["other_col"]) == [1, 2, 3, 4]


def test_qc_negative_clipped():
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=4, freq="h"),
            "pm25_ugm3": [-4.0, -0.1, 0.0, 10.0],
        }
    )
    out = apply_qc(df)

    assert out.loc[0, "pm25_ugm3"] == 0.0
    assert out.loc[0, "qc_flags"] & NEGATIVE_CLIPPED

    assert out.loc[1, "pm25_ugm3"] == 0.0
    assert out.loc[1, "qc_flags"] & NEGATIVE_CLIPPED

    assert out.loc[2, "pm25_ugm3"] == 0.0
    assert out.loc[2, "qc_flags"] == 0

    assert out.loc[3, "pm25_ugm3"] == 10.0
    assert out.loc[3, "qc_flags"] == 0


def test_qc_flatline():
    # Exactly 5 identical -> no flatline
    # Exactly 6 identical -> flatline
    # Zero -> no flatline
    df = pd.DataFrame(
        {
            "sensor_id": [1] * 12,
            "ts_utc": pd.date_range("2026-01-01", periods=12, freq="h"),
            "pm25_ugm3": [
                10,
                10,
                10,
                10,
                10,  # 5 identical -> no
                0,
                0,
                0,
                0,
                0,
                0,
                0,  # 7 identical zeros -> no
            ],
        }
    )
    out = apply_qc(df)
    assert out["qc_flags"].sum() == 0
    assert out["pm25_ugm3"].notna().all()

    df2 = pd.DataFrame(
        {
            "sensor_id": [1] * 7,
            "ts_utc": pd.date_range("2026-01-01", periods=7, freq="h"),
            "pm25_ugm3": [10, 10, 10, 10, 10, 10, 10],
        }
    )
    out2 = apply_qc(df2)
    assert (out2["qc_flags"] == FLATLINE).all()
    assert out2["pm25_ugm3"].isna().all()


def test_qc_flatline_interrupted_and_gaps():
    df = pd.DataFrame(
        {
            "sensor_id": [1] * 6,
            "ts_utc": [
                pd.Timestamp("2026-01-01 00:00:00"),
                pd.Timestamp("2026-01-01 01:00:00"),
                pd.Timestamp("2026-01-01 02:00:00"),
                pd.Timestamp("2026-01-01 03:00:00"),
                pd.Timestamp("2026-01-01 04:00:00"),
                pd.Timestamp("2026-01-01 06:00:00"),  # gap
            ],
            "pm25_ugm3": [10, 10, 10, 10, 10, 10],
        }
    )
    out = apply_qc(df)
    assert out["qc_flags"].sum() == 0


def test_qc_spike():
    df = pd.DataFrame(
        {
            "sensor_id": [1] * 10,
            "ts_utc": pd.date_range("2026-01-01", periods=10, freq="h"),
            "pm25_ugm3": [10, 12, 11, 200, 10, 9, 11, 10, 10, 10],
        }
    )
    out = apply_qc(df)
    # 200 is a spike.
    assert pd.isna(out.loc[3, "pm25_ugm3"])
    assert out.loc[3, "qc_flags"] & SPIKE
    # ensure it didn't delete rows
    assert len(out) == 10


def test_qc_flags_combine():
    df = pd.DataFrame(
        {
            "sensor_id": [1],
            "ts_utc": pd.date_range("2026-01-01", periods=1, freq="h"),
            "pm25_ugm3": [-10],
            "qc_flags": [512],
        }
    )
    out = apply_qc(df)
    assert out.loc[0, "qc_flags"] == (512 | RANGE_REJECT)


def test_pchip_impute_gaps():
    s = pd.Series(
        [10.0, np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, np.nan, 30.0],
        index=pd.date_range("2026-01-01", periods=9, freq="h"),
    )
    out = pchip_impute(s, max_gap=3)

    # The 2-gap should be filled
    assert pd.notna(out.iloc[1])
    assert pd.notna(out.iloc[2])

    # The 4-gap should remain NaN
    assert pd.isna(out.iloc[4])
    assert pd.isna(out.iloc[5])
    assert pd.isna(out.iloc[6])
    assert pd.isna(out.iloc[7])


def test_pchip_impute_edges():
    s = pd.Series([np.nan, 10.0, 20.0, np.nan])
    out = pchip_impute(s, max_gap=3)
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[3])


def test_pchip_regression_overshoot():
    # 3-hour internal gap between approx 520 and 480
    y = np.array([200.0, 520.0, np.nan, np.nan, np.nan, 480.0, 300.0])
    s = pd.Series(y, index=pd.date_range("2026-01-01", periods=7, freq="h"))

    # Ordinary cubic spline overshoot check
    valid = ~np.isnan(y)
    x = np.arange(len(y))
    cs = CubicSpline(x[valid], y[valid])
    cs_out = cs(x)
    assert cs_out[2:5].max() > 550  # Overshoots envelope (mathematical requirement)

    out = pchip_impute(s, max_gap=3)
    # PCHIP remains within envelope (roughly between 470 and 530)
    assert (out.iloc[2:5] <= 520).all()
    assert (out.iloc[2:5] >= 480).all()


def test_imputed_flag():
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=3, freq="h"),
            "pm25_ugm3": [10.0, np.nan, 20.0],
            "qc_flags": [0, 8, 0],  # existing spike flag on the NaN
        }
    )

    # Mocking how pipeline might do it
    imputed = pchip_impute(df["pm25_ugm3"], max_gap=3)
    mask = imputed.notna() & df["pm25_ugm3"].isna()

    df["pm25_ugm3"] = imputed
    df.loc[mask, "qc_flags"] |= IMPUTED

    assert df.loc[1, "pm25_ugm3"] > 0
    assert df.loc[1, "qc_flags"] == (8 | IMPUTED)
    assert df.loc[0, "qc_flags"] == 0
    assert df.loc[2, "qc_flags"] == 0


def test_qc_empty_input():
    df = pd.DataFrame(columns=["sensor_id", "ts_utc", "pm25_ugm3"])
    out = apply_qc(df)
    assert len(out) == 0

    s = pd.Series(dtype=float)
    out_s = pchip_impute(s)
    assert len(out_s) == 0


def test_impute_invalid_max_gap():
    with pytest.raises(ValueError):
        pchip_impute(pd.Series([10, np.nan, 20]), max_gap=-1)
