"""Data contracts: every Pandera YAML schema loads and enforces what it claims; the public JSON schema is valid."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.errors
import pandera.pandas as pa
import pytest
from jsonschema import Draft202012Validator, FormatChecker

pytestmark = pytest.mark.contract

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "data" / "schemas"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.yaml"))
EXPECTED = {
    "observations_hourly",
    "station_registry",
    "cams_station_series",
    "fire_exposure",
    "feature_store",
    "forecast_log",
    "score_log",
}
T0 = pd.Timestamp("2026-11-05T00:00:00Z")
Q_NAMES = [f"q{round((0.05 * (i + 1)) * 100):02d}" for i in range(19)]
VALIDATION_ERRORS = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)


def validate_frame(schema: pa.DataFrameSchema, frame: pd.DataFrame) -> pd.DataFrame:
    """The pattern production code must follow: reset the index first (Pandera 0.33 + pandas 3 bug on duplicate index)."""
    return schema.validate(frame.reset_index(drop=True))


@pytest.fixture(scope="module")
def schemas() -> dict[str, pa.DataFrameSchema]:
    return {
        p.name.removesuffix(".schema.yaml"): pa.DataFrameSchema.from_yaml(str(p))
        for p in SCHEMA_FILES
    }


def test_all_expected_schemas_present_and_loadable(schemas: dict[str, pa.DataFrameSchema]) -> None:
    assert set(schemas) == EXPECTED


def test_every_column_is_documented(schemas: dict[str, pa.DataFrameSchema]) -> None:
    undocumented = [
        (n, c) for n, s in schemas.items() for c, col in s.columns.items() if not col.description
    ]
    assert not undocumented, undocumented


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [1, 1],
            "sensor_id": [10, 10],
            "domain": ["lahore"] * 2,
            "ts_utc": [T0, T0 + pd.Timedelta(hours=1)],
            "pm25_ugm3": [180.0, np.nan],
            "pm25_raw_ugm3": [181.0, np.nan],
            "rh_pct": [70.0, 71.0],
            "temp_c": [12.0, 12.0],
            "qc_flags": np.array([0, 64], dtype="int32"),
            "imputed": [False, False],
            "is_reference": [False, False],
            "provider": ["x", "x"],
            "source": ["openaq_api"] * 2,
            "ingested_at_utc": [T0 + pd.Timedelta(hours=2)] * 2,
            "colocated_group_id": [None, None],
            "n_revisions": np.array([1, 1], dtype="int32"),
            "last_revised_utc": [T0, T0],
            "available_at_utc": [T0, T0],
        }
    )


def test_observations_accepts_valid_and_rejects_out_of_range(
    schemas: dict[str, pa.DataFrameSchema],
) -> None:
    schema = schemas["observations_hourly"]
    validate_frame(schema, _observations())
    bad = _observations()
    bad.loc[0, "pm25_ugm3"] = 5000.0
    with pytest.raises(VALIDATION_ERRORS):
        validate_frame(schema, bad)


def test_observations_rejects_duplicate_keys_even_with_duplicate_index(
    schemas: dict[str, pa.DataFrameSchema],
) -> None:
    obs = _observations()
    duplicated = pd.concat([obs, obs.iloc[[0]]])  # index is [0, 1, 0]
    assert not duplicated.index.is_unique
    with pytest.raises(VALIDATION_ERRORS):
        validate_frame(schemas["observations_hourly"], duplicated)


def test_pandera_duplicate_index_pitfall_is_still_present(
    schemas: dict[str, pa.DataFrameSchema],
) -> None:
    """Documents the upstream bug. If this starts raising a clean SchemaError, the reset_index workaround can go."""
    obs = _observations()
    duplicated = pd.concat([obs, obs.iloc[[0]]])
    with pytest.raises((ValueError, *VALIDATION_ERRORS)):
        schemas["observations_hourly"].validate(duplicated)


def _forecast_row() -> pd.DataFrame:
    row: dict[str, Any] = {
        "run_id": "test_run",
        "issuance_utc": T0,
        "generated_at_utc": T0 + pd.Timedelta(minutes=25),
        "domain": "lahore",
        "level": "city",
        "point_id": "city:lahore",
        "horizon_h": np.int32(24),
        "window_start_utc": T0,
        "window_end_utc": T0 + pd.Timedelta(hours=24),
        "method": "m7_hybrid_maml",
        "model_version": "models-v0.2.0",
        "mode": "full",
        "cams_base_time_utc": T0 - pd.Timedelta(hours=12),
        "data_cutoff_utc": T0 - pd.Timedelta(hours=2),
        "git_sha": "abc",
        "config_hash": "def",
        "calibrated": False,
    }
    row.update({q: 100.0 + 10 * i for i, q in enumerate(Q_NAMES)})
    return pd.DataFrame([row])


def test_forecast_log_has_nineteen_quantile_columns(schemas: dict[str, pa.DataFrameSchema]) -> None:
    schema = schemas["forecast_log"]
    assert all(q in schema.columns for q in Q_NAMES)
    validate_frame(schema, _forecast_row())


def test_forecast_log_rejects_unknown_method(schemas: dict[str, pa.DataFrameSchema]) -> None:
    bad = _forecast_row()
    bad["method"] = "not_a_method"
    with pytest.raises(VALIDATION_ERRORS):
        validate_frame(schemas["forecast_log"], bad)


def test_feature_store_regex_columns_and_strictness(schemas: dict[str, pa.DataFrameSchema]) -> None:
    frame = pd.DataFrame(
        {
            "domain": ["delhi"],
            "location_id": [5],
            "issuance_utc": [T0],
            "source_split": ["train"],
            "feature_set_version": ["fs1"],
            "config_hash": ["h"],
            "y_h24": [210.0],
            "y_h48": [np.nan],
            "y_h72": [np.nan],
            "valid_h24": [True],
            "valid_h48": [False],
            "valid_h72": [False],
            "n_available_stations": np.array([3], dtype="int32"),
            "f_met_vc10_min": np.array([1500.0], dtype="float32"),
            "m_met_vc10_min": [False],
        }
    )
    validate_frame(schemas["feature_store"], frame)
    frame["unexpected_column"] = 1
    with pytest.raises(VALIDATION_ERRORS):
        validate_frame(schemas["feature_store"], frame)


# ----------------------------------------------------------------------------- public JSON contract
@pytest.fixture(scope="module")
def bulletin_schema() -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / "bulletin.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bulletin_example() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "bulletin_valid_example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_bulletin_schema_is_valid_json_schema(bulletin_schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(bulletin_schema)


def test_bulletin_example_validates(
    bulletin_schema: dict[str, Any], bulletin_example: dict[str, Any]
) -> None:
    errors = list(
        Draft202012Validator(bulletin_schema, format_checker=FormatChecker()).iter_errors(
            bulletin_example
        )
    )
    assert not errors, errors[:3]


def test_bulletin_example_semantic_invariants(bulletin_example: dict[str, Any]) -> None:
    order = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    for horizon in bulletin_example["horizons"]:
        values = [horizon["quantiles_ugm3"][k] for k in order]
        assert values == sorted(values), "quantiles must be monotone"
        assert abs(sum(horizon["probabilities"]["by_category"].values()) - 1.0) < 2e-3
        exceed = [e["probability"] for e in horizon["probabilities"]["exceed"]]
        assert exceed == sorted(exceed, reverse=True)


MUTATIONS = {
    "missing_q90": lambda d: d["horizons"][0]["quantiles_ugm3"].pop("q90"),
    "unknown_mode": lambda d: d.__setitem__("mode", "turbo"),
    "two_horizons_only": lambda d: d.__setitem__("horizons", d["horizons"][:2]),
    "negative_concentration": lambda d: d["horizons"][1]["quantiles_ugm3"].__setitem__("q05", -3),
    "legacy_aqi_scheme": lambda d: d.__setitem__("aqi_scheme", "us_epa_2012"),
    "extrapolated_aqi_field": lambda d: d.__setitem__("aqi_index", 1900),
    "bad_git_sha": lambda d: d["model"].__setitem__("git_sha", "not-a-sha"),
    "probability_above_one": lambda d: d["horizons"][0]["probabilities"]["by_category"].__setitem__(
        "good", 1.4
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_bulletin_schema_rejects_broken_documents(
    name: str, bulletin_schema: dict[str, Any], bulletin_example: dict[str, Any]
) -> None:
    broken = copy.deepcopy(bulletin_example)
    MUTATIONS[name](broken)
    assert list(
        Draft202012Validator(bulletin_schema, format_checker=FormatChecker()).iter_errors(broken)
    )
