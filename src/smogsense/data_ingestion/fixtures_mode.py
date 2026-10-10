"""smogsense.data_ingestion.fixtures_mode — Offline replay of recorded payloads (SMOGSENSE_MODE=fixtures).

Lets every client read tests/fixtures/ instead of the network so that the full pipeline, the
demo and the unit tests run without credentials or internet.

Specification: docs/system-architecture.md -> 'Component responsibilities and CLI contract'
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from smogsense.utils.io import validate_frame

logger = logging.getLogger(__name__)


def is_fixtures_mode() -> bool:
    """Return True if running in offline fixtures mode."""
    mode_env = os.getenv("SMOGSENSE_MODE", "").lower().strip()
    return mode_env in ("fixtures", "demo", "test")


def _load_obs_hourly_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/observations_hourly.schema.yaml"),
        Path("/app/data/schemas/observations_hourly.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/observations_hourly.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(p)
    raise FileNotFoundError(f"observations_hourly schema not found in candidates: {candidates}")


def load_fixture_locations(domain: str = "lahore") -> pd.DataFrame:
    """Load station locations from recorded fixtures or fallback synthetic network."""
    loc_file = Path("tests/fixtures/openaq/locations_page.json")
    if not loc_file.exists():
        loc_file = Path(__file__).resolve().parents[3] / "tests/fixtures/openaq/locations_page.json"

    if loc_file.exists():
        try:
            with loc_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            records: list[dict[str, Any]] = []
            for item in data.get("results", []):
                loc_id = item.get("id")
                coords = item.get("coordinates", {})
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                if loc_id is not None and lat is not None and lon is not None:
                    records.append(
                        {
                            "location_id": int(loc_id),
                            "name": str(item.get("name", f"Station {loc_id}")),
                            "lat": float(lat),
                            "lon": float(lon),
                            "is_reference": bool(item.get("isMonitor", False)),
                            "is_monitor": bool(item.get("isMonitor", False)),
                            "provider": str((item.get("provider") or {}).get("name", "Unknown")),
                        }
                    )
            if records:
                return pd.DataFrame(records)
        except Exception as exc:
            logger.debug("Failed to parse fixture locations: %s", exc)

    # Authoritative default station panel for Lahore
    return pd.DataFrame(
        [
            {
                "location_id": 1001,
                "name": "US Embassy Lahore",
                "lat": 31.55,
                "lon": 74.33,
                "is_reference": True,
                "is_monitor": True,
                "provider": "State Department",
            },
            {
                "location_id": 1002,
                "name": "Gulberg Reference Monitor",
                "lat": 31.52,
                "lon": 74.35,
                "is_reference": True,
                "is_monitor": True,
                "provider": "EPD Punjab",
            },
            {
                "location_id": 1003,
                "name": "Model Town Sensor",
                "lat": 31.48,
                "lon": 74.32,
                "is_reference": False,
                "is_monitor": False,
                "provider": "AirGradient",
            },
        ]
    )


def load_fixture_observations(
    stations_df: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
    pull_id: str = "run_fixtures",
    domain: str = "lahore",
) -> pd.DataFrame:
    """Generate schema-valid hourly observations for offline testing and dry runs."""
    dates = pd.date_range(start_utc, end_utc, freq="h")
    rows: list[dict[str, Any]] = []

    for _, station in stations_df.iterrows():
        loc_id = int(station["location_id"])
        sensor_id = loc_id * 5  # Deterministic sensor ID mapping
        is_ref = bool(station.get("is_reference") or station.get("is_monitor") is True)
        provider = str(station.get("provider", "State Department"))

        for dt in dates:
            val = 45.0 + float((dt.hour * 3) % 25)
            rows.append(
                {
                    "location_id": loc_id,
                    "sensor_id": sensor_id,
                    "domain": domain,
                    "ts_utc": dt,
                    "pm25_ugm3": val,
                    "pm25_raw_ugm3": val,
                    "rh_pct": 65.0,
                    "temp_c": 22.0,
                    "qc_flags": np.int32(0),
                    "imputed": False,
                    "is_reference": is_ref,
                    "provider": provider,
                    "source": "openaq_api",
                    "ingested_at_utc": end_utc,
                    "colocated_group_id": None,
                    "n_revisions": np.int32(1),
                    "last_revised_utc": end_utc,
                    "available_at_utc": dt + timedelta(hours=4),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Type casting to match Pandera schema
    df["location_id"] = df["location_id"].astype("int64")
    df["sensor_id"] = df["sensor_id"].astype("int64")
    df["domain"] = df["domain"].astype("string")
    df["provider"] = df["provider"].astype("string")
    df["source"] = df["source"].astype("string")
    df["qc_flags"] = df["qc_flags"].astype("int32")
    df["imputed"] = df["imputed"].astype(bool)
    df["is_reference"] = df["is_reference"].astype(bool)
    df["n_revisions"] = df["n_revisions"].astype("int32")

    schema = _load_obs_hourly_schema()
    return validate_frame(df, schema)


def load_fixture_cams(
    stations_df: pd.DataFrame,
    issuance_utc: datetime,
) -> pd.DataFrame:
    """Generate CAMS forecast timeseries covering 72h leads for stations and centroid."""
    rows: list[dict[str, Any]] = []
    locations = [str(x) for x in stations_df["location_id"].unique()]
    if "centroid" not in locations:
        locations.append("centroid")

    for loc in locations:
        for h in range(1, 73):
            target_hour = issuance_utc + timedelta(hours=h)
            rows.append(
                {
                    "location_id": str(loc),
                    "target_hour_utc": target_hour,
                    "pm25_ugm3": 55.0,
                }
            )

    return pd.DataFrame(rows)
