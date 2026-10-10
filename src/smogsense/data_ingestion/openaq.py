"""smogsense.data_ingestion.openaq -- OpenAQ v3 REST client (near-real-time observations).

Discovers PM2.5 sensors inside a domain bounding box, paginates /v3/locations and
/v3/sensors/{id}/hours, and returns hourly records plus coverage metadata. Free tier is 60
requests/minute and 2,000/hour per API key.

Public contract (implemented in Phase 1a):
- list_locations(client, domain) -> DataFrame (registry candidates).
- fetch_hourly(client, locations_df, start_utc, end_utc) -> DataFrame (pivoted, one row per station-hour).
- Respects pagination via limit/page and stops on found/limit arithmetic, never on empty-page
  guessing.

Specification: docs/data-engineering.md -> 'Phase 1 data audit (acceptance criteria)'
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
from structlog import get_logger

from smogsense.config import Settings
from smogsense.data_ingestion.base import CircuitBreakerError, ResilientClient
from smogsense.errors import QuotaExceeded

logger = get_logger(__name__)

# Authoritative OpenAQ v3 Parameter IDs
# PM2.5 = 2, Temperature = 100, Relative Humidity = 98
PARAM_ID_PM25 = 2
PARAM_ID_TEMP = 100
PARAM_ID_RH = 98


async def list_locations(client: ResilientClient, domain: str) -> pd.DataFrame:
    """Fetch all OpenAQ locations in a bounding box."""
    settings = Settings.load("configs")
    domain_conf = settings.domains.get("domains", {}).get(domain)
    if not domain_conf:
        raise ValueError(f"Domain {domain} not found in configs/domains.yaml")

    bbox = domain_conf["station_bbox"]
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    locations: list[dict[str, Any]] = []
    page = 1
    limit = 1000

    # To handle duplicates from pagination
    seen_locations = set()

    while True:
        url = "https://api.openaq.org/v3/locations"
        params = {"bbox": bbox_str, "parameters_id": PARAM_ID_PM25, "limit": limit, "page": page}

        resp = await client.get(url, params=params)
        data = resp.json()

        # Infinite loop protection
        if not data.get("results"):
            break

        for loc in data.get("results", []):
            loc_id = loc.get("id")
            if loc_id is None or loc_id in seen_locations:
                continue

            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")

            # Coordinate validation
            if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue

            seen_locations.add(loc_id)

            name = loc.get("name")
            provider = (loc.get("provider") or {}).get("name")

            # Preserve True/False/None
            is_monitor = loc.get("isMonitor")

            first_dt = (loc.get("datetimeFirst") or {}).get("utc")
            last_dt = (loc.get("datetimeLast") or {}).get("utc")

            # Collect multiple sensors per parameter, picking the one with the highest coverage
            best_pm25, best_pm25_cov = None, -1.0
            best_rh, best_rh_cov = None, -1.0
            best_temp, best_temp_cov = None, -1.0

            for sensor in loc.get("sensors", []):
                param_id = sensor.get("parameter", {}).get("id")
                s_id = sensor.get("id")
                cov = float(sensor.get("coverage", {}).get("percentComplete", 0.0)) / 100.0

                if param_id == PARAM_ID_PM25:
                    if cov > best_pm25_cov or (
                        cov == best_pm25_cov and (best_pm25 is None or s_id < best_pm25)
                    ):
                        best_pm25 = s_id
                        best_pm25_cov = cov
                elif param_id == PARAM_ID_RH:
                    if cov > best_rh_cov or (
                        cov == best_rh_cov and (best_rh is None or s_id < best_rh)
                    ):
                        best_rh = s_id
                        best_rh_cov = cov
                elif param_id == PARAM_ID_TEMP and (
                    cov > best_temp_cov
                    or (cov == best_temp_cov and (best_temp is None or s_id < best_temp))
                ):
                    best_temp = s_id
                    best_temp_cov = cov

            locations.append(
                {
                    "location_id": loc_id,
                    "name": name,
                    "provider": provider,
                    "lat": float(lat),
                    "lon": float(lon),
                    "is_monitor": is_monitor,
                    "sensor_id_pm25": best_pm25,
                    "sensor_id_rh": best_rh,
                    "sensor_id_temp": best_temp,
                    "first_datetime": first_dt,
                    "last_datetime": last_dt,
                    "lifecycle_uptime": max(best_pm25_cov, 0.0),
                }
            )

        found = data.get("meta", {}).get("found", 0)
        if limit * page >= found:
            break
        page += 1

    df = pd.DataFrame(locations)
    if not df.empty:
        df["first_datetime"] = pd.to_datetime(df["first_datetime"], utc=True)
        df["last_datetime"] = pd.to_datetime(df["last_datetime"], utc=True)
    else:
        df = pd.DataFrame(
            columns=[
                "location_id",
                "name",
                "provider",
                "lat",
                "lon",
                "is_monitor",
                "sensor_id_pm25",
                "sensor_id_rh",
                "sensor_id_temp",
                "first_datetime",
                "last_datetime",
                "lifecycle_uptime",
            ]
        )
    return df


async def _fetch_single_sensor_hourly(
    client: ResilientClient,
    location_id: int,
    sensor_id: int,
    param_col: str,
    start_utc: datetime,
    end_utc: datetime,
    pull_id: str | None = None,
    first_seen_utc: datetime | None = None,
) -> pd.DataFrame:
    page = 1
    limit = 1000
    rows: list[dict[str, Any]] = []

    date_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    active_pull_id = pull_id or str(uuid.uuid4())
    active_first_seen = first_seen_utc or datetime.now(UTC)

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        params = {"datetime_from": date_from, "datetime_to": date_to, "limit": limit, "page": page}

        resp = await client.get(url, params=params)
        data = resp.json()

        if not data.get("results"):
            break

        for row in data.get("results", []):
            period = row.get("period") or {}
            dt_from = (period.get("datetimeFrom") or {}).get("utc")
            dt_to = (period.get("datetimeTo") or {}).get("utc")
            val = row.get("value")

            if dt_from is not None:
                ts_dt = pd.to_datetime(dt_from, utc=True)
                end_dt = (
                    pd.to_datetime(dt_to, utc=True)
                    if dt_to is not None
                    else ts_dt + pd.Timedelta(hours=1)
                )
            elif dt_to is not None:
                end_dt = pd.to_datetime(dt_to, utc=True)
                ts_dt = end_dt - pd.Timedelta(hours=1)
            else:
                continue

            rows.append(
                {
                    "location_id": location_id,
                    "sensor_id": sensor_id,
                    "ts_utc": ts_dt,
                    "hour_end_utc": end_dt,
                    "first_seen_utc": active_first_seen,
                    "pull_id": active_pull_id,
                    param_col: val,
                }
            )

        found = data.get("meta", {}).get("found", 0)
        if limit * page >= found:
            break
        page += 1

    return pd.DataFrame(rows)


async def fetch_hourly(
    client: ResilientClient,
    locations_df: pd.DataFrame,
    start_utc: datetime,
    end_utc: datetime,
    pull_id: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly data for multiple locations and pivot to one row per station-hour."""
    active_pull_id = pull_id or str(uuid.uuid4())
    batch_first_seen = datetime.now(UTC)
    tasks = []

    for _, row in locations_df.iterrows():
        loc_id = row["location_id"]
        if pd.notna(row.get("sensor_id_pm25")):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client,
                    loc_id,
                    int(row["sensor_id_pm25"]),
                    "pm25_ugm3",
                    start_utc,
                    end_utc,
                    pull_id=active_pull_id,
                    first_seen_utc=batch_first_seen,
                )
            )
        if pd.notna(row.get("sensor_id_rh")):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client,
                    loc_id,
                    int(row["sensor_id_rh"]),
                    "rh_pct",
                    start_utc,
                    end_utc,
                    pull_id=active_pull_id,
                    first_seen_utc=batch_first_seen,
                )
            )
        if pd.notna(row.get("sensor_id_temp")):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client,
                    loc_id,
                    int(row["sensor_id_temp"]),
                    "temperature_c",
                    start_utc,
                    end_utc,
                    pull_id=active_pull_id,
                    first_seen_utc=batch_first_seen,
                )
            )

    empty_cols = [
        "location_id",
        "sensor_id",
        "ts_utc",
        "hour_end_utc",
        "first_seen_utc",
        "pull_id",
        "pm25_ugm3",
        "rh_pct",
        "temperature_c",
        "provider",
    ]

    if not tasks:
        return pd.DataFrame(columns=empty_cols)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_dfs: list[pd.DataFrame] = []
    for res in results:
        if isinstance(res, (CircuitBreakerError, QuotaExceeded)) or (
            isinstance(res, httpx.HTTPStatusError) and res.response.status_code == 429
        ):
            raise res
        elif isinstance(res, Exception):
            logger.warning("sensor_hourly_fetch_failed", error=str(res))
        elif isinstance(res, pd.DataFrame) and not res.empty:
            valid_dfs.append(res)

    if not valid_dfs:
        return pd.DataFrame(columns=empty_cols)

    # Combine all long-form rows
    long_df = pd.concat(valid_dfs, ignore_index=True)
    long_df["ts_utc"] = pd.to_datetime(long_df["ts_utc"], utc=True)
    long_df["hour_end_utc"] = pd.to_datetime(long_df["hour_end_utc"], utc=True)
    long_df["first_seen_utc"] = pd.to_datetime(long_df["first_seen_utc"], utc=True)

    # Pivot into the required format: one row per location_id, ts_utc
    pivoted = long_df.groupby(["location_id", "ts_utc"], as_index=False).first()

    # Map provider and pm25 sensor_id from locations_df if present
    if "provider" in locations_df.columns:
        prov_map = locations_df.set_index("location_id")["provider"].to_dict()
        pivoted["provider"] = pivoted["location_id"].map(prov_map).fillna("openaq")
    else:
        pivoted["provider"] = "openaq"

    if "sensor_id_pm25" in locations_df.columns:
        s_pm25_map = locations_df.set_index("location_id")["sensor_id_pm25"].to_dict()
        pivoted["sensor_id"] = (
            pivoted["location_id"].map(s_pm25_map).fillna(pivoted.get("sensor_id"))
        )

    if "sensor_id" not in pivoted.columns:
        pivoted["sensor_id"] = pivoted["location_id"]
    pivoted["sensor_id"] = pivoted["sensor_id"].fillna(pivoted["location_id"]).astype(str)

    if "pull_id" not in pivoted.columns:
        pivoted["pull_id"] = active_pull_id
    pivoted["pull_id"] = pivoted["pull_id"].fillna(active_pull_id).astype(str)

    if "first_seen_utc" not in pivoted.columns:
        pivoted["first_seen_utc"] = batch_first_seen
    pivoted["first_seen_utc"] = pd.to_datetime(
        pivoted["first_seen_utc"].fillna(batch_first_seen), utc=True
    )

    if "hour_end_utc" not in pivoted.columns:
        pivoted["hour_end_utc"] = pivoted["ts_utc"] + pd.Timedelta(hours=1)
    pivoted["hour_end_utc"] = pd.to_datetime(
        pivoted["hour_end_utc"].fillna(pivoted["ts_utc"] + pd.Timedelta(hours=1)), utc=True
    )

    # Ensure all target columns exist even if no data was returned for one parameter
    for col in ["pm25_ugm3", "rh_pct", "temperature_c"]:
        if col not in pivoted.columns:
            pivoted[col] = None

    return pivoted
