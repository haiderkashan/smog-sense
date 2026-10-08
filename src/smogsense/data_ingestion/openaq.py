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
from datetime import datetime
from typing import Any

import pandas as pd
from structlog import get_logger

from smogsense.config import Settings
from smogsense.data_ingestion.base import ResilientClient

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
) -> pd.DataFrame:
    page = 1
    limit = 1000
    rows: list[dict[str, Any]] = []

    date_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        params = {"datetime_from": date_from, "datetime_to": date_to, "limit": limit, "page": page}

        resp = await client.get(url, params=params)
        data = resp.json()

        if not data.get("results"):
            break

        for row in data.get("results", []):
            ts = (row.get("period") or {}).get("datetimeFrom", {}).get("utc")
            val = row.get("value")

            rows.append({"location_id": location_id, "ts_utc": ts, param_col: val})

        found = data.get("meta", {}).get("found", 0)
        if limit * page >= found:
            break
        page += 1

    return pd.DataFrame(rows)


async def fetch_hourly(
    client: ResilientClient, locations_df: pd.DataFrame, start_utc: datetime, end_utc: datetime
) -> pd.DataFrame:
    """Fetch hourly data for multiple locations and pivot to one row per station-hour."""
    tasks = []

    for _, row in locations_df.iterrows():
        loc_id = row["location_id"]
        if pd.notna(row["sensor_id_pm25"]):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client, loc_id, int(row["sensor_id_pm25"]), "pm25_ugm3", start_utc, end_utc
                )
            )
        if pd.notna(row["sensor_id_rh"]):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client, loc_id, int(row["sensor_id_rh"]), "rh_pct", start_utc, end_utc
                )
            )
        if pd.notna(row["sensor_id_temp"]):
            tasks.append(
                _fetch_single_sensor_hourly(
                    client, loc_id, int(row["sensor_id_temp"]), "temperature_c", start_utc, end_utc
                )
            )

    if not tasks:
        return pd.DataFrame(
            columns=["location_id", "ts_utc", "pm25_ugm3", "rh_pct", "temperature_c"]
        )

    dfs = await asyncio.gather(*tasks)
    valid_dfs = [df for df in dfs if not df.empty]

    if not valid_dfs:
        return pd.DataFrame(
            columns=["location_id", "ts_utc", "pm25_ugm3", "rh_pct", "temperature_c"]
        )

    # Combine all long-form rows
    long_df = pd.concat(valid_dfs, ignore_index=True)
    long_df["ts_utc"] = pd.to_datetime(long_df["ts_utc"], utc=True)

    # Pivot into the required format: one row per location_id, ts_utc
    # Grouping by location_id and ts_utc, taking the first non-null value per column
    pivoted = long_df.groupby(["location_id", "ts_utc"], as_index=False).first()

    # Ensure all target columns exist even if no data was returned for one parameter
    for col in ["pm25_ugm3", "rh_pct", "temperature_c"]:
        if col not in pivoted.columns:
            pivoted[col] = None

    return pivoted
