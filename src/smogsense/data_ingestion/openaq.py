"""smogsense.data_ingestion.openaq -- OpenAQ v3 REST client (near-real-time observations).

Discovers PM2.5 sensors inside a domain bounding box, paginates /v3/locations and
/v3/sensors/{id}/hours, and returns hourly records plus coverage metadata. Free tier is 60
requests/minute and 2,000/hour per API key.

Public contract (implemented in Phase 1a):
- list_locations(client, domain) -> DataFrame (registry candidates).
- fetch_hourly(client, sensor_ids, start_utc, end_utc) -> DataFrame (raw, un-QC'd).
- Respects pagination via limit/page and stops on found/limit arithmetic, never on empty-page
  guessing.
"""

import asyncio
from datetime import datetime
from typing import Any

import pandas as pd
from structlog import get_logger

from smogsense.config import Settings
from smogsense.data_ingestion.base import ResilientClient

logger = get_logger(__name__)


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

    while True:
        url = "https://api.openaq.org/v3/locations"
        params = {
            "bbox": bbox_str,
            "parameters_id": 2,  # PM2.5
            "limit": limit,
            "page": page
        }

        resp = await client.get(url, params=params)
        data = resp.json()

        for loc in data.get("results", []):
            loc_id = loc.get("id")
            name = loc.get("name")
            provider = loc.get("provider", {}).get("name")
            coords = loc.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            is_monitor = loc.get("isMonitor", False)
            first_dt = loc.get("datetimeFirst", {}).get("utc")
            last_dt = loc.get("datetimeLast", {}).get("utc")

            sensor_pm25 = None
            sensor_rh = None
            sensor_temp = None
            pm25_uptime = 0.0

            for sensor in loc.get("sensors", []):
                param_name = sensor.get("parameter", {}).get("name", "").lower()
                s_id = sensor.get("id")
                if param_name == "pm25":
                    sensor_pm25 = s_id
                    pm25_uptime = float(sensor.get("coverage", {}).get("percentComplete", 0.0)) / 100.0
                elif param_name in ("relativehumidity", "rh"):
                    sensor_rh = s_id
                elif param_name in ("temperature", "temp"):
                    sensor_temp = s_id

            locations.append({
                "location_id": loc_id,
                "name": name,
                "provider": provider,
                "lat": lat,
                "lon": lon,
                "is_monitor": is_monitor,
                "sensor_id_pm25": sensor_pm25,
                "sensor_id_rh": sensor_rh,
                "sensor_id_temp": sensor_temp,
                "first_datetime": first_dt,
                "last_datetime": last_dt,
                "pm25_uptime_90d": pm25_uptime
            })

        found = data.get("meta", {}).get("found", 0)
        if limit * page >= found:
            break
        page += 1

    df = pd.DataFrame(locations)
    if not df.empty:
        df["first_datetime"] = pd.to_datetime(df["first_datetime"])
        df["last_datetime"] = pd.to_datetime(df["last_datetime"])
    else:
        df = pd.DataFrame(columns=[
            "location_id", "name", "provider", "lat", "lon", "is_monitor",
            "sensor_id_pm25", "sensor_id_rh", "sensor_id_temp",
            "first_datetime", "last_datetime", "pm25_uptime_90d"
        ])
    return df


async def _fetch_single_sensor_hourly(client: ResilientClient, sensor_id: int, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    page = 1
    limit = 1000
    rows: list[dict[str, Any]] = []

    date_from = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        params = {
            "datetime_from": date_from,
            "datetime_to": date_to,
            "limit": limit,
            "page": page
        }
        resp = await client.get(url, params=params)
        data = resp.json()

        for row in data.get("results", []):
            ts = row.get("datetime", {}).get("utc")
            val = row.get("value")
            param = row.get("parameter", {}).get("name", "").lower()

            pm25 = val if param == "pm25" else None
            rh = val if param in ("relativehumidity", "rh") else None
            temp = val if param in ("temperature", "temp") else None

            rows.append({
                "sensor_id": sensor_id,
                "ts_utc": ts,
                "pm25_ugm3": pm25,
                "rh_pct": rh,
                "temperature_c": temp
            })

        found = data.get("meta", {}).get("found", 0)
        if limit * page >= found:
            break
        page += 1

    return pd.DataFrame(rows)


async def fetch_hourly(client: ResilientClient, sensor_ids: list[int], start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Fetch hourly data for multiple sensors concurrently."""
    tasks = [
        _fetch_single_sensor_hourly(client, s_id, start_utc, end_utc)
        for s_id in sensor_ids
    ]
    dfs = await asyncio.gather(*tasks)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["sensor_id", "ts_utc", "pm25_ugm3", "rh_pct", "temperature_c"])

    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    return df
