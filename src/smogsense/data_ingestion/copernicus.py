"""smogsense.data_ingestion.copernicus - CDS and ADS client wrapper.

Wraps cdsapi for two distinct data stores with separate endpoints and personal access tokens:
ADS (cams-global-atmospheric-composition-forecasts) and CDS (reanalysis-era5-single-levels).

Specification: docs/data-engineering.md -> 'Copernicus ADS: CAMS global forecasts'
"""

import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cdsapi
import requests.exceptions

from smogsense.errors import SourceUnavailable

logger = logging.getLogger(__name__)


class CamsClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self._active_requests: dict[str, str] = {}
        self._last_request_id: str | None = None
        self.config = config

    def _get_ads_client(self) -> cdsapi.Client:
        ads_conf = self.config["ads"]
        url = ads_conf["url"]
        key_env = ads_conf["key_env"]
        key = os.environ.get(key_env)
        if not key:
            raise SourceUnavailable(f"Missing {key_env} in environment")
        return cdsapi.Client(url=url, key=key, wait_until_complete=False)

    def fetch_cams(
        self,
        base_time: datetime,
        leadtime_hours: list[int],
        variables: list[str],
        area: list[float],
        dest_path: Path,
    ) -> Path:
        """
        Fetch CAMS forecast data using ADS.
        """
        # Validate base_time
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=UTC)
        else:
            base_time = base_time.astimezone(UTC)

        if base_time.minute != 0 or base_time.second != 0 or base_time.microsecond != 0:
            raise ValueError(f"Invalid CAMS base time {base_time}: must be on the hour")
        if base_time.hour not in (0, 12):
            raise ValueError(f"Invalid CAMS base time {base_time}: must be 00Z or 12Z")

        # Validate area
        if len(area) != 4:
            raise ValueError(
                f"Area must contain exactly 4 numbers [north, west, south, east], got {len(area)}"
            )
        if area[0] < area[2]:
            raise ValueError(f"Invalid area: north ({area[0]}) must be >= south ({area[2]})")

        # Validate lead times
        if not leadtime_hours:
            raise ValueError("leadtime_hours must not be empty")
        for h in leadtime_hours:
            if not isinstance(h, int) or h < 0:
                raise ValueError(f"Invalid lead time {h}: must be non-negative integer")

        # Validate variables
        if not variables:
            raise ValueError("variables must not be empty")

        ads_conf = self.config["ads"]
        dataset = ads_conf["dataset"]
        req_conf = ads_conf.get("request", {})

        # Build request according to actual API schema
        request = {
            "data_format": req_conf.get("data_format", "grib"),
            "type": req_conf.get("type", ["forecast"]),
            "date": base_time.strftime("%Y-%m-%d"),
            "time": base_time.strftime("%H:%M"),
            "leadtime_hour": [str(h) for h in sorted(set(leadtime_hours))],
            "variable": variables,
            "area": area,
        }

        # Unique key for isolating reattachment per exact request parameters
        req_key = f"{dataset}:{base_time.isoformat()}:{sorted(leadtime_hours)}:{sorted(variables)}:{area}"

        client = self._get_ads_client()

        logger.info("Submitting ADS request to %s for %s", dataset, base_time)
        try:
            import cdsapi.api

            existing_id = self._active_requests.get(req_key)
            if existing_id:
                logger.info("Reattaching to ADS request %s for %s", existing_id, req_key)
                result = cdsapi.api.Result(client, {"request_id": existing_id})
            else:
                result = client.retrieve(dataset, request)
                if result.reply and "request_id" in result.reply:
                    req_id = result.reply["request_id"]
                    self._active_requests[req_key] = req_id
                    self._last_request_id = req_id
        except Exception as e:
            self._active_requests.pop(req_key, None)
            self._last_request_id = None
            msg = str(e).lower()
            if "accept the terms" in msg or "licence" in msg or "license" in msg:
                raise PermissionError("Accept the dataset licence on the ADS website once") from e
            raise SourceUnavailable(f"ADS retrieve failed: {e}") from e

        queue_conf = ads_conf.get("queue", {})
        poll_interval = queue_conf.get("poll_interval_s", 30)
        max_wall_minutes = queue_conf.get("max_wall_minutes_operational", 20)
        max_wall_seconds = max_wall_minutes * 60

        start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_time
            remaining = max_wall_seconds - elapsed
            if remaining <= 0:
                self._active_requests.pop(req_key, None)
                self._last_request_id = None
                raise SourceUnavailable(f"ADS queue timeout: exceeded {max_wall_minutes} minutes")

            try:
                result.update()
            except requests.exceptions.RequestException as e:
                # transient network error
                logger.warning("Transient error updating ADS result status: %s", e)
                time.sleep(min(poll_interval, remaining))
                continue
            except Exception as e:
                # terminal API error from cdsapi itself (not a network level request error)
                if "unknown api state" in str(e).lower():
                    self._active_requests.pop(req_key, None)
                    self._last_request_id = None
                    raise SourceUnavailable(f"ADS request failed: terminal error: {e}") from e
                logger.warning("Error updating ADS result status: %s", e)
                time.sleep(min(poll_interval, remaining))
                continue

            state = result.reply.get("state")
            if state == "completed":
                break
            elif state in ("failed", "deleted"):
                self._active_requests.pop(req_key, None)
                self._last_request_id = None
                error_msg = result.reply.get("error", {}).get("message", "Unknown error")
                raise SourceUnavailable(f"ADS request failed: {state} - {error_msg}")

            time.sleep(min(poll_interval, remaining))

        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic landing with unique temporary file
        fd, temp_path_str = tempfile.mkstemp(
            dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp.grib"
        )
        os.close(fd)
        temp_path = Path(temp_path_str)

        try:
            result.download(str(temp_path))
            temp_path.replace(dest_path)
        except Exception as e:
            raise SourceUnavailable(f"Failed to download GRIB: {e}") from e
        finally:
            self._active_requests.pop(req_key, None)
            self._last_request_id = None
            if temp_path.exists():
                temp_path.unlink()

        return dest_path
