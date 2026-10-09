"""smogsense.pipeline.seed_history - Generate empirical residual quantiles (Phase 1a.7).

Specification: docs/system-architecture.md
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smogsense.config import Settings
from smogsense.data_ingestion.base import (
    CircuitBreaker,
    RateBudget,
    ResilientClient,
    create_client,
)
from smogsense.data_ingestion.copernicus import CamsClient
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations
from smogsense.preprocessing.alignment import asof_cams_run
from smogsense.preprocessing.gridded import extract_stations
from smogsense.preprocessing.qc import apply_qc

logger = logging.getLogger(__name__)


async def determine_cutoff_dates(
    start_utc: datetime, end_utc: datetime, settings: dict[str, Any]
) -> tuple[datetime, pd.DataFrame, pd.DataFrame]:
    """Fetch OpenAQ and determine the resolved historical period.

    Cutoff Rule (Point-in-Time Correctness & Common Data Support):
    An issuance at reference time T requires observations at T+24h, T+48h, and T+72h.
    Therefore, the maximum usable historical issuance date is bounded by:
    T_issuance_max <= t_obs_max - 72h.
    Furthermore, T_issuance_max cannot exceed the execution cutoff (end_utc / now_utc),
    and the corresponding CAMS as-of cycle B*(T) must be knowable at execution time:
    B*(T) + 10h <= end_utc.
    """
    raw_key = settings.get("openaq_api_key")
    api_key: str | None = None
    if raw_key is not None:
        api_key = (
            str(raw_key.get_secret_value())
            if hasattr(raw_key, "get_secret_value")
            else str(raw_key)
        )
    ac = create_client(api_key=api_key)
    rb = RateBudget()
    cb = CircuitBreaker()
    client = ResilientClient(ac, rb, cb)

    stations_df = await list_locations(client, "lahore")
    if stations_df.empty:
        await ac.aclose()
        raise ValueError("No stations returned from registry")

    logger.info(f"Fetching OpenAQ history from {start_utc} to {end_utc}")
    obs_raw = await fetch_hourly(client, stations_df, start_utc, end_utc)
    await ac.aclose()

    if obs_raw.empty:
        raise ValueError("No historical OpenAQ data retrieved")

    # Use production QC rules and filter strictly to valid non-imputed observations (qc_flags == 0)
    qc_df = apply_qc(obs_raw)
    valid_obs = qc_df[(qc_df["qc_flags"] == 0) & (qc_df["pm25_ugm3"].notna())].copy()
    if valid_obs.empty:
        raise ValueError("No valid observations after QC")

    t_obs_max = valid_obs["ts_utc"].max()
    t_obs_max = t_obs_max.to_pydatetime() if hasattr(t_obs_max, "to_pydatetime") else t_obs_max
    if t_obs_max.tzinfo is None:
        t_obs_max = t_obs_max.replace(tzinfo=UTC)

    # 1. Guard against future leakage if observations have timestamps beyond execution time
    t_obs_effective = min(end_utc, t_obs_max)

    logger.info(f"Latest valid OpenAQ observation: {t_obs_effective}")

    # 2. An issuance T requires +72h observation for horizon 72
    t_issuance_max = min(end_utc, t_obs_effective - timedelta(hours=72))
    # Align to 00:00 UTC
    t_issuance_max = t_issuance_max.replace(hour=0, minute=0, second=0, microsecond=0)

    # 3. CAMS availability check: issuance T requires CAMS run B*(T) knowable as of end_utc
    # Availability rule from docs/system-architecture.md §4: B*(T) + 10h <= end_utc
    while (
        t_issuance_max >= start_utc
        and (asof_cams_run(t_issuance_max) + timedelta(hours=10)) > end_utc
    ):
        t_issuance_max -= timedelta(days=1)

    if t_issuance_max < start_utc:
        raise ValueError(
            f"Resolved historical issuance end date {t_issuance_max} is earlier than start date {start_utc}. "
            "Insufficient observations or CAMS history to evaluate 72h horizons."
        )

    logger.info(f"Resolved historical issuance end date: {t_issuance_max}")
    return t_issuance_max, valid_obs, stations_df


def generate_seed_history(
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    output_path: Path | None = None,
    min_city_stations: int = 3,
    min_processed_days: int = 1,
    test_start_utc: datetime | None = None,
    test_now_utc: datetime | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    settings = Settings.load("configs")

    start_dt = test_start_utc or start_utc or datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_dt = test_now_utc or end_utc or datetime.now(UTC)
    out_file = output_path or Path(".state/artifacts/seed_history.json")

    end_issuance, obs_df, stations_df = asyncio.run(
        determine_cutoff_dates(start_dt, now_dt, settings.model_dump())
    )

    cams_client = CamsClient(settings.model_dump())

    residuals: list[dict[str, Any]] = []

    coverage: dict[str, list[str]] = {"expected": [], "processed": [], "missing": [], "failed": []}

    # Loop over issuance times daily
    current_issuance = start_dt
    while current_issuance <= end_issuance:
        cams_base = asof_cams_run(current_issuance)
        coverage["expected"].append(current_issuance.isoformat())

        cams_path = Path("data/raw/cams") / f"cams_hist_{cams_base.strftime('%Y%m%d_%H')}.grib"

        # We need targets at T+24, T+48, T+72
        targets = [current_issuance + timedelta(hours=h) for h in [24, 48, 72]]
        cams_leadtimes = [int((t - cams_base).total_seconds() / 3600.0) for t in targets]

        try:
            if not cams_path.exists():
                cams_path.parent.mkdir(parents=True, exist_ok=True)
                cams_client.fetch_cams(
                    base_time=cams_base,
                    leadtime_hours=cams_leadtimes,
                    variables=["particulate_matter_2.5um"],
                    area=[
                        settings.domains["domains"]["lahore"]["station_bbox"]["north"],
                        settings.domains["domains"]["lahore"]["station_bbox"]["west"],
                        settings.domains["domains"]["lahore"]["station_bbox"]["south"],
                        settings.domains["domains"]["lahore"]["station_bbox"]["east"],
                    ],
                    dest_path=cams_path,
                )

            cams_df = extract_stations(
                cams_path, stations_df, domain="lahore", settings=settings.model_dump()
            )

            if cams_df.empty:
                logger.warning(f"No CAMS data extracted for base {cams_base}")
                coverage["missing"].append(current_issuance.isoformat())
                current_issuance += timedelta(days=1)
                continue

            cams_df["location_id"] = cams_df["location_id"].astype(str)

            day_processed = False
            for h in [24, 48, 72]:
                target = current_issuance + timedelta(hours=h)
                # Match observation
                obs_target = obs_df[obs_df["ts_utc"] == target]
                if obs_target.empty:
                    continue

                # Match CAMS
                if "target_hour_utc" in cams_df.columns:
                    cams_target = cams_df[cams_df["target_hour_utc"] == target]
                else:
                    cams_target = pd.DataFrame()

                if cams_target.empty:
                    continue

                for loc_id in stations_df["location_id"].unique():
                    o_match = obs_target[obs_target["location_id"].astype(str) == str(loc_id)]
                    c_match = cams_target[cams_target["location_id"].astype(str) == str(loc_id)]

                    if not o_match.empty and not c_match.empty:
                        o_val = o_match.iloc[0]["pm25_ugm3"]
                        c_val = c_match.iloc[0]["pm25_ugm3"]
                        if pd.notna(o_val) and pd.notna(c_val):
                            res = float(o_val) - float(c_val)
                            residuals.append(
                                {
                                    "location_id": str(loc_id),
                                    "horizon": str(h),
                                    "target_hour_utc": target,
                                    "residual": res,
                                }
                            )
                            day_processed = True

                # Compute centroid residual
                # Specification: docs/ml-architecture.md §1 & PRD.md FR-38:
                # "The city value is a pseudo-location with target = mean of group-collapsed
                # station block means over a panel frozen at season freeze, requiring >= 3 valid stations."
                valid_station_obs = obs_target[
                    obs_target["location_id"].isin(stations_df["location_id"])
                ]
                valid_o_vals = valid_station_obs["pm25_ugm3"].dropna()
                c_match_cent = cams_target[cams_target["location_id"].astype(str) == "centroid"]
                if len(valid_o_vals) >= min_city_stations and not c_match_cent.empty:
                    o_val_cent = float(valid_o_vals.mean())
                    c_val_cent = float(c_match_cent.iloc[0]["pm25_ugm3"])
                    if pd.notna(o_val_cent) and pd.notna(c_val_cent):
                        res_cent = o_val_cent - c_val_cent
                        residuals.append(
                            {
                                "location_id": "centroid",
                                "horizon": str(h),
                                "target_hour_utc": target,
                                "residual": res_cent,
                            }
                        )
                        day_processed = True

            if day_processed:
                coverage["processed"].append(current_issuance.isoformat())
            else:
                coverage["missing"].append(current_issuance.isoformat())

        except Exception as e:
            logger.warning(f"Error processing issuance {current_issuance}: {e}")
            coverage["failed"].append(current_issuance.isoformat())

        current_issuance += timedelta(days=1)

    if len(coverage["failed"]) > 0:
        logger.error(
            f"Seed history generation failed: {len(coverage['failed'])} days failed due to infrastructure/pipeline errors."
        )
        return 1

    if len(coverage["processed"]) < min_processed_days:
        logger.error(
            f"Seed history generation failed: only {len(coverage['processed'])} days processed, "
            f"minimum required is {min_processed_days}."
        )
        return 1

    res_df = pd.DataFrame(residuals)
    if res_df.empty:
        logger.error("Seed history generation failed: no valid residuals computed.")
        return 1

    actual_start_issuance = min(coverage["processed"])
    actual_end_issuance = max(coverage["processed"])
    actual_end_target = (
        datetime.fromisoformat(actual_end_issuance) + timedelta(hours=72)
    ).isoformat()

    artifact: dict[str, Any] = {
        "metadata": {
            "requested_start": start_dt.isoformat(),
            "requested_end": now_dt.isoformat(),
            "resolved_start_issuance": actual_start_issuance,
            "resolved_end_issuance": actual_end_issuance,
            "resolved_end_target": actual_end_target,
            "generated_at": now_dt.isoformat(),
            "horizons": [24, 48, 72],
            "quantile_levels": [
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
            ],
            "units": "ug/m3",
            "min_city_stations": min_city_stations,
            "coverage": coverage,
            "total_residuals": len(res_df),
        },
        "quantiles": {},
        "counts": {},
    }

    if not res_df.empty:
        qs = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]
        for key, group in res_df.groupby(["location_id", "horizon"]):
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            loc_h, horiz_val = key
            loc_str_val = str(loc_h)
            if loc_str_val.lower() == "nan" or loc_h is None:
                continue
            try:
                loc_str = str(int(float(loc_str_val)))
            except (ValueError, TypeError):
                loc_str = loc_str_val

            if loc_str not in artifact["quantiles"]:
                artifact["quantiles"][loc_str] = {}
                artifact["counts"][loc_str] = {}
            if len(group) > 0:
                h_str = str(int(float(str(horiz_val))))
                vals = np.quantile(group["residual"], qs, method="linear").tolist()
                # Strict monotonicity check without silent sorting repair:
                if not all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
                    raise ValueError(
                        f"Non-monotonic quantiles computed for {loc_str} horizon {h_str}: {vals}"
                    )
                artifact["quantiles"][loc_str][h_str] = vals
                artifact["counts"][loc_str][h_str] = len(group)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Seed history saved to {out_file} with {len(res_df)} total residuals.")
    return 0


if __name__ == "__main__":
    generate_seed_history()
