"""smogsense.pipeline.seed_history - Generate empirical residual quantiles (Phase 1a.7).

Specification: docs/system-architecture.md
"""
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smogsense.config import Settings
from smogsense.data_ingestion.copernicus import CamsClient
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations
from smogsense.preprocessing.alignment import asof_cams_run
from smogsense.preprocessing.gridded import extract_stations
from smogsense.preprocessing.qc import clean_observations

logger = logging.getLogger(__name__)

import asyncio
from smogsense.data_ingestion.base import CircuitBreaker, RateBudget, ResilientClient
import httpx

async def determine_cutoff_dates(
    start_utc: datetime,
    end_utc: datetime,
    settings: dict[str, Any]
) -> tuple[datetime, pd.DataFrame, pd.DataFrame]:
    """Fetch OpenAQ and determine the resolved historical period."""
    ac = httpx.AsyncClient(timeout=30.0)
    rb = RateBudget(calls=60, period=60.0)
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

    obs_clean = clean_observations(obs_raw)
    valid_obs = obs_clean[obs_clean["pm25_ugm3"].notna()]
    if valid_obs.empty:
        raise ValueError("No valid observations after QC")

    t_obs_max = valid_obs["ts_utc"].max()
    t_obs_max = t_obs_max.to_pydatetime() if hasattr(t_obs_max, "to_pydatetime") else t_obs_max
    
    logger.info(f"Latest valid OpenAQ observation: {t_obs_max}")
    
    # The latest issuance time we can evaluate targets for (requires +72h observation)
    t_issuance_max = t_obs_max - timedelta(hours=72)
    # Align to 00:00
    t_issuance_max = t_issuance_max.replace(hour=0, minute=0, second=0, microsecond=0)
    
    logger.info(f"Resolved historical issuance end date: {t_issuance_max}")
    return t_issuance_max, obs_clean, stations_df


def generate_seed_history() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.load("configs")
    
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = datetime.now(UTC)

    try:
        end_issuance, obs_df, stations_df = asyncio.run(determine_cutoff_dates(start_utc, now_utc, settings.model_dump()))
    except Exception as e:
        logger.error(f"Failed to resolve cutoff: {e}")
        return 1

    cams_client = CamsClient(settings.model_dump())
    
    residuals: list[dict[str, Any]] = []

    # Loop over issuance times daily
    current_issuance = start_utc
    while current_issuance <= end_issuance:
        cams_base = asof_cams_run(current_issuance)
        
        cams_path = Path(".state/inputs") / f"cams_hist_{cams_base.strftime('%Y%m%d_%H')}.grib"
        
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
                current_issuance += timedelta(days=1)
                continue

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
                    o_match = obs_target[obs_target["location_id"] == loc_id]
                    c_match = cams_target[cams_target["location_id"] == loc_id]
                    
                    if not o_match.empty and not c_match.empty:
                        o_val = o_match.iloc[0]["pm25_ugm3"]
                        c_val = c_match.iloc[0]["pm25_ugm3"]
                        if pd.notna(o_val) and pd.notna(c_val):
                            res = float(o_val) - float(c_val)
                            residuals.append({
                                "location_id": str(loc_id),
                                "horizon": str(h),
                                "target_hour_utc": target,
                                "residual": res
                            })
                            
        except Exception as e:
            logger.warning(f"Error processing issuance {current_issuance}: {e}")

        current_issuance += timedelta(days=1)

    res_df = pd.DataFrame(residuals)
    
    artifact = {
        "metadata": {
            "requested_start": start_utc.isoformat(),
            "requested_end": now_utc.isoformat(),
            "resolved_start": start_utc.isoformat(),
            "resolved_end_issuance": end_issuance.isoformat(),
            "resolved_end_target": (end_issuance + timedelta(hours=72)).isoformat(),
            "generated_at": now_utc.isoformat(),
            "horizons": [24, 48, 72],
            "quantile_levels": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            "units": "ug/m3"
        },
        "quantiles": {},
        "counts": {}
    }

    if not res_df.empty:
        qs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for (loc, h), group in res_df.groupby(["location_id", "horizon"]):
            if pd.isna(loc): continue
            try:
                loc_str = str(int(float(loc)))
            except (ValueError, TypeError):
                loc_str = str(loc)
                
            if loc_str not in artifact["quantiles"]:
                artifact["quantiles"][loc_str] = {}
                artifact["counts"][loc_str] = {}
            if len(group) > 0:
                vals = np.quantile(group["residual"], qs, method="linear").tolist()
                # Ensure monotonicity just in case
                vals = sorted(vals)
                artifact["quantiles"][loc_str][str(int(h))] = vals
                artifact["counts"][loc_str][str(int(h))] = len(group)

    out_path = Path(".state/artifacts/seed_history.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Seed history saved to {out_path} with {len(res_df)} total residuals.")
    return 0

if __name__ == "__main__":
    generate_seed_history()
