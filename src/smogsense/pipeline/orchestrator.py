"""smogsense.pipeline.orchestrator - Daily pipeline orchestrator.

Integrates data ingestion, preprocessing, baselines, and publishing.
Specification: docs/system-architecture.md
"""

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from typing import Any

from smogsense.inference.degradation import determine_mode
from smogsense.preprocessing.alignment import asof_cams_run
from smogsense.publishing.bulletin import generate_bulletin_json
from smogsense.publishing.site import generate_site

logger = logging.getLogger(__name__)


def generate_forecast_log(
    issuance_utc: datetime,
    cams_base_time: datetime | None,
    mode: str,
    baseline_used: str,
    stations_data: list[dict[str, Any]],
    city_forecasts: dict[int, list[float]],
) -> pd.DataFrame:
    """Generate the forecast log dataframe."""
    now_utc = datetime.now(UTC)

    rows = []

    # Process stations
    for s in stations_data:
        point_id = f"station:{s['location_id']}"
        for h in [24, 48, 72]:
            row = {
                "issuance_utc": issuance_utc,
                "generated_at_utc": now_utc,
                "domain": "lahore",
                "level": "station",
                "point_id": point_id,
                "horizon_h": h,
                "window_start_utc": issuance_utc + pd.Timedelta(hours=h - 24),
                "window_end_utc": issuance_utc + pd.Timedelta(hours=h),
                "method": baseline_used,
                "model_version": "baseline",
                "mode": mode,
                "cams_base_time_utc": cams_base_time,
                "data_cutoff_utc": issuance_utc,
                "git_sha": "unknown",
                "config_hash": "unknown",
                "calibrated": False,
                "run_id": "run_daily",
                "is_rerun": False,
                "adaptation_status": None,
                "cams_lead_offset_h": 0.0,
                "n_available_stations": len(stations_data),
            }
            # Assign quantiles from station forecast for horizon h
            qf_vals = s["forecasts"].get(h)
            if qf_vals is None:
                continue
            for q_idx, q_level in enumerate(
                [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
            ):
                row[f"q{q_level:02d}"] = qf_vals[q_idx]
            rows.append(row)

    # Process city aggregate
    for h in [24, 48, 72]:
        row = {
            "issuance_utc": issuance_utc,
            "generated_at_utc": now_utc,
            "domain": "lahore",
            "level": "city",
            "point_id": "city:lahore",
            "horizon_h": h,
            "window_start_utc": issuance_utc + pd.Timedelta(hours=h - 24),
            "window_end_utc": issuance_utc + pd.Timedelta(hours=h),
            "method": baseline_used,
            "model_version": "baseline",
            "mode": mode,
            "cams_base_time_utc": cams_base_time,
            "data_cutoff_utc": issuance_utc,
            "git_sha": "unknown",
            "config_hash": "unknown",
            "calibrated": False,
            "run_id": "run_daily",
            "is_rerun": False,
            "adaptation_status": None,
            "cams_lead_offset_h": 0.0,
            "n_available_stations": len(stations_data),
        }
        qf_vals = city_forecasts.get(h)
        if qf_vals is None:
            continue
        for q_idx, q_level in enumerate(
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        ):
            row[f"q{q_level:02d}"] = qf_vals[q_idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    # validate
    schema = pa.DataFrameSchema.from_yaml("data/schemas/forecast_log.schema.yaml")
    return schema.validate(df)


def run_daily_pipeline(issuance_utc: datetime, force: bool = False) -> int:
    """Run the complete daily forecast pipeline."""
    start_time = time.time()

    # 1. State check
    manifest_dir = Path(".state/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{issuance_utc.strftime('%Y%m%d_%H%M')}"
    manifest_file = manifest_dir / f"{run_id}.json"

    if not force and manifest_file.exists():
        try:
            with open(manifest_file) as f:
                prev_manifest = json.load(f)
            if prev_manifest.get("published") is True:
                logger.info(f"Issuance {issuance_utc} already published. Use --force to override.")
                return 11
        except Exception:
            pass

    # 2. Try fetching data (Mocked for integration test)
    # In a real run, we would call openaq.fetch_hourly and copernicus.fetch_cams.
    # For Phase 1a.6 integration, we will simulate the variables:
    has_obs = True
    has_cams = True
    cams_is_stale = False

    # Determine mode
    try:
        mode = determine_mode(
            has_observations=has_obs,
            has_cams=has_cams,
            cams_is_stale=cams_is_stale,
            promoted_model_available=False,
        )
    except RuntimeError as e:
        logger.error(f"Degradation failed: {e}")
        # Write failure manifest
        write_manifest(manifest_file, run_id, issuance_utc, False, "unknown", 20)
        return 20

    baseline_used = "m1_cams_raw" if has_cams else "m0_persistence"

    # Generate mock station data to fulfill the schema (N=1)
    # At N=0 M1 is deterministic, but here we simulate N=1 for integration completeness
    cams_base = asof_cams_run(issuance_utc)

    # Since Phase 1a has no historical residuals, we will use a dummy flat residual array
    # or just flat quantiles for the output.
    dummy_quantiles = np.linspace(40.0, 60.0, 19).tolist()

    stations_data = [
        {
            "location_id": 1,
            "forecasts": {24: dummy_quantiles, 48: dummy_quantiles, 72: dummy_quantiles},
        }
    ]
    city_forecasts = {24: dummy_quantiles, 48: dummy_quantiles, 72: dummy_quantiles}

    # 3. Forecast Log
    try:
        log_df = generate_forecast_log(
            issuance_utc=issuance_utc,
            cams_base_time=cams_base,
            mode=mode,
            baseline_used=baseline_used,
            stations_data=stations_data,
            city_forecasts=city_forecasts,
        )

        forecast_dir = Path(".state/forecasts")
        forecast_dir.mkdir(parents=True, exist_ok=True)
        log_path = forecast_dir / f"forecast_{issuance_utc.strftime('%Y%m%d_%H%M%S')}.parquet"
        log_df.to_parquet(log_path)
    except Exception as e:
        logger.error(f"Forecast log generation failed: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, mode, 50)
        return 50

    # 4. JSON Bulletin
    try:
        sources_status = {
            "openaq": {"status": "ok", "as_of_utc": issuance_utc.isoformat(timespec="seconds")},
            "cams_global": {
                "status": "ok" if has_cams else "missing",
                "as_of_utc": cams_base.isoformat(timespec="seconds"),
            },
        }

        bulletin = generate_bulletin_json(
            forecast_df=log_df,
            issuance_utc=issuance_utc,
            mode=mode,
            model=baseline_used,
            data_cutoff_utc=issuance_utc,
            domain="lahore",
            sources_status=sources_status,
        )
    except Exception as e:
        logger.error(f"Bulletin generation failed: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, mode, 30)
        return 30

    # 5. Minimal Site
    try:
        site_dir = Path("gh-pages")
        generate_site(bulletin, site_dir)

        # Also save bulletin to state
        state_site_dir = Path(".state/site_json")
        state_site_dir.mkdir(parents=True, exist_ok=True)
        with open(
            state_site_dir / f"bulletin_{issuance_utc.strftime('%Y%m%d_%H%M%S')}.json", "w"
        ) as f:
            json.dump(bulletin, f)

    except Exception as e:
        logger.error(f"Site generation failed: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, mode, 50)
        return 50

    # 6. Run Manifest
    write_manifest(manifest_file, run_id, issuance_utc, True, mode, 10 if mode != "full" else 0)

    return 10 if mode != "full" else 0


def write_manifest(
    path: Path, run_id: str, issuance_utc: datetime, published: bool, mode: str, exit_code: int
) -> None:
    manifest = {
        "run_id": run_id,
        "issued_at_utc": issuance_utc.isoformat(timespec="seconds"),
        "is_rerun": False,
        "published": published,
        "adaptation_status": None,
        "observed_latency": 0.0,
        "git_sha": "unknown",
        "config_hash": "unknown",
        "issuance": issuance_utc.isoformat(timespec="seconds"),
        "degradation_level": 2 if mode == "baseline_only" else 3,
        "exit_code": exit_code,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
