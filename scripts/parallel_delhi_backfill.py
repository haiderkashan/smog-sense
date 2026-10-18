"""scripts/parallel_delhi_backfill.py - Multi-process feature store backfill for Delhi.

Executes feature extraction for Delhi across all 8 smog seasons (2018-2026) using
a ProcessPoolExecutor to saturate multi-core runner hardware.
Strictly enforces ADR-024:
- Block E: 6 seasons (2018-2024) across 3 folds:
    2018-2019: fold 0
    2019-2020: fold 0
    2020-2021: fold 1
    2021-2022: fold 1
    2022-2023: fold 2
    2023-2024: fold 2
- Block V: 2024-2025 (Validation)
- Block T: 2025-2026 (Test)
Computes unified deterministic OOF partition hash and saves partitioned Parquet files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smogsense.features.backfill import (
    DEFAULT_FEATURE_SET_VERSION,
    compute_config_hash,
    compute_oof_partition_hash,
    get_season_date_range,
    is_boundary_purged,
    load_backfilled_observations,
)
from smogsense.features.store import (
    _load_feature_store_schema,
    build_feature_table,
    save_feature_table,
)
from smogsense.utils.io import validate_frame

SEASON_CONFIGS = [
    {"season": "2018-2019", "split": "E", "fold_id": 0, "start": "2018-10-01", "end": "2019-02-28"},
    {"season": "2019-2020", "split": "E", "fold_id": 0, "start": "2019-10-01", "end": "2020-02-29"},
    {"season": "2020-2021", "split": "E", "fold_id": 1, "start": "2020-10-01", "end": "2021-02-28"},
    {"season": "2021-2022", "split": "E", "fold_id": 1, "start": "2021-10-01", "end": "2022-02-28"},
    {"season": "2022-2023", "split": "E", "fold_id": 2, "start": "2022-10-01", "end": "2023-02-28"},
    {"season": "2023-2024", "split": "E", "fold_id": 2, "start": "2023-10-01", "end": "2024-02-29"},
    {
        "season": "2024-2025",
        "split": "V",
        "fold_id": None,
        "start": "2024-10-01",
        "end": "2025-02-28",
    },
    {
        "season": "2025-2026",
        "split": "T",
        "fold_id": None,
        "start": "2025-10-01",
        "end": "2026-02-28",
    },
]


def _extract_season_worker(
    cfg: dict[str, Any], backfill_dir: str
) -> tuple[str, list[pd.DataFrame], int, int]:
    """Worker function executed in separate process to extract features for one season."""
    season = cfg["season"]
    split = cfg["split"]
    fold_id = cfg["fold_id"]
    s_date = datetime.strptime(cfg["start"], "%Y-%m-%d").date()
    e_date = datetime.strptime(cfg["end"], "%Y-%m-%d").date()

    print(f"[{season}] Worker started (Split: {split}, Fold: {fold_id})...")

    domain = "delhi"
    c_hash = compute_config_hash()

    # Load real observations for domain
    obs_df = load_backfilled_observations(domain, backfill_dir=backfill_dir)

    # Load registry
    reg_file = Path("data/registry") / f"registry_{domain}.parquet"
    registry_df = pd.read_parquet(reg_file) if reg_file.exists() else None

    # Synthesize CAMS & FIRMS
    from smogsense.features.backfill import _synthesize_historical_cams, _synthesize_historical_fire

    stations = (
        sorted([int(x) for x in registry_df["location_id"].unique()])
        if registry_df is not None
        else [17]
    )

    start_utc = datetime(s_date.year, s_date.month, s_date.day, 0, 0, 0, tzinfo=UTC) - timedelta(
        hours=96
    )
    end_utc = datetime(e_date.year, e_date.month, e_date.day, 23, 59, 59, tzinfo=UTC) + timedelta(
        hours=96
    )

    cams_df = _synthesize_historical_cams(domain, stations, start_utc, end_utc)
    fire_df = _synthesize_historical_fire(domain, start_utc, end_utc)

    # Pre-filter to season bounds
    s_min_ts = datetime(s_date.year, s_date.month, s_date.day, 0, 0, 0, tzinfo=UTC) - timedelta(
        days=9
    )
    s_max_ts = datetime(e_date.year, e_date.month, e_date.day, 23, 59, 59, tzinfo=UTC) + timedelta(
        days=9
    )

    season_obs = (
        obs_df[(obs_df["ts_utc"] >= s_min_ts) & (obs_df["ts_utc"] <= s_max_ts)]
        .sort_values(by="ts_utc")
        .reset_index(drop=True)
    )
    obs_ts_vals = season_obs["ts_utc"].to_numpy()

    season_cams = (
        cams_df[(cams_df["ts_utc"] >= s_min_ts) & (cams_df["ts_utc"] <= s_max_ts)]
        .sort_values(by="ts_utc")
        .reset_index(drop=True)
    )
    cams_ts_vals = season_cams["ts_utc"].to_numpy()

    season_fire = fire_df[
        (fire_df["ts_utc"] >= s_min_ts) & (fire_df["ts_utc"] <= s_max_ts)
    ].reset_index(drop=True)

    season_start, season_end = get_season_date_range(season)

    lattice_hours = (0, 6, 12, 18)
    embargo_hours = 72.0

    season_dfs: list[pd.DataFrame] = []
    purged_count = 0
    total_count = 0

    c_cur = s_date
    while c_cur <= e_date:
        for hour in lattice_hours:
            issuance_utc = datetime(c_cur.year, c_cur.month, c_cur.day, hour, 0, 0, tzinfo=UTC)

            if is_boundary_purged(issuance_utc, season_start, season_end, embargo_hours):
                purged_count += len(stations)
                continue

            t_lookback_start = issuance_utc - timedelta(days=8)
            t_lookback_np = pd.Timestamp(t_lookback_start).to_numpy()
            t_iss_np = pd.Timestamp(issuance_utc).to_numpy()

            i_obs_start = int(np.searchsorted(obs_ts_vals, t_lookback_np))
            i_obs_end = int(np.searchsorted(obs_ts_vals, t_iss_np))
            obs_win = season_obs.iloc[i_obs_start:i_obs_end]
            obs_slice = obs_win[obs_win["available_at_utc"] <= issuance_utc]

            i_cams_start = int(np.searchsorted(cams_ts_vals, t_lookback_np))
            i_cams_end = int(np.searchsorted(cams_ts_vals, t_iss_np))
            cams_win = season_cams.iloc[i_cams_start:i_cams_end]
            cams_slice = cams_win[cams_win["available_at_utc"] <= issuance_utc]

            fire_slice = season_fire[
                (season_fire["available_at_utc"] <= issuance_utc)
                & (season_fire["ts_utc"] >= t_lookback_start)
                & (season_fire["ts_utc"] < issuance_utc)
            ]

            t_fut_end_np = pd.Timestamp(issuance_utc + timedelta(hours=72)).to_numpy()
            i_fut_end = int(np.searchsorted(obs_ts_vals, t_fut_end_np))
            fut_slice = season_obs.iloc[i_obs_end:i_fut_end]

            tbl = build_feature_table(
                domain=domain,
                issuance_utc=issuance_utc,
                obs_df=obs_slice,
                cams_df=cams_slice,
                fire_exposure_df=fire_slice,
                registry_df=registry_df,
                future_obs_df=fut_slice,
                feature_set_version=DEFAULT_FEATURE_SET_VERSION,
                config_hash=c_hash,
                source_split=split,
                fold_id=fold_id,
                verify_schema=False,
            )
            tbl["season"] = season
            season_dfs.append(tbl)
            total_count += len(tbl)

        c_cur += timedelta(days=1)

    print(f"[{season}] Worker finished: {total_count:,} samples extracted ({purged_count} purged).")
    return season, season_dfs, total_count, purged_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel multi-season backfill for Delhi.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent worker processes"
    )
    parser.add_argument(
        "--backfill-dir", default="data/processed/backfill", help="Path to raw backfill partitions"
    )
    parser.add_argument(
        "--out-dir", default="data/processed/feature_store", help="Path to feature store root"
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING PARALLEL MULTI-SEASON FEATURE STORE BACKFILL (DELHI)")
    print(f"Workers: {args.workers} | Seasons: {len(SEASON_CONFIGS)}")
    print("=" * 80)

    all_dfs: list[pd.DataFrame] = []
    total_samples = 0
    total_purged = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_extract_season_worker, cfg, args.backfill_dir): cfg["season"]
            for cfg in SEASON_CONFIGS
        }
        for future in as_completed(futures):
            s_name = futures[future]
            try:
                _season, s_dfs, s_count, s_purged = future.result()
                all_dfs.extend(s_dfs)
                total_samples += s_count
                total_purged += s_purged
            except Exception as exc:
                print(f"ERROR in season {s_name}: {exc}")
                raise

    print(
        f"\nAll workers complete. Total extracted: {total_samples:,} samples ({total_purged} purged)."
    )

    # Combine into single DataFrame to compute global OOF hash & validate schema
    combined_df = pd.concat(all_dfs, ignore_index=True)
    oof_hash = compute_oof_partition_hash(combined_df)
    print(f"Computed unified OOF partition hash: {oof_hash}")

    # Schema verification
    print("Verifying schema against feature_store.schema.yaml...")
    schema = _load_feature_store_schema()
    validate_frame(combined_df.drop(columns=["season"], errors="ignore"), schema)
    print(f"Schema verification passed on all {len(combined_df):,} rows.")

    # Save partitioned Parquet files
    print(f"Writing daily partitioned Parquet files to {out_path} ...")
    written_count = 0
    for raw_season, s_group in combined_df.groupby("season"):
        s_str = str(raw_season)
        for _iss_dt, d_group in s_group.groupby(s_group["issuance_utc"].dt.date):
            save_df = d_group.drop(columns=["season"]).copy()
            save_feature_table(
                df=save_df,
                base_dir=out_path,
                season=s_str,
                oof_partition_hash=oof_hash,
            )
            written_count += 1

    print(f"Successfully wrote {written_count} daily partitions for Delhi.")
    print("=" * 80)


if __name__ == "__main__":
    main()
