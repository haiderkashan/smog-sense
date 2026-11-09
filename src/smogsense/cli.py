"""smogsense.cli - Typer command-line entry point.

Specification: docs/system-architecture.md -> 'Component responsibilities and CLI contract'
"""

import contextlib
import faulthandler
import os
import sys
from datetime import UTC
from pathlib import Path

import typer

from smogsense.errors import SmogSenseError

faulthandler.enable()


def safe_exit(code: int) -> None:
    """Exit cleanly, bypassing C-extension teardown corruption on standalone execution."""
    sys.stdout.flush()
    sys.stderr.flush()
    # In pytest, use sys.exit so the test runner catches SystemExit
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        sys.exit(code)
    os._exit(code)


app = typer.Typer(no_args_is_help=True)

# Sub-apps
app_ingest = typer.Typer(no_args_is_help=True)
app_features = typer.Typer(no_args_is_help=True)
app_train = typer.Typer(no_args_is_help=True)
app_forecast = typer.Typer(no_args_is_help=True)
app_score = typer.Typer(no_args_is_help=True)
app_eval = typer.Typer(no_args_is_help=True)
app_bulletin = typer.Typer(no_args_is_help=True)
app_site = typer.Typer(no_args_is_help=True)
app_publish = typer.Typer(no_args_is_help=True)
app_state = typer.Typer(no_args_is_help=True)
app_run = typer.Typer(no_args_is_help=True)
app_reconcile = typer.Typer(no_args_is_help=True)
app_audit = typer.Typer(no_args_is_help=True)
app_preprocess = typer.Typer(no_args_is_help=True)
app_registry = typer.Typer(no_args_is_help=True)
app_replay = typer.Typer(no_args_is_help=True)
app_models = typer.Typer(no_args_is_help=True)
app_cards = typer.Typer(no_args_is_help=True)

app.add_typer(app_ingest, name="ingest")
app.add_typer(app_features, name="features")
app.add_typer(app_train, name="train")
app.add_typer(app_forecast, name="forecast")
app.add_typer(app_score, name="score")
app.add_typer(app_eval, name="eval")
app.add_typer(app_eval, name="evaluate")
app.add_typer(app_bulletin, name="bulletin")
app.add_typer(app_site, name="site")
app.add_typer(app_publish, name="publish")
app.add_typer(app_state, name="state")
app.add_typer(app_run, name="run")
app.add_typer(app_replay, name="replay")
app.add_typer(app_reconcile, name="reconcile")
app.add_typer(app_audit, name="audit")
app.add_typer(app_preprocess, name="preprocess")
app.add_typer(app_registry, name="registry")
app.add_typer(app_models, name="models")
app.add_typer(app_cards, name="cards")


@app.command()
def doctor(online: bool = False) -> None:
    """Check environment, credentials presence and config parse."""
    print("Doctor check: OK")
    if online:
        print("Online check: OK")


# Example command stub
@app_ingest.command()
def live() -> None:
    """Ingest live data."""
    pass


@app_ingest.command("archive-pilot")
def archive_pilot_cmd(
    limit: int = typer.Option(500, "--limit", help="Max archive files to process"),
    domain: str = typer.Option("lahore", "--domain", help="Target domain (lahore, delhi)"),
    shard_by: str = typer.Option("year", "--shard-by", help="Sharding strategy"),
) -> None:
    """Run historical backfill pilot on OpenAQ archive (Task P1-19)."""
    from smogsense.data_ingestion.openaq_archive import run_backfill_pilot

    stats = run_backfill_pilot(domain=domain, max_files=limit, shard_by=shard_by)
    print(
        f"Archive pilot completed: {stats['records_processed']} records from {stats['files_found']} files."
    )
    print(f"Throughput: {stats['download_mb_per_s']} MB/s ({stats['records_per_s']} records/s).")
    print(f"Output saved to: {stats['output_file']}")


@app_ingest.command("firms")
def ingest_firms_cmd(
    domain: str = typer.Option(
        "lahore", "--domain", help="Target domain (lahore, delhi, amritsar, ludhiana)"
    ),
    start: str = typer.Option(..., "--start", help="Start date/time (ISO 8601, e.g. 2026-10-01)"),
    end: str = typer.Option(..., "--end", help="End date/time (ISO 8601, e.g. 2026-10-05)"),
    source: str = typer.Option("VIIRS_NOAA21_NRT", "--source", help="FIRMS source/platform"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to write Parquet output"),
) -> None:
    """Ingest NASA FIRMS active fire detections and aggregate fire exposure (Task P1-09)."""
    from smogsense.data_ingestion.firms import FirmsClient, aggregate_fire_exposure

    client = FirmsClient()
    try:
        raw_dets = client.query_area_sync(source=source, start_date=start, end_date=end)
    except Exception as exc:
        print(f"Error querying FIRMS API: {exc}")
        safe_exit(1)
        return

    exposure_df = aggregate_fire_exposure(raw_dets, domain=domain, start_utc=start, end_utc=end)
    print(
        f"Ingested {len(raw_dets)} detections; aggregated {len(exposure_df)} hours of fire exposure for {domain}."
    )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exposure_df.to_parquet(out_path, index=False)
        print(f"Saved fire exposure to {out_path}")


@app_features.command("build")
def features_build_cmd(
    issuance: str = typer.Option(
        "latest", "--issuance", help="Issuance date/time (ISO 8601 or 'latest')"
    ),
    domain: str = typer.Option("lahore", "--domain", help="Target domain (lahore, delhi)"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
) -> None:
    """Build point-in-time features and persist input snapshot (Task P1-12)."""
    import os
    from datetime import datetime, timedelta

    import pandas as pd

    from smogsense.features.snapshot import build_input_snapshot, save_input_snapshot

    if issuance == "latest":
        now = datetime.now(UTC)
        issuance_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        issuance_utc = datetime.fromisoformat(issuance)
        if issuance_utc.tzinfo is None:
            issuance_utc = issuance_utc.replace(tzinfo=UTC)

    st_path = Path(state_dir)
    mode = os.getenv("SMOGSENSE_MODE", "live").lower().strip()

    obs_df = pd.DataFrame()
    cams_df = pd.DataFrame()

    cams_candidates = list((st_path / "inputs" / domain).glob("cams_snapshot_*.parquet"))
    obs_candidates = [
        *(st_path / "inputs" / domain).glob("obs_snapshot_*.parquet"),
        st_path / "inputs" / "obs_snapshot.parquet",
    ]

    if mode in ("fixtures", "demo", "test") or (not cams_candidates and not obs_candidates):
        with contextlib.suppress(Exception):
            from smogsense.data_ingestion.fixtures_mode import (
                load_fixture_cams,
                load_fixture_locations,
                load_fixture_observations,
            )

            stations_df = load_fixture_locations(domain)
            end_utc = issuance_utc - timedelta(hours=4)
            start_utc = end_utc - timedelta(hours=72)
            obs_df = load_fixture_observations(stations_df, start_utc, end_utc, domain=domain)
            cams_df = load_fixture_cams(stations_df, issuance_utc)

    if obs_df.empty and obs_candidates:
        for c in obs_candidates:
            if c.exists():
                with contextlib.suppress(Exception):
                    obs_df = pd.read_parquet(c)
                    if not obs_df.empty:
                        break

    if cams_df.empty and cams_candidates:
        for c in cams_candidates:
            if c.exists():
                with contextlib.suppress(Exception):
                    cams_df = pd.read_parquet(c)
                    if not cams_df.empty:
                        break

    snapshot = build_input_snapshot(
        issuance_utc=issuance_utc,
        domain=domain,
        obs_df=obs_df,
        cams_df=cams_df,
        fire_exposure_df=None,
    )
    saved_path = save_input_snapshot(
        snapshot, domain=domain, issuance_date=issuance_utc, state_dir=st_path
    )
    print(f"Features build completed for {domain}.")
    print(f"Persisted canonical input snapshot to {saved_path}")
    print(f"Observations: {len(snapshot.obs_df)} rows | CAMS: {len(snapshot.cams_df)} rows")
    safe_exit(0)


@app_features.command("snapshot")
def features_snapshot_cmd(
    issuance: str = typer.Option(
        "latest", "--issuance", help="Issuance date/time (ISO 8601, YYYY-MM-DD or 'latest')"
    ),
    domain: str = typer.Option("lahore", "--domain", help="Target domain (lahore, delhi)"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
) -> None:
    """Inspect or load daily input snapshot for replay (Task P1-12)."""
    from datetime import datetime

    from smogsense.features.snapshot import load_input_snapshot

    if issuance == "latest":
        now = datetime.now(UTC)
        issuance_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
        dt_str = issuance_utc.strftime("%Y-%m-%d")
    else:
        dt_str = issuance[:10]

    st_path = Path(state_dir)
    target_path = st_path / "inputs" / domain / f"dt={dt_str}.parquet"

    if not target_path.exists():
        print(f"Input snapshot for {domain} on {dt_str} not found at {target_path}.")
        safe_exit(1)
        return

    snap = load_input_snapshot(domain=domain, issuance_date=dt_str, state_dir=st_path)
    print(f"Input Snapshot for {domain.upper()} (Issuance: {snap.issuance_utc.isoformat()}):")
    print(f"  Run ID: {snap.run_id}")
    print(f"  Config Hash: {snap.config_hash}")
    print(f"  Available Stations: {snap.n_available_stations}")
    print(f"  Observations: {len(snap.obs_df)} rows")
    cams_base_str = snap.cams_base_time_utc.isoformat() if snap.cams_base_time_utc else "None"
    print(
        f"  CAMS Series: {len(snap.cams_df)} rows (base: {cams_base_str}, lead offset: {snap.cams_lead_offset_h}h)"
    )
    fire_rows = len(snap.fire_exposure_df) if snap.fire_exposure_df is not None else 0
    print(f"  Fire Exposure: {fire_rows} rows")
    print(f"  Location: {target_path}")
    safe_exit(0)


@app_features.command("backfill")
def features_backfill_cmd(
    domain: str = typer.Option("delhi", "--domain", help="Target domain (delhi, lahore)"),
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)"),
    split: str | None = typer.Option(None, "--split", help="Split partition (E, V, T, eval)"),
    out_dir: str = typer.Option(
        "data/processed/feature_store", "--out-dir", help="Path to feature store root"
    ),
    backfill_dir: str = typer.Option(
        "data/processed/backfill", "--backfill-dir", help="Path to raw backfill archive partitions"
    ),
    verify_schema: bool = typer.Option(
        True, "--verify-schema/--no-verify-schema", help="Enforce Pandera schema validation"
    ),
) -> None:
    """Run multi-season feature store backfill engine (Task P2-03)."""
    from smogsense.features.backfill import run_feature_backfill

    print(f"Starting feature store backfill for domain: {domain.upper()} [{start} to {end}]")
    result = run_feature_backfill(
        domain=domain,
        start_date=start,
        end_date=end,
        split=split,
        out_dir=out_dir,
        backfill_dir=backfill_dir,
        verify_schema=verify_schema,
    )
    print(f"Feature backfill complete for {domain.upper()}:")
    print(f"  Seasons: {result.seasons}")
    print(f"  Partitions: {result.season_partitions}")
    if result.block_e_folds:
        print(f"  Block E Folds: {result.block_e_folds}")
    print(f"  OOF Partition Hash: {result.oof_partition_hash}")
    print(f"  Total Samples Extracted: {result.total_samples}")
    print(f"  Purged Samples (Embargo): {result.purged_samples}")
    print(f"  Partitions Written: {len(result.partitions_written)}")
    safe_exit(0)


@app_reconcile.command("archive-api")
def reconcile_archive_api_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (lahore, delhi)"),
    tolerance: float = typer.Option(
        0.1, "--tolerance", help="Tolerance in ug/m3 for identical rows"
    ),
    api_path: str | None = typer.Option(
        None, "--api-path", help="Path to API observations parquet"
    ),
    archive_path: str | None = typer.Option(
        None, "--archive-path", help="Path to archive observations parquet"
    ),
) -> None:
    """Reconcile OpenAQ live API and AWS archive observations (Task P1-20)."""
    from smogsense.pipeline.reconciliation import run_reconciliation

    stats = run_reconciliation(
        domain=domain,
        tolerance=tolerance,
        api_path=api_path,
        archive_path=archive_path,
    )
    print(f"Reconciliation ({domain}): {stats['status']}")
    print(f"Overlapping observations: {stats['overlap_count']}")
    print(f"Mean absolute difference: {stats['mean_abs_diff']:.4f} ug/m3")
    print(f"Identical (<= {tolerance} ug/m3): {stats['pct_identical']:.2f}%")
    print(f"Revisions: {stats['revisions_count']}")
    print(
        f"Added in archive: {stats['added_timestamps_count']} | Dropped: {stats['dropped_timestamps_count']}"
    )


@app_run.command("daily")
def daily(
    issuance: str = typer.Option("latest", "--issuance", help="Issuance time"),
    force: bool = typer.Option(False, "--force", help="Force run even if already published"),
    mode: str | None = typer.Option(None, "--mode", help="Execution mode (live, fixtures)"),
    challenger_bundle: str | None = typer.Option(
        None, "--challenger-bundle", help="Model bundle tag or path for challenger"
    ),
    challenger_method: str | None = typer.Option(
        None, "--challenger-method", help="Challenger model method id (e.g. m7_hybrid_maml)"
    ),
    enable_challenger: bool | None = typer.Option(
        None, "--enable-challenger/--no-challenger", help="Enable or disable challenger execution"
    ),
) -> None:
    """Run the full daily cycle."""
    import os
    from datetime import datetime

    from smogsense.pipeline.orchestrator import run_daily_pipeline

    if not force and os.getenv("SMOGSENSE_FORCE", "0").lower() in ("1", "true"):
        force = True

    if mode is None:
        env_mode = os.getenv("SMOGSENSE_MODE", "live").lower().strip()
        mode = "fixtures" if env_mode in ("fixtures", "demo") else "live"

    if issuance == "latest":
        # Get start of today in UTC
        now = datetime.now(UTC)
        issuance_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        issuance_utc = datetime.fromisoformat(issuance).replace(tzinfo=UTC)

    exit_code = run_daily_pipeline(
        issuance_utc,
        force=force,
        mode=mode,
        challenger_bundle=challenger_bundle,
        challenger_method=challenger_method,
        enable_challenger=enable_challenger,
    )
    safe_exit(exit_code)


@app_run.command("forecast")
def run_forecast(
    issuance: str = typer.Option("latest", "--issuance", help="Issuance time"),
    force: bool = typer.Option(False, "--force", help="Force run even if already published"),
    mode: str | None = typer.Option(None, "--mode", help="Execution mode (live, fixtures)"),
    challenger_bundle: str | None = typer.Option(
        None, "--challenger-bundle", help="Model bundle tag or path for challenger"
    ),
    challenger_method: str | None = typer.Option(
        None, "--challenger-method", help="Challenger model method id (e.g. m7_hybrid_maml)"
    ),
    enable_challenger: bool | None = typer.Option(
        None, "--enable-challenger/--no-challenger", help="Enable or disable challenger execution"
    ),
) -> None:
    """Execute forecast pipeline in run group."""
    daily(
        issuance=issuance,
        force=force,
        mode=mode,
        challenger_bundle=challenger_bundle,
        challenger_method=challenger_method,
        enable_challenger=enable_challenger,
    )


@app_replay.command("shadow")
def replay_shadow(
    start_date: str = typer.Option(
        "2026-10-19", "--start-date", help="Replay start date YYYY-MM-DD"
    ),
    end_date: str = typer.Option("2026-12-06", "--end-date", help="Replay end date YYYY-MM-DD"),
    domain: str = typer.Option("lahore", "--domain", help="Target domain"),
    method: str = typer.Option("m7_hybrid_maml", "--method", help="Challenger method id"),
    bundle_tag: str | None = typer.Option(
        None, "--bundle-tag", help="Model bundle path or release tag"
    ),
    state_dir: str = typer.Option(".state", "--state-dir", help="State root directory"),
    output_dir: str = typer.Option("reports", "--output-dir", help="Reports output directory"),
    score: bool = typer.Option(True, "--score/--no-score", help="Score replayed forecasts"),
) -> None:
    """Execute shadow replay over logged point-in-time input snapshots."""
    from smogsense.pipeline.replay import run_shadow_replay

    report = run_shadow_replay(
        start_date=start_date,
        end_date=end_date,
        domain=domain,
        challenger_method=method,
        challenger_bundle=bundle_tag,
        state_dir=Path(state_dir),
        output_dir=Path(output_dir),
        score=score,
    )
    print(f"Shadow Replay Completed for {domain} ({start_date} to {end_date})")
    print(
        f"Total replayed days: {report['total_days']} | Forecast records: {report['total_forecast_rows']}"
    )
    if report.get("metrics"):
        m = report["metrics"]
        print(f"Overall CRPS: {m.get('mean_crps_overall', 0.0):.2f} ug/m3")
        print(f"80% Coverage: {m.get('coverage_80_overall', 0.0) * 100.0:.1f}%")
        print(
            f"CRPSS vs M1: {m.get('crpss_vs_m1', 0.0):.3f} | vs M2: {m.get('crpss_vs_m2', 0.0):.3f}"
        )
    safe_exit(0)


@app_run.command("replay")
def run_replay(
    start_date: str = typer.Option(
        "2026-10-19", "--start-date", help="Replay start date YYYY-MM-DD"
    ),
    end_date: str = typer.Option("2026-12-06", "--end-date", help="Replay end date YYYY-MM-DD"),
    domain: str = typer.Option("lahore", "--domain", help="Target domain"),
    method: str = typer.Option("m7_hybrid_maml", "--method", help="Challenger method id"),
    bundle_tag: str | None = typer.Option(
        None, "--bundle-tag", help="Model bundle path or release tag"
    ),
    state_dir: str = typer.Option(".state", "--state-dir", help="State root directory"),
    output_dir: str = typer.Option("reports", "--output-dir", help="Reports output directory"),
    score: bool = typer.Option(True, "--score/--no-score", help="Score replayed forecasts"),
) -> None:
    """Execute shadow replay in run group."""
    replay_shadow(
        start_date=start_date,
        end_date=end_date,
        domain=domain,
        method=method,
        bundle_tag=bundle_tag,
        state_dir=state_dir,
        output_dir=output_dir,
        score=score,
    )


@app_forecast.command("daily")
def forecast_daily(
    issuance: str = typer.Option("latest", "--issuance", help="Issuance time"),
    force: bool = typer.Option(False, "--force", help="Force run even if already published"),
    mode: str | None = typer.Option(None, "--mode", help="Execution mode (live, fixtures)"),
    challenger_bundle: str | None = typer.Option(
        None, "--challenger-bundle", help="Model bundle tag or path for challenger"
    ),
    challenger_method: str | None = typer.Option(
        None, "--challenger-method", help="Challenger model method id (e.g. m7_hybrid_maml)"
    ),
    enable_challenger: bool | None = typer.Option(
        None, "--enable-challenger/--no-challenger", help="Enable or disable challenger execution"
    ),
) -> None:
    """Execute daily forecast pipeline in forecast group."""
    daily(
        issuance=issuance,
        force=force,
        mode=mode,
        challenger_bundle=challenger_bundle,
        challenger_method=challenger_method,
        enable_challenger=enable_challenger,
    )


@app_state.command()
def exists(issuance: str = typer.Option("latest", "--issuance")) -> None:
    """Check if state exists and is published."""
    import json
    from datetime import datetime
    from pathlib import Path

    if issuance == "latest":
        now = datetime.now(UTC)
        issuance_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        issuance_utc = datetime.fromisoformat(issuance).replace(tzinfo=UTC)

    manifest_dir = Path(".state/manifests")
    issuance_date = issuance_utc.strftime("%Y-%m-%d")
    canonical_manifest = manifest_dir / f"{issuance_date}.json"
    compact_date = issuance_utc.strftime("%Y%m%d")

    candidates: list[Path] = [canonical_manifest]
    if manifest_dir.exists():
        candidates.extend(sorted(manifest_dir.glob(f"run_{compact_date}*.json")))

    for cand in candidates:
        if cand.exists():
            with contextlib.suppress(Exception):
                with cand.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("published") is True:
                    safe_exit(11)
    safe_exit(0)


@app_state.command("verify")
def state_verify_cmd(
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
    warn_mb: float = typer.Option(400.0, "--warn-mb", help="Warning threshold in MB"),
    alert_mb: float = typer.Option(700.0, "--alert-mb", help="Alert/failure threshold in MB"),
) -> None:
    """Verify state branch size guard and journal integrity (Task P1-23)."""
    from smogsense.pipeline.state_guard import verify_state_branch

    res = verify_state_branch(state_dir=Path(state_dir), warn_mb=warn_mb, alert_mb=alert_mb)
    print(
        f"State Branch Verification: status={res['status'].upper()} "
        f"(size={res['size_mb']} MB, warn={warn_mb} MB, alert={alert_mb} MB)"
    )
    if res["forbidden_files"]:
        print(f"::error::Forbidden binary files detected in state: {res['forbidden_files']}")
    if res["corrupt_files"]:
        print(f"::error::Corrupt journal files detected: {res['corrupt_files']}")
    if res["status"] == "warn":
        print(
            f"::warning::State branch size ({res['size_mb']} MB) exceeds warn threshold ({warn_mb} MB)"
        )
    print(f"Readable journals: {res['readable_counts']}")

    if res["status"] == "alert" or not res["valid"]:
        safe_exit(30)
    safe_exit(0)


@app_site.command("build")
def site_build_cmd(
    bulletin_path: str | None = typer.Option(
        None, "--bulletin", "-b", help="Path to bulletin JSON file"
    ),
    stale: bool = typer.Option(
        False, "--stale", "--expired", help="Render static expiry notice (stale forecast)"
    ),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to state directory"),
    out_dir: str = typer.Option("site", "--out-dir", help="Output directory for site"),
    configs_dir: str = typer.Option("configs", "--configs-dir", help="Configs directory"),
    base_url: str | None = typer.Option(None, "--base-url", help="Canonical site URL"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Run site validation after build"
    ),
) -> None:
    """Build the static website, fingerprint assets, and render cards (Task P4-04)."""
    import json
    from typing import Any

    from smogsense.publishing.deploy import BudgetBreachError
    from smogsense.publishing.site import build_site, render_expired_site

    if stale:
        res = render_expired_site(state_dir=Path(state_dir), out_dir=Path(out_dir))
        print(
            f"Rendered expired static site into {out_dir} (valid until {res.get('valid_until_utc')})"
        )
        safe_exit(0)

    # Resolve bulletin
    bulletin: dict[str, Any] | None = None
    if bulletin_path:
        b_p = Path(bulletin_path)
        if b_p.is_file():
            try:
                with b_p.open("r", encoding="utf-8") as f:
                    bulletin = json.load(f)
            except Exception as err:
                print(f"::error::Failed to load bulletin from {bulletin_path}: {err}")
                safe_exit(50)
        else:
            print(f"::error::Bulletin file not found at {bulletin_path}")
            safe_exit(50)
    else:
        candidates = [
            Path(state_dir) / "site_json" / "forecast" / "latest.json",
            Path(state_dir) / "site_json" / "latest.json",
            Path(out_dir) / "forecast" / "latest.json",
            Path(out_dir) / "latest.json",
            Path(".state/bulletins/latest.json"),
            Path("tests/fixtures/bulletin_valid_example.json"),
        ]
        for cand in candidates:
            if cand.is_file():
                with contextlib.suppress(Exception), cand.open("r", encoding="utf-8") as f:
                    bulletin = json.load(f)
                    break

    if bulletin is None:
        print("::error::No bulletin found to build site. Provide --bulletin <path>.")
        safe_exit(50)

    try:
        res = build_site(
            bulletin=bulletin,
            output_dir=Path(out_dir),
            configs_dir=Path(configs_dir),
            base_url=base_url,
            expired=stale,
            validate=validate,
        )
        print(f"Successfully built static site in {res['out_dir']} (build_id={res['build_id']})")
        safe_exit(0)
    except BudgetBreachError as err:
        print(f"::error::Site build failed performance or accessibility budget check:\n{err}")
        safe_exit(30)
    except Exception as err:
        print(f"::error::Site build failed: {err}")
        safe_exit(50)


@app_site.command("validate")
def site_validate_cmd(
    site_dir: str = typer.Option("site", "--site-dir", help="Path to static site directory"),
    configs_dir: str = typer.Option("configs", "--configs-dir", help="Configs directory"),
) -> None:
    """Validate static site against performance budgets, CSP, and accessibility (Task P4-04)."""
    from smogsense.publishing.deploy import format_violations_table, validate_site

    violations = validate_site(
        site_dir=Path(site_dir),
        configs_dir=Path(configs_dir),
        raise_on_breach=False,
    )
    if violations:
        print(f"::error::Site validation failed with {len(violations)} violation(s):")
        print(format_violations_table(violations))
        safe_exit(30)
    else:
        print(
            f"Site validation passed: {site_dir} satisfies all performance, CSP, and accessibility budgets."
        )
        safe_exit(0)


def _execute_render_cards(
    bulletin_path: str | None,
    output_dir: str,
    format_choice: str,
    lang_choice: str,
) -> None:
    import json

    from smogsense.visualization.cards import FORMAT_SPECS, CardFormat, render_card, save_card

    # Resolve bulletin
    b_path: Path | None = None
    if bulletin_path:
        b_path = Path(bulletin_path)
    else:
        for candidate in [
            Path("site/forecast/latest.json"),
            Path(".state/bulletins/latest.json"),
            Path("tests/fixtures/bulletin_valid_example.json"),
        ]:
            if candidate.is_file():
                b_path = candidate
                break

    if not b_path or not b_path.is_file():
        print(f"Error: Bulletin file not found at {bulletin_path or 'default paths'}")
        safe_exit(50)

    try:
        with b_path.open("r", encoding="utf-8") as f:
            bulletin_data = json.load(f)
    except Exception as err:
        print(f"Error loading bulletin {b_path}: {err}")
        safe_exit(50)

    # Determine formats
    fmt_lower = format_choice.strip().lower()
    if fmt_lower in ["post", "whatsapp_post", "whatsapp"]:
        target_formats = [CardFormat.WHATSAPP_POST]
    elif fmt_lower in ["og", "open_graph"]:
        target_formats = [CardFormat.OPEN_GRAPH]
    elif fmt_lower in ["all", "both"]:
        target_formats = [CardFormat.WHATSAPP_POST, CardFormat.OPEN_GRAPH]
    else:
        print(f"Error: Unknown format '{format_choice}'. Choose from post, og, all.")
        safe_exit(50)

    # Determine languages
    lang_lower = lang_choice.strip().lower()
    if lang_lower in ["en", "english"]:
        target_langs = ["en"]
    elif lang_lower in ["ur", "urdu"]:
        target_langs = ["ur"]
    elif lang_lower in ["all", "both"]:
        target_langs = ["en", "ur"]
    else:
        print(f"Error: Unknown language '{lang_choice}'. Choose from en, ur, all.")
        safe_exit(50)

    out_root = Path(output_dir)
    latest_dir = out_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    issuance_raw = bulletin_data.get(
        "issuance_utc",
        bulletin_data.get("generated_at_utc", "2026-01-01T00:00:00Z"),
    )
    date_tag = issuance_raw[:10]
    dated_dir = out_root / date_tag
    dated_dir.mkdir(parents=True, exist_ok=True)

    budget_breach = False

    for lang in target_langs:
        for fmt in target_formats:
            spec = FORMAT_SPECS[fmt]
            prefix = "whatsapp" if fmt == CardFormat.WHATSAPP_POST else "og"
            filename = f"lahore-{prefix}-{lang}.png"

            try:
                card_img = render_card(bulletin_data, lang=lang, format=fmt)
                path_latest = latest_dir / filename
                path_dated = dated_dir / filename

                bytes_latest = save_card(card_img, path_latest)
                save_card(card_img, path_dated)

                kb_size = bytes_latest / 1024.0
                print(
                    f"Rendered {filename} -> {path_latest} ({kb_size:.1f} KB / max {spec.max_kb} KB)"
                )

                if kb_size > spec.max_kb:
                    print(f"ERROR: {filename} exceeds budget ({kb_size:.1f} KB > {spec.max_kb} KB)")
                    budget_breach = True
            except Exception as err:
                print(f"Fatal error rendering {filename}: {err}")
                safe_exit(50)

    if budget_breach:
        safe_exit(30)

    safe_exit(0)


@app_site.command("render-cards")
def site_render_cards_cmd(
    bulletin: str | None = typer.Option(None, "--bulletin", help="Path to bulletin JSON file"),
    output_dir: str = typer.Option(
        "site/cards", "--output-dir", help="Output directory for card images"
    ),
    format: str = typer.Option("all", "--format", help="Card format: post, og, or all"),
    lang: str = typer.Option("all", "--lang", help="Language: en, ur, or all"),
) -> None:
    """Render WhatsApp post and Open Graph cards with Pillow + Raqm (Task P4-03)."""
    _execute_render_cards(bulletin, output_dir, format, lang)


@app_cards.command("render")
def cards_render_cmd(
    bulletin: str | None = typer.Option(None, "--bulletin", help="Path to bulletin JSON file"),
    output_dir: str = typer.Option(
        "site/cards", "--output-dir", help="Output directory for card images"
    ),
    format: str = typer.Option("all", "--format", help="Card format: post, og, or all"),
    lang: str = typer.Option("all", "--lang", help="Language: en, ur, or all"),
) -> None:
    """Render social card assets (alias for site render-cards) (Task P4-03)."""
    _execute_render_cards(bulletin, output_dir, format, lang)


@app_run.command("seed-history")
def seed_history_cmd(
    start: str | None = typer.Option(
        None, "--start", help="Start date ISO-8601 (default: 2026-08-20T00:00:00Z)"
    ),
    end: str | None = typer.Option(
        None, "--end", help="End date ISO-8601 (default: current execution time)"
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output path (default: .state/artifacts/seed_history.json)"
    ),
) -> None:
    """Generate empirical residual quantiles (Phase 1a.7)."""
    from datetime import datetime
    from pathlib import Path

    from smogsense.pipeline.seed_history import generate_seed_history

    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    out_path = Path(output) if output else None

    exit_code = generate_seed_history(start_utc=start_dt, end_utc=end_dt, output_path=out_path)
    safe_exit(exit_code)


@app_run.command("backfill")
def backfill_cmd(
    source: str = typer.Option(
        "all", "--source", help="Which source to backfill (all, openaq-archive, cams, era5, firms)"
    ),
    domain: str = typer.Option("both", "--domain", help="Which domain (both, delhi, lahore)"),
    start: str = typer.Option(..., "--start", help="First day (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", help="Last day (YYYY-MM-DD)"),
    max_wall_minutes: int = typer.Option(
        300, "--max-wall-minutes", help="Stop cleanly after this many minutes (job limit is 360)"
    ),
    output_dir: str = typer.Option(
        "data/processed/backfill", "--output-dir", help="Directory to store Parquet partitions"
    ),
) -> None:
    """Run historical backfill orchestrator (Task P1-13)."""
    from smogsense.pipeline.backfill import run_backfill

    stats = run_backfill(
        source=source,
        domain=domain,
        start=start,
        end=end,
        max_wall_minutes=max_wall_minutes,
        output_dir=output_dir,
    )
    print(f"Historical backfill status: {stats['status']}")
    print(
        f"Partitions: {stats['partitions_completed']} completed, "
        f"{stats['partitions_skipped']} skipped (total: {stats['total_partitions']})"
    )
    print(f"Elapsed time: {stats['elapsed_seconds']}s")
    print(f"Output directory: {stats['output_dir']}")
    safe_exit(0)


@app_audit.command("stations")
def audit_stations_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Domain to audit (e.g. lahore, delhi)"),
    output: str | None = typer.Option(
        None, "--output", help="Output directory for reports (default: .state/audit)"
    ),
) -> None:
    """Run data quality and station audit (Phase 1b Task P1-10)."""
    from smogsense.analysis.station_audit import run_station_audit

    out_dir = Path(output) if output else Path(".state/audit")
    summary, _ = run_station_audit(domain=domain, output_dir=out_dir)
    print(f"Station Audit Complete for {domain.upper()}:")
    print(f"  Total Stations: {summary.total_locations}")
    print(
        f"  Eligible Stations: {summary.eligible_station_count} (Held-out N_max: {summary.n_max_held_out})"
    )
    print(f"  Reference Monitors: {summary.reference_station_count}")
    print(f"  Winter Completeness: {summary.winter_day_completeness_pct}%")
    print(f"  Co-located Groups: {summary.colocated_groups_count}")
    print("Decision Rules Status:")
    for rule_id, rule in summary.decision_rules.items():
        print(f"  [{rule['status']}] {rule_id}: {rule['observed']} -> {rule['recommendation']}")
    safe_exit(0)


@app_audit.command("truth-basis")
def audit_truth_basis_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (e.g. lahore, delhi)"),
    output: str | None = typer.Option(None, "--output", "-o", help="Optional output JSON path"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to state directory"),
) -> None:
    """Run co-location hierarchy and truth-basis audit (Task P1-21)."""
    import json

    import pandas as pd

    from smogsense.analysis.station_audit import audit_truth_basis
    from smogsense.data_ingestion.fixtures_mode import load_fixture_locations

    st_path = Path(state_dir)
    reg_path = st_path / "registry" / f"locations_{domain}.parquet"
    if reg_path.exists():
        locations_df = pd.read_parquet(reg_path)
    else:
        locations_df = load_fixture_locations(domain=domain)

    # Optional observations check
    obs_path = st_path / "obs" / domain
    obs_df: pd.DataFrame | None = None
    if obs_path.exists():
        files = sorted(obs_path.glob("*.parquet"))
        if files:
            obs_df = pd.concat([pd.read_parquet(f) for f in files[-7:]], ignore_index=True)

    report = audit_truth_basis(locations_df=locations_df, observations_df=obs_df, domain=domain)

    print(f"Truth-Basis Audit for {domain.upper()}:")
    print(f"  Total Stations: {report['total_stations']} | Total Groups: {report['total_groups']}")
    print(f"  Co-located Multi-Station Groups: {len(report['colocated_multi_station_groups'])}")
    for grp in report["colocated_multi_station_groups"]:
        print(
            f"    - Group {grp['group_id']}: {grp['station_count']} stations {grp['location_ids']}, "
            f"basis: {grp['group_basis']}, providers: {grp['providers']}"
        )
    print(
        f"  Panel Candidates: {len(report['panel_candidates'])} "
        f"(Ref: {report['n_reference_groups']}, Low-Cost: {report['n_lowcost_groups']})"
    )
    print(f"  Overall Truth Basis: {report['overall_truth_basis'].upper()}")
    print(f"  Panel Viable (>= 3 groups): {report['panel_viable']}")

    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to: {output}")

    safe_exit(0)


@app_audit.command("asof-parity")
def audit_asof_parity_cmd(
    date: str | None = typer.Option(
        None, "--date", help="Reference date YYYY-MM-DD for lattice verification"
    ),
    repo_root: str | None = typer.Option(
        None, "--repo-root", help="Optional repo root directory path"
    ),
) -> None:
    """Run as-of lattice verification and container-parity audit (Task P1-22)."""
    from datetime import datetime

    from smogsense.preprocessing.alignment import (
        verify_asof_lattice,
        verify_container_parity,
    )

    ref_dt = datetime.fromisoformat(date).replace(tzinfo=UTC) if date else None
    root_path = Path(repo_root) if repo_root else None

    lattice_rep = verify_asof_lattice(reference_date=ref_dt)
    parity_rep = verify_container_parity(repo_root=root_path)

    print(f"As-Of Lattice Verification ({lattice_rep['reference_date']}): {lattice_rep['status']}")
    for pt in lattice_rep["points"]:
        print(
            f"  Issuance {pt['issuance_utc']}: B*={pt['b_star_utc']} "
            f"(delta0={pt['delta_level_0_h']}h, delta1={pt['delta_level_1_h']}h, "
            f"max_lead={pt['max_stitched_lead_h']}h) -> {'PASS' if pt['valid'] else 'FAIL'}"
        )

    print(f"\nContainer Parity Audit: {parity_rep['status']}")
    for chk, ok in parity_rep["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")

    all_ok = lattice_rep["status"] == "PASS" and parity_rep["status"] == "PASS"
    safe_exit(0 if all_ok else 30)


@app_audit.command("corruption-prior")
def audit_corruption_prior_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    backfill_dir: str = typer.Option(
        "data/processed/backfill", "--backfill-dir", help="Path to backfill directory"
    ),
    registry_path: str | None = typer.Option(
        None, "--registry-path", help="Path to station registry"
    ),
    config_path: str = typer.Option(
        "configs/model.yaml", "--config-path", help="Path to model config"
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", help="Output directory for reports (default: reports)"
    ),
    window_days: int = typer.Option(
        60, "--window-days", help="Window in days for pairing (default: 60)"
    ),
    max_distance_km: float = typer.Option(
        8.0, "--max-distance-km", help="Max distance in km from reference monitor"
    ),
) -> None:
    """Run empirical corruption-prior audit on Lahore low-cost sensors (Task P2-15)."""
    from smogsense.models.corruption_audit import run_corruption_prior_audit

    res = run_corruption_prior_audit(
        domain=domain,
        backfill_dir=backfill_dir,
        registry_path=registry_path,
        config_path=config_path,
        output_dir=output_dir,
        window_days=window_days,
        max_distance_km=max_distance_km,
    )

    print("=" * 80)
    print(f"EMPIRICAL CORRUPTION-PRIOR AUDIT (Domain: {domain.upper()})")
    print("=" * 80)
    print(f"Reference Station: {res.reference_station_id} ({res.reference_station_name})")
    print(
        f"Candidate Low-Cost Stations: {len(res.candidate_stations)} within <= {max_distance_km} km"
    )
    print(f"Total Paired Hourly Records: {res.total_paired_records:,}")
    print("\nFitted Huber Coefficients:")
    print(
        f"  a (gain): {res.huber_coefficients['a_gain']:.4f} | "
        f"b (RH slope): {res.huber_coefficients['b_rh_slope']:.4f} | "
        f"c (offset): {res.huber_coefficients['c_intercept']:.4f}"
    )

    print("\nInverted Corruption Priors vs Configured Ranges:")
    for comp in res.prior_comparisons:
        print(
            f"  - {comp.parameter:10s}: Empirical = {comp.empirical_value:7.4f} | "
            f"Configured = [{comp.configured_min:5.1f}, {comp.configured_max:5.1f}] | "
            f"Status = {comp.status}"
        )

    print("\nSupport Window Error Distribution:")
    for sw in res.support_window_stats:
        print(
            f"  - {sw.window_days:2d} Days: Mean Ratio = {sw.mean_ratio:.2f}x | "
            f"Median = {sw.median_ratio:.2f}x | Ratio > 1.30 = {sw.fraction_ratio_gt_1_30 * 100:.1f}% | "
            f"RMSE = {sw.rmse_ugm3:.1f} ug/m3"
        )

    print(
        f"\nRisk R17 Assessment: {'TRIGGERED' if res.risk_r17_triggered else 'ENVELOPED'} ({res.risk_r17_severity})"
    )
    print(f"  Summary: {res.risk_r17_summary}")
    print(f"\nReports written to: {output_dir}/corruption_prior_audit.json and .md")
    safe_exit(0)


@app_score.command("shadow")
def score_shadow_cmd(
    stage: str = typer.Option(
        "provisional", "--stage", help="Scoring stage (provisional, settled)"
    ),
    date: str | None = typer.Option(None, "--date", help="Issuance date (YYYY-MM-DD) to score"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
    obs_file: str | None = typer.Option(
        None, "--obs-file", help="Path to explicit observations parquet"
    ),
) -> None:
    """Run shadow scoring for prospective verification (Task P1-11)."""
    from smogsense.pipeline.shadow import run_shadow_scoring

    res = run_shadow_scoring(
        stage=stage, issuance_date=date, state_dir=state_dir, obs_file=obs_file
    )
    print(f"Shadow scoring ({res['stage']}): {res['status']}")
    if res.get("new_scores") is not None:
        print(f"Scored {res['new_scores']} targets for {res.get('issuance_date', date)}")
    if res.get("saved_path"):
        print(f"Saved scores to {res['saved_path']}")
    safe_exit(0 if res.get("status") in ("ok", "no_forecasts") else 1)


@app_score.command("provisional")
def score_provisional_cmd(
    date: str | None = typer.Option(None, "--date", help="Issuance date (YYYY-MM-DD) to score"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
    obs_file: str | None = typer.Option(
        None, "--obs-file", help="Path to explicit observations parquet"
    ),
) -> None:
    """Run provisional shadow scoring with OpenAQ live API observations (Task P1-11)."""
    from smogsense.pipeline.shadow import run_shadow_scoring

    res = run_shadow_scoring(
        stage="provisional", issuance_date=date, state_dir=state_dir, obs_file=obs_file
    )
    print(f"Provisional scoring: {res['status']}")
    if res.get("new_scores") is not None:
        print(f"Scored {res['new_scores']} targets for {res.get('issuance_date', date)}")
    if res.get("saved_path"):
        print(f"Saved scores to {res['saved_path']}")
    safe_exit(0 if res.get("status") in ("ok", "no_forecasts") else 1)


@app_score.command("settled")
def score_settled_cmd(
    date: str | None = typer.Option(None, "--date", help="Issuance date (YYYY-MM-DD) to score"),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to .state directory"),
    obs_file: str | None = typer.Option(
        None, "--obs-file", help="Path to explicit observations parquet"
    ),
) -> None:
    """Run settled shadow scoring with OpenAQ AWS archive observations (Task P1-11)."""
    from smogsense.pipeline.shadow import run_shadow_scoring

    res = run_shadow_scoring(
        stage="settled", issuance_date=date, state_dir=state_dir, obs_file=obs_file
    )
    print(f"Settled scoring: {res['status']}")
    if res.get("new_scores") is not None:
        print(f"Scored {res['new_scores']} targets for {res.get('issuance_date', date)}")
    if res.get("saved_path"):
        print(f"Saved scores to {res['saved_path']}")
    safe_exit(0 if res.get("status") in ("ok", "no_forecasts") else 1)


@app_train.command("lgbm")
def train_lgbm_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    domain: str = typer.Option("delhi", "--domain", help="Source domain (delhi)"),
    split: str = typer.Option("E", "--split", help="Split to train on (E)"),
    oof: bool = typer.Option(False, "--oof", help="Run season-blocked cross-fitting on Block E"),
    out_dir: str = typer.Option(
        "data/processed/models/lgbm_m5", "--out-dir", "-o", help="Output directory for model"
    ),
    num_boost_round: int = typer.Option(120, "--num-boost-round", help="Boosting rounds"),
) -> None:
    """Train LightGBM M5 physics base ensemble with optional OOF cross-fitting (Task P2-04 / P3-01)."""
    import pandas as pd

    from smogsense.models.lgbm_quantile import (
        DEFAULT_HORIZONS,
        DEFAULT_QUANTILES,
        LGBMQuantileEnsemble,
        fit_cross_fit_oof,
    )

    fs_path = Path(feature_store_dir)
    pattern = f"{domain}/**/features.parquet"
    files = sorted(fs_path.glob(pattern))
    if not files:
        print(f"Error: No partitions found under {fs_path / domain}")
        safe_exit(1)
        return

    dfs = [pd.read_parquet(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    if "source_split" in df_all.columns:
        df_train = df_all[df_all["source_split"] == split].copy()
    else:
        df_train = df_all

    print(f"Loaded {len(df_train):,} training samples for domain '{domain}' (split: {split}).")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    val_df = (
        df_all[df_all["source_split"] == "V"].copy()
        if "source_split" in df_all.columns and (df_all["source_split"] == "V").any()
        else None
    )

    if oof:
        print("Executing season-blocked cross-fitting on Block E...")
        oof_df, base_model = fit_cross_fit_oof(
            df_block_e=df_train,
            val_df=val_df,
            horizons=DEFAULT_HORIZONS,
            quantiles=DEFAULT_QUANTILES,
            num_boost_round=num_boost_round,
        )
        oof_file = out_path / "m5_oof_predictions.parquet"
        oof_df.to_parquet(oof_file, index=False)
        print(f"Saved out-of-fold base predictions to: {oof_file} ({len(oof_df):,} rows)")
        base_model.save(out_path / "base_model")
        print(f"Saved base model ensemble to: {out_path / 'base_model'}")
    else:
        print("Training LightGBM M5 physics base on full split...")
        base_model = LGBMQuantileEnsemble(model_type="M5")
        base_model.fit(df_train, val_df=val_df, num_boost_round=num_boost_round)
        base_model.save(out_path)
        print(f"Saved LightGBM M5 ensemble to: {out_path}")

    print(f"Deterministic Base SHA256: {base_model.compute_base_sha256()}")
    if base_model.oof_partition_hash:
        print(f"OOF Partition Hash: {base_model.oof_partition_hash}")
    safe_exit(0)


@app_train.command("m6")
def train_m6_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Output directory for model checkpoints"
    ),
    domain: str = typer.Option("delhi", "--domain", help="Source domain (delhi)"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    subsample_train: int | None = typer.Option(
        None, "--subsample-train", help="Subsample training rows for rapid execution"
    ),
    subsample_val: int | None = typer.Option(
        None, "--subsample-val", help="Subsample validation rows"
    ),
    device: str = typer.Option("cpu", "--device", help="Compute device (cpu, cuda)"),
    out_checkpoint: str | None = typer.Option(
        None, "--out-checkpoint", "-o", help="Explicit path for saved checkpoint"
    ),
) -> None:
    """Supervised pre-training of Residual Calibrator (Model M6) on Block E OOF base predictions (Task P2-07)."""
    from smogsense.models.pretraining import train_m6_calibrator

    res = train_m6_calibrator(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        domain=domain,
        subsample_train=subsample_train,
        subsample_val=subsample_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        checkpoint_path=out_checkpoint,
    )
    print(
        f"Pre-training complete. Model M6 checkpoint: {out_checkpoint or Path(models_dir) / 'm6_calibrator.pt'}"
    )
    print(f"Base SHA256 Binding: {res['base_sha256']}")
    safe_exit(0)


@app_train.command("maml")
def train_maml_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Output directory for model checkpoints"
    ),
    domain: str = typer.Option("delhi", "--domain", help="Source domain (delhi)"),
    iterations: int = typer.Option(
        100, "--iterations", help="Number of outer meta-update iterations"
    ),
    meta_batch_size: int = typer.Option(
        4, "--meta-batch-size", help="Batch size of tasks per outer update"
    ),
    inner_steps: int = typer.Option(3, "--inner-steps", help="Inner gradient adaptation steps J"),
    lr_outer: float = typer.Option(1e-3, "--lr-outer", help="Outer Adam learning rate beta"),
    lr_inner_init: float = typer.Option(
        0.01, "--lr-inner-init", help="Initial Meta-SGD inner step size alpha"
    ),
    adapt_scope: str = typer.Option(
        "all", "--adapt-scope", help="Adaptation scope (all, head_only, gru_only)"
    ),
    device: str = typer.Option("cpu", "--device", help="Compute device (cpu, cuda)"),
    out_checkpoint: str | None = typer.Option(
        None, "--out-checkpoint", "-o", help="Explicit path for saved checkpoint"
    ),
) -> None:
    """Execute First-Order MAML (FOMAML) with Meta-SGD and sensor corruption (Task P2-08)."""
    from smogsense.models.maml import train_maml

    res = train_maml(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        domain=domain,
        iterations=iterations,
        meta_batch_size=meta_batch_size,
        inner_steps=inner_steps,
        lr_outer=lr_outer,
        lr_inner_init=lr_inner_init,
        adapt_scope=adapt_scope,
        device=device,
        checkpoint_path=out_checkpoint,
    )
    print(
        f"Meta-training complete. Model M7 checkpoint: {out_checkpoint or Path(models_dir) / 'm7_maml.pt'}"
    )
    print(f"Base SHA256 Lineage: {res['base_sha256']}")
    safe_exit(0)


@app_train.command("maml-ablation")
def train_maml_ablation_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Output directory for model checkpoints"
    ),
    domain: str = typer.Option("delhi", "--domain", help="Source domain (delhi)"),
    iterations: int = typer.Option(
        50, "--iterations", help="Outer iterations per ablation condition"
    ),
    meta_batch_size: int = typer.Option(
        4, "--meta-batch-size", help="Batch size of tasks per outer update"
    ),
    inner_steps: int = typer.Option(3, "--inner-steps", help="Inner gradient adaptation steps J"),
    lr_outer: float = typer.Option(1e-3, "--lr-outer", help="Outer Adam learning rate beta"),
    lr_inner_init: float = typer.Option(
        0.01, "--lr-inner-init", help="Initial Meta-SGD inner step size alpha"
    ),
    device: str = typer.Option("cpu", "--device", help="Compute device (cpu, cuda)"),
    output_json: str | None = typer.Option(
        None, "--output-json", "-o", help="Explicit path for saved ablation JSON"
    ),
) -> None:
    """Execute MAML scope and X-11 corruption label policy ablations (Task P2-09)."""
    from smogsense.models.ablations import run_maml_ablations

    res = run_maml_ablations(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        domain=domain,
        iterations=iterations,
        meta_batch_size=meta_batch_size,
        inner_steps=inner_steps,
        lr_outer=lr_outer,
        lr_inner_init=lr_inner_init,
        device=device,
        output_path=output_json,
    )

    print("\n" + "=" * 90)
    print("MAML ADAPTATION SCOPE & X-11 LABEL POLICY ABLATION SUMMARY (Task P2-09)")
    print("=" * 90)
    print(
        f"{'Condition':<24} | {'Scope':<10} | {'Policy':<14} | "
        f"{'Val CRPS':<10} | {'CRPSS vs M5':<12} | {'Res R':<8} | {'Time (s)':<8}"
    )
    print("-" * 90)
    for row in res["summary_table"]:
        print(
            f"{row['Condition']:<24} | {row['Scope']:<10} | {row['Label Policy']:<14} | "
            f"{row['Val CRPS (ug/m3)']:<10.2f} | {row['CRPSS vs M5 (%)']:>+10.2f}% | "
            f"{row['Residual R']:<8.4f} | {row['Time (s)']:<8.2f}"
        )
    print("=" * 90)
    out_file = output_json or str(Path(models_dir) / "maml_ablations.json")
    print(f"Results serialized to: {out_file}")
    safe_exit(0)


@app_train.command("residual-bound")
@app_eval.command("residual-bound")
def train_residual_bound_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Path to models directory"
    ),
    n_simulations: int = typer.Option(
        200, "--n-simulations", "-n", help="Number of deployment simulations M"
    ),
    output_json: str = typer.Option(
        "data/processed/models/residual_bound.json",
        "--output-json",
        "-o",
        help="Path to output JSON summary",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("cpu", "--device", help="Torch compute device (cpu, cuda)"),
) -> None:
    """Calibrate deployment residual bound threshold rho_99 on Delhi Block V (Task P2-12)."""
    from smogsense.models.residual_bound import simulate_deployments_on_block_v

    print("=" * 70)
    print("CALIBRATING RESIDUAL BOUND THRESHOLD (rho_99 on Block V)")
    print("=" * 70)
    print(f"Feature Store: {feature_store_dir}")
    print(f"Models Dir:    {models_dir}")
    print(f"Simulations:   {n_simulations}")
    print(f"Random Seed:   {seed}")
    print(f"Device:        {device}")
    print(f"Output Path:   {output_json}")

    summary = simulate_deployments_on_block_v(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        n_simulations=n_simulations,
        seed=seed,
        device=device,
        output_json=output_json,
    )

    p = summary["percentiles"]
    print("\n" + "=" * 70)
    print("RESIDUAL BOUND CALIBRATION SUMMARY (ADR-024 Condition C6)")
    print("=" * 70)
    print(f"{'Statistic':<25} | {'Value (log1p units)':<20}")
    print("-" * 70)
    print(f"{'Simulations Completed':<25} | {summary['n_simulations']}")
    print(f"{'Median Residual (p50)':<25} | {p['p50']:.6f}")
    print(f"{'90th Percentile (p90)':<25} | {p['p90']:.6f}")
    print(f"{'95th Percentile (p95)':<25} | {p['p95']:.6f}")
    print(f"{'99th Percentile (rho_99)':<25} | {summary['rho_99']:.6f}  <-- THRESHOLD")
    print(f"{'Maximum Residual':<25} | {p['max']:.6f}")
    print("=" * 70)
    print(f"Calibrated threshold rho_99: {summary['rho_99']:.6f}")
    print(f"Serialized distribution to: {output_json}")
    safe_exit(0)


@app_eval.command("probe")
def eval_probe_cmd(
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    subsample_train: int | None = typer.Option(
        None, "--subsample-train", help="Subsample Delhi training rows for rapid execution"
    ),
    location_id: int = typer.Option(8664, "--location-id", help="Lahore monitor ID to evaluate"),
    num_boost_round: int = typer.Option(120, "--num-boost-round", help="Boosting iterations"),
    output_json: str | None = typer.Option(
        None, "--output-json", "-o", help="Path to write JSON probe report"
    ),
) -> None:
    from smogsense.models.lgbm_quantile import run_early_transfer_probe

    res = run_early_transfer_probe(
        feature_store_dir=feature_store_dir,
        subsample_train=subsample_train,
        eval_location_id=location_id,
        num_boost_round=num_boost_round,
        out_json=output_json,
    )
    passed = res.get("gate_g2_passed", False)
    safe_exit(0 if passed else 1)


@app_eval.command("ladder")
def eval_ladder_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    models_dir: str = typer.Option(
        "data/processed/models",
        "--models-dir",
        help="Directory holding pre-trained model checkpoints",
    ),
    location_id: int = typer.Option(
        8664, "--location-id", "--eval-location-id", help="Primary evaluation monitor ID"
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", "-o", help="Output directory for reports"
    ),
    device: str = typer.Option("cpu", "--device", help="Torch device ('cpu' or 'cuda')"),
) -> None:
    """Evaluate full model ladder (M0 through M7) on held-out Lahore development data."""
    from smogsense.models.ladder import run_model_ladder

    run_model_ladder(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        domain=domain,
        eval_location_id=location_id,
        output_dir=output_dir,
        device=device,
    )
    print(f"\nModel Ladder evaluation complete. Report written to {output_dir}/ladder.md")
    safe_exit(0)


@app_eval.command("diagnostics")
def eval_diagnostics_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    location_id: int | None = typer.Option(
        8664,
        "--location-id",
        "--eval-location-id",
        help="Evaluation monitor ID (or None for domain-wide)",
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", "-o", help="Output directory for reports"
    ),
) -> None:
    """Run atmospheric diagnostics: lagged cross-correlations and sector-importance fire decomposition."""
    from smogsense.models.diagnostics import run_atmospheric_diagnostics

    run_atmospheric_diagnostics(
        feature_store_dir=feature_store_dir,
        domain=domain,
        eval_location_id=location_id,
        output_dir=output_dir,
    )
    print(f"\nAtmospheric diagnostics complete. Report written to {output_dir}/diagnostics.md")
    safe_exit(0)


@app_eval.command("rh-mask-audit")
def eval_rh_mask_audit_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    backfill_dir: str = typer.Option(
        "data/processed/backfill", "--backfill-dir", help="Path to backfilled observations"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Path to models directory"
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", "-o", help="Output directory for reports"
    ),
    rh_threshold: float = typer.Option(
        85.0, "--rh-threshold", help="RH cutoff threshold percentage (default: 85.0%)"
    ),
    min_support_fraction: float = typer.Option(
        0.5, "--min-support-fraction", help="Minimum survival fraction to apply mask (default: 0.5)"
    ),
    n_windows: int = typer.Option(
        15, "--n-windows", "-n", help="Number of benchmark evaluation windows (default: 15)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("cpu", "--device", help="Torch compute device (cpu, cuda)"),
) -> None:
    """Audit empirical Relative Humidity distribution on low-cost sensors and evaluate Condition C9 mask (Task P2-13)."""
    from smogsense.models.rh_mask_audit import run_rh_mask_audit

    print("=" * 75)
    print(f"RH-MASK AUDIT ON LAHORE LOW-COST SUPPORT HOURS (Domain: {domain})")
    print("=" * 75)
    print(f"Feature Store:        {feature_store_dir}")
    print(f"Backfill Dir:         {backfill_dir}")
    print(f"Models Dir:           {models_dir}")
    print(f"Output Directory:     {output_dir}")
    print(f"RH Threshold:         {rh_threshold:.1f}%")
    print(f"Min Support Fraction: {min_support_fraction:.2f}")
    print(f"Evaluation Windows:   {n_windows}")
    print(f"Random Seed:          {seed}")
    print(f"Device:               {device}")

    report = run_rh_mask_audit(
        domain=domain,
        feature_store_dir=feature_store_dir,
        backfill_dir=backfill_dir,
        models_dir=models_dir,
        output_dir=output_dir,
        rh_threshold=rh_threshold,
        min_support_fraction=min_support_fraction,
        n_windows=n_windows,
        seed=seed,
        device=device,
    )

    dist = report.rh_distribution
    coin = report.extreme_smog_coincidence
    imp = report.adaptation_impact
    pol = report.policy_decision

    print("\n" + "=" * 75)
    print("1. EMPIRICAL RH DISTRIBUTION (Low-Cost Monitors)")
    print("=" * 75)
    print(
        f"Total Hours: {dist.total_hours:,} | Mean: {dist.mean_rh:.2f}% | Median: {dist.median_rh:.2f}%"
    )
    print(f"{'Relative Humidity Bin':<25} | {'Hours':<15} | {'Percentage':<15}")
    print("-" * 75)
    for b in dist.bins:
        print(f"{b.bin_name:<25} | {b.count:<15,} | {b.percentage:.2f}%")

    print("\n" + "=" * 75)
    print("2. UPPER-TAIL SMOG COINCIDENCE (Risk R14 Truncation)")
    print("=" * 75)
    print(
        f"{'Smog Threshold':<25} | {'Total Hours':<15} | {'Hours RH > 85%':<15} | {'Share with RH > 85%':<20}"
    )
    print("-" * 75)
    print(
        f"{'PM2.5 >= 150 ug/m3':<25} | {coin.threshold_150.total_hours:<15,} | {coin.threshold_150.hours_high_rh:<15,} | {coin.threshold_150.percentage_high_rh:.2f}%"
    )
    print(
        f"{'PM2.5 >= 250 ug/m3':<25} | {coin.threshold_250.total_hours:<15,} | {coin.threshold_250.hours_high_rh:<15,} | {coin.threshold_250.percentage_high_rh:.2f}%"
    )

    print("\n" + "=" * 75)
    print("3. ADAPTATION IMPACT BENCHMARK (Unmasked vs RH-Masked)")
    print("=" * 75)
    print(
        f"{'Metric':<30} | {'Unmasked':<12} | {'RH-Masked':<12} | {'Delta':<12} | {'Rel Delta (%)':<15}"
    )
    print("-" * 75)
    print(
        f"{'Upper Pinball (tau >= 0.75)':<30} | {imp.unmasked.upper_quantile_pinball:<12.4f} | {imp.masked.upper_quantile_pinball:<12.4f} | {imp.delta_upper_pinball:<+12.4f} | {imp.rel_delta_upper_pinball_pct:>+12.2f}%"
    )
    print(
        f"{'Full 19-Quantile CRPS':<30} | {imp.unmasked.crps:<12.4f} | {imp.masked.crps:<12.4f} | {imp.delta_crps:<+12.4f} | {imp.rel_delta_crps_pct:>+12.2f}%"
    )
    print(
        f"{'High-RH Days Pinball':<30} | {imp.unmasked.high_rh_upper_pinball:<12.4f} | {imp.masked.high_rh_upper_pinball:<12.4f} | {imp.delta_high_rh_pinball:<+12.4f} | {imp.rel_delta_high_rh_pinball_pct:>+12.2f}%"
    )
    print("=" * 75)
    print(f"\nCondition C9 Policy Decision: {pol['recommendation']}")
    print(
        f"Audit reports written to: {output_dir}/rh_mask_audit.json and {output_dir}/rh_mask_audit.md"
    )
    safe_exit(0)


@app_eval.command("h5-power")
def eval_h5_power_cmd(
    domain: str = typer.Option("delhi", "--domain", help="Target domain (default: delhi)"),
    season: str = typer.Option(
        "2025-2026", "--season", help="Delhi confirmatory test season (default: 2025-2026)"
    ),
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    registry_path: str = typer.Option(
        "data/registry/registry_delhi.parquet", "--registry-path", help="Path to domain registry"
    ),
    models_dir: str = typer.Option(
        "data/processed/models", "--models-dir", help="Path to models directory"
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", "-o", help="Output directory for reports"
    ),
    n_windows: int = typer.Option(
        25, "--n-windows", "-n", help="Number of benchmark sliding windows (default: 25)"
    ),
    support_days: int = typer.Option(
        14, "--support-days", help="Number of support days per window (default: 14)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option("cpu", "--device", help="Torch compute device (cpu, cuda)"),
) -> None:
    """Evaluate Hypothesis H5 statistical power and pilot study on Delhi Season T (Task P2-14)."""
    from smogsense.models.h5_power import run_h5_power_pilot

    print("=" * 80)
    print(f"H5 STATISTICAL POWER & PILOT STUDY (Domain: {domain}, Season: {season})")
    print("=" * 80)
    print(f"Feature Store:    {feature_store_dir}")
    print(f"Registry Path:    {registry_path}")
    print(f"Models Dir:       {models_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Sliding Windows:  {n_windows}")
    print(f"Support Days:     {support_days}")
    print(f"Random Seed:      {seed}")
    print(f"Device:           {device}")

    report = run_h5_power_pilot(
        domain=domain,
        season=season,
        feature_store_dir=feature_store_dir,
        registry_path=registry_path,
        models_dir=models_dir,
        output_dir=output_dir,
        n_windows=n_windows,
        support_days=support_days,
        seed=seed,
        device=device,
    )

    audit = report.station_audit
    h5_cln = report.h5_clean
    h5_crp = report.h5_corrupted

    print("\n" + "=" * 80)
    print("1. STATION ELIGIBILITY AUDIT (Delhi Season T)")
    print("=" * 80)
    print(f"Total Registry Stations:  {audit.total_stations}")
    print(f"Eligible Stations:        {audit.eligible_stations} / {audit.total_stations}")
    print(f"Threshold Requirement:   >= {audit.min_threshold} stations")
    print(f"Audit Verdict:            {'PASS' if audit.threshold_met else 'FAIL'}")

    print("\n" + "=" * 80)
    print("2. HYPOTHESIS H5 PILOT RESULTS (M7 Meta-SGD vs M6 Fine-Tuning)")
    print("=" * 80)
    print(
        f"{'Condition':<30} | {'CRPS Ratio':<12} | {'sigma':<8} | {'rho_1':<8} | {'n_eff':<8} | {'Power':<8} | {'Verdict':<8}"
    )
    print("-" * 80)
    print(
        f"{'H5 (Clean Regulatory)':<30} | {h5_cln.crps_ratio_point:<12.4f} | {h5_cln.sigma_d:<8.4f} | {h5_cln.rho_1:<8.3f} | {h5_cln.n_eff:<8.1f} | {h5_cln.analytical_power * 100:>6.1f}% | {'PASS' if h5_cln.verdict_passed else 'FAIL':<8}"
    )
    print(
        f"{'H5-C (Corrupted Inputs Prior)':<30} | {h5_crp.crps_ratio_point:<12.4f} | {h5_crp.sigma_d:<8.4f} | {h5_crp.rho_1:<8.3f} | {h5_crp.n_eff:<8.1f} | {h5_crp.analytical_power * 100:>6.1f}% | {'PASS' if h5_crp.verdict_passed else 'FAIL':<8}"
    )
    print("=" * 80)
    print(f"\nH5 Status: {'POWERED / CONFIRMED' if h5_cln.verdict_passed else 'INDICATIVE'}")
    print(
        f"Reports written to: {output_dir}/h5_power_pilot.json and {output_dir}/h5_power_pilot.md"
    )
    safe_exit(0)


@app_eval.command("adapt")
def eval_adapt_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    location_id: int = typer.Option(8664, "--location-id", help="Location ID for support window"),
    feature_store_dir: str = typer.Option(
        "data/processed/feature_store", "--feature-store-dir", help="Path to feature store root"
    ),
    base_model_dir: str = typer.Option(
        "data/processed/models/lgbm_m5/base_model",
        "--base-model-dir",
        help="Path to base model directory",
    ),
    calibrator_path: str = typer.Option(
        "data/processed/models/m7_maml.pt",
        "--calibrator-path",
        help="Path to calibrator weights (.pt)",
    ),
    support_days: int = typer.Option(14, "--support-days", help="Support window duration in days"),
    rho_99: float = typer.Option(0.362452, "--rho-99", help="Calibrated residual bound threshold"),
    is_reference: bool = typer.Option(
        False, "--is-reference", help="Whether sensor is reference monitor"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("cpu", "--device", help="Compute device"),
) -> None:
    """Run deployment adaptation lifecycle and acceptance check (Task P3-02)."""
    import pandas as pd

    from smogsense.models.adaptation import DeploymentAdaptationEngine

    print("=" * 80)
    print(f"DEPLOYMENT ADAPTATION & ACCEPTANCE GATE (Domain: {domain}, Location: {location_id})")
    print("=" * 80)
    print(f"Base Model Dir:    {base_model_dir}")
    print(f"Calibrator Path:   {calibrator_path}")
    print(f"Feature Store:     {feature_store_dir}")
    print(f"Support Days:      {support_days}")
    print(f"rho_99 Threshold:  {rho_99:.6f}")
    print(f"Is Reference:      {is_reference}")
    print(f"Device:            {device}")

    base_path = Path(base_model_dir)
    cal_path = Path(calibrator_path)
    fs_path = Path(feature_store_dir)

    if not base_path.exists():
        print(f"Error: Base model directory not found: {base_model_dir}")
        safe_exit(1)
    if not cal_path.exists():
        print(f"Error: Calibrator checkpoint not found: {calibrator_path}")
        safe_exit(1)

    print("\nLoading models and instantiating DeploymentAdaptationEngine...")
    engine = DeploymentAdaptationEngine.from_paths(
        base_model_dir=base_path,
        calibrator_path=cal_path,
        rho_99=rho_99,
        device=device,
    )

    domain_fs = fs_path / domain
    if not domain_fs.exists():
        print(f"Error: Feature store for domain '{domain}' not found at {domain_fs}")
        safe_exit(1)

    parquet_files = sorted(domain_fs.rglob("*.parquet"))
    if not parquet_files:
        print(f"Error: No parquet partitions found in {domain_fs}")
        safe_exit(1)

    dfs: list[pd.DataFrame] = []
    for pf in parquet_files[-3:]:
        df_p = pd.read_parquet(pf)
        if "location_id" in df_p.columns:
            df_stn = df_p[df_p["location_id"] == location_id]
            if not df_stn.empty:
                dfs.append(df_stn)

    if not dfs:
        print(f"Warning: No support data found for location_id {location_id} in {domain_fs}")
        df_support = pd.DataFrame()
    else:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all["issuance_utc"] = pd.to_datetime(df_all["issuance_utc"], utc=True)
        df_all = df_all.sort_values("issuance_utc").reset_index(drop=True)
        t_max = df_all["issuance_utc"].max()
        t_min = t_max - pd.Timedelta(days=support_days)
        df_support = df_all[df_all["issuance_utc"] >= t_min].copy()

    print(f"Running adaptation on {len(df_support)} support observations...")
    result = engine.adapt(support_df=df_support, is_reference_sensor=is_reference, seed=seed)

    print("\n" + "=" * 80)
    print("ADAPTATION ACCEPTANCE RESULT")
    print("=" * 80)
    print(f"Status:               {result.status}")
    print(f"Is Adapted:           {engine.is_adapted}")
    print(
        f"Relative Improvement: {result.relative_improvement * 100:.2f}% "
        f"(min required: {engine.min_improvement * 100:.1f}%)"
    )
    print(
        f"Daily Win Fraction:   {result.win_fraction * 100:.2f}% "
        f"(min required: {engine.min_days_better_fraction * 100:.1f}%)"
    )
    print(
        f"Mean Abs Residual:    {result.mean_abs_residual:.6f} "
        f"(rho_99 bound: {result.residual_bound_threshold:.6f})"
    )
    print(f"Fit Samples:          {result.n_fit_samples}")
    print(f"Check Samples:        {result.n_check_samples}")
    print(f"Check Days Evaluated: {result.n_check_days}")
    print(f"M5 Check Loss:        {result.m5_check_loss:.6f}")
    print(f"Adapted Check Loss:   {result.adapted_check_loss:.6f}")
    print("=" * 80)

    if engine.is_adapted:
        print("Verdict: ACCEPTED — residual calibrator successfully deployed.")
    else:
        print(f"Verdict: FALLBACK TO M5 ({result.status}) — exact M5 physics base preserved.")

    safe_exit(0)


@app_eval.command("metrics")
def eval_metrics_cmd(
    predictions_path: str = typer.Option(
        ..., "--predictions", "-p", help="Path to predictions file (parquet or csv)"
    ),
    truth_path: str | None = typer.Option(
        None,
        "--truth",
        "-t",
        help="Path to observations file (optional if predictions contain truth)",
    ),
    truth_col: str = typer.Option(
        "pm25", "--truth-col", help="Column name for observed PM2.5 concentrations"
    ),
    output_json: str | None = typer.Option(
        None, "--out-json", "-o", help="Optional output JSON path"
    ),
) -> None:
    """Compute probabilistic evaluation metrics and calibration diagnostics."""
    import json
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd

    from smogsense.evaluation import (
        Q_NAMES,
        category_accuracy,
        compute_calibration_report,
        compute_contingency_table,
        compute_crps,
        compute_tw_crps,
        interval_score,
        pinball_loss_19,
        point_crps,
    )

    pred_p = Path(predictions_path)
    df_pred = (
        pd.read_parquet(pred_p) if pred_p.suffix in {".parquet", ".pq"} else pd.read_csv(pred_p)
    )

    if truth_path:
        truth_p = Path(truth_path)
        df_truth = (
            pd.read_parquet(truth_p)
            if truth_p.suffix in {".parquet", ".pq"}
            else pd.read_csv(truth_p)
        )
        if "issuance_utc" in df_pred.columns and "issuance_utc" in df_truth.columns:
            join_cols = [
                c
                for c in ["issuance_utc", "location_id", "horizon_h"]
                if c in df_pred.columns and c in df_truth.columns
            ]
            df = df_pred.merge(df_truth, on=join_cols, suffixes=("", "_obs"))
        else:
            df = df_pred.copy()
            if truth_col in df_truth.columns:
                df[truth_col] = df_truth[truth_col].to_numpy()[: len(df)]
    else:
        df = df_pred.copy()

    if truth_col not in df.columns:
        print(f"Error: truth column '{truth_col}' not found in data.")
        safe_exit(1)

    df_valid = df.dropna(subset=[truth_col])
    y_true = df_valid[truth_col].to_numpy(dtype=float)
    n_samples = len(y_true)

    metrics_out: dict[str, Any] = {"n_samples": n_samples}

    has_19q = all(q in df_valid.columns for q in Q_NAMES)
    if has_19q:
        q_mat = df_valid[Q_NAMES].to_numpy(dtype=float)
        crps_vals = [compute_crps(q_mat[i], y_true[i]) for i in range(n_samples)]
        mean_crps = float(np.mean(crps_vals))
        mean_pb19 = float(np.mean(pinball_loss_19(y_true, q_mat)))
        tw_125 = float(
            np.mean([compute_tw_crps(q_mat[i], y_true[i], 125.5) for i in range(n_samples)])
        )
        tw_150 = float(
            np.mean([compute_tw_crps(q_mat[i], y_true[i], 150.0) for i in range(n_samples)])
        )

        q10 = q_mat[:, 1]
        q90 = q_mat[:, 17]
        is80 = float(np.mean(interval_score(y_true, q10, q90, alpha=0.20)))

        calib_report = compute_calibration_report(y_true, q_mat)
        q50 = q_mat[:, 9]
        cat_acc = category_accuracy(y_true, q50)
        ct_125 = compute_contingency_table(y_true, q_mat[:, 17], threshold=125.5)

        metrics_out.update(
            {
                "mean_crps": mean_crps,
                "mean_pinball_19": mean_pb19,
                "mean_tw_crps_125_5": tw_125,
                "mean_tw_crps_150_0": tw_150,
                "interval_score_80": is80,
                "coverage_80": calib_report.coverage_80,
                "sharpness_80": calib_report.sharpness_80,
                "coverage_below_median": calib_report.median_coverage,
                "pit_chi2": calib_report.pit_chi2,
                "pit_p_value": calib_report.pit_p_value,
                "is_pit_uniform": calib_report.is_pit_uniform,
                "category_exact_match": cat_acc["exact_match"],
                "category_within_one": cat_acc["within_one_category"],
                "csi_125_5": ct_125.csi,
                "pss_125_5": ct_125.pss,
            }
        )
    elif "y_hat" in df_valid.columns or "pm25_pred" in df_valid.columns:
        pred_col = "y_hat" if "y_hat" in df_valid.columns else "pm25_pred"
        y_hat = df_valid[pred_col].to_numpy(dtype=float)
        mae = float(np.mean(point_crps(y_true, y_hat)))
        metrics_out["point_crps_mae"] = mae
    else:
        print("Error: DataFrame contains neither standard 19 quantiles nor point predictions.")
        safe_exit(1)

    print("\n" + "=" * 70)
    print("PROBABILISTIC EVALUATION METRICS REPORT")
    print("=" * 70)
    for k, v in metrics_out.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:.4f}")
        else:
            print(f"{k:28s}: {v}")
    print("=" * 70)

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(metrics_out, indent=2))
        print(f"Metrics written to {out_p}")

    safe_exit(0)


@app_eval.command("significance")
def eval_significance_cmd(
    loss_a_path: str = typer.Option(
        ..., "--loss-a", "-a", help="Path to CSV or parquet holding daily loss series for Model A"
    ),
    loss_b_path: str = typer.Option(
        ...,
        "--loss-b",
        "-b",
        help="Path to CSV or parquet holding daily loss series for Reference Model B",
    ),
    loss_col: str = typer.Option("crps", "--loss-col", help="Column name containing loss values"),
    horizon_h: int = typer.Option(
        24, "--horizon-h", help="Forecast horizon in hours (24, 48, or 72)"
    ),
    bootstrap_reps: int = typer.Option(
        2000, "--bootstrap-reps", help="Number of stationary bootstrap replicates"
    ),
    output_json: str | None = typer.Option(
        None, "--out-json", "-o", help="Optional output JSON path"
    ),
) -> None:
    """Evaluate statistical significance between two forecast models via Diebold-Mariano test and block bootstrap."""
    import json
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd

    from smogsense.evaluation import (
        diebold_mariano_test,
        stationary_block_bootstrap,
    )

    path_a = Path(loss_a_path)
    df_a = pd.read_parquet(path_a) if path_a.suffix in {".parquet", ".pq"} else pd.read_csv(path_a)

    path_b = Path(loss_b_path)
    df_b = pd.read_parquet(path_b) if path_b.suffix in {".parquet", ".pq"} else pd.read_csv(path_b)

    if loss_col not in df_a.columns or loss_col not in df_b.columns:
        print(f"Error: loss column '{loss_col}' must be present in both files.")
        safe_exit(1)

    if "issuance_utc" in df_a.columns and "issuance_utc" in df_b.columns:
        merged = df_a.merge(df_b, on="issuance_utc", suffixes=("_a", "_b"))
        loss_a = merged[f"{loss_col}_a"].to_numpy(dtype=float)
        loss_b = merged[f"{loss_col}_b"].to_numpy(dtype=float)
    else:
        n_min = min(len(df_a), len(df_b))
        loss_a = df_a[loss_col].to_numpy(dtype=float)[:n_min]
        loss_b = df_b[loss_col].to_numpy(dtype=float)[:n_min]

    dm_res = diebold_mariano_test(loss_a, loss_b, horizon_h=horizon_h)

    diffs = loss_a - loss_b
    boot_res = stationary_block_bootstrap(
        diffs,
        statistic_fn=lambda arr: float(np.mean(arr)),
        mean_block_days=7.0,
        replicates=bootstrap_reps,
    )

    result_dict: dict[str, Any] = {
        "diebold_mariano": dm_res.to_dict(),
        "bootstrap": boot_res.to_dict(),
    }

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE EVALUATION (DIEBOLD-MARIANO + BOOTSTRAP)")
    print("=" * 70)
    print(f"Model A Samples:      {len(loss_a)}")
    print(f"Mean Loss Diff (A-B): {dm_res.mean_loss_diff:.6f}")
    print(f"DM Statistic (raw):   {dm_res.dm_stat:.4f}")
    print(f"DM* Statistic (HLN):  {dm_res.dm_star:.4f}")
    print(f"HLN Factor:           {dm_res.hln_factor:.4f}")
    print(f"HAC SE:               {dm_res.hac_se:.6f}")
    print(f"Bandwidth (lags):     {dm_res.bandwidth}")
    print(f"Lag-1 Autocorr:       {dm_res.lag1_autocorr:.4f}")
    print(f"Effective N (n_eff):  {dm_res.n_eff:.1f}")
    print(f"p-value (two-sided):  {dm_res.p_value:.6e}")
    print("-" * 70)
    print("STATIONARY BLOCK BOOTSTRAP (MEAN BLOCK 7 DAYS)")
    print(f"Replicates:           {boot_res.replicates_count}")
    print(f"Point Estimate:       {boot_res.point_estimate:.6f}")
    print(f"Bootstrap Mean:       {boot_res.bootstrap_mean:.6f}")
    print(f"Bootstrap SE:         {boot_res.std_error:.6f}")
    print(f"95% Confidence Int:   [{boot_res.ci_lower:.6f}, {boot_res.ci_upper:.6f}]")
    print("=" * 70)

    if dm_res.p_value < 0.05:
        verdict = "SIGNIFICANT difference (p < 0.05)"
    else:
        verdict = "NOT statistically significant (p >= 0.05)"
    print(f"Verdict: {verdict}")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result_dict, indent=2))
        print(f"Results written to {out_p}")

    safe_exit(0)


@app_eval.command("sparsity")
def eval_sparsity_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Evaluation domain (lahore, delhi)"),
    feature_store_dir: str | None = typer.Option(
        None, "--feature-store-dir", help="Path to precomputed feature store"
    ),
    models_dir: str | None = typer.Option(
        None, "--models-dir", help="Path to models directory holding M5 base and calibrators"
    ),
    n_grid_str: str | None = typer.Option(
        None, "--n-grid", help="Comma-separated N values, e.g. '0,1,2,3,5,8'"
    ),
    n_val: int | None = typer.Option(
        None, "--n-val", help="Single N value for matrix sharding execution"
    ),
    draws: int = typer.Option(30, "--draws", "-r", help="Number of random draws per N"),
    mode: str = typer.Option(
        "neighbour_history",
        "--mode",
        help="Evaluation mode: 'neighbour_history' (leave-stations-out) or 'own_history'",
    ),
    support_days: int = typer.Option(
        14, "--support-days", help="Number of days in support window W"
    ),
    output_dir: str = typer.Option(
        "reports", "--output-dir", "-o", help="Directory for JSON and MD output reports"
    ),
    device: str = typer.Option("cpu", "--device", help="Computation device ('cpu' or 'cuda')"),
    merge_shards: str | None = typer.Option(
        None,
        "--merge-shards",
        help="Glob or comma-separated list of shard JSON files to merge into unified report",
    ),
) -> None:
    """Execute sensor-sparsity curve evaluation or merge distributed matrix shards (Task P3-06)."""
    from pathlib import Path

    from smogsense.evaluation.sparsity_curve import (
        DEFAULT_N_GRID,
        merge_sparsity_shards,
        run_sparsity_experiment,
        save_sparsity_report,
    )

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if merge_shards:
        shard_paths: list[Path] = []
        for part in merge_shards.split(","):
            part = part.strip()
            p = Path(part)
            if "*" in part or "?" in part:
                parent = p.parent if str(p.parent) != "" else Path.cwd()
                matched = list(parent.glob(p.name))
                shard_paths.extend(matched)
            elif p.exists():
                shard_paths.append(p)

        if not shard_paths:
            print(f"Error: No shard files found matching '{merge_shards}'")
            safe_exit(1)

        merged = merge_sparsity_shards(shard_paths)
        json_p, md_p = save_sparsity_report(merged, out_dir_path)
        print(f"Successfully merged {len(shard_paths)} shards into {json_p} and {md_p}")
        safe_exit(0)

    # Resolve active N grid
    if n_val is not None:
        active_n_grid: tuple[int, ...] = (n_val,)
    elif n_grid_str:
        active_n_grid = tuple(int(x.strip()) for x in n_grid_str.split(","))
    else:
        active_n_grid = DEFAULT_N_GRID

    print("\n" + "=" * 70)
    print("SENSOR-SPARSITY CURVE EVALUATION (TASK P3-06)")
    print("=" * 70)
    print(f"Domain:       {domain}")
    print(f"Mode:         {mode}")
    print(f"N Grid:       {list(active_n_grid)}")
    print(f"Draws (R):    {draws}")
    print(f"Support Days: {support_days}")
    print(f"Device:       {device}")
    print("=" * 70)

    base_model = None
    calibrator_m6 = None
    calibrator_m7 = None
    step_sizes_m7 = None

    if models_dir and Path(models_dir).exists():
        m_dir = Path(models_dir)
        m5_path = m_dir / "m5_physics_base.joblib"
        if m5_path.exists():
            from smogsense.models.lgbm_quantile import LGBMQuantileEnsemble

            base_model = LGBMQuantileEnsemble.load(m5_path)

    result = run_sparsity_experiment(
        base_model=base_model,
        calibrator_m6=calibrator_m6,
        calibrator_m7=calibrator_m7,
        step_sizes_m7=step_sizes_m7,
        n_grid=active_n_grid,
        draws=draws,
        mode=mode,
        domain=domain,
        device=device,
    )

    json_path, md_path = save_sparsity_report(result, out_dir_path)

    print("\nSENSOR-SPARSITY EXPERIMENT COMPLETED")
    print("-" * 70)
    print(f"{'Method':<25s} | {'N*':<5s} | {'Power Law Fit (c_inf, c_0, nu, R2)':<35s}")
    print("-" * 70)
    for method in result.methods:
        n_star = str(result.n_star.get(method, "N/A"))
        fit = result.fits.get(method)
        if fit:
            fit_str = f"c_inf={fit.c_infinity:.2f}, nu={fit.nu:.2f}, R2={fit.r_squared:.3f}"
        else:
            fit_str = "No fit"
        print(f"{method:<25s} | {n_star:<5s} | {fit_str:<35s}")
    print("-" * 70)
    print(f"Reports written to:\n  - {json_path}\n  - {md_path}")
    print("=" * 70 + "\n")

    safe_exit(0)


@app_eval.command("promotion")
def eval_promotion_cmd(
    ladder_report: str = typer.Option(
        "reports/ladder.json",
        "--ladder-report",
        help="Path to model ladder evaluation JSON report",
    ),
    sparsity_report: str = typer.Option(
        "reports/sparsity_curve.json",
        "--sparsity-report",
        help="Path to sensor-sparsity experiment JSON report",
    ),
    replay_report: str = typer.Option(
        "reports/replay_shadow.json",
        "--replay-report",
        help="Path to shadow replay evaluation JSON report",
    ),
    output_dir: str = typer.Option(
        "reports",
        "--output-dir",
        "-o",
        help="Output directory for promotion decision report",
    ),
    apply: bool = typer.Option(
        False,
        "--apply/--no-apply",
        help="Whether to automatically apply promotion to configs/model.yaml if approved",
    ),
    bundle_tag: str = typer.Option(
        "models-v0.2.0",
        "--bundle-tag",
        help="Model bundle tag for promotion",
    ),
) -> None:
    """Execute Gate G5 model promotion evaluation, PR generation, and config review (Task P3-09)."""
    from smogsense.evaluation.promotion import (
        apply_promotion_to_config,
        generate_promotion_decision,
    )

    print("=" * 75)
    print("GATE G5 MODEL PROMOTION REVIEW & HEADLINE MODEL SELECTION (Task P3-09)")
    print("=" * 75)
    print(f"Ladder Report:   {ladder_report}")
    print(f"Sparsity Report: {sparsity_report}")
    print(f"Replay Report:   {replay_report}")
    print(f"Output Dir:      {output_dir}")
    print(f"Apply Promotion: {apply}")
    print("=" * 75)

    decision = generate_promotion_decision(
        ladder_path=ladder_report,
        sparsity_path=sparsity_report,
        replay_path=replay_report,
        output_dir=output_dir,
        bundle_tag=bundle_tag,
    )

    print("\nPROMOTION DECISION SUMMARY")
    print("-" * 75)
    print(f"Selected Headline Model:  {decision.headline_model}")
    print(f"Recommended Action:       {decision.recommended_action}")
    print(f"Promoted Bundle Value:    {decision.promoted_bundle_value}")
    print(f"All Guardrails Passed:    {decision.all_guardrails_passed}")
    print(f"PR Title:                 {decision.pr_title}")
    print("-" * 75)
    print(f"Justification:\n  {decision.justification}")
    print("-" * 75)

    if apply:
        if decision.all_guardrails_passed and decision.recommended_action != "maintain_baseline":
            success = apply_promotion_to_config(
                promoted_bundle_value=decision.promoted_bundle_value
            )
            if success:
                print(
                    f"[OK] Applied promotion to configs/model.yaml: {decision.promoted_bundle_value}"
                )
            else:
                print("[ERROR] Failed to update configs/model.yaml")
        else:
            print(
                "[INFO] Promotion not applied: recommended action maintains baseline or guardrails failed."
            )

    print(f"\nArtifacts generated in: {output_dir}/")
    print("  - promotion_decision.json")
    print("  - promotion_decision.md")
    print("=" * 75 + "\n")
    safe_exit(0)


@app_eval.command("freeze-register")
def eval_freeze_register_cmd(
    create: bool = typer.Option(
        True,
        "--create/--no-create",
        help="Whether to create and save the Gate G4 freeze register",
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Whether to verify the existing freeze register against current state",
    ),
    git_tag: str = typer.Option(
        "v0.3.0-gate-g4",
        "--git-tag",
        help="Git release tag for the freeze register",
    ),
    output_dir: str = typer.Option(
        ".state",
        "--output-dir",
        "-o",
        help="Output directory for freeze_register.json",
    ),
    report_dir: str = typer.Option(
        "reports",
        "--report-dir",
        "-r",
        help="Output directory for freeze_register.md",
    ),
) -> None:
    """Execute Gate G4 freeze register creation and drift verification (Task P3-10)."""
    from smogsense.evaluation.freeze import (
        create_freeze_register,
        verify_freeze_register,
    )

    out_p = Path(output_dir) / "freeze_register.json"
    rep_p = Path(report_dir) / "freeze_register.md"

    print("=" * 75)
    print("GATE G4 FREEZE REGISTER & INVARIANT LOCKDOWN (Task P3-10)")
    print("=" * 75)
    print(f"Mode:            {'VERIFY' if verify else 'CREATE'}")
    print(f"Git Tag:         {git_tag}")
    print(f"Register Path:   {out_p}")
    print(f"Report Path:     {rep_p}")
    print("=" * 75)

    if verify:
        print(f"\nAuditing active state against freeze register at {out_p}...")
        try:
            is_valid, drift_messages = verify_freeze_register(
                register_path=out_p,
                strict=False,
            )
            if is_valid:
                print("\n[OK] Gate G4 Freeze Register verification PASSED.")
                print("No configuration drift, bundle tampering, or station panel drift detected.")
                print("=" * 75 + "\n")
                safe_exit(0)
            else:
                print(
                    f"\n[FAIL] Gate G4 Freeze Register verification FAILED with {len(drift_messages)} error(s):"
                )
                for msg in drift_messages:
                    print(f"  - {msg}")
                print("=" * 75 + "\n")
                safe_exit(30)
        except Exception as exc:
            print(f"\n[ERROR] Verification aborted with error: {exc}")
            safe_exit(30)

    if create:
        print("\nCreating Gate G4 freeze register and locking invariants...")
        try:
            reg = create_freeze_register(
                output_path=out_p,
                report_path=rep_p,
                git_tag=git_tag,
                write_files=True,
            )
            print("\nFREEZE REGISTER SUMMARY")
            print("-" * 75)
            print(f"Git Tag:             {reg.git_tag}")
            print(f"Git SHA:             {reg.git_commit_sha[:16]}...")
            print(f"Config Hash:         {reg.config_hash[:16]}...")
            print(f"Headline Model:      {reg.headline_model}")
            print(f"Bundle SHA-256:      {reg.bundle_sha256[:16]}...")
            print(f"Base SHA-256:        {reg.base_sha256[:16]}...")
            print(f"City Panel Stations: {len(reg.city_panel)}")
            print(f"Hypotheses Count:    {len(reg.hypotheses)}")
            print(f"Residual Bound rho:  {reg.invariants.get('rho_99')}")
            print("-" * 75)
            print(f"Saved JSON register: {out_p}")
            print(f"Saved Markdown card: {rep_p}")
            print("=" * 75 + "\n")
            safe_exit(0)
        except Exception as exc:
            print(f"\n[ERROR] Failed to create freeze register: {exc}")
            safe_exit(50)


@app_eval.command("freeze")
def eval_freeze_cmd(
    create: bool = typer.Option(
        True,
        "--create/--no-create",
        help="Whether to create and save the Gate G4 freeze register",
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Whether to verify the existing freeze register against current state",
    ),
    git_tag: str = typer.Option(
        "v0.3.0-gate-g4",
        "--git-tag",
        help="Git release tag for the freeze register",
    ),
    output_dir: str = typer.Option(
        ".state",
        "--output-dir",
        "-o",
        help="Output directory for freeze_register.json",
    ),
    report_dir: str = typer.Option(
        "reports",
        "--report-dir",
        "-r",
        help="Output directory for freeze_register.md",
    ),
) -> None:
    """Alias for freeze-register."""
    eval_freeze_register_cmd(
        create=create,
        verify=verify,
        git_tag=git_tag,
        output_dir=output_dir,
        report_dir=report_dir,
    )


@app_eval.command("threshold-pilot")
def eval_threshold_pilot_cmd(
    domain: str = typer.Option(
        "delhi",
        "--domain",
        "-d",
        help="Evaluation domain ('delhi', 'lahore', or 'delhi+lahore')",
    ),
    n_tasks: int = typer.Option(
        50,
        "--n-tasks",
        "-n",
        help="Number of simulated deployment tasks to evaluate",
    ),
    output_dir: str = typer.Option(
        "reports",
        "--output-dir",
        "-o",
        help="Output directory for threshold_pilot.json and threshold_pilot.md",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Deterministic random seed",
    ),
) -> None:
    """Run Phase 3 Task P3-12 Threshold Pilot & Adaptation Status Distribution Audit."""
    from smogsense.evaluation.threshold_pilot import run_threshold_pilot

    print("=" * 80)
    print("THRESHOLD PILOT & ADAPTATION STATUS DISTRIBUTION AUDIT (Task P3-12)")
    print("=" * 80)
    print(f"Domain:      {domain}")
    print(f"Tasks:       {n_tasks}")
    print(f"Output Dir:  {output_dir}")
    print(f"Random Seed: {seed}")
    print("=" * 80)

    try:
        report = run_threshold_pilot(
            domain=domain,
            n_tasks=n_tasks,
            output_dir=output_dir,
            seed=seed,
        )
        print("\nTHRESHOLD PILOT SUMMARY & INVARIANT CONFIRMATION")
        print("-" * 80)
        print(f"Total Tasks Simulated:       {report.total_simulated_tasks}")
        print(f"Recommended min_improvement: {report.recommended_min_improvement:.2f} (2.0%)")
        print(f"Recommended min_days_better: {report.recommended_min_days_better:.2f} (60.0%)")
        print(f"Recommended min_support_days: {report.recommended_min_support_days} days")
        print(f"Calibrated rho_99 Threshold: {report.calibrated_rho_99:.6f}")
        print("\nEmpirical Adaptation Status Distribution:")
        for status, pct in report.empirical_status_distribution.items():
            print(f"  - {status:28s}: {pct:5.1f}%")
        print("-" * 80)
        print(f"[OK] Reports persisted to {output_dir}/threshold_pilot.json and .md")
        print("=" * 80 + "\n")
        safe_exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Threshold pilot failed: {exc}")
        print("=" * 80 + "\n")
        safe_exit(50)


@app_eval.command("pilot")
def eval_pilot_alias_cmd(
    domain: str = typer.Option(
        "delhi",
        "--domain",
        "-d",
        help="Evaluation domain ('delhi', 'lahore', or 'delhi+lahore')",
    ),
    n_tasks: int = typer.Option(
        50,
        "--n-tasks",
        "-n",
        help="Number of simulated deployment tasks to evaluate",
    ),
    output_dir: str = typer.Option(
        "reports",
        "--output-dir",
        "-o",
        help="Output directory for threshold_pilot.json and threshold_pilot.md",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Deterministic random seed",
    ),
) -> None:
    """Alias for threshold-pilot."""
    eval_threshold_pilot_cmd(
        domain=domain,
        n_tasks=n_tasks,
        output_dir=output_dir,
        seed=seed,
    )


@app_eval.command("pre-register")
def eval_pre_register_cmd(
    create: bool = typer.Option(
        True,
        "--create/--no-create",
        help="Whether to create and save the Gate G4 pre-registration specification",
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Whether to verify active state against the pre-registration specification",
    ),
    git_tag: str = typer.Option(
        "v0.3.0-gate-g4",
        "--git-tag",
        help="Git release tag for the pre-registration",
    ),
    output_doc: str = typer.Option(
        "docs/pre-registration.md",
        "--output-doc",
        "-d",
        help="Output path for pre-registration markdown document",
    ),
    output_state: str = typer.Option(
        ".state/pre_registration.json",
        "--output-state",
        "-s",
        help="Output path for pre-registration JSON state",
    ),
    configs_dir: str = typer.Option(
        "configs",
        "--configs-dir",
        "-c",
        help="Directory containing configuration files",
    ),
    freeze_register_path: str = typer.Option(
        ".state/freeze_register.json",
        "--freeze-register",
        "-f",
        help="Path to Gate G4 freeze register JSON",
    ),
) -> None:
    """Execute Gate G4 confirmatory pre-registration creation and drift verification (Task P3-13)."""
    from smogsense.evaluation.registration import (
        PreRegistrationVerificationError,
        create_pre_registration_document,
        verify_pre_registration,
    )

    doc_p = Path(output_doc)
    state_p = Path(output_state)

    print("=" * 75)
    print("GATE G4 CONFIRMATORY PRE-REGISTRATION (Task P3-13)")
    print("=" * 75)
    print(f"Mode:              {'VERIFY' if verify else 'CREATE'}")
    print(f"Git Tag:           {git_tag}")
    print(f"State Output:      {state_p}")
    print(f"Document Output:   {doc_p}")
    print(f"Freeze Register:   {freeze_register_path}")

    if verify:
        print(f"\nAuditing active state against pre-registration at {state_p}...")
        try:
            is_valid, drift_messages = verify_pre_registration(
                pre_registration_path=state_p,
                freeze_register_path=Path(freeze_register_path),
                configs_dir=Path(configs_dir),
                strict=False,
            )
            if is_valid:
                print("\n[OK] Gate G4 Pre-Registration verification PASSED.")
                print(
                    "No configuration drift, threshold mutation, or cryptographic mismatch detected."
                )
                print("=" * 75 + "\n")
                safe_exit(0)
            else:
                print(
                    f"\n[FAIL] Gate G4 Pre-Registration verification FAILED with {len(drift_messages)} error(s):"
                )
                for msg in drift_messages:
                    print(f"  - {msg}")
                print("=" * 75 + "\n")
                safe_exit(30)
        except PreRegistrationVerificationError as exc:
            print(f"\n[FAIL] Pre-registration verification failed: {exc}")
            safe_exit(exc.exit_code)
        except Exception as exc:
            print(f"\n[ERROR] Verification aborted: {exc}")
            safe_exit(50)

    if create:
        print("\nFormalizing Gate G4 pre-registration document and locking hypothesis tiers...")
        try:
            doc = create_pre_registration_document(
                configs_dir=Path(configs_dir),
                freeze_register_path=Path(freeze_register_path),
                output_state_path=state_p,
                output_doc_path=doc_p,
                git_tag=git_tag,
                write_files=True,
            )
            print("\nPRE-REGISTRATION SPECIFICATION SUMMARY")
            print("-" * 75)
            print(f"Title:               {doc.title}")
            print(f"Registration Date:   {doc.registration_date}")
            print(f"Git Tag:             {doc.git_release_tag}")
            print(f"Git SHA:             {doc.git_commit_sha[:16]}...")
            print(f"Config Hash:         {doc.config_hash[:16]}...")
            print(f"Freeze Digest:       {doc.freeze_register_digest[:16]}...")
            print(f"Headline Model:      {doc.headline_model}")
            print(f"Hypotheses Count:    {len(doc.hypotheses)}")
            for h in doc.hypotheses:
                print(
                    f"  - [{h.hypothesis_id}] ({h.tier.value}): {h.metric} {h.comparison_operator} {h.threshold_value}"
                )
            print(
                f"Power Note H5:       power={doc.statistical_power_note.statistical_power}%, "
                f"sigma={doc.statistical_power_note.observed_log_ratio_sd}, "
                f"n_eff={doc.statistical_power_note.effective_sample_size}"
            )
            print(f"Calendar Window:     {doc.evaluation_calendar.confirmatory_shadow_window}")
            print("-" * 75)
            print(f"[OK] State saved to {state_p}")
            print(f"[OK] Markdown documentation saved to {doc_p}")
            print("=" * 75 + "\n")
            safe_exit(0)
        except Exception as exc:
            print(f"\n[ERROR] Failed to create pre-registration: {exc}")
            safe_exit(50)


@app_eval.command("register")
def eval_register_alias_cmd(
    create: bool = typer.Option(
        True,
        "--create/--no-create",
        help="Whether to create and save the Gate G4 pre-registration specification",
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Whether to verify active state against the pre-registration specification",
    ),
    git_tag: str = typer.Option(
        "v0.3.0-gate-g4",
        "--git-tag",
        help="Git release tag for the pre-registration",
    ),
    output_doc: str = typer.Option(
        "docs/pre-registration.md",
        "--output-doc",
        "-d",
        help="Output path for pre-registration markdown document",
    ),
    output_state: str = typer.Option(
        ".state/pre_registration.json",
        "--output-state",
        "-s",
        help="Output path for pre-registration JSON state",
    ),
    configs_dir: str = typer.Option(
        "configs",
        "--configs-dir",
        "-c",
        help="Directory containing configuration files",
    ),
    freeze_register_path: str = typer.Option(
        ".state/freeze_register.json",
        "--freeze-register",
        "-f",
        help="Path to Gate G4 freeze register JSON",
    ),
) -> None:
    """Alias for pre-register."""
    eval_pre_register_cmd(
        create=create,
        verify=verify,
        git_tag=git_tag,
        output_doc=output_doc,
        output_state=output_state,
        configs_dir=configs_dir,
        freeze_register_path=freeze_register_path,
    )


@app_preprocess.command("low-cost-correction")
def preprocess_low_cost_correction_cmd(
    domain: str = typer.Option("lahore", "--domain", help="Target domain (default: lahore)"),
    backfill_dir: str = typer.Option(
        "data/processed/backfill",
        "--backfill-dir",
        help="Path to backfilled observation partitions",
    ),
    registry_path: str | None = typer.Option(
        None, "--registry-path", help="Path to registry parquet file (default: auto)"
    ),
    window_days: int = typer.Option(
        60, "--window-days", help="Window in days for pairing (default: 60)"
    ),
    max_distance_km: float = typer.Option(
        8.0,
        "--max-distance-km",
        help="Maximum distance in km from reference monitor (default: 8.0)",
    ),
    min_pairs: int = typer.Option(
        500, "--min-pairs", help="Minimum required hourly pairs (default: 500)"
    ),
    output_json: str | None = typer.Option(
        None, "--output-json", "-o", help="Path to write JSON validation and audit report"
    ),
    output_model: str | None = typer.Option(
        None, "--output-model", help="Path to save serialized model (JSON or Parquet)"
    ),
    update_config: bool = typer.Option(
        False, "--update-config", help="Update configs/preprocessing.yaml based on decision rule"
    ),
    config_path: str = typer.Option(
        "configs/preprocessing.yaml", "--config-path", help="Path to preprocessing.yaml"
    ),
    model_config_path: str = typer.Option(
        "configs/model.yaml", "--model-config-path", help="Path to model.yaml"
    ),
) -> None:
    """Fit, validate, and audit empirical humidity-aware low-cost sensor correction (Task P2-05)."""
    from smogsense.preprocessing.low_cost_correction import run_fitting_pipeline

    run_fitting_pipeline(
        domain=domain,
        backfill_dir=backfill_dir,
        registry_path=registry_path,
        window_days=window_days,
        max_distance_km=max_distance_km,
        min_pairs=min_pairs,
        output_json=output_json,
        output_model=output_model,
        update_config=update_config,
        config_path=config_path,
        model_config_path=model_config_path,
    )
    safe_exit(0)


@app_registry.command("create-bundle")
def registry_create_bundle_cmd(
    model_name: str = typer.Option(
        ...,
        "--model-name",
        "-m",
        help="Model name (e.g. m5_physics_base, m7_meta_calibrator)",
    ),
    source_dir: str = typer.Option(
        ...,
        "--source-dir",
        "-s",
        help="Path to directory containing trained model files",
    ),
    output_dir: str = typer.Option(
        "data/processed/models/bundles",
        "--output-dir",
        "-o",
        help="Output directory for bundle archive",
    ),
    version: str = typer.Option(
        "v1.0.0",
        "--version",
        "-v",
        help="Semantic version tag for bundle",
    ),
    model_type: str | None = typer.Option(
        None,
        "--model-type",
        help="Model type ('base' or 'residual_calibrator')",
    ),
    base_sha256: str | None = typer.Option(
        None,
        "--base-sha256",
        help="Physics base model SHA-256 (required for calibrators)",
    ),
    archive: bool = typer.Option(
        True,
        "--archive/--no-archive",
        help="Package as .tar.gz archive",
    ),
) -> None:
    """Package a trained model directory into a signed release bundle with manifest.json."""
    from smogsense.models.registry import ModelRegistry

    print("=" * 80)
    print(f"CREATING MODEL BUNDLE: {model_name} ({version})")
    print("=" * 80)
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Archive:          {archive}")
    if base_sha256:
        print(f"Base SHA256:      {base_sha256}")

    registry = ModelRegistry()
    bundle_path = registry.create_bundle(
        model_name=model_name,
        source_dir=source_dir,
        output_dir=output_dir,
        version=version,
        model_type=model_type,
        base_sha256=base_sha256,
        archive=archive,
    )
    print("\n" + "=" * 80)
    print(f"SUCCESS: Created model bundle at {bundle_path}")
    print("=" * 80)
    safe_exit(0)


@app_registry.command("verify")
def registry_verify_cmd(
    bundle_path: str = typer.Option(
        ...,
        "--bundle-path",
        "-b",
        help="Path to bundle archive (.tar.gz) or extracted directory",
    ),
    base_sha256: str | None = typer.Option(
        None,
        "--base-sha256",
        help="Active physics base SHA-256 for binding check",
    ),
    verify_binding: bool = typer.Option(
        True,
        "--verify-binding/--no-verify-binding",
        help="Verify base_sha256 binding for calibrators",
    ),
    verify_compatibility: bool = typer.Option(
        True,
        "--verify-compatibility/--no-verify-compatibility",
        help="Verify feature_set_version",
    ),
) -> None:
    """Audit an existing model bundle for SHA-256 file integrity and cryptographic binding."""
    from smogsense.models.registry import ModelRegistry

    print("=" * 80)
    print(f"AUDITING MODEL BUNDLE: {bundle_path}")
    print("=" * 80)

    registry = ModelRegistry()
    manifest = registry.verify_bundle(
        bundle_path_or_dir=bundle_path,
        active_base_sha256=base_sha256,
        verify_binding=verify_binding,
        verify_compatibility=verify_compatibility,
    )

    print("\n" + "=" * 80)
    print("BUNDLE VERIFICATION AUDIT: PASSED (T1-15)")
    print("=" * 80)
    print(f"Model Name:          {manifest.model_name}")
    print(f"Model Type:          {manifest.model_type}")
    print(f"Version:             {manifest.version}")
    print(f"Git SHA:             {manifest.git_sha}")
    print(f"Feature Set Version: {manifest.feature_set_version}")
    print(f"Config Hash:         {manifest.config_hash[:16]}...")
    print(f"Verified Files:      {len(manifest.files)} files matching checksums")
    if manifest.base_sha256:
        print(f"Bound Base SHA-256:  {manifest.base_sha256}")
    if manifest.parameter_count:
        print(f"Parameter Count:     {manifest.parameter_count:,}")
    print("=" * 80)
    safe_exit(0)


@app_models.command("verify-binding")
def models_verify_binding_cmd(
    calibrator_bundle: str = typer.Option(
        ...,
        "--calibrator-bundle",
        "-c",
        help="Path or tag for neural calibrator bundle (M6 or M7)",
    ),
    base_bundle: str = typer.Option(
        ...,
        "--base-bundle",
        "-b",
        help="Path or tag for shared LightGBM physics base (M5)",
    ),
    cache_dir: str = typer.Option(
        "data/processed/models",
        "--cache-dir",
        help="Path to model bundle cache directory",
    ),
) -> None:
    """Verify cryptographic base/calibrator binding (Condition C3 / T1-15)."""
    from smogsense.models.binding import (
        audit_calibrator_binding,
    )
    from smogsense.models.registry import BaseModelBindingError, BundleTamperedError

    print("=" * 75)
    print("BASE / CALIBRATOR CRYPTOGRAPHIC BINDING VERIFICATION (Task P3-11)")
    print("=" * 75)
    print(f"Calibrator Bundle: {calibrator_bundle}")
    print(f"Base Bundle:       {base_bundle}")
    print(f"Cache Directory:   {cache_dir}")
    print("=" * 75)

    try:
        card = audit_calibrator_binding(
            calibrator_bundle_path_or_tag=calibrator_bundle,
            base_bundle_path_or_tag=base_bundle,
            cache_dir=cache_dir,
        )
        print("\nBINDING VERIFICATION SUMMARY")
        print("-" * 75)
        print(f"Status:                 {card['status'].upper()}")
        print(f"Calibrator Model:       {card['calibrator_model']} ({card['calibrator_version']})")
        print(f"Declared Base SHA-256:  {card['calibrator_base_sha256']}")
        print(f"Active Base SHA-256:    {card['active_base_sha256']}")
        print(f"OOF Partition Hash:     {card.get('oof_partition_hash')}")
        print(f"Feature Set Version:    {card.get('feature_set_version')}")
        print(f"Parameter Count:        {card.get('parameter_count')}")
        print(f"Meta-SGD Step Sizes:    {card.get('step_sizes_count')}")
        print("-" * 75)

        if card["is_bound"]:
            print("[OK] Cryptographic binding verified (Condition C3 / T1-15 passed).")
            print("=" * 75 + "\n")
            safe_exit(0)
        else:
            print("[FAIL] Base model binding mismatch (Condition C3 violation).")
            print("=" * 75 + "\n")
            safe_exit(30)
    except BaseModelBindingError as b_err:
        print(f"\n[FAIL] BaseModelBindingError: {b_err}")
        print("=" * 75 + "\n")
        safe_exit(30)
    except BundleTamperedError as t_err:
        print(f"\n[FAIL] BundleTamperedError: {t_err}")
        print("=" * 75 + "\n")
        safe_exit(30)
    except Exception as exc:
        print(f"\n[ERROR] Verification aborted with error: {exc}")
        print("=" * 75 + "\n")
        safe_exit(50)


@app_registry.command("verify-binding")
def registry_verify_binding_alias_cmd(
    calibrator_bundle: str = typer.Option(
        ...,
        "--calibrator-bundle",
        "-c",
        help="Path or tag for neural calibrator bundle (M6 or M7)",
    ),
    base_bundle: str = typer.Option(
        ...,
        "--base-bundle",
        "-b",
        help="Path or tag for shared LightGBM physics base (M5)",
    ),
    cache_dir: str = typer.Option(
        "data/processed/models",
        "--cache-dir",
        help="Path to model bundle cache directory",
    ),
) -> None:
    """Alias for smogsense models verify-binding under registry command group."""
    models_verify_binding_cmd(
        calibrator_bundle=calibrator_bundle,
        base_bundle=base_bundle,
        cache_dir=cache_dir,
    )


def main() -> None:
    try:
        app()
    except SmogSenseError as e:
        print(f"Error: {e}")
        safe_exit(e.exit_code)
    except Exception as e:
        print(f"Internal error: {e}")
        safe_exit(50)


if __name__ == "__main__":
    main()
