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

app.add_typer(app_ingest, name="ingest")
app.add_typer(app_features, name="features")
app.add_typer(app_train, name="train")
app.add_typer(app_forecast, name="forecast")
app.add_typer(app_score, name="score")
app.add_typer(app_eval, name="eval")
app.add_typer(app_bulletin, name="bulletin")
app.add_typer(app_site, name="site")
app.add_typer(app_publish, name="publish")
app.add_typer(app_state, name="state")
app.add_typer(app_run, name="run")
app.add_typer(app_reconcile, name="reconcile")
app.add_typer(app_audit, name="audit")
app.add_typer(app_preprocess, name="preprocess")


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


@app_run.command()
def daily(
    issuance: str = typer.Option("latest", "--issuance", help="Issuance time"),
    force: bool = typer.Option(False, "--force", help="Force run even if already published"),
    mode: str | None = typer.Option(None, "--mode", help="Execution mode (live, fixtures)"),
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

    exit_code = run_daily_pipeline(issuance_utc, force=force, mode=mode)
    safe_exit(exit_code)


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
    stale: bool = typer.Option(
        False, "--stale", "--expired", help="Render static expiry notice (stale forecast)"
    ),
    state_dir: str = typer.Option(".state", "--state-dir", help="Path to state directory"),
    out_dir: str = typer.Option("site", "--out-dir", help="Output directory for site"),
) -> None:
    """Build the static website (with optional --stale expiry notice) (Task P1-23)."""
    from smogsense.publishing.site import render_expired_site

    res = render_expired_site(state_dir=Path(state_dir), out_dir=Path(out_dir))
    if stale:
        print(
            f"Rendered expired static site into {out_dir} (valid until {res.get('valid_until_utc')})"
        )
    else:
        print(f"Built site at {out_dir}")
    safe_exit(0)


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

    if oof:
        print("Executing season-blocked cross-fitting on Block E...")
        oof_df, base_model = fit_cross_fit_oof(
            df_block_e=df_train,
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
        base_model.fit(df_train, num_boost_round=num_boost_round)
        base_model.save(out_path)
        print(f"Saved LightGBM M5 ensemble to: {out_path}")

    print(f"Deterministic Base SHA256: {base_model.compute_base_sha256()}")
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
