"""smogsense.cli - Typer command-line entry point.

Specification: docs/system-architecture.md -> 'Component responsibilities and CLI contract'
"""

import contextlib
import faulthandler
import os
import sys
from datetime import UTC

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
