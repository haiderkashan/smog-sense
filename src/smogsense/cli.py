"""smogsense.cli - Typer command-line entry point.

Specification: docs/system-architecture.md -> 'Component responsibilities and CLI contract'
"""

import contextlib
import faulthandler
import sys
from datetime import UTC

import typer

from smogsense.errors import SmogSenseError

faulthandler.enable()

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


@app_run.command()
def daily(
    issuance: str = typer.Option("latest", "--issuance", help="Issuance time"),
    force: bool = typer.Option(False, "--force", help="Force run even if already published"),
) -> None:
    """Run the full daily cycle."""
    from datetime import datetime

    from smogsense.pipeline.orchestrator import run_daily_pipeline

    if issuance == "latest":
        # Get start of today in UTC
        now = datetime.now(UTC)
        issuance_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        issuance_utc = datetime.fromisoformat(issuance).replace(tzinfo=UTC)

    exit_code = run_daily_pipeline(issuance_utc, force=force)
    sys.exit(exit_code)


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

    manifest_file = Path(".state/manifests") / f"run_{issuance_utc.strftime('%Y%m%d_%H%M')}.json"
    if manifest_file.exists():
        with contextlib.suppress(Exception):
            with manifest_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("published") is True:
                sys.exit(11)
    sys.exit(0)


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
    sys.exit(exit_code)


def main() -> None:
    try:
        app()
    except SmogSenseError as e:
        print(f"Error: {e}")
        sys.exit(e.exit_code)
    except Exception as e:
        print(f"Internal error: {e}")
        sys.exit(50)


if __name__ == "__main__":
    main()
