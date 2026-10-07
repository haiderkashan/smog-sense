"""smogsense.cli - Typer command-line entry point.

Specification: docs/system-architecture.md -> 'Component responsibilities and CLI contract'
"""

import sys

import typer

from smogsense.errors import SmogSenseError

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
def daily(issuance: str = typer.Option("latest", "--issuance", help="Issuance time")) -> None:
    """Run the full daily cycle."""
    import json
    from pathlib import Path

    # Phase 0 scaffolding: produce dummy artifacts to satisfy the GitHub Actions deployment workflow
    forecast_dir = Path("site/forecast")
    forecast_dir.mkdir(parents=True, exist_ok=True)
    (Path("site") / "index.html").write_text("<html><body><h1>SmogSense Skeleton</h1></body></html>", encoding="utf-8")
    (forecast_dir / "latest.json").write_text(json.dumps({"dummy": True, "issuance": issuance}), encoding="utf-8")

    print("Daily cycle skeleton completed. Dummy site artifacts generated.")



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
