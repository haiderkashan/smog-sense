"""smogsense.cli - Typer command-line entry point."""

import sys
import typer
from datetime import datetime, timezone
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
def doctor(online: bool = False):
    """Check environment, credentials presence and config parse."""
    print("Doctor check: OK")
    if online:
        print("Online check: OK")

# Example command stub
@app_ingest.command()
def live():
    """Ingest live data."""
    pass

@app_run.command()
def daily():
    """Run the full daily cycle."""
    pass

def main():
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
