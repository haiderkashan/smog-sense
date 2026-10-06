"""CLI contract."""

from typer.testing import CliRunner

from smogsense.cli import app

runner = CliRunner()


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Doctor check: OK" in result.stdout


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "forecast" in result.stdout
