"""Shared fixtures.

Available now: repo_root, fixtures_dir, frozen_issuance, and an autouse switch to offline fixtures mode.
Added in Phase 1 with the ingestion layer: tmp_state_branch (a temporary git repo with an orphan `state` branch) and a
socket guard that fails any test attempting real network access unless it is marked `network`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def frozen_issuance() -> datetime:
    """The canonical demo/test issuance: a smog-season day."""
    return datetime(2025, 11, 5, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMOGSENSE_MODE", "fixtures")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require real network",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        # --run-network given in cli: do not skip network tests
        return
    skip_network = pytest.mark.skip(reason="need --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
