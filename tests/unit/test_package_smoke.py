"""Every module imports, carries a contract docstring, and the package version is well-formed."""

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest

import smogsense

pytestmark = pytest.mark.unit

MODULE_NAMES = [m.name for m in pkgutil.walk_packages(smogsense.__path__, "smogsense.")]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", smogsense.__version__)


def test_expected_number_of_modules() -> None:
    assert len(MODULE_NAMES) >= 55


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_module_imports_and_documents_its_contract(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no docstring"
    assert "Specification:" in module.__doc__, f"{name} does not point to its specification"
