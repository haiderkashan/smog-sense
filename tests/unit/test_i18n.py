"""Message-catalogue parity (en/ur) and palette parity between configs/bulletin.yaml and tokens.css."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")
# Values that are legitimately Latin-only in the Urdu catalogue
NO_ARABIC_REQUIRED = {"meta.lang", "meta.dir", "nav.switch_language"}


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(_flatten(value, f"{prefix.rstrip('.')}[{index}]."))
    else:
        out[prefix.rstrip(".")] = node
    return out


@pytest.fixture(scope="module")
def catalogues(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    en = yaml.safe_load((repo_root / "web/i18n/en.yaml").read_text(encoding="utf-8"))
    ur = yaml.safe_load((repo_root / "web/i18n/ur.yaml").read_text(encoding="utf-8"))
    return _flatten(en), _flatten(ur)


def test_key_sets_are_identical(catalogues: tuple[dict[str, Any], dict[str, Any]]) -> None:
    en, ur = catalogues
    assert set(en) == set(ur), sorted(set(en) ^ set(ur))


def test_placeholders_match(catalogues: tuple[dict[str, Any], dict[str, Any]]) -> None:
    en, ur = catalogues
    placeholder = re.compile(r"\{(\w+)\}")
    mismatched = [
        k
        for k in en
        if set(placeholder.findall(str(en[k]))) != set(placeholder.findall(str(ur[k])))
    ]
    assert not mismatched, mismatched


def test_urdu_values_are_actually_urdu(catalogues: tuple[dict[str, Any], dict[str, Any]]) -> None:
    _, ur = catalogues
    offenders = [
        k for k, v in ur.items() if k not in NO_ARABIC_REQUIRED and not ARABIC_SCRIPT.search(str(v))
    ]
    assert not offenders, offenders


def test_direction_metadata(catalogues: tuple[dict[str, Any], dict[str, Any]]) -> None:
    en, ur = catalogues
    assert (en["meta.dir"], ur["meta.dir"]) == ("ltr", "rtl")


def test_every_category_has_advice_in_both_languages(
    catalogues: tuple[dict[str, Any], dict[str, Any]], repo_root: Path
) -> None:
    en, ur = catalogues
    bulletin = yaml.safe_load((repo_root / "configs/bulletin.yaml").read_text(encoding="utf-8"))
    for category in bulletin["aqi"]["categories"]:
        for catalogue in (en, ur):
            assert f"aqi.{category['id']}" in catalogue
            assert f"advice.{category['id']}.headline" in catalogue
            assert f"advice.{category['id']}.actions[0]" in catalogue


def test_palette_parity_between_bulletin_yaml_and_tokens_css(repo_root: Path) -> None:
    bulletin = yaml.safe_load((repo_root / "configs/bulletin.yaml").read_text(encoding="utf-8"))
    css = (repo_root / "web/static/css/tokens.css").read_text(encoding="utf-8")
    for category in bulletin["aqi"]["categories"]:
        cid = category["id"].replace("_", "-")
        bg = re.search(rf"--aqi-{cid}-bg:\s*(#[0-9a-fA-F]{{6}})", css)
        fg = re.search(rf"--aqi-{cid}-fg:\s*(#[0-9a-fA-F]{{6}})", css)
        assert bg and fg, f"tokens.css lacks colours for {cid}"
        assert bg.group(1).lower() == category["bg"].lower()
        assert fg.group(1).lower() == category["fg"].lower()


def test_example_bulletin_references_existing_keys(
    catalogues: tuple[dict[str, Any], dict[str, Any]], fixtures_dir: Path
) -> None:
    en, _ = catalogues
    example = json.loads((fixtures_dir / "bulletin_valid_example.json").read_text(encoding="utf-8"))
    needed = [f"advice.{example['advisory']['category']}.headline", example["disclaimer_key"]]
    needed += [f"notes.{note}" for note in example["advisory"]["notes"]]
    assert all(key in en for key in needed)
