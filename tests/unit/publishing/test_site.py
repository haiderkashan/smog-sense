"""Unit tests for static site builder, asset fingerprinting, and font subsetting.

Specification: docs/dissemination-and-ui.md -> 'Static site architecture'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.publishing.site import (
    SiteContext,
    build_site,
    copy_and_fingerprint_assets,
    generate_site,
    render_expired_site,
    subset_urdu_font,
)


@pytest.fixture
def sample_bulletin() -> dict[str, Any]:
    """Load canonical test fixture bulletin."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def test_site_context_asset_resolution() -> None:
    """Verify SiteContext resolves fingerprinted asset paths via manifest."""
    manifest = {
        "css/base.css": "/assets/css/base.a1b2c3d4.css",
        "js/dashboard.js": "/assets/js/dashboard.e5f60718.js",
    }
    ctx = SiteContext(base_url="https://smogsense.org", manifest=manifest)

    # Path with no prefix
    assert ctx.asset("css/base.css") == "/assets/css/base.a1b2c3d4.css"

    # Path with leading slash
    assert ctx.asset("/css/base.css") == "/assets/css/base.a1b2c3d4.css"

    # Path with assets/ prefix
    assert ctx.asset("assets/css/base.css") == "/assets/css/base.a1b2c3d4.css"
    assert ctx.asset("/assets/css/base.css") == "/assets/css/base.a1b2c3d4.css"

    # Unmanifested path fallback
    assert ctx.asset("images/logo.png") == "/assets/images/logo.png"

    # Subpath base_url (e.g. GitHub Pages repo root)
    ctx_subpath = SiteContext(
        base_url="https://haiderkashan.github.io/smog-sense", manifest=manifest
    )
    assert ctx_subpath.asset("css/base.css") == "/smog-sense/assets/css/base.a1b2c3d4.css"
    assert ctx_subpath.asset("images/logo.png") == "/smog-sense/assets/images/logo.png"


def test_copy_and_fingerprint_assets(tmp_path: Path) -> None:
    """Verify SHA-256 fingerprinting of static assets and manifest.json emission."""
    static_dir = tmp_path / "web/static"
    (static_dir / "css").mkdir(parents=True)
    (static_dir / "js").mkdir(parents=True)

    css_file = static_dir / "css" / "base.css"
    css_file.write_text("body { color: #111; }", encoding="utf-8")

    js_file = static_dir / "js" / "app.js"
    js_file.write_text("console.log('test');", encoding="utf-8")

    out_assets = tmp_path / "site/assets"
    manifest = copy_and_fingerprint_assets(static_dir=static_dir, output_assets_dir=out_assets)

    assert "css/base.css" in manifest
    assert "js/app.js" in manifest

    # Check hashed file names
    hashed_css = manifest["css/base.css"]
    assert hashed_css.startswith("/assets/css/base.")
    assert hashed_css.endswith(".css")

    # Check that both fingerprinted and un-fingerprinted files exist on disk
    assert (out_assets / "css" / "base.css").is_file()
    assert (out_assets / "js" / "app.js").is_file()

    # Check manifest.json on disk
    manifest_file = out_assets / "manifest.json"
    assert manifest_file.is_file()
    with manifest_file.open("r", encoding="utf-8") as f:
        saved_manifest = json.load(f)
    assert saved_manifest == manifest


def test_subset_urdu_font(tmp_path: Path) -> None:
    """Verify WOFF2 font subsetting fallback and output existence."""
    out_font = tmp_path / "site/assets/fonts/noto-nastaliq-urdu.woff2"
    result_path = subset_urdu_font(
        source_font_path=None,
        output_font_path=out_font,
        config_path="configs/bulletin.yaml",
    )
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_build_site_end_to_end(tmp_path: Path, sample_bulletin: dict[str, Any]) -> None:
    """Verify complete static site build, including sitemap, robots, _headers, cards, and audit validation."""
    site_dir = tmp_path / "site"

    summary = build_site(
        bulletin=sample_bulletin,
        output_dir=site_dir,
        configs_dir="configs",
        templates_dir="web/templates",
        static_dir="web/static",
        validate=True,
    )

    assert summary["status"] == "ok"
    assert summary["violations"] == []

    # Verify generated pages
    index_en = site_dir / "index.html"
    index_ur = site_dir / "ur" / "index.html"
    not_found = site_dir / "404.html"

    assert index_en.is_file()
    assert index_ur.is_file()
    assert not_found.is_file()

    # Content verification
    content_en = index_en.read_text(encoding="utf-8")
    assert '<html lang="en" dir="ltr"' in content_en
    assert 'hreflang="ur"' in content_en
    assert 'class="skip-link"' in content_en
    assert 'class="fan-chart"' in content_en

    content_ur = index_ur.read_text(encoding="utf-8")
    assert '<html lang="ur" dir="rtl"' in content_ur
    assert 'hreflang="en"' in content_ur
    assert 'class="skip-link"' in content_ur

    # Verify JSON APIs
    assert (site_dir / "latest.json").is_file()
    assert (site_dir / "forecast" / "latest.json").is_file()
    date_tag = summary["date_tag"]
    assert (site_dir / "forecast" / f"{date_tag}.json").is_file()

    # Verify social cards rendered
    cards_dir = site_dir / "cards" / "latest"
    assert (cards_dir / "lahore-whatsapp-en.png").is_file()
    assert (cards_dir / "lahore-whatsapp-ur.png").is_file()
    assert (cards_dir / "lahore-og-en.png").is_file()
    assert (cards_dir / "lahore-og-ur.png").is_file()

    # Verify sitemap.xml
    sitemap = (site_dir / "sitemap.xml").read_text(encoding="utf-8")
    assert "<loc>https://smogsense.org/</loc>" in sitemap
    assert '<xhtml:link rel="alternate" hreflang="ur"' in sitemap
    assert '<xhtml:link rel="alternate" hreflang="en"' in sitemap

    # Verify robots.txt
    robots = (site_dir / "robots.txt").read_text(encoding="utf-8")
    assert "Sitemap: https://smogsense.org/sitemap.xml" in robots

    # Verify _headers
    headers = (site_dir / "_headers").read_text(encoding="utf-8")
    assert "max-age=31536000, immutable" in headers
    assert "Access-Control-Allow-Origin: *" in headers

    # Verify .nojekyll
    assert (site_dir / ".nojekyll").is_file()


def test_generate_site_backward_compatibility(
    tmp_path: Path, sample_bulletin: dict[str, Any]
) -> None:
    """Verify generate_site helper preserves compatibility with earlier pipeline calls."""
    site_dir = tmp_path / "compat_site"
    generate_site(sample_bulletin, site_dir)
    assert (site_dir / "index.html").is_file()


def test_render_expired_site(tmp_path: Path) -> None:
    """Verify render_expired_site emits expired forecast site."""
    site_dir = tmp_path / "expired_site"
    res = render_expired_site(state_dir=tmp_path / ".state", out_dir=site_dir)
    assert res["status"] == "ok"
    assert res["expired"] is True
    assert (site_dir / "index.html").is_file()

    content = (site_dir / "index.html").read_text(encoding="utf-8")
    # Verify expiry notice / banner element exists in page
    assert 'id="expiry-banner"' in content


def test_cli_site_build_and_validate(tmp_path: Path, sample_bulletin: dict[str, Any]) -> None:
    """Verify CLI site build and site validate commands."""
    runner = CliRunner()
    bulletin_file = tmp_path / "bulletin.json"
    with bulletin_file.open("w", encoding="utf-8") as f:
        json.dump(sample_bulletin, f)

    site_dir = tmp_path / "cli_site"

    # 1. Run smogsense site build
    build_result = runner.invoke(
        app,
        [
            "site",
            "build",
            "--bulletin",
            str(bulletin_file),
            "--out-dir",
            str(site_dir),
            "--validate",
        ],
    )
    assert build_result.exit_code == 0
    assert "Successfully built static site" in build_result.output

    # 2. Run smogsense site validate on clean site
    val_result = runner.invoke(
        app,
        [
            "site",
            "validate",
            "--site-dir",
            str(site_dir),
            "--configs-dir",
            "configs",
        ],
    )
    assert val_result.exit_code == 0
    assert "Site validation passed" in val_result.output

    # 3. Simulate failure by removing required file
    (site_dir / ".nojekyll").unlink()
    fail_result = runner.invoke(
        app,
        [
            "site",
            "validate",
            "--site-dir",
            str(site_dir),
            "--configs-dir",
            "configs",
        ],
    )
    assert fail_result.exit_code == 30
    assert "Site validation failed" in fail_result.output
