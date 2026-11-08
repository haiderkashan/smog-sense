"""Unit tests for Jinja2 template rendering, static asset budgets, and security invariants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import jinja2
import pytest

from smogsense.publishing.i18n import create_jinja_env

pytestmark = pytest.mark.unit


class MockSite:
    """Mock site context adhering to the Jinja2 template contract."""

    base_url = "https://smogsense.org"
    build_id = "test-build-1"
    generated_at_utc = "2025-11-05T00:26:41Z"

    @staticmethod
    def asset(path: str) -> str:
        """Return fingerprinted mock asset path."""
        return f"/assets/{path}"


def load_fixture_bulletin() -> dict[str, Any]:
    """Load sample valid bulletin fixture."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def test_jinja_rendering_en_and_ur() -> None:
    """Verify that index.html.j2 renders cleanly in both English and Urdu with StrictUndefined."""
    bulletin = load_fixture_bulletin()
    site = MockSite()

    page_en = {
        "title": "SmogSense Lahore - Daily Smog Forecast",
        "canonical_url": "https://smogsense.org/",
        "og_image_url": "https://smogsense.org/og/latest-en.png",
        "alternate_lang_url": "https://smogsense.org/ur/",
    }

    page_ur = {
        "title": "اسموگ سینس لاہور - روزانہ اسموگ کی پیش گوئی",
        "canonical_url": "https://smogsense.org/ur/",
        "og_image_url": "https://smogsense.org/og/latest-ur.png",
        "alternate_lang_url": "https://smogsense.org/",
    }

    # 1. English rendering
    env_en = create_jinja_env(lang="en")
    template_en = env_en.get_template("index.html.j2")
    html_en = template_en.render(site=site, page=page_en, bulletin=bulletin)

    assert "<!DOCTYPE html>" in html_en
    assert '<html lang="en" dir="ltr"' in html_en
    assert "SmogSense Lahore" in html_en
    assert 'href="#forecast"' in html_en
    assert '<main id="forecast"' in html_en
    assert '<header class="site-header"' in html_en
    assert '<footer class="site-footer"' in html_en
    assert "243" in html_en  # Hero median value
    assert "Valid until" in html_en
    assert 'hreflang="ur"' in html_en
    assert 'table class="sr-only"' in html_en

    # 2. Urdu rendering
    env_ur = create_jinja_env(lang="ur")
    template_ur = env_ur.get_template("index.html.j2")
    html_ur = template_ur.render(site=site, page=page_ur, bulletin=bulletin)

    assert "<!DOCTYPE html>" in html_ur
    assert '<html lang="ur" dir="rtl"' in html_ur
    assert "اسموگ سینس لاہور" in html_ur
    assert "خطرناک" in html_ur
    assert "243" in html_ur
    assert 'hreflang="en"' in html_ur
    assert "rtl.css" in html_ur
    assert "noto-nastaliq-urdu.woff2" in html_ur


def test_strict_undefined_catches_missing_variables() -> None:
    """Verify that StrictUndefined raises build error when required context is missing."""
    env = create_jinja_env(lang="en")
    template = env.get_template("index.html.j2")

    # Missing bulletin
    with pytest.raises(jinja2.UndefinedError):
        template.render(
            site=MockSite(),
            page={
                "title": "Test",
                "canonical_url": "",
                "og_image_url": "",
                "alternate_lang_url": "",
            },
        )


def test_static_file_budgets() -> None:
    """Verify that stylesheets, script, and rendered pages satisfy performance budgets."""
    base_css_size = Path("web/static/css/base.css").stat().st_size
    rtl_css_size = Path("web/static/css/rtl.css").stat().st_size
    dashboard_js_size = Path("web/static/js/dashboard.js").stat().st_size

    # Budgets from configs/bulletin.yaml
    assert base_css_size <= 25 * 1024, f"base.css ({base_css_size} bytes) exceeds 25 KB budget"
    assert rtl_css_size <= 10 * 1024, f"rtl.css ({rtl_css_size} bytes) exceeds 10 KB budget"
    assert dashboard_js_size <= 30 * 1024, (
        f"dashboard.js ({dashboard_js_size} bytes) exceeds 30 KB budget"
    )

    # HTML budget <= 60 KB
    bulletin = load_fixture_bulletin()
    page = {
        "title": "SmogSense Test",
        "canonical_url": "https://smogsense.org/",
        "og_image_url": "https://smogsense.org/og.png",
        "alternate_lang_url": "https://smogsense.org/ur/",
    }
    env = create_jinja_env(lang="en")
    html_str = env.get_template("index.html.j2").render(
        site=MockSite(), page=page, bulletin=bulletin
    )
    html_bytes = len(html_str.encode("utf-8"))

    assert html_bytes <= 60 * 1024, f"Rendered HTML ({html_bytes} bytes) exceeds 60 KB budget"


def test_csp_and_security_invariants() -> None:
    """Verify that rendered HTML enforces CSP and contains no forbidden inline script or style blocks."""
    bulletin = load_fixture_bulletin()
    page = {
        "title": "SmogSense Security Test",
        "canonical_url": "https://smogsense.org/",
        "og_image_url": "https://smogsense.org/og.png",
        "alternate_lang_url": "https://smogsense.org/ur/",
    }
    env = create_jinja_env(lang="en")
    html = env.get_template("index.html.j2").render(site=MockSite(), page=page, bulletin=bulletin)

    # 1. CSP meta tag is present
    assert '<meta http-equiv="Content-Security-Policy"' in html
    assert "default-src 'self'" in html

    # 2. No inline <style>...</style> tags
    assert "<style" not in html

    # 3. No inline <script>...</script> containing executable code (only deferred external script link allowed)
    inline_scripts = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", html, flags=re.DOTALL)
    assert len(inline_scripts) == 0, f"Found forbidden inline script: {inline_scripts}"
