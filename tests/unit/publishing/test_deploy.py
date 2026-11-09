"""Unit tests for static site budget, CSP, and accessibility validator.

Specification: docs/dissemination-and-ui.md -> 'Accessibility and performance budgets'
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smogsense.publishing.deploy import (
    BudgetBreachError,
    Violation,
    ViolationType,
    format_violations_table,
    load_site_budgets,
    validate_site,
)


def _create_minimal_valid_site(site_dir: Path) -> None:
    """Helper to populate a minimal valid site satisfying all budgets and checks."""
    site_dir.mkdir(parents=True, exist_ok=True)

    # Core required files
    (site_dir / "forecast").mkdir(parents=True, exist_ok=True)
    (site_dir / "forecast" / "latest.json").write_text("{}", encoding="utf-8")
    (site_dir / "latest.json").write_text("{}", encoding="utf-8")
    (site_dir / "sitemap.xml").write_text("<urlset></urlset>", encoding="utf-8")
    (site_dir / "robots.txt").write_text("User-agent: *\nAllow: /", encoding="utf-8")
    (site_dir / ".nojekyll").touch()
    (site_dir / "_headers").write_text("/*\n  Cache-Control: public", encoding="utf-8")

    # Assets
    css_dir = site_dir / "assets" / "css"
    css_dir.mkdir(parents=True, exist_ok=True)
    (css_dir / "base.css").write_text("body { margin: 0; }", encoding="utf-8")

    js_dir = site_dir / "assets" / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (js_dir / "dashboard.js").write_text("console.log('init');", encoding="utf-8")

    fonts_dir = site_dir / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "noto-nastaliq-urdu.woff2").write_bytes(b"wOF2" + b"\x00" * 32)

    cards_dir = site_dir / "cards" / "latest"
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / "lahore-post-en.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    (cards_dir / "lahore-og-en.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # English page
    html_en = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self';">
  <link rel="alternate" hreflang="ur" href="/ur/">
  <link rel="stylesheet" href="/assets/css/base.css">
  <script src="/assets/js/dashboard.js"></script>
</head>
<body>
  <a class="skip-link" href="#forecast">Skip</a>
  <h1>Daily Forecast</h1>
  <div class="forecast-metadata__valid-until">Valid until 2026-11-10T06:00:00Z</div>
</body>
</html>"""
    (site_dir / "index.html").write_text(html_en, encoding="utf-8")

    # Urdu page
    ur_dir = site_dir / "ur"
    ur_dir.mkdir(parents=True, exist_ok=True)
    html_ur = """<!DOCTYPE html>
<html lang="ur" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self';">
  <link rel="alternate" hreflang="en" href="/">
  <link rel="stylesheet" href="/assets/css/base.css">
  <script src="/assets/js/dashboard.js"></script>
</head>
<body>
  <a class="skip-link" href="#forecast">چھلانگ لگائیں</a>
  <h1>روزانہ پیش گوئی</h1>
  <div class="forecast-metadata__valid-until">درست تا 2026-11-10T06:00:00Z</div>
</body>
</html>"""
    (ur_dir / "index.html").write_text(html_ur, encoding="utf-8")

    # 404 page
    html_404 = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self';">
</head>
<body>
  <h1>404 Not Found</h1>
</body>
</html>"""
    (site_dir / "404.html").write_text(html_404, encoding="utf-8")


def test_load_site_budgets(tmp_path: Path) -> None:
    """Verify loading performance budgets from real or default config."""
    # Real repo config
    real_budgets = load_site_budgets("configs/bulletin.yaml")
    assert real_budgets["html_kb"] == 60
    assert real_budgets["css_kb"] == 25
    assert real_budgets["js_kb"] == 30
    assert real_budgets["page_weight_kb_excluding_fonts"] == 500
    assert real_budgets["urdu_font_woff2_kb"] == 160
    assert real_budgets["whatsapp_post_kb"] == 400
    assert real_budgets["open_graph_kb"] == 300

    # Non-existent config falls back to defaults
    missing_budgets = load_site_budgets(tmp_path / "nonexistent.yaml")
    assert missing_budgets["html_kb"] == 60
    assert missing_budgets["whatsapp_post_kb"] == 400


def test_validate_site_clean(tmp_path: Path) -> None:
    """Verify that a compliant static site passes audit with zero violations."""
    _create_minimal_valid_site(tmp_path)
    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    assert violations == []


def test_validate_site_missing_files(tmp_path: Path) -> None:
    """Verify that missing core files trigger MISSING_FILE violations."""
    _create_minimal_valid_site(tmp_path)
    (tmp_path / ".nojekyll").unlink()
    (tmp_path / "robots.txt").unlink()

    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    missing = [v for v in violations if v.violation_type == ViolationType.MISSING_FILE]
    missing_paths = {v.file_path for v in missing}
    assert ".nojekyll" in missing_paths
    assert "robots.txt" in missing_paths


def test_validate_site_file_size_breach(tmp_path: Path) -> None:
    """Verify that oversized assets trigger FILE_SIZE violations."""
    _create_minimal_valid_site(tmp_path)

    # Oversize HTML (> 60 KB)
    (tmp_path / "404.html").write_text("A" * (65 * 1024), encoding="utf-8")

    # Oversize CSS (> 25 KB)
    (tmp_path / "assets" / "css" / "huge.css").write_text(
        "/* " + "X" * (30 * 1024) + " */", encoding="utf-8"
    )

    # Oversize JS (> 30 KB)
    (tmp_path / "assets" / "js" / "huge.js").write_text("// " + "Y" * (35 * 1024), encoding="utf-8")

    # Oversize Font (> 160 KB)
    (tmp_path / "assets" / "fonts" / "noto-nastaliq-urdu.woff2").write_bytes(b"F" * (170 * 1024))

    # Oversize Card (> 400 KB)
    (tmp_path / "cards" / "latest" / "lahore-post-en.png").write_bytes(b"P" * (450 * 1024))

    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    size_violations = [v for v in violations if v.violation_type == ViolationType.FILE_SIZE]

    paths = {v.file_path for v in size_violations}
    assert "404.html" in paths
    assert "assets/css/huge.css" in paths
    assert "assets/js/huge.js" in paths
    assert "assets/fonts/noto-nastaliq-urdu.woff2" in paths
    assert "cards/latest/lahore-post-en.png" in paths


def test_validate_site_page_weight_breach(tmp_path: Path) -> None:
    """Verify that combined page weight (> 500 KB) triggers PAGE_WEIGHT violation."""
    _create_minimal_valid_site(tmp_path)

    # Create large linked CSS of 550 KB
    (tmp_path / "assets" / "css" / "base.css").write_text(
        "/* " + "Z" * (550 * 1024) + " */", encoding="utf-8"
    )

    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    pw_violations = [v for v in violations if v.violation_type == ViolationType.PAGE_WEIGHT]
    assert len(pw_violations) >= 1
    assert any(v.file_path == "index.html" for v in pw_violations)


def test_validate_site_csp_violations(tmp_path: Path) -> None:
    """Verify detection of missing CSP meta, inline style tags, and inline script tags."""
    _create_minimal_valid_site(tmp_path)

    bad_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <title>Bad CSP</title>
  <style>body { background: #000; }</style>
  <script>console.log("forbidden inline");</script>
</head>
<body>
  <h1>Bad</h1>
</body>
</html>"""
    (tmp_path / "404.html").write_text(bad_html, encoding="utf-8")

    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    csp_violations = [v for v in violations if v.violation_type == ViolationType.CSP_INLINE]

    messages = [v.message for v in csp_violations]
    assert any("missing Content-Security-Policy" in m for m in messages)
    assert any("forbidden inline <style>" in m for m in messages)
    assert any("forbidden inline <script>" in m for m in messages)


def test_validate_site_accessibility_violations(tmp_path: Path) -> None:
    """Verify accessibility checks: single <h1>, skip link, valid-until notice, reciprocal hreflang."""
    _create_minimal_valid_site(tmp_path)

    bad_index = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self';">
</head>
<body>
  <h1>First Title</h1>
  <h1>Second Title</h1>
  <div>Plain text without required accessible features</div>
</body>
</html>"""
    (tmp_path / "index.html").write_text(bad_index, encoding="utf-8")

    violations = validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=False)
    a11y_violations = [v for v in violations if v.violation_type == ViolationType.ACCESSIBILITY]

    types = {v.measured_value for v in a11y_violations}
    assert 2 in types  # 2 h1 elements
    assert "missing_skip_link" in types
    assert "missing_valid_until" in types
    assert "missing_hreflang" in types


def test_validate_site_raises_budget_breach_error(tmp_path: Path) -> None:
    """Verify raise_on_breach=True raises BudgetBreachError with exit code 30."""
    _create_minimal_valid_site(tmp_path)
    (tmp_path / ".nojekyll").unlink()

    with pytest.raises(BudgetBreachError) as exc_info:
        validate_site(site_dir=tmp_path, configs_dir="configs", raise_on_breach=True)

    assert exc_info.value.exit_code == 30
    assert "Site audit failed" in str(exc_info.value)


def test_format_violations_table() -> None:
    """Verify terminal formatting for empty and populated violation lists."""
    assert "0 violations detected" in format_violations_table([])

    v = Violation(
        file_path="assets/css/base.css",
        violation_type=ViolationType.FILE_SIZE,
        measured_value=32.5,
        budget_value=25.0,
        message="CSS file exceeds budget",
    )
    formatted = format_violations_table([v])
    assert "file_size" in formatted
    assert "assets/css/base.css" in formatted
    assert "32.5" in formatted
    assert "25.0" in formatted
