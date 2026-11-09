"""smogsense.publishing.deploy — Delivery preparation and static site budget validation.

Validates the built static site against strict performance budgets (HTML <= 60 KB,
CSS <= 25 KB, JS <= 30 KB, page weight <= 500 KB, Urdu font <= 160 KB), CSP security
invariants (zero inline scripts or styles), and WCAG 2.2 AA accessibility landmarks.

Specification: docs/dissemination-and-ui.md → 'Accessibility and performance budgets'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from smogsense.errors import SmogSenseError

logger = logging.getLogger(__name__)


class ViolationType(StrEnum):
    """Categorization of audit and budget violations."""

    FILE_SIZE = "file_size"
    PAGE_WEIGHT = "page_weight"
    CSP_INLINE = "csp_inline"
    ACCESSIBILITY = "accessibility"
    MISSING_FILE = "missing_file"
    KEY_PARITY = "key_parity"


@dataclass(frozen=True)
class Violation:
    """Detailed record of an audit budget breach or accessibility violation."""

    file_path: str
    violation_type: ViolationType
    measured_value: str | float | int
    budget_value: str | float | int
    message: str


class BudgetBreachError(SmogSenseError, ValueError):
    """Raised when static site assets or rendered pages breach performance or security budgets."""

    def __init__(self, message: str, exit_code: int = 30) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def load_site_budgets(config_path: Path | str = "configs/bulletin.yaml") -> dict[str, Any]:
    """Load performance and size budgets from bulletin config."""
    cfg_file = Path(config_path)
    defaults = {
        "html_kb": 60,
        "css_kb": 25,
        "js_kb": 30,
        "page_weight_kb_excluding_fonts": 500,
        "urdu_font_woff2_kb": 160,
        "whatsapp_post_kb": 400,
        "open_graph_kb": 300,
    }
    if not cfg_file.is_file():
        return defaults

    try:
        with cfg_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            site_budgets = data.get("site", {}).get("budgets", {})
            cards_cfg = data.get("cards", {})
            res = dict(defaults)
            res.update(site_budgets)
            if "whatsapp_post" in cards_cfg and "max_kb" in cards_cfg["whatsapp_post"]:
                res["whatsapp_post_kb"] = cards_cfg["whatsapp_post"]["max_kb"]
            if "open_graph" in cards_cfg and "max_kb" in cards_cfg["open_graph"]:
                res["open_graph_kb"] = cards_cfg["open_graph"]["max_kb"]
            return res
    except Exception as err:
        logger.warning("Error reading %s: %s; using defaults", config_path, err)
        return defaults


def validate_site(
    site_dir: Path | str = "site",
    configs_dir: Path | str = "configs",
    raise_on_breach: bool = False,
) -> list[Violation]:
    """Audit the built static website for file weight budgets, CSP rules, and accessibility landmarks.

    Parameters
    ----------
    site_dir : Path | str
        Path to the built site directory.
    configs_dir : Path | str
        Path to directory containing bulletin.yaml.
    raise_on_breach : bool
        If True, raises BudgetBreachError on the first violation.

    Returns
    -------
    list[Violation]
        List of all detected audit violations.
    """
    root = Path(site_dir)
    cfg_path = Path(configs_dir) / "bulletin.yaml"
    budgets = load_site_budgets(cfg_path)

    violations: list[Violation] = []

    # 1. Verify existence of required core static files
    required_files = [
        "index.html",
        "ur/index.html",
        "404.html",
        "forecast/latest.json",
        "sitemap.xml",
        "robots.txt",
        ".nojekyll",
        "_headers",
    ]

    for rel_path in required_files:
        target = root / rel_path
        if not target.is_file():
            violations.append(
                Violation(
                    file_path=str(rel_path),
                    violation_type=ViolationType.MISSING_FILE,
                    measured_value="missing",
                    budget_value="present",
                    message=f"Required file {rel_path} was not generated in {root}",
                )
            )

    # 2. File size audits
    html_max_kb = float(budgets.get("html_kb", 60))
    css_max_kb = float(budgets.get("css_kb", 25))
    js_max_kb = float(budgets.get("js_kb", 30))
    font_max_kb = float(budgets.get("urdu_font_woff2_kb", 160))

    # Audit all HTML files
    for html_file in root.rglob("*.html"):
        rel_str = html_file.relative_to(root).as_posix()
        size_kb = html_file.stat().st_size / 1024.0
        if size_kb > html_max_kb:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value=round(size_kb, 2),
                    budget_value=html_max_kb,
                    message=f"HTML file {rel_str} ({size_kb:.1f} KB) exceeds {html_max_kb} KB budget",
                )
            )

    # Audit CSS assets
    for css_file in root.glob("assets/css/*.css"):
        rel_str = css_file.relative_to(root).as_posix()
        size_kb = css_file.stat().st_size / 1024.0
        if size_kb > css_max_kb:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value=round(size_kb, 2),
                    budget_value=css_max_kb,
                    message=f"CSS file {rel_str} ({size_kb:.1f} KB) exceeds {css_max_kb} KB budget",
                )
            )

    # Audit JS assets
    for js_file in root.glob("assets/js/*.js"):
        rel_str = js_file.relative_to(root).as_posix()
        size_kb = js_file.stat().st_size / 1024.0
        if size_kb > js_max_kb:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value=round(size_kb, 2),
                    budget_value=js_max_kb,
                    message=f"JS file {rel_str} ({size_kb:.1f} KB) exceeds {js_max_kb} KB budget",
                )
            )

    # Audit WOFF2 font asset
    for font_file in root.glob("assets/fonts/*.woff2"):
        rel_str = font_file.relative_to(root).as_posix()
        size_kb = font_file.stat().st_size / 1024.0
        if size_kb > font_max_kb:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value=round(size_kb, 2),
                    budget_value=font_max_kb,
                    message=f"Font file {rel_str} ({size_kb:.1f} KB) exceeds {font_max_kb} KB budget",
                )
            )

    # Audit social cards
    card_post_max_kb = float(budgets.get("whatsapp_post_kb", 400))
    card_og_max_kb = float(budgets.get("open_graph_kb", 300))
    for card_file in root.glob("cards/**/*.png"):
        rel_str = card_file.relative_to(root).as_posix()
        size_kb = card_file.stat().st_size / 1024.0
        card_limit = card_og_max_kb if "og" in card_file.name.lower() else card_post_max_kb
        if size_kb > card_limit:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value=round(size_kb, 2),
                    budget_value=card_limit,
                    message=f"Card {rel_str} ({size_kb:.1f} KB) exceeds {card_limit} KB budget",
                )
            )

    # 3. Security (CSP), Page Weight and Accessibility audits on HTML pages
    page_weight_max_kb = float(budgets.get("page_weight_kb_excluding_fonts", 500))

    for html_file in root.rglob("*.html"):
        rel_str = html_file.relative_to(root).as_posix()
        try:
            content = html_file.read_text(encoding="utf-8")
        except Exception as err:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.FILE_SIZE,
                    measured_value="read_error",
                    budget_value="utf-8",
                    message=f"Could not read {rel_str} as UTF-8: {err}",
                )
            )
            continue

        # Page weight check (HTML + referenced CSS + referenced JS, excluding fonts)
        page_bytes = html_file.stat().st_size
        css_hrefs = re.findall(
            r'<link\b[^>]*rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        css_hrefs += re.findall(
            r'<link\b[^>]*href=["\']([^"\']+)["\'][^>]*rel=["\']stylesheet["\']',
            content,
            re.IGNORECASE,
        )
        js_srcs = re.findall(r'<script\b[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE)

        asset_bytes = 0
        for ref in set(css_hrefs + js_srcs):
            clean_ref = ref.lstrip("/").split("?")[0]
            if "assets/" in clean_ref:
                clean_ref = clean_ref[clean_ref.find("assets/") :]
            cand = root / clean_ref
            if cand.is_file():
                asset_bytes += cand.stat().st_size

        total_page_kb = (page_bytes + asset_bytes) / 1024.0
        if total_page_kb > page_weight_max_kb:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.PAGE_WEIGHT,
                    measured_value=round(total_page_kb, 2),
                    budget_value=page_weight_max_kb,
                    message=f"Total page weight for {rel_str} ({total_page_kb:.1f} KB) exceeds {page_weight_max_kb} KB budget",
                )
            )

        # CSP check: must contain meta CSP tag
        if '<meta http-equiv="Content-Security-Policy"' not in content:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.CSP_INLINE,
                    measured_value="missing_csp_meta",
                    budget_value="csp_meta_required",
                    message=f"{rel_str} is missing Content-Security-Policy meta tag",
                )
            )

        # Zero inline <style> tags permitted
        inline_styles = re.findall(
            r"<style\b[^>]*>(.*?)</style>", content, flags=re.DOTALL | re.IGNORECASE
        )
        if inline_styles:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.CSP_INLINE,
                    measured_value=len(inline_styles),
                    budget_value=0,
                    message=f"{rel_str} contains {len(inline_styles)} forbidden inline <style> block(s)",
                )
            )

        # Zero inline <script> tags permitted (except external script tags with src=)
        inline_scripts = re.findall(
            r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Filter out empty script tags or non-executable types
        executable_inline_scripts = [s for s in inline_scripts if s.strip()]
        if executable_inline_scripts:
            violations.append(
                Violation(
                    file_path=rel_str,
                    violation_type=ViolationType.CSP_INLINE,
                    measured_value=len(executable_inline_scripts),
                    budget_value=0,
                    message=f"{rel_str} contains {len(executable_inline_scripts)} forbidden inline <script> block(s)",
                )
            )

        # Accessibility checks on forecast pages (index.html, ur/index.html)
        if rel_str in ["index.html", "ur/index.html"]:
            # Landmark: exactly one <h1>
            h1_tags = re.findall(r"<h1\b[^>]*>", content, flags=re.IGNORECASE)
            if len(h1_tags) != 1:
                violations.append(
                    Violation(
                        file_path=rel_str,
                        violation_type=ViolationType.ACCESSIBILITY,
                        measured_value=len(h1_tags),
                        budget_value=1,
                        message=f"{rel_str} has {len(h1_tags)} <h1> elements; exactly 1 is required",
                    )
                )

            # Skip link check
            if (
                'class="skip-link"' not in content
                and 'href="#forecast"' not in content
                and 'href="#main"' not in content
            ):
                violations.append(
                    Violation(
                        file_path=rel_str,
                        violation_type=ViolationType.ACCESSIBILITY,
                        measured_value="missing_skip_link",
                        budget_value="skip_link_required",
                        message=f"{rel_str} is missing an accessible skip link",
                    )
                )

            # Static valid-until line check
            if (
                "valid-until" not in content.lower()
                and "درست" not in content
                and "valid_until" not in content.lower()
            ):
                violations.append(
                    Violation(
                        file_path=rel_str,
                        violation_type=ViolationType.ACCESSIBILITY,
                        measured_value="missing_valid_until",
                        budget_value="valid_until_notice_required",
                        message=f"{rel_str} is missing static 'Valid until' notice",
                    )
                )

            # Hreflang reciprocal links
            expected_alt = "ur" if rel_str == "index.html" else "en"
            if f'hreflang="{expected_alt}"' not in content:
                violations.append(
                    Violation(
                        file_path=rel_str,
                        violation_type=ViolationType.ACCESSIBILITY,
                        measured_value="missing_hreflang",
                        budget_value=f'hreflang="{expected_alt}"',
                        message=f"{rel_str} is missing reciprocal hreflang={expected_alt} link",
                    )
                )

    if raise_on_breach and violations:
        summary_msg = f"Site audit failed with {len(violations)} violation(s):\n" + "\n".join(
            f"  - [{v.violation_type}] {v.file_path}: {v.message}" for v in violations
        )
        raise BudgetBreachError(summary_msg)

    return violations


def format_violations_table(violations: list[Violation]) -> str:
    """Format violations into a readable terminal summary table."""
    if not violations:
        return "Audit passed cleanly: 0 violations detected."

    lines: list[str] = [
        f"{'TYPE':<16} | {'FILE':<24} | {'MEASURED':<10} | {'BUDGET':<10} | MESSAGE",
        "-" * 90,
    ]
    lines.extend(
        f"{v.violation_type:<16} | {v.file_path:<24} | {v.measured_value!s:<10} | {v.budget_value!s:<10} | {v.message}"
        for v in violations
    )
    return "\n".join(lines)
