"""Unit tests for Pillow + Raqm card renderer, bidirectional text layout, and size budgets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.visualization.cards import (
    CardFormat,
    render_all_cards,
    render_card,
    save_card,
)
from smogsense.visualization.fan_chart import FanChartLayout, compute_fan_chart_geometry
from smogsense.visualization.text_layout import (
    draw_multiline_text,
    load_card_font,
    measure_multiline_text,
    wrap_text_multiline,
)

pytestmark = pytest.mark.unit


def load_fixture_bulletin() -> dict[str, Any]:
    """Load sample valid bulletin fixture."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def test_text_layout_wrapping_and_measurement() -> None:
    """Verify that wrap_text_multiline and measure_multiline_text wrap and size text accurately."""
    font = load_card_font(script="latin", size=24, bold=False)
    canvas = Image.new("RGB", (800, 400), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    sample_text = (
        "Stay indoors and limit heavy outdoor exertion during the early morning and late evening "
        "when particulate pollution levels are highest across the city."
    )

    # 1. Wrap text with constrained max_width
    max_w = 300
    lines = wrap_text_multiline(sample_text, font=font, max_width=max_w, draw=draw)
    assert len(lines) >= 3, f"Expected multi-line wrap, got {len(lines)} lines"

    # Verify no line exceeds max_width (with small margin for kerning)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        assert w <= max_w + 10, f"Line '{line}' width {w} exceeded max_width {max_w}"

    # 2. Measurement
    measured_w, measured_h = measure_multiline_text(
        lines, font=font, line_height_factor=1.5, draw=draw
    )
    assert 0 < measured_w <= max_w + 10
    assert measured_h > 0

    # 3. Drawing
    end_y = draw_multiline_text(
        draw=draw,
        xy=(50, 50),
        lines=lines,
        font=font,
        fill=(0, 0, 0),
        line_height_factor=1.5,
        align="left",
    )
    assert end_y > 50


def test_render_whatsapp_post_card_en_and_ur() -> None:
    """Verify rendering of 1080x1350 WhatsApp post cards in both English and Urdu."""
    bulletin = load_fixture_bulletin()

    for lang in ["en", "ur"]:
        im = render_card(bulletin, lang=lang, format=CardFormat.WHATSAPP_POST)

        assert isinstance(im, Image.Image)
        assert im.size == (1080, 1350)
        assert im.mode == "RGB"

        # Verify image contains non-trivial content (not a blank canvas)
        extrema = cast(tuple[tuple[int, int], ...], im.getextrema())
        assert extrema is not None
        # Verify color variation exists across all channels
        for ch_min, ch_max in extrema:
            assert ch_max > ch_min


def test_render_open_graph_card_en_and_ur() -> None:
    """Verify rendering of 1200x630 Open Graph cards in both English and Urdu."""
    bulletin = load_fixture_bulletin()

    for lang in ["en", "ur"]:
        im = render_card(bulletin, lang=lang, format=CardFormat.OPEN_GRAPH)

        assert isinstance(im, Image.Image)
        assert im.size == (1200, 630)
        assert im.mode == "RGB"

        extrema = cast(tuple[tuple[int, int], ...], im.getextrema())
        assert extrema is not None
        for ch_min, ch_max in extrema:
            assert ch_max > ch_min


def test_card_file_size_budgets(tmp_path: Path) -> None:
    """Verify that exported quantized PNGs satisfy strict file budgets (<=400KB post, <=300KB OG)."""
    bulletin = load_fixture_bulletin()

    for lang in ["en", "ur"]:
        # 1. WhatsApp post: budget <= 400 KB
        im_post = render_card(bulletin, lang=lang, format=CardFormat.WHATSAPP_POST)
        post_path = tmp_path / f"post-{lang}.png"
        post_bytes = save_card(im_post, post_path, optimize=True)
        post_kb = post_bytes / 1024.0

        assert post_path.is_file()
        assert post_bytes > 5 * 1024, "Image unexpectedly small/empty"
        assert post_kb <= 400.0, f"WhatsApp post ({post_kb:.1f} KB) exceeded 400 KB budget"

        # 2. Open Graph: budget <= 300 KB
        im_og = render_card(bulletin, lang=lang, format=CardFormat.OPEN_GRAPH)
        og_path = tmp_path / f"og-{lang}.png"
        og_bytes = save_card(im_og, og_path, optimize=True)
        og_kb = og_bytes / 1024.0

        assert og_path.is_file()
        assert og_bytes > 5 * 1024, "Image unexpectedly small/empty"
        assert og_kb <= 300.0, f"Open Graph ({og_kb:.1f} KB) exceeded 300 KB budget"


def test_fan_chart_geometry_reuse_on_canvas() -> None:
    """Verify that FanChartLayout coordinates are directly reusable for Pillow drawing."""
    bulletin = load_fixture_bulletin()
    layout = FanChartLayout(
        width=1000.0,
        height=500.0,
        margin_left=100.0,
        margin_right=50.0,
        margin_top=40.0,
        margin_bottom=80.0,
    )
    geom = compute_fan_chart_geometry(bulletin=bulletin, layout=layout)

    # Geometry bounds must be strictly within layout canvas
    assert geom.plot_x == 100.0
    assert geom.plot_width == 850.0
    assert geom.plot_height == 380.0
    assert len(geom.columns) == 3

    # All columns must lie strictly inside plot horizontal span
    for col in geom.columns:
        assert 100.0 <= col.x_center <= 950.0
        assert col.bar_80.width > 0
        assert col.bar_50.width > 0


def test_render_all_cards(tmp_path: Path) -> None:
    """Verify that render_all_cards produces all 4 cards in latest and dated directory."""
    bulletin = load_fixture_bulletin()
    out_dir = tmp_path / "cards"

    artifacts = render_all_cards(bulletin=bulletin, output_dir=out_dir)

    assert "whatsapp_en" in artifacts
    assert "whatsapp_ur" in artifacts
    assert "og_en" in artifacts
    assert "og_ur" in artifacts

    latest_dir = out_dir / "latest"
    assert (latest_dir / "lahore-whatsapp-en.png").is_file()
    assert (latest_dir / "lahore-whatsapp-ur.png").is_file()
    assert (latest_dir / "lahore-og-en.png").is_file()
    assert (latest_dir / "lahore-og-ur.png").is_file()

    # Dated directory (2025-11-05 from fixture)
    dated_dir = out_dir / "2025-11-05"
    assert dated_dir.is_dir()
    assert (dated_dir / "lahore-whatsapp-en.png").is_file()
    assert (dated_dir / "lahore-og-en.png").is_file()


def test_cli_render_cards_command(tmp_path: Path) -> None:
    """Verify CLI site render-cards and cards render commands execute cleanly via CliRunner."""
    runner = CliRunner()
    fixture_path = "tests/fixtures/bulletin_valid_example.json"
    out_dir = str(tmp_path / "cli_cards")

    # 1. Test 'site render-cards'
    res1 = runner.invoke(
        app,
        [
            "site",
            "render-cards",
            "--bulletin",
            fixture_path,
            "--output-dir",
            out_dir,
            "--format",
            "all",
            "--lang",
            "all",
        ],
    )
    assert res1.exit_code == 0
    assert "Rendered lahore-whatsapp-en.png" in res1.stdout
    assert "Rendered lahore-og-en.png" in res1.stdout

    # 2. Test 'cards render' alias
    out_dir2 = str(tmp_path / "cli_cards_alias")
    res2 = runner.invoke(
        app,
        [
            "cards",
            "render",
            "--bulletin",
            fixture_path,
            "--output-dir",
            out_dir2,
            "--format",
            "post",
            "--lang",
            "en",
        ],
    )
    assert res2.exit_code == 0
    assert "Rendered lahore-whatsapp-en.png" in res2.stdout
