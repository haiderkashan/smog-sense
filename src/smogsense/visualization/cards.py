"""smogsense.visualization.cards — Pillow + Raqm renderer for WhatsApp cards and Open Graph images.

Draws 1080x1350 cards and 1200x630 OG images with 3x supersampling; all text (English and Urdu
Nastaliq) goes through Pillow's Raqm layout engine with direction='rtl', language='ur' when available.
Adaptive 256-color palette quantization guarantees strict performance budgets (<= 400 KB for post,
<= 300 KB for OG).

Specification: docs/dissemination-and-ui.md → 'WhatsApp-ready cards'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from smogsense.publishing.i18n import category_name, load_catalogs, local_time
from smogsense.visualization.fan_chart import (
    FanChartGeometry,
    FanChartLayout,
    compute_fan_chart_geometry,
)
from smogsense.visualization.text_layout import (
    draw_multiline_text,
    load_card_font,
    wrap_text_multiline,
)

logger = logging.getLogger(__name__)

# Design system color tokens (RGB tuples)
COLOR_BG = (244, 241, 236)  # --haze-50 (#F4F1EC)
COLOR_SURFACE = (235, 230, 222)  # Card box neutral
COLOR_SURFACE_ALT = (226, 221, 212)
COLOR_INK = (27, 31, 42)  # --ink-900 (#1B1F2A)
COLOR_MUTED = (91, 86, 80)  # --ash-600 (#5B5650)
COLOR_RULE = (217, 211, 202)  # --haze-200 (#D9D3CA)
COLOR_RAVI = (26, 82, 112)  # --ravi-700 (#1A5270)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Quantile band fills
COLOR_BAND_80 = (196, 206, 209)  # Ravi blue mixed with haze bg ~22%
COLOR_BAND_50 = (146, 173, 184)  # Ravi blue mixed with haze bg ~45%

# Category identity colors
CATEGORY_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "good": ((0, 228, 0), COLOR_BLACK),
    "moderate": ((255, 255, 0), COLOR_BLACK),
    "usg": ((255, 126, 0), COLOR_BLACK),
    "unhealthy": ((255, 0, 0), COLOR_BLACK),
    "very_unhealthy": ((143, 63, 151), COLOR_WHITE),
    "hazardous": ((126, 0, 35), COLOR_WHITE),
}


class CardFormat(StrEnum):
    """Supported export dimensions and aspect ratios for social cards."""

    WHATSAPP_POST = "whatsapp_post"
    OPEN_GRAPH = "open_graph"
    STATUS_STORY = "status_story"


@dataclass(frozen=True)
class CardSpec:
    """Format dimensions and maximum size budget in kilobytes."""

    name: str
    width: int
    height: int
    max_kb: int
    supersample: int = 3


FORMAT_SPECS: dict[CardFormat, CardSpec] = {
    CardFormat.WHATSAPP_POST: CardSpec(
        name="whatsapp_post",
        width=1080,
        height=1350,
        max_kb=400,
        supersample=3,
    ),
    CardFormat.OPEN_GRAPH: CardSpec(
        name="open_graph",
        width=1200,
        height=630,
        max_kb=300,
        supersample=3,
    ),
    CardFormat.STATUS_STORY: CardSpec(
        name="status_story",
        width=1080,
        height=1920,
        max_kb=450,
        supersample=3,
    ),
}


def _blend_rgb(
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Alpha blend foreground color over background color."""
    return (
        round(fg[0] * alpha + bg[0] * (1.0 - alpha)),
        round(fg[1] * alpha + bg[1] * (1.0 - alpha)),
        round(fg[2] * alpha + bg[2] * (1.0 - alpha)),
    )


def _draw_non_colour_cues(
    draw: ImageDraw.ImageDraw,
    cat_id: str,
    box: tuple[float, float, float, float],
    step: int = 36,
) -> None:
    """Draw distinctive accessible pattern cues for red-ish categories (WCAG 2.2 AA non-colour)."""
    x1, y1, x2, y2 = box
    width = int(x2 - x1)
    height = int(y2 - y1)

    if cat_id == "unhealthy":
        # Diagonal stripes (//)
        line_color = (255, 0, 0)
        for offset in range(-height, width + height, step):
            draw.line(
                [(x1 + offset, y2), (x1 + offset + height, y1)],
                fill=line_color,
                width=3,
            )
    elif cat_id == "very_unhealthy":
        # Cross-hatch (XX)
        line_color = (143, 63, 151)
        for offset in range(-height, width + height, step):
            draw.line(
                [(x1 + offset, y2), (x1 + offset + height, y1)],
                fill=line_color,
                width=2,
            )
            draw.line(
                [(x1 + offset, y1), (x1 + offset + height, y2)],
                fill=line_color,
                width=2,
            )
    elif cat_id == "hazardous":
        # Stipple dots grid
        dot_color = (126, 0, 35)
        dot_r = 4
        for dx in range(int(x1) + 12, int(x2) - 8, step):
            for dy in range(int(y1) + 12, int(y2) - 8, step):
                draw.ellipse(
                    [(dx - dot_r, dy - dot_r), (dx + dot_r, dy + dot_r)],
                    fill=dot_color,
                )


def _draw_canvas_fan_chart(
    draw: ImageDraw.ImageDraw,
    geom: FanChartGeometry,
    offset_x: int,
    offset_y: int,
    font_ticks: Any,
    font_labels: Any,
    unit_label: str = "µg/m³",
) -> None:
    """Draw the 3-day range fan chart directly onto the Pillow canvas."""
    # 1. Horizontal Category Bands
    for band in geom.category_bands:
        cat_bg, _ = CATEGORY_COLORS.get(band.category_id, (COLOR_INK, COLOR_WHITE))
        blended_bg = _blend_rgb(cat_bg, COLOR_BG, alpha=0.18)

        bx1 = offset_x + band.x
        by1 = offset_y + band.y_top
        bx2 = bx1 + band.width
        by2 = by1 + band.height

        draw.rectangle([(bx1, by1), (bx2, by2)], fill=blended_bg)

        # Accessible non-colour pattern overlays for red-ish categories
        if band.category_id in ["unhealthy", "very_unhealthy", "hazardous"]:
            _draw_non_colour_cues(draw, band.category_id, (bx1, by1, bx2, by2), step=48)

    # 2. Grid Rule Lines & Boundary Ticks
    for tick in geom.ticks:
        tx1 = offset_x + tick.x1
        ty = offset_y + tick.y
        tx2 = offset_x + tick.x2
        draw.line([(tx1, ty), (tx2, ty)], fill=COLOR_RULE, width=2)

        # Tick numeric label
        draw.text(
            (tx1 - 18, ty - 12),
            tick.label,
            font=font_ticks,
            fill=COLOR_MUTED,
            anchor="ra",
        )

    # Unit indicator above the Y-axis
    draw.text(
        (offset_x + geom.plot_x - 18, offset_y + geom.plot_y - 45),
        unit_label,
        font=font_ticks,
        fill=COLOR_MUTED,
        anchor="ra",
    )

    # 3. Median Connecting Line
    if len(geom.median_line_points) >= 2:
        pts = [(offset_x + p[0], offset_y + p[1]) for p in geom.median_line_points]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=COLOR_MUTED, width=4)

    # 4. Forecast Horizon Columns
    for col in geom.columns:
        col_cx = offset_x + col.x_center

        # Vertical guide dashed line
        draw.line(
            [
                (col_cx, offset_y + geom.plot_y),
                (col_cx, offset_y + geom.plot_y + geom.plot_height),
            ],
            fill=COLOR_RULE,
            width=2,
        )

        # 80% range bar
        b80_x1 = offset_x + col.bar_80.x
        b80_y1 = offset_y + col.bar_80.y
        b80_x2 = b80_x1 + col.bar_80.width
        b80_y2 = b80_y1 + col.bar_80.height
        draw.rounded_rectangle(
            [(b80_x1, b80_y1), (b80_x2, b80_y2)],
            radius=int(col.bar_80.rx * 2),
            fill=COLOR_BAND_80,
            outline=COLOR_RAVI,
            width=2,
        )

        # 50% range bar
        b50_x1 = offset_x + col.bar_50.x
        b50_y1 = offset_y + col.bar_50.y
        b50_x2 = b50_x1 + col.bar_50.width
        b50_y2 = b50_y1 + col.bar_50.height
        draw.rounded_rectangle(
            [(b50_x1, b50_y1), (b50_x2, b50_y2)],
            radius=int(col.bar_50.rx * 2),
            fill=COLOR_BAND_50,
            outline=COLOR_RAVI,
            width=2,
        )

        # Median tick line & marker
        med_y = offset_y + col.median.y
        med_x1 = offset_x + col.median.tick_x1
        med_x2 = offset_x + col.median.tick_x2
        draw.line([(med_x1, med_y), (med_x2, med_y)], fill=COLOR_INK, width=8)
        draw.ellipse([(col_cx - 8, med_y - 8), (col_cx + 8, med_y + 8)], fill=COLOR_INK)

        # Lead time label (+24h, +48h, +72h)
        lead_str = f"+{col.lead_h}h"
        draw.text(
            (col_cx, offset_y + geom.plot_y + geom.plot_height + 40),
            lead_str,
            font=font_labels,
            fill=COLOR_INK,
            anchor="mm",
        )

        # Median numeric value
        val_str = f"{col.median.value:.0f} {unit_label}"
        draw.text(
            (col_cx, offset_y + geom.plot_y + geom.plot_height + 85),
            val_str,
            font=font_ticks,
            fill=COLOR_MUTED,
            anchor="mm",
        )


def _render_whatsapp_post(
    bulletin: dict[str, Any],
    lang: str = "en",
) -> Image.Image:
    """Render 1080x1350 WhatsApp post card with 3x supersampling."""
    scale = 3
    canvas_w = 1080 * scale  # 3240
    canvas_h = 1350 * scale  # 4050

    im = Image.new("RGB", (canvas_w, canvas_h), COLOR_BG)
    draw = ImageDraw.Draw(im)

    catalogs = load_catalogs()
    cat = catalogs.get(lang, catalogs["en"])
    is_urdu = lang == "ur"
    script = "urdu" if is_urdu else "latin"
    direction = "rtl" if is_urdu else "ltr"
    lh_factor = 2.1 if is_urdu else 1.5

    # Load typography at 3x scale
    font_brand = load_card_font(script=script, size=64 * scale, bold=True)
    font_date = load_card_font(script=script, size=32 * scale, bold=False)
    font_chip = load_card_font(script=script, size=40 * scale, bold=True)
    font_hero = load_card_font(script="latin", size=115 * scale, bold=True)
    font_unit = load_card_font(script=script, size=38 * scale, bold=False)
    font_sub = load_card_font(script=script, size=30 * scale, bold=False)
    font_advisory = load_card_font(script=script, size=36 * scale, bold=True)
    font_note = load_card_font(script=script, size=28 * scale, bold=False)
    font_footer = load_card_font(script=script, size=24 * scale, bold=False)
    font_ticks = load_card_font(script="latin", size=24 * scale, bold=False)
    font_labels = load_card_font(script="latin", size=30 * scale, bold=True)

    pad_x = 140 * scale
    curr_y = 100 * scale

    # Outer decorative card border
    draw.rounded_rectangle(
        [(40 * scale, 40 * scale), (canvas_w - 40 * scale, canvas_h - 40 * scale)],
        radius=30 * scale,
        outline=COLOR_RULE,
        width=4 * scale,
    )

    # 1. Header (Title & Local Date)
    title_text = cat.get("card", {}).get(
        "title",
        "لاہور اسموگ کی پیش گوئی" if is_urdu else "Lahore Smog Forecast",
    )
    draw.text(
        (canvas_w // 2, curr_y),
        title_text,
        font=font_brand,
        fill=COLOR_INK,
        anchor="mt",
    )
    curr_y += 85 * scale

    gen_time_str = bulletin.get("generated_at_utc", bulletin.get("issuance_utc", ""))
    date_formatted = local_time(gen_time_str, "%A, %d %b %Y")
    draw.text(
        (canvas_w // 2, curr_y),
        date_formatted,
        font=font_date,
        fill=COLOR_MUTED,
        anchor="mt",
    )
    curr_y += 65 * scale

    # 2. AQI Category Chip
    h1 = bulletin.get("horizons", [{}])[0]
    cat_median = h1.get("category", {}).get("median_q50", "hazardous")
    chip_colors = CATEGORY_COLORS.get(cat_median, (COLOR_INK, COLOR_WHITE))
    chip_bg, chip_fg = chip_colors
    chip_name = category_name(cat_median, lang=lang).upper()

    chip_w = 620 * scale
    chip_h = 70 * scale
    chip_x1 = (canvas_w - chip_w) // 2
    chip_y1 = curr_y
    chip_x2 = chip_x1 + chip_w
    chip_y2 = chip_y1 + chip_h

    draw.rounded_rectangle(
        [(chip_x1, chip_y1), (chip_x2, chip_y2)],
        radius=35 * scale,
        fill=chip_bg,
        outline=COLOR_INK,
        width=3 * scale,
    )
    # Non-colour cue inside chip for red-ish categories
    if cat_median in ["unhealthy", "very_unhealthy", "hazardous"]:
        _draw_non_colour_cues(
            draw,
            cat_median,
            (float(chip_x1), float(chip_y1), float(chip_x2), float(chip_y2)),
            step=28 * scale,
        )

    draw.text(
        (canvas_w // 2, chip_y1 + chip_h // 2),
        chip_name,
        font=font_chip,
        fill=chip_fg,
        anchor="mm",
    )
    curr_y += chip_h + 35 * scale

    # 3. Hero Metric
    q_dict = h1.get("quantiles_ugm3", {})
    q50_val = q_dict.get("q50", 0.0)
    q90_val = q_dict.get("q90", 0.0)

    hero_str = f"{q50_val:.0f}"
    unit_str = cat.get("units", {}).get("ugm3", "µg/m³")

    draw.text(
        (canvas_w // 2 - 80 * scale, curr_y),
        hero_str,
        font=font_hero,
        fill=COLOR_INK,
        anchor="mt",
    )
    draw.text(
        (canvas_w // 2 + 160 * scale, curr_y + 40 * scale),
        unit_str,
        font=font_unit,
        fill=COLOR_MUTED,
        anchor="la",
    )
    curr_y += 125 * scale

    # Most likely & Bad case sub-label
    most_likely_lbl = cat.get("card", {}).get(
        "most_likely", "سب سے ممکنہ" if is_urdu else "Most likely"
    )
    bad_case_lbl = cat.get("card", {}).get(
        "bad_case", "بدترین امکان" if is_urdu else "Bad case (1 in 10)"
    )
    sub_text = f"{most_likely_lbl} {q50_val:.0f}  ·  {bad_case_lbl} {q90_val:.0f} {unit_str}"
    draw.text(
        (canvas_w // 2, curr_y),
        sub_text,
        font=font_sub,
        fill=COLOR_MUTED,
        anchor="mt",
    )
    curr_y += 65 * scale

    # 4. 3-Day Fan Chart Range Plot
    chart_box_w = canvas_w - 2 * pad_x
    chart_box_h = 360 * scale
    chart_offset_x = pad_x
    chart_offset_y = curr_y

    # Compute geometry for card canvas
    fan_layout = FanChartLayout(
        width=float(chart_box_w),
        height=float(chart_box_h),
        margin_left=90.0 * scale,
        margin_right=40.0 * scale,
        margin_top=25.0 * scale,
        margin_bottom=75.0 * scale,
        bar_80_width=52.0 * scale,
        bar_50_width=34.0 * scale,
        median_tick_width=42.0 * scale,
    )
    geom = compute_fan_chart_geometry(bulletin=bulletin, layout=fan_layout)

    _draw_canvas_fan_chart(
        draw=draw,
        geom=geom,
        offset_x=chart_offset_x,
        offset_y=chart_offset_y,
        font_ticks=font_ticks,
        font_labels=font_labels,
        unit_label=unit_str,
    )
    curr_y += chart_box_h + 30 * scale

    # 5. Protective Action Guidance Box
    advice_actions = cat.get("advice", {}).get(cat_median, {}).get("actions", [])
    advisory_action = (
        advice_actions[0] if advice_actions else "Stay indoors and limit heavy outdoor exertion."
    )

    adv_box_x1 = pad_x
    adv_box_y1 = curr_y
    adv_box_w = canvas_w - 2 * pad_x
    adv_box_h = 105 * scale
    adv_box_x2 = adv_box_x1 + adv_box_w
    adv_box_y2 = adv_box_y1 + adv_box_h

    draw.rounded_rectangle(
        [(adv_box_x1, adv_box_y1), (adv_box_x2, adv_box_y2)],
        radius=16 * scale,
        fill=COLOR_SURFACE,
        outline=COLOR_RULE,
        width=2 * scale,
    )

    wrapped_adv = wrap_text_multiline(
        text=advisory_action,
        font=font_advisory,
        max_width=adv_box_w - 60 * scale,
        draw=draw,
        direction=direction,
        language="ur" if is_urdu else None,
    )
    adv_text_y = adv_box_y1 + (adv_box_h - len(wrapped_adv) * int(36 * scale * lh_factor)) // 2
    draw_multiline_text(
        draw=draw,
        xy=(canvas_w // 2, adv_text_y),
        lines=wrapped_adv[:2],
        font=font_advisory,
        fill=COLOR_INK,
        line_height_factor=lh_factor,
        align="center",
        direction=direction,
        language="ur" if is_urdu else None,
    )
    curr_y += adv_box_h + 30 * scale

    # 6. Uncertainty Statement
    uncertainty_txt = (
        "متوقع دائرہ 10 میں سے 8 دن کا احاطہ کرتا ہے · 10 میں سے 1 دن اس سے بھی بدتر ہو سکتا ہے"
        if is_urdu
        else "Likely range covers 8 days in 10 · 1 in 10 is worse"
    )
    draw.text(
        (canvas_w // 2, curr_y),
        uncertainty_txt,
        font=font_note,
        fill=COLOR_MUTED,
        anchor="mt",
    )
    curr_y += 50 * scale

    # 7. Footer (Expiry & Source Attribution)
    valid_until_raw = bulletin.get("valid_until_utc", "")
    valid_until_local = local_time(valid_until_raw, "%d %b, %I:%M %p")
    footer_text = (
        f"{valid_until_local} تک درست  ·  اسموگ سینس  ·  smogsense.org"
        if is_urdu
        else f"Valid until {valid_until_local}  ·  SmogSense  ·  smogsense.org"
    )
    draw.text(
        (canvas_w // 2, canvas_h - 90 * scale),
        footer_text,
        font=font_footer,
        fill=COLOR_MUTED,
        anchor="mb",
    )

    # Downscale 3x supersampled canvas with LANCZOS for crisp edges
    return im.resize((1080, 1350), resample=Image.Resampling.LANCZOS)


def _render_open_graph(
    bulletin: dict[str, Any],
    lang: str = "en",
) -> Image.Image:
    """Render 1200x630 Open Graph preview image (landscape split layout)."""
    scale = 3
    canvas_w = 1200 * scale  # 3600
    canvas_h = 630 * scale  # 1890

    im = Image.new("RGB", (canvas_w, canvas_h), COLOR_BG)
    draw = ImageDraw.Draw(im)

    catalogs = load_catalogs()
    cat = catalogs.get(lang, catalogs["en"])
    is_urdu = lang == "ur"
    script = "urdu" if is_urdu else "latin"
    direction = "rtl" if is_urdu else "ltr"
    lh_factor = 2.1 if is_urdu else 1.5

    font_brand = load_card_font(script=script, size=52 * scale, bold=True)
    font_date = load_card_font(script=script, size=26 * scale, bold=False)
    font_chip = load_card_font(script=script, size=34 * scale, bold=True)
    font_hero = load_card_font(script="latin", size=95 * scale, bold=True)
    font_unit = load_card_font(script=script, size=32 * scale, bold=False)
    font_sub = load_card_font(script=script, size=24 * scale, bold=False)
    font_advice = load_card_font(script=script, size=28 * scale, bold=True)
    font_footer = load_card_font(script=script, size=22 * scale, bold=False)
    font_ticks = load_card_font(script="latin", size=20 * scale, bold=False)
    font_labels = load_card_font(script="latin", size=26 * scale, bold=True)

    # Outer decorative card border
    draw.rounded_rectangle(
        [(30 * scale, 30 * scale), (canvas_w - 30 * scale, canvas_h - 30 * scale)],
        radius=24 * scale,
        outline=COLOR_RULE,
        width=3 * scale,
    )

    # Split: Left side (info), Right side (3-day chart)
    split_x = int(canvas_w * 0.52)

    # Divider line
    draw.line(
        [(split_x, 60 * scale), (split_x, canvas_h - 60 * scale)],
        fill=COLOR_RULE,
        width=2 * scale,
    )

    left_pad = 80 * scale
    curr_y = 70 * scale

    # Left Column: Brand & Date
    title_text = "لاہور اسموگ کی پیش گوئی" if is_urdu else "SmogSense Lahore"
    draw.text(
        (left_pad, curr_y),
        title_text,
        font=font_brand,
        fill=COLOR_INK,
    )
    curr_y += 65 * scale

    gen_time_str = bulletin.get("generated_at_utc", bulletin.get("issuance_utc", ""))
    date_formatted = local_time(gen_time_str, "%A, %d %b %Y")
    draw.text(
        (left_pad, curr_y),
        date_formatted,
        font=font_date,
        fill=COLOR_MUTED,
    )
    curr_y += 55 * scale

    # AQI Chip
    h1 = bulletin.get("horizons", [{}])[0]
    cat_median = h1.get("category", {}).get("median_q50", "hazardous")
    chip_colors = CATEGORY_COLORS.get(cat_median, (COLOR_INK, COLOR_WHITE))
    chip_bg, chip_fg = chip_colors
    chip_name = category_name(cat_median, lang=lang).upper()

    chip_w = 460 * scale
    chip_h = 56 * scale
    draw.rounded_rectangle(
        [(left_pad, curr_y), (left_pad + chip_w, curr_y + chip_h)],
        radius=28 * scale,
        fill=chip_bg,
        outline=COLOR_INK,
        width=2 * scale,
    )
    if cat_median in ["unhealthy", "very_unhealthy", "hazardous"]:
        _draw_non_colour_cues(
            draw,
            cat_median,
            (float(left_pad), float(curr_y), float(left_pad + chip_w), float(curr_y + chip_h)),
            step=24 * scale,
        )

    draw.text(
        (left_pad + chip_w // 2, curr_y + chip_h // 2),
        chip_name,
        font=font_chip,
        fill=chip_fg,
        anchor="mm",
    )
    curr_y += chip_h + 35 * scale

    # Hero concentration
    q_dict = h1.get("quantiles_ugm3", {})
    q50_val = q_dict.get("q50", 0.0)
    q90_val = q_dict.get("q90", 0.0)
    hero_str = f"{q50_val:.0f}"
    unit_str = cat.get("units", {}).get("ugm3", "µg/m³")

    draw.text(
        (left_pad, curr_y),
        hero_str,
        font=font_hero,
        fill=COLOR_INK,
    )
    draw.text(
        (left_pad + 200 * scale, curr_y + 40 * scale),
        unit_str,
        font=font_unit,
        fill=COLOR_MUTED,
    )
    curr_y += 110 * scale

    most_likely_lbl = cat.get("card", {}).get("most_likely", "Most likely")
    bad_case_lbl = cat.get("card", {}).get("bad_case", "Bad case")
    sub_text = f"{most_likely_lbl} {q50_val:.0f}  ·  {bad_case_lbl} {q90_val:.0f}"
    draw.text(
        (left_pad, curr_y),
        sub_text,
        font=font_sub,
        fill=COLOR_MUTED,
    )
    curr_y += 50 * scale

    # Advisory snippet
    advice_actions = cat.get("advice", {}).get(cat_median, {}).get("actions", [])
    advisory_action = advice_actions[0] if advice_actions else "Limit outdoor activity."
    wrapped_adv = wrap_text_multiline(
        text=advisory_action,
        font=font_advice,
        max_width=split_x - left_pad - 40 * scale,
        draw=draw,
        direction=direction,
        language="ur" if is_urdu else None,
    )
    draw_multiline_text(
        draw=draw,
        xy=(left_pad, curr_y),
        lines=wrapped_adv[:2],
        font=font_advice,
        fill=COLOR_INK,
        line_height_factor=lh_factor,
        align="left",
        direction=direction,
        language="ur" if is_urdu else None,
    )

    # Footer validity
    valid_until_raw = bulletin.get("valid_until_utc", "")
    valid_until_local = local_time(valid_until_raw, "%d %b, %I:%M %p")
    footer_txt = (
        f"{valid_until_local} تک درست  ·  smogsense.org"
        if is_urdu
        else f"Valid until {valid_until_local}  ·  smogsense.org"
    )
    draw.text(
        (left_pad, canvas_h - 70 * scale),
        footer_txt,
        font=font_footer,
        fill=COLOR_MUTED,
    )

    # Right Column: 3-Day Fan Chart
    chart_x = split_x + 60 * scale
    chart_y = 60 * scale
    chart_w = canvas_w - chart_x - 60 * scale
    chart_h = canvas_h - 120 * scale

    fan_layout = FanChartLayout(
        width=float(chart_w),
        height=float(chart_h),
        margin_left=75.0 * scale,
        margin_right=30.0 * scale,
        margin_top=25.0 * scale,
        margin_bottom=65.0 * scale,
        bar_80_width=44.0 * scale,
        bar_50_width=28.0 * scale,
        median_tick_width=36.0 * scale,
    )
    geom = compute_fan_chart_geometry(bulletin=bulletin, layout=fan_layout)

    _draw_canvas_fan_chart(
        draw=draw,
        geom=geom,
        offset_x=chart_x,
        offset_y=chart_y,
        font_ticks=font_ticks,
        font_labels=font_labels,
        unit_label=unit_str,
    )

    return im.resize((1200, 630), resample=Image.Resampling.LANCZOS)


def render_card(
    bulletin: dict[str, Any],
    lang: str = "en",
    format: CardFormat | str = CardFormat.WHATSAPP_POST,
) -> Image.Image:
    """Render a social card image for the specified language and format.

    Parameters
    ----------
    bulletin : dict[str, Any]
        Parsed bulletin dictionary adhering to bulletin.schema.json.
    lang : str
        Target language code ('en' or 'ur').
    format : CardFormat | str
        Target card format (whatsapp_post, open_graph, status_story).

    Returns
    -------
    Image.Image
        Rendered RGB image.
    """
    card_fmt = CardFormat(format)
    if card_fmt == CardFormat.WHATSAPP_POST:
        return _render_whatsapp_post(bulletin=bulletin, lang=lang)
    if card_fmt == CardFormat.OPEN_GRAPH:
        return _render_open_graph(bulletin=bulletin, lang=lang)
    # Default fallback to whatsapp post
    return _render_whatsapp_post(bulletin=bulletin, lang=lang)


def save_card(
    image: Image.Image,
    output_path: Path | str,
    optimize: bool = True,
    max_colors: int = 256,
) -> int:
    """Save rendered card image to disk with adaptive palette quantization.

    Parameters
    ----------
    image : Image.Image
        Input PIL image.
    output_path : Path | str
        Destination file path (.png or .jpg).
    optimize : bool
        Whether to enable PNG compression optimization.
    max_colors : int
        Maximum palette colors for PNG quantization (default 256).

    Returns
    -------
    int
        Saved file size in bytes.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_file.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        rgb_im = image.convert("RGB")
        rgb_im.save(out_file, format="JPEG", quality=88, subsampling=0)
    else:
        # Default PNG with adaptive palette quantization
        quantized = image.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
        quantized.save(out_file, format="PNG", optimize=optimize)

    return out_file.stat().st_size


def render_all_cards(
    bulletin: dict[str, Any],
    output_dir: Path | str = "site/cards",
) -> dict[str, Path]:
    """Generate all standard social cards (WhatsApp post & Open Graph, en & ur).

    Outputs both to latest/ and dated directory YYYY-MM-DD/.

    Parameters
    ----------
    bulletin : dict[str, Any]
        Parsed bulletin dictionary.
    output_dir : Path | str
        Root destination directory for card assets.

    Returns
    -------
    dict[str, Path]
        Mapping of asset identifier to saved file path.
    """
    out_root = Path(output_dir)
    latest_dir = out_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Determine date subdirectory YYYY-MM-DD
    issuance_raw = bulletin.get(
        "issuance_utc", bulletin.get("generated_at_utc", "2026-01-01T00:00:00Z")
    )
    date_tag = issuance_raw[:10]  # YYYY-MM-DD
    dated_dir = out_root / date_tag
    dated_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    for lang in ["en", "ur"]:
        # 1. WhatsApp Post Card
        im_post = render_card(bulletin, lang=lang, format=CardFormat.WHATSAPP_POST)
        post_name = f"lahore-whatsapp-{lang}.png"
        path_post_latest = latest_dir / post_name
        path_post_dated = dated_dir / post_name
        save_card(im_post, path_post_latest)
        save_card(im_post, path_post_dated)
        artifacts[f"whatsapp_{lang}"] = path_post_latest

        # 2. Open Graph Card
        im_og = render_card(bulletin, lang=lang, format=CardFormat.OPEN_GRAPH)
        og_name = f"lahore-og-{lang}.png"
        path_og_latest = latest_dir / og_name
        path_og_dated = dated_dir / og_name
        save_card(im_og, path_og_latest)
        save_card(im_og, path_og_dated)
        artifacts[f"og_{lang}"] = path_og_latest

    return artifacts
