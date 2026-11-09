"""smogsense.visualization.text_layout — Bidirectional text measurement and wrapping.

Measures with the same layout engine used for drawing so that Nastaliq line wrapping and tall
line boxes (line height >= 2.0 x font size) are correct.

Specification: docs/dissemination-and-ui.md → 'Bilingual and RTL rules (Urdu Nastaliq)'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from PIL import ImageDraw, ImageFont, features

logger = logging.getLogger(__name__)

# Fallback font paths by OS
WINDOWS_FONTS: dict[str, list[str]] = {
    "latin": [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ],
    "urdu": [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf",
        r"C:\Windows\Fonts\seguisb.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ],
}

LINUX_FONTS: dict[str, list[str]] = {
    "latin": [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "urdu": [
        "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    ],
}

MACOS_FONTS: dict[str, list[str]] = {
    "latin": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    "urdu": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/GeezaPro.ttc",
    ],
}


def is_raqm_available() -> bool:
    """Check whether libraqm is available in Pillow for complex text shaping."""
    return bool(features.check("raqm"))


def load_card_font(
    script: str = "latin",
    size: int = 24,
    bold: bool = False,
    config_path: Path | str = "configs/bulletin.yaml",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the appropriate font for cards with Raqm engine when available.

    Parameters
    ----------
    script : str
        'latin' or 'urdu'.
    size : int
        Target font size in pixels (supersampled canvas size).
    bold : bool
        Whether bold weight is requested.
    config_path : Path | str
        Path to configs/bulletin.yaml to read configured font paths.

    Returns
    -------
    ImageFont.FreeTypeFont | ImageFont.ImageFont
        Loaded font object ready for drawing and measurement.
    """
    has_raqm = is_raqm_available()
    engine = ImageFont.Layout.RAQM if has_raqm else ImageFont.Layout.BASIC

    # 1. Try path from configs/bulletin.yaml
    cfg_file = Path(config_path)
    if cfg_file.is_file():
        try:
            with cfg_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                fonts_cfg = data.get("cards", {}).get("fonts", {})
                key = f"{script}_bold" if bold else script
                candidate = fonts_cfg.get(key)
                if candidate and Path(candidate).is_file():
                    try:
                        return ImageFont.truetype(str(candidate), size=size, layout_engine=engine)
                    except Exception as err:
                        logger.warning("Failed loading configured font %s: %s", candidate, err)
        except Exception as err:
            logger.warning("Error reading %s: %s", config_path, err)

    # 2. Try platform-specific candidate fonts
    candidates: list[str] = []
    candidates.extend(LINUX_FONTS.get(script, []))
    candidates.extend(WINDOWS_FONTS.get(script, []))
    candidates.extend(MACOS_FONTS.get(script, []))

    for path_str in candidates:
        p = Path(path_str)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size, layout_engine=engine)
            except Exception as err:
                logger.debug("Failed loading font candidate %s: %s", path_str, err)
                continue

    # 3. Fallback to Pillow's built-in default font
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def get_layout_kwargs(direction: str = "ltr", language: str | None = None) -> dict[str, Any]:
    """Return layout keyword arguments for Pillow draw methods when Raqm is supported."""
    if is_raqm_available() and direction == "rtl":
        kwargs: dict[str, Any] = {"direction": "rtl"}
        if language:
            kwargs["language"] = language
        return kwargs
    return {}


def wrap_text_multiline(
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
    direction: str = "ltr",
    language: str | None = None,
) -> list[str]:
    """Wrap a block of text into lines that do not exceed max_width.

    Parameters
    ----------
    text : str
        Input string (may contain newlines).
    font : ImageFont.ImageFont | ImageFont.FreeTypeFont
        Font used for rendering.
    max_width : int
        Maximum permitted width in pixels.
    draw : ImageDraw.ImageDraw
        Active ImageDraw surface for bbox measurement.
    direction : str
        'ltr' or 'rtl'.
    language : str | None
        Optional language code (e.g. 'ur' for Urdu).

    Returns
    -------
    list[str]
        Wrapped lines that fit within max_width.
    """
    if not text.strip():
        return []

    layout_kwargs = get_layout_kwargs(direction=direction, language=language)
    wrapped_lines: list[str] = []

    # Split by explicit newlines first
    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line: list[str] = []

        for word in words:
            candidate = " ".join([*current_line, word])
            bbox = draw.textbbox((0, 0), candidate, font=font, **layout_kwargs)
            cand_width = bbox[2] - bbox[0]

            if cand_width <= max_width or not current_line:
                current_line.append(word)
            else:
                wrapped_lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            wrapped_lines.append(" ".join(current_line))

    return wrapped_lines


def measure_multiline_text(
    lines: list[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    line_height_factor: float,
    draw: ImageDraw.ImageDraw,
    direction: str = "ltr",
    language: str | None = None,
) -> tuple[int, int]:
    """Measure total width and height of a list of wrapped lines.

    Parameters
    ----------
    lines : list[str]
        Lines of text.
    font : ImageFont.ImageFont | ImageFont.FreeTypeFont
        Font used for rendering.
    line_height_factor : float
        Line height multiplier (e.g. 1.5 for Latin, 2.1 for Urdu).
    draw : ImageDraw.ImageDraw
        Active ImageDraw surface.
    direction : str
        'ltr' or 'rtl'.
    language : str | None
        Language code.

    Returns
    -------
    tuple[int, int]
        (max_width, total_height) in pixels.
    """
    if not lines:
        return 0, 0

    layout_kwargs = get_layout_kwargs(direction=direction, language=language)
    max_w = 0

    for line in lines:
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font, **layout_kwargs)
        w = round(bbox[2] - bbox[0])
        if w > max_w:
            max_w = w

    font_size = getattr(font, "size", 24)
    line_h = round(font_size * line_height_factor)
    total_h = len(lines) * line_h

    return max_w, total_h


def draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] | str,
    line_height_factor: float,
    align: str = "left",
    direction: str = "ltr",
    language: str | None = None,
    box_width: int | None = None,
) -> int:
    """Draw wrapped lines onto canvas with alignment and line-height spacing.

    Parameters
    ----------
    draw : ImageDraw.ImageDraw
        Active drawing context.
    xy : tuple[int, int]
        Starting (x, y) coordinates.
    lines : list[str]
        Text lines to draw.
    font : ImageFont.ImageFont | ImageFont.FreeTypeFont
        Font used for rendering.
    fill : tuple[int, int, int] | str
        Text fill color.
    line_height_factor : float
        Line height factor (e.g. 1.5 for Latin, 2.1 for Urdu).
    align : str
        'left', 'center', or 'right'.
    direction : str
        'ltr' or 'rtl'.
    language : str | None
        Language code.
    box_width : int | None
        Optional bounding box width for center or right alignment.

    Returns
    -------
    int
        Final Y position after all lines have been rendered.
    """
    if not lines:
        return xy[1]

    layout_kwargs = get_layout_kwargs(direction=direction, language=language)
    start_x, start_y = xy
    font_size = getattr(font, "size", 24)
    line_h = round(font_size * line_height_factor)
    current_y = start_y

    for line in lines:
        if not line:
            current_y += line_h
            continue

        bbox = draw.textbbox((0, 0), line, font=font, **layout_kwargs)
        line_w = round(bbox[2] - bbox[0])

        if align == "center":
            line_x = (
                start_x + (box_width - line_w) // 2
                if box_width is not None
                else start_x - line_w // 2
            )
        elif align == "right":
            line_x = start_x + box_width - line_w if box_width is not None else start_x - line_w
        else:
            line_x = start_x

        draw.text((line_x, current_y), line, font=font, fill=fill, **layout_kwargs)
        current_y += line_h

    return current_y
