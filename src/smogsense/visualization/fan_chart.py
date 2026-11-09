"""smogsense.visualization.fan_chart — Fan-chart geometry shared by diagnostics, site, and cards.

Pure geometry engine for the category-banded range chart:
- Six equal-height horizontal bands tinted with AQI category colours at low opacity.
- Piecewise-linear value-to-position mapping between category boundaries.
- Dynamic axis top at max(400, 1.05 * q95).
- 80% range (q10-q90), 50% range (q25-q75), median tick (q50), and faint median connecting line.
- Non-colour cues (patterns) for the three red-ish categories (unhealthy, very_unhealthy, hazardous)
  to ensure accessibility in greyscale printing and for colour-blind users.

Specification: docs/dissemination-and-ui.md → 'Information architecture'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Standard AQI breakpoints (US EPA 2024 PM2.5 24h)
DEFAULT_CATEGORY_BREAKPOINTS: list[tuple[str, float, str, str]] = [
    ("good", 0.0, "#00E400", "#000000"),
    ("moderate", 9.1, "#FFFF00", "#000000"),
    ("usg", 35.5, "#FF7E00", "#000000"),
    ("unhealthy", 55.5, "#FF0000", "#000000"),
    ("very_unhealthy", 125.5, "#8F3F97", "#FFFFFF"),
    ("hazardous", 225.5, "#7E0023", "#FFFFFF"),
]

RED_ISH_CATEGORIES: dict[str, str] = {
    "unhealthy": "pattern-unhealthy",
    "very_unhealthy": "pattern-very-unhealthy",
    "hazardous": "pattern-hazardous",
}


@dataclass(frozen=True)
class FanChartLayout:
    """Configurable layout dimensions for the fan chart canvas."""

    width: float = 720.0
    height: float = 360.0
    margin_left: float = 65.0
    margin_right: float = 35.0
    margin_top: float = 30.0
    margin_bottom: float = 60.0
    bar_80_width: float = 56.0
    bar_50_width: float = 36.0
    median_tick_width: float = 44.0
    rx_80: float = 6.0
    rx_50: float = 4.0


@dataclass(frozen=True)
class CategoryBandGeometry:
    """Geometry and style attributes for one horizontal AQI category band."""

    category_id: str
    lower_val: float
    upper_val: float
    y_top: float
    y_bottom: float
    height: float
    x: float
    width: float
    bg: str
    fg: str
    pattern_id: str | None = None


@dataclass(frozen=True)
class AxisTickGeometry:
    """Geometry for a horizontal boundary tick and label."""

    value: float
    label: str
    y: float
    x1: float
    x2: float


@dataclass(frozen=True)
class BarGeometry:
    """Geometry for a quantile range bar."""

    x: float
    y: float
    width: float
    height: float
    q_low_val: float
    q_high_val: float
    rx: float = 6.0


@dataclass(frozen=True)
class MedianGeometry:
    """Geometry for a median tick line and circle marker."""

    x_center: float
    y: float
    value: float
    tick_x1: float
    tick_x2: float


@dataclass(frozen=True)
class ColumnGeometry:
    """Geometry for one 24-hour block forecast horizon."""

    lead_h: int
    x_center: float
    bar_80: BarGeometry
    bar_50: BarGeometry
    median: MedianGeometry
    window_start_utc: str | None = None
    window_end_utc: str | None = None
    category_id: str | None = None


@dataclass(frozen=True)
class FanChartGeometry:
    """Complete layout geometry for the category-banded range chart."""

    width: float
    height: float
    plot_x: float
    plot_y: float
    plot_width: float
    plot_height: float
    axis_top: float
    category_bands: list[CategoryBandGeometry]
    ticks: list[AxisTickGeometry]
    columns: list[ColumnGeometry]
    median_line_points: list[tuple[float, float]]
    median_line_path: str
    layout: FanChartLayout

    def value_to_y(self, value: float) -> float:
        """Map PM2.5 concentration in µg/m³ to pixel y coordinate (piecewise-linear)."""
        num_bands = len(self.category_bands)
        if num_bands == 0:
            return self.plot_y + self.plot_height

        band_h = self.plot_height / num_bands

        # Clamp lower bound
        if value <= 0.0:
            return self.plot_y + self.plot_height

        # Clamp upper bound
        if value >= self.axis_top:
            return self.plot_y

        # Locate which category band the value falls into
        for idx, band in enumerate(self.category_bands):
            if band.lower_val <= value <= band.upper_val or (
                idx == num_bands - 1 and value >= band.lower_val
            ):
                span = band.upper_val - band.lower_val
                if span <= 0.0:
                    fraction = 0.0
                else:
                    fraction = min(1.0, max(0.0, (value - band.lower_val) / span))
                # y decreases as value increases (SVG origin is top-left)
                return band.y_bottom - fraction * band_h

        return self.plot_y

    def y_to_value(self, y: float) -> float:
        """Inverse mapping: convert pixel y coordinate back to PM2.5 concentration in µg/m³."""
        num_bands = len(self.category_bands)
        if num_bands == 0:
            return 0.0

        band_h = self.plot_height / num_bands
        plot_bottom = self.plot_y + self.plot_height

        if y >= plot_bottom:
            return 0.0
        if y <= self.plot_y:
            return self.axis_top

        # Distance from bottom
        dist_from_bottom = plot_bottom - y
        band_idx = int(dist_from_bottom / band_h)
        band_idx = min(num_bands - 1, max(0, band_idx))

        band = self.category_bands[band_idx]
        fraction = (dist_from_bottom - band_idx * band_h) / band_h
        fraction = min(1.0, max(0.0, fraction))

        return band.lower_val + fraction * (band.upper_val - band.lower_val)


def load_aqi_categories_from_config(
    config_path: Path | str = "configs/bulletin.yaml",
) -> list[tuple[str, float, str, str]]:
    """Load AQI categories and threshold breakpoints from bulletin config file."""
    c_path = Path(config_path)
    if not c_path.is_file():
        return DEFAULT_CATEGORY_BREAKPOINTS

    try:
        with c_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            raw_cats = data.get("aqi", {}).get("categories", [])
            if not raw_cats or len(raw_cats) != 6:
                return DEFAULT_CATEGORY_BREAKPOINTS
            categories: list[tuple[str, float, str, str]] = []
            for item in raw_cats:
                cat_id = str(item.get("id"))
                lower = float(item.get("lower", 0.0))
                bg = str(item.get("bg", "#000000"))
                fg = str(item.get("fg", "#FFFFFF"))
                categories.append((cat_id, lower, bg, fg))
            return categories
    except Exception:
        return DEFAULT_CATEGORY_BREAKPOINTS


def compute_fan_chart_geometry(
    bulletin: dict[str, Any],
    layout: FanChartLayout | None = None,
    config_path: Path | str = "configs/bulletin.yaml",
) -> FanChartGeometry:
    """Compute complete pure-data geometry for the category-banded range chart.

    Parameters
    ----------
    bulletin : dict[str, Any]
        Parsed bulletin dictionary adhering to bulletin.schema.json.
    layout : FanChartLayout | None
        Optional layout dimensions and margins (defaults to 720x360).
    config_path : Path | str
        Path to configs/bulletin.yaml for category colors and breakpoints.

    Returns
    -------
    FanChartGeometry
        Calculated coordinate paths, bars, ticks, and layout geometry.
    """
    if layout is None:
        layout = FanChartLayout()

    raw_categories = load_aqi_categories_from_config(config_path)

    horizons_data = bulletin.get("horizons", [])

    # Dynamic axis top: max(400, 1.05 * max(q95))
    max_q95 = 0.0
    for h in horizons_data:
        q_dict = h.get("quantiles_ugm3", {})
        val = float(q_dict.get("q95", q_dict.get("q90", 0.0)))
        if val > max_q95:
            max_q95 = val

    axis_top = max(400.0, 1.05 * max_q95)
    # Clean rounding for axis top
    axis_top = round(axis_top, 1)

    plot_x = layout.margin_left
    plot_y = layout.margin_top
    plot_width = layout.width - layout.margin_left - layout.margin_right
    plot_height = layout.height - layout.margin_top - layout.margin_bottom

    num_bands = len(raw_categories)
    band_height = plot_height / num_bands

    # Build category band geometries from bottom (index 0) to top (index 5)
    # In SVG coordinates: band 0 (good) is at the bottom, band 5 (hazardous) is at the top.
    category_bands: list[CategoryBandGeometry] = []
    ticks: list[AxisTickGeometry] = []

    # First tick at value 0.0 (bottom axis)
    y_bottom_chart = plot_y + plot_height
    ticks.append(
        AxisTickGeometry(
            value=0.0,
            label="0",
            y=round(y_bottom_chart, 2),
            x1=plot_x,
            x2=plot_x + plot_width,
        )
    )

    for i in range(num_bands):
        cat_id, lower, bg, fg = raw_categories[i]
        upper = raw_categories[i + 1][1] if i + 1 < num_bands else axis_top

        y_bottom = y_bottom_chart - i * band_height
        y_top = y_bottom - band_height

        pattern_id = RED_ISH_CATEGORIES.get(cat_id)

        category_bands.append(
            CategoryBandGeometry(
                category_id=cat_id,
                lower_val=lower,
                upper_val=upper,
                y_top=round(y_top, 2),
                y_bottom=round(y_bottom, 2),
                height=round(band_height, 2),
                x=round(plot_x, 2),
                width=round(plot_width, 2),
                bg=bg,
                fg=fg,
                pattern_id=pattern_id,
            )
        )

        # Upper boundary tick for this band
        tick_label = f"{upper:.1f}".rstrip("0").rstrip(".") if upper != 0.0 else "0"
        ticks.append(
            AxisTickGeometry(
                value=round(upper, 1),
                label=tick_label,
                y=round(y_top, 2),
                x1=plot_x,
                x2=plot_x + plot_width,
            )
        )

    # Temporary geometry instance to evaluate value_to_y for horizons
    temp_geom = FanChartGeometry(
        width=layout.width,
        height=layout.height,
        plot_x=plot_x,
        plot_y=plot_y,
        plot_width=plot_width,
        plot_height=plot_height,
        axis_top=axis_top,
        category_bands=category_bands,
        ticks=ticks,
        columns=[],
        median_line_points=[],
        median_line_path="",
        layout=layout,
    )

    # Compute columns
    columns: list[ColumnGeometry] = []
    median_points: list[tuple[float, float]] = []

    num_cols = max(1, len(horizons_data))
    col_step = plot_width / num_cols

    for idx, h in enumerate(horizons_data):
        lead_h = int(h.get("lead_h", (idx + 1) * 24))
        q = h.get("quantiles_ugm3", {})
        q10 = float(q.get("q10", 0.0))
        q25 = float(q.get("q25", 0.0))
        q50 = float(q.get("q50", 0.0))
        q75 = float(q.get("q75", 0.0))
        q90 = float(q.get("q90", 0.0))

        # Column center X
        col_x = plot_x + (idx + 0.5) * col_step

        # 80% interval (q10 to q90)
        y_q90 = temp_geom.value_to_y(q90)
        y_q10 = temp_geom.value_to_y(q10)
        bar_80_h = max(2.0, y_q10 - y_q90)
        bar_80 = BarGeometry(
            x=round(col_x - layout.bar_80_width / 2, 2),
            y=round(y_q90, 2),
            width=layout.bar_80_width,
            height=round(bar_80_h, 2),
            q_low_val=q10,
            q_high_val=q90,
            rx=layout.rx_80,
        )

        # 50% interval (q25 to q75)
        y_q75 = temp_geom.value_to_y(q75)
        y_q25 = temp_geom.value_to_y(q25)
        bar_50_h = max(2.0, y_q25 - y_q75)
        bar_50 = BarGeometry(
            x=round(col_x - layout.bar_50_width / 2, 2),
            y=round(y_q75, 2),
            width=layout.bar_50_width,
            height=round(bar_50_h, 2),
            q_low_val=q25,
            q_high_val=q75,
            rx=layout.rx_50,
        )

        # Median tick
        y_q50 = temp_geom.value_to_y(q50)
        med_tick_x1 = col_x - layout.median_tick_width / 2
        med_tick_x2 = col_x + layout.median_tick_width / 2
        median_geo = MedianGeometry(
            x_center=round(col_x, 2),
            y=round(y_q50, 2),
            value=q50,
            tick_x1=round(med_tick_x1, 2),
            tick_x2=round(med_tick_x2, 2),
        )

        median_points.append((round(col_x, 2), round(y_q50, 2)))

        col_cat = h.get("category", {}).get("median_q50")

        columns.append(
            ColumnGeometry(
                lead_h=lead_h,
                x_center=round(col_x, 2),
                bar_80=bar_80,
                bar_50=bar_50,
                median=median_geo,
                window_start_utc=h.get("window_start_utc"),
                window_end_utc=h.get("window_end_utc"),
                category_id=col_cat,
            )
        )

    # Build median connecting line SVG path (e.g. M x1 y1 L x2 y2 L x3 y3)
    if median_points:
        path_parts = [f"M {median_points[0][0]} {median_points[0][1]}"]
        path_parts.extend(f"L {pt[0]} {pt[1]}" for pt in median_points[1:])
        median_line_path = " ".join(path_parts)
    else:
        median_line_path = ""

    return FanChartGeometry(
        width=layout.width,
        height=layout.height,
        plot_x=plot_x,
        plot_y=plot_y,
        plot_width=plot_width,
        plot_height=plot_height,
        axis_top=axis_top,
        category_bands=category_bands,
        ticks=ticks,
        columns=columns,
        median_line_points=median_points,
        median_line_path=median_line_path,
        layout=layout,
    )


def render_fan_chart_svg(
    geometry: FanChartGeometry,
    title: str = "Forecast chart for the next three days",
    desc: str = "The line shows the most likely level; the shaded bands show the likely range.",
    unit_label: str = "µg/m³",
) -> str:
    """Render a standalone or embeddable SVG string from computed FanChartGeometry.

    Enforces WCAG 2.2 AA non-colour cues, semantic elements, and CSP compliance
    (strictly no inline <style> tags).
    """
    svg_lines: list[str] = [
        f'<svg class="fan-chart" viewBox="0 0 {geometry.width:.0f} {geometry.height:.0f}" role="img" aria-labelledby="fanchart-title fanchart-desc" xmlns="http://www.w3.org/2000/svg">',
        f'  <title id="fanchart-title">{title}</title>',
        f'  <desc id="fanchart-desc">{desc}</desc>',
        "  <defs>",
        '    <pattern id="pattern-unhealthy" width="10" height="10" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">',
        '      <line x1="0" y1="0" x2="0" y2="10" stroke="var(--aqi-unhealthy-bg, #ff0000)" stroke-width="2.5" stroke-opacity="0.35" />',
        "    </pattern>",
        '    <pattern id="pattern-very-unhealthy" width="12" height="12" patternUnits="userSpaceOnUse">',
        '      <path d="M 0 0 L 12 12 M 12 0 L 0 12" stroke="var(--aqi-very-unhealthy-bg, #8f3f97)" stroke-width="2" stroke-opacity="0.4" />',
        "    </pattern>",
        '    <pattern id="pattern-hazardous" width="8" height="8" patternUnits="userSpaceOnUse">',
        '      <circle cx="4" cy="4" r="1.5" fill="var(--aqi-hazardous-bg, #7e0023)" fill-opacity="0.5" />',
        "    </pattern>",
        "  </defs>",
        "  <!-- 1. Category Bands & Accessibility Patterns -->",
        '  <g class="fanchart-bands">',
    ]

    for band in geometry.category_bands:
        svg_lines.append(
            f'    <rect class="band-rect band-rect--{band.category_id}" x="{band.x}" y="{band.y_top}" width="{band.width}" height="{band.height}" fill="var(--aqi-{band.category_id}-bg, {band.bg})" fill-opacity="0.15" />'
        )
        if band.pattern_id:
            svg_lines.append(
                f'    <rect class="band-pattern band-pattern--{band.category_id}" x="{band.x}" y="{band.y_top}" width="{band.width}" height="{band.height}" fill="url(#{band.pattern_id})" />'
            )

    svg_lines.extend(
        [
            "  </g>",
            "  <!-- 2. Grid & Boundary Lines -->",
            '  <g class="fanchart-grid">',
        ]
    )

    for tick in geometry.ticks:
        svg_lines.append(
            f'    <line x1="{tick.x1}" y1="{tick.y}" x2="{tick.x2}" y2="{tick.y}" stroke="var(--rule, #d9d3ca)" stroke-width="1" stroke-dasharray="3 3" />'
        )
        svg_lines.append(
            f'    <text x="{tick.x1 - 8:.1f}" y="{tick.y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--fg-muted, #5b5650)">{tick.label}</text>'
        )

    # Unit label at the top-left of the Y-axis
    svg_lines.append(
        f'    <text x="{geometry.plot_x - 8:.1f}" y="{geometry.plot_y - 12:.1f}" text-anchor="end" font-size="11" font-weight="700" fill="var(--fg-muted, #5b5650)">{unit_label}</text>'
    )
    svg_lines.append("  </g>")

    # Connecting median line
    if geometry.median_line_path:
        svg_lines.extend(
            [
                "  <!-- 3. Median Connecting Line -->",
                f'  <path d="{geometry.median_line_path}" fill="none" stroke="var(--fg-muted, #5b5650)" stroke-width="1.5" stroke-dasharray="4 4" stroke-opacity="0.65" />',
            ]
        )

    # Forecast columns
    svg_lines.extend(
        [
            "  <!-- 4. Forecast Columns -->",
            '  <g class="fanchart-columns">',
        ]
    )

    for col in geometry.columns:
        svg_lines.extend(
            [
                f"    <!-- Column +{col.lead_h}h -->",
                f'    <line x1="{col.x_center}" y1="{geometry.plot_y}" x2="{col.x_center}" y2="{geometry.plot_y + geometry.plot_height}" stroke="var(--rule, #d9d3ca)" stroke-width="1" stroke-dasharray="2 2" stroke-opacity="0.5" />',
                f'    <rect class="bar-80" x="{col.bar_80.x}" y="{col.bar_80.y}" width="{col.bar_80.width}" height="{col.bar_80.height}" rx="{col.bar_80.rx}" fill="var(--band-80, rgba(26, 82, 112, 0.22))" />',
                f'    <rect class="bar-50" x="{col.bar_50.x}" y="{col.bar_50.y}" width="{col.bar_50.width}" height="{col.bar_50.height}" rx="{col.bar_50.rx}" fill="var(--band-50, rgba(26, 82, 112, 0.40))" />',
                f'    <line class="median-tick" x1="{col.median.tick_x1}" y1="{col.median.y}" x2="{col.median.tick_x2}" y2="{col.median.y}" stroke="var(--median, #1b1f2a)" stroke-width="3.5" stroke-linecap="round" />',
                f'    <circle class="median-dot" cx="{col.x_center}" cy="{col.median.y}" r="3.5" fill="var(--median, #1b1f2a)" />',
                f'    <text class="col-lead-label" x="{col.x_center}" y="{geometry.plot_y + geometry.plot_height + 25:.1f}" text-anchor="middle" font-size="13" font-weight="700" fill="var(--fg, #1b1f2a)">+{col.lead_h}h</text>',
                f'    <text class="col-median-label" x="{col.x_center}" y="{geometry.plot_y + geometry.plot_height + 42:.1f}" text-anchor="middle" font-size="12" fill="var(--fg-muted, #5b5650)">{col.median.value:.0f} {unit_label}</text>',
            ]
        )

    svg_lines.extend(
        [
            "  </g>",
            "</svg>",
        ]
    )

    return "\n".join(svg_lines)
