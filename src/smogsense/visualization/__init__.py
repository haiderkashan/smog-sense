"""smogsense.visualization — Charts and cards.

Charts and cards.

Specification: docs/dissemination-and-ui.md → 'Information architecture'
"""

from smogsense.visualization.cards import (
    CardFormat,
    CardSpec,
    render_all_cards,
    render_card,
    save_card,
)
from smogsense.visualization.fan_chart import (
    AxisTickGeometry,
    BarGeometry,
    CategoryBandGeometry,
    ColumnGeometry,
    FanChartGeometry,
    FanChartLayout,
    MedianGeometry,
    compute_fan_chart_geometry,
    render_fan_chart_svg,
)
from smogsense.visualization.text_layout import (
    draw_multiline_text,
    is_raqm_available,
    load_card_font,
    measure_multiline_text,
    wrap_text_multiline,
)

__all__ = [
    "AxisTickGeometry",
    "BarGeometry",
    "CardFormat",
    "CardSpec",
    "CategoryBandGeometry",
    "ColumnGeometry",
    "FanChartGeometry",
    "FanChartLayout",
    "MedianGeometry",
    "compute_fan_chart_geometry",
    "draw_multiline_text",
    "is_raqm_available",
    "load_card_font",
    "measure_multiline_text",
    "render_all_cards",
    "render_card",
    "render_fan_chart_svg",
    "save_card",
    "wrap_text_multiline",
]
