"""smogsense.visualization — Charts and cards.

Charts and cards.

Specification: docs/dissemination-and-ui.md → 'Information architecture'
"""

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

__all__ = [
    "AxisTickGeometry",
    "BarGeometry",
    "CategoryBandGeometry",
    "ColumnGeometry",
    "FanChartGeometry",
    "FanChartLayout",
    "MedianGeometry",
    "compute_fan_chart_geometry",
    "render_fan_chart_svg",
]
