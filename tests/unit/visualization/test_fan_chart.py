"""Unit tests for Fan Chart geometry engine, SVG generation, and accessibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from smogsense.publishing.i18n import create_jinja_env
from smogsense.visualization.fan_chart import (
    FanChartLayout,
    compute_fan_chart_geometry,
    render_fan_chart_svg,
)

pytestmark = pytest.mark.unit


def load_fixture_bulletin() -> dict[str, Any]:
    """Load sample valid bulletin fixture."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def test_fan_chart_geometry_computation() -> None:
    """Verify that compute_fan_chart_geometry calculates exact layout properties."""
    bulletin = load_fixture_bulletin()
    layout = FanChartLayout(width=720.0, height=360.0)

    geom = compute_fan_chart_geometry(bulletin=bulletin, layout=layout)

    # 1. Canvas and plot dimensions
    assert geom.width == 720.0
    assert geom.height == 360.0
    assert geom.plot_x == 65.0
    assert geom.plot_y == 30.0
    assert geom.plot_width == 720.0 - 65.0 - 35.0  # 620.0
    assert geom.plot_height == 360.0 - 30.0 - 60.0  # 270.0

    # 2. Category bands: exactly 6 equal-height bands
    assert len(geom.category_bands) == 6
    expected_ids = ["good", "moderate", "usg", "unhealthy", "very_unhealthy", "hazardous"]
    expected_lowers = [0.0, 9.1, 35.5, 55.5, 125.5, 225.5]

    band_h = geom.plot_height / 6  # 45.0
    for idx, band in enumerate(geom.category_bands):
        assert band.category_id == expected_ids[idx]
        assert band.lower_val == expected_lowers[idx]
        assert band.height == band_h
        assert band.width == geom.plot_width
        assert band.x == geom.plot_x

        # Bands are ordered from bottom to top in pixel coordinates
        expected_y_bottom = (geom.plot_y + geom.plot_height) - idx * band_h
        expected_y_top = expected_y_bottom - band_h
        assert band.y_bottom == pytest.approx(expected_y_bottom)
        assert band.y_top == pytest.approx(expected_y_top)

        # Non-colour cue patterns for red-ish categories
        if band.category_id in ["unhealthy", "very_unhealthy", "hazardous"]:
            assert band.pattern_id == f"pattern-{band.category_id.replace('_', '-')}"
        else:
            assert band.pattern_id is None

    # 3. Ticks: 7 boundary ticks (0.0 + 6 category boundaries)
    assert len(geom.ticks) == 7
    tick_values = [t.value for t in geom.ticks]
    assert tick_values == [0.0, 9.1, 35.5, 55.5, 125.5, 225.5, geom.axis_top]
    for tick in geom.ticks:
        assert tick.x1 == geom.plot_x
        assert tick.x2 == geom.plot_x + geom.plot_width

    # 4. Forecast columns: exactly 3 horizons
    assert len(geom.columns) == 3
    for idx, col in enumerate(geom.columns):
        assert col.lead_h == (idx + 1) * 24
        assert col.bar_80.width == layout.bar_80_width
        assert col.bar_50.width == layout.bar_50_width
        assert col.bar_80.height >= col.bar_50.height
        assert col.bar_80.height > 0
        assert col.bar_50.height > 0
        assert col.median.tick_x2 > col.median.tick_x1

        # Monotonic quantile nesting in y-coordinates (higher quantile = smaller y in SVG)
        y_q90 = col.bar_80.y
        y_q75 = col.bar_50.y
        y_q50 = col.median.y
        y_q25 = col.bar_50.y + col.bar_50.height
        y_q10 = col.bar_80.y + col.bar_80.height
        assert y_q90 <= y_q75 <= y_q50 <= y_q25 <= y_q10

    # 5. Median connecting line
    assert len(geom.median_line_points) == 3
    assert geom.median_line_path.startswith("M ")
    assert geom.median_line_path.count(" L ") == 2


def test_piecewise_linear_value_mapping_and_inverse() -> None:
    """Verify piecewise-linear mapping properties, boundary continuity, and inverse bijection."""
    bulletin = load_fixture_bulletin()
    geom = compute_fan_chart_geometry(bulletin=bulletin)

    plot_top = geom.plot_y
    plot_bottom = geom.plot_y + geom.plot_height
    band_h = geom.plot_height / 6

    # 1. Exact boundary values
    assert geom.value_to_y(0.0) == pytest.approx(plot_bottom)
    assert geom.value_to_y(9.1) == pytest.approx(plot_bottom - band_h)
    assert geom.value_to_y(35.5) == pytest.approx(plot_bottom - 2 * band_h)
    assert geom.value_to_y(55.5) == pytest.approx(plot_bottom - 3 * band_h)
    assert geom.value_to_y(125.5) == pytest.approx(plot_bottom - 4 * band_h)
    assert geom.value_to_y(225.5) == pytest.approx(plot_bottom - 5 * band_h)
    assert geom.value_to_y(geom.axis_top) == pytest.approx(plot_top)

    # 2. Strict monotonicity: higher PM2.5 concentration -> smaller SVG y (higher position)
    test_values = [
        0.0,
        5.0,
        9.1,
        20.0,
        35.5,
        45.0,
        55.5,
        90.0,
        125.5,
        180.0,
        225.5,
        300.0,
        geom.axis_top,
    ]
    mapped_ys = [geom.value_to_y(v) for v in test_values]
    for i in range(len(mapped_ys) - 1):
        assert mapped_ys[i] > mapped_ys[i + 1], (
            f"Monotonicity failed between {test_values[i]} and {test_values[i + 1]}"
        )

    # 3. Clamping of out-of-range values
    assert geom.value_to_y(-50.0) == geom.value_to_y(0.0)
    assert geom.value_to_y(geom.axis_top + 500.0) == geom.value_to_y(geom.axis_top)

    # 4. Inverse bijection round-trip
    for val in test_values:
        recovered = geom.y_to_value(geom.value_to_y(val))
        assert recovered == pytest.approx(val, rel=1e-3, abs=1e-3)


def test_dynamic_axis_top() -> None:
    """Verify that axis top dynamically scales to max(400, 1.05 * max(q95))."""
    bulletin = load_fixture_bulletin()

    # Case 1: Standard moderate smog where max(q95) = 340 -> 1.05 * 340 = 357 <= 400
    geom1 = compute_fan_chart_geometry(bulletin=bulletin)
    assert geom1.axis_top == 400.0

    # Case 2: Extreme smog episode where max(q95) = 500 -> 1.05 * 500 = 525 > 400
    bulletin_extreme = json.loads(json.dumps(bulletin))
    bulletin_extreme["horizons"][0]["quantiles_ugm3"]["q95"] = 500.0
    geom2 = compute_fan_chart_geometry(bulletin=bulletin_extreme)
    assert geom2.axis_top == 525.0

    # Case 3: Empty horizons
    geom_empty = compute_fan_chart_geometry(bulletin={"horizons": []})
    assert geom_empty.axis_top == 400.0


def test_render_fan_chart_svg() -> None:
    """Verify standalone render_fan_chart_svg produces valid XML, patterns, and no inline styles."""
    bulletin = load_fixture_bulletin()
    geom = compute_fan_chart_geometry(bulletin=bulletin)

    svg_str = render_fan_chart_svg(
        geometry=geom,
        title="Custom Fan Chart Title",
        desc="Custom Fan Chart Description",
    )

    assert svg_str.startswith('<svg class="fan-chart"')
    assert svg_str.endswith("</svg>")
    assert 'viewBox="0 0 720 360"' in svg_str
    assert 'role="img"' in svg_str
    assert '<title id="fanchart-title">Custom Fan Chart Title</title>' in svg_str
    assert '<desc id="fanchart-desc">Custom Fan Chart Description</desc>' in svg_str

    # Non-colour patterns
    assert 'id="pattern-unhealthy"' in svg_str
    assert 'id="pattern-very-unhealthy"' in svg_str
    assert 'id="pattern-hazardous"' in svg_str

    # Category bands and pattern fills
    assert 'fill="url(#pattern-unhealthy)"' in svg_str
    assert 'fill="url(#pattern-very-unhealthy)"' in svg_str
    assert 'fill="url(#pattern-hazardous)"' in svg_str

    # Columns and median line
    assert 'class="bar-80"' in svg_str
    assert 'class="bar-50"' in svg_str
    assert 'class="median-tick"' in svg_str
    assert 'class="col-lead-label"' in svg_str
    assert "+24h" in svg_str
    assert "+48h" in svg_str
    assert "+72h" in svg_str

    # Strict CSP: no inline style blocks
    assert "<style" not in svg_str
    assert "<script" not in svg_str


def test_fan_chart_jinja_rendering_and_accessibility() -> None:
    """Verify that _fan_chart.svg.j2 renders cleanly in Jinja2 in both English and Urdu with accessible table."""
    bulletin = load_fixture_bulletin()

    for lang in ["en", "ur"]:
        env = create_jinja_env(lang=lang)
        template = env.get_template("partials/_fan_chart.svg.j2")
        html = template.render(bulletin=bulletin)

        # SVG container and role="img"
        assert 'class="fan-chart-container"' in html
        assert '<svg class="fan-chart"' in html
        assert 'role="img"' in html
        assert 'viewBox="0 0 720 360"' in html

        # Accessible title and description from i18n catalog
        t_func = cast(Any, env.globals["t"])
        assert str(t_func("a11y.chart_alt")) in html
        assert str(t_func("uncertainty.explain_short")) in html

        # Non-colour patterns
        assert 'id="pattern-unhealthy"' in html
        assert 'id="pattern-very-unhealthy"' in html
        assert 'id="pattern-hazardous"' in html

        # Screen reader fallback table
        assert '<table class="sr-only">' in html
        assert "<caption>" in html
        assert '<th scope="col">' in html
        assert "+24h" in html
        assert "+48h" in html
        assert "+72h" in html

        # Exact fixture values present in accessible table
        # Day 1: q50=243, q10=182, q90=316
        assert "243" in html
        assert "182" in html
        assert "316" in html


def test_fan_chart_with_explicit_geometry_override() -> None:
    """Verify that templates respect custom geometry passed explicitly in context."""
    bulletin = load_fixture_bulletin()
    custom_layout = FanChartLayout(
        width=960.0,
        height=480.0,
        margin_left=80.0,
        margin_right=40.0,
        margin_top=40.0,
        margin_bottom=80.0,
        bar_80_width=70.0,
        bar_50_width=45.0,
    )
    custom_geom = compute_fan_chart_geometry(bulletin=bulletin, layout=custom_layout)

    env = create_jinja_env(lang="en")
    template = env.get_template("partials/_fan_chart.svg.j2")
    html = template.render(bulletin=bulletin, geometry=custom_geom)

    assert 'viewBox="0 0 960 480"' in html
    assert f'width="{custom_geom.plot_width}"' in html
