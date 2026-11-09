"""Unit tests for truth-basis statement, clock-window horizon labels, true generation time, and exceedance rounding.

Task: Phase 4 Task P4-10
Specification: docs/dissemination-and-ui.md, docs/PRD.md (FR-22-FR-37)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from smogsense.publishing.bulletin import generate_bulletin_json
from smogsense.publishing.i18n import (
    Translator,
    _extract_placeholders,
    create_jinja_env,
    load_catalogs,
    verify_catalog_parity,
)


@pytest.fixture
def sample_bulletin() -> dict[str, Any]:
    """Load canonical valid bulletin test fixture."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _build_test_forecast_df(issuance: datetime) -> pd.DataFrame:
    """Build a minimal valid forecast DataFrame containing all 19 quantiles for city and station."""
    rows = []
    for lead in [24, 48, 72]:
        row_city: dict[str, Any] = {
            "issuance_utc": issuance,
            "generated_at_utc": issuance,
            "domain": "lahore",
            "level": "city",
            "point_id": "city:lahore",
            "horizon_h": lead,
            "window_start_utc": issuance + pd.Timedelta(hours=lead - 24),
            "window_end_utc": issuance + pd.Timedelta(hours=lead),
            "method": "m2_cams_bc",
            "model_version": "baseline",
            "mode": "full",
            "cams_base_time_utc": issuance,
            "data_cutoff_utc": issuance,
            "git_sha": "0123456789abcdef",
            "config_hash": "cfg123",
            "calibrated": False,
            "run_id": "test_run",
            "is_rerun": False,
            "adaptation_status": "skipped_n0",
            "cams_lead_offset_h": 0.0,
            "n_available_stations": 2,
            "forecast_ugm3": 175.0,
        }
        for q in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            row_city[f"q{q:02d}"] = 100.0 + q * 1.5
        rows.append(row_city)

        # Station 1 (Reference)
        row_s1 = dict(row_city)
        row_s1["level"] = "station"
        row_s1["point_id"] = "station:42"
        rows.append(row_s1)

        # Station 2 (Low-cost)
        row_s2 = dict(row_city)
        row_s2["level"] = "station"
        row_s2["point_id"] = "station:43"
        rows.append(row_s2)

    return pd.DataFrame(rows)


def test_six_revision_keys_parity_and_rendering() -> None:
    """Verify that all six critical revision keys exist, maintain placeholder parity, and render in en/ur."""
    catalogs = load_catalogs()
    verify_catalog_parity(catalogs)

    t_en = Translator("en", catalogs["en"])
    t_ur = Translator("ur", catalogs["ur"])

    revision_keys = [
        ("horizon.window", {"start": "Thu 05:00 AM", "end": "Fri 05:00 AM"}),
        ("status.valid_until", {"time": "06 Nov 2026, 06:00 AM"}),
        ("status.expired_banner", {}),
        ("uncertainty.chance_exceed_low", {"threshold": "55.5"}),
        ("uncertainty.chance_exceed_high", {"threshold": "55.5"}),
        ("methodology.truth", {"n": 6, "ref": 3, "lc": 3}),
    ]

    for key, kwargs in revision_keys:
        en_rendered = t_en(key, **kwargs)
        ur_rendered = t_ur(key, **kwargs)

        assert isinstance(en_rendered, str) and len(en_rendered) > 0
        assert isinstance(ur_rendered, str) and len(ur_rendered) > 0

        # Verify placeholders match
        parts = key.split(".")
        en_raw = catalogs["en"]
        ur_raw = catalogs["ur"]
        for p in parts:
            en_raw = en_raw[p]
            ur_raw = ur_raw[p]
        assert _extract_placeholders(en_raw) == _extract_placeholders(ur_raw)

    # Specific substring assertions
    assert "Thu 05:00 AM to Fri 05:00 AM, Pakistan time" in t_en(
        "horizon.window", start="Thu 05:00 AM", end="Fri 05:00 AM"
    )
    assert "تک، پاکستانی وقت" in t_ur("horizon.window", start="Thu 05:00 AM", end="Fri 05:00 AM")
    assert "under 5%" in t_en("uncertainty.chance_exceed_low", threshold="55.5")
    assert "5 فیصد سے کم" in t_ur("uncertainty.chance_exceed_low", threshold="55.5")
    assert "over 95%" in t_en("uncertainty.chance_exceed_high", threshold="55.5")
    assert "95 فیصد سے زیادہ" in t_ur("uncertainty.chance_exceed_high", threshold="55.5")


def test_truth_basis_statement_interpolation() -> None:
    """Verify that truth-basis statement accurately interpolates station counts and supports defensive stubs."""
    env_en = create_jinja_env(lang="en")
    env_ur = create_jinja_env(lang="ur")
    template_name = "partials/_truth_basis.html.j2"

    # Test with varying station counts
    test_cases = [
        {"n_panel": 6, "n_reference": 3, "n_lowcost": 3},
        {"n_panel": 1, "n_reference": 1, "n_lowcost": 0},
        {"n_panel": 12, "n_reference": 4, "n_lowcost": 8},
    ]

    for counts in test_cases:
        bulletin_mock = {"basis": counts}

        rendered_en = env_en.get_template(template_name).render(bulletin=bulletin_mock)
        rendered_ur = env_ur.get_template(template_name).render(bulletin=bulletin_mock)

        assert '<section class="truth-basis"' in rendered_en
        assert (
            f"average of {counts['n_panel']} stations ({counts['n_reference']} reference, {counts['n_lowcost']} low-cost)"
            in rendered_en
        )
        assert "US Consulate" not in rendered_en  # Follows catalog truth wording

        assert '<section class="truth-basis"' in rendered_ur
        assert (
            f"{counts['n_panel']} اسٹیشنوں ({counts['n_reference']} ریفرنس، {counts['n_lowcost']} سستے سینسر) کی اوسط"
            in rendered_ur
        )

    # Test defensive omission: empty or missing basis does not fail and renders empty string
    bulletin_stub: dict[str, Any] = {}
    rendered_empty = env_en.get_template(template_name).render(bulletin=bulletin_stub)
    assert rendered_empty.strip() == ""


def test_clock_window_horizon_labels_formatting() -> None:
    """Verify horizon cards render day labels and Pakistan Standard Time (UTC+5) window bounds."""
    env_en = create_jinja_env(lang="en")
    env_ur = create_jinja_env(lang="ur")
    template_name = "partials/_horizon_card.html.j2"

    # Horizon definitions matching 00:00 UTC (05:00 PKT)
    horizons = [
        {
            "lead_h": 24,
            "window_start_utc": "2026-11-05T00:00:00Z",
            "window_end_utc": "2026-11-06T00:00:00Z",
            "quantiles_ugm3": {"q10": 80.0, "q50": 120.0, "q90": 180.0},
            "category": {"median_q50": "unhealthy"},
            "probabilities": {"exceed": []},
        },
        {
            "lead_h": 48,
            "window_start_utc": "2026-11-06T00:00:00Z",
            "window_end_utc": "2026-11-07T00:00:00Z",
            "quantiles_ugm3": {"q10": 90.0, "q50": 140.0, "q90": 210.0},
            "category": {"median_q50": "very_unhealthy"},
            "probabilities": {"exceed": []},
        },
        {
            "lead_h": 72,
            "window_start_utc": "2026-11-07T00:00:00Z",
            "window_end_utc": "2026-11-08T00:00:00Z",
            "quantiles_ugm3": {"q10": 110.0, "q50": 160.0, "q90": 240.0},
            "category": {"median_q50": "very_unhealthy"},
            "probabilities": {"exceed": []},
        },
    ]

    expected_labels_en = ["Next 24 hours", "24 to 48 hours ahead", "48 to 72 hours ahead"]
    expected_labels_ur = ["اگلے 24 گھنٹے", "24 سے 48 گھنٹے بعد", "48 سے 72 گھنٹے بعد"]
    expected_windows_en = [
        "Thu 05:00 AM to Fri 05:00 AM, Pakistan time",
        "Fri 05:00 AM to Sat 05:00 AM, Pakistan time",
        "Sat 05:00 AM to Sun 05:00 AM, Pakistan time",
    ]
    expected_windows_ur = [
        "Thu 05:00 AM سے Fri 05:00 AM تک، پاکستانی وقت",
        "Fri 05:00 AM سے Sat 05:00 AM تک، پاکستانی وقت",
        "Sat 05:00 AM سے Sun 05:00 AM تک، پاکستانی وقت",
    ]

    for idx, h in enumerate(horizons):
        out_en = env_en.get_template(template_name).render(h=h)
        out_ur = env_ur.get_template(template_name).render(h=h)

        assert expected_labels_en[idx] in out_en
        assert expected_windows_en[idx] in out_en
        assert '<time class="horizon-card__window">' in out_en

        assert expected_labels_ur[idx] in out_ur
        assert expected_windows_ur[idx] in out_ur
        assert '<time class="horizon-card__window">' in out_ur


def test_exceedance_probability_natural_frequency_rounding() -> None:
    """Verify natural frequency rounding for exceedances: <5% low clamp, >95% high clamp, 5% increments."""
    env_en = create_jinja_env(lang="en")
    env_ur = create_jinja_env(lang="ur")
    template_name = "partials/_horizon_card.html.j2"

    h = {
        "lead_h": 24,
        "window_start_utc": "2026-11-05T00:00:00Z",
        "window_end_utc": "2026-11-06T00:00:00Z",
        "quantiles_ugm3": {"q10": 80.0, "q50": 120.0, "q90": 180.0},
        "category": {"median_q50": "unhealthy"},
        "probabilities": {
            "exceed": [
                {"threshold_ugm3": 35.5, "probability": 0.98},  # > 0.95 -> high clamp
                {"threshold_ugm3": 55.5, "probability": 0.54},  # rounded to 55%
                {"threshold_ugm3": 125.5, "probability": 0.32},  # rounded to 30%
                {"threshold_ugm3": 150.0, "probability": 0.18},  # rounded to 20%
                {"threshold_ugm3": 225.5, "probability": 0.03},  # < 0.05 -> low clamp
                {"threshold_ugm3": 250.0, "probability": 0.05},  # exact boundary -> 5%
                {"threshold_ugm3": 300.0, "probability": 0.95},  # exact boundary -> 95%
            ]
        },
    }

    out_en = env_en.get_template(template_name).render(h=h)
    out_ur = env_ur.get_template(template_name).render(h=h)

    # English checks
    assert '<ul class="exceed-list">' in out_en
    assert "Chance of exceeding 35.5 µg/m³: over 95%" in out_en
    assert "Chance of exceeding 55.5 µg/m³: 55%" in out_en
    assert "Chance of exceeding 125.5 µg/m³: 30%" in out_en
    assert "Chance of exceeding 150 µg/m³: 20%" in out_en
    assert "Chance of exceeding 225.5 µg/m³: under 5%" in out_en
    assert "Chance of exceeding 250 µg/m³: 5%" in out_en
    assert "Chance of exceeding 300 µg/m³: 95%" in out_en

    # Urdu checks
    assert '<ul class="exceed-list">' in out_ur
    assert "35.5 مائیکروگرام/مکعب میٹر سے زیادہ ہونے کا امکان: 95 فیصد سے زیادہ" in out_ur
    assert "55.5 مائیکروگرام/مکعب میٹر سے تجاوز کا امکان: 55 فیصد" in out_ur
    assert "125.5 مائیکروگرام/مکعب میٹر سے تجاوز کا امکان: 30 فیصد" in out_ur
    assert "150 مائیکروگرام/مکعب میٹر سے تجاوز کا امکان: 20 فیصد" in out_ur
    assert "225.5 مائیکروگرام/مکعب میٹر سے زیادہ ہونے کا امکان: 5 فیصد سے کم" in out_ur
    assert "250 مائیکروگرام/مکعب میٹر سے تجاوز کا امکان: 5 فیصد" in out_ur
    assert "300 مائیکروگرام/مکعب میٹر سے تجاوز کا امکان: 95 فیصد" in out_ur


def test_true_generation_time_after_rerun() -> None:
    """Verify bulletin generation preserves actual generation timestamp and rerun mode reason."""
    issuance = datetime(2026, 11, 5, 0, 0, tzinfo=UTC)
    # Delayed rerun triggered 14h 30m after scheduled T0
    actual_gen = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)

    df = _build_test_forecast_df(issuance)
    meta = {
        "42": {"name": "Ref Station", "lat": 31.5, "lon": 74.3, "is_reference": True},
        "43": {"name": "Low Cost Station", "lat": 31.6, "lon": 74.4, "is_reference": False},
    }
    src_status = {"openaq": {"status": "ok", "as_of_utc": "2026-11-05T14:00:00Z"}}

    bulletin = generate_bulletin_json(
        forecast_df=df,
        issuance_utc=issuance,
        mode="full",
        model="m2_cams_bc",
        data_cutoff_utc=issuance,
        domain="lahore",
        sources_status=src_status,
        stations_metadata=meta,
        git_sha="0123456789abcdef",
        generated_at_utc=actual_gen,
        is_rerun=True,
    )

    assert bulletin["issuance_utc"] == "2026-11-05T00:00:00Z"
    assert bulletin["generated_at_utc"] == "2026-11-05T14:30:00Z"
    assert bulletin["mode_reason"] == "Automated pipeline rerun"

    # Verify rendered display in Pakistan time (14:30 UTC = 19:30 PKT = 07:30 PM)
    env_en = create_jinja_env(lang="en")
    env_ur = create_jinja_env(lang="ur")

    # Render timing fragment
    timing_tmpl_en = env_en.from_string(
        "{{ t('status.issued', time=bulletin.generated_at_utc | local_time) }}"
    )
    timing_tmpl_ur = env_ur.from_string(
        "{{ t('status.issued', time=bulletin.generated_at_utc | local_time) }}"
    )

    rendered_en = timing_tmpl_en.render(bulletin=bulletin)
    rendered_ur = timing_tmpl_ur.render(bulletin=bulletin)

    assert "Issued 05 Nov 2026, 07:30 PM (Pakistan time)" in rendered_en
    assert "جاری کردہ: 05 Nov 2026, 07:30 PM (پاکستانی وقت)" in rendered_ur


def test_horizon_cards_responsive_markup_and_accessibility() -> None:
    """Verify semantic markup, a11y structure, and CSS rule integration for horizon cards."""
    env_en = create_jinja_env(lang="en")
    h = {
        "lead_h": 24,
        "window_start_utc": "2026-11-05T00:00:00Z",
        "window_end_utc": "2026-11-06T00:00:00Z",
        "quantiles_ugm3": {"q10": 80.0, "q50": 120.0, "q90": 180.0},
        "category": {"median_q50": "unhealthy"},
        "probabilities": {"exceed": [{"threshold_ugm3": 55.5, "probability": 0.40}]},
    }

    html = env_en.get_template("partials/_horizon_card.html.j2").render(h=h)

    # Semantic element checks
    assert "<article" in html
    assert 'class="horizon-card horizon-card--unhealthy"' in html
    assert '<header class="horizon-card__header">' in html
    assert '<time class="horizon-card__window">' in html
    assert '<div class="horizon-card__metrics">' in html
    assert "metric-row--median" in html
    assert "metric-row--range" in html
    assert "metric-row--badcase" in html
    assert '<ul class="exceed-list">' in html
    assert '<li class="exceed-item">' in html

    # Verify CSS contracts in base.css and rtl.css
    base_css = Path("web/static/css/base.css").read_text(encoding="utf-8")
    rtl_css = Path("web/static/css/rtl.css").read_text(encoding="utf-8")

    assert ".horizons-grid" in base_css
    assert "@media (min-width: 48rem)" in base_css
    assert "grid-template-columns: repeat(3, 1fr);" in base_css
    assert ".horizon-card" in base_css
    assert ".horizon-card__window" in base_css
    assert ".exceed-list" in base_css

    assert ".horizon-card__window" in rtl_css
    assert "direction: ltr;" in rtl_css
    assert "unicode-bidi: isolate;" in rtl_css
