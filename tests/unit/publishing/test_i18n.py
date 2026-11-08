"""Unit tests for smogsense.publishing.i18n - Message catalog loader and formatting filters."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import jinja2
import pytest

from smogsense.publishing.i18n import (
    KeyParityError,
    Translator,
    category_name,
    create_jinja_env,
    fmt_ugm3,
    load_catalogs,
    local_time,
    verify_catalog_parity,
)

pytestmark = pytest.mark.unit


def test_catalog_loading_and_exact_key_parity() -> None:
    """Verify that both en.yaml and ur.yaml load successfully and pass parity verification."""
    catalogs = load_catalogs()
    assert "en" in catalogs
    assert "ur" in catalogs

    # Must pass without raising KeyParityError
    verify_catalog_parity(catalogs)


def test_parity_fails_on_missing_key() -> None:
    """Verify that missing keys in one catalog trigger KeyParityError."""
    catalogs = load_catalogs()
    tampered = copy.deepcopy(catalogs)
    del tampered["ur"]["status"]["valid_until"]

    with pytest.raises(KeyParityError) as exc_info:
        verify_catalog_parity(tampered)
    assert "status.valid_until" in str(exc_info.value) or "valid_until" in str(exc_info.value)


def test_parity_fails_on_mismatched_placeholders() -> None:
    """Verify that placeholder variable mismatches trigger KeyParityError."""
    catalogs = load_catalogs()
    tampered = copy.deepcopy(catalogs)
    tampered["ur"]["status"]["valid_until"] = "{wrong_time} تک درست"

    with pytest.raises(KeyParityError) as exc_info:
        verify_catalog_parity(tampered)
    assert "Placeholder mismatch" in str(exc_info.value)


def test_translation_lookup_and_interpolation() -> None:
    """Verify dotted key lookup, placeholder substitution, and error handling."""
    catalogs = load_catalogs()
    t_en = Translator("en", catalogs["en"])
    t_ur = Translator("ur", catalogs["ur"])

    # 1. Simple lookup
    assert t_en("aqi.hazardous") == "Hazardous"
    assert t_ur("aqi.hazardous") == "خطرناک"

    # 2. Dotted lookup with placeholders
    res_en = t_en("status.valid_until", time="06:00 AM")
    assert "06:00 AM" in res_en
    assert "Valid until" in res_en

    res_ur = t_ur("status.valid_until", time="06:00 AM")
    assert "06:00 AM" in res_ur

    # 3. Missing key raises KeyError
    with pytest.raises(KeyError):
        t_en("non_existent_section.key")

    # 4. Missing placeholder raises ValueError
    with pytest.raises(ValueError):
        t_en("status.valid_until")  # missing {time}


def test_formatting_filters() -> None:
    """Verify fmt_ugm3, local_time (Asia/Karachi UTC+5), and category_name filters."""
    # 1. fmt_ugm3
    assert fmt_ugm3(243.0) == "243"
    assert fmt_ugm3(243.4) == "243.4"
    assert fmt_ugm3(0.0) == "0"
    assert fmt_ugm3(-5.0) == "0"
    assert fmt_ugm3(None) == "—"
    assert fmt_ugm3("not_a_number") == "—"

    # 2. local_time (Asia/Karachi UTC+5)
    # 00:00 UTC on 5 Nov 2025 -> 05:00 AM PKT on 5 Nov 2025
    utc_str = "2025-11-05T00:00:00Z"
    formatted = local_time(utc_str, fmt="%d %b %Y, %I:%M %p")
    assert "05 Nov 2025" in formatted
    assert "05:00 AM" in formatted

    # datetime object
    dt_utc = datetime(2025, 11, 5, 12, 0, 0, tzinfo=UTC)
    formatted_dt = local_time(dt_utc, fmt="%I:%M %p")
    assert formatted_dt == "05:00 PM"

    assert local_time(None) == "—"

    # 3. category_name
    assert category_name("good", lang="en") == "Good"
    assert category_name("good", lang="ur") == "اچھی"
    assert category_name("hazardous", lang="en") == "Hazardous"
    assert category_name("hazardous", lang="ur") == "خطرناک"
    assert category_name("very_unhealthy", lang="en") == "Very unhealthy"
    assert category_name("very_unhealthy", lang="ur") == "بہت زیادہ غیر صحت بخش"


def test_create_jinja_env() -> None:
    """Verify that create_jinja_env initializes filters, globals, and strict undefined mode."""
    env = create_jinja_env(lang="en")
    assert "fmt_ugm3" in env.filters
    assert "local_time" in env.filters
    assert "category_name" in env.filters
    assert "t" in env.globals
    assert env.globals["lang"] == "en"
    assert env.globals["dir"] == "ltr"

    # StrictUndefined check
    template = env.from_string("Hello {{ missing_var }}")
    with pytest.raises(jinja2.UndefinedError):
        template.render()
