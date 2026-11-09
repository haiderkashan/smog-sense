"""smogsense.publishing.i18n - Message catalogue loader and internationalization engine.

Loads web/i18n/*.yaml, enforces key and placeholder parity across languages,
provides translation lookup with dotted keys, and formats numbers and dates
in Asia/Karachi (Pakistan Standard Time, UTC+5).

Specification: docs/dissemination-and-ui.md → 'Message catalog', 'Jinja2 template contract'
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import jinja2
import yaml

from smogsense.errors import SmogSenseError
from smogsense.visualization.fan_chart import compute_fan_chart_geometry

PKT_TZ = ZoneInfo("Asia/Karachi")

CATEGORY_KEYS = [
    "good",
    "moderate",
    "usg",
    "unhealthy",
    "very_unhealthy",
    "hazardous",
]


class KeyParityError(SmogSenseError, ValueError):
    """Raised when message catalogs between languages have mismatched keys or placeholders."""

    def __init__(self, message: str, exit_code: int = 30) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def load_catalogs(i18n_dir: Path | str = "web/i18n") -> dict[str, dict[str, Any]]:
    """Load all YAML message catalogs from the specified directory."""
    dir_path = Path(i18n_dir)
    catalogs: dict[str, dict[str, Any]] = {}

    for lang in ["en", "ur"]:
        file_path = dir_path / f"{lang}.yaml"
        if not file_path.is_file():
            raise FileNotFoundError(f"Message catalog file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Catalog {file_path} must be a YAML dictionary")
            catalogs[lang] = data

    return catalogs


def _extract_placeholders(text: str) -> set[str]:
    """Extract all format placeholders from a string (e.g. '{name}' -> 'name')."""
    return set(re.findall(r"\{([a-zA-Z0-9_]+)\}", text))


def verify_catalog_parity(catalogs: dict[str, dict[str, Any]]) -> None:
    """Verify exact structural and placeholder parity between English and Urdu catalogs.

    Raises KeyParityError if there are missing keys, mismatched list lengths,
    or inconsistent placeholder variables.
    """
    if "en" not in catalogs or "ur" not in catalogs:
        raise KeyParityError("Both 'en' and 'ur' catalogs must be present for parity verification")

    en_cat = catalogs["en"]
    ur_cat = catalogs["ur"]

    def _check_dict(en_sub: dict[str, Any], ur_sub: dict[str, Any], path: str) -> None:
        en_keys = set(en_sub.keys())
        ur_keys = set(ur_sub.keys())

        missing_in_ur = en_keys - ur_keys
        missing_in_en = ur_keys - en_keys

        if missing_in_ur:
            raise KeyParityError(
                f"Keys missing in Urdu catalog at '{path}': {sorted(missing_in_ur)}"
            )
        if missing_in_en:
            raise KeyParityError(
                f"Keys missing in English catalog at '{path}': {sorted(missing_in_en)}"
            )

        for key in en_keys:
            current_path = f"{path}.{key}" if path else key
            en_val = en_sub[key]
            ur_val = ur_sub[key]

            if type(en_val) is not type(ur_val):
                raise KeyParityError(
                    f"Type mismatch at '{current_path}': English is {type(en_val).__name__}, "
                    f"Urdu is {type(ur_val).__name__}"
                )

            if isinstance(en_val, dict):
                _check_dict(en_val, ur_val, current_path)
            elif isinstance(en_val, list):
                if len(en_val) != len(ur_val):
                    raise KeyParityError(
                        f"List length mismatch at '{current_path}': English has {len(en_val)} items, "
                        f"Urdu has {len(ur_val)}"
                    )
                for idx, (en_item, ur_item) in enumerate(zip(en_val, ur_val, strict=True)):
                    item_path = f"{current_path}[{idx}]"
                    if isinstance(en_item, str) and isinstance(ur_item, str):
                        en_p = _extract_placeholders(en_item)
                        ur_p = _extract_placeholders(ur_item)
                        if en_p != ur_p:
                            raise KeyParityError(
                                f"Placeholder mismatch at '{item_path}': English={sorted(en_p)} vs Urdu={sorted(ur_p)}"
                            )
            elif isinstance(en_val, str) and isinstance(ur_val, str):
                en_p = _extract_placeholders(en_val)
                ur_p = _extract_placeholders(ur_val)
                if en_p != ur_p:
                    raise KeyParityError(
                        f"Placeholder mismatch at '{current_path}': English={sorted(en_p)} vs Urdu={sorted(ur_p)}"
                    )

    _check_dict(en_cat, ur_cat, "")


class Translator:
    """Localized translation callable supporting dotted key lookups and placeholders."""

    def __init__(self, lang: str, catalog: dict[str, Any]) -> None:
        self.lang = lang
        self.catalog = catalog

    def __call__(self, key: str, **placeholders: Any) -> Any:
        """Look up key and interpolate placeholders."""
        parts = key.split(".")
        current: Any = self.catalog

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(
                    f"Translation key '{key}' not found in '{self.lang}' catalog (failed at '{part}')"
                )
            current = current[part]

        if isinstance(current, str):
            expected = _extract_placeholders(current)
            if expected:
                missing = expected - set(placeholders.keys())
                if missing:
                    raise ValueError(
                        f"Missing placeholder(s) {sorted(missing)} for translation key '{key}'"
                    )
                try:
                    return current.format(**placeholders)
                except Exception as exc:
                    raise ValueError(
                        f"Formatting failed for key '{key}' with {placeholders}: {exc}"
                    ) from exc
            return current

        # Return lists or dictionaries directly if accessed
        return current


def fmt_ugm3(val: float | int | str | None) -> str:
    """Format particulate PM2.5 concentration in µg/m³.

    Non-negative rounded value, gracefully handling None or string numeric inputs.
    """
    if val is None:
        return "—"
    try:
        f_val = float(val)
        if f_val < 0.0:
            f_val = 0.0
        # If integer or effectively an integer, format as integer
        if abs(f_val - round(f_val)) < 1e-4:
            return f"{round(f_val)}"
        return f"{f_val:.1f}"
    except (ValueError, TypeError):
        return "—"


def local_time(
    ts: datetime | str | None,
    fmt: str = "%d %b %Y, %I:%M %p",
) -> str:
    """Format timestamp converted to Pakistan Standard Time (Asia/Karachi, UTC+5)."""
    if ts is None:
        return "—"

    dt: datetime
    if isinstance(ts, str):
        cleaned = ts.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            return ts
    elif isinstance(ts, datetime):
        dt = ts
    else:
        return "—"

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    pkt_dt = dt.astimezone(PKT_TZ)
    return pkt_dt.strftime(fmt)


def category_name(cat_id: str, lang: str = "en") -> str:
    """Resolve localized AQI category name for the given category ID."""
    clean_id = str(cat_id).strip().lower()
    catalogs = load_catalogs()
    cat = catalogs.get(lang, catalogs["en"])
    aqi_dict = cat.get("aqi", {})
    if clean_id in aqi_dict:
        return str(aqi_dict[clean_id])
    return clean_id.replace("_", " ").title()


def create_jinja_env(
    templates_dir: Path | str = "web/templates",
    lang: str = "en",
    catalog: dict[str, Any] | None = None,
) -> jinja2.Environment:
    """Construct configured Jinja2 Environment for bilingual rendering.

    Enforces strict undefined variables, HTML autoescape, localized filters,
    and translation callable global.
    """
    t_dir = Path(templates_dir)
    if catalog is None:
        catalogs = load_catalogs()
        cat = catalogs.get(lang, catalogs["en"])
    else:
        cat = catalog

    translator = Translator(lang=lang, catalog=cat)
    direction = str(cat.get("meta", {}).get("dir", "ltr" if lang == "en" else "rtl"))

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(t_dir),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )

    # Register custom formatting filters
    env.filters["fmt_ugm3"] = fmt_ugm3
    env.filters["local_time"] = local_time
    env.filters["category_name"] = lambda c: category_name(c, lang=lang)

    # Register global functions and context
    env.globals["t"] = translator
    env.globals["lang"] = lang
    env.globals["dir"] = direction
    env.globals["compute_fan_chart_geometry"] = compute_fan_chart_geometry

    return env
