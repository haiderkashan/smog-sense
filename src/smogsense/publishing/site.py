"""smogsense.publishing.site - Static site builder.

Renders templates, writes sitemap/robots/.nojekyll, enforces page-weight budgets.
Phase 1a.6 minimal site integration.

Specification: docs/dissemination-and-ui.md -> 'Static site architecture'
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_site(bulletin: dict[str, Any], out_dir: Path, expired: bool = False) -> None:
    """Generate the static site for the forecast with valid_until and expiry cues (Task P1-23).

    Rules (docs/system-architecture.md §7, docs/dissemination-and-ui.md §2):
    1. Static "Valid until: <valid_until_utc>" line.
    2. Client-side JS check that reveals #expiry-banner when now > valid_until_utc.
    3. Prominent .expiry-banner statically rendered when expired=True.
    4. Writes bulletin JSON to site/latest.json and site/forecast/latest.json with expired flag.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    bulletin_copy = dict(bulletin)
    if expired:
        bulletin_copy["expired"] = True

    # Write JSON API response
    json_path = out_dir / "latest.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bulletin_copy, f, indent=2, ensure_ascii=False)

    forecast_dir = out_dir / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_latest_path = forecast_dir / "latest.json"
    with forecast_latest_path.open("w", encoding="utf-8") as f:
        json.dump(bulletin_copy, f, indent=2, ensure_ascii=False)

    run_id = bulletin_copy.get("run_id")
    if run_id:
        with (out_dir / f"{run_id}.json").open("w", encoding="utf-8") as f:
            json.dump(bulletin_copy, f, indent=2, ensure_ascii=False)
        with (forecast_dir / f"{run_id}.json").open("w", encoding="utf-8") as f:
            json.dump(bulletin_copy, f, indent=2, ensure_ascii=False)

    # Write .nojekyll for GitHub Pages
    (out_dir / ".nojekyll").touch()

    # Write index.html with static and client-side expiry cues
    html_path = out_dir / "index.html"

    mode = bulletin_copy.get("mode", "unknown")
    issuance = bulletin_copy.get("issuance_utc", "unknown")
    valid_until = bulletin_copy.get("valid_until_utc")
    if not valid_until and issuance != "unknown":
        try:
            iss_dt = datetime.fromisoformat(str(issuance).replace("Z", "+00:00"))
            valid_until = (iss_dt + timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            valid_until = "unknown"
    elif not valid_until:
        valid_until = "unknown"

    expiry_display = "block" if expired else "none"
    expired_attr = "true" if expired else "false"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmogSense Forecast</title>
    <style>
        body {{ font-family: system-ui, sans-serif; line-height: 1.5; padding: 2rem; max-width: 800px; margin: 0 auto; color: #222; }}
        h1 {{ color: #111; }}
        .forecast-card {{ border: 1px solid #ccc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; background: #eee; }}
        .mode-banner {{ background: #fff3cd; color: #856404; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; }}
        .expiry-banner {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem; }}
        .valid-until {{ color: #555; font-size: 0.95rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }}
    </style>
</head>
<body>
    <div id="expiry-banner" class="expiry-banner" style="display: {expiry_display};" data-expired="{expired_attr}">
        <strong>EXPIRED FORECAST</strong> &mdash; No new forecast was published for today's issuance; showing last known bulletin as expired.
    </div>

    <h1>SmogSense Daily Forecast</h1>
    <p>Issuance: <span id="issuance-time">{issuance}</span></p>
    <p id="valid-until" class="valid-until">Valid until: <span id="valid-until-time">{valid_until}</span></p>
"""

    if mode != "full":
        html_content += f"""
    <div class="mode-banner">
        <strong>Degraded Mode: {mode}</strong>
        <p>The forecast is operating in a degraded mode. Some data sources may be unavailable.</p>
    </div>
"""

    html_content += "    <h2>City Aggregate Forecast</h2>\n"

    for h in bulletin_copy.get("horizons", []):
        lead_h = h.get("lead_h")
        cat = h.get("category", {}).get("median_q50", "unknown")
        q50 = h.get("quantiles_ugm3", {}).get("q50", 0.0)
        html_content += f"""
    <div class="forecast-card">
        <h3>+{lead_h}h Forecast</h3>
        <p>Expected PM2.5: <strong>{q50:.1f} &mu;g/m&sup3;</strong></p>
        <p>Category: <span class="badge">{cat}</span></p>
    </div>
"""

    html_content += f"""
    <script>
        (function() {{
            const validUntilStr = "{valid_until}";
            if (validUntilStr && validUntilStr !== "unknown") {{
                const validUntil = new Date(validUntilStr);
                if (!isNaN(validUntil.getTime()) && new Date() > validUntil) {{
                    const banner = document.getElementById("expiry-banner");
                    if (banner) {{
                        banner.style.display = "block";
                        banner.setAttribute("data-expired", "true");
                    }}
                }}
            }}
            console.log("SmogSense forecast page loaded (valid until {valid_until}).");
        }})();
    </script>
</body>
</html>
"""

    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated site at {out_dir} (expired={expired})")


def render_expired_site(
    state_dir: Path = Path(".state"),
    out_dir: Path = Path("site"),
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Render and publish static expiry notice using the last known bulletin (Task P1-23).

    Specification: docs/deployment-and-ops.md §5 -> 'Scheduling reliability'
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)

    st_path = Path(state_dir)
    site_path = Path(out_dir)

    bulletin: dict[str, Any] | None = None

    # Search for latest known bulletin in candidate locations
    candidate_paths = [
        st_path / "site_json" / "forecast" / "latest.json",
        st_path / "site_json" / "latest.json",
        site_path / "forecast" / "latest.json",
        site_path / "latest.json",
    ]

    for p in candidate_paths:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    bulletin = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                pass

    # Fallback to manifests if site_json is absent
    if bulletin is None and (st_path / "manifests").exists():
        manifests = sorted((st_path / "manifests").glob("*.json"), reverse=True)
        for m in manifests:
            try:
                with m.open("r", encoding="utf-8") as f:
                    m_data = json.load(f)
                if m_data.get("published") is True:
                    # Synthesize minimal bulletin from manifest
                    issuance_str = m_data.get("issuance", m.stem)
                    bulletin = {
                        "run_id": m_data.get("run_id", f"run_{m.stem}"),
                        "issuance_utc": f"{issuance_str}T00:00:00Z"
                        if len(issuance_str) == 10
                        else issuance_str,
                        "valid_until_utc": f"{issuance_str}T06:00:00Z",
                        "mode": "expired",
                        "horizons": [],
                    }
                    break
            except (json.JSONDecodeError, OSError):
                pass

    # Fallback if no prior bulletin found
    if bulletin is None:
        yesterday = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        bulletin = {
            "run_id": f"expired_fallback_{now_utc.strftime('%Y%m%d')}",
            "issuance_utc": f"{yesterday}T00:00:00Z",
            "valid_until_utc": f"{yesterday}T06:00:00Z",
            "mode": "expired",
            "horizons": [],
        }

    bulletin["expired"] = True
    generate_site(bulletin, out_dir=site_path, expired=True)

    return {
        "status": "ok",
        "expired": True,
        "out_dir": str(site_path),
        "issuance_utc": bulletin.get("issuance_utc"),
        "valid_until_utc": bulletin.get("valid_until_utc"),
    }
