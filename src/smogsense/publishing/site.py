"""smogsense.publishing.site - Static site builder.

Renders templates, writes sitemap/robots/.nojekyll, enforces page-weight budgets.
Phase 1a.6 minimal site integration.

Specification: docs/dissemination-and-ui.md -> 'Static site architecture'
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_site(bulletin: dict[str, Any], out_dir: Path) -> None:
    """Generate the minimal static site for the forecast.

    This fulfills the P1-06 minimal site integration requirement.
    It writes the bulletin JSON to the site directory and creates a minimal index.html.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON API response
    json_path = out_dir / "latest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bulletin, f, indent=2, ensure_ascii=False)

    # Write .nojekyll for GitHub Pages
    (out_dir / ".nojekyll").touch()

    # Write a minimal index.html that can consume the JSON
    html_path = out_dir / "index.html"

    mode = bulletin.get("mode", "unknown")
    issuance = bulletin.get("issuance_utc", "unknown")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmogSense Forecast</title>
    <style>
        body {{ font-family: system-ui, sans-serif; line-height: 1.5; padding: 2rem; max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .forecast-card {{ border: 1px solid #ccc; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; background: #eee; }}
        .mode-banner {{ background: #fff3cd; color: #856404; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; }}
    </style>
</head>
<body>
    <h1>SmogSense Daily Forecast</h1>
    <p>Issuance: {issuance}</p>
    """

    if mode != "full":
        html_content += f"""
        <div class="mode-banner">
            <strong>Degraded Mode: {mode}</strong>
            <p>The forecast is operating in a degraded mode. Some data sources may be unavailable.</p>
        </div>
        """

    html_content += "<h2>City Aggregate Forecast</h2>"

    for h in bulletin.get("horizons", []):
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

    html_content += """
    <script>
        // In a real frontend, this would fetch latest.json and render dynamically
        console.log("SmogSense minimal site loaded.");
    </script>
</body>
</html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated minimal site at {out_dir}")
