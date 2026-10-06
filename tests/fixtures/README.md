# Test fixtures

Fixtures make the whole pipeline runnable **offline and without credentials** (`SMOGSENSE_MODE=fixtures`).

* The canonical demo/test issuance is **2025-11-05T00:00:00Z** (a smog-season day).
* `openaq/` — recorded `/v3/locations` and `/v3/sensors/{id}/hours` pages (trimmed), including one 429 response with
  `x-ratelimit-*` headers and one page with a coverage gap.
* `cams/` — a tiny GRIB (a few grid cells, three lead times) generated programmatically with ecCodes so that no
  copyrighted bulk data is stored.
* `firms/` — FIRMS CSV samples per platform, including a detection-free day.
* `archive/` — two `location-<id>-<yyyymmdd>.csv.gz` files.
* `bulletin_valid_example.json` — the public JSON contract example (validated by `tests/contract`).

Rules: fixtures are synthetic or trimmed public data, each < 200 KB; never include API keys in recorded headers
(the recorder strips `X-API-Key`, `Authorization`); every fixture has a one-line provenance note in its directory.
