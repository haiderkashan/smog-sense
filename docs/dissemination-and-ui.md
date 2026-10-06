# Dissemination and UI

> **Audience:** whoever builds Phase 4 (templates, cards, site) and the reviewers who will judge it. The model is useless to a person who cannot understand it in five seconds on a phone, in Urdu, on a bad connection.
> **No framework code is specified here on purpose**; this document fixes *behaviour, content, design tokens, contracts and budgets*.

## 1. Audiences and communication goals

| Audience | Situation | Needs | Surface |
|---|---|---|---|
| **Citizens** (parents, commuters, workers) | Phone, often Urdu-first, variable bandwidth, WhatsApp-centric | "What should I do today?" in one glance; trust; shareability | Forecast page, WhatsApp card |
| **Schools, employers, event organisers** | Decide closures/schedules | Next-72-hour outlook; chance of the worst category | Forecast page (3 horizons), JSON |
| **Journalists** | Quote and cite | Numbers with units, issuance time, provenance, method, track record | Forecast, methodology, accuracy, JSON |
| **Hospital/municipal planners** | Surge planning | Upper-tail risk (`P(exceed 125.5/225.5)`) | Horizon cards, JSON |
| **Scientists/officials** | Evaluate credibility | Methodology, data sources, verification statistics, honest limitations | Methodology, accuracy |

Goals: **(G1)** understand the headline in ≤ 5 seconds; **(G2)** one clear protective action; **(G3)** uncertainty stated honestly and *understood*; **(G4)** every number traceable to data and time; **(G5)** works offline-ish on 2G (small pages), with JavaScript disabled, in both languages.

## 2. Risk-communication rules for probabilistic forecasts

1. **Lead with category and action, then the number.** A chip ("Hazardous") and one sentence of advice come before any µg m⁻³ value.
2. **Natural frequencies for uncertainty.** "Real levels fall inside this range on about **8 days out of 10**; about **1 day in 10** is worse." Natural frequencies are understood far better than probabilities or percentiles (Gigerenzer & Hoffrage, 1995). Percentages are reserved for *exceedance* statements ("chance of exceeding 125.5 µg m⁻³: 93 %").
3. **Never "worst case".** $\hat q_{.90}$ is the *bad case* (1 day in 10 is worse), not a bound. The plan's phrase "90th percentile worst-case scenario" is replaced throughout (labels: *Bad case*).
4. **The interval is 80 %, not 90 %.** $[q_{.10},q_{.90}]$ contains 80 % of outcomes if calibrated; the label says "8 days in 10".
5. **Never extrapolate an index.** The official EPA AQI ends at 500; dramatic figures such as "AQI 1900" come from consumer apps' extended scales. SmogSense reports **concentration (µg m⁻³) plus category**, and caps nothing artificially: a 400 µg m⁻³ forecast is "Hazardous", full stop.
6. **One transparent scheme.** US EPA 2024 PM2.5 breakpoints on 24-hour means (below), named on the page. Pakistani authorities may publish their own categories; reconcile with the local authority before launch and show both if they differ.
7. **Visible degradation.** Any mode other than `full` shows a banner in plain words (catalogue `mode_banner.*`).
8. **Time stamps everywhere.** "Issued 05:26 Pakistan time"; a 30-hour `valid_until` with a three-path banner when stale.
9. **Calm tone.** The page turns "alarm red" only for ≥ Unhealthy; no exclamation marks, no emojis in advice text; avoid both false reassurance and alarm fatigue.
10. **Escalate cautiously.** Advice is written for the **next-24-hour median category**, escalated one level if the chance of the next-worse category is ≥ 35 % (`configs/bulletin.yaml → advisory_policy`). Illustration with the fixture's day-2 numbers: a *Very unhealthy* median with a 38.6 % chance of *Hazardous* would escalate the advice to Hazardous.
11. **Attribution and disclaimer** on every page and card: research project, not an official forecast; follow official health guidance; data credits including the Copernicus wording ("Generated using Copernicus Atmosphere Monitoring Service information 2026").
12. **Exceedance rounding.** Round exceedance to 5 %, clamp to <5 % / >95 %.
13. **Truth basis.** Clearly state the truth basis.

**AQI categories** (US EPA 2024, 24-hour PM2.5; concentrations truncated to 0.1 µg m⁻³ before categorising):

| Category | Lower bound (µg m⁻³) | Category | Lower bound (µg m⁻³) |
|---|---|---|---|
| Good | 0.0 | Unhealthy | 55.5 |
| Moderate | 9.1 | Very unhealthy | 125.5 |
| Unhealthy for sensitive groups | 35.5 | Hazardous | 225.5 |

## 3. Information architecture

```
/                       Forecast (English)           /ur/                  Forecast (Urdu)
/methodology/           How it works                 /ur/methodology/
/accuracy/              Public scorecard             /ur/accuracy/
/archive/               Past bulletins               /ur/archive/
/forecast/latest.json   Machine-readable bulletin   /forecast/YYYY-MM-DD.json
/cards/latest/lahore-whatsapp-{en,ur}.png            /cards/YYYY-MM-DD/…   (last 14 days)
/og/latest-{en,ur}.png  Open Graph preview           /404.html  /robots.txt  /sitemap.xml
```

Mobile-first home (both languages share the structure; Urdu mirrors inline direction):

```
┌────────────────────────────────┐
│ SmogSense Lahore          اردو │   header + language switch
│ Issued at real generation time  │
├────────────────────────────────┤
│ [ HAZARDOUS ]                   │   category chip (colour + text)
│ 243 µg/m³       most likely     │   hero number
│ Bad case 316 · 1 day in 10      │
├────────────────────────────────┤
│  range chart, 3 horizons on     │   the signature element
│  category-banded background     │
├────────────────────────────────┤
│ What to do                      │   headline + 3–4 actions + notes
├────────────────────────────────┤
│ window_start_local to window_end_local │   clock windows replacing horizon cards
├────────────────────────────────┤
│ How to read this · Sources      │
└────────────────────────────────┘
```

**The signature element — the category-banded range chart.** Three columns (one per 24-hour block) float over six equal-height horizontal bands tinted with the AQI colours at low opacity. Each column shows the 80 % range (light), the 50 % range (darker) and the median tick; a faint line links the medians
(blocks are *means*, so the chart never implies hourly resolution). Readers see *which categories each day could fall into* at a glance. Value-to-position is piecewise-linear between category boundaries (equal band heights), so boundary ticks are labelled with actual µg m⁻³; the axis top is `max(400, 1.05·q95)`.

## 4. Design system (tokens)

**Concept.** Winter-dusk paper and ash neutrals, near-black blue ink, **one** accent — *Ravi blue* (after the river by Lahore; no AQI category is blue). The six AQI colours are the **only saturated colours** on the page, so colour always means "category", never decoration.
Tokens live in `web/static/css/tokens.css`; `configs/bulletin.yaml` mirrors the AQI pairs for the Pillow renderer and a test enforces parity.

| Token | Light | Dark | Contrast (measured, WCAG) |
|---|---|---|---|
| `--bg` / `--fg` | `#F4F1EC` / `#1B1F2A` | `#14171F` / `#ECE8E1` | **14.61** / **14.67** |
| `--fg-muted` | `#5B5650` | `#A8A29A` | 6.45 / 7.08 |
| `--accent` (links, focus) | `#1A5270` | `#7FB6D1` | 7.50 / 8.12 |
| `--rule` | `#D9D3CA` | `#2C313D` | decorative only (1.32) |

| AQI chip | Background | Text | Contrast |
|---|---|---|---|
| Good | `#00E400` | black | 12.10 |
| Moderate | `#FFFF00` | black | 19.56 |
| Unhealthy for sensitive groups | `#FF7E00` | black | 8.24 |
| Unhealthy | `#FF0000` | **black** | 5.25 (white would fail AA at 4.00) |
| Very unhealthy | `#8F3F97` | white | 6.31 |
| Hazardous | `#7E0023` | white | 10.99 |

**Type.** One family per script, both SIL-OFL and already in the container (`fonts-noto-core`): **Noto Sans** (Latin) and **Noto Nastaliq Urdu**. Scale: 1 / 1.125 / 1.375 / 1.75 / 2.5 rem, hero `clamp(3.25rem, 14vw, 6rem)`. Latin line-height 1.5; **Urdu 2.1** (Nastaliq glyphs are tall; the measured line box was ≈ 2× the font size).
**Space:** 4 / 8 / 16 / 24 / 40 px; radius 14 px; reading measure 62 ch. **Focus:** 3 px outline in the accent, offset 2 px. **Motion:** 160 ms, zero under `prefers-reduced-motion`. **Themes:** light and dark via `prefers-color-scheme` plus an explicit `data-theme` override.
**Non-colour cues:** every chip carries its text label; the three red-ish categories add a distinct shape/pattern in the chart and card so the information survives colour-blindness and greyscale printing.

## 5. Bilingual and RTL rules (Urdu Nastaliq)

* **Shaping is a hard technical constraint, verified experimentally.** Rendering the same Urdu sentence with Pillow's **Raqm** layout produced correctly joined, right-to-left Nastaliq; with the **basic** layout (what Matplotlib uses) it produced disconnected, *reversed* glyphs, unreadable. Therefore: **Matplotlib is used only for English-language diagnostics**; all public text, in either language, is drawn with Pillow+Raqm (cards) or by the browser (site). The Pillow wheel bundles Raqm/HarfBuzz/FriBiDi, so no browser is needed in the container (CI asserts `features.check("raqm")`).
* **Fonts.** Noto Nastaliq Urdu is 571 KB as TTF; the web subset (Arabic block + Basic Latin + punctuation, `layout_features=*`; Nastaliq depends on GSUB/GPOS contextual rules, so *never* subset layout features away) is **≈ 107 KB WOFF2**. Budget 160 KB; load only on Urdu pages with `font-display: swap`.
* **Direction and structure.** `<html lang="ur" dir="rtl">`; CSS logical properties (`margin-inline-start`, …) so one stylesheet serves both directions; numerals and units are LTR islands (`<bdi>` / `unicode-bidi: isolate`).
* **Typography.** Line-height ≥ 2.0 (default 2.1); body ≥ 1.125 rem; **no letter-spacing, italics or synthetic bold**; never truncate with an ellipsis (it clips ascenders); allow 30–40 % text expansion.
* **Digits (decision).** **Western digits in both languages** (common in Pakistani media and apps, avoids bidi digit reordering around units). Extended Arabic-Indic digits (۰–۹) render correctly with the same stack and are available as a config switch for a later user-research decision.
* **Time axis (decision, to be validated with readers).** Charts keep time running **left→right** in both languages, with Urdu labels; the card and site share the geometry.
* **Language switch.** Reciprocal links with `hreflang`; labelled in the *other language's own script* ("اردو" / "English"). No automatic redirect; JavaScript may *suggest* Urdu when `navigator.language` starts with `ur` and no choice is stored.
* **Review gates.** Native-speaker editorial review (**G-LANG**) and health-professional review of every advice string (**G-HLTH**) before launch.

## 6. Static site architecture

```
smogsense run daily ─▶ bulletin render ─▶ site build ─▶ site validate ─▶ (host) scripts/publish_ghpages.sh ─▶ GitHub Pages
                                                                                      └─▶ optional: Cloudflare Pages direct upload
```

* **Output** (`site/`): HTML per language, fingerprinted assets under `/assets/` (immutable cache), `forecast/*.json`, `cards/…`, `og/…`, `sitemap.xml`, `robots.txt`, `404.html`, `.nojekyll`, and for the Cloudflare mirror a `_headers` file carrying the CSP and cache policy.
* **No framework, no bundler, no Node.** Jinja2 renders pages inside the Python pipeline. JavaScript (≤ 30 KB, vanilla, `defer`) is *progressive enhancement only*: chart tooltips, `navigator.share` with a `wa.me` fallback, remembering language/theme in `localStorage` inside try/catch, the staleness banner.
* **Strict CSP** (`default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'`): no inline script or style, no third-party hosts, no analytics, no cookies, no CDN fonts. GitHub Pages cannot set response headers, so the CSP is also emitted as a `<meta>` tag (note: `frame-ancestors` is ignored in `<meta>`; Cloudflare's `_headers` sets it properly).
* **No service worker / offline cache** for forecast pages: a stale cached forecast is a safety risk. Pages are served with `max-age=600`.
* **Open data.** `forecast/latest.json` follows `data/schemas/bulletin.schema.json` for journalists and third-party apps (verify the host's CORS header with `curl -I` in Phase 4; Pages typically allows cross-origin reads of static files).
* **Retention.** JSON for every day (rebuilt from the `state` branch if needed); cards for the last 14 days only, which bounds the site at ≈ 50 MB.
* **Share cache-busting.** Share links append `?d=YYYYMMDD`, because WhatsApp caches link previews per URL; the daily query value forces a fresh Open Graph preview.

## 7. Jinja2 template contract

Environment: `autoescape=True`, `StrictUndefined` (a missing variable is a build error), no `|safe` except for SVG generated by our own code, and three filters: `fmt_ugm3(x)`, `local_time(ts, fmt)` (Asia/Karachi), `category_name(id)`. The translator `t(key, **kw)` raises on an unknown key or a missing placeholder.
The header comment of every template in `web/templates/` states its context; the shared contract:

| Variable | Type | Source |
|---|---|---|
| `lang`, `dir` | `'en'\|'ur'`, `'ltr'\|'rtl'` | catalogue `meta.*` |
| `t` | callable | `publishing/i18n.py` (key parity enforced) |
| `site` | `{base_url, build_id, generated_at_utc, asset(path)}` | `publishing/site.py` |
| `page` | `{title, canonical_url, og_image_url, alternate_lang_url}` | per page |
| `bulletin` | dict validated against `bulletin.schema.json` | `publishing/bulletin.py` |
| `categories` | list of `{id, bg, fg, name}` | `configs/bulletin.yaml` + catalogue |
| `geometry` | chart paths/ticks, pure data | `visualization/fan_chart.py` |

Templates and partials: `base`, `index`, `methodology`, `archive`, `accuracy`, `404`; partials `_aqi_chip`, `_fan_chart.svg` (real `<text>`, never outlined glyphs; `role="img"` with `<title>`/`<desc>`; a visually-hidden data table repeats the numbers), `_advisory_block`, `_uncertainty_legend`, `_language_switch`, `_mode_banner`, `_horizon_card`.

## 8. WhatsApp-ready cards

| Format | Size | Budget | Use |
|---|---|---|---|
| Post (4:5) | 1080 × 1350 | ≤ 400 KB | primary share asset (WhatsApp, Telegram, Facebook) |
| Open Graph | 1200 × 630 | ≤ 300 KB | link preview |
| Story/status (9:16) | 1080 × 1920 | ≤ 450 KB | optional |

```
┌──────────────────────────────────┐
│ لاہور اسموگ کی پیش گوئی           │  title · date (local)
│ ┌──────────────────────────────┐ │
│ │   خطرناک   ·  HAZARDOUS      │ │  chip: colour + label (card language first)
│ └──────────────────────────────┘ │
│        243                       │  hero number, µg/m³
│   most likely · bad case 316     │
│   ░░ range chart, 3 days ░░      │  same geometry as the website
│   one action sentence            │
│   "8 days in 10 inside the range"│
│  SmogSense · research forecast   │  source line + short URL
└──────────────────────────────────┘
```

* **Rendering:** Pillow, text through **Raqm** with `direction="rtl"`, `language="ur"`; measurement uses the same engine as drawing so wrapping is correct; 3× supersampling then LANCZOS downscale for crisp edges; PNG quantised to a small palette with `optimize=True` to meet the size budget. Colours come from `configs/bulletin.yaml` (tokens parity-tested).
* **Why PNG + optional JPEG.** WhatsApp recompresses photos sent as images; keeping cards ≤ 400 KB limits recompression artefacts on Nastaliq diacritics. A high-quality JPEG twin is provided for senders who prefer it.
* **Stable URLs.** `/cards/latest/lahore-whatsapp-{en,ur}.png` always points at today's card, so channel admins can bookmark them; dated copies live under `/cards/YYYY-MM-DD/`.
* **Share text** (catalogue `card.whatsapp_text`, both languages) and a `wa.me` link are on every page; the card has an alt text in the JSON.
* **Automation boundary.** Automatic *sending* to WhatsApp subscribers is out of scope (billed; unofficial libraries violate the terms). The free **Telegram Bot API** can post the card to a channel automatically (`publish telegram`).

## 9. Message catalog

Single source of truth: `web/i18n/en.yaml` and `ur.yaml` (73 keys each; key, nesting, list-length and `{placeholder}` parity is a CI test). Category names and headlines (draft v0.1 — **gates G-LANG and G-HLTH pending**):

| Id | English | اردو | Headline (English) |
|---|---|---|---|
| good | Good | اچھی | Air quality is good. |
| moderate | Moderate | معتدل | Air quality is acceptable. Unusually sensitive people may notice symptoms. |
| usg | Unhealthy for sensitive groups | حساس افراد کے لیے غیر صحت بخش | Unhealthy for sensitive groups. |
| unhealthy | Unhealthy | غیر صحت بخش | Unhealthy for everyone. |
| very_unhealthy | Very unhealthy | بہت زیادہ غیر صحت بخش | Very unhealthy: health alert. |
| hazardous | Hazardous | خطرناک | Hazardous: emergency conditions. |

Uncertainty strings (`uncertainty.*`): *Most likely level* · *Likely range (8 days in 10)* · *Bad case (1 day in 10 is worse)* · *Chance of exceeding {threshold} µg/m³: {pct}%*. Notes (`notes.*`): mask effectiveness, indoor smoke, who is most at risk.
Mode banners (`mode_banner.*`): stale CAMS, observations only, baseline only. Rules: placeholders are never translated; sentences are written natively in Urdu (not literal calques); health claims cite an authority and are reviewed before every season.

## 10. Accessibility and performance budgets

| Area | Requirement |
|---|---|
| Standard | WCAG 2.2 AA |
| Contrast | text ≥ 4.5:1 (tokens above all pass, minimum 5.25); non-text UI ≥ 3:1 |
| Keyboard | everything operable; visible focus; skip link ("Skip to the forecast") |
| Semantics | landmarks, one `<h1>` per page, `lang` on every language island, `<bdi>` for numbers in RTL text |
| Chart | `role="img"` + text alternative + hidden data table |
| Motion & zoom | `prefers-reduced-motion` honoured; usable at 200 % zoom; tap targets ≥ 44 px |
| Weight | HTML ≤ 60 KB, CSS ≤ 25 KB, JS ≤ 30 KB, page ≤ 500 KB excluding fonts, Urdu WOFF2 ≤ 160 KB; `site validate` fails the build otherwise |
| Speed | Largest Contentful Paint < 2.5 s on a slow-3G profile; works with JavaScript disabled |
| Privacy | no cookies, no third-party requests, no analytics |

Automated checks in CI: token contrast computed from `tokens.css`, HTML validity of the rendered pages, link checking, per-page weight, language-tree parity. Manual: screen-reader pass (TalkBack in Urdu and English), print/greyscale pass of the card.

## 11. Expert review protocol

**Gates (all before public launch; target: December, before the January peak):**

| Gate | Reviewer (by role) | Scope | Pass criterion |
|---|---|---|---|
| **G-SCI** | Environmental scientist | Methodology page, uncertainty wording, accuracy page honesty | no scientific objection outstanding |
| **G-LANG** (= G-LANG) | Native Urdu editor | all `ur.yaml` strings, card typography | approved text; no calques |
| **G-HLTH** | Public-health physician | all `advice.*` and `notes.*` | approved wording per category |
| **G-UX** | Local climate journalist + 5–8 lay users in Lahore (mixed literacy, phone-only) | comprehension test below | criteria below |

**Comprehension test (15 min per participant, think-aloud, five-second first look):** (1) "What is the air like tomorrow and what would you do?"; (2) "On how many days out of ten will it be *worse* than the upper number?" — correct answer **1**; (3) "Is the upper number the worst possible?" — correct answer **no**; (4) "Which colour/word is worse, orange or purple?"; (5) "Would you share this card? What would you change?"; (6) trust and clarity rating.
**Success:** ≥ 80 % correct on (2)–(4) and ≥ 80 % choose the protective action in (1). Failures are fixed in wording or layout and the test repeated with new participants.
**Record:** one closed GitHub issue per gate with reviewer *role*, date and outcome (no personal data). The shadow-run season is also the field test: every issue raised by readers is triaged within 48 hours.
