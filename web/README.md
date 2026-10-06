# web/

Source for the static public site and the WhatsApp cards. There is **no JavaScript framework, no bundler and no Node
toolchain**: pages are rendered by Jinja2 inside the Python pipeline and the result (`site/`) is deployable on any static host.

```
web/
  i18n/en.yaml, ur.yaml        every user-facing string; key parity enforced by tests
  templates/                   Jinja2 pages and partials (context contracts documented at the top of each file)
  static/css/tokens.css        design tokens (the only place colours are defined for the web)
  static/css/base.css, rtl.css layout and Urdu typography (Phase 4)
  static/js/dashboard.js       optional progressive enhancement (Phase 4)
  static/fonts/                generated at build time (git-ignored): subset WOFF2 of Noto Sans / Noto Nastaliq Urdu
  static/img/                  favicon and static artwork
```

Rules (details in `docs/dissemination-and-ui.md`): strict CSP (no inline script/style, no external hosts), full function
with JavaScript disabled, both languages for every string, colour is never the only carrier of meaning, Urdu line-height
2.1, uncertainty is always shown with the natural-frequency wording ("8 days in 10", "1 day in 10").
