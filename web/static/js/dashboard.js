/* Progressive enhancement ONLY. The site is fully usable with JavaScript disabled. Implemented in Phase 4.
   Contract (vanilla ES module, <= 30 KB, no dependencies, no network calls):
   - hover/focus tooltips on the fan chart (reads data-* attributes already in the SVG)
   - 'Share' button: navigator.share({text,url}) when available, else a wa.me click-to-share link
   - remember the visitor's language/theme choice in localStorage (try/catch; works without it)
   - nothing here may be required to read the forecast
   Specification: docs/dissemination-and-ui.md -> 'Static site architecture'. */
