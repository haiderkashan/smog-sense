/**
 * SmogSense Progressive Enhancement Client Script
 * Contract: Vanilla JS, <= 30 KB, zero dependencies, zero network calls.
 * Specification: docs/dissemination-and-ui.md §6
 */

(function () {
  'use strict';

  // --------------------------------------------------------------------------
  // 1. Expiry Banner & Staleness Check (Scheduler-Independent)
  // --------------------------------------------------------------------------
  function checkForecastExpiry() {
    const banner = document.getElementById('expiry-banner');
    if (!banner) return;

    const validUntilIso = banner.getAttribute('data-valid-until');
    if (!validUntilIso) return;

    try {
      const validUntilEpoch = new Date(validUntilIso).getTime();
      const nowEpoch = Date.now();

      if (!isNaN(validUntilEpoch) && nowEpoch > validUntilEpoch) {
        banner.removeAttribute('hidden');
        banner.setAttribute('data-expired', 'true');
      }
    } catch (err) {
      // Graceful fallback: do not interrupt user experience
      console.warn('SmogSense expiry check error:', err);
    }
  }

  // --------------------------------------------------------------------------
  // 2. Web Share API & Native WhatsApp Fallback
  // --------------------------------------------------------------------------
  function setupShareButton() {
    const btn = document.getElementById('btn-share');
    if (!btn) return;

    btn.addEventListener('click', async function (event) {
      event.preventDefault();

      const title = btn.getAttribute('data-title') || document.title;
      const text = btn.getAttribute('data-text') || '';
      const url = btn.getAttribute('data-url') || window.location.href;

      if (navigator.share) {
        try {
          await navigator.share({ title, text, url });
          return;
        } catch (err) {
          // If aborted by user (AbortError), no fallback needed
          if (err.name === 'AbortError') return;
        }
      }

      // Fallback: Copy URL or open WhatsApp
      const waUrl = 'https://wa.me/?text=' + encodeURIComponent(text + '\n' + url);
      window.open(waUrl, '_blank', 'noopener,noreferrer');
    });
  }

  // --------------------------------------------------------------------------
  // 3. Theme & Preference Persistence (Wrapped in try/catch)
  // --------------------------------------------------------------------------
  function initPreferences() {
    try {
      const savedTheme = localStorage.getItem('smogsense_theme');
      if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
      }
    } catch (e) {
      // localStorage may be disabled in restricted/private browsing modes
    }
  }

  // --------------------------------------------------------------------------
  // 4. Initialisation
  // --------------------------------------------------------------------------
  function init() {
    initPreferences();
    checkForecastExpiry();
    setupShareButton();

    // Check expiry periodically (every 60 seconds)
    setInterval(checkForecastExpiry, 60000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
