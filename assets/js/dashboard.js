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
        if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
          window.dispatchEvent(
            new CustomEvent('smogsense:expired', {
              detail: { validUntil: validUntilIso, expiredAt: nowEpoch },
            })
          );
        }
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
  function updateThemeButtonIcon(theme) {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const iconSpan = btn.querySelector('.theme-toggle__icon');
    if (iconSpan) {
      iconSpan.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }

  function initPreferences() {
    try {
      const savedTheme = localStorage.getItem('smogsense_theme');
      if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeButtonIcon(savedTheme);
      } else {
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        updateThemeButtonIcon(prefersDark ? 'dark' : 'light');
      }
    } catch (e) {
      // localStorage may be disabled in restricted/private browsing modes
    }
  }

  function setupThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;

    btn.addEventListener('click', function () {
      const currentTheme = document.documentElement.getAttribute('data-theme') ||
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';

      document.documentElement.setAttribute('data-theme', nextTheme);
      updateThemeButtonIcon(nextTheme);

      try {
        localStorage.setItem('smogsense_theme', nextTheme);
      } catch (e) {
        // Ignored in private browsing
      }
    });
  }

  // --------------------------------------------------------------------------
  // 4. Initialisation
  // --------------------------------------------------------------------------
  function init() {
    initPreferences();
    setupThemeToggle();
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
