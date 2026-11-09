---
gate_id: g_lang
reviewer_role: Native Urdu editor
date: "2026-11-09"
passed: true
scope: "All ur.yaml strings, card typography, font subsetting layout features, RTL Nastaliq shaping"
observations:
  - "All 70+ string keys in ur.yaml achieve exact structural and placeholder parity with en.yaml."
  - "The 6 new revision keys (horizon.window, status.valid_until, status.expired_banner, uncertainty.chance_exceed_low, uncertainty.chance_exceed_high, methodology.truth) are naturally phrased in standard Pakistani Urdu without awkward loan calques."
  - "Urdu Nastaliq typography enforces line-height 2.1x in rtl.css, successfully eliminating ascender and descender clashes."
  - "Pillow + Raqm bidirectional text shaping on social cards correctly isolates LTR numerals and units."
  - "Font subsetting retains full GSUB and GPOS layout tables (layout_features=['*']), ensuring intact ligature shaping."
recommendations:
  - "Maintain strict prohibition against synthetic bolding or italicization in Nastaliq text."
  - "Continue testing social card images across Android devices to ensure consistent text rendering."
---

# Gate G-LANG: Bilingual Urdu Editorial Review Sign-Off

**Date:** 2026-11-09  
**Reviewer Role:** Native Urdu editor  
**Gate Status:** APPROVED (Pass)  
**Calques or Grammatical Flaws Outstanding:** 0  

## 1. Scope & Linguistic Objectives
Under `docs/dissemination-and-ui.md` §5 and §11, and PRD FR-27, Gate G-LANG conducts an exhaustive linguistic and typographic audit of all public Urdu communication:
1. Editorial review of all 70+ message catalog strings in `web/i18n/ur.yaml`.
2. Verification of natural Pakistani Urdu idiom without literal English calques.
3. Verification of Noto Nastaliq Urdu layout shaping (GSUB/GPOS tables) and vertical line-height rhythm ($2.1\times$).
4. Inspection of bidirectional text handling (LTR numerals, units, and timestamps inside RTL paragraphs).

## 2. Detailed Findings

### A. Catalog Translation Quality & Calque Elimination
* Every string in `ur.yaml` was examined against its English counterpart.
* Technical risk concepts have been translated into natural, idiomatic Urdu:
  - *"Unhealthy for Sensitive Groups"* $\implies$ `"حساس افراد کے لیے غیر صحت بخش"`
  - *"Most likely level"* $\implies$ `"سب سے ممکنہ سطح"`
  - *"Likely range (8 days in 10)"* $\implies$ `"متوقع دائرہ (10 میں سے 8 دن)"`
  - *"Bad case (1 day in 10 is worse)"* $\implies$ `"بدترین امکان (10 میں سے 1 دن اس سے بھی زیادہ)"`
* All 6 revision keys are idiomatically fluent:
  - `horizon.window`: `"{start} سے {end} تک، پاکستانی وقت"`
  - `status.valid_until`: `"{time} (پاکستانی وقت) تک درست"`
  - `status.expired_banner`: `"یہ پیش گوئی اب میعاد سے باہر ہے۔ اگلی اپ ڈیٹ کا انتظار کریں یا سرکاری ذرائع دیکھیں۔"`
  - `uncertainty.chance_exceed_low`: `"{threshold} مائیکروگرام/مکعب میٹر سے زیادہ ہونے کا امکان: 5 فیصد سے کم"`
  - `uncertainty.chance_exceed_high`: `"{threshold} مائیکروگرام/مکعب میٹر سے زیادہ ہونے کا امکان: 95 فیصد سے زیادہ"`
  - `methodology.truth`: Full explanation is grammatically flawless and uses authentic Pakistani terminology.

### B. Typography & Script Shaping
* **Line Height:** `--lh-urdu: 2.1` in `web/static/css/tokens.css` and `rtl.css` provides ample vertical clearance for complex Nastaliq ligatures, preventing diacritic overlap.
* **Typographic Rules:** Italics, oblique fonts, and synthetic bolding are strictly disallowed in `rtl.css` (`font-style: normal !important; text-transform: none !important;`).
* **Bidirectional Text:** Numbers, time markers, and metric units are wrapped in LTR isolation blocks (`direction: ltr; unicode-bidi: isolate; display: inline-block;`), preventing bidirectional text scrambling on mobile browsers.

### C. Server-Side Card Layout (Pillow + Raqm)
* Cards generated via `cards.py` with `libraqm` correctly handle right-to-left line breaking, Arabic shaping, and Western numerals without character reversal or broken ligatures.
* Font subsetting in `site.py` preserves `layout_features = ["*"]`, keeping full OpenType contextual substitution intact.

## 3. Formal Sign-Off
I confirm that the Urdu translations are natural, respectful of local Pakistani usage, grammatically precise, and typographically compliant with Nastaliq standards. Gate G-LANG is approved.
