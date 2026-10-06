"""smogsense.visualization.cards — Pillow + Raqm renderer for WhatsApp cards and Open Graph images.

Draws 1080x1350 cards and 1200x630 OG images with 3x supersampling; all text (English and Urdu
Nastaliq) goes through Pillow's Raqm layout engine with direction='rtl', language='ur'.

Specification: docs/dissemination-and-ui.md → 'WhatsApp-ready cards'
"""
