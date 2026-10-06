"""smogsense.data_ingestion.maiac — OPTIONAL MODIS MCD19A2 (MAIAC) AOD reader for research ablations.

Disabled by default. MODIS Aqua/Terra end-of-mission is planned for 2026-27 and orbits are
drifting, so MAIAC is excluded from the operational feature set; this module supports historical
missingness studies only.

Public contract (implemented in Phase 2):
- Gated by sources.maiac.enabled; imports earthaccess/pyhdf lazily.

Specification: docs/data-engineering.md → 'Aerosol optical depth strategy'
"""
