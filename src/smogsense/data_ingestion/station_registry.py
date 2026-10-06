"""smogsense.data_ingestion.station_registry — Versioned station registry builder.

Turns OpenAQ location metadata into the station registry (domain assignment, reference vs low-
cost class, uptime, eligibility) and snapshots it with a registry_version so that experiments
are reproducible even when OpenAQ metadata changes.

Public contract (implemented in Phase 1):
- build_registry(domain) -> DataFrame validated by data/schemas/station_registry.schema.yaml

Specification: docs/data-engineering.md → 'Quality control and low-cost sensor handling'
"""
