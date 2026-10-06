"""smogsense.data_ingestion.fixtures_mode — Offline replay of recorded payloads (SMOGSENSE_MODE=fixtures).

Lets every client read tests/fixtures/ instead of the network so that the full pipeline, the
demo and the unit tests run without credentials or internet.

Specification: docs/system-architecture.md → 'Component responsibilities and CLI contract'
"""
