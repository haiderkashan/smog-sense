"""smogsense.features.fire_exposure — Target-centric, wind-aligned fire exposure indices.

Aggregates FIRMS detections into distance rings and compass sectors around each target and
computes transport-weighted exposure E(t) from FRP, distance, bearing and advective time.
Platform-normalised so that Suomi-NPP retirement and MODIS decline do not create artificial
trends.

Public contract (implemented in Phase 2):
- exposure(detections, target, wind_series, params) -> DataFrame validated by
  fire_exposure.schema.yaml

Specification: docs/ml-architecture.md → 'Fire transport exposure'

Contract update: bearing convention.
"""
