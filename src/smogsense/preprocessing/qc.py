"""smogsense.preprocessing.qc — Rule-based quality control with bit-flagged outcomes.

Range, flat-line, spike, spatial-outlier, humidity and completeness checks. Never deletes rows:
sets bits in qc_flags and nulls the cleaned value so that raw and cleaned series remain
auditable.

Public contract (implemented in Phase 1):
- apply_qc(df, rules) -> df with pm25_ugm3, qc_flags
- Bit meanings are documented in data/schemas/observations_hourly.schema.yaml.

Specification: docs/data-engineering.md → 'Quality control and low-cost sensor handling'
"""
