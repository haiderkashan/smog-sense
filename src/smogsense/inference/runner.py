"""smogsense.inference.runner — Forecast runner.

Builds features as-of T0, loads the promoted model bundle, predicts the 19-level quantiles for h
in {24,48,72} per station and city aggregate, validates against forecast_log.schema.yaml.

Specification: docs/system-architecture.md → 'Daily execution cycle (00:00 UTC)'
"""
