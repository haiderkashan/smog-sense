"""smogsense.features.meteorology — RH (Magnus), wind speed/direction, ventilation coefficient, dilution index, stagnation run-lengths, lower-tropospheric stability.

VC = PBLH x WS (m2/s) in 10 m and 100 m wind variants; dilution index DI = mean(1/VC) over the
block; stagnation run-length = consecutive hours with VC below configurable thresholds.

Public contract (implemented in Phase 2):
- ventilation_coefficient(blh_m, ws_ms, floor=10.0)
- dilution_index(vc_series)

Specification: docs/ml-architecture.md → 'Feature engineering (physics-informed)'
"""
