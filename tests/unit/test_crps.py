"""CRPS estimator.

Test specification (cases to implement):
- Equals analytic CRPS of a Normal within 0.5% (19 levels) and 5% (7 levels)
- Equals |y - q| for point forecasts
- Cross-check against scoringrules ensemble CRPS
- 2 x integral of pinball over (0,1) identity
"""
