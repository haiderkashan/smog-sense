"""Bearing and Transport calculations.

Test specification (cases to implement):
- Bearing of vector fire -> target: beta
- Wind transport direction: phi = atan2(u, v) (clockwise from North)
- Exposure: a = max(0, cos(phi - beta))
- Worked example: Lahore to Amritsar azimuth matches ~75.5 degrees clockwise from North.
"""
