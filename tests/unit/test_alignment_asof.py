"""As-of availability rule.

Test specification (cases to implement):
- For T0 = 00:00Z the selected CAMS base time is 12Z of the previous day
- For T = 06Z: previous 12Z; T = 12Z and 18Z: same-day 00Z
- No feature uses data with available_at > T
"""
