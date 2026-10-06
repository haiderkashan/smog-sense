"""QC bit flags and gap filling.

Test specification (cases to implement):
- Negative values below -5 are rejected, [-5,0) clipped to 0
- Flat-line of 6 h flagged
- PCHIP never exceeds neighbour envelope (regression test for the cubic-spline overshoot)
- Gaps > 3 h stay missing
"""
