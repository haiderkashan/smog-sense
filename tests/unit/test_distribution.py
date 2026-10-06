"""Quantile-function object and rearrangement.

Test specification (cases to implement):
- Crossing quantiles are sorted and pinball loss does not increase
- Lower tail never negative
- ppf(cdf(x)) == x on knots
"""
