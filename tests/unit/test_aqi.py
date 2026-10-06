"""US EPA 2024 PM2.5 breakpoints.

Test specification (cases to implement):
- Boundary values 9.0/9.1, 35.4/35.5, 55.4/55.5, 125.4/125.5, 225.4/225.5 map to the documented categories
- P(category) sums to 1 for any monotone quantile function
- No index above 500 is ever produced
"""
