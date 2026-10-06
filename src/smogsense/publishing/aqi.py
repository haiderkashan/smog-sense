"""smogsense.publishing.aqi — US EPA 2024 PM2.5 AQI categories and probabilities.

Breakpoints on 24-h means: 0-9.0, 9.1-35.4, 35.5-55.4, 55.5-125.4, 125.5-225.4, 225.5+. Computes
P(category) from the quantile function. Never reports an index above 500.

Specification: docs/dissemination-and-ui.md → 'Risk-communication rules for probabilistic forecasts'
"""
