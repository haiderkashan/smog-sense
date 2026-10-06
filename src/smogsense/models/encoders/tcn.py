"""smogsense.models.encoders.tcn — Causal dilated temporal convolution encoder.

Residual blocks with dilations {1,2,4,8,16}, kernel 3, two convolutions per block: receptive
field 1 + 2(k-1) sum(d) = 125 hours >= L = 72.

Specification: docs/ml-architecture.md → 'Sequence encoder'
"""
