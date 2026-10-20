"""smogsense.models.encoders.gru — Two-layer GRU encoder with missingness-aware inputs.

Consumes the six local observation channels (C = 6) per hour and returns latent vector z in R^{32}.

Specification: docs/ml-architecture.md → 'Sequence encoder'
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class LocalGRUEncoder(nn.Module):
    """Two-layer GRU encoder with linear projection to latent space R^32.

    Consumes sequence tensors of shape (B, 72, 6) or (72, 6) across the six local channels:
    1. pm25_log1p_masked (x * m)
    2. obs_minus_cams_pm25 (observed - CAMS PM2.5 error in log1p)
    3. rh_pct (relative humidity 0-100)
    4. t2m_c (2m temperature in Celsius)
    5. pm25_mask (missingness indicator m in {0, 1})
    6. delta_t (elapsed hours since last valid observation)

    Parameter count:
        - Layer 1 GRU (6 -> 64):  13,824
        - Layer 2 GRU (64 -> 64): 24,960
        - Linear proj (64 -> 32):  2,080
        Total parameters:         40,864
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.latent_dim = latent_dim

        # GRU with batch_first=True
        # Dropout is applied between GRU layers if num_layers > 1
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Linear projection from top-layer hidden state to latent space
        self.proj = nn.Linear(hidden_size, latent_dim)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, L, C) or (L, C), where L=72 and C=6.

        Returns:
            Latent representation z of shape (B, latent_dim) or (latent_dim,).
        """
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)

        # gru output: out of shape (B, L, hidden_size), h_n of shape (num_layers, B, hidden_size)
        _, h_n = self.gru(x)

        # Top-layer hidden state at final step: h_n[-1] has shape (B, hidden_size)
        h_last = h_n[-1]

        z = self.proj(h_last)  # shape (B, latent_dim)

        if unbatched:
            z = z.squeeze(0)

        return cast(torch.Tensor, z)
