"""smogsense.models.heads — Residual quantile head (46 -> 64 -> 57).

MLP 46 -> 64 (ReLU) -> 57 (linear) predicting local quantile residuals Delta z_{tau, h}
for 19 levels x 3 horizons. Last layer is zero-initialised so an unadapted calibrator
reproduces M5 exactly. Monotone rearrangement applied at inference.

Specification: docs/ml-architecture.md → 'Quantile loss, the residual head, and the quantile function'
"""

from __future__ import annotations

from typing import cast, overload

import numpy as np
import torch
from torch import nn

from smogsense.models.encoders.gru import LocalGRUEncoder

# Canonical 19 quantile knots: tau in {0.05, 0.10, ..., 0.95}
CANONICAL_QUANTILES: tuple[float, ...] = tuple(round(0.05 + 0.05 * i, 2) for i in range(19))
CANONICAL_HORIZONS: tuple[int, ...] = (24, 48, 72)


@overload
def rearrange_quantiles(q_preds: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


@overload
def rearrange_quantiles(q_preds: np.ndarray, dim: int = -1) -> np.ndarray: ...


def rearrange_quantiles(
    q_preds: torch.Tensor | np.ndarray,
    dim: int = -1,
) -> torch.Tensor | np.ndarray:
    """Monotone quantile rearrangement (Chernozhukov et al., 2010).

    Sorts predicted quantiles along the specified dimension to restore monotonicity.
    By the rearrangement inequality, sorting never increases total pinball loss.

    Args:
        q_preds: Quantile predictions tensor or numpy array.
        dim: Dimension along which quantiles are indexed (default: -1).

    Returns:
        Monotonically non-decreasing quantile predictions of identical shape.
    """
    if isinstance(q_preds, np.ndarray):
        return np.sort(q_preds, axis=dim)
    sorted_q, _ = torch.sort(q_preds, dim=dim)
    return sorted_q


class ResidualQuantileHead(nn.Module):
    """Residual quantile regression head predicting local quantile residuals.

    Consumes a 46-dimensional context vector u:
    - 32 latent features z (from LocalGRUEncoder)
    - 1 trailing recent CAMS bias
    - 1 station relative humidity at issuance T
    - 3 ln(VC) ventilation coefficient block means (24h, 48h, 72h)
    - 9 base quantiles z_hat^(5)_{tau, h} for tau in {0.10, 0.50, 0.90} x h in {24, 48, 72}

    Emits 57 outputs (3 horizons x 19 quantiles).
    The output linear layer is strictly zero-initialized so that Delta z == 0
    before adaptation, bit-for-bit recovering the frozen physics base (M5).

    Parameter count:
        - Layer 1 (46 -> 64):  3,008 (weight: 64*46=2944, bias: 64)
        - Layer 2 (64 -> 57):  3,705 (weight: 57*64=3648, bias: 57)
        Total parameters:      6,713
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dim: int = 64,
        output_dim: int = 57,
        n_horizons: int = 3,
        n_quantiles: int = 19,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.dropout_rate = dropout

        if n_horizons * n_quantiles != output_dim:
            raise ValueError(
                f"n_horizons ({n_horizons}) * n_quantiles ({n_quantiles}) "
                f"must equal output_dim ({output_dim})"
            )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if zero_init:
            self.zero_init_last_layer()

    def zero_init_last_layer(self) -> None:
        """Zero-initialize the final layer (weights and bias).

        Guarantees Delta z == 0 before adaptation, reproducing M5 exactly.
        """
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        u: torch.Tensor,
        rearrange: bool = False,
        reshape: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            u: Context tensor of shape (B, 46) or (46,).
            rearrange: Whether to apply monotone rearrangement along quantiles (dim=-1).
            reshape: Whether to reshape output to (B, 3, 19) or keep (B, 57).

        Returns:
            Residual quantile tensor Delta z of shape (B, 3, 19) (if reshape=True)
            or (B, 57) (if reshape=False), or unbatched (3, 19) / (57,).
        """
        unbatched = u.ndim == 1
        if unbatched:
            u = u.unsqueeze(0)

        h1 = self.act(self.fc1(u))
        h1 = self.dropout(h1)
        raw_out = self.fc2(h1)  # shape (B, 57)

        grid = raw_out.view(-1, self.n_horizons, self.n_quantiles)  # shape (B, 3, 19)
        if rearrange:
            grid = rearrange_quantiles(grid, dim=-1)

        out = grid if reshape else grid.reshape(-1, self.output_dim)

        if unbatched:
            out = out.squeeze(0)

        return cast(torch.Tensor, out)


class ResidualCalibrator(nn.Module):
    """Complete differentiable neural calibrator (LocalGRUEncoder + ResidualQuantileHead).

    Total parameter count: 40,864 + 6,713 = 47,577.
    """

    def __init__(
        self,
        encoder: LocalGRUEncoder | None = None,
        head: ResidualQuantileHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else LocalGRUEncoder()
        self.head = head if head is not None else ResidualQuantileHead()

    @property
    def num_parameters(self) -> int:
        """Total parameters across encoder and head."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x_seq: torch.Tensor,
        context_tabular: torch.Tensor,
        base_quantiles: torch.Tensor | None = None,
        rearrange: bool = False,
    ) -> torch.Tensor:
        """Forward pass through encoder and residual head.

        Args:
            x_seq: Local 6-channel sequence tensor (B, 72, 6).
            context_tabular: 14-dimensional tabular context vector (B, 14)
                (CAMS bias (1) + station RH (1) + ln(VC) means (3) + base quantiles subset (9)).
            base_quantiles: Optional full base quantiles (B, 3, 19) to add Delta z to.
            rearrange: Whether to apply monotone sorting.

        Returns:
            If base_quantiles is provided:
                Predicted quantiles z_hat^(7) = z_hat^(5) + Delta z of shape (B, 3, 19).
            Else:
                Local residuals Delta z of shape (B, 3, 19).
        """
        z = self.encoder(x_seq)  # (B, 32)
        u = torch.cat([z, context_tabular], dim=-1)  # (B, 46)
        delta_z = self.head(u, rearrange=False, reshape=True)  # (B, 3, 19)

        if base_quantiles is not None:
            if base_quantiles.ndim == 2 and base_quantiles.shape[-1] == 57:
                base_quantiles = base_quantiles.view(-1, 3, 19)
            pred = base_quantiles + delta_z
            if rearrange:
                pred = rearrange_quantiles(pred, dim=-1)
            return cast(torch.Tensor, pred)

        if rearrange:
            delta_z = rearrange_quantiles(delta_z, dim=-1)
        return cast(torch.Tensor, delta_z)
