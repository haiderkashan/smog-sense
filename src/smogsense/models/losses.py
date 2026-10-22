"""smogsense.models.losses — Pinball (quantile) loss and aggregates.

rho_tau(u) = u (tau - 1[u<0]); masked mean over levels, horizons and valid targets.

Specification: docs/ml-architecture.md → 'Quantile loss, the residual head, and the quantile function'
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from smogsense.models.heads import CANONICAL_QUANTILES


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    taus: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Compute asymmetric quantile pinball loss.

    Formula:
        rho_tau(u) = u * (tau - 1[u < 0]) = max(tau * u, (tau - 1) * u)
        where error u = y_true - y_pred.

    Args:
        y_true: Ground truth target tensor. Broadcastable against y_pred.
        y_pred: Predicted quantile tensor. Last dimension typically indexes quantiles.
        taus: Quantile levels tau in (0, 1). Can be a scalar, sequence, or tensor.

    Returns:
        Pinball loss tensor matching the broadcast shape of y_true and y_pred.
    """
    if isinstance(taus, (float, int)):
        tau_tensor = torch.tensor(float(taus), device=y_pred.device, dtype=y_pred.dtype)
    elif isinstance(taus, torch.Tensor):
        tau_tensor = taus.to(device=y_pred.device, dtype=y_pred.dtype)
    else:
        tau_tensor = torch.tensor(list(taus), device=y_pred.device, dtype=y_pred.dtype)

    error = y_true - y_pred

    # Ensure tau_tensor can broadcast across the quantile dimension
    # If tau_tensor is 1D and y_pred has >1 dimensions, align tau along dim=-1
    if tau_tensor.ndim == 1 and y_pred.ndim > 1:
        shape = [1] * (y_pred.ndim - 1) + [tau_tensor.shape[0]]
        tau_tensor = tau_tensor.view(*shape)

    return torch.maximum(tau_tensor * error, (tau_tensor - 1.0) * error)


class MaskedResidualPinballLoss(nn.Module):
    """Masked pinball loss over 19 quantiles x 3 horizons with target validity masking.

    Formula (docs/ml-architecture.md §5):
        L(theta) = (1 / sum(mu)) * sum_{(s,T,h) in B} mu_{s,T,h} * (1/19) * sum_{tau in Q} rho_tau(z - z_hat)
        where z_hat = z_base + delta_z (or z_hat provided directly).

    Stateless and compatible with torch.func / functional_call for Meta-SGD.
    """

    taus: torch.Tensor

    def __init__(
        self,
        quantiles: Sequence[float] = CANONICAL_QUANTILES,
        n_horizons: int = 3,
    ) -> None:
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = len(quantiles)
        tau_tensor = torch.tensor(list(quantiles), dtype=torch.float32)
        self.register_buffer("taus", tau_tensor)

    def forward(
        self,
        y_true: torch.Tensor,
        delta_z: torch.Tensor,
        z_base: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the masked residual pinball loss.

        Args:
            y_true: Ground truth targets (log1p PM2.5) of shape (B, 3) or (B, 3, 1).
            delta_z: Predicted residuals of shape (B, 3, 19) or (B, 57).
            z_base: Optional base predictions of shape (B, 3, 19) or (B, 57).
                If None, delta_z is treated as full quantile prediction z_hat.
            mask: Optional validity mask of shape (B, 3) or (B, 3, 1).
                1.0 for valid target block means (>= 18/24h), 0.0 otherwise.

        Returns:
            Scalar loss tensor (reduction="mean" weighted by mask).
        """
        # Ensure delta_z is (B, n_horizons, n_quantiles)
        n_q = self.n_quantiles
        if delta_z.ndim == 2 and delta_z.shape[-1] == self.n_horizons * n_q:
            delta_z = delta_z.view(-1, self.n_horizons, n_q)

        # Compute full prediction z_hat
        if z_base is not None:
            if z_base.ndim == 2 and z_base.shape[-1] == self.n_horizons * n_q:
                z_base = z_base.view(-1, self.n_horizons, n_q)
            y_pred = z_base + delta_z
        else:
            y_pred = delta_z

        # Ensure y_true is broadcastable: (B, 3, 1)
        if y_true.ndim == 2:
            y_true = y_true.unsqueeze(-1)
        elif y_true.ndim == 1:
            y_true = y_true.view(-1, 1, 1)

        # Pinball loss per quantile: shape (B, 3, 19)
        taus = self.taus.to(device=y_pred.device, dtype=y_pred.dtype).view(1, 1, -1)
        error = y_true - y_pred
        loss_per_q = torch.maximum(taus * error, (taus - 1.0) * error)

        # Average over the 19 quantiles: shape (B, 3)
        loss_per_horizon = loss_per_q.mean(dim=-1)

        # Apply validity mask
        if mask is not None:
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask_float = mask.to(dtype=loss_per_horizon.dtype, device=loss_per_horizon.device)
            masked_loss = loss_per_horizon * mask_float
            mask_sum = mask_float.sum()
            safe_mask_sum = torch.clamp(mask_sum, min=1.0)
            loss = torch.where(
                mask_sum > 0,
                masked_loss.sum() / safe_mask_sum,
                torch.zeros((), dtype=loss_per_horizon.dtype, device=loss_per_horizon.device),
            )
        else:
            loss = loss_per_horizon.mean()

        return loss
