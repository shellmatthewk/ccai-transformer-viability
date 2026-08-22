"""A compact convolutional conditional neural process for spatial fields."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvCNP(nn.Module):
    """Rasterize irregular contexts, convolve a latent grid, and query it bilinearly.

    Coordinates must be normalized to [-1, 1], ordered as (longitude, latitude).
    Values are standardized using statistics fit on the training partition only.
    """

    def __init__(self, grid_size: int = 64, channels: int = 64, sigma: float = 0.08) -> None:
        super().__init__()
        self.grid_size, self.sigma = grid_size, sigma
        self.encoder = nn.Sequential(nn.Linear(1, channels), nn.GELU(), nn.Linear(channels, channels))
        self.decoder = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 5, padding=2), nn.GELU(),
            nn.Conv2d(channels, channels, 5, padding=2), nn.GELU(), nn.Conv2d(channels, 2, 1),
        )

    def _rasterize(self, context_x: Tensor, context_y: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        axis = torch.linspace(-1, 1, self.grid_size, device=context_x.device, dtype=context_x.dtype)
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        grid = torch.stack((gx, gy), dim=-1).view(1, 1, self.grid_size, self.grid_size, 2)
        squared_distance = (context_x[:, :, None, None, :] - grid).square().sum(-1)
        weights = torch.exp(-0.5 * squared_distance / self.sigma**2) * context_mask[:, :, None, None]
        density = weights.sum(1).clamp_min(1e-6)
        encoded = self.encoder(context_y.unsqueeze(-1))
        features = torch.einsum("bnhw,bnc->bchw", weights, encoded) / density[:, None]
        return features, density

    def forward(self, context_x: Tensor, context_y: Tensor, target_x: Tensor, context_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if context_mask is None:
            context_mask = torch.ones(context_x.shape[:2], dtype=torch.bool, device=context_x.device)
        features, density = self._rasterize(context_x, context_y, context_mask)
        field = self.decoder(torch.cat((features, density[:, None]), dim=1))
        samples = F.grid_sample(field, target_x[:, None], mode="bilinear", align_corners=True)
        mean, log_std = samples[:, 0, 0], samples[:, 1, 0]
        return mean, log_std.clamp(-7.0, 5.0)
