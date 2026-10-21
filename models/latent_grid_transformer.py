"""All-station transformer using a fixed grid of latent inducing tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import sinusoidal_coordinates


class LatentGridTransformer(nn.Module):
    """Compress all irregular stations into gridded latents before decoding targets.

    Complexity is O(N*L + L^2 + L*T), where N is the station count, L is the
    number of inducing tokens, and T is the number of target queries. This avoids
    the O(N^2) station self-attention used by the pilot transformer.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 3, latent_side: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        coord_dim = 26
        axis = torch.linspace(-1, 1, latent_side)
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        self.register_buffer("latent_coordinates", torch.stack((gx, gy), dim=-1).reshape(-1, 2), persistent=True)
        self.station_embed = nn.Sequential(nn.Linear(coord_dim + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.latent_embed = nn.Sequential(nn.Linear(coord_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.latent_bias = nn.Parameter(torch.zeros(latent_side * latent_side, d_model))
        self.station_to_latent = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.latent_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.target_embed = nn.Sequential(nn.Linear(coord_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.target_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, context_x: Tensor, context_y: Tensor, target_x: Tensor, context_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if context_mask is None:
            context_mask = torch.ones(context_x.shape[:2], dtype=torch.bool, device=context_x.device)
        stations = self.station_embed(torch.cat((sinusoidal_coordinates(context_x), context_y.unsqueeze(-1)), dim=-1))
        coordinates = self.latent_coordinates.to(dtype=context_x.dtype)
        latent = self.latent_embed(sinusoidal_coordinates(coordinates)) + self.latent_bias
        latent = latent.unsqueeze(0).expand(context_x.shape[0], -1, -1)
        latent_update, _ = self.station_to_latent(latent, stations, stations, key_padding_mask=~context_mask, need_weights=False)
        latent = self.latent_encoder(latent + latent_update)
        targets = self.target_embed(sinusoidal_coordinates(target_x))
        decoded, _ = self.target_attention(targets, latent, latent, need_weights=False)
        mean, log_std = self.head(decoded).unbind(-1)
        return mean, log_std.clamp(-7.0, 5.0)
