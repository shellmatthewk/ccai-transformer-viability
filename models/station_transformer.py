"""Cross-attention transformer neural process for irregular station contexts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import sinusoidal_coordinates


class StationTransformer(nn.Module):
    """Encode station tokens and decode predictions at grid pseudo-token positions."""

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        coord_dim = 2 + 4 * 6
        self.context_embed = nn.Sequential(nn.Linear(coord_dim + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.target_embed = nn.Sequential(nn.Linear(coord_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, context_x: Tensor, context_y: Tensor, target_x: Tensor, context_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if context_mask is None:
            context_mask = torch.ones(context_x.shape[:2], dtype=torch.bool, device=context_x.device)
        context = torch.cat((sinusoidal_coordinates(context_x), context_y.unsqueeze(-1)), dim=-1)
        memory = self.encoder(self.context_embed(context), src_key_padding_mask=~context_mask)
        queries = self.target_embed(sinusoidal_coordinates(target_x))
        decoded, _ = self.cross_attention(queries, memory, memory, key_padding_mask=~context_mask, need_weights=False)
        mean, log_std = self.head(decoded).unbind(-1)
        return mean, log_std.clamp(-7.0, 5.0)
