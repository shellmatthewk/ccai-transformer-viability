"""Shared probability and coordinate utilities."""

from __future__ import annotations

import math
import torch
from torch import Tensor


def normal_nll(mean: Tensor, log_std: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean Gaussian negative log likelihood, optionally over a boolean mask."""
    log_std = log_std.clamp(-7.0, 5.0)
    loss = 0.5 * ((target - mean) / log_std.exp()).square() + log_std + 0.5 * math.log(2.0 * math.pi)
    return loss.masked_select(mask).mean() if mask is not None else loss.mean()


def sinusoidal_coordinates(x: Tensor, num_frequencies: int = 6) -> Tensor:
    """Encode normalized latitude/longitude coordinates in [-1, 1]."""
    frequencies = (2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype)) * math.pi
    angles = x.unsqueeze(-1) * frequencies
    return torch.cat((x, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
