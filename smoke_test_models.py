"""CPU-only architecture and gradient check; it does not train on climate data."""

import torch
from models import ConvCNP, StationTransformer
from models.common import normal_nll


def run(model: torch.nn.Module) -> None:
    context_x, context_y = torch.rand(2, 11, 2).mul(2).sub(1), torch.randn(2, 11)
    target_x, target_y = torch.rand(2, 17, 2).mul(2).sub(1), torch.randn(2, 17)
    mask = torch.tensor([[True] * 11, [True] * 7 + [False] * 4])
    mean, log_std = model(context_x, context_y, target_x, mask)
    assert mean.shape == log_std.shape == target_y.shape
    loss = normal_nll(mean, log_std, target_y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"{model.__class__.__name__}: shape={tuple(mean.shape)}, nll={loss.item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run(ConvCNP(grid_size=32, channels=16))
    run(StationTransformer(d_model=32, nhead=4, layers=1))
