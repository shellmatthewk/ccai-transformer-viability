"""Short synthetic GPU benchmark at the default training tensor sizes."""

import time

import torch

from models import ConvCNP, StationTransformer
from models.common import normal_nll


def benchmark(device_index: int, model_type: type[torch.nn.Module]) -> None:
    device = torch.device(f"cuda:{device_index}")
    batch, contexts, targets = 2, 256, 1024
    x = torch.rand(batch, contexts, 2, device=device).mul(2).sub(1)
    y = torch.randn(batch, contexts, device=device)
    query = torch.rand(batch, targets, 2, device=device).mul(2).sub(1)
    reference = torch.randn(batch, targets, device=device)
    model = model_type().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        mean, log_std = model(x, y, query)
        normal_nll(mean, log_std, reference).backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        mean, log_std = model(x, y, query)
        normal_nll(mean, log_std, reference).backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    print(f"{torch.cuda.get_device_name(device_index)} | {model_type.__name__}: {(time.perf_counter() - start) / 10:.4f} seconds/step")


if __name__ == "__main__":
    for gpu in range(torch.cuda.device_count()):
        for architecture in (ConvCNP, StationTransformer):
            benchmark(gpu, architecture)
