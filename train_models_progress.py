"""CUDA-aware training with per-batch progress and epoch-level ETA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import ConvCNP, StationTransformer
from models.common import normal_nll
from train_models import DailyERA5Pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convcnp", "transformer"), required=True)
    parser.add_argument("--stations", type=Path, default=Path("processed/daily_temperature_2010_2021.parquet"))
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--station-variable", default="tg")
    parser.add_argument("--era5-variable", default="t2m")
    parser.add_argument("--era5-offset", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(f"device={device}; torch={torch.__version__}; cuda_available={torch.cuda.is_available()}")
    dataset = DailyERA5Pairs(args.stations, args.era5, args.station_variable, args.era5_variable, args.era5_offset, args.max_context, args.targets_per_day, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    model = (ConvCNP() if args.model == "convcnp" else StationTransformer()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_start = time.perf_counter(); total_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            batch = {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            mean, log_std = model(batch["context_x"], batch["context_y"], batch["target_x"])
            loss = normal_nll(mean, log_std, batch["target_y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            elapsed = time.perf_counter() - epoch_start
            eta = elapsed / step * (len(loader) - step)
            progress.set_postfix(nll=f"{total_loss / step:.4f}", eta=f"{eta / 60:.1f}m")
        epoch_seconds = time.perf_counter() - epoch_start
        remaining_epochs = args.epochs - epoch
        print(f"epoch={epoch:03d} mean_nll={total_loss / len(loader):.5f} elapsed={epoch_seconds / 60:.2f}m estimated_remaining={epoch_seconds * remaining_epochs / 60:.1f}m")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_type": args.model, "device": str(device), "value_mean": dataset.mean, "value_std": dataset.std, "coord_min": dataset.coord_min, "coord_max": dataset.coord_max, "args": vars(args), "training_seconds": time.perf_counter() - total_start}, args.output / f"{args.model}.pt")


if __name__ == "__main__":
    main()
