"""Final split-aware training with validation, early stopping, and best checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from experiment_data import ReconstructionDataset, fit_normalization, read_stations
from models import ConvCNP, StationTransformer
from models.common import normal_nll


def make_model(name: str) -> torch.nn.Module:
    return ConvCNP() if name == "convcnp" else StationTransformer()


@torch.inference_mode()
def validation_nll(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); total = 0.0
    for batch in loader:
        context_x = batch["context_x"].to(device, non_blocking=True)
        context_y = batch["context_y"].to(device, non_blocking=True)
        target_x = batch["target_x"].to(device, non_blocking=True)
        target_y = batch["target_y"].to(device, non_blocking=True)
        mean, log_std = model(context_x, context_y, target_x)
        total += normal_nll(mean, log_std, target_y).item()
    return total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convcnp", "transformer"), required=True)
    parser.add_argument("--experiment", choices=("global", "source"), required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--station-variable", default="tg")
    parser.add_argument("--era5-variable", default="t2m")
    parser.add_argument("--era5-offset", type=float, default=-273.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--longitude-cutoff", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    output = args.output_root / args.experiment / args.model
    best_path = output / "best.pt"
    if best_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {best_path}; pass --overwrite intentionally.")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_region = "global" if args.experiment == "global" else "source"
    print(f"device={device}; model={args.model}; experiment={args.experiment}; region={train_region}")
    frame = read_stations(args.stations, args.station_variable, ("train", "validation"))
    normalization = fit_normalization(frame, args.era5, args.station_variable, train_region, args.longitude_cutoff)
    common = dict(frame=frame, era5_path=args.era5, station_variable=args.station_variable, era5_variable=args.era5_variable, era5_offset=args.era5_offset, context_region=train_region, target_region=train_region, normalization=normalization, max_context=args.max_context, targets_per_day=args.targets_per_day, cutoff=args.longitude_cutoff)
    train_data = ReconstructionDataset(split="train", seed=args.seed, training=True, **common)
    validation_data = ReconstructionDataset(split="validation", seed=args.seed + 1, training=False, **common)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    daily_means = train_data.daily_target_means()
    extreme_thresholds = {"low": float(np.quantile(daily_means, 0.01)), "high": float(np.quantile(daily_means, 0.99)), "definition": "training-region daily spatial mean 1st/99th percentiles"}
    model = make_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_value, best_epoch, stale, history = float("inf"), 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        model.train(); total = 0.0; epoch_start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            context_x = batch["context_x"].to(device, non_blocking=True)
            context_y = batch["context_y"].to(device, non_blocking=True)
            target_x = batch["target_x"].to(device, non_blocking=True)
            target_y = batch["target_y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mean, log_std = model(context_x, context_y, target_x)
            loss = normal_nll(mean, log_std, target_y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item(); progress.set_postfix(train_nll=f"{total / step:.4f}")
        train_value = total / len(train_loader)
        val_value = validation_nll(model, validation_loader, device)
        elapsed = time.perf_counter() - epoch_start
        improved = val_value < best_value
        if improved:
            best_value, best_epoch, stale = val_value, epoch, 0
            torch.save({"model": model.state_dict(), "model_type": args.model, "experiment": args.experiment, "epoch": epoch, "validation_nll": val_value, "normalization": normalization.to_dict(), "extreme_thresholds": extreme_thresholds, "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, best_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_nll": train_value, "validation_nll": val_value, "seconds": elapsed, "best": improved})
        print(f"epoch={epoch:03d} train_nll={train_value:.5f} val_nll={val_value:.5f} best_epoch={best_epoch} stale={stale}/{args.patience} elapsed={elapsed:.1f}s")
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch}; best_epoch={best_epoch}; best_val_nll={best_value:.5f}")
            break
    torch.save({"model": model.state_dict(), "model_type": args.model, "experiment": args.experiment, "epoch": history[-1]["epoch"], "normalization": normalization.to_dict(), "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, output / "last.pt")
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "best_validation_nll": best_value, "total_seconds": time.perf_counter() - started, "history": history}, handle, indent=2)
    print(f"saved best={best_path} (epoch {best_epoch}); last={output / 'last.pt'}")


if __name__ == "__main__":
    main()
