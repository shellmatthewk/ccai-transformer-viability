"""Train full-context ConvCNP and latent transformer models for daily precipitation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.common import normal_nll
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer
from precipitation_data import (
    PrecipitationDataset,
    fit_precipitation_normalization,
    padded_collate,
    read_precipitation_stations,
)


def make_model(name: str, latent_side: int) -> torch.nn.Module:
    return ConvCNP() if name == "convcnp" else LatentGridTransformer(latent_side=latent_side)


def add_variance_floor(log_std: torch.Tensor, minimum_std: float) -> torch.Tensor:
    """Smoothly enforce a minimum predictive variance in normalized log1p space."""
    log_floor_variance = log_std.new_tensor(2.0 * math.log(minimum_std))
    return 0.5 * torch.logaddexp(2.0 * log_std, log_floor_variance)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    minimum_std: float,
) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        mean, log_std = model(
            batch["context_x"].to(device, non_blocking=True),
            batch["context_y"].to(device, non_blocking=True),
            batch["target_x"].to(device, non_blocking=True),
            batch["context_mask"].to(device, non_blocking=True),
        )
        log_std = add_variance_floor(log_std, minimum_std)
        total += normal_nll(mean, log_std, batch["target_y"].to(device, non_blocking=True)).item()
    return total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("convcnp", "latent_transformer"), required=True)
    parser.add_argument("--experiment", choices=("global", "source"), required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--latent-side", type=int, default=16)
    parser.add_argument("--longitude-cutoff", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-normalized-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("experiments_precipitation"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.minimum_normalized_std <= 0:
        parser.error("--minimum-normalized-std must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    output = args.output_root / f"seed_{args.seed}" / args.experiment / args.model
    best_path = output / "best.pt"
    if best_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {best_path}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_region = "global" if args.experiment == "global" else "source"
    frame = read_precipitation_stations(args.stations, ("train", "validation"), args.era5)
    norm = fit_precipitation_normalization(frame, args.era5, train_region, args.longitude_cutoff)
    common = dict(
        frame=frame,
        era5_path=args.era5,
        context_region=train_region,
        target_region=train_region,
        normalization=norm,
        targets_per_day=args.targets_per_day,
        retention=1.0,
        cutoff=args.longitude_cutoff,
    )
    train_data = PrecipitationDataset(split="train", seed=args.seed, training=True, **common)
    validation_data = PrecipitationDataset(split="validation", seed=args.seed + 1, training=False, **common)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=padded_collate)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=padded_collate)
    counts = np.asarray([len(rows) for rows in train_data.groups.values()])
    print(
        f"device={device}; variable=precipitation; model={args.model}; region={train_region}; "
        f"daily_stations={counts.min()}-{counts.max()} median={int(np.median(counts))}; "
        f"log1p_mean={norm.value_mean:.4f}; log1p_std={norm.value_std:.4f}"
    )

    model = make_model(args.model, args.latent_side).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_value, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        total = 0.0
        epoch_start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="day", dynamic_ncols=True)
        for step, batch in enumerate(progress, 1):
            optimizer.zero_grad(set_to_none=True)
            mean, log_std = model(
                batch["context_x"].to(device, non_blocking=True),
                batch["context_y"].to(device, non_blocking=True),
                batch["target_x"].to(device, non_blocking=True),
                batch["context_mask"].to(device, non_blocking=True),
            )
            log_std = add_variance_floor(log_std, args.minimum_normalized_std)
            loss = normal_nll(mean, log_std, batch["target_y"].to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            progress.set_postfix(train_nll=f"{total / step:.4f}", stations=int(batch["context_count"].float().mean()))
        train_nll = total / len(train_loader)
        validation_nll = validate(model, validation_loader, device, args.minimum_normalized_std)
        elapsed = time.perf_counter() - epoch_start
        improved = validation_nll < best_value
        if improved:
            best_value, best_epoch, stale = validation_nll, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "model_type": args.model,
                "architecture": "full_context_precipitation",
                "variable": "daily_total_precipitation",
                "target_units": "mm day-1",
                "target_transform": "log1p",
                "experiment": args.experiment,
                "epoch": epoch,
                "validation_nll_log1p": validation_nll,
                "normalization": norm.to_dict(),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            }, best_path)
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "train_nll_log1p": train_nll,
            "validation_nll_log1p": validation_nll,
            "seconds": elapsed,
            "best": improved,
        })
        print(
            f"epoch={epoch:03d} train_nll_log1p={train_nll:.5f} "
            f"val_nll_log1p={validation_nll:.5f} best_epoch={best_epoch} "
            f"stale={stale}/{args.patience} elapsed={elapsed:.1f}s"
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early_stop epoch={epoch}; best_epoch={best_epoch}; best_val_nll_log1p={best_value:.5f}")
            break
    torch.save({
        "model": model.state_dict(),
        "model_type": args.model,
        "architecture": "full_context_precipitation",
        "variable": "daily_total_precipitation",
        "target_units": "mm day-1",
        "target_transform": "log1p",
        "experiment": args.experiment,
        "epoch": history[-1]["epoch"],
        "normalization": norm.to_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }, output / "last.pt")
    (output / "history.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_validation_nll_log1p": best_value,
        "total_seconds": time.perf_counter() - started,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"saved {best_path}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()

