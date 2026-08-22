"""Evaluate reconstruction, sparse networks, extremes, and split conformal intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from experiment_data import Normalization, ReconstructionDataset, read_stations
from models import ConvCNP, StationTransformer


def make_model(name: str) -> torch.nn.Module:
    return ConvCNP() if name == "convcnp" else StationTransformer()


@torch.inference_mode()
def predict(model: torch.nn.Module, data: ReconstructionDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    output = {"mean": [], "std": [], "target": [], "date_ns": []}
    model.eval()
    for batch in tqdm(loader, desc="predict", unit="batch", dynamic_ncols=True):
        mean, log_std = model(batch["context_x"].to(device, non_blocking=True), batch["context_y"].to(device, non_blocking=True), batch["target_x"].to(device, non_blocking=True))
        mean = mean.cpu().numpy() * data.norm.value_std + data.norm.value_mean
        std = log_std.exp().cpu().numpy() * data.norm.value_std
        target = batch["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        output["mean"].append(mean.reshape(-1)); output["std"].append(std.reshape(-1)); output["target"].append(target.reshape(-1))
        output["date_ns"].append(np.repeat(batch["date_ns"].numpy(), mean.shape[1]))
    return {key: np.concatenate(values) for key, values in output.items()}


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    finite = np.sort(scores[np.isfinite(scores)])
    rank = min(len(finite) - 1, math.ceil((len(finite) + 1) * coverage) - 1)
    return float(finite[rank])


def extreme_mask(predictions: dict[str, np.ndarray], low: float, high: float) -> np.ndarray:
    dates = predictions["date_ns"]
    extreme_dates = []
    for date in np.unique(dates):
        daily_mean = predictions["target"][dates == date].mean()
        if daily_mean <= low or daily_mean >= high:
            extreme_dates.append(date)
    return np.isin(dates, extreme_dates)


def metrics(predictions: dict[str, np.ndarray], conformal_q: float, mask: np.ndarray, label: str, retention: float) -> dict[str, object]:
    mean, std, target = predictions["mean"][mask], predictions["std"][mask].clip(1e-6), predictions["target"][mask]
    error = target - mean
    raw_q = 1.6448536269514722
    raw_covered = np.abs(error) <= raw_q * std
    conformal_covered = np.abs(error) <= conformal_q * std
    return {
        "subset": label,
        "retention": retention,
        "context_count": max(1, round(256 * retention)),
        "n": int(mask.sum()),
        "rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "mae_c": float(np.mean(np.abs(error))),
        "gaussian_nll": float(np.mean(0.5 * (error / std) ** 2 + np.log(std) + 0.5 * np.log(2 * np.pi))),
        "raw_90_coverage": float(raw_covered.mean()),
        "raw_mean_width_c": float(np.mean(2 * raw_q * std)),
        "conformal_90_coverage": float(conformal_covered.mean()),
        "conformal_coverage_gap": float(conformal_covered.mean() - 0.90),
        "conformal_mean_width_c": float(np.mean(2 * conformal_q * std)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--calibration-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--test-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--calibration-label", required=True, help="global, transferred, or oracle_target")
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25, 0.10))
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    normalization = Normalization.from_dict(checkpoint["normalization"])
    model = make_model(checkpoint["model_type"]); model.load_state_dict(checkpoint["model"]); model.to(device)
    frame = read_stations(args.stations, config["station_variable"], ("calibration", "test"))
    common = dict(frame=frame, era5_path=args.era5, station_variable=config["station_variable"], era5_variable=config["era5_variable"], era5_offset=float(config["era5_offset"]), normalization=normalization, max_context=int(config["max_context"]), targets_per_day=int(config["targets_per_day"]), cutoff=float(config["longitude_cutoff"]), training=False)
    calibration_data = ReconstructionDataset(split="calibration", context_region=args.calibration_region, target_region=args.calibration_region, retention=1.0, seed=int(config["seed"]) + 10, **common)
    calibration = predict(model, calibration_data, device, args.batch_size)
    scores = np.abs(calibration["target"] - calibration["mean"]) / calibration["std"].clip(1e-6)
    conformal_q = conformal_quantile(scores, args.coverage)
    thresholds = checkpoint["extreme_thresholds"]
    rows = []
    for retention in args.retentions:
        test_data = ReconstructionDataset(split="test", context_region=args.test_region, target_region=args.test_region, retention=retention, seed=int(config["seed"]) + 20, **common)
        predictions = predict(model, test_data, device, args.batch_size)
        all_mask = np.ones(len(predictions["target"]), dtype=bool)
        sparse_extreme = extreme_mask(predictions, float(thresholds["low"]), float(thresholds["high"]))
        rows.append(metrics(predictions, conformal_q, all_mask, "all", retention))
        if sparse_extreme.any():
            rows.append(metrics(predictions, conformal_q, sparse_extreme, "extreme_days", retention))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "model_type": checkpoint["model_type"],
        "experiment": checkpoint["experiment"],
        "calibration_region": args.calibration_region,
        "test_region": args.test_region,
        "calibration_label": args.calibration_label,
        "coverage": args.coverage,
        "conformal_q": conformal_q,
        "calibration_scores": len(scores),
        "extreme_definition": thresholds,
        "results_csv": str(args.output),
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"wrote {args.output}; conformal_q={conformal_q:.5f}; rows={len(rows)}")


if __name__ == "__main__":
    main()
