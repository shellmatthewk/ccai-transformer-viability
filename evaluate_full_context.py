"""Full-network and nested sparse evaluation for the redesigned architectures."""

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

from experiment_data import Normalization
from full_context_data import FullContextDataset, padded_collate, read_domain_stations
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer


def make_model(name: str, latent_side: int) -> torch.nn.Module:
    return ConvCNP() if name == "convcnp" else LatentGridTransformer(latent_side=latent_side)


@torch.inference_mode()
def predict(model: torch.nn.Module, data: FullContextDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=padded_collate)
    output = {key: [] for key in ("mean", "std", "target", "date_ns", "context_count")}; model.eval()
    for batch in tqdm(loader, desc="predict", unit="day", dynamic_ncols=True):
        mean, log_std = model(batch["context_x"].to(device, non_blocking=True), batch["context_y"].to(device, non_blocking=True), batch["target_x"].to(device, non_blocking=True), batch["context_mask"].to(device, non_blocking=True))
        mean = mean.cpu().numpy() * data.norm.value_std + data.norm.value_mean
        std = log_std.exp().cpu().numpy() * data.norm.value_std
        target = batch["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        output["mean"].append(mean.reshape(-1)); output["std"].append(std.reshape(-1)); output["target"].append(target.reshape(-1))
        output["date_ns"].append(np.repeat(batch["date_ns"].numpy(), mean.shape[1]))
        output["context_count"].append(batch["context_count"].numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def conformal_quantile(scores: np.ndarray, coverage: float = .9) -> float:
    scores = np.sort(scores[np.isfinite(scores)]); rank = min(len(scores) - 1, math.ceil((len(scores) + 1) * coverage) - 1)
    return float(scores[rank])


def extreme_days(prediction: dict[str, np.ndarray], low: float, high: float) -> np.ndarray:
    dates = prediction["date_ns"]; selected = []
    for date in np.unique(dates):
        value = prediction["target"][dates == date].mean()
        if value <= low or value >= high: selected.append(date)
    return np.isin(dates, selected)


def metric_row(prediction: dict[str, np.ndarray], q: float, mask: np.ndarray, subset: str, retention: float) -> dict[str, object]:
    mean, std, target = prediction["mean"][mask], prediction["std"][mask].clip(1e-6), prediction["target"][mask]
    error = target - mean; raw_q = 1.6448536269514722
    raw_coverage = (np.abs(error) <= raw_q * std).mean(); coverage = (np.abs(error) <= q * std).mean()
    return {"subset": subset, "retention": retention, "context_count_min": int(prediction["context_count"].min()), "context_count_median": int(np.median(prediction["context_count"])), "context_count_max": int(prediction["context_count"].max()), "n": int(mask.sum()), "rmse_c": float(np.sqrt(np.mean(error ** 2))), "mae_c": float(np.mean(np.abs(error))), "gaussian_nll": float(np.mean(.5 * (error / std) ** 2 + np.log(std) + .5 * np.log(2 * np.pi))), "raw_90_coverage": float(raw_coverage), "raw_mean_width_c": float(np.mean(2 * raw_q * std)), "conformal_90_coverage": float(coverage), "conformal_coverage_gap": float(coverage - .9), "conformal_mean_width_c": float(np.mean(2 * q * std))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--stations", type=Path, required=True); parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--calibration-region", choices=("global", "source", "target"), required=True); parser.add_argument("--test-region", choices=("global", "source", "target"), required=True); parser.add_argument("--calibration-label", required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, .75, .5, .25, .1)); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto"); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "full_context": raise ValueError("Checkpoint is not a full-context architecture.")
    config = checkpoint["config"]; norm = Normalization.from_dict(checkpoint["normalization"])
    model = make_model(checkpoint["model_type"], int(config["latent_side"])); model.load_state_dict(checkpoint["model"]); model.to(device)
    frame = read_domain_stations(args.stations, "tg", ("calibration", "test"), args.era5)
    common = dict(frame=frame, era5_path=args.era5, station_variable="tg", era5_variable="t2m", era5_offset=-273.15, normalization=norm, targets_per_day=int(config["targets_per_day"]), cutoff=float(config["longitude_cutoff"]), training=False)
    calibration_data = FullContextDataset(split="calibration", context_region=args.calibration_region, target_region=args.calibration_region, retention=1.0, seed=int(config["seed"]) + 10, **common)
    calibration = predict(model, calibration_data, device, args.batch_size)
    q = conformal_quantile(np.abs(calibration["target"] - calibration["mean"]) / calibration["std"].clip(1e-6))
    threshold = checkpoint["extreme_thresholds"]; rows = []
    for retention in args.retentions:
        test_data = FullContextDataset(split="test", context_region=args.test_region, target_region=args.test_region, retention=retention, seed=int(config["seed"]) + 20, **common)
        prediction = predict(model, test_data, device, args.batch_size); all_mask = np.ones(len(prediction["target"]), dtype=bool)
        extreme = extreme_days(prediction, float(threshold["low"]), float(threshold["high"]))
        rows.append(metric_row(prediction, q, all_mask, "all", retention))
        if extreme.any(): rows.append(metric_row(prediction, q, extreme, "extreme_days", retention))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {"model_type": checkpoint["model_type"], "architecture": "full_context", "experiment": checkpoint["experiment"], "calibration_region": args.calibration_region, "test_region": args.test_region, "calibration_label": args.calibration_label, "conformal_q": q, "calibration_scores": len(calibration["target"]), "results_csv": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output}; conformal_q={q:.5f}; rows={len(rows)}")


if __name__ == "__main__": main()
