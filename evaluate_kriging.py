"""Validation-tuned ordinary kriging baseline on the shared split/sparse protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from evaluate_experiment import conformal_quantile, extreme_mask, metrics
from experiment_data import ReconstructionDataset, fit_normalization, read_stations


def coordinates_km(coords: np.ndarray) -> np.ndarray:
    mean_latitude = np.deg2rad(coords[:, 1].mean())
    return np.column_stack((coords[:, 0] * 111.32 * np.cos(mean_latitude), coords[:, 1] * 110.57))


def pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))


def ordinary_kriging(context_x: np.ndarray, context_y: np.ndarray, target_x: np.ndarray, lengthscale_km: float, nugget_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    x = coordinates_km(context_x); q = coordinates_km(target_x)
    sill = max(float(np.var(context_y, ddof=1)), 1e-4)
    covariance = sill * np.exp(-pairwise_distance(x, x) / lengthscale_km)
    covariance.flat[:: len(x) + 1] += sill * nugget_fraction + 1e-6
    target_covariance = sill * np.exp(-pairwise_distance(x, q) / lengthscale_km)
    system = np.empty((len(x) + 1, len(x) + 1), dtype="float64")
    system[:-1, :-1] = covariance; system[:-1, -1] = 1.0; system[-1, :-1] = 1.0; system[-1, -1] = 0.0
    rhs = np.vstack((target_covariance, np.ones((1, len(q)))))
    solution = np.linalg.solve(system, rhs)
    weights, multiplier = solution[:-1], solution[-1]
    mean = context_y @ weights
    variance = sill * (1.0 + nugget_fraction) - np.sum(weights * target_covariance, axis=0) - multiplier
    return mean.astype("float32"), np.sqrt(np.maximum(variance, 1e-6)).astype("float32")


def recover_coords(normalized: np.ndarray, data: ReconstructionDataset) -> np.ndarray:
    return ((normalized + 1) * 0.5 * (data.norm.coord_max - data.norm.coord_min) + data.norm.coord_min).astype("float32")


def predict(data: ReconstructionDataset, lengthscale_km: float, nugget_fraction: float, indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    output = {"mean": [], "std": [], "target": [], "date_ns": []}
    selected = np.arange(len(data)) if indices is None else indices
    for index in tqdm(selected, desc="kriging", unit="day", dynamic_ncols=True):
        sample = data[int(index)]
        context_x = recover_coords(sample["context_x"].numpy(), data)
        target_x = recover_coords(sample["target_x"].numpy(), data)
        context_y = sample["context_y"].numpy() * data.norm.value_std + data.norm.value_mean
        target_y = sample["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        mean, std = ordinary_kriging(context_x, context_y, target_x, lengthscale_km, nugget_fraction)
        output["mean"].append(mean); output["std"].append(std); output["target"].append(target_y)
        output["date_ns"].append(np.repeat(sample["date_ns"].item(), len(target_y)))
    return {key: np.concatenate(values) for key, values in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("global", "source"), required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--calibration-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--test-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--calibration-label", required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25, 0.10))
    parser.add_argument("--lengthscale-candidates-km", type=float, nargs="+", default=(200, 400, 800, 1600))
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--nugget-fraction", type=float, default=0.05)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = read_stations(args.stations, "tg", ("train", "validation", "calibration", "test"))
    train_region = "global" if args.experiment == "global" else "source"
    norm = fit_normalization(frame, args.era5, "tg", train_region, 15.0)
    common = dict(frame=frame, era5_path=args.era5, station_variable="tg", era5_variable="t2m", era5_offset=-273.15, normalization=norm, max_context=args.max_context, targets_per_day=args.targets_per_day, seed=args.seed, training=False, cutoff=15.0)
    validation = ReconstructionDataset(split="validation", context_region=train_region, target_region=train_region, retention=1.0, **common)
    validation_indices = np.linspace(0, len(validation) - 1, min(args.validation_days, len(validation)), dtype=int)
    tuning = {}
    for candidate in args.lengthscale_candidates_km:
        prediction = predict(validation, candidate, args.nugget_fraction, validation_indices)
        tuning[str(candidate)] = float(np.sqrt(np.mean((prediction["target"] - prediction["mean"]) ** 2)))
    lengthscale = min(args.lengthscale_candidates_km, key=lambda value: tuning[str(value)])
    calibration_data = ReconstructionDataset(split="calibration", context_region=args.calibration_region, target_region=args.calibration_region, retention=1.0, **common)
    calibration = predict(calibration_data, lengthscale, args.nugget_fraction)
    conformal_q = conformal_quantile(np.abs(calibration["target"] - calibration["mean"]) / calibration["std"].clip(1e-6), 0.90)
    training_targets = ReconstructionDataset(split="train", context_region=train_region, target_region=train_region, retention=1.0, **common)
    daily_means = training_targets.daily_target_means()
    low, high = float(np.quantile(daily_means, 0.01)), float(np.quantile(daily_means, 0.99))
    rows = []
    for retention in args.retentions:
        test = ReconstructionDataset(split="test", context_region=args.test_region, target_region=args.test_region, retention=retention, **common)
        prediction = predict(test, lengthscale, args.nugget_fraction)
        all_mask = np.ones(len(prediction["target"]), dtype=bool)
        extremes = extreme_mask(prediction, low, high)
        rows.append(metrics(prediction, conformal_q, all_mask, "all", retention))
        if extremes.any():
            rows.append(metrics(prediction, conformal_q, extremes, "extreme_days", retention))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {"model_type": "ordinary_kriging", "experiment": args.experiment, "calibration_region": args.calibration_region, "test_region": args.test_region, "calibration_label": args.calibration_label, "lengthscale_km": lengthscale, "validation_rmse_by_lengthscale": tuning, "nugget_fraction": args.nugget_fraction, "conformal_q": conformal_q, "results_csv": str(args.output)}
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"wrote {args.output}; lengthscale={lengthscale}km; conformal_q={conformal_q:.5f}")


if __name__ == "__main__":
    main()
