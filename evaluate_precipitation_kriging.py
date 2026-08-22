"""Validation-tuned full-network local ordinary kriging for daily precipitation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from evaluate_local_kriging import local_ordinary_kriging
from precipitation_data import PrecipitationDataset, fit_precipitation_normalization, read_precipitation_stations
from precipitation_protocol import (
    build_heavy_precipitation_definition,
    conformal_quantile,
    day_cluster_conformal_quantile,
    metric_summary,
    normalized_log_scores,
    predictive_mean_mm,
    subset_mask,
)


def recover_coords(normalized: np.ndarray, data: PrecipitationDataset) -> np.ndarray:
    return ((normalized + 1.0) * 0.5 * (data.norm.coord_max - data.norm.coord_min) + data.norm.coord_min).astype("float64")


def selected_indices(length: int, max_days: int | None) -> np.ndarray:
    if max_days is None or max_days >= length:
        return np.arange(length)
    return np.linspace(0, length - 1, max_days, dtype=int)


def predict(
    data: PrecipitationDataset,
    neighbors: int,
    lengthscale_km: float,
    nugget_fraction: float,
    max_days: int | None = None,
) -> dict[str, np.ndarray]:
    output = {key: [] for key in ("mean_log", "std_log", "target_mm", "date_ns", "context_count", "context_date_ns")}
    for index in tqdm(selected_indices(len(data), max_days), desc="precipitation kriging", unit="day", dynamic_ncols=True):
        sample = data[int(index)]
        context_coords = recover_coords(sample["context_x"].numpy(), data)
        target_coords = recover_coords(sample["target_x"].numpy(), data)
        context_log = sample["context_y"].numpy() * data.norm.value_std + data.norm.value_mean
        target_log = sample["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        mean_log, std_log = local_ordinary_kriging(
            context_coords,
            context_log,
            target_coords,
            neighbors,
            lengthscale_km,
            nugget_fraction,
        )
        date = int(sample["date_ns"].item())
        output["mean_log"].append(mean_log)
        output["std_log"].append(std_log)
        output["target_mm"].append(np.maximum(np.expm1(target_log), 0.0))
        output["date_ns"].append(np.repeat(date, len(target_log)))
        output["context_count"].append(np.asarray([int(sample["context_count"].item())]))
        output["context_date_ns"].append(np.asarray([date], dtype="int64"))
    return {key: np.concatenate(values) for key, values in output.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("global", "source"), required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--calibration-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--test-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--calibration-label", required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25, 0.1))
    parser.add_argument("--neighbor-candidates", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--lengthscale-candidates-km", type=float, nargs="+", default=(100.0, 200.0, 400.0, 800.0))
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--nugget-fraction", type=float, default=0.05)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--evaluation-seed", type=int, default=19102)
    parser.add_argument("--calibration-seed", type=int, default=19101)
    parser.add_argument("--tuning-seed", type=int, default=19100)
    parser.add_argument("--heavy-tail-probability", type=float, default=0.05)
    parser.add_argument("--climatology-half-window-days", type=int, default=15)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-calibration-days", type=int)
    parser.add_argument("--max-test-days", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_region = "global" if args.experiment == "global" else "source"
    frame = read_precipitation_stations(args.stations, ("train", "validation", "calibration", "test"), args.era5)
    norm = fit_precipitation_normalization(frame, args.era5, train_region, 15.0)
    common = dict(
        frame=frame,
        era5_path=args.era5,
        normalization=norm,
        targets_per_day=args.targets_per_day,
        cutoff=15.0,
        training=False,
    )
    validation = PrecipitationDataset(
        split="validation",
        context_region=train_region,
        target_region=train_region,
        retention=1.0,
        seed=args.tuning_seed,
        **common,
    )
    tuning: dict[str, float] = {}
    for neighbors in args.neighbor_candidates:
        for lengthscale in args.lengthscale_candidates_km:
            prediction = predict(validation, neighbors, lengthscale, args.nugget_fraction, args.validation_days)
            estimate = predictive_mean_mm(prediction["mean_log"], prediction["std_log"])
            daily_rmse = []
            for date in np.unique(prediction["date_ns"]):
                selected = prediction["date_ns"] == date
                daily_rmse.append(float(np.sqrt(np.mean((prediction["target_mm"][selected] - estimate[selected]) ** 2))))
            tuning[f"k={neighbors},lengthscale_km={lengthscale:g}"] = float(np.mean(daily_rmse))
    best_key = min(tuning, key=tuning.get)
    parts = dict(item.split("=") for item in best_key.split(","))
    neighbors = int(parts["k"])
    lengthscale = float(parts["lengthscale_km"])
    print(f"selected {best_key}; validation day-mean RMSE={tuning[best_key]:.4f} mm/day")

    calibration_predictions: dict[float, dict[str, np.ndarray]] = {}
    point_q: dict[float, float] = {}
    cluster_q: dict[float, float] = {}
    for retention in sorted(set(args.retentions)):
        data = PrecipitationDataset(
            split="calibration",
            context_region=args.calibration_region,
            target_region=args.calibration_region,
            retention=retention,
            seed=args.calibration_seed,
            **common,
        )
        prediction = predict(data, neighbors, lengthscale, args.nugget_fraction, args.max_calibration_days)
        calibration_predictions[retention] = prediction
        point_q[retention] = conformal_quantile(normalized_log_scores(prediction), 0.90)
        cluster_q[retention] = day_cluster_conformal_quantile(prediction, 0.90, 0.90)

    heavy = build_heavy_precipitation_definition(
        args.era5,
        "tp",
        args.test_region,
        15.0,
        args.heavy_tail_probability,
        args.climatology_half_window_days,
    )
    rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for retention_index, retention in enumerate(args.retentions):
        data = PrecipitationDataset(
            split="test",
            context_region=args.test_region,
            target_region=args.test_region,
            retention=retention,
            seed=args.evaluation_seed,
            **common,
        )
        prediction = predict(data, neighbors, lengthscale, args.nugget_fraction, args.max_test_days)
        modes = [("deployed_full_density", point_q[1.0], cluster_q[1.0])]
        if retention != 1.0:
            modes.append(("matched_density", point_q[retention], cluster_q[retention]))
        for mode_index, (mode, q, q_cluster) in enumerate(modes):
            for subset_index, subset in enumerate(("all", "heavy_precipitation_days")):
                mask = subset_mask(prediction, subset, heavy)
                if not mask.any():
                    print(f"warning: no {subset} samples at retention={retention}")
                    continue
                seed = args.evaluation_seed + 1000 * retention_index + 100 * mode_index + subset_index
                row, daily = metric_summary(
                    prediction,
                    q,
                    q_cluster,
                    mask,
                    subset,
                    retention,
                    mode,
                    seed,
                    args.bootstrap_repetitions,
                )
                rows.append(row)
                daily_rows.extend(daily)

    write_csv(args.output, rows)
    daily_path = args.output.with_name(f"{args.output.stem}.daily.csv")
    write_csv(daily_path, daily_rows)
    metadata = {
        "model_type": "ordinary_kriging",
        "architecture": "local_neighbor_ordinary_kriging_log1p_precipitation",
        "variable": "daily_total_precipitation",
        "target_units": "mm day-1",
        "target_transform": "log1p",
        "training_experiment": args.experiment,
        "training_seed": None,
        "calibration_region": args.calibration_region,
        "test_region": args.test_region,
        "calibration_label": args.calibration_label,
        "neighbor_count": neighbors,
        "lengthscale_km": lengthscale,
        "nugget_fraction": args.nugget_fraction,
        "validation_day_mean_rmse_by_candidate": tuning,
        "conformal_q_log1p_by_retention": {str(key): value for key, value in point_q.items()},
        "day_cluster_conformal_q_log1p_by_retention": {str(key): value for key, value in cluster_q.items()},
        "calibration_scores_per_retention": {
            str(key): int(len(value["target_mm"])) for key, value in calibration_predictions.items()
        },
        "evaluation_seed": args.evaluation_seed,
        "calibration_seed": args.calibration_seed,
        "targets_per_day": args.targets_per_day,
        "heavy_precipitation_event": heavy.metadata(),
        "results_csv": str(args.output),
        "daily_results_csv": str(daily_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows) and {daily_path} ({len(daily_rows)} rows)")


if __name__ == "__main__":
    main()

