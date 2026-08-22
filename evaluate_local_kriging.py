"""Full-network local-neighbor ordinary kriging under the final evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from evaluation_protocol import (
    build_extreme_definition,
    conformal_quantile,
    day_cluster_conformal_quantile,
    metric_summary,
    normalized_scores,
    subset_mask,
)
from full_context_data import FullContextDataset, fit_full_normalization, read_domain_stations


REFERENCE_LATITUDE_DEG = 53.0
EARTH_RADIUS_KM = 6371.0088


def coordinates_km(coords: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(coords, dtype="float64"))
    return np.column_stack((
        EARTH_RADIUS_KM * np.cos(np.deg2rad(REFERENCE_LATITUDE_DEG)) * radians[:, 0],
        EARTH_RADIUS_KM * radians[:, 1],
    ))


def recover_coords(normalized: np.ndarray, data: FullContextDataset) -> np.ndarray:
    return ((normalized + 1.0) * 0.5 * (data.norm.coord_max - data.norm.coord_min) + data.norm.coord_min).astype("float64")


def local_ordinary_kriging(
    context_coords: np.ndarray,
    context_values: np.ndarray,
    target_coords: np.ndarray,
    neighbors: int,
    lengthscale_km: float,
    nugget_fraction: float,
    target_batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    context_xy = coordinates_km(context_coords)
    target_xy = coordinates_km(target_coords)
    k = min(int(neighbors), len(context_xy))
    if k < 2:
        raise ValueError("Kriging requires at least two context stations.")
    tree = cKDTree(context_xy)
    target_distance, neighbor_index = tree.query(target_xy, k=k, workers=1)
    if k == 1:
        target_distance = target_distance[:, None]
        neighbor_index = neighbor_index[:, None]
    sill = max(float(np.var(context_values, ddof=1)), 1e-4)
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for start in range(0, len(target_xy), target_batch_size):
        stop = min(start + target_batch_size, len(target_xy))
        index = neighbor_index[start:stop]
        points = context_xy[index]
        pairwise = np.sqrt(np.maximum(np.sum((points[:, :, None] - points[:, None, :]) ** 2, axis=-1), 0.0))
        covariance = sill * np.exp(-pairwise / lengthscale_km)
        diagonal = np.arange(k)
        covariance[:, diagonal, diagonal] += sill * nugget_fraction + 1e-6
        count = stop - start
        system = np.empty((count, k + 1, k + 1), dtype="float64")
        system[:, :k, :k] = covariance
        system[:, :k, k] = 1.0
        system[:, k, :k] = 1.0
        system[:, k, k] = 0.0
        target_covariance = sill * np.exp(-target_distance[start:stop] / lengthscale_km)
        rhs = np.empty((count, k + 1), dtype="float64")
        rhs[:, :k] = target_covariance
        rhs[:, k] = 1.0
        solution = np.linalg.solve(system, rhs[..., None])[..., 0]
        weights = solution[:, :k]
        multiplier = solution[:, k]
        local_values = context_values[index]
        means.append(np.sum(weights * local_values, axis=1))
        variance = sill * (1.0 + nugget_fraction) - np.sum(weights * target_covariance, axis=1) - multiplier
        stds.append(np.sqrt(np.maximum(variance, 1e-6)))
    return np.concatenate(means).astype("float32"), np.concatenate(stds).astype("float32")


def selected_indices(length: int, max_days: int | None) -> np.ndarray:
    if max_days is None or max_days >= length:
        return np.arange(length)
    return np.linspace(0, length - 1, max_days, dtype=int)


def predict(
    data: FullContextDataset,
    neighbors: int,
    lengthscale_km: float,
    nugget_fraction: float,
    max_days: int | None = None,
) -> dict[str, np.ndarray]:
    output = {key: [] for key in ("mean", "std", "target", "date_ns", "context_count", "context_date_ns")}
    for index in tqdm(selected_indices(len(data), max_days), desc="local kriging", unit="day", dynamic_ncols=True):
        sample = data[int(index)]
        context_coords = recover_coords(sample["context_x"].numpy(), data)
        target_coords = recover_coords(sample["target_x"].numpy(), data)
        context_values = sample["context_y"].numpy() * data.norm.value_std + data.norm.value_mean
        targets = sample["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        mean, std = local_ordinary_kriging(
            context_coords,
            context_values,
            target_coords,
            neighbors,
            lengthscale_km,
            nugget_fraction,
        )
        date = int(sample["date_ns"].item())
        output["mean"].append(mean)
        output["std"].append(std)
        output["target"].append(targets)
        output["date_ns"].append(np.repeat(date, len(targets)))
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
    parser.add_argument("--lengthscale-candidates-km", type=float, nargs="+", default=(200.0, 400.0, 800.0))
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--nugget-fraction", type=float, default=0.05)
    parser.add_argument("--targets-per-day", type=int, default=1024)
    parser.add_argument("--evaluation-seed", type=int, default=9102)
    parser.add_argument("--calibration-seed", type=int, default=9101)
    parser.add_argument("--tuning-seed", type=int, default=9100)
    parser.add_argument("--extreme-tail-probability", type=float, default=0.05)
    parser.add_argument("--climatology-half-window-days", type=int, default=15)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-calibration-days", type=int)
    parser.add_argument("--max-test-days", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_region = "global" if args.experiment == "global" else "source"
    frame = read_domain_stations(args.stations, "tg", ("train", "validation", "calibration", "test"), args.era5)
    norm = fit_full_normalization(frame, args.era5, "tg", train_region, 15.0)
    common = dict(
        frame=frame,
        era5_path=args.era5,
        station_variable="tg",
        era5_variable="t2m",
        era5_offset=-273.15,
        normalization=norm,
        targets_per_day=args.targets_per_day,
        cutoff=15.0,
        training=False,
    )
    validation = FullContextDataset(
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
            daily_rmse = []
            for date in np.unique(prediction["date_ns"]):
                selected = prediction["date_ns"] == date
                daily_rmse.append(float(np.sqrt(np.mean((prediction["target"][selected] - prediction["mean"][selected]) ** 2))))
            tuning[f"k={neighbors},lengthscale_km={lengthscale:g}"] = float(np.mean(daily_rmse))
    best_key = min(tuning, key=tuning.get)
    parts = dict(item.split("=") for item in best_key.split(","))
    neighbors = int(parts["k"])
    lengthscale = float(parts["lengthscale_km"])
    print(f"selected {best_key}; validation day-mean RMSE={tuning[best_key]:.4f} C")

    calibration_predictions: dict[float, dict[str, np.ndarray]] = {}
    conformal_q: dict[float, float] = {}
    cluster_conformal_q: dict[float, float] = {}
    for retention in sorted(set(args.retentions)):
        data = FullContextDataset(
            split="calibration",
            context_region=args.calibration_region,
            target_region=args.calibration_region,
            retention=retention,
            seed=args.calibration_seed,
            **common,
        )
        prediction = predict(data, neighbors, lengthscale, args.nugget_fraction, args.max_calibration_days)
        calibration_predictions[retention] = prediction
        conformal_q[retention] = conformal_quantile(normalized_scores(prediction), 0.90)
        cluster_conformal_q[retention] = day_cluster_conformal_quantile(prediction, 0.90, 0.90)

    extreme = build_extreme_definition(
        args.era5,
        "t2m",
        -273.15,
        args.test_region,
        15.0,
        args.extreme_tail_probability,
        args.climatology_half_window_days,
    )
    rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    full_q = conformal_q[1.0]
    full_cluster_q = cluster_conformal_q[1.0]
    for retention_index, retention in enumerate(args.retentions):
        test = FullContextDataset(
            split="test",
            context_region=args.test_region,
            target_region=args.test_region,
            retention=retention,
            seed=args.evaluation_seed,
            **common,
        )
        prediction = predict(test, neighbors, lengthscale, args.nugget_fraction, args.max_test_days)
        modes = [("deployed_full_density", full_q, full_cluster_q)]
        if retention != 1.0:
            modes.append(("matched_density", conformal_q[retention], cluster_conformal_q[retention]))
        for mode_index, (mode, q, cluster_q) in enumerate(modes):
            for subset_index, subset in enumerate(("all", "extreme_days")):
                mask = subset_mask(prediction, subset, extreme)
                if not mask.any():
                    print(f"warning: no {subset} samples at retention={retention}")
                    continue
                seed = args.evaluation_seed + 1000 * retention_index + 100 * mode_index + subset_index
                row, per_day = metric_summary(
                    prediction,
                    q,
                    cluster_q,
                    mask,
                    subset,
                    retention,
                    mode,
                    seed,
                    args.bootstrap_repetitions,
                )
                rows.append(row)
                daily_rows.extend(per_day)

    write_csv(args.output, rows)
    daily_path = args.output.with_name(f"{args.output.stem}.daily.csv")
    write_csv(daily_path, daily_rows)
    metadata = {
        "model_type": "ordinary_kriging",
        "architecture": "local_neighbor_ordinary_kriging",
        "training_experiment": args.experiment,
        "training_seed": None,
        "calibration_region": args.calibration_region,
        "test_region": args.test_region,
        "calibration_label": args.calibration_label,
        "neighbor_count": neighbors,
        "lengthscale_km": lengthscale,
        "nugget_fraction": args.nugget_fraction,
        "validation_day_mean_rmse_by_candidate": tuning,
        "conformal_q_by_retention": {str(key): value for key, value in conformal_q.items()},
        "day_cluster_conformal_q_by_retention": {str(key): value for key, value in cluster_conformal_q.items()},
        "calibration_scores_per_retention": {
            str(key): int(len(value["target"])) for key, value in calibration_predictions.items()
        },
        "calibration_unit": "sampled ERA5 target point; reported uncertainty intervals are bootstrap intervals across days",
        "evaluation_seed": args.evaluation_seed,
        "calibration_seed": args.calibration_seed,
        "targets_per_day": args.targets_per_day,
        "extreme_event": extreme.metadata(),
        "results_csv": str(args.output),
        "daily_results_csv": str(daily_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows) and {daily_path} ({len(daily_rows)} rows)")


if __name__ == "__main__":
    main()

