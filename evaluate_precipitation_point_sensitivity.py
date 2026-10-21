"""Neural precipitation point-estimate sensitivity evaluation; original evaluator is unchanged."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from experiment_data import Normalization
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer
from precipitation_data import PrecipitationDataset, padded_collate, read_precipitation_stations
from precipitation_protocol_point_sensitivity import (
    build_heavy_precipitation_definition,
    conformal_quantile,
    day_cluster_conformal_quantile,
    metric_summary,
    normalized_log_scores,
    subset_mask,
)


def make_model(name: str, latent_side: int) -> torch.nn.Module:
    if name == "convcnp":
        return ConvCNP()
    if name == "latent_transformer":
        return LatentGridTransformer(latent_side=latent_side)
    raise ValueError(f"Unknown model {name!r}")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    data: PrecipitationDataset,
    device: torch.device,
    batch_size: int,
    minimum_normalized_std: float,
    max_days: int | None = None,
) -> dict[str, np.ndarray]:
    dataset = data
    if max_days is not None and max_days < len(data):
        dataset = Subset(data, np.linspace(0, len(data) - 1, max_days, dtype=int).tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda", collate_fn=padded_collate)
    output = {key: [] for key in ("mean_log", "std_log", "target_mm", "date_ns", "context_count", "context_date_ns")}
    model.eval()
    for batch in tqdm(loader, desc="predict precipitation", unit="day", dynamic_ncols=True):
        mean, log_std = model(
            batch["context_x"].to(device, non_blocking=True),
            batch["context_y"].to(device, non_blocking=True),
            batch["target_x"].to(device, non_blocking=True),
            batch["context_mask"].to(device, non_blocking=True),
        )
        mean_log = mean.cpu().numpy() * data.norm.value_std + data.norm.value_mean
        normalized_std = torch.sqrt(torch.exp(2.0 * log_std) + minimum_normalized_std**2)
        std_log = normalized_std.cpu().numpy() * data.norm.value_std
        target_log = batch["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        dates = batch["date_ns"].numpy()
        output["mean_log"].append(mean_log.reshape(-1))
        output["std_log"].append(std_log.reshape(-1))
        output["target_mm"].append(np.maximum(np.expm1(target_log.reshape(-1)), 0.0))
        output["date_ns"].append(np.repeat(dates, mean_log.shape[1]))
        output["context_count"].append(batch["context_count"].numpy())
        output["context_date_ns"].append(dates)
    return {key: np.concatenate(values) for key, values in output.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--era5", type=Path, required=True)
    parser.add_argument("--calibration-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--test-region", choices=("global", "source", "target"), required=True)
    parser.add_argument("--calibration-label", required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25, 0.1))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--evaluation-seed", type=int, default=19102)
    parser.add_argument("--calibration-seed", type=int, default=19101)
    parser.add_argument("--targets-per-day", type=int)
    parser.add_argument("--heavy-tail-probability", type=float, default=0.05)
    parser.add_argument("--climatology-half-window-days", type=int, default=15)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-calibration-days", type=int)
    parser.add_argument("--max-test-days", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "full_context_precipitation":
        raise ValueError("Checkpoint is not a full-context precipitation model.")
    config = checkpoint["config"]
    norm = Normalization.from_dict(checkpoint["normalization"])
    model = make_model(checkpoint["model_type"], int(config["latent_side"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    targets_per_day = args.targets_per_day or int(config["targets_per_day"])
    cutoff = float(config["longitude_cutoff"])
    minimum_normalized_std = float(config.get("minimum_normalized_std", 0.0))
    frame = read_precipitation_stations(args.stations, ("calibration", "test"), args.era5)
    common = dict(
        frame=frame,
        era5_path=args.era5,
        normalization=norm,
        targets_per_day=targets_per_day,
        cutoff=cutoff,
        training=False,
    )

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
        prediction = predict(
            model,
            data,
            device,
            args.batch_size,
            minimum_normalized_std,
            args.max_calibration_days,
        )
        calibration_predictions[retention] = prediction
        point_q[retention] = conformal_quantile(normalized_log_scores(prediction), 0.90)
        cluster_q[retention] = day_cluster_conformal_quantile(prediction, 0.90, 0.90)

    heavy = build_heavy_precipitation_definition(
        args.era5,
        "tp",
        args.test_region,
        cutoff,
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
        prediction = predict(
            model,
            data,
            device,
            args.batch_size,
            minimum_normalized_std,
            args.max_test_days,
        )
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
        "model_type": checkpoint["model_type"],
        "architecture": "full_context_precipitation",
        "variable": "daily_total_precipitation",
        "target_units": "mm day-1",
        "target_transform": "log1p",
        "primary_point_estimate": "inverse_log1p_median",
        "sensitivity_point_estimate": "shifted_lognormal_mean",
        "interval_evaluation": "unchanged transformed Gaussian and conformal intervals",
        "minimum_normalized_std": minimum_normalized_std,
        "training_experiment": checkpoint["experiment"],
        "training_seed": int(config["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "calibration_region": args.calibration_region,
        "test_region": args.test_region,
        "calibration_label": args.calibration_label,
        "calibration_modes": {
            "deployed_full_density": "90% split-conformal q from 100% station retention",
            "matched_density": "90% split-conformal q from the same station retention as test",
        },
        "conformal_q_log1p_by_retention": {str(key): value for key, value in point_q.items()},
        "day_cluster_conformal_q_log1p_by_retention": {str(key): value for key, value in cluster_q.items()},
        "calibration_scores_per_retention": {
            str(key): int(len(value["target_mm"])) for key, value in calibration_predictions.items()
        },
        "evaluation_seed": args.evaluation_seed,
        "calibration_seed": args.calibration_seed,
        "targets_per_day": targets_per_day,
        "heavy_precipitation_event": heavy.metadata(),
        "results_csv": str(args.output),
        "daily_results_csv": str(daily_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows) and {daily_path} ({len(daily_rows)} rows)")


if __name__ == "__main__":
    main()
