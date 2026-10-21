"""Final neural evaluation with corrected extremes, conformal regimes, and day-level uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from evaluation_protocol import (
    build_extreme_definition,
    conformal_quantile,
    day_cluster_conformal_quantile,
    metric_summary,
    normalized_scores,
    subset_mask,
)
from experiment_data import Normalization
from full_context_data import FullContextDataset, padded_collate, read_domain_stations
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer


def make_model(name: str, latent_side: int) -> torch.nn.Module:
    if name == "convcnp":
        return ConvCNP()
    if name == "latent_transformer":
        return LatentGridTransformer(latent_side=latent_side)
    raise ValueError(f"Unknown model {name!r}")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    data: FullContextDataset,
    device: torch.device,
    batch_size: int,
    max_days: int | None = None,
) -> dict[str, np.ndarray]:
    dataset = data
    if max_days is not None and max_days < len(data):
        indices = np.linspace(0, len(data) - 1, max_days, dtype=int)
        dataset = Subset(data, indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda", collate_fn=padded_collate)
    output = {key: [] for key in ("mean", "std", "target", "date_ns", "context_count", "context_date_ns")}
    model.eval()
    for batch in tqdm(loader, desc="predict", unit="day", dynamic_ncols=True):
        mean, log_std = model(
            batch["context_x"].to(device, non_blocking=True),
            batch["context_y"].to(device, non_blocking=True),
            batch["target_x"].to(device, non_blocking=True),
            batch["context_mask"].to(device, non_blocking=True),
        )
        mean_np = mean.cpu().numpy() * data.norm.value_std + data.norm.value_mean
        std_np = log_std.exp().cpu().numpy() * data.norm.value_std
        target_np = batch["target_y"].numpy() * data.norm.value_std + data.norm.value_mean
        dates_np = batch["date_ns"].numpy()
        output["mean"].append(mean_np.reshape(-1))
        output["std"].append(std_np.reshape(-1))
        output["target"].append(target_np.reshape(-1))
        output["date_ns"].append(np.repeat(dates_np, mean_np.shape[1]))
        output["context_count"].append(batch["context_count"].numpy())
        output["context_date_ns"].append(dates_np)
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
    parser.add_argument("--evaluation-seed", type=int, default=9102)
    parser.add_argument("--calibration-seed", type=int, default=9101)
    parser.add_argument("--targets-per-day", type=int)
    parser.add_argument("--extreme-tail-probability", type=float, default=0.05)
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
    if checkpoint.get("architecture") != "full_context":
        raise ValueError("Checkpoint is not a full-context architecture.")
    config = checkpoint["config"]
    norm = Normalization.from_dict(checkpoint["normalization"])
    model = make_model(checkpoint["model_type"], int(config["latent_side"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    targets_per_day = args.targets_per_day or int(config["targets_per_day"])
    cutoff = float(config["longitude_cutoff"])
    frame = read_domain_stations(args.stations, "tg", ("calibration", "test"), args.era5)
    common = dict(
        frame=frame,
        era5_path=args.era5,
        station_variable="tg",
        era5_variable="t2m",
        era5_offset=-273.15,
        normalization=norm,
        targets_per_day=targets_per_day,
        cutoff=cutoff,
        training=False,
    )
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
        prediction = predict(model, data, device, args.batch_size, args.max_calibration_days)
        calibration_predictions[retention] = prediction
        conformal_q[retention] = conformal_quantile(normalized_scores(prediction), 0.90)
        cluster_conformal_q[retention] = day_cluster_conformal_quantile(prediction, 0.90, 0.90)

    extreme = build_extreme_definition(
        args.era5,
        "t2m",
        -273.15,
        args.test_region,
        cutoff,
        args.extreme_tail_probability,
        args.climatology_half_window_days,
    )
    rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    full_q = conformal_q[1.0]
    full_cluster_q = cluster_conformal_q[1.0]
    for retention_index, retention in enumerate(args.retentions):
        test_data = FullContextDataset(
            split="test",
            context_region=args.test_region,
            target_region=args.test_region,
            retention=retention,
            seed=args.evaluation_seed,
            **common,
        )
        prediction = predict(model, test_data, device, args.batch_size, args.max_test_days)
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
        "model_type": checkpoint["model_type"],
        "architecture": "full_context",
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
        "conformal_q_by_retention": {str(key): value for key, value in conformal_q.items()},
        "day_cluster_conformal_q_by_retention": {str(key): value for key, value in cluster_conformal_q.items()},
        "calibration_scores_per_retention": {
            str(key): int(len(value["target"])) for key, value in calibration_predictions.items()
        },
        "calibration_unit": "sampled ERA5 target point; reported uncertainty intervals are bootstrap intervals across days",
        "evaluation_seed": args.evaluation_seed,
        "calibration_seed": args.calibration_seed,
        "targets_per_day": targets_per_day,
        "extreme_event": extreme.metadata(),
        "results_csv": str(args.output),
        "daily_results_csv": str(daily_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows) and {daily_path} ({len(daily_rows)} rows)")


if __name__ == "__main__":
    main()

