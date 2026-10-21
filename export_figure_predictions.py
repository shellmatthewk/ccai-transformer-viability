"""Export aligned, point-level predictions for spatial reconstruction figures.

The final evaluation scripts intentionally persist only aggregate and per-day
metrics.  This utility reruns test inference for one neural training seed and
writes a wide Parquet file in which ERA5 truth, ConvCNP, latent-transformer,
and ordinary-kriging predictions share the same date and target coordinates.

The default deployment is the full-density western-source to eastern-target
transfer experiment, because that is the paper's central deployment shift.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from evaluation_protocol import build_extreme_definition
from evaluate_local_kriging import local_ordinary_kriging
from experiment_data import Normalization
from full_context_data import FullContextDataset, read_domain_stations
from models.convcnp import ConvCNP
from models.latent_grid_transformer import LatentGridTransformer
from precipitation_data import PrecipitationDataset, read_precipitation_stations
from precipitation_protocol_point_sensitivity import (
    build_heavy_precipitation_definition,
    transformed_interval_mm,
)


MODEL_NAMES = ("convcnp", "latent_transformer")
METHOD_NAMES = (*MODEL_NAMES, "ordinary_kriging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned truth and predictions for one-seed spatial figures."
    )
    parser.add_argument("--variable", choices=("temperature", "precipitation"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--deployment",
        choices=("global", "transfer", "oracle_target"),
        default="transfer",
    )
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("figure_data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature-stations", type=Path, default=Path("data/tg/daily_temperature_2010_2021.parquet"))
    parser.add_argument("--temperature-era5", type=Path, default=Path("data/era5_daily_t2m_2010_2021.nc"))
    parser.add_argument("--precipitation-stations", type=Path, default=Path("data/rr/daily_precipitation_2010_2021.parquet"))
    parser.add_argument("--precipitation-era5", type=Path, default=Path("data/era5_daily_tp_mm_2010_2021.nc"))
    args = parser.parse_args()
    if not 0.0 < args.retention <= 1.0:
        parser.error("--retention must be in (0, 1].")
    if args.max_days is not None and args.max_days < 1:
        parser.error("--max-days must be positive.")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    name = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if name == "auto":
        name = "cpu"
    return torch.device(name)


def checkpoint_path(variable: str, seed: int, experiment: str, model: str) -> Path:
    if variable == "temperature":
        if seed == 2026:
            return Path("experiments_full_context") / experiment / model / "best.pt"
        return Path("experiments_multiseed") / f"seed_{seed}" / experiment / model / "best.pt"
    return Path("experiments_precipitation") / f"seed_{seed}" / experiment / model / "best.pt"


def result_root(variable: str) -> Path:
    if variable == "temperature":
        return Path("final_results_temperature")
    return Path("final_results_precipitation_point_sensitivity")


def result_paths(variable: str, seed: int, deployment: str) -> dict[str, dict[str, Path]]:
    root = result_root(variable)
    result_label = "oracle" if deployment == "oracle_target" else deployment
    paths: dict[str, dict[str, Path]] = {}
    for model in MODEL_NAMES:
        base = root / "neural" / f"seed_{seed}" / f"{model}_{result_label}"
        paths[model] = {"csv": base.with_suffix(".csv"), "json": base.with_suffix(".json")}
    base = root / "kriging" / f"ordinary_kriging_{result_label}"
    paths["ordinary_kriging"] = {"csv": base.with_suffix(".csv"), "json": base.with_suffix(".json")}
    return paths


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_model(name: str, latent_side: int) -> torch.nn.Module:
    if name == "convcnp":
        return ConvCNP()
    if name == "latent_transformer":
        return LatentGridTransformer(latent_side=latent_side)
    raise ValueError(f"Unknown model {name!r}.")


def norms_match(left: Normalization, right: Normalization) -> bool:
    return (
        np.isclose(left.value_mean, right.value_mean)
        and np.isclose(left.value_std, right.value_std)
        and np.allclose(left.coord_min, right.coord_min)
        and np.allclose(left.coord_max, right.coord_max)
    )


def recover_coords(normalized: np.ndarray, norm: Normalization) -> np.ndarray:
    return (
        (normalized + 1.0) * 0.5 * (norm.coord_max - norm.coord_min)
        + norm.coord_min
    ).astype("float32")


def full_density_q(variable: str, metadata: dict[str, Any]) -> float:
    key = "conformal_q_by_retention" if variable == "temperature" else "conformal_q_log1p_by_retention"
    values = metadata[key]
    if "1.0" in values:
        return float(values["1.0"])
    return float(values[str(1.0)])


def selected_indices(length: int, max_days: int | None) -> np.ndarray:
    if max_days is None or max_days >= length:
        return np.arange(length)
    return np.linspace(0, length - 1, max_days, dtype=int)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    retention_label = f"retention_{args.retention:g}".replace(".", "p")
    stem = f"{args.variable}_seed_{args.seed}_{args.deployment}_{retention_label}"
    parquet = args.output_dir / f"{stem}.parquet"
    daily = args.output_dir / f"{stem}.daily.csv"
    metadata = args.output_dir / f"{stem}.json"
    partial = args.output_dir / f"{stem}.partial.parquet"
    return parquet, daily, metadata, partial


def expected_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["subset"] == "all")
        & np.isclose(frame["retention"], 1.0)
        & (frame["calibration_mode"] == "deployed_full_density")
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one full-density all-day row in {path}, found {len(selected)}.")
    return selected.iloc[0]


def validate_against_final_results(
    variable: str,
    daily: pd.DataFrame,
    paths: dict[str, dict[str, Path]],
    enabled: bool,
) -> dict[str, dict[str, float | bool]]:
    validation: dict[str, dict[str, float | bool]] = {}
    if not enabled:
        return validation
    rmse_suffix = "rmse_c" if variable == "temperature" else "rmse_median_mm_day"
    expected_rmse_key = "day_mean_rmse_c" if variable == "temperature" else "day_mean_rmse_median_mm"
    for method in METHOD_NAMES:
        expected = expected_row(paths[method]["csv"])
        actual_rmse = float(daily[f"{method}_{rmse_suffix}"].mean())
        actual_coverage = float(daily[f"{method}_coverage90"].mean())
        expected_rmse = float(expected[expected_rmse_key])
        expected_coverage = float(expected["day_mean_conformal_coverage"])
        rmse_ok = bool(np.isclose(actual_rmse, expected_rmse, rtol=5e-5, atol=5e-5))
        coverage_ok = bool(np.isclose(actual_coverage, expected_coverage, rtol=5e-5, atol=5e-5))
        validation[method] = {
            "actual_day_mean_rmse": actual_rmse,
            "expected_day_mean_rmse": expected_rmse,
            "rmse_absolute_difference": abs(actual_rmse - expected_rmse),
            "actual_day_mean_coverage90": actual_coverage,
            "expected_day_mean_coverage90": expected_coverage,
            "coverage_absolute_difference": abs(actual_coverage - expected_coverage),
            "passed": rmse_ok and coverage_ok,
        }
    failures = [method for method, values in validation.items() if not values["passed"]]
    if failures:
        raise RuntimeError(f"Export validation disagrees with final results for: {', '.join(failures)}")
    return validation


def load_models_and_config(
    variable: str,
    seed: int,
    experiment: str,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]], Normalization, int, float]:
    expected_architecture = "full_context" if variable == "temperature" else "full_context_precipitation"
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    shared_norm: Normalization | None = None
    targets_per_day: int | None = None
    cutoff: float | None = None
    for model_name in MODEL_NAMES:
        path = checkpoint_path(variable, seed, experiment, model_name)
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != expected_architecture:
            raise ValueError(f"Unexpected architecture in {path}: {checkpoint.get('architecture')!r}")
        if checkpoint.get("experiment") != experiment:
            raise ValueError(f"Unexpected experiment in {path}: {checkpoint.get('experiment')!r}")
        config = checkpoint["config"]
        if int(config["seed"]) != seed:
            raise ValueError(f"Unexpected training seed in {path}: {config['seed']!r}")
        norm = Normalization.from_dict(checkpoint["normalization"])
        if shared_norm is None:
            shared_norm = norm
            targets_per_day = int(config["targets_per_day"])
            cutoff = float(config["longitude_cutoff"])
        else:
            if not norms_match(shared_norm, norm):
                raise ValueError("ConvCNP and transformer checkpoints use different normalization.")
            if targets_per_day != int(config["targets_per_day"]):
                raise ValueError("ConvCNP and transformer checkpoints use different target counts.")
            if not np.isclose(cutoff, float(config["longitude_cutoff"])):
                raise ValueError("ConvCNP and transformer checkpoints use different longitude cutoffs.")
        model = make_model(model_name, int(config["latent_side"]))
        model.load_state_dict(checkpoint["model"])
        models[model_name] = model.to(device).eval()
        checkpoint["_path"] = str(path)
        checkpoints[model_name] = checkpoint
    assert shared_norm is not None and targets_per_day is not None and cutoff is not None
    return models, checkpoints, shared_norm, targets_per_day, cutoff


def build_dataset(
    args: argparse.Namespace,
    norm: Normalization,
    targets_per_day: int,
    cutoff: float,
    target_region: str,
    evaluation_seed: int,
) -> tuple[FullContextDataset | PrecipitationDataset, dict[str, Any]]:
    if args.variable == "temperature":
        frame = read_domain_stations(args.temperature_stations, "tg", ("test",), args.temperature_era5)
        dataset = FullContextDataset(
            frame=frame,
            era5_path=args.temperature_era5,
            station_variable="tg",
            era5_variable="t2m",
            era5_offset=-273.15,
            split="test",
            context_region=target_region,
            target_region=target_region,
            normalization=norm,
            targets_per_day=targets_per_day,
            retention=args.retention,
            seed=evaluation_seed,
            training=False,
            cutoff=cutoff,
        )
        event = build_extreme_definition(
            args.temperature_era5, "t2m", -273.15, target_region, cutoff
        )
        return dataset, {
            "name": "temperature_extreme_day",
            "selected_dates_ns": event.selected_test_dates_ns,
            "definition": event.metadata(),
        }
    frame = read_precipitation_stations(
        args.precipitation_stations, ("test",), args.precipitation_era5
    )
    dataset = PrecipitationDataset(
        frame=frame,
        era5_path=args.precipitation_era5,
        split="test",
        context_region=target_region,
        target_region=target_region,
        normalization=norm,
        targets_per_day=targets_per_day,
        retention=args.retention,
        seed=evaluation_seed,
        training=False,
        cutoff=cutoff,
    )
    event = build_heavy_precipitation_definition(
        args.precipitation_era5, "tp", target_region, cutoff
    )
    return dataset, {
        "name": "heavy_precipitation_day",
        "selected_dates_ns": event.selected_test_dates_ns,
        "definition": event.metadata(),
    }


def temperature_method_values(
    mean_normalized: np.ndarray,
    log_std_normalized: np.ndarray,
    norm: Normalization,
    q: float,
) -> dict[str, np.ndarray]:
    mean = mean_normalized * norm.value_std + norm.value_mean
    std = np.exp(log_std_normalized) * norm.value_std
    return {
        "prediction": mean.astype("float32"),
        "std": std.astype("float32"),
        "lower": (mean - q * std).astype("float32"),
        "upper": (mean + q * std).astype("float32"),
    }


def precipitation_method_values(
    mean_normalized: np.ndarray,
    log_std_normalized: np.ndarray,
    norm: Normalization,
    minimum_normalized_std: float,
    q: float,
) -> dict[str, np.ndarray]:
    mean_log = mean_normalized * norm.value_std + norm.value_mean
    normalized_std = np.sqrt(np.exp(2.0 * log_std_normalized) + minimum_normalized_std**2)
    std_log = normalized_std * norm.value_std
    prediction = np.maximum(np.expm1(np.clip(mean_log, -20.0, 10.0)), 0.0)
    lower, upper = transformed_interval_mm(mean_log, std_log, q)
    return {
        "prediction": prediction.astype("float32"),
        "mean_log": mean_log.astype("float32"),
        "std_log": std_log.astype("float32"),
        "lower": lower.astype("float32"),
        "upper": upper.astype("float32"),
    }


def frame_for_day(
    variable: str,
    date_ns: int,
    coords: np.ndarray,
    context_count: int,
    truth: np.ndarray,
    event_selected: bool,
    methods: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    count = len(truth)
    common: dict[str, Any] = {
        "date": np.full(count, np.datetime64(date_ns, "ns")),
        "day_target_index": np.arange(count, dtype="int16"),
        "longitude": coords[:, 0].astype("float32"),
        "latitude": coords[:, 1].astype("float32"),
        "context_count": np.full(count, context_count, dtype="int32"),
        "is_event_day": np.full(count, event_selected, dtype=bool),
    }
    if variable == "temperature":
        common["truth_c"] = truth.astype("float32")
        for method, values in methods.items():
            common[f"{method}_mean_c"] = values["prediction"]
            common[f"{method}_std_c"] = values["std"]
            common[f"{method}_lower90_c"] = values["lower"]
            common[f"{method}_upper90_c"] = values["upper"]
    else:
        common["truth_mm_day"] = truth.astype("float32")
        for method, values in methods.items():
            common[f"{method}_median_mm_day"] = values["prediction"]
            common[f"{method}_mean_log1p"] = values["mean_log"]
            common[f"{method}_std_log1p"] = values["std_log"]
            common[f"{method}_lower90_mm_day"] = values["lower"]
            common[f"{method}_upper90_mm_day"] = values["upper"]
    return pd.DataFrame(common)


def daily_row(
    variable: str,
    date_ns: int,
    context_count: int,
    truth: np.ndarray,
    event_selected: bool,
    methods: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "date": pd.Timestamp(date_ns).strftime("%Y-%m-%d"),
        "date_ns": date_ns,
        "context_count": context_count,
        "n_targets": len(truth),
        "is_event_day": event_selected,
    }
    suffix = "rmse_c" if variable == "temperature" else "rmse_median_mm_day"
    for method, values in methods.items():
        error = truth - values["prediction"]
        row[f"{method}_{suffix}"] = float(np.sqrt(np.mean(error**2)))
        row[f"{method}_mae"] = float(np.mean(np.abs(error)))
        row[f"{method}_coverage90"] = float(
            np.mean((truth >= values["lower"]) & (truth <= values["upper"]))
        )
        row[f"{method}_mean_interval_width"] = float(
            np.mean(values["upper"] - values["lower"])
        )
    return row


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    experiment = "global" if args.deployment == "global" else "source"
    target_region = "global" if args.deployment == "global" else "target"
    evaluation_seed = 9102 if args.variable == "temperature" else 19102
    paths = result_paths(args.variable, args.seed, args.deployment)
    for method_paths in paths.values():
        for path in method_paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

    parquet_path, daily_path, metadata_path, partial_path = output_paths(args)
    outputs = (parquet_path, daily_path, metadata_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"Output exists; use --overwrite to replace: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        partial_path.unlink()

    models, checkpoints, norm, targets_per_day, cutoff = load_models_and_config(
        args.variable, args.seed, experiment, device
    )
    dataset, event = build_dataset(
        args, norm, targets_per_day, cutoff, target_region, evaluation_seed
    )
    indices = selected_indices(len(dataset), args.max_days)
    result_metadata = {method: read_json(values["json"]) for method, values in paths.items()}
    conformal_q = {
        method: full_density_q(args.variable, metadata)
        for method, metadata in result_metadata.items()
    }
    kriging_metadata = result_metadata["ordinary_kriging"]
    kriging_neighbors = int(kriging_metadata["neighbor_count"])
    kriging_lengthscale = float(kriging_metadata["lengthscale_km"])
    kriging_nugget = float(kriging_metadata["nugget_fraction"])
    minimum_std = {
        model: float(checkpoints[model]["config"].get("minimum_normalized_std", 0.0))
        for model in MODEL_NAMES
    }

    writer: pq.ParquetWriter | None = None
    daily_rows: list[dict[str, Any]] = []
    row_count = 0
    started = time.perf_counter()
    total_days = len(indices)
    try:
        for day_number, index in enumerate(indices, start=1):
            sample = dataset[int(index)]
            context_x_np = sample["context_x"].numpy()
            target_x_np = sample["target_x"].numpy()
            coords = recover_coords(target_x_np, norm)
            context_coords = recover_coords(context_x_np, norm)
            date_ns = int(sample["date_ns"].item())
            context_count = int(sample["context_count"].item())
            event_selected = date_ns in event["selected_dates_ns"]

            context_x = sample["context_x"].unsqueeze(0).to(device, non_blocking=True)
            context_y = sample["context_y"].unsqueeze(0).to(device, non_blocking=True)
            target_x = sample["target_x"].unsqueeze(0).to(device, non_blocking=True)
            context_mask = torch.ones(
                (1, context_count), dtype=torch.bool, device=device
            )
            methods: dict[str, dict[str, np.ndarray]] = {}
            for model_name, model in models.items():
                mean, log_std = model(context_x, context_y, target_x, context_mask)
                mean_np = mean[0].cpu().numpy()
                log_std_np = log_std[0].cpu().numpy()
                if args.variable == "temperature":
                    methods[model_name] = temperature_method_values(
                        mean_np, log_std_np, norm, conformal_q[model_name]
                    )
                else:
                    methods[model_name] = precipitation_method_values(
                        mean_np,
                        log_std_np,
                        norm,
                        minimum_std[model_name],
                        conformal_q[model_name],
                    )

            if args.variable == "temperature":
                truth = sample["target_y"].numpy() * norm.value_std + norm.value_mean
                context_values = sample["context_y"].numpy() * norm.value_std + norm.value_mean
                kriging_mean, kriging_std = local_ordinary_kriging(
                    context_coords,
                    context_values,
                    coords,
                    kriging_neighbors,
                    kriging_lengthscale,
                    kriging_nugget,
                )
                q = conformal_q["ordinary_kriging"]
                methods["ordinary_kriging"] = {
                    "prediction": kriging_mean.astype("float32"),
                    "std": kriging_std.astype("float32"),
                    "lower": (kriging_mean - q * kriging_std).astype("float32"),
                    "upper": (kriging_mean + q * kriging_std).astype("float32"),
                }
            else:
                truth_log = sample["target_y"].numpy() * norm.value_std + norm.value_mean
                truth = np.maximum(np.expm1(truth_log), 0.0)
                context_log = sample["context_y"].numpy() * norm.value_std + norm.value_mean
                kriging_mean_log, kriging_std_log = local_ordinary_kriging(
                    context_coords,
                    context_log,
                    coords,
                    kriging_neighbors,
                    kriging_lengthscale,
                    kriging_nugget,
                )
                kriging_prediction = np.maximum(
                    np.expm1(np.clip(kriging_mean_log, -20.0, 10.0)), 0.0
                )
                lower, upper = transformed_interval_mm(
                    kriging_mean_log,
                    kriging_std_log,
                    conformal_q["ordinary_kriging"],
                )
                methods["ordinary_kriging"] = {
                    "prediction": kriging_prediction.astype("float32"),
                    "mean_log": kriging_mean_log.astype("float32"),
                    "std_log": kriging_std_log.astype("float32"),
                    "lower": lower.astype("float32"),
                    "upper": upper.astype("float32"),
                }

            frame = frame_for_day(
                args.variable,
                date_ns,
                coords,
                context_count,
                truth,
                event_selected,
                methods,
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial_path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
            daily_rows.append(
                daily_row(
                    args.variable,
                    date_ns,
                    context_count,
                    truth,
                    event_selected,
                    methods,
                )
            )
            if day_number == 1 or day_number % 25 == 0 or day_number == total_days:
                elapsed = time.perf_counter() - started
                rate = day_number / elapsed if elapsed > 0.0 else 0.0
                remaining = total_days - day_number
                eta = remaining / rate if rate > 0.0 else float("nan")
                print(
                    f"export {args.variable} {args.deployment}: "
                    f"{day_number}/{total_days} days, {elapsed:.1f}s elapsed, "
                    f"{eta:.1f}s ETA",
                    flush=True,
                )
    except Exception:
        if writer is not None:
            writer.close()
        if partial_path.exists():
            partial_path.unlink()
        raise
    else:
        if writer is None:
            raise RuntimeError("No rows were exported.")
        writer.close()
    finally:
        dataset.ds.close()

    daily_frame = pd.DataFrame(daily_rows)
    expected_rows = int(sum(min(targets_per_day, len(dataset.target_indices)) for _ in indices))
    if row_count != expected_rows:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {expected_rows} rows, wrote {row_count}.")
    if args.deployment in ("transfer", "oracle_target"):
        longitude_min = float(pq.read_table(partial_path, columns=["longitude"]).column(0).to_numpy().min())
        if longitude_min <= cutoff:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Transfer export contains longitude {longitude_min}, expected all values > {cutoff}."
            )

    full_run = args.max_days is None and np.isclose(args.retention, 1.0)
    validation = validate_against_final_results(
        args.variable, daily_frame, paths, enabled=full_run
    )

    partial_daily = daily_path.with_suffix(".partial.csv")
    partial_metadata = metadata_path.with_suffix(".partial.json")
    daily_frame.to_csv(partial_daily, index=False)
    metadata: dict[str, Any] = {
        "variable": args.variable,
        "training_seed": args.seed,
        "deployment": args.deployment,
        "training_experiment": experiment,
        "test_region": target_region,
        "retention": args.retention,
        "evaluation_seed": evaluation_seed,
        "targets_per_day": targets_per_day,
        "days": len(indices),
        "rows": row_count,
        "date_min": daily_frame["date"].min(),
        "date_max": daily_frame["date"].max(),
        "longitude_cutoff": cutoff,
        "device": str(device),
        "checkpoint_paths": {
            model: checkpoints[model]["_path"] for model in MODEL_NAMES
        },
        "checkpoint_epochs": {
            model: int(checkpoints[model]["epoch"]) for model in MODEL_NAMES
        },
        "kriging": {
            "neighbors": kriging_neighbors,
            "lengthscale_km": kriging_lengthscale,
            "nugget_fraction": kriging_nugget,
        },
        "conformal_q_full_density": conformal_q,
        "event": event["definition"],
        "primary_precipitation_point_estimate": (
            "inverse_log1p_median" if args.variable == "precipitation" else None
        ),
        "validation_against_final_results": validation,
        "parquet": str(parquet_path),
        "daily_metrics_csv": str(daily_path),
    }
    partial_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.overwrite:
        for path in outputs:
            path.unlink(missing_ok=True)
    partial_path.replace(parquet_path)
    partial_daily.replace(daily_path)
    partial_metadata.replace(metadata_path)
    print(
        f"wrote {parquet_path} ({row_count:,} rows), {daily_path}, and {metadata_path}; "
        f"validation={'passed' if validation else 'not run for partial export'}"
    )


if __name__ == "__main__":
    main()
