"""Strict summary for median-primary precipitation point-estimate sensitivity outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SUMMARY_METRICS = (
    "day_mean_rmse_median_mm",
    "day_mean_mae_median_mm",
    "day_mean_rmse_lognormal_mean_mm",
    "day_mean_mae_lognormal_mean_mm",
    "day_mean_gaussian_nll_log1p",
    "day_mean_wet_day_brier",
    "day_mean_wet_day_csi",
    "day_mean_conformal_coverage",
    "conformal_mean_width_mm",
    "day_mean_cluster_coverage",
    "day_fraction_at_least_90pct_covered",
    "day_cluster_90x90_mean_width_mm",
    "p95_predicted_std_log",
    "p99_predicted_std_log",
    "maximum_predicted_std_log",
    "fraction_lognormal_mean_exponent_clipped",
    "p95_one_plus_mean_to_median_ratio",
)


def mean_sd_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean())
    if len(array) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(array.std(ddof=1))
    half_width = float(student_t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half_width, mean + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("final_results_precipitation_point_sensitivity"))
    parser.add_argument("--output-dir", type=Path, default=Path("final_results_precipitation_point_sensitivity/summary"))
    parser.add_argument("--expected-neural-seeds", type=int, nargs="+", default=tuple(range(2026, 2034)))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(args.input_root.rglob("*.csv")):
        if csv_path.name.endswith(".daily.csv") or "_smoke" in csv_path.parts or args.output_dir in csv_path.parents:
            continue
        metadata_path = csv_path.with_suffix(".json")
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        frame.insert(0, "source_file", str(csv_path))
        frame.insert(1, "model", metadata["model_type"])
        frame.insert(2, "training_experiment", metadata["training_experiment"])
        frame.insert(3, "training_seed", metadata.get("training_seed"))
        frame.insert(4, "calibration", metadata["calibration_label"])
        frame.insert(5, "calibration_region", metadata["calibration_region"])
        frame.insert(6, "test_region", metadata["test_region"])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No precipitation result CSV/JSON pairs under {args.input_root}")
    all_results = pd.concat(frames, ignore_index=True)

    if not args.allow_incomplete:
        expected_neural = {
            (model, seed, calibration)
            for model in ("convcnp", "latent_transformer")
            for seed in args.expected_neural_seeds
            for calibration in ("global", "transferred", "oracle_target")
        }
        neural = all_results.loc[all_results.model != "ordinary_kriging"]
        observed_neural = set(zip(neural.model, neural.training_seed.astype(int), neural.calibration))
        missing_neural = sorted(expected_neural - observed_neural)
        kriging = all_results.loc[all_results.model == "ordinary_kriging"]
        missing_kriging = sorted({"global", "transferred", "oracle_target"} - set(kriging.calibration))
        if missing_neural or missing_kriging:
            raise RuntimeError(
                f"Incomplete precipitation point-sensitivity results. Missing neural runs={missing_neural}; "
                f"missing kriging calibrations={missing_kriging}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output_dir / "all_results.csv", index=False)
    neural = all_results.loc[all_results.model != "ordinary_kriging"].copy()
    group_columns = [
        "model",
        "training_experiment",
        "calibration",
        "calibration_region",
        "test_region",
        "subset",
        "retention",
        "calibration_mode",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in neural.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        seeds = sorted(int(value) for value in group.training_seed.dropna().unique())
        row["training_seeds"] = ",".join(map(str, seeds))
        row["n_training_seeds"] = len(seeds)
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_sd_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd_across_seeds"] = sd
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    seed_summary.to_csv(args.output_dir / "neural_seed_summary.csv", index=False)
    headline = all_results.loc[
        (all_results.subset == "all")
        & (all_results.calibration_mode == "deployed_full_density")
        & (all_results.retention.isin([1.0, 0.1]))
    ].copy()
    headline.to_csv(args.output_dir / "headline_results.csv", index=False)
    print(
        f"wrote {args.output_dir / 'all_results.csv'} ({len(all_results)} rows), "
        f"{args.output_dir / 'neural_seed_summary.csv'} ({len(seed_summary)} rows)"
    )


if __name__ == "__main__":
    main()
