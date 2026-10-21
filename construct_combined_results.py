"""Construct harmonized paper tables from completed temperature and precipitation results.

This script is read-only with respect to the source result directories. Temperature
uses the final multi-seed summaries in ``final_results``. Precipitation uses the
median-primary point-sensitivity summaries, while retaining the lognormal-mean
back-transformation as an explicit sensitivity metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = {"ordinary_kriging": 0, "convcnp": 1, "latent_transformer": 2}
CALIBRATION_ORDER = {"global": 0, "transferred": 1, "oracle_target": 2}
VARIABLE_ORDER = {"temperature": 0, "precipitation": 1}


def _neural_rows(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    common = [
        "model", "training_experiment", "calibration", "calibration_region",
        "test_region", "subset", "retention", "calibration_mode",
        "n_training_seeds",
    ]
    out = frame[common].copy()
    out["variable"] = variable
    out["estimate_kind"] = "eight_seed_mean"
    out["uncertainty_basis"] = "training_runs"
    if variable == "temperature":
        out["unit"] = "degC"
        out["primary_point_estimate"] = "predictive_mean"
        mappings = {
            "rmse": "day_mean_rmse_c",
            "mae": "day_mean_mae_c",
            "nll": "day_mean_gaussian_nll",
            "conformal_coverage": "day_mean_conformal_coverage",
            "conformal_width": "conformal_mean_width_c",
            "cluster_coverage": "day_mean_cluster_coverage",
            "fraction_days_ge90": "day_fraction_at_least_90pct_covered",
        }
    else:
        out["unit"] = "mm/day"
        out["primary_point_estimate"] = "inverse_log1p_median"
        mappings = {
            "rmse": "day_mean_rmse_median_mm",
            "mae": "day_mean_mae_median_mm",
            "nll": "day_mean_gaussian_nll_log1p",
            "conformal_coverage": "day_mean_conformal_coverage",
            "conformal_width": "conformal_mean_width_mm",
            "cluster_coverage": "day_mean_cluster_coverage",
            "fraction_days_ge90": "day_fraction_at_least_90pct_covered",
        }
    for target, source in mappings.items():
        out[f"{target}_mean"] = frame[f"{source}_mean"]
        out[f"{target}_sd"] = frame[f"{source}_sd_across_seeds"]
        out[f"{target}_ci95_low"] = frame[f"{source}_ci95_low"]
        out[f"{target}_ci95_high"] = frame[f"{source}_ci95_high"]
    if variable == "precipitation":
        for target, source in {
            "rmse_lognormal_mean_sensitivity": "day_mean_rmse_lognormal_mean_mm",
            "mae_lognormal_mean_sensitivity": "day_mean_mae_lognormal_mean_mm",
        }.items():
            out[f"{target}_mean"] = frame[f"{source}_mean"]
            out[f"{target}_sd"] = frame[f"{source}_sd_across_seeds"]
            out[f"{target}_ci95_low"] = frame[f"{source}_ci95_low"]
            out[f"{target}_ci95_high"] = frame[f"{source}_ci95_high"]
    return out


def _kriging_rows(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    frame = frame.loc[frame["model"] == "ordinary_kriging"].copy()
    common = [
        "model", "training_experiment", "calibration", "calibration_region",
        "test_region", "subset", "retention", "calibration_mode",
    ]
    out = frame[common].copy()
    out["variable"] = variable
    out["n_training_seeds"] = 1
    out["estimate_kind"] = "single_deterministic_baseline"
    out["uncertainty_basis"] = "test_days"
    if variable == "temperature":
        out["unit"] = "degC"
        out["primary_point_estimate"] = "predictive_mean"
        mappings = {
            "rmse": "day_mean_rmse_c",
            "mae": "day_mean_mae_c",
            "nll": "day_mean_gaussian_nll",
            "conformal_coverage": "day_mean_conformal_coverage",
            "conformal_width": "conformal_mean_width_c",
            "cluster_coverage": "day_mean_cluster_coverage",
            "fraction_days_ge90": "day_fraction_at_least_90pct_covered",
        }
    else:
        out["unit"] = "mm/day"
        out["primary_point_estimate"] = "inverse_log1p_median"
        mappings = {
            "rmse": "day_mean_rmse_median_mm",
            "mae": "day_mean_mae_median_mm",
            "nll": "day_mean_gaussian_nll_log1p",
            "conformal_coverage": "day_mean_conformal_coverage",
            "conformal_width": "conformal_mean_width_mm",
            "cluster_coverage": "day_mean_cluster_coverage",
            "fraction_days_ge90": "day_fraction_at_least_90pct_covered",
        }
    for target, source in mappings.items():
        out[f"{target}_mean"] = frame[source]
        out[f"{target}_sd"] = np.nan
        low = f"{source}_ci95_low"
        high = f"{source}_ci95_high"
        out[f"{target}_ci95_low"] = frame[low] if low in frame else np.nan
        out[f"{target}_ci95_high"] = frame[high] if high in frame else np.nan
    if variable == "precipitation":
        out["rmse_lognormal_mean_sensitivity_mean"] = frame["day_mean_rmse_lognormal_mean_mm"]
        out["mae_lognormal_mean_sensitivity_mean"] = frame["day_mean_mae_lognormal_mean_mm"]
        out["rmse_lognormal_mean_sensitivity_sd"] = np.nan
        out["mae_lognormal_mean_sensitivity_sd"] = np.nan
        out["rmse_lognormal_mean_sensitivity_ci95_low"] = frame["day_mean_rmse_lognormal_mean_ci95_low"]
        out["rmse_lognormal_mean_sensitivity_ci95_high"] = frame["day_mean_rmse_lognormal_mean_ci95_high"]
        out["mae_lognormal_mean_sensitivity_ci95_low"] = frame["day_mean_mae_lognormal_mean_ci95_low"]
        out["mae_lognormal_mean_sensitivity_ci95_high"] = frame["day_mean_mae_lognormal_mean_ci95_high"]
    return out


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame.assign(
        _variable=frame["variable"].map(VARIABLE_ORDER),
        _calibration=frame["calibration"].map(CALIBRATION_ORDER),
        _model=frame["model"].map(MODEL_ORDER),
    ).sort_values(
        ["_variable", "_calibration", "subset", "retention", "calibration_mode", "_model"]
    )
    return order.drop(columns=["_variable", "_calibration", "_model"]).reset_index(drop=True)


def _format_estimate(row: pd.Series, metric: str) -> str:
    mean = row[f"{metric}_mean"]
    sd = row[f"{metric}_sd"]
    if pd.notna(sd):
        return f"{mean:.3f} +/- {sd:.3f}"
    low = row[f"{metric}_ci95_low"]
    high = row[f"{metric}_ci95_high"]
    if pd.notna(low) and pd.notna(high):
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"
    return f"{mean:.3f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "Variable": row["variable"].title(),
            "Deployment": row["calibration"],
            "Model": row["model"],
            "RMSE": _format_estimate(row, "rmse"),
            "MAE": _format_estimate(row, "mae"),
            "Coverage": f"{100 * row['conformal_coverage_mean']:.1f}%",
            "Width": f"{row['conformal_width_mean']:.3f}",
        })
    display = pd.DataFrame(rows)
    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature-root", type=Path, default=Path("final_results/summary"))
    parser.add_argument(
        "--precipitation-root",
        type=Path,
        default=Path("final_results_precipitation_point_sensitivity/summary"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results"))
    args = parser.parse_args()

    temperature_neural = pd.read_csv(args.temperature_root / "neural_seed_summary.csv")
    temperature_all = pd.read_csv(args.temperature_root / "all_results.csv")
    precipitation_neural = pd.read_csv(args.precipitation_root / "neural_seed_summary.csv")
    precipitation_all = pd.read_csv(args.precipitation_root / "all_results.csv")

    combined = _sort(pd.concat([
        _neural_rows(temperature_neural, "temperature"),
        _kriging_rows(temperature_all, "temperature"),
        _neural_rows(precipitation_neural, "precipitation"),
        _kriging_rows(precipitation_all, "precipitation"),
    ], ignore_index=True, sort=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "combined_aggregate_results.csv", index=False)

    main_table = _sort(combined.loc[
        (combined["subset"] == "all")
        & (combined["retention"] == 1.0)
        & (combined["calibration_mode"] == "deployed_full_density")
    ].copy())
    sparse_table = _sort(combined.loc[
        (combined["subset"] == "all")
        & (combined["calibration"].isin(["global", "transferred"]))
    ].copy())
    extreme_table = _sort(combined.loc[
        (combined["subset"].isin(["extreme_days", "heavy_precipitation_days"]))
        & (combined["retention"] == 1.0)
        & (combined["calibration_mode"] == "deployed_full_density")
    ].copy())
    calibration_table = main_table[[
        "variable", "model", "calibration", "calibration_region", "test_region",
        "conformal_coverage_mean", "conformal_coverage_sd",
        "conformal_coverage_ci95_low", "conformal_coverage_ci95_high",
        "conformal_width_mean", "conformal_width_sd", "uncertainty_basis",
    ]].copy()

    main_table.to_csv(args.output_dir / "main_results.csv", index=False)
    sparse_table.to_csv(args.output_dir / "sparsity_results.csv", index=False)
    extreme_table.to_csv(args.output_dir / "extreme_event_results.csv", index=False)
    calibration_table.to_csv(args.output_dir / "calibration_results.csv", index=False)

    precip_seed_diagnostics = precipitation_all.loc[
        (precipitation_all["model"] == "convcnp")
        & (precipitation_all["subset"] == "all")
        & (precipitation_all["retention"] == 1.0)
        & (precipitation_all["calibration_mode"] == "deployed_full_density"),
        [
            "training_seed", "calibration", "day_mean_rmse_median_mm",
            "day_mean_rmse_lognormal_mean_mm", "day_mean_gaussian_nll_log1p",
            "day_mean_conformal_coverage", "p99_predicted_std_log",
            "maximum_predicted_std_log", "fraction_lognormal_mean_exponent_clipped",
        ],
    ].sort_values(["calibration", "training_seed"])
    precip_seed_diagnostics.to_csv(
        args.output_dir / "convcnp_precipitation_seed_diagnostics.csv", index=False
    )

    readme = f"""# Combined temperature and precipitation results

Generated by `construct_combined_results.py` from the completed temperature and
precipitation summaries. Existing result directories are read-only inputs.

## Main full-density results

{_markdown_table(main_table)}

Neural entries are means +/- standard deviations across eight independently trained
seeds. Kriging entries are single deterministic estimates with 95% intervals obtained
by resampling test days. Coverage is empirical conformal coverage for nominal 90%
intervals. Width uses degC for temperature and mm/day for precipitation.

For precipitation, RMSE and MAE use the inverse-log1p median as the primary physical-
scale point estimate. The lognormal-mean back-transformation remains in the CSV files
as a sensitivity metric and should not be silently substituted for the primary value.

## Files

- `combined_aggregate_results.csv`: every harmonized aggregate result.
- `main_results.csv`: all-day, full-density global, transfer, and oracle results.
- `sparsity_results.csv`: all-day global and transfer results at every retention and
  both deployed-full-density and matched-density calibration where applicable.
- `extreme_event_results.csv`: temperature extreme days and heavy-precipitation days.
- `calibration_results.csv`: compact coverage and interval-width table.
- `convcnp_precipitation_seed_diagnostics.csv`: seed-level instability evidence.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    expected_main = 2 * 3 * 3
    if len(main_table) != expected_main:
        raise RuntimeError(f"Expected {expected_main} main rows, found {len(main_table)}")
    if set(main_table["n_training_seeds"]) != {1, 8}:
        raise RuntimeError("Main table does not contain deterministic and eight-seed results.")
    key_metrics = ["rmse_mean", "mae_mean", "nll_mean", "conformal_coverage_mean"]
    if not np.isfinite(main_table[key_metrics].to_numpy(dtype="float64")).all():
        raise RuntimeError("Main table contains non-finite headline metrics.")
    print(
        f"wrote {args.output_dir}: combined={len(combined)}, main={len(main_table)}, "
        f"sparse={len(sparse_table)}, extreme={len(extreme_table)}, "
        f"calibration={len(calibration_table)}, diagnostics={len(precip_seed_diagnostics)}"
    )


if __name__ == "__main__":
    main()
