"""Shared final-evaluation protocol: regional anomalies, conformal intervals, and day-level uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from experiment_data import SPLITS, region_mask


RAW_90_QUANTILE = 1.6448536269514722


def conformal_quantile(scores: np.ndarray, coverage: float = 0.90) -> float:
    finite = np.sort(np.asarray(scores, dtype="float64")[np.isfinite(scores)])
    if not len(finite):
        raise ValueError("No finite conformal scores.")
    rank = min(len(finite) - 1, math.ceil((len(finite) + 1) * coverage) - 1)
    return float(finite[rank])


def normalized_scores(prediction: dict[str, np.ndarray]) -> np.ndarray:
    return np.abs(prediction["target"] - prediction["mean"]) / prediction["std"].clip(1e-6)


def day_cluster_conformal_quantile(
    prediction: dict[str, np.ndarray],
    within_day_coverage: float = 0.90,
    day_coverage: float = 0.90,
) -> float:
    scores = normalized_scores(prediction)
    daily_quantiles = [
        conformal_quantile(scores[prediction["date_ns"] == date], within_day_coverage)
        for date in np.unique(prediction["date_ns"])
    ]
    return conformal_quantile(np.asarray(daily_quantiles), day_coverage)

def _region_daily_means(
    era5_path: Path,
    variable: str,
    offset: float,
    region: str,
    cutoff: float,
    split: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    start, end = SPLITS[split]
    with xr.open_dataset(era5_path) as ds:
        field = ds[variable].sel(time=slice(start, end))
        longitudes = field.longitude.values
        selected = np.flatnonzero(region_mask(longitudes, region, cutoff))
        means = field.isel(longitude=selected).mean(dim=("latitude", "longitude"), skipna=True).load().values
        dates = pd.DatetimeIndex(field.time.values).normalize()
    return dates, np.asarray(means, dtype="float64") + offset


def _calendar_climatology(dates: pd.DatetimeIndex, values: np.ndarray, half_window_days: int) -> np.ndarray:
    day = dates.dayofyear.to_numpy()
    climatology = np.empty(366, dtype="float64")
    for target in range(1, 367):
        distance = np.abs(day - target)
        distance = np.minimum(distance, 366 - distance)
        selected = distance <= half_window_days
        climatology[target - 1] = np.nanmean(values[selected])
    return climatology


@dataclass(frozen=True)
class ExtremeDefinition:
    selected_test_dates_ns: frozenset[int]
    low_anomaly_c: float
    high_anomaly_c: float
    tail_probability: float
    climatology_half_window_days: int
    target_region: str
    test_days: int
    selected_test_days: int

    def metadata(self) -> dict[str, object]:
        return {
            "definition": "target-region ERA5 daily spatial-mean anomaly relative to a training-period calendar-day climatology",
            "reference_period": f"{SPLITS['train'][0]} to {SPLITS['train'][1]}",
            "tail_probability_each_side": self.tail_probability,
            "low_anomaly_c": self.low_anomaly_c,
            "high_anomaly_c": self.high_anomaly_c,
            "climatology_half_window_days": self.climatology_half_window_days,
            "target_region": self.target_region,
            "test_days": self.test_days,
            "selected_test_days": self.selected_test_days,
        }


def build_extreme_definition(
    era5_path: Path,
    variable: str,
    offset: float,
    target_region: str,
    cutoff: float,
    tail_probability: float = 0.05,
    climatology_half_window_days: int = 15,
) -> ExtremeDefinition:
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be between 0 and 0.5.")
    train_dates, train_means = _region_daily_means(era5_path, variable, offset, target_region, cutoff, "train")
    climatology = _calendar_climatology(train_dates, train_means, climatology_half_window_days)
    train_anomalies = train_means - climatology[train_dates.dayofyear.to_numpy() - 1]
    low, high = np.quantile(train_anomalies, [tail_probability, 1.0 - tail_probability])
    test_dates, test_means = _region_daily_means(era5_path, variable, offset, target_region, cutoff, "test")
    test_anomalies = test_means - climatology[test_dates.dayofyear.to_numpy() - 1]
    selected = (test_anomalies <= low) | (test_anomalies >= high)
    selected_ns = frozenset(int(value) for value in test_dates.view("int64")[selected])
    return ExtremeDefinition(
        selected_test_dates_ns=selected_ns,
        low_anomaly_c=float(low),
        high_anomaly_c=float(high),
        tail_probability=tail_probability,
        climatology_half_window_days=climatology_half_window_days,
        target_region=target_region,
        test_days=len(test_dates),
        selected_test_days=int(selected.sum()),
    )


def subset_mask(prediction: dict[str, np.ndarray], subset: str, extreme: ExtremeDefinition) -> np.ndarray:
    if subset == "all":
        return np.ones(len(prediction["target"]), dtype=bool)
    if subset == "extreme_days":
        return np.fromiter(
            (int(value) in extreme.selected_test_dates_ns for value in prediction["date_ns"]),
            dtype=bool,
            count=len(prediction["date_ns"]),
        )
    raise ValueError(f"Unknown subset {subset!r}")


def _bootstrap_mean_ci(values: np.ndarray, seed: int, repetitions: int) -> tuple[float, float]:
    values = np.asarray(values, dtype="float64")
    if len(values) < 2:
        value = float(values.mean()) if len(values) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype="float64")
    chunk = 250
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def metric_summary(
    prediction: dict[str, np.ndarray],
    q: float,
    cluster_q: float,
    mask: np.ndarray,
    subset: str,
    retention: float,
    calibration_mode: str,
    bootstrap_seed: int,
    bootstrap_repetitions: int = 2000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not mask.any():
        raise ValueError(f"No points selected for subset={subset}.")
    context_lookup = {
        int(date): int(count)
        for date, count in zip(prediction["context_date_ns"], prediction["context_count"], strict=True)
    }
    daily: list[dict[str, object]] = []
    for date in np.unique(prediction["date_ns"][mask]):
        selected = mask & (prediction["date_ns"] == date)
        mean = prediction["mean"][selected]
        std = prediction["std"][selected].clip(1e-6)
        target = prediction["target"][selected]
        error = target - mean
        daily.append({
            "date": pd.Timestamp(int(date)).strftime("%Y-%m-%d"),
            "date_ns": int(date),
            "subset": subset,
            "retention": retention,
            "calibration_mode": calibration_mode,
            "context_count": context_lookup[int(date)],
            "n_targets": int(selected.sum()),
            "rmse_c": float(np.sqrt(np.mean(error**2))),
            "mae_c": float(np.mean(np.abs(error))),
            "gaussian_nll": float(np.mean(0.5 * (error / std) ** 2 + np.log(std) + 0.5 * np.log(2 * np.pi))),
            "raw_90_coverage": float(np.mean(np.abs(error) <= RAW_90_QUANTILE * std)),
            "raw_mean_width_c": float(np.mean(2 * RAW_90_QUANTILE * std)),
            "conformal_90_coverage": float(np.mean(np.abs(error) <= q * std)),
            "conformal_mean_width_c": float(np.mean(2 * q * std)),
            "day_cluster_90x90_coverage": float(np.mean(np.abs(error) <= cluster_q * std)),
            "day_cluster_90x90_mean_width_c": float(np.mean(2 * cluster_q * std)),
        })
    daily_rmse = np.asarray([row["rmse_c"] for row in daily])
    daily_mae = np.asarray([row["mae_c"] for row in daily])
    daily_coverage = np.asarray([row["conformal_90_coverage"] for row in daily])
    rmse_ci = _bootstrap_mean_ci(daily_rmse, bootstrap_seed, bootstrap_repetitions)
    daily_cluster_coverage = np.asarray([row["day_cluster_90x90_coverage"] for row in daily])
    mae_ci = _bootstrap_mean_ci(daily_mae, bootstrap_seed + 1, bootstrap_repetitions)
    coverage_ci = _bootstrap_mean_ci(daily_coverage, bootstrap_seed + 2, bootstrap_repetitions)
    mean = prediction["mean"][mask]
    cluster_coverage_ci = _bootstrap_mean_ci(daily_cluster_coverage, bootstrap_seed + 3, bootstrap_repetitions)
    std = prediction["std"][mask].clip(1e-6)
    target = prediction["target"][mask]
    error = target - mean
    coverage = float(np.mean(np.abs(error) <= q * std))
    cluster_coverage = float(np.mean(np.abs(error) <= cluster_q * std))
    row = {
        "subset": subset,
        "retention": retention,
        "calibration_mode": calibration_mode,
        "conformal_q": q,
        "day_cluster_conformal_q": cluster_q,
        "context_count_min": int(prediction["context_count"].min()),
        "context_count_median": int(np.median(prediction["context_count"])),
        "context_count_max": int(prediction["context_count"].max()),
        "days": len(daily),
        "n": int(mask.sum()),
        "pooled_rmse_c": float(np.sqrt(np.mean(error**2))),
        "pooled_mae_c": float(np.mean(np.abs(error))),
        "day_mean_rmse_c": float(daily_rmse.mean()),
        "day_mean_rmse_ci95_low": rmse_ci[0],
        "day_mean_rmse_ci95_high": rmse_ci[1],
        "day_mean_mae_c": float(daily_mae.mean()),
        "day_mean_mae_ci95_low": mae_ci[0],
        "day_mean_mae_ci95_high": mae_ci[1],
        "day_mean_gaussian_nll": float(np.mean([item["gaussian_nll"] for item in daily])),
        "raw_90_coverage": float(np.mean(np.abs(error) <= RAW_90_QUANTILE * std)),
        "raw_mean_width_c": float(np.mean(2 * RAW_90_QUANTILE * std)),
        "conformal_90_coverage": coverage,
        "conformal_coverage_gap": coverage - 0.90,
        "day_mean_conformal_coverage": float(daily_coverage.mean()),
        "day_mean_conformal_coverage_ci95_low": coverage_ci[0],
        "day_mean_conformal_coverage_ci95_high": coverage_ci[1],
        "conformal_mean_width_c": float(np.mean(2 * q * std)),
        "day_cluster_90x90_coverage": cluster_coverage,
        "day_mean_cluster_coverage": float(daily_cluster_coverage.mean()),
        "day_mean_cluster_coverage_ci95_low": cluster_coverage_ci[0],
        "day_mean_cluster_coverage_ci95_high": cluster_coverage_ci[1],
        "day_fraction_at_least_90pct_covered": float(np.mean(daily_cluster_coverage >= 0.90)),
        "day_cluster_90x90_mean_width_c": float(np.mean(2 * cluster_q * std)),
    }
    return row, daily
