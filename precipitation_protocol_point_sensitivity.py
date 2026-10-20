"""Precipitation-specific extremes, conformal intervals, metrics, and day-level uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import xarray as xr

from experiment_data import SPLITS, region_mask


RAW_90_QUANTILE = 1.6448536269514722
WET_DAY_THRESHOLD_MM = 1.0


def conformal_quantile(scores: np.ndarray, coverage: float = 0.90) -> float:
    finite = np.sort(np.asarray(scores, dtype="float64")[np.isfinite(scores)])
    if not len(finite):
        raise ValueError("No finite conformal scores.")
    rank = min(len(finite) - 1, math.ceil((len(finite) + 1) * coverage) - 1)
    return float(finite[rank])


def normalized_log_scores(prediction: dict[str, np.ndarray]) -> np.ndarray:
    target_log = np.log1p(np.maximum(prediction["target_mm"], 0.0))
    return np.abs(target_log - prediction["mean_log"]) / prediction["std_log"].clip(1e-6)


def day_cluster_conformal_quantile(
    prediction: dict[str, np.ndarray],
    within_day_coverage: float = 0.90,
    day_coverage: float = 0.90,
) -> float:
    scores = normalized_log_scores(prediction)
    daily_quantiles = [
        conformal_quantile(scores[prediction["date_ns"] == date], within_day_coverage)
        for date in np.unique(prediction["date_ns"])
    ]
    return conformal_quantile(np.asarray(daily_quantiles), day_coverage)


def predictive_median_mm(mean_log: np.ndarray) -> np.ndarray:
    """Primary point estimate: inverse transform of the predicted log1p mean."""
    return np.maximum(np.expm1(np.clip(mean_log, -20.0, 10.0)), 0.0)


def predictive_lognormal_mean_mm(mean_log: np.ndarray, std_log: np.ndarray) -> np.ndarray:
    """Sensitivity estimate: mean of the implied shifted-lognormal distribution."""
    exponent = np.clip(mean_log + 0.5 * std_log**2, -20.0, 10.0)
    return np.maximum(np.expm1(exponent), 0.0)


predictive_mean_mm = predictive_lognormal_mean_mm


def transformed_interval_mm(
    mean_log: np.ndarray,
    std_log: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(np.expm1(np.clip(mean_log - q * std_log, -20.0, 10.0)), 0.0)
    upper = np.maximum(np.expm1(np.clip(mean_log + q * std_log, -20.0, 10.0)), 0.0)
    return lower, upper


def wet_probability(mean_log: np.ndarray, std_log: np.ndarray) -> np.ndarray:
    threshold = np.log1p(WET_DAY_THRESHOLD_MM)
    return 1.0 - ndtr((threshold - mean_log) / std_log.clip(1e-6))


def _region_daily_means(
    era5_path: Path,
    variable: str,
    region: str,
    cutoff: float,
    split: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    start, end = SPLITS[split]
    with xr.open_dataset(era5_path) as ds:
        field = ds[variable].sel(time=slice(start, end))
        selected = np.flatnonzero(region_mask(field.longitude.values, region, cutoff))
        means = field.isel(longitude=selected).mean(dim=("latitude", "longitude"), skipna=True).load().values
        dates = pd.DatetimeIndex(field.time.values).normalize()
    return dates, np.asarray(means, dtype="float64")


def _calendar_climatology(dates: pd.DatetimeIndex, values: np.ndarray, half_window_days: int) -> np.ndarray:
    day = dates.dayofyear.to_numpy()
    climatology = np.empty(366, dtype="float64")
    for target in range(1, 367):
        distance = np.abs(day - target)
        distance = np.minimum(distance, 366 - distance)
        climatology[target - 1] = np.nanmean(values[distance <= half_window_days])
    return climatology


@dataclass(frozen=True)
class HeavyPrecipitationDefinition:
    selected_test_dates_ns: frozenset[int]
    high_anomaly_mm: float
    tail_probability: float
    climatology_half_window_days: int
    target_region: str
    test_days: int
    selected_test_days: int

    def metadata(self) -> dict[str, object]:
        return {
            "definition": "target-region ERA5 daily spatial-mean precipitation anomaly above a training-period calendar-day threshold",
            "reference_period": f"{SPLITS['train'][0]} to {SPLITS['train'][1]}",
            "upper_tail_probability": self.tail_probability,
            "high_anomaly_mm_day": self.high_anomaly_mm,
            "climatology_half_window_days": self.climatology_half_window_days,
            "target_region": self.target_region,
            "test_days": self.test_days,
            "selected_test_days": self.selected_test_days,
        }


def build_heavy_precipitation_definition(
    era5_path: Path,
    variable: str,
    target_region: str,
    cutoff: float,
    tail_probability: float = 0.05,
    climatology_half_window_days: int = 15,
) -> HeavyPrecipitationDefinition:
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be between 0 and 0.5.")
    train_dates, train_means = _region_daily_means(era5_path, variable, target_region, cutoff, "train")
    climatology = _calendar_climatology(train_dates, train_means, climatology_half_window_days)
    train_anomalies = train_means - climatology[train_dates.dayofyear.to_numpy() - 1]
    high = float(np.quantile(train_anomalies, 1.0 - tail_probability))
    test_dates, test_means = _region_daily_means(era5_path, variable, target_region, cutoff, "test")
    test_anomalies = test_means - climatology[test_dates.dayofyear.to_numpy() - 1]
    selected = test_anomalies >= high
    return HeavyPrecipitationDefinition(
        selected_test_dates_ns=frozenset(int(value) for value in test_dates.view("int64")[selected]),
        high_anomaly_mm=high,
        tail_probability=tail_probability,
        climatology_half_window_days=climatology_half_window_days,
        target_region=target_region,
        test_days=len(test_dates),
        selected_test_days=int(selected.sum()),
    )


def subset_mask(
    prediction: dict[str, np.ndarray],
    subset: str,
    heavy: HeavyPrecipitationDefinition,
) -> np.ndarray:
    if subset == "all":
        return np.ones(len(prediction["target_mm"]), dtype=bool)
    if subset == "heavy_precipitation_days":
        return np.fromiter(
            (int(value) in heavy.selected_test_dates_ns for value in prediction["date_ns"]),
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
    for start in range(0, repetitions, 250):
        count = min(250, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _daily_metrics(
    mean_log: np.ndarray,
    std_log: np.ndarray,
    target_mm: np.ndarray,
    point_q: float,
    cluster_q: float,
) -> dict[str, float]:
    prediction_median_mm = predictive_median_mm(mean_log)
    prediction_lognormal_mean_mm = predictive_lognormal_mean_mm(mean_log, std_log)
    median_error = target_mm - prediction_median_mm
    lognormal_mean_error = target_mm - prediction_lognormal_mean_mm
    raw_lognormal_exponent = mean_log + 0.5 * std_log**2
    one_plus_mean_to_median_ratio = np.exp(np.clip(0.5 * std_log**2, 0.0, 20.0))
    target_log = np.log1p(np.maximum(target_mm, 0.0))
    raw_lower, raw_upper = transformed_interval_mm(mean_log, std_log, RAW_90_QUANTILE)
    lower, upper = transformed_interval_mm(mean_log, std_log, point_q)
    cluster_lower, cluster_upper = transformed_interval_mm(mean_log, std_log, cluster_q)
    wet = target_mm >= WET_DAY_THRESHOLD_MM
    probability = wet_probability(mean_log, std_log)
    predicted_wet = probability >= 0.5
    intersection = np.sum(predicted_wet & wet)
    union = np.sum(predicted_wet | wet)
    return {
        "rmse_mm": float(np.sqrt(np.mean(median_error**2))),
        "mae_mm": float(np.mean(np.abs(median_error))),
        "rmse_median_mm": float(np.sqrt(np.mean(median_error**2))),
        "mae_median_mm": float(np.mean(np.abs(median_error))),
        "rmse_lognormal_mean_mm": float(np.sqrt(np.mean(lognormal_mean_error**2))),
        "mae_lognormal_mean_mm": float(np.mean(np.abs(lognormal_mean_error))),
        "gaussian_nll_log1p": float(np.mean(0.5 * ((target_log - mean_log) / std_log.clip(1e-6)) ** 2 + np.log(std_log.clip(1e-6)) + 0.5 * np.log(2 * np.pi))),
        "wet_day_brier": float(np.mean((probability - wet.astype("float64")) ** 2)),
        "wet_day_csi": float(intersection / union) if union else 1.0,
        "observed_wet_fraction": float(wet.mean()),
        "predicted_wet_fraction": float(predicted_wet.mean()),
        "raw_90_coverage": float(np.mean((target_mm >= raw_lower) & (target_mm <= raw_upper))),
        "raw_mean_width_mm": float(np.mean(raw_upper - raw_lower)),
        "conformal_90_coverage": float(np.mean((target_mm >= lower) & (target_mm <= upper))),
        "conformal_mean_width_mm": float(np.mean(upper - lower)),
        "day_cluster_90x90_coverage": float(np.mean((target_mm >= cluster_lower) & (target_mm <= cluster_upper))),
        "day_cluster_90x90_mean_width_mm": float(np.mean(cluster_upper - cluster_lower)),
        "median_predicted_std_log": float(np.median(std_log)),
        "p95_predicted_std_log": float(np.quantile(std_log, 0.95)),
        "p99_predicted_std_log": float(np.quantile(std_log, 0.99)),
        "maximum_predicted_std_log": float(np.max(std_log)),
        "fraction_lognormal_mean_exponent_clipped": float(np.mean((raw_lognormal_exponent < -20.0) | (raw_lognormal_exponent > 10.0))),
        "median_one_plus_mean_to_median_ratio": float(np.median(one_plus_mean_to_median_ratio)),
        "p95_one_plus_mean_to_median_ratio": float(np.quantile(one_plus_mean_to_median_ratio, 0.95)),
    }


def metric_summary(
    prediction: dict[str, np.ndarray],
    point_q: float,
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
        metrics = _daily_metrics(
            prediction["mean_log"][selected],
            prediction["std_log"][selected],
            prediction["target_mm"][selected],
            point_q,
            cluster_q,
        )
        daily.append({
            "date": pd.Timestamp(int(date)).strftime("%Y-%m-%d"),
            "date_ns": int(date),
            "subset": subset,
            "retention": retention,
            "calibration_mode": calibration_mode,
            "context_count": context_lookup[int(date)],
            "n_targets": int(selected.sum()),
            **metrics,
        })
    arrays = {key: np.asarray([row[key] for row in daily], dtype="float64") for key in (
        "rmse_mm",
        "mae_mm",
        "rmse_median_mm",
        "mae_median_mm",
        "rmse_lognormal_mean_mm",
        "mae_lognormal_mean_mm",
        "gaussian_nll_log1p",
        "wet_day_brier",
        "wet_day_csi",
        "conformal_90_coverage",
        "day_cluster_90x90_coverage",
    )}
    rmse_ci = _bootstrap_mean_ci(arrays["rmse_mm"], bootstrap_seed, bootstrap_repetitions)
    mae_ci = _bootstrap_mean_ci(arrays["mae_mm"], bootstrap_seed + 1, bootstrap_repetitions)
    lognormal_rmse_ci = _bootstrap_mean_ci(arrays["rmse_lognormal_mean_mm"], bootstrap_seed + 4, bootstrap_repetitions)
    lognormal_mae_ci = _bootstrap_mean_ci(arrays["mae_lognormal_mean_mm"], bootstrap_seed + 5, bootstrap_repetitions)
    coverage_ci = _bootstrap_mean_ci(arrays["conformal_90_coverage"], bootstrap_seed + 2, bootstrap_repetitions)
    cluster_ci = _bootstrap_mean_ci(arrays["day_cluster_90x90_coverage"], bootstrap_seed + 3, bootstrap_repetitions)
    selected_prediction = {
        key: value[mask] for key, value in prediction.items() if key in ("mean_log", "std_log", "target_mm")
    }
    pooled = _daily_metrics(
        selected_prediction["mean_log"],
        selected_prediction["std_log"],
        selected_prediction["target_mm"],
        point_q,
        cluster_q,
    )
    row = {
        "primary_point_estimate": "inverse_log1p_median",
        "subset": subset,
        "retention": retention,
        "calibration_mode": calibration_mode,
        "conformal_q_log1p": point_q,
        "day_cluster_conformal_q_log1p": cluster_q,
        "context_count_min": int(prediction["context_count"].min()),
        "context_count_median": int(np.median(prediction["context_count"])),
        "context_count_max": int(prediction["context_count"].max()),
        "days": len(daily),
        "n": int(mask.sum()),
        "pooled_rmse_mm": pooled["rmse_mm"],
        "pooled_mae_mm": pooled["mae_mm"],
        "pooled_rmse_median_mm": pooled["rmse_median_mm"],
        "pooled_mae_median_mm": pooled["mae_median_mm"],
        "pooled_rmse_lognormal_mean_mm": pooled["rmse_lognormal_mean_mm"],
        "pooled_mae_lognormal_mean_mm": pooled["mae_lognormal_mean_mm"],
        "day_mean_rmse_mm": float(arrays["rmse_mm"].mean()),
        "day_mean_rmse_ci95_low": rmse_ci[0],
        "day_mean_rmse_ci95_high": rmse_ci[1],
        "day_mean_mae_mm": float(arrays["mae_mm"].mean()),
        "day_mean_mae_ci95_low": mae_ci[0],
        "day_mean_mae_ci95_high": mae_ci[1],
        "day_mean_rmse_median_mm": float(arrays["rmse_median_mm"].mean()),
        "day_mean_rmse_median_ci95_low": rmse_ci[0],
        "day_mean_rmse_median_ci95_high": rmse_ci[1],
        "day_mean_mae_median_mm": float(arrays["mae_median_mm"].mean()),
        "day_mean_mae_median_ci95_low": mae_ci[0],
        "day_mean_mae_median_ci95_high": mae_ci[1],
        "day_mean_rmse_lognormal_mean_mm": float(arrays["rmse_lognormal_mean_mm"].mean()),
        "day_mean_rmse_lognormal_mean_ci95_low": lognormal_rmse_ci[0],
        "day_mean_rmse_lognormal_mean_ci95_high": lognormal_rmse_ci[1],
        "day_mean_mae_lognormal_mean_mm": float(arrays["mae_lognormal_mean_mm"].mean()),
        "day_mean_mae_lognormal_mean_ci95_low": lognormal_mae_ci[0],
        "day_mean_mae_lognormal_mean_ci95_high": lognormal_mae_ci[1],
        "day_mean_gaussian_nll_log1p": float(arrays["gaussian_nll_log1p"].mean()),
        "day_mean_wet_day_brier": float(arrays["wet_day_brier"].mean()),
        "day_mean_wet_day_csi": float(arrays["wet_day_csi"].mean()),
        "observed_wet_fraction": pooled["observed_wet_fraction"],
        "predicted_wet_fraction": pooled["predicted_wet_fraction"],
        "raw_90_coverage": pooled["raw_90_coverage"],
        "raw_mean_width_mm": pooled["raw_mean_width_mm"],
        "conformal_90_coverage": pooled["conformal_90_coverage"],
        "day_mean_conformal_coverage": float(arrays["conformal_90_coverage"].mean()),
        "day_mean_conformal_coverage_ci95_low": coverage_ci[0],
        "day_mean_conformal_coverage_ci95_high": coverage_ci[1],
        "conformal_mean_width_mm": pooled["conformal_mean_width_mm"],
        "day_cluster_90x90_coverage": pooled["day_cluster_90x90_coverage"],
        "day_mean_cluster_coverage": float(arrays["day_cluster_90x90_coverage"].mean()),
        "day_mean_cluster_coverage_ci95_low": cluster_ci[0],
        "day_mean_cluster_coverage_ci95_high": cluster_ci[1],
        "day_fraction_at_least_90pct_covered": float(np.mean(arrays["day_cluster_90x90_coverage"] >= 0.90)),
        "day_cluster_90x90_mean_width_mm": pooled["day_cluster_90x90_mean_width_mm"],
        "median_predicted_std_log": pooled["median_predicted_std_log"],
        "p95_predicted_std_log": pooled["p95_predicted_std_log"],
        "p99_predicted_std_log": pooled["p99_predicted_std_log"],
        "maximum_predicted_std_log": pooled["maximum_predicted_std_log"],
        "fraction_lognormal_mean_exponent_clipped": pooled["fraction_lognormal_mean_exponent_clipped"],
        "median_one_plus_mean_to_median_ratio": pooled["median_one_plus_mean_to_median_ratio"],
        "p95_one_plus_mean_to_median_ratio": pooled["p95_one_plus_mean_to_median_ratio"],
    }
    return row, daily
