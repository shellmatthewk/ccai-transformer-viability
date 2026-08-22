"""Compute reproducible ECA&D and ERA5 statistics used by the experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd


def streaming_field_stats(variable, offset: float, chunk_days: int = 31) -> dict[str, float | int]:
    count = 0; total = 0.0; total_square = 0.0
    minimum, maximum = float("inf"), float("-inf")
    missing = 0
    for start in range(0, variable.shape[0], chunk_days):
        values = np.ma.asarray(variable[start:start + chunk_days], dtype="float64") + offset
        missing += int(np.ma.count_masked(values))
        compressed = values.compressed() if np.ma.isMaskedArray(values) else values.reshape(-1)
        finite = compressed[np.isfinite(compressed)]
        count += len(finite); total += float(finite.sum()); total_square += float(np.square(finite).sum())
        if len(finite):
            minimum = min(minimum, float(finite.min())); maximum = max(maximum, float(finite.max()))
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return {"valid_values": count, "missing_values": missing, "mean_c": mean, "std_c": variance ** 0.5, "min_c": minimum, "max_c": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=Path, default=Path("data/tg/daily_temperature_2010_2021.parquet"))
    parser.add_argument("--era5", type=Path, default=Path("data/era5_daily_t2m_2010_2021.nc"))
    parser.add_argument("--output", type=Path, default=Path("dataset_stats.json"))
    args = parser.parse_args()
    frame = pd.read_parquet(args.stations, columns=["station_id", "source_id", "date", "tg", "latitude", "longitude"])
    daily = frame.groupby("date").size()
    station_stats = {
        "rows": len(frame),
        "unique_station_ids": int(frame.station_id.nunique()),
        "unique_source_ids": int(frame.source_id.nunique()),
        "unique_coordinates": int(len(frame[["latitude", "longitude"]].drop_duplicates())),
        "start_date": str(pd.Timestamp(frame.date.min()).date()),
        "end_date": str(pd.Timestamp(frame.date.max()).date()),
        "days": int(frame.date.nunique()),
        "observations_per_day_min": int(daily.min()),
        "observations_per_day_median": int(daily.median()),
        "observations_per_day_max": int(daily.max()),
        "latitude_min": float(frame.latitude.min()),
        "latitude_max": float(frame.latitude.max()),
        "longitude_min": float(frame.longitude.min()),
        "longitude_max": float(frame.longitude.max()),
        "temperature_mean_c": float(frame.tg.mean()),
        "temperature_std_c": float(frame.tg.std()),
        "temperature_min_c": float(frame.tg.min()),
        "temperature_max_c": float(frame.tg.max()),
    }
    with Dataset(args.era5) as dataset:
        field = dataset.variables["t2m"]
        latitude = np.asarray(dataset.variables["latitude"][:]); longitude = np.asarray(dataset.variables["longitude"][:])
        time = dataset.variables["time"]
        dates = num2date(time[:], units=time.units, calendar=getattr(time, "calendar", "standard"), only_use_cftime_datetimes=False)
        field_stats = streaming_field_stats(field, -273.15)
        era5_stats = {
            "variable": "2 metre temperature (t2m)",
            "source_units": getattr(field, "units", "K"),
            "analysis_units": "degrees C",
            "temporal_processing": "hourly values aggregated to daily means",
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
            "daily_fields_in_file": len(dates),
            "experiment_daily_fields_2010_2021": int(sum(date.year <= 2021 for date in dates)),
            "latitude_points": len(latitude),
            "longitude_points": len(longitude),
            "grid_cells_per_day": len(latitude) * len(longitude),
            "latitude_min": float(latitude.min()),
            "latitude_max": float(latitude.max()),
            "longitude_min": float(longitude.min()),
            "longitude_max": float(longitude.max()),
            "latitude_resolution_degrees": float(abs(np.diff(latitude).mean())),
            "longitude_resolution_degrees": float(abs(np.diff(longitude).mean())),
            **field_stats,
        }
    report = {"ecad_daily_temperature": station_stats, "era5_daily_t2m": era5_stats}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
