"""Create or resume an hourly-to-daily ERA5 NetCDF target without xarray append."""

from __future__ import annotations

import argparse
from pathlib import Path

from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
import xarray as xr


def append_daily(output_path: Path, variable: str, daily: xr.Dataset) -> int:
    """Append new daily records, preserving the first xarray-written file schema."""
    if not output_path.exists():
        daily.to_netcdf(
            output_path,
            mode="w",
            engine="netcdf4",
            unlimited_dims=["time"],
            encoding={variable: {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        return daily.sizes["time"]
    with Dataset(output_path, mode="a") as output:
        time_out = output.variables["time"]
        field_out = output.variables[variable]
        units = time_out.units
        calendar = getattr(time_out, "calendar", "standard")
        numeric_time = date2num(pd.to_datetime(daily.time.values).to_pydatetime(), units=units, calendar=calendar)
        last = float(time_out[-1]) if len(time_out) else -np.inf
        keep = numeric_time > last
        count = int(keep.sum())
        if count:
            start = len(time_out)
            time_out[start:start + count] = numeric_time[keep]
            field_out[start:start + count] = daily[variable].values[keep].astype("float32")
        return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.nc")
    args = parser.parse_args()
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No {args.pattern!r} files in {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        with Dataset(args.output) as output:
            if args.variable not in output.variables or "time" not in output.variables:
                raise ValueError(f"Existing {args.output} is not a daily {args.variable} target.")
            print(f"resuming with {len(output.variables['time'])} daily records")
    for index, path in enumerate(files, start=1):
        with xr.open_dataset(path) as raw:
            if args.variable not in raw:
                raise KeyError(f"{args.variable!r} missing from {path}; found {list(raw.data_vars)}")
            time_name = "valid_time" if "valid_time" in raw.coords else "time"
            daily = raw[[args.variable]].resample({time_name: "1D"}).mean().rename({time_name: "time"}).load()
        added = append_daily(args.output, args.variable, daily)
        print(f"[{index}/{len(files)}] {path.name}: added {added} days")


if __name__ == "__main__":
    main()
