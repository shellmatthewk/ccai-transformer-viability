"""Aggregate monthly hourly ERA5 files into one daily reference-target NetCDF.

Example:
  python prepare_daily_era5.py --input-dir data/rr_era5/era5_hourly_tg --variable t2m \
      --output data/era5_daily_t2m_2010_2021.nc

For temperature this creates daily mean 2 m temperature in kelvin. Pass
``--era5-offset -273.15`` to train_models.py when pairing it with the station
``tg`` values, which are degrees C. This conversion is intentionally deferred
to training so the derived ERA5 file retains its published units and provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


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
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived target: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, start=1):
        with xr.open_dataset(path) as raw:
            if args.variable not in raw:
                raise KeyError(f"{args.variable!r} missing from {path}; found {list(raw.data_vars)}")
            time_name = "valid_time" if "valid_time" in raw.coords else "time"
            daily = raw[[args.variable]].resample({time_name: "1D"}).mean().rename({time_name: "time"})
            encoding = {args.variable: {"zlib": True, "complevel": 4, "dtype": "float32"}}
            if index == 1:
                daily.to_netcdf(args.output, mode="w", engine="netcdf4", unlimited_dims=["time"], encoding=encoding)
            else:
                daily.to_netcdf(args.output, mode="a", engine="netcdf4", append_dim="time", encoding=encoding)
        print(f"[{index}/{len(files)}] appended {path.name}")


if __name__ == "__main__":
    main()
