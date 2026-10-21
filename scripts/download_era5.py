"""Download ERA5 hourly t2m/tp for Europe from Copernicus CDS and aggregate to daily.

Setup (one-time):
    pip install cdsapi xarray netCDF4
    Register at https://cds.climate.copernicus.eu/, get your API key from your
    profile page, and save it to ~/.cdsapirc (format documented on the CDS site).
    Accept the license for the "ERA5 hourly data on single levels" dataset on
    its CDS page (registering alone is not enough - the first request fails
    silently/rejects until the license is accepted via the web UI).

Usage:
    python scripts/download_era5.py --year-start 2010 --year-end 2021

Requests are batched by month (not year) to stay under CDS per-request field
limits: 2 variables x ~30 days x 24 hours = ~1440 fields/month vs. ~17,000+
for a full year.

CAVEAT (verify before trusting daily precip totals): ERA5 single-levels
reanalysis 'total_precipitation' is an accumulated field. This script assumes
each hourly value already represents that hour's accumulation (not cumulative
since 00 UTC), so daily total = sum of the 24 hourly values. Confirm this
against the current CDS documentation for reanalysis-era5-single-levels
before relying on RR (precipitation) numbers - accumulation conventions are a
well-known ERA5 gotcha and this has not been independently verified here.
"""

import argparse
import calendar
from pathlib import Path

import cdsapi
import xarray as xr

# Europe bounding box: North, West, South, East (matches CHAPTER_PLAN.md Sec 4.1)
AREA = [72, -25, 35, 45]

VARIABLES = ["2m_temperature", "total_precipitation"]

RAW_DIR = Path("data/raw/era5_hourly")
DAILY_DIR = Path("data/processed/era5_daily")

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def download_month(client: cdsapi.Client, year: int, month: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_{year}{month:02d}.nc"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    print(f"[request] {year}-{month:02d} ({n_days} days)")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": ALL_HOURS,
            "area": AREA,
            "grid": "0.25/0.25",
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"[done] {out_path}")
    return out_path


def to_daily(hourly_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / hourly_path.name.replace("era5_", "era5_daily_")
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path

    ds = xr.open_dataset(hourly_path)
    t2m_daily = ds["t2m"].resample(time="1D").mean()
    tp_daily = ds["tp"].resample(time="1D").sum()  # see accumulation caveat in module docstring
    daily = xr.Dataset({"t2m": t2m_daily, "tp": tp_daily})
    daily.to_netcdf(out_path)
    ds.close()
    print(f"[aggregated] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end", type=int, default=2021)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    args = parser.parse_args()

    client = cdsapi.Client()

    for year in range(args.year_start, args.year_end + 1):
        for month in range(1, 13):
            hourly_path = download_month(client, year, month, args.raw_dir)
            to_daily(hourly_path, args.daily_dir)


if __name__ == "__main__":
    main()
