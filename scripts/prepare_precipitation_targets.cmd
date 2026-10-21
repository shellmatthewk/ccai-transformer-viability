@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe prepare_daily_era5_precipitation.py --input-dir data/rr_era5/era5_hourly_rr --variable tp --start 2010-01-01 --end 2021-12-31 --output data/era5_daily_tp_mm_2010_2021.nc

