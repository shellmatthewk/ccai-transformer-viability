@echo off
cd /d "%~dp0.."
start "Kriging Global" cmd /k ".venv\Scripts\python.exe kriging_experiment.py --experiment global --calibration-region global --test-region global --calibration-label global --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output experiments/results/kriging_global_sparse.csv"
start "Kriging Transfer" cmd /k ".venv\Scripts\python.exe kriging_experiment.py --experiment source --calibration-region source --test-region target --calibration-label transferred --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output experiments/results/kriging_transfer_sparse.csv"
start "Kriging Oracle" cmd /k ".venv\Scripts\python.exe kriging_experiment.py --experiment source --calibration-region target --test-region target --calibration-label oracle_target --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output experiments/results/kriging_oracle_sparse.csv"
echo Started validation-tuned ordinary kriging evaluations. These are CPU jobs.
