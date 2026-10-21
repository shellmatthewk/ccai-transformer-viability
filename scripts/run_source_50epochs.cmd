@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Create the CUDA environment first.
  exit /b 1
)
start "Source Transformer - RTX 4080" cmd /k "set CUDA_VISIBLE_DEVICES=0&& .venv\Scripts\python.exe train_experiment.py --model transformer --experiment source --epochs 50 --min-epochs 10 --patience 8 --longitude-cutoff 15 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
start "Source ConvCNP - RTX 5060 Ti" cmd /k "set CUDA_VISIBLE_DEVICES=1&& .venv\Scripts\python.exe train_experiment.py --model convcnp --experiment source --epochs 50 --min-epochs 10 --patience 8 --longitude-cutoff 15 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
echo Started both western/source experiments in separate windows.
