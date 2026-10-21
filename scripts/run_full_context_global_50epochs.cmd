@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" exit /b 1
start "Full Global Latent Transformer - RTX 4080" cmd /k "set CUDA_VISIBLE_DEVICES=0&& .venv\Scripts\python.exe train_full_context.py --model latent_transformer --experiment global --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
start "Full Global ConvCNP - RTX 5060 Ti" cmd /k "set CUDA_VISIBLE_DEVICES=1&& .venv\Scripts\python.exe train_full_context.py --model convcnp --experiment global --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
echo Started both all-station global experiments.
