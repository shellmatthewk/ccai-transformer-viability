@echo off
cd /d "%~dp0.."
start "Full Source Latent Transformer - RTX 4080" cmd /k "set CUDA_VISIBLE_DEVICES=0&& .venv\Scripts\python.exe train_full_context.py --model latent_transformer --experiment source --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --longitude-cutoff 15 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
start "Full Source ConvCNP - RTX 5060 Ti" cmd /k "set CUDA_VISIBLE_DEVICES=1&& .venv\Scripts\python.exe train_full_context.py --model convcnp --experiment source --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --longitude-cutoff 15 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc"
echo Started both all-station western/source experiments.
