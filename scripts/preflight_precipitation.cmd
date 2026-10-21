@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Missing .venv\Scripts\python.exe
  exit /b 1
)
if not exist "data\rr\daily_precipitation_2010_2021.parquet" (
  echo ERROR: Missing ECA^&D precipitation parquet.
  exit /b 1
)
if not exist "data\era5_daily_tp_mm_2010_2021.nc" (
  echo ERROR: Missing daily ERA5 precipitation target.
  exit /b 1
)

.venv\Scripts\python.exe -m py_compile prepare_daily_era5_precipitation.py precipitation_data.py precipitation_protocol.py train_precipitation.py evaluate_precipitation.py evaluate_precipitation_kriging.py summarize_precipitation_results.py
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.device_count() >= 2, f'Expected two CUDA GPUs, found {torch.cuda.device_count()}'; print('torch=' + torch.__version__); [print(f'gpu {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
if errorlevel 1 exit /b 1

echo.
echo Precipitation setup is ready.
echo Daily target: 4383 days, 2010-01-01 through 2021-12-31, mm/day.
echo Tomorrow run: scripts\run_precipitation_training.cmd
exit /b 0
