@echo off
cd /d "%~dp0.."
set "MODEL=%~1"
set "GPU=%~2"
if "%MODEL%"=="" exit /b 2
if not exist "data\era5_daily_tp_mm_2010_2021.nc" (
  echo Missing daily precipitation target. Run scripts\prepare_precipitation_targets.cmd first.
  exit /b 1
)
set "CUDA_VISIBLE_DEVICES=%GPU%"
for %%S in (2026 2027 2028 2029 2030 2031 2032 2033) do (
  for %%R in (global source) do (
    if exist "experiments_precipitation\seed_%%S\%%R\%MODEL%\best.pt" (
      echo Skipping existing %MODEL% seed %%S region %%R
    ) else (
      echo Training precipitation %MODEL% seed %%S region %%R on physical GPU %GPU%
      .venv\Scripts\python.exe train_precipitation.py --model %MODEL% --experiment %%R --seed %%S --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --device cuda --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc
      if errorlevel 1 exit /b 1
    )
  )
)
echo Finished precipitation %MODEL% seeds 2026 through 2033.

