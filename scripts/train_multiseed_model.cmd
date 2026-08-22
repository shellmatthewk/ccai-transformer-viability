@echo off
cd /d "%~dp0.."
set "MODEL=%~1"
set "GPU=%~2"
if "%MODEL%"=="" exit /b 2
set "CUDA_VISIBLE_DEVICES=%GPU%"
for %%S in (2027 2028) do (
  for %%R in (global source) do (
    echo Training %MODEL% seed %%S region %%R on physical GPU %GPU%
    .venv\Scripts\python.exe train_full_context.py --model %MODEL% --experiment %%R --seed %%S --epochs 50 --min-epochs 10 --patience 8 --batch-size 1 --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output-root experiments_multiseed/seed_%%S
    if errorlevel 1 exit /b 1
  )
)
echo Finished %MODEL% seeds 2027 and 2028.

