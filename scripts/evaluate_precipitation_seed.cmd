@echo off
cd /d "%~dp0.."
set "SEED=%~1"
set "MODEL=%~2"
set "GPU=%~3"
set "ROOT=experiments_precipitation\seed_%SEED%"
set "OUT=final_results_precipitation\neural\seed_%SEED%"
if not exist "%ROOT%\global\%MODEL%\best.pt" exit /b 1
if not exist "%ROOT%\source\%MODEL%\best.pt" exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
set "CUDA_VISIBLE_DEVICES=%GPU%"
.venv\Scripts\python.exe evaluate_precipitation.py --checkpoint "%ROOT%/global/%MODEL%/best.pt" --calibration-region global --test-region global --calibration-label global --device cuda --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc --output "%OUT%/%MODEL%_global.csv"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe evaluate_precipitation.py --checkpoint "%ROOT%/source/%MODEL%/best.pt" --calibration-region source --test-region target --calibration-label transferred --device cuda --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc --output "%OUT%/%MODEL%_transfer.csv"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe evaluate_precipitation.py --checkpoint "%ROOT%/source/%MODEL%/best.pt" --calibration-region target --test-region target --calibration-label oracle_target --device cuda --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc --output "%OUT%/%MODEL%_oracle.csv"
if errorlevel 1 exit /b 1
echo Finished precipitation %MODEL% seed %SEED%.

