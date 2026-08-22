@echo off
cd /d "%~dp0.."
set "ROOT=%~1"
set "SEED=%~2"
set "MODEL=%~3"
set "GPU=%~4"
set "OUT=final_results\neural\seed_%SEED%"
if not exist "%ROOT%\global\%MODEL%\best.pt" (
  echo Missing %ROOT%\global\%MODEL%\best.pt
  exit /b 1
)
if not exist "%ROOT%\source\%MODEL%\best.pt" (
  echo Missing %ROOT%\source\%MODEL%\best.pt
  exit /b 1
)
if not exist "%OUT%" mkdir "%OUT%"
set "CUDA_VISIBLE_DEVICES=%GPU%"
.venv\Scripts\python.exe evaluate_full_context_v2.py --checkpoint "%ROOT%/global/%MODEL%/best.pt" --calibration-region global --test-region global --calibration-label global --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output "%OUT%/%MODEL%_global.csv"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe evaluate_full_context_v2.py --checkpoint "%ROOT%/source/%MODEL%/best.pt" --calibration-region source --test-region target --calibration-label transferred --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output "%OUT%/%MODEL%_transfer.csv"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe evaluate_full_context_v2.py --checkpoint "%ROOT%/source/%MODEL%/best.pt" --calibration-region target --test-region target --calibration-label oracle_target --device cuda --stations data/tg/daily_temperature_2010_2021.parquet --era5 data/era5_daily_t2m_2010_2021.nc --output "%OUT%/%MODEL%_oracle.csv"
if errorlevel 1 exit /b 1
echo Finished %MODEL% seed %SEED%.

