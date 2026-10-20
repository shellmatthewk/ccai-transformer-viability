@echo off
setlocal
cd /d "%~dp0.."
set "SEED=%~1"
set "MODEL=%~2"
set "GPU=%~3"
set "ROOT=experiments_precipitation\seed_%SEED%"
set "OUT=final_results_precipitation_point_sensitivity\neural\seed_%SEED%"

if "%SEED%"=="" (
  echo Usage: %~nx0 SEED MODEL GPU
  exit /b 2
)
if "%MODEL%"=="" (
  echo Usage: %~nx0 SEED MODEL GPU
  exit /b 2
)
if "%GPU%"=="" (
  echo Usage: %~nx0 SEED MODEL GPU
  exit /b 2
)
if not exist "%ROOT%\global\%MODEL%\best.pt" exit /b 1
if not exist "%ROOT%\source\%MODEL%\best.pt" exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
set "CUDA_VISIBLE_DEVICES=%GPU%"

call :run global "%ROOT%/global/%MODEL%/best.pt" global global global
if errorlevel 1 exit /b 1
call :run transfer "%ROOT%/source/%MODEL%/best.pt" source target transferred
if errorlevel 1 exit /b 1
call :run oracle "%ROOT%/source/%MODEL%/best.pt" target target oracle_target
if errorlevel 1 exit /b 1
echo Finished point-sensitivity precipitation %MODEL% seed %SEED%.
exit /b 0

:run
set "NAME=%~1"
set "CHECKPOINT=%~2"
set "CAL_REGION=%~3"
set "TEST_REGION=%~4"
set "CAL_LABEL=%~5"
set "RESULT=%OUT%\%MODEL%_%NAME%.csv"
if exist "%RESULT%" if exist "%RESULT:.csv=.json%" if exist "%RESULT:.csv=.daily.csv%" (
  echo Skipping complete %MODEL% seed %SEED% %NAME%.
  exit /b 0
)
.venv\Scripts\python.exe evaluate_precipitation_point_sensitivity.py --checkpoint "%CHECKPOINT%" --calibration-region %CAL_REGION% --test-region %TEST_REGION% --calibration-label %CAL_LABEL% --device cuda --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc --output "%RESULT%"
exit /b %errorlevel%
