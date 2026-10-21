@echo off
setlocal
cd /d "%~dp0.."
set "OUT=final_results_precipitation_point_sensitivity\kriging"
if not exist "%OUT%" mkdir "%OUT%"

call :run global global global global
if errorlevel 1 exit /b 1
call :run transfer source target transferred
if errorlevel 1 exit /b 1
call :run oracle source target oracle_target
if errorlevel 1 exit /b 1
echo Finished point-sensitivity precipitation kriging evaluations.
exit /b 0

:run
set "NAME=%~1"
set "EXPERIMENT=%~2"
set "TEST_REGION=%~3"
set "CAL_LABEL=%~4"
if "%NAME%"=="oracle" (set "CAL_REGION=target") else (set "CAL_REGION=%EXPERIMENT%")
set "RESULT=%OUT%\ordinary_kriging_%NAME%.csv"
if exist "%RESULT%" if exist "%RESULT:.csv=.json%" if exist "%RESULT:.csv=.daily.csv%" (
  echo Skipping complete kriging %NAME%.
  exit /b 0
)
.venv\Scripts\python.exe evaluate_precipitation_kriging_point_sensitivity.py --experiment %EXPERIMENT% --calibration-region %CAL_REGION% --test-region %TEST_REGION% --calibration-label %CAL_LABEL% --stations data/rr/daily_precipitation_2010_2021.parquet --era5 data/era5_daily_tp_mm_2010_2021.nc --output "%RESULT%"
exit /b %errorlevel%
