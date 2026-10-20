@echo off
setlocal
cd /d "%~dp0.."

for %%V in (temperature precipitation) do (
  for %%D in (global transfer oracle_target) do (
    .venv\Scripts\python.exe export_figure_predictions.py --variable %%V --seed 2026 --deployment %%D --device cuda --overwrite
    if errorlevel 1 exit /b 1
  )
)

echo Finished seed-2026 global, transferred, and oracle-target figure-data exports.
