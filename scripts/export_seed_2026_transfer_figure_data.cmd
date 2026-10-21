@echo off
setlocal
cd /d "%~dp0.."

.venv\Scripts\python.exe export_figure_predictions.py --variable temperature --seed 2026 --deployment transfer --device cuda
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe export_figure_predictions.py --variable precipitation --seed 2026 --deployment transfer --device cuda
if errorlevel 1 exit /b 1

echo Finished seed-2026 transfer figure-data exports.
