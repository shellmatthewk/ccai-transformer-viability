@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe summarize_results.py --input-dir experiments/results --output experiments/results/all_results.csv
