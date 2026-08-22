@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe summarize_results.py --input-dir experiments_full_context/results --output experiments_full_context/results/all_results.csv
