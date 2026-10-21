@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe summarize_precipitation_results.py --input-root final_results_precipitation --output-dir final_results_precipitation/summary

