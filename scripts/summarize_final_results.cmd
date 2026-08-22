@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe summarize_final_results.py --input-root final_results --output-dir final_results/summary

