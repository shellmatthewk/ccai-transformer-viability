@echo off
setlocal
cd /d "%~dp0.."
.venv\Scripts\python.exe summarize_precipitation_point_sensitivity.py
exit /b %errorlevel%
