@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" exit /b 1
start "Precipitation Transformer Seeds - RTX 4080" cmd /k "scripts\train_precipitation_all_seeds_model.cmd latent_transformer 0"
start "Precipitation ConvCNP Seeds - RTX 5060 Ti" cmd /k "scripts\train_precipitation_all_seeds_model.cmd convcnp 1"
echo Started precipitation seeds 2026 through 2033.

