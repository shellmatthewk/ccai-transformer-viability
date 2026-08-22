@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" exit /b 1
start "Five More Transformer Seeds - RTX 4080" cmd /k "scripts\train_five_more_seeds_model.cmd latent_transformer 0"
start "Five More ConvCNP Seeds - RTX 5060 Ti" cmd /k "scripts\train_five_more_seeds_model.cmd convcnp 1"
echo Started seeds 2029 through 2033. Existing seeds 2026 through 2028 are preserved.

