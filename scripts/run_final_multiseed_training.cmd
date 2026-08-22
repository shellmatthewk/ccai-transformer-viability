@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" exit /b 1
start "Multi-seed Latent Transformer - RTX 4080" cmd /k "scripts\train_multiseed_model.cmd latent_transformer 0"
start "Multi-seed ConvCNP - RTX 5060 Ti" cmd /k "scripts\train_multiseed_model.cmd convcnp 1"
echo Started seeds 2027 and 2028. Existing seed 2026 checkpoints are preserved and reused.

