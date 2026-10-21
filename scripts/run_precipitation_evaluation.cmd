@echo off
cd /d "%~dp0.."
start "Precipitation Transformer Evaluation - RTX 4080" cmd /k "scripts\evaluate_all_precipitation_seeds.cmd latent_transformer 0"
start "Precipitation ConvCNP Evaluation - RTX 5060 Ti" cmd /k "scripts\evaluate_all_precipitation_seeds.cmd convcnp 1"
echo Started corrected precipitation evaluation for seeds 2026 through 2033.

