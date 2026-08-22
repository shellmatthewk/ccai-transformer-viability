@echo off
cd /d "%~dp0.."
start "Final Transformer Evaluation - RTX 4080" cmd /k "scripts\evaluate_all_neural_seeds.cmd latent_transformer 0"
start "Final ConvCNP Evaluation - RTX 5060 Ti" cmd /k "scripts\evaluate_all_neural_seeds.cmd convcnp 1"
echo Started corrected neural evaluation for seeds 2026, 2027, and 2028.

