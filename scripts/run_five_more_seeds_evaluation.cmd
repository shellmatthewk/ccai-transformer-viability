@echo off
cd /d "%~dp0.."
start "Five More Transformer Evaluations - RTX 4080" cmd /k "scripts\evaluate_five_more_neural_seeds.cmd latent_transformer 0"
start "Five More ConvCNP Evaluations - RTX 5060 Ti" cmd /k "scripts\evaluate_five_more_neural_seeds.cmd convcnp 1"
echo Started corrected evaluation for seeds 2029 through 2033.

