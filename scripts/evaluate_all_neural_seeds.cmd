@echo off
cd /d "%~dp0.."
set "MODEL=%~1"
set "GPU=%~2"
call scripts\evaluate_neural_seed.cmd experiments_full_context 2026 %MODEL% %GPU%
if errorlevel 1 exit /b 1
call scripts\evaluate_neural_seed.cmd experiments_multiseed\seed_2027 2027 %MODEL% %GPU%
if errorlevel 1 exit /b 1
call scripts\evaluate_neural_seed.cmd experiments_multiseed\seed_2028 2028 %MODEL% %GPU%
if errorlevel 1 exit /b 1

