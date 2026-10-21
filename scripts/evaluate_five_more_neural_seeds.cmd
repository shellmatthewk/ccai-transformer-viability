@echo off
cd /d "%~dp0.."
set "MODEL=%~1"
set "GPU=%~2"
for %%S in (2029 2030 2031 2032 2033) do (
  call scripts\evaluate_neural_seed.cmd experiments_multiseed\seed_%%S %%S %MODEL% %GPU%
  if errorlevel 1 exit /b 1
)
echo Finished corrected evaluation for %MODEL% seeds 2029 through 2033.

