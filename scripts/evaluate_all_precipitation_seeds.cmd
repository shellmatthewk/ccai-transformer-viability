@echo off
cd /d "%~dp0.."
set "MODEL=%~1"
set "GPU=%~2"
for %%S in (2026 2027 2028 2029 2030 2031 2032 2033) do (
  call scripts\evaluate_precipitation_seed.cmd %%S %MODEL% %GPU%
  if errorlevel 1 exit /b 1
)

