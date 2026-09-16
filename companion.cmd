@echo off
REM One-command launcher for the companion. Double-click or run `companion`.
REM Passes every argument through, so `companion --voice` starts the mic loop.

setlocal
set "HERE=%~dp0"
set "PY=%HERE%orchestrator\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo   The orchestrator environment is missing.
  echo   Run setup first:  powershell -ExecutionPolicy Bypass -File "%HERE%scripts\setup.ps1"
  echo.
  exit /b 1
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

"%PY%" "%HERE%orchestrator\start.py" %*
endlocal
