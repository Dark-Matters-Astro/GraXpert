@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv-windows\Scripts\python.exe" (
    echo The Windows test environment is missing.
    echo Run tools\setup_windows_test.cmd first.
    exit /b 1
)

".venv-windows\Scripts\python.exe" tools\run_sample_free_gui.py
