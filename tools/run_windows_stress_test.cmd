@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv-windows\Scripts\python.exe" (
    echo The Windows test environment is missing.
    echo Run tools\setup_windows_test.cmd first.
    exit /b 1
)

if "%~1"=="" (
    echo Usage: tools\run_windows_stress_test.cmd "C:\path\to\test-image.fit"
    echo You can also drag a FITS file onto this command file.
    exit /b 1
)

".venv-windows\Scripts\python.exe" tools\stress_test_sample_free.py ^
    --image "%~1" ^
    --full-runs 10 ^
    --tk ^
    --report ".local-test-data\onnx-windows-stress-report.json"
if errorlevel 1 goto :failed

echo.
echo Stress test passed.
echo Report: .local-test-data\onnx-windows-stress-report.json
exit /b 0

:failed
echo.
echo Stress test failed. Keep the console output and the GraXpert log.
exit /b 1
