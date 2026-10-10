@echo off
setlocal
cd /d "%~dp0\.."

echo [1/4] Checking Python 3.10...
where py >nul 2>nul
if errorlevel 1 goto :no_python
py -3.10 -c "import sys; print(sys.version)"
if errorlevel 1 goto :no_python

echo [2/4] Creating isolated environment...
if not exist ".venv-windows\Scripts\python.exe" py -3.10 -m venv .venv-windows
if errorlevel 1 goto :failed

echo [3/4] Installing GraXpert and test dependencies...
".venv-windows\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv-windows\Scripts\python.exe" -m pip install -r requirements.txt pytest
if errorlevel 1 goto :failed

echo [4/4] Verifying the checked-in ONNX model...
".venv-windows\Scripts\python.exe" -c "from graxpert.sample_free_onnx import MODEL_PATH,get_sample_free_session; print(MODEL_PATH); print(get_sample_free_session().get_providers())"
if errorlevel 1 goto :failed

echo.
echo Windows test environment is ready.
echo Start the GUI with tools\start_windows_test.cmd
exit /b 0

:no_python
echo.
echo Python 3.10 64-bit was not found.
echo Install it from python.org, including Tcl/Tk, then run this file again.
exit /b 1

:failed
echo.
echo Setup failed. Copy the complete output and send it to the GraXpert team.
exit /b 1
