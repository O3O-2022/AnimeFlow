@echo off
rem AnimeFlow setup: Python venv + libraries (PyTorch CUDA, ~5GB) + AI models + FFmpeg (~350MB)
rem Messages are ASCII-only so this file works on any Windows language/codepage.
rem The venv lives in %USERPROFILE%\.animeflow\venv (outside cloud-synced folders such as OneDrive).
set "VENV=%USERPROFILE%\.animeflow\venv"
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
    echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run setup.bat again.
    pause
    exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    %PY% -m venv "%VENV%" || goto :fail
)
echo [2/4] Installing libraries...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip -q || goto :fail
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt -q || goto :fail
echo [3/4] Installing PyTorch (NVIDIA CUDA). This takes a while...
"%VENV%\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q || goto :fail
echo [4/4] Downloading AI models and FFmpeg (~350MB, skipped if already present)...
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\download_models.ps1" -Python "%VENV%\Scripts\python.exe" || goto :fail
echo.
echo Setup complete! Start AnimeFlow with run.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. Check the messages above, then run setup.bat again (finished steps are skipped).
pause
exit /b 1
