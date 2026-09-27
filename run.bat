@echo off
rem AnimeFlow launcher (opens the GUI without a console window)
rem Messages are ASCII-only so this file works on any Windows language/codepage.
set "VENV=%USERPROFILE%\.animeflow\venv"
if not exist "%VENV%\Scripts\pythonw.exe" (
    echo [AnimeFlow] Not installed yet. Run setup.bat first.
    pause
    exit /b 1
)
cd /d "%~dp0"
start "" "%VENV%\Scripts\pythonw.exe" -m animeflow
