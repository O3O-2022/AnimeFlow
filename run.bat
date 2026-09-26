@echo off
rem AnimeFlow 실행 (콘솔 창 없이 GUI만 뜹니다)
rem 이 파일은 CP949(한국어 Windows 기본)로 저장되어 있습니다. UTF-8로 저장하면 한글이 깨집니다.
set "VENV=%USERPROFILE%\.animeflow\venv"
if not exist "%VENV%\Scripts\pythonw.exe" (
    echo [AnimeFlow] 실행 환경이 없습니다. setup.bat 을 먼저 실행하세요.
    pause
    exit /b 1
)
cd /d "%~dp0"
start "" "%VENV%\Scripts\pythonw.exe" -m animeflow
