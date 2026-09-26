@echo off
rem AnimeFlow 설치: 파이썬 가상환경 + 라이브러리 (PyTorch CUDA 포함, 약 5GB)
rem 이 파일은 CP949(한국어 Windows 기본)로 저장되어 있습니다.
rem 가상환경은 OneDrive 동기화를 피하려고 사용자 폴더(%USERPROFILE%\.animeflow)에 만듭니다.
set "VENV=%USERPROFILE%\.animeflow\venv"
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3.12"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
    echo Python 3.12 를 찾을 수 없습니다. https://www.python.org/downloads/ 에서 설치하세요.
    pause
    exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
    echo [1/3] 가상환경 만드는 중...
    %PY% -m venv "%VENV%" || goto :fail
)
echo [2/3] 기본 라이브러리 설치 중...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip -q || goto :fail
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt -q || goto :fail
echo [3/3] PyTorch (NVIDIA CUDA) 설치 중... 시간이 걸립니다.
"%VENV%\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q || goto :fail

if not exist "tools\ffmpeg\ffmpeg.exe" echo [주의] tools\ffmpeg\ffmpeg.exe 가 없습니다. README 참고.
if not exist "third_party\Practical-RIFE\train_log\flownet.pkl" echo [주의] RIFE 모델이 없습니다. README 참고.
echo.
echo 설치 완료! run.bat 으로 실행하세요.
pause
exit /b 0

:fail
echo 설치 중 오류가 발생했습니다.
pause
exit /b 1
