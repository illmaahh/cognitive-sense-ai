@echo off
title CognitiveLoad AI — Live Multimodal Research Console

cd /d "%~dp0.."

echo.
echo ============================================================
echo   CognitiveLoad AI — Live Multimodal Research Console
echo ============================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found.
    echo Install Python 3.11+ and try again.
    pause
    exit /b 1
)

echo [1/2] Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing project dependencies...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo [2/2] Starting live server...
echo.
echo Open: http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python -m uvicorn live_cbas.server:app --host 0.0.0.0 --port 8000

pause
