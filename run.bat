@echo off
setlocal
title ResearchFinderAI
cd /d "%~dp0"

echo ==================================
echo       ResearchFinderAI
echo ==================================
echo.

if not exist "backend\venv\Scripts\python.exe" (
    echo ERROR: Backend venv not found.
    pause
    exit /b 1
)

if not exist "frontend\node_modules\vite\bin\vite.js" (
    echo Installing frontend dependencies...
    cd /d "%~dp0frontend"
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed.
        pause
        exit /b 1
    )
)

echo Starting Backend...
start "ResearchFinderAI Backend" cmd /k "cd /d "%~dp0backend" && venv\Scripts\activate && python -m uvicorn research_finder:app --port 8000"

echo Starting Frontend...
start "ResearchFinderAI Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1"

echo Waiting for services...
timeout /t 5 /nobreak >nul

echo Opening ResearchFinderAI in browser...
start "" "http://localhost:5173"

echo.
echo ResearchFinderAI is running.
pause
