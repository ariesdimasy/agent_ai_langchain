@echo off
setlocal

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0backend\.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0backend\.venv\Scripts\python.exe"
) else (
    echo Python virtual environment was not found.
    echo Create it and install backend\requirements.txt first.
    pause
    exit /b 1
)

if not exist "%~dp0backend\.env" (
    echo backend\.env was not found.
    echo Copy backend\.env.example to backend\.env and set DEEPSEEK_API_KEY.
    pause
    exit /b 1
)

if not exist "%~dp0frontend\node_modules" (
    echo Frontend dependencies were not found.
    echo Run npm install in the frontend folder first.
    pause
    exit /b 1
)

where node.exe >nul 2>&1
if errorlevel 1 (
    echo Node.js was not found on PATH. Install Node.js 18 or newer first.
    pause
    exit /b 1
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo npm was not found on PATH. Reinstall Node.js or add npm to PATH.
    pause
    exit /b 1
)

echo Starting the Rangkumin backend and frontend...
start "Rangkumin - Backend" /D "%~dp0backend" "%PYTHON%" -m uvicorn main:app --reload
start "Rangkumin - Frontend" /D "%~dp0frontend" "%ComSpec%" /k "npm run dev -- --host 127.0.0.1"

echo.
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Close the two terminal windows to stop the services.
timeout /t 3 /nobreak >nul
start "" "http://localhost:5173"

endlocal
