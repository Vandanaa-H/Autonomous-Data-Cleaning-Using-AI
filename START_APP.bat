@echo off
echo ============================================
echo   Starting Data Cleaning Application
echo ============================================
echo.

REM Kill any existing processes
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [1/2] Starting Backend on port 8003...
cd /d "%~dp0backend"
start "Backend API (Port 8003)" cmd /k "python -m uvicorn autonomous_api:app --host 0.0.0.0 --port 8003"
timeout /t 5 /nobreak >nul

echo [2/2] Starting Frontend on port 8502...
cd /d "%~dp0frontend"
start "Frontend UI (Port 8502)" cmd /k "streamlit run final_professional_app.py --server.port 8502"
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo   Backend API:  http://localhost:8003
echo   Frontend UI:  http://localhost:8502
echo.
echo   Open your browser and go to:
echo   http://localhost:8502
echo.
echo   Press any key to open browser...
pause >nul
start http://localhost:8502
