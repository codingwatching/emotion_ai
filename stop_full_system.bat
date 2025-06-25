@echo off
REM Aura AI - Stop All Services Script for Windows

echo 🛑 Stopping Aura AI Services...
echo ===============================

REM Kill processes on specific ports
echo Stopping Backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo Stopping Frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5173" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

REM Kill any remaining related processes
echo Cleaning up remaining processes...
taskkill /f /im "uvicorn.exe" >nul 2>&1
taskkill /f /im "node.exe" >nul 2>&1

echo.
echo 🛑 All Aura AI services stopped
echo You can now safely restart the system with start_full_system.bat
echo.
pause
