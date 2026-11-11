@echo off
rem Demo mode: a simulated live market (no API keys or market hours needed). Stop with stop.bat.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  call "%~dp0setup.bat"
  if errorlevel 1 exit /b 1
)
start "" /b ".venv\Scripts\pythonw.exe" -m forecaster.launcher --demo --open
echo Demo starting: the app opens in your browser in about a minute (first time: the lab runs once).
echo Log in with your dashboard password. Stop with stop.bat.
timeout /t 8 >nul
