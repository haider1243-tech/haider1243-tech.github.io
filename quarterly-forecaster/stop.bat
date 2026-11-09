@echo off
rem Stop the monitor, webhook, dashboard and tunnel started by start.bat.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m forecaster.launcher --stop
timeout /t 3 >nul
