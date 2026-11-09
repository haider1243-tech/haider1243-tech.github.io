@echo off
rem Quarterly Forecaster: start monitor + webhook + dashboard together, then open the app.
rem   start.bat          start (first run: runs setup.bat) and open the dashboard
rem   start.bat /boot    start in the background without opening a browser (used at sign-in/boot)
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"

if not exist "%PY%" (
  echo First run: setting things up...
  call "%~dp0setup.bat"
  if errorlevel 1 exit /b 1
)

"%PY%" -m forecaster.launcher --check
if errorlevel 1 (
  echo.
  echo Fix the ERROR lines above ^(run setup.bat^), then start again.
  if /i not "%~1"=="/boot" pause
  exit /b 1
)

if /i "%~1"=="/boot" (
  start "" /b "%PYW%" -m forecaster.launcher
) else (
  start "" /b "%PYW%" -m forecaster.launcher --open
  echo Started. The dashboard opens in your browser in a few seconds: http://127.0.0.1:8501
  echo Logs: %~dp0logs    Stop everything: stop.bat
  timeout /t 5 >nul
)
exit /b 0
