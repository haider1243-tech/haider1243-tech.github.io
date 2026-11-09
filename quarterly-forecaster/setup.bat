@echo off
rem One-time setup for Windows. Safe to run again (it skips what's already done).
rem  1. Python virtual environment + packages
rem  2. .env (API keys) and the dashboard password
rem  3. portfolio.yaml from the example
rem  4. Desktop shortcut + start automatically (at sign-in, or at boot)
rem  5. Tailscale: private HTTPS link for your phone (never public)
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo === Quarterly Forecaster setup ===
echo.

rem ---- 1. Python ----------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  set "SYS_PY="
  where py >nul 2>&1 && set "SYS_PY=py -3"
  if not defined SYS_PY ( where python >nul 2>&1 && set "SYS_PY=python" )
  if not defined SYS_PY (
    echo Python 3.11 or newer is needed: install it from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^), then run setup.bat again.
    pause & exit /b 1
  )
  echo Creating the Python environment...
  !SYS_PY! -m venv .venv || ( echo Could not create .venv & pause & exit /b 1 )
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || (
  echo Python 3.11 or newer is needed. Delete the .venv folder, install a newer Python, and run again.
  pause & exit /b 1
)
echo Installing packages ^(first time takes a few minutes^)...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r requirements.txt -q || ( echo Package install failed. & pause & exit /b 1 )
"%PY%" -m pip install --no-deps -e . -q || ( echo Package install failed. & pause & exit /b 1 )

rem ---- 2. .env + password -------------------------------------------------
if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo Created .env. Add your API keys to it later ^(Notepad is fine^).
)
findstr /r /c:"^APP_PASSWORD_HASH=..*" ".env" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Choose a password for the dashboard ^(you'll type it on your phone^).
  "%PY%" -m forecaster.auth --write-env || ( pause & exit /b 1 )
)

rem ---- 3. portfolio --------------------------------------------------------
if not exist "portfolio.yaml" if exist "portfolio.example.yaml" (
  copy /y "portfolio.example.yaml" "portfolio.yaml" >nul
  echo Created portfolio.yaml from the example: edit it with your holdings.
)

rem ---- 4. shortcuts + auto-start ------------------------------------------
echo.
echo Auto-start options:
echo   1 = when you sign in to Windows ^(recommended, no admin needed^)
echo   2 = at PC boot, even before anyone signs in ^(asks for admin + your Windows password^)
echo   3 = no auto-start
set "AUTO=1"
set /p "AUTO=Choose 1, 2 or 3 [1]: "
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\create_shortcuts.ps1" -Mode "%AUTO%"

rem ---- 5. Tailscale (private phone access) --------------------------------
echo.
where tailscale >nul 2>&1
if errorlevel 1 (
  echo Tailscale isn't installed. For phone access install it from https://tailscale.com/download
  echo on this PC and your phone ^(same account^), then run setup.bat again.
) else (
  echo Sharing the dashboard on your private tailnet ^(HTTPS, only your devices; never public^)...
  tailscale serve --bg 8501
  tailscale serve status
  echo Open the https://...ts.net address above on your phone, then "Add to Home Screen".
)

echo.
echo Setup done. Start it now with start.bat ^(or the desktop shortcut^).
pause
exit /b 0
