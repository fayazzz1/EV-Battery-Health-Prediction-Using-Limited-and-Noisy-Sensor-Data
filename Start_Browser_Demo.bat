@echo off
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  python start_browser.py
) else (
  py -3.12 start_browser.py
)
pause
