@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (set "PYTHON=python") else (set "PYTHON=py -3")
%PYTHON% -c "import streamlit, plotly, sklearn" >nul 2>nul
if errorlevel 1 (
  echo Installing project dependencies...
  %PYTHON% -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
%PYTHON% -m streamlit run app.py
pause
exit /b
:failed
echo Setup failed. Ensure Python 3.10 or newer is installed.
pause
