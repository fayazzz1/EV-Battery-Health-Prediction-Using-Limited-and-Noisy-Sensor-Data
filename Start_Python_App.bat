@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto error
  .venv\Scripts\python.exe -m pip install -r python_app\requirements.txt
  if errorlevel 1 goto error
)
.venv\Scripts\python.exe -m streamlit run python_app\app.py
goto end
:error
echo Setup failed. Install Python 3.12 and check your internet connection.
:end
pause
