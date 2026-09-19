@echo off
REM ============================================================
REM  CivicProof - double-click this file to run (Windows)
REM ============================================================
cd /d %~dp0
echo.
echo  CivicProof is starting... (first run may take 1-2 minutes)
echo.
where python >nul 2>nul
if errorlevel 1 (
    echo  ERROR: Python is not installed.
    echo  Get it from https://www.python.org/downloads/ and tick "Add to PATH".
    pause
    exit /b 1
)
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo.
echo  Open this in your browser:  http://localhost:8000
echo.
python app.py
pause
