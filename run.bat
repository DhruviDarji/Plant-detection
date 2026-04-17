@echo off
echo ==========================================
echo  Verdana Backend Setup (Windows)
echo ==========================================

cd /d "%~dp0"

IF NOT EXIST "venv" (
    IF NOT EXIST ".venv" (
        echo [1/3] Creating virtual environment...
        python -m venv .venv
    )
)

IF EXIST "venv" (
    set VENV_DIR=venv
) ELSE (
    set VENV_DIR=.venv
)

echo [2/3] Activating virtual environment...
call %VENV_DIR%\Scripts\activate

echo [3/3] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo ==========================================
echo Starting server on http://localhost:5000
echo ==========================================
echo.
python app.py
pause
