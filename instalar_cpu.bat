@echo off
setlocal
py -3.11 -c "import sys" >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m venv .venv
) else (
    py -m venv .venv
)
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1
python instalar.py --cpu
