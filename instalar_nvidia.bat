@echo off
setlocal

rem Si Conda o un venv ya están activos, usa ese Python y no crea otro entorno.
if defined CONDA_PREFIX goto use_current
if defined VIRTUAL_ENV goto use_current

py -3.11 -c "import sys" >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m venv .venv
) else (
    py -m venv .venv
)
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

:use_current
python instalar.py --nvidia
exit /b %errorlevel%
