@echo off
rem Double-click once to install stainID (needs Python 3.10 or newer from python.org; reading .vsi slides also needs Java).
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[app,deep,slides]"
echo stainID is installed. Double-click "Start stainID.bat" to open it.
pause
