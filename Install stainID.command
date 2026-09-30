#!/bin/bash
# Double-click once to install stainID (needs Python 3.10 or newer; reading .vsi slides also needs Java).
cd "$(dirname "$0")"
set -e
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "stainID needs Python 3.10 or newer: https://www.python.org/downloads/"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[app,deep,slides]"
echo
echo "stainID is installed. Double-click 'Start stainID' to open it."
read -r -p "Press Enter to close this window."
