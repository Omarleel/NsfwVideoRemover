#!/usr/bin/env sh
set -eu
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
else
    PYTHON=python3
fi
"$PYTHON" -m venv .venv
. .venv/bin/activate
python instalar.py --detector falconsai --auto
