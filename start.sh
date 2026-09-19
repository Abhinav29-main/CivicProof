#!/usr/bin/env bash
# ============================================================
#  CivicProof — one command to run (macOS / Linux)
#  Usage:  ./start.sh
# ============================================================
cd "$(dirname "$0")"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
command -v $PY >/dev/null 2>&1 || { echo "ERROR: Python 3.10+ not found — install from https://www.python.org/downloads/"; exit 1; }

echo ""
echo " CivicProof is starting... (first run may take 1-2 minutes)"
echo ""

[ -d .venv ] || $PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt

echo ""
echo " Open this in your browser:  http://localhost:8000"
echo ""

python app.py
