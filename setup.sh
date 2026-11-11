#!/bin/bash
set -e
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "Activate with: source .venv/bin/activate"
