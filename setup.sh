#!/usr/bin/env bash
# Clean-setup script for a fresh Linux (or macOS) machine.
# Creates a virtual environment and installs all dependencies from requirements.txt.
#
# Usage:
#   bash setup.sh
# then:
#   source .venv/bin/activate
#   bash run_pipeline.sh
set -euo pipefail

# Move to the directory containing this script (the MLOP project root).
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Using Python: $($PYTHON --version)"

echo "==> Creating virtual environment (.venv)"
$PYTHON -m venv .venv

echo "==> Activating venv and upgrading pip"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  bash run_pipeline.sh        # run the full ML pipeline"
echo "  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000   # serve the API"
