#!/usr/bin/env bash
# Run the full HeartCare ML pipeline end-to-end, then the test suite.
# Assumes dependencies are installed (run setup.sh first, or activate your venv).
#
# Usage:
#   bash run_pipeline.sh            # run pipeline + tests
#   bash run_pipeline.sh --no-tests # run pipeline only
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Running full ML pipeline"
$PYTHON -m src.orchestration.pipeline

if [[ "${1:-}" != "--no-tests" ]]; then
  echo ""
  echo "==> Running test suite"
  $PYTHON -m pytest tests/ -q
fi

echo ""
echo "Pipeline finished. Key artifacts:"
echo "  models/heart_disease_pipeline.pkl   (packaged model)"
echo "  models/model_card.json              (model metadata)"
echo "  docs/eda_report.md, docs/model_report.md"
echo "  plots/*.png                         (EDA + run plots)"
