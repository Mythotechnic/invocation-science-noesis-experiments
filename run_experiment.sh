#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/experiment_config.yaml
RESULT_DIR=results

mkdir -p "${RESULT_DIR}"
mkdir -p reports
mkdir -p models

echo "1) Environment check"
python - << 'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
PY

echo "2) (Optional) Train pooled→inputs_embeds mapper (can fail safely)"
python scripts/mapping_trainer.py --config "${CONFIG}" --out models/decoder.pt || echo "Mapper training skipped"

echo "3) Run main NOESIS embedding-reentry experiment"
python scripts/mvp_embedding_reentry_full.py --config "${CONFIG}" --outdir "${RESULT_DIR}"

echo "4) Run simple task suite"
python scripts/mvp_task_suite.py --config "${CONFIG}" --resultsdir "${RESULT_DIR}"

echo "5) Run batch Lyapunov estimates"
python scripts/batch_lyapunov.py --config "${CONFIG}" --resultsdir "${RESULT_DIR}"

echo "6) Aggregate and generate report"
python scripts/analysis_and_plots.py --resultsdir "${RESULT_DIR}" --out reports/experiment_report.md

echo "Done. Results in ${RESULT_DIR}, report in reports/experiment_report.md"
