#!/bin/bash

set -euo pipefail

: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be exported before running this script}"

MODEL_VERSION="${MODEL_VERSION:-gemini-3.1-pro-preview}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PYTHON="${PYTHON:-python}"

OUTPUT_DIR="results/longvid-reasoning-eval_recorded/gemini31_pro"
LOG_SAMPLES_SUFFIX="gemini31_pro_longvid-reasoning-eval_recorded"

export LMMS_EVAL_LAUNCHER=python

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},timeout=120"

SUBTASKS=(
    "longvid-reasoning-eval_recorded"
)

for TASK in "${SUBTASKS[@]}"; do
    echo "=========================================="
    echo "Running: $TASK"
    echo "=========================================="
    "$PYTHON" -m lmms_eval \
        --model gemini_api \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --log_samples \
        --log_samples_suffix "${LOG_SAMPLES_SUFFIX}" \
        --output_path "${OUTPUT_DIR}" || echo "FAILED: $TASK"
    echo "Done: $TASK"
    echo ""
done

echo "All subtasks completed!"
