#!/bin/bash
#
# Brief description:
# Run `lmms_eval` with the Gemini 3.1 Pro Preview API model on the long-video reasoning task.
#
# Usage:
# `export GOOGLE_API_KEY="<your_api_key>"`
# `bash longvid-reasoning-eval/scripts/gemini3_0_pro.sh`
# `TASK_NAME=longvid-reasoning-eval_5min BATCH_SIZE=1 bash longvid-reasoning-eval/scripts/gemini3_0_pro.sh`
#
# Input spec:
# Environment variables:
# `GOOGLE_API_KEY` required by `lmms_eval.models.simple.gemini_api`.
# `MODEL_VERSION` optional, default `gemini-3.1-pro-preview`.
# `TASK_NAME` optional, default `longvid-reasoning-eval_5min`.
# `BATCH_SIZE` optional, default `1`.
# `PYTHON` optional, default `/nas2/edwin/miniconda/envs/lmms_eval/bin/python`.
# `HF_HOME` optional, default `/nas2/edwin/lmms-eval/.cache/huggingface`.
# `OUTPUT_DIR` optional, default `/nas2/longvideo_eval/results/gemini3_1_pro_preview/<task>_debug`.
# `LOG_SAMPLES_SUFFIX` optional, default `gemini3_1_pro_preview_<task>_debug`.
#
# Output spec:
# Writes evaluation outputs, aggregate metrics, and sample logs under `OUTPUT_DIR`.

set -euo pipefail

: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY before running this script.}"

MODEL_VERSION="${MODEL_VERSION:-gemini-3.1-pro-preview}"
TASK_NAME="${TASK_NAME:-longvid-reasoning-eval_5min}"
TASK_DIR_SUFFIX="${TASK_NAME//-/_}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PYTHON="${PYTHON:-/nas2/edwin/miniconda/envs/lmms_eval/bin/python}"
HF_HOME="${HF_HOME:-/nas2/edwin/lmms-eval/.cache/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas2/longvideo_eval/results/gemini3_1_pro_preview/${TASK_DIR_SUFFIX}_debug}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-gemini3_1_pro_preview_${TASK_DIR_SUFFIX}_debug}"

export GOOGLE_API_KEY="AIzaSyA3NdztiSwFQzogen7oG9cic1rTyXkM8YI"
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},timeout=120"

"$PYTHON" -m lmms_eval \
    --model gemini_api \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --log_samples \
    --log_samples_suffix "${LOG_SAMPLES_SUFFIX}" \
    --output_path "${OUTPUT_DIR}"
