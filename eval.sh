#!/bin/bash

set -euo pipefail

# : "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be exported before running this script}"

export GOOGLE_API_KEY="AIzaSyBVnyjRzjlglmaYj7ESgq12yt8d96VM-ck"

MODEL_VERSION="${MODEL_VERSION:-gemini-3.1-pro-preview}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PYTHON="${PYTHON:-python}"

THINKING_BUDGET="${THINKING_BUDGET:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=${MAX_NEW_TOKENS},temperature=0,top_p=1.0,num_beams=1,do_sample=false}"

if [ "$MODEL_VERSION" == "gemini-3.1-pro-preview" ]; then
    OUTPUT_DIR="results/longvid-reasoning-eval_ytb/gemini31_pro"
    LOG_SAMPLES_SUFFIX="gemini31_pro_longvid-reasoning-eval_ytb"
elif [ "$MODEL_VERSION" == "gemini-3-flash-preview" ]; then
    OUTPUT_DIR="results/longvid-reasoning-eval_ytb/gemini3_flash"
    LOG_SAMPLES_SUFFIX="gemini3_flash_longvid-reasoning-eval_ytb"
else
    echo "Invalid model version: $MODEL_VERSION"
    exit 1
fi

export LMMS_EVAL_LAUNCHER=python

# mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},timeout=120,no_audio=True,thinking_budget=${THINKING_BUDGET},response_persistent_folder=${OUTPUT_DIR}/cache"

SUBTASKS=(
    # "longvid-reasoning-eval_ytb"
    # "longvid-reasoning-eval_recorded"
    "longvid-reasoning-eval_ytb_stretch_0p2s"
)

for TASK in "${SUBTASKS[@]}"; do
    echo "=========================================="
    echo "Running: $TASK with model: $MODEL_VERSION"
    echo "=========================================="

    MODEL_SHORT="$(basename "$OUTPUT_DIR")"
    TASK_OUTPUT_DIR="results/${TASK}/${MODEL_SHORT}"
    TASK_LOG_SAMPLES_SUFFIX="${MODEL_SHORT}_${TASK}"
    mkdir -p "$TASK_OUTPUT_DIR"

    echo "=========================================="
    echo "Output directory: $TASK_OUTPUT_DIR"
    echo "Log samples suffix: $TASK_LOG_SAMPLES_SUFFIX"
    echo "=========================================="

    "$PYTHON" -m lmms_eval \
        --model gemini_api \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --gen_kwargs "${GEN_KWARGS}" \
        --log_samples \
        --log_samples_suffix "${TASK_LOG_SAMPLES_SUFFIX}" \
        --output_path "${TASK_OUTPUT_DIR}" || echo "FAILED: $TASK"
    echo "Done: $TASK"
    echo ""
done

echo "All subtasks completed!"
