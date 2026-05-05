#!/bin/bash

set -euo pipefail

: "${GOOGLE_API_KEY:?GOOGLE_API_KEY must be exported before running this script}"

MODEL_VERSION="${MODEL_VERSION:-gemini-3.1-pro-preview}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PYTHON="${PYTHON:-python}"

TIMEOUT="${TIMEOUT:-240}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-1.0}"
CACHE_WRITE_EVERY_N="${CACHE_WRITE_EVERY_N:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-65536}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=${MAX_NEW_TOKENS},temperature=0,top_p=1.0,num_beams=1,do_sample=false}"

if [ "$MODEL_VERSION" == "gemini-3.1-pro-preview" ]; then
    MODEL_SHORT="gemini31_pro"
    LEVELS="${LEVELS:-LOW MEDIUM HIGH}"
elif [ "$MODEL_VERSION" == "gemini-3-flash-preview" ]; then
    MODEL_SHORT="gemini3_flash"
    LEVELS="${LEVELS:-MINIMAL LOW MEDIUM HIGH}"
else
    echo "Invalid model version: $MODEL_VERSION"
    exit 1
fi

export LMMS_EVAL_LAUNCHER=python

SUBTASKS=(
    "longvid-reasoning-eval_yista"
)

for TASK in "${SUBTASKS[@]}"; do
    for LEVEL in $LEVELS; do
        LEVEL_LOWER="${LEVEL,,}"

        echo "=========================================="
        echo "Running: $TASK | model: $MODEL_VERSION | thinking_level: $LEVEL"
        echo "=========================================="

        TASK_OUTPUT_DIR="results/${TASK}/${MODEL_SHORT}/${LEVEL_LOWER}"
        TASK_LOG_SAMPLES_SUFFIX="${MODEL_SHORT}_${LEVEL_LOWER}_${TASK}"
        mkdir -p "$TASK_OUTPUT_DIR"

        MODEL_ARGS="model_version=${MODEL_VERSION},timeout=${TIMEOUT},num_concurrent=${CONCURRENCY},continual_mode=True,response_persistent_folder=${TASK_OUTPUT_DIR}/cache,max_retries=${MAX_RETRIES},retry_backoff_s=${RETRY_BACKOFF_S},cache_write_every_n=${CACHE_WRITE_EVERY_N},thinking_level=${LEVEL},include_thoughts=True"

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
            --output_path "${TASK_OUTPUT_DIR}" || echo "FAILED: $TASK ($LEVEL)"
        echo "Done: $TASK ($LEVEL)"
        echo ""
    done
done

echo "All subtasks completed!"