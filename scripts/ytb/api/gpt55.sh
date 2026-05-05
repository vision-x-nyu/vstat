#!/bin/bash

set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be exported before running this script}"

MODEL_VERSION="${MODEL_VERSION:-gpt-5.5}"
TASK="${TASK:-longvid-reasoning-eval_ytb}"
MAX_FRAMES="${MAX_FRAMES:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PYTHON="${PYTHON:-python}"

TIMEOUT="${TIMEOUT:-240}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF_S="${RETRY_BACKOFF_S:-2.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
REASONING_SUMMARY="${REASONING_SUMMARY:-auto}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=${MAX_NEW_TOKENS},temperature=0,do_sample=false}"

MODEL_SHORT="${MODEL_SHORT:-gpt55}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${TASK}/${MODEL_SHORT}/max_frames_${MAX_FRAMES}}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-${MODEL_SHORT}_max_frames_${MAX_FRAMES}_${TASK}}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_DIR}/response_cache}"

LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

CACHE_ARGS=()
if [ "${USE_CACHE:-1}" != "0" ]; then
    CACHE_ARGS=(--use_cache "${CACHE_PATH}")
fi

MODEL_ARGS="model_version=${MODEL_VERSION},timeout=${TIMEOUT},num_concurrent=${CONCURRENCY},max_retries=${MAX_RETRIES},retry_backoff_s=${RETRY_BACKOFF_S},max_frames_num=${MAX_FRAMES},reasoning_effort=${REASONING_EFFORT},use_responses_api=True,reasoning_summary=${REASONING_SUMMARY}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Running: ${TASK}"
echo "Model: ${MODEL_VERSION}"
echo "Max frames: ${MAX_FRAMES}"
echo "Reasoning effort: ${REASONING_EFFORT}"
echo "Reasoning summary: ${REASONING_SUMMARY}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

"${PYTHON}" -m lmms_eval \
    --model openai \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --log_samples \
    --log_samples_suffix "${LOG_SAMPLES_SUFFIX}" \
    --output_path "${OUTPUT_DIR}" \
    "${CACHE_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"
