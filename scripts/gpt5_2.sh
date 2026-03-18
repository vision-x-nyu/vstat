MODEL_VERSION="gpt-5.2"
TASK_NAME="longvid-reasoning-eval_5min"
OUTPUT_DIR="/nas2/longvideo_eval/results/gpt5p2/longvid_reasoning_eval_5min_debug"
BATCH_SIZE=64
NUM_CONCURRENT=32

PYTHON="/nas2/edwin/miniconda/envs/lmms_eval/bin/python"

export OPENAI_API_KEY="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"
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="model_version=${MODEL_VERSION},num_concurrent=${NUM_CONCURRENT},max_retries=5,timeout=120,continual_mode=True,response_persistent_folder=${OUTPUT_DIR}/cache"

$PYTHON -m lmms_eval \
    --model openai \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix gpt5p2 \
    --output_path ${OUTPUT_DIR}