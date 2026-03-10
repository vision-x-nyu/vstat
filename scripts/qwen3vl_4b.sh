MODEL="Qwen/Qwen3-VL-4B-Instruct"
TASK_NAME="longvid-reasoning-eval"
OUTPUT_DIR="/nas2/longvideo_eval/results/qwen3vl_4b"
GPUS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
BATCH_SIZE=40
MIN_PIXELS=784
MAX_PIXELS=50176

export HF_HUB_OFFLINE=1
PYTHON="/nas2/edwin/miniconda/envs/lmms_eval/bin/python"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},device_map=auto"

export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} -m lmms_eval \
    --model qwen3_vl \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix qwen3_4b \
    --output_path ${OUTPUT_DIR}
