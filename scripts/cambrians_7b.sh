MODEL="nyu-visionx/Cambrian-S-7B"
TASK_NAME="longvid-reasoning-eval"
OUTPUT_DIR="/nas2/longvideo_eval/results/cambrians_7b"
GPUS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
BATCH_SIZE=1

MIV_TOKEN_LEN="${MIV_TOKEN_LEN:-64}"
SI_TOKEN_LEN="${SI_TOKEN_LEN:-729}"

PYTHON="/nas2/edwin/miniconda/envs/cambrians_eval/bin/python"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},conv_template=qwen_2,miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN},device_map=cuda:0"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} -m lmms_eval \
    --model cambrians \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix cambrians_7b \
    --output_path ${OUTPUT_DIR}