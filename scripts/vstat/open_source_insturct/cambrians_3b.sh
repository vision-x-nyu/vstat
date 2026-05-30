MODEL="nyu-visionx/Cambrian-S-3B"
TASK_NAME="vstat"
GPUS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
BATCH_SIZE=1

MIV_TOKEN_LEN="${MIV_TOKEN_LEN:-64}"
SI_TOKEN_LEN="${SI_TOKEN_LEN:-729}"

PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
TASKS_INCLUDE="${LMMS_EVAL_TASKS_PATH:-$REPO_ROOT/lmms_eval/tasks}"

OPEN_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frame_sweep_env.sh
. "${OPEN_SRC}/frame_sweep_env.sh"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},conv_template=qwen_2,miv_token_len=${MIV_TOKEN_LEN},si_token_len=${SI_TOKEN_LEN},video_max_frames=${MAX_FRAMES}"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} --main_process_port=${MAIN_PROCESS_PORT:-29517} -m lmms_eval \
    --include_path "${TASKS_INCLUDE}" \
    --model cambrians \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix "${LOG_SAMPLES_SUFFIX}" \
    --output_path ${OUTPUT_DIR}
