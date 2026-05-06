MODEL="OpenGVLab/InternVL3_5-8B"
# InternVL3.5-8B with Thinking mode enabled (R1_SYSTEM_PROMPT, sampling at T=0.6).
# Reference: https://huggingface.co/OpenGVLab/InternVL3_5-8B
TASK_NAME="longvid-reasoning-eval_vstat"
GPUS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
BATCH_SIZE=1
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
TASKS_INCLUDE="${LMMS_EVAL_TASKS_PATH:-$REPO_ROOT/lmms_eval/tasks}"

OPEN_SRC="$REPO_ROOT/scripts/vstat/open_source_insturct"
# shellcheck source=../open_source_insturct/frame_sweep_env.sh
. "${OPEN_SRC}/frame_sweep_env.sh"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},modality=video,num_frame=${MAX_FRAMES},thinking=True"

export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} --main_process_port=${MAIN_PROCESS_PORT:-29517} -m lmms_eval \
    --include_path "${TASKS_INCLUDE}" \
    --model internvl3_5 \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix "${LOG_SAMPLES_SUFFIX}" \
    --output_path ${OUTPUT_DIR}
