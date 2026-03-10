#pip install transformers==4.51.3

MODEL="OpenGVLab/InternVL3_5-8B"
TASK_NAME="longvid-reasoning-eval"
OUTPUT_DIR="/nas2/longvideo_eval/results/internvl3p5_8b"
GPUS="0,1,2,3"
NUM_PROCESSES=4
BATCH_SIZE=1
PYTHON="/nas2/edwin/miniconda/envs/internvl/bin/python"

export HF_HUB_OFFLINE=1

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},device_map=auto"

export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} -m lmms_eval \
    --model internvl3_5 \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --log_samples \
    --log_samples_suffix internvl3_8b \
    --output_path ${OUTPUT_DIR}
