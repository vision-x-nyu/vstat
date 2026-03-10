MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
TASK_NAME="longvid-reasoning-eval"
OUTPUT_DIR="/nas2/longvideo_eval/results/llava_onevision_7b"
GPUS="1,2,3"
NUM_PROCESSES=3
BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1
export HF_HOME="/nas2/edwin/lmms-eval/.cache/huggingface"

PYTHON="/nas2/edwin/miniconda/envs/llava_onevision/bin/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="pretrained=${MODEL},conv_template=qwen_1_5,device_map=cuda:0"

export LMMS_EVAL_LAUNCHER="accelerate"

$PYTHON -m accelerate.commands.launch --num_processes=${NUM_PROCESSES} -m lmms_eval \
    --model llava_onevision \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASK_NAME} \
    --batch_size ${BATCH_SIZE} \
    --log_samples \
    --log_samples_suffix llava_onevision_7b \
    --output_path ${OUTPUT_DIR}
