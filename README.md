# VISTA (Video Inference & Spatial/Temporal Assessment)

This repository is a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with the **VISTA** benchmark task: **`longvid-reasoning-eval_vista`**, plus reference shell launchers for open-weight models under [`scripts/vista/open_source/`](scripts/vista/open_source).

## Install

From the repository root:

```bash
git submodule update --init --recursive
pip install -e ".[all]"
```

## Reference Environments

The checked launchers default to `PYTHON="${PYTHON:-python}"`, so they use the active environment by default. To run with a specific environment, prefix the command with `PYTHON=/path/to/env/bin/python`.

| Scripts | Suggested env | Python | `transformers` | Other observed pins |
|---------|---------------|--------|----------------|---------------------|
| `cambrians_*.sh` | `cambrians_eval` | `3.10` | `4.37.0` | `torch==2.5.1`, `accelerate==0.23.0` |
| `internvl3p5_*.sh` | `internvl` | `3.10` | `4.51.3` | `torch==2.10.0`, `accelerate==0.34.2` |
| `qwen3vl_*.sh` | `lmms_eval` | `3.14` | `5.0.0` | `torch==2.10.0`, `accelerate==1.12.0`, `qwen-vl-utils==0.0.14` |

Minimal setup pattern for each backend:

```bash
# Cambrian-S
conda create -n cambrians_eval python=3.10
conda activate cambrians_eval
python -m pip install -e ".[video]"
python -m pip install -e third_party/cambrian-s
python -m pip install "transformers==4.37.0" "accelerate==0.23.0"

# InternVL3.5
conda create -n internvl python=3.10
conda activate internvl
python -m pip install -e ".[video]"
python -m pip install "transformers==4.51.3" "accelerate==0.34.2"

# Qwen3-VL
conda create -n lmms_eval python=3.14
conda activate lmms_eval
python -m pip install -e ".[video,qwen]"
python -m pip install "transformers==5.0.0"
python -m pip install -e "third_party/Qwen3-VL/qwen-vl-utils[decord]"
```

Install the matching CUDA build of `torch`/`torchvision` for your machine before running large model evaluations. The versions above record the tested maintainer environments.

## Dataset

The task expects a local VSTAT folder at `data/vstat/`. Download the Hugging Face dataset [`VSTAT-NeurIPS2026/VSTAT`](https://huggingface.co/datasets/VSTAT-NeurIPS2026/VSTAT) there from the repository root:

```bash
mkdir -p data
huggingface-cli download VSTAT-NeurIPS2026/VSTAT \
  --repo-type=dataset \
  --local-dir data/vstat
```

Then run the HF-provided YouTube download and redaction scripts unchanged from inside that folder:

```bash
cd data/vstat
python scripts/download_youtube.py --resolution-map youtube_resolutions.json
bash scripts/redact.sh
cd ../..
```

For a quick smoke test before the full YouTube run, add `--limit 1` to the downloader command.

The task config reads `data/vstat/vstat_qa_clean.json`; video paths in that QA file are resolved relative to `data/vstat/`. If you keep VSTAT somewhere else, set `VSTAT_QA_PATH=/path/to/vstat_qa_clean.json` and `VSTAT_VIDEO_ROOT=/path/to/vstat`.

## Run VISTA

It's recommended to download the dataset into a local `data/` folder.

Task group name:

```text
longvid-reasoning-eval_vista
```

Example:

```bash
python -m lmms_eval \
  --include_path "$(pwd)/lmms_eval/tasks" \
  --model <model_name> \
  --model_args "<args>" \
  --tasks longvid-reasoning-eval_vista \
  --batch_size 1 \
  --output_path ./results/vista \
  --log_samples
```

## Reference launchers (`scripts/vista/open_source/`)

Each `*.sh` script runs **`longvid-reasoning-eval_vista`** with a fixed Hugging Face `MODEL=`, Accelerate, and model-specific `model_args`. They `cd` to the repository root before invoking `python -m lmms_eval`. Override `PYTHON`, `HF_HOME`, or `HF_HUB_OFFLINE` in your shell if your setup differs.

| Variable | Role |
|----------|------|
| `MAX_FRAMES` | Video frame budget (default: `32`; see `frame_sweep_env.sh`). |
| `LMMS_EVAL_TASKS_PATH` | Override for `--include_path` (default: `$REPO_ROOT/lmms_eval/tasks`). |
| `VISTA_RESULTS_OPEN_SOURCE` | Root directory for `OUTPUT_DIR` (default: `$REPO_ROOT/results/vista/open_source`). |
| `GPUS` | Set in each script header; controls `CUDA_VISIBLE_DEVICES`. |

**Batch queue:** [`submit_all_vista.sh`](scripts/vista/open_source/submit_all_vista.sh) enqueues all model scripts over a grid of `MAX_FRAMES` values using **task-spooler** (`ts`). Optional env: `VISTA_OPEN_SOURCE_FRAMES`, `VISTA_MANIFEST`.
