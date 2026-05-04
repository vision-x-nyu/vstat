# VISTA (Video Inference & Spatial/Temporal Assessment)

This repository is a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with the **VISTA** benchmark task: **`longvid-reasoning-eval_vista`**, plus reference shell launchers for open-weight models under [`scripts/vista/open_source/`](scripts/vista/open_source).

## Install

From the repository root:

```bash
git submodule update --init --recursive
pip install -e ".[all]"
```

## Reference Environments

The checked launchers set `PYTHON=` to maintainer-local conda paths. If you create environments elsewhere, update the `PYTHON=` line in the corresponding script.

| Scripts | Maintainer `PYTHON` | Python | `transformers` | Other observed pins |
|---------|----------------------|--------|----------------|---------------------|
| `cambrians_*.sh` | `/nas2/edwin/miniconda/envs/cambrians_eval/bin/python` | `3.10.19` | `4.37.0` | `torch==2.5.1`, `accelerate==0.23.0` |
| `internvl3p5_*.sh` | `/nas2/edwin/miniconda/envs/internvl/bin/python` | `3.10.19` | `4.51.3` | `torch==2.10.0`, `accelerate==0.34.2` |
| `qwen3vl_*.sh` | `/nas2/edwin/miniconda/envs/lmms_eval/bin/python` | `3.14.2` | `5.0.0` | `torch==2.10.0`, `accelerate==1.12.0`, `qwen-vl-utils==0.0.14` |

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

## Dataset path

Task YAMLs include [`lmms_eval/tasks/longvid-reasoning-eval_vista/_default_template_yaml`](lmms_eval/tasks/longvid-reasoning-eval_vista/_default_template_yaml), which sets `dataset_kwargs.data_files.test` to the merged QA JSON. That path is **machine-specific** in the template; replace it with the path to your copy of the benchmark JSON (or maintain a local override file and point `include_path` / custom YAMLs at it).

Each sub-task YAML in that directory extends the template and sets `task:` names for lmms-eval.

## Run VISTA

Task group name:

```text
longvid-reasoning-eval_vista
```

Example (after editing dataset paths to valid files on your machine):

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

Each `*.sh` script runs **`longvid-reasoning-eval_vista`** with a fixed Hugging Face `MODEL=`, Accelerate, and model-specific `model_args`. They `cd` to `scripts/vista` (parent of `open_source`) before invoking `python -m lmms_eval`. Interpreters and cache paths are set **per script** for the maintainers’ conda environments; edit the `PYTHON=`, `HF_HOME=`, and `HF_HUB_OFFLINE=` lines if your setup differs.

| Variable | Role |
|----------|------|
| `MAX_FRAMES` | Video frame budget (default: `32`; see `frame_sweep_env.sh`). |
| `LMMS_EVAL_TASKS_PATH` | Override for `--include_path` (default: `$REPO_ROOT/lmms_eval/tasks`). |
| `VISTA_RESULTS_OPEN_SOURCE` | Root directory for `OUTPUT_DIR` (default: `$REPO_ROOT/results/vista/open_source`). |
| `GPUS` | Set in each script header; controls `CUDA_VISIBLE_DEVICES`. |

**Batch queue:** [`submit_all_vista.sh`](scripts/vista/open_source/submit_all_vista.sh) enqueues all model scripts over a grid of `MAX_FRAMES` values using **task-spooler** (`ts`). Optional env: `VISTA_OPEN_SOURCE_FRAMES`, `VISTA_MANIFEST`.

