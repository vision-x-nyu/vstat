# LongVid reasoning evaluation (lmms-eval fork)

This repository is a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with a single added benchmark task: **`longvid-reasoning-eval_ytb`**, plus reference shell launchers for open-weight models under [`scripts/ytb/open_source/`](scripts/ytb/open_source).

## Install

From the repository root:

```bash
pip install -e ".[all]"
```

Use a Python environment that already has CUDA and the vision-language dependencies required for your chosen model backend (see upstream lmms-eval docs).

## Dataset path

Task YAMLs include [`lmms_eval/tasks/longvid-reasoning-eval_ytb/_default_template_yaml`](lmms_eval/tasks/longvid-reasoning-eval_ytb/_default_template_yaml), which sets `dataset_kwargs.data_files.test` to the merged QA JSON. That path is **machine-specific** in the template; replace it with the path to your copy of the benchmark JSON (or maintain a local override file and point `include_path` / custom YAMLs at it).

Each sub-task YAML in that directory extends the template and sets `task:` / `group:` names for lmms-eval.

## Run the benchmark

Task group name:

```text
longvid-reasoning-eval_ytb
```

Example (after editing dataset paths to valid files on your machine):

```bash
python -m lmms_eval \
  --include_path "$(pwd)/lmms_eval/tasks" \
  --model <model_name> \
  --model_args "<args>" \
  --tasks longvid-reasoning-eval_ytb \
  --batch_size 1 \
  --output_path ./results/ytb \
  --log_samples
```

## Reference launchers (`scripts/ytb/open_source/`)

Each `*.sh` script runs **`longvid-reasoning-eval_ytb`** with a fixed Hugging Face `MODEL=`, Accelerate, and model-specific `model_args`. They `cd` to `scripts/ytb` (parent of `open_source`) before invoking `python -m lmms_eval`; install the package from the repo root as above.

**Environment variables** (all optional unless noted):

| Variable | Role |
|----------|------|
| `PYTHON` | Interpreter (default: `python`). |
| `HF_HOME` | Hugging Face cache root (default: `$HOME/.cache/huggingface`). |
| `HF_HUB_OFFLINE` | `0` or `1` (default: `0`). |
| `MAX_FRAMES` | Passed into model args as the video frame budget (default: `32`; see `frame_sweep_env.sh`). |
| `LMMS_EVAL_TASKS_PATH` | Override for `--include_path` (default: `$REPO_ROOT/lmms_eval/tasks`). |
| `YTb_RESULTS_OPEN_SOURCE` | Root directory for `OUTPUT_DIR` (default: `$REPO_ROOT/results/ytb/open_source`). |
| `GPUS` / `CUDA_VISIBLE_DEVICES` | Set in the script header; edit before running if needed. |

**Batch queue:** [`submit_all_ytb.sh`](scripts/ytb/open_source/submit_all_ytb.sh) enqueues all model scripts over a grid of `MAX_FRAMES` values using **task-spooler** (`ts`).

## Upstream

For general lmms-eval usage, models, and task mechanics, see the [upstream README](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [`docs/`](docs/).
