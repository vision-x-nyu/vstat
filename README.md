# VISTA (Video Inference & Spatial/Temporal Assessment)

This repository is a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with the **VISTA** benchmark task: **`longvid-reasoning-eval_vista`**, plus reference shell launchers for open-weight models under [`scripts/vista/open_source/`](scripts/vista/open_source).

## Install

From the repository root:

```bash
pip install -e ".[all]"
```

Use a Python environment that already has CUDA and the vision-language dependencies required for your chosen model backend (see upstream lmms-eval docs).

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

## Upstream

For general lmms-eval usage, models, and task mechanics, see the [upstream README](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [`docs/`](docs/).
