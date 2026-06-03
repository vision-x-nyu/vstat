<h1>
  <img src="assets/logo.png" alt="" height="32" valign="middle">&nbsp;VSTAT: Benchmarking Visual State Tracking in Multimodal Video Understanding
</h1>

<p align="center">
  <img src="assets/teaser.gif" alt="VSTAT teaser" width="100%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.03920"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a>
  <a href="https://vision-x-nyu.github.io/vstat-site/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/nyu-visionx/vstat"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Benchmark-yellow"></a>
</p>

<p align="center">
  <a href="https://sihyun.me/">Sihyun Yu</a><sup>1,2†*</sup>,
  <a href="https://willisma.github.io/">Nanye Ma</a><sup>1†*</sup>,
  <a href="https://pinzhihuang.github.io/">Pinzhi Huang</a><sup>1†*</sup>,
  <a href="https://hyunseoklee-ai.github.io">Hyunseok Lee</a><sup>2*</sup>,
  <a href="https://shushengyang.com/">Shusheng Yang</a><sup>1</sup>,
  <a href="https://choi403.github.io/">June Suk Choi</a><sup>2</sup>,
  <a href="https://ellisbrown.github.io/">Ellis Brown</a><sup>1</sup>,<br>
  <a href="https://ojmichel.github.io/">Oscar Michel</a><sup>1</sup>,
  <a href="https://bytetriper.github.io/">Boyang Zheng</a><sup>1</sup>,
  <a href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo Shin</a><sup>2</sup>,
  <a href="https://www.sainingxie.com/">Saining Xie</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>New York University&nbsp;&nbsp;&nbsp;<sup>2</sup>KAIST
</p>

<p align="center">
  <sup>†</sup> Project lead.&nbsp;&nbsp;&nbsp;<sup>*</sup> Equal technical contribution.
</p>

---

## Overview

**VSTAT** (Visual STAte Tracking) is a benchmark for evaluating the ability of Multimodal Large Language Models (MLLMs) to track fine-grained visual state changes in long-form videos. Unlike benchmarks that test static scene understanding or simple event recognition, VSTAT requires models to maintain a running mental model of object states, count changes, and temporal order across extended video sequences sourced from YouTube.

## Release

- `2026-05` 🚀 We release VSTAT benchmark and evaluation code.

## Contents

- [Overview](#overview)
- [Release](#release)
- [Results](#results)
- [Installation](#installation)
- [Benchmark](#benchmark)
- [Evaluation](#evaluation)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Results

We benchmark video-supporting MLLMs from diverse model families in zero-shot settings with greedy decoding. Each question is labeled by **state element** (Count, Location, Attribute) and **state structure** (Atomic, Sequence, Set, Dict). MCQ tasks are scored by accuracy; numerical tasks by Mean Relative Accuracy (MRA). The reported average is computed over all questions.

<p align="center">
  <img src="assets/leaderboard.png" alt="VSTAT leaderboard" width="100%">
</p>

Even the strongest proprietary model (Gemini-3.1 Pro) reaches only **44.4** average, far below human performance (**90.5**), highlighting the difficulty of visual state tracking for current MLLMs. See our [project website](https://vision-x-nyu.github.io/vstat-site/) for the full, interactive leaderboard.

## Installation

```bash
conda create --name vstat python=3.10
conda activate vstat

git clone https://github.com/vision-x-nyu/vstat.git
cd vstat

git submodule update --init --recursive
pip install -e ".[video]"
```

## Benchmark

VSTAT is hosted on HuggingFace: [`nyu-visionx/VSTAT`](https://huggingface.co/collections/nyu-visionx/vstat).

Download it into the local `data/` folder:

```bash
mkdir -p data
huggingface-cli download nyu-visionx/VSTAT \
  --repo-type=dataset \
  --local-dir data/vstat
```

Then run the video download and redaction scripts bundled with the benchmark:

```bash
cd data/vstat
python scripts/download_youtube.py --resolution-map youtube_resolutions.json
bash scripts/redact.sh
cd ../..
```

Ensure every video referenced in `vstat_qa_clean.json` exists under `data/vstat/` before evaluation. Missing files can cause silent multi-rank hangs during distributed runs.

**Custom paths:** If your data lives elsewhere, set:
```bash
export VSTAT_QA_PATH=/path/to/vstat_qa_clean.json
export VSTAT_VIDEO_ROOT=/path/to/vstat
```

## Evaluation

Task name: **`vstat`**

**Open-weight model** (single process):

```bash
python -m lmms_eval \
  --include_path "$(pwd)/lmms_eval/tasks" \
  --model qwen3_vl \
  --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=784,max_pixels=50176,max_num_frames=128" \
  --tasks vstat \
  --batch_size 1 \
  --output_path ./results/vstat \
  --log_samples
```

**Open-weight model** (multi-GPU with Accelerate):

```bash
python -m accelerate.commands.launch \
  --num_processes 8 \
  -m lmms_eval \
  --include_path "$(pwd)/lmms_eval/tasks" \
  --model qwen3_vl \
  --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=784,max_pixels=50176,max_num_frames=128" \
  --tasks vstat \
  --batch_size 1 \
  --output_path ./results/vstat \
  --log_samples
```

**API model** (Gemini):

```bash
export GOOGLE_API_KEY=your_key

python -m lmms_eval \
  --include_path "$(pwd)/lmms_eval/tasks" \
  --model gemini_api \
  --model_args "model_version=gemini-3.1-pro-preview,timeout=240,num_concurrent=64" \
  --tasks vstat \
  --batch_size 1 \
  --output_path ./results/vstat \
  --log_samples
```

Swap `--model` and `--model_args` for other supported backends (e.g. `internvl3`, `cambrians`, `llava_onevision`, `mimo_vl`). See `lmms_eval/models/simple/` for available model wrappers.

| Variable | Role | Default |
|----------|------|---------|
| `VSTAT_QA_PATH` | Path to QA JSON | `data/vstat/vstat_qa_clean.json` |
| `VSTAT_VIDEO_ROOT` | Root directory for video files | `data/vstat/` |
| `HF_HOME` | HuggingFace cache | `~/.cache/huggingface` |

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgement

Our evaluation framework is built upon [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the LMMs-Lab team for providing this excellent toolkit for evaluating multimodal large language models.

## Citation

If you find our benchmark and code useful, please consider citing our work:

```bibtex
@article{vstat2026,
    title={Benchmarking Visual State Tracking in Multimodal Video Understanding},
    author={Sihyun Yu and Nanye Ma and Pinzhi Huang and Hyunseok Lee and Shusheng Yang and June Suk Choi and Ellis Brown and Oscar Michel and Boyang Zheng and Jinwoo Shin and Saining Xie},
    year={2026},
    journal={arXiv preprint arXiv:2606.03920},
}
```
