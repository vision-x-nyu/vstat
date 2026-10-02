# Download prepared VSTAT clips

The [private Google Drive folder](https://drive.google.com/drive/folders/1IAUiBXHhqCP_jXM9wZe42dEbIdd-pWoZ) contains all **834 clips / 1,500 questions**, including clips whose YouTube sources are unavailable. The 1.87 GB of videos are already trimmed and redacted. Keep the supplied paths and do not rerun `download_youtube.py` or `redact.sh` on this copy.

## Request access and authorize once

1. Open the folder while signed in to Google and click **Request access**. Access is managed by **pinzhihuang23@gmail.com**. The folder is restricted to approved accounts; a link alone does not grant access.
2. Install [rclone](https://rclone.org/downloads/) and Python 3.10+. Create your own [Google OAuth client for rclone](https://rclone.org/drive/#making-your-own-client-id); rclone's shared client is being retired during 2026.
3. Run `rclone config`. Create a remote named **`vstat_drive`**, choose **Google Drive**, enter your OAuth client details, select **`drive.readonly`**, and sign in with the **same Google account that requested access**. Leave the root folder ID blank and answer **No** to Shared Drive / Team Drive. For a server without a browser, use [rclone's remote authorization instructions](https://rclone.org/remote_setup/).

Keep the rclone configuration and OAuth tokens private. Each downloader uses their own approved account; the script contains no shared credentials and cannot grant access.

## Download and verify

From the [evaluation repository](https://github.com/vision-x-nyu/vstat):

```bash
git clone https://github.com/vision-x-nyu/vstat.git
cd vstat
python scripts/download_drive.py --output data/vstat
```

The same script is available in the Hugging Face dataset's `scripts/` directory. From that dataset directory, use `--output .` to keep the existing layout.

The downloader checks access, copies the release with parallel transfers and retries, reuses completed matching files on a rerun, and validates every manifest file's size and SHA-256. It verifies the pinned QA file and coverage of all 834 clips / 1,500 questions. It does not delete extra local files.

```bash
# Check access without downloading videos.
python scripts/download_drive.py --check-access

# Validate an existing download; no rclone or Google login is needed.
python scripts/download_drive.py --output data/vstat --verify-only

# Optional: fully decode each video too (requires ffmpeg).
python scripts/download_drive.py --output data/vstat --verify-only --decode
```

For browser downloads, download the folder after approval, extract all parts into `data/vstat/`, and run `--verify-only`. The layout must include `data/vstat/vstat_qa_clean.json` and `data/vstat/videos/` directly, without an extra enclosing folder.

## Agent workflow

The agent can read this page and run the following commands. The user must complete the initial Google login and access request; credentials must never be posted in a chat or committed to a repository.

```bash
python scripts/download_drive.py --check-access --json
python scripts/download_drive.py --output data/vstat --json
```

With `--json`, stdout contains one final JSON object. Transfer progress goes to stderr. Use `status` and the exit code to decide what to do:

- **`ready` (0):** access check passed; start downloading.
- **`verified` (0):** all media and QA checks passed; evaluation can start.
- **`needs_setup` / `setup_error` (2):** install or configure rclone; follow `setup_url`.
- **`needs_auth` (3):** ask the user to reconnect the remote with `rclone config reconnect vstat_drive:`.
- **`access_required` (4):** show `access_url` and ask the user to request access with the same account. If already approved, check the account and OAuth scope.
- **`service_error` / `download_failed` (5):** retry after resolving connectivity, quota, or transfer errors.
- **`verification_failed` (6):** repair the download; do not start evaluation or silently drop questions.

For an already submitted request, `--wait-for-access 600` waits up to ten minutes for access, checking every 30 seconds. It does not submit or approve a request. `--remote NAME` selects another configured remote. `--dry-run` prints the copy command without contacting Drive.

## Start evaluation

From the evaluation repository root, install the evaluator and model dependencies:

```bash
python -m pip install -e '.[video]'
export VSTAT_QA_PATH="$PWD/data/vstat/vstat_qa_clean.json"
export VSTAT_VIDEO_ROOT="$PWD/data/vstat"

python -m lmms_eval \
  --include_path "$PWD/lmms_eval/tasks" \
  --model qwen3_vl \
  --model_args 'pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=784,max_pixels=50176,max_num_frames=128' \
  --tasks vstat \
  --batch_size 1 \
  --output_path ./results/vstat \
  --log_samples
```

This example requires suitable CUDA hardware and model dependencies. Replace the model and arguments for your model. Append `--limit 3` for a smoke check, then remove it for the full **1,500-question** evaluation. Use the official evaluator's accuracy and MRA metrics.

## Reproducibility

- Dataset revision: `21b3198fb46627acea9770216a36651a407394b8`.
- QA SHA-256: `a4df882c2872f54d4fa0ec58f2678a8ea0ae2c9f1a0a022e9817834fee74f59c`.
- The download contains `manifest.json`, `SHA256SUMS`, `media_specs.json`, and `MEDIA_NOTES.md`. The notes document differences between archived media and upstream reference metadata. The backup bytes are preserved.
- All 834 videos were fully decoded; all 530 synthetic / self-recorded files match the pinned Hugging Face LFS hashes. Record the code revision and model settings with results.

Annotations, synthetic videos, and self-recorded videos retain CC BY 4.0. YouTube videos retain the original uploaders' rights; private access does not change their license.
