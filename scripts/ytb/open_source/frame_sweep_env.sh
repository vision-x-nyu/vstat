# Sourced by model launcher scripts in this directory after REPO_ROOT is set.
#
# Env (optional overrides; see repository README):
#   MAX_FRAMES — video frame budget (default 32).
#   YTb_RESULTS_OPEN_SOURCE — root for run outputs (default $REPO_ROOT/results/ytb/open_source).
#
# BASH_SOURCE[1] is the calling launcher (used for output slug).
MAX_FRAMES="${MAX_FRAMES:-32}"
MODEL_SLUG="$(basename "${BASH_SOURCE[1]}" .sh)"
YTb_RESULTS_OPEN_SOURCE="${YTb_RESULTS_OPEN_SOURCE:-$REPO_ROOT/results/ytb/open_source}"
OUTPUT_DIR="${YTb_RESULTS_OPEN_SOURCE}/max_frames_${MAX_FRAMES}/${MODEL_SLUG}/longvid_reasoning_eval_ytb"
LOG_SAMPLES_SUFFIX="${MODEL_SLUG}_f${MAX_FRAMES}_ytb"
