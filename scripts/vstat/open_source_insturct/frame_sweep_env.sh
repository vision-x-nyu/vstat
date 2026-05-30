# Sourced by VSTAT model launcher scripts in this directory after REPO_ROOT is set.
#
# Env (optional overrides; see repository README):
#   MAX_FRAMES — video frame budget (default 32).
#   VSTAT_RESULTS_OPEN_SOURCE — root for run outputs (default $REPO_ROOT/results/vstat/open_source).
#
# BASH_SOURCE[1] is the calling launcher (used for output slug).
MAX_FRAMES="${MAX_FRAMES:-32}"
MODEL_SLUG="$(basename "${BASH_SOURCE[1]}" .sh)"
VSTAT_RESULTS_OPEN_SOURCE="${VSTAT_RESULTS_OPEN_SOURCE:-$REPO_ROOT/results/vstat/open_source}"
OUTPUT_DIR="${VSTAT_RESULTS_OPEN_SOURCE}/max_frames_${MAX_FRAMES}/${MODEL_SLUG}/vstat"
LOG_SAMPLES_SUFFIX="${MODEL_SLUG}_f${MAX_FRAMES}_vstat"
