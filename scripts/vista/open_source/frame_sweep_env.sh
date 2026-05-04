# Sourced by VISTA model launcher scripts in this directory after REPO_ROOT is set.
#
# Env (optional overrides; see repository README):
#   MAX_FRAMES — video frame budget (default 32).
#   VISTA_RESULTS_OPEN_SOURCE — root for run outputs (default $REPO_ROOT/results/vista/open_source).
#
# BASH_SOURCE[1] is the calling launcher (used for output slug).
MAX_FRAMES="${MAX_FRAMES:-32}"
MODEL_SLUG="$(basename "${BASH_SOURCE[1]}" .sh)"
VISTA_RESULTS_OPEN_SOURCE="${VISTA_RESULTS_OPEN_SOURCE:-$REPO_ROOT/results/vista/open_source}"
OUTPUT_DIR="${VISTA_RESULTS_OPEN_SOURCE}/max_frames_${MAX_FRAMES}/${MODEL_SLUG}/longvid_reasoning_eval_vista"
LOG_SAMPLES_SUFFIX="${MODEL_SLUG}_f${MAX_FRAMES}_vista"
