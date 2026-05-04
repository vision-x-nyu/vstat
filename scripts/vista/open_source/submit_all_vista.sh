#!/bin/bash
# Queue every open_source VISTA evaluator at multiple max-frame budgets via task-spooler (ts).
#
# Usage: bash submit_all_vista.sh
# Requires: GNU task-spooler (`ts`) on PATH.
#
# Optional env:
#   VISTA_OPEN_SOURCE_FRAMES — space-separated frame counts (default: 16 32 64 128).
#   VISTA_MANIFEST — path to the run log TSV (default: $REPO_ROOT/results/vista/open_source/run_manifest.tsv).
# Launcher env is passed through by each child job.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST="${VISTA_MANIFEST:-$REPO_ROOT/results/vista/open_source/run_manifest.tsv}"
if [[ -n "${VISTA_OPEN_SOURCE_FRAMES:-}" ]]; then
  read -r -a FRAMES_SWEEP <<<"${VISTA_OPEN_SOURCE_FRAMES}"
else
  FRAMES_SWEEP=(16 32 64 128)
fi

MODEL_SCRIPTS=(
  qwen3vl_8b.sh
  qwen3vl_4b.sh
  qwen3vl_2b.sh
  internvl3p5_2b.sh
  internvl3p5_8b.sh
  cambrians_7b.sh
  cambrians_3b.sh
  cambrians_1p5b.sh
  llava_onevision_7b.sh
  llava_onevision_0.5b.sh
)

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'timestamp\tevent\tmax_frames\tmodel_script\toutput_dir\n' >"$MANIFEST"
fi

append_manifest() {
  local event="$1" nf="$2" script_base="$3" out_dir="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$event" "$nf" "$script_base" "$out_dir" >>"$MANIFEST"
}

echo "Queuing ${#MODEL_SCRIPTS[@]} models x ${#FRAMES_SWEEP[@]} max_frames for VISTA (longvid-reasoning-eval_vista, open_source sweep)..."

for nf in "${FRAMES_SWEEP[@]}"; do
  for s in "${MODEL_SCRIPTS[@]}"; do
    out="${REPO_ROOT}/results/vista/open_source/max_frames_${nf}/$(basename "$s" .sh)/longvid_reasoning_eval_vista"
    append_manifest queued "$nf" "$s" "$out"
    ts bash -c "export MAX_FRAMES=${nf}; bash \"${SCRIPT_DIR}/${s}\""
  done
done

echo "Queued $(( ${#MODEL_SCRIPTS[@]} * ${#FRAMES_SWEEP[@]} )) jobs. Manifest: $MANIFEST"
echo "Run ts to inspect the queue."
