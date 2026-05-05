#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$REPO_ROOT/scripts/ytb/open_source/mimo_vl_7b_think.sh" "$@"
