#!/usr/bin/env bash
# Run from a versioned release directory, e.g. cd /home/ubuntu/uma-releases/<SHA>
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse HEAD)
RUN_ID="baseline_${SHA:0:9}_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="/home/ubuntu/uma-data/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export MPLBACKEND=Agg
# Ensure an environment where torch, fairchem-core and dependencies are installed is activated first.
(
  cd "$ROOT"
  echo "git_commit=$SHA"
  python -c 'import torch, fairchem.core; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())'
  python scripts/uma_cuh_uma_m_baseline.py --workdir "$RUN_DIR" --calibration 12 --test 30
) 2>&1 | tee "$RUN_DIR/baseline.log"
printf '%s\n' "$SHA" > "$RUN_DIR/git_commit.txt"
python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
if [[ -n "${S3_RESULTS_URI:-}" ]]; then
  aws s3 sync "$RUN_DIR" "${S3_RESULTS_URI%/}/$RUN_ID/"
fi
echo "Finished. Results in $RUN_DIR/results"
