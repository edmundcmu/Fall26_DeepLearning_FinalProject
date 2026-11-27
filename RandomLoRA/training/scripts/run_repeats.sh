#!/usr/bin/env bash
set -u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate uma
cd "$HOME/uma_lora_project"
export PYTHONPATH="$PWD"
for mask in 42 43; do
  run="runs/random/seed41_mask${mask}"
  python -u "scripts/train_random_mask${mask}.py" > "$run/console.log" 2>&1
  rc=$?
  echo "TRAIN_EXIT_CODE=$rc" >> "$run/console.log"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  python -u "scripts/evaluate_random_mask${mask}.py" > "$run/evaluation_console.log" 2>&1
  rc=$?
  echo "EVAL_EXIT_CODE=$rc" >> "$run/evaluation_console.log"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
echo REPEATS_COMPLETE > runs/random/repeats_complete.txt
