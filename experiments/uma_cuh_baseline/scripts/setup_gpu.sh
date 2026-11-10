#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run in a NEW Linux GPU environment; do not change a shared course environment.
mkdir -p external
if [ ! -d external/fairchem/.git ]; then
  git clone https://github.com/facebookresearch/fairchem.git external/fairchem
fi
git -C external/fairchem checkout --detach 645af7d03e91138ef07e51de387316c9e2e152dc
python -m pip install -r requirements-data.txt
python -m pip install -e 'external/fairchem/packages/fairchem-core[dev]'
python -m pip freeze > reports/gpu_environment_lock.txt
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available()'
