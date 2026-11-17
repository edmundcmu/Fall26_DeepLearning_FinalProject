#!/usr/bin/env bash
# Called as ubuntu by AWS SSM, deploys a pinned GitHub commit without changing running experiments.
set -euo pipefail
SHA="${1:-}"
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected a full 40-character Git commit SHA" >&2; exit 2
fi
REPO=/home/ubuntu/uma-project
RELEASES=/home/ubuntu/uma-releases
mkdir -p "$RELEASES"
git -C "$REPO" fetch origin main
# Only deploy code reachable from main, and never modify a running experiment's worktree.
git -C "$REPO" cat-file -e "${SHA}^{commit}"
git -C "$REPO" merge-base --is-ancestor "$SHA" origin/main
if [[ ! -d "$RELEASES/$SHA" ]]; then
  git -C "$REPO" worktree add --detach "$RELEASES/$SHA" "$SHA"
fi
ln -sfn "$RELEASES/$SHA" /home/ubuntu/uma-current
printf 'Synced commit %s\nUse /home/ubuntu/uma-releases/%s for reproducible runs.\n' "$SHA" "$SHA"
