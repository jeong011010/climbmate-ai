#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "🧹 Cleaning workspace artifacts... (root=$ROOT_DIR)"

# Candidate paths for deletion
DEL_PATHS=(
  "outputs"
  "venv_new"
  "holdcheck/dataset"
  "holdcheck/runs"
  "holdcheck/roboflow_weights"
)

for p in "${DEL_PATHS[@]}"; do
  if [ -e "$p" ]; then
    echo " - removing $p"
    rm -rf "$p"
  fi
done

# Remove top-level extra weights except required ones
find "$ROOT_DIR" -maxdepth 1 -type f -name '*.pt' \
  ! -name 'weights-keep.pt' \
  -print -exec rm -f {} \; || true

echo "✅ Cleanup done."


