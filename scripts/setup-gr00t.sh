#!/usr/bin/env bash
# setup-gr00t.sh — prepare NVIDIA Isaac-GR00T for the GR00T trainer.
#
# Run this ON THE GPU MACHINE (CUDA required; fine-tuning wants 40GB+ VRAM,
# H100/L40 recommended — a 4090 works with small batch sizes).
#
# Usage:
#   ./scripts/setup-gr00t.sh [target-dir]     # default: ~/Isaac-GR00T
#
# Afterwards set in the worker's .env:
#   GR00T_REPO_DIR=<target-dir>
#   TRAINING_DEVICE=cuda

set -euo pipefail

GR00T_DIR="${1:-$HOME/Isaac-GR00T}"

if ! command -v uv >/dev/null; then
  echo "▶ Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -d "$GR00T_DIR/.git" ]; then
  echo "▶ Updating existing clone at $GR00T_DIR"
  git -C "$GR00T_DIR" pull --ff-only
else
  echo "▶ Cloning Isaac-GR00T to $GR00T_DIR"
  git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_DIR"
fi

# ffmpeg is needed for dataset video handling on dGPU systems
if ! command -v ffmpeg >/dev/null; then
  echo "⚠ ffmpeg not found — install it (e.g. sudo apt-get install -y ffmpeg)"
fi

echo "▶ Syncing Isaac-GR00T environment (Python 3.10 + CUDA deps, takes a while)"
cd "$GR00T_DIR"
uv sync --python 3.10

echo
echo "✓ Done. Add to training-worker/.env:"
echo "    GR00T_REPO_DIR=$GR00T_DIR"
echo "    TRAINING_DEVICE=cuda"
