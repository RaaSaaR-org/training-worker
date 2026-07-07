#!/usr/bin/env bash
# setup-lerobot-gpu.sh — prepare LeRobot ≥0.6.0 on the GPU machine for
# GR00T N1.7 (native), world-model policies, and reward models.
#
# Run this ON THE GPU MACHINE (Linux + CUDA; LeRobot 0.6.0 pins CUDA 12.8
# wheels, torch 2.7–2.11). See docs/lerobot-0.6-gpu.md for VRAM guidance.
#
# This is the NATIVE LeRobot path (in-process GR00T N1.7). It coexists with
# the Isaac-GR00T shell-out flow (scripts/setup-gr00t.sh + PolicyServer/ZMQ)
# — set up either or both.
#
# Usage:
#   ./scripts/setup-lerobot-gpu.sh [target-dir]   # default: ~/lerobot-gpu

set -euo pipefail

LEROBOT_DIR="${1:-$HOME/lerobot-gpu}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.0}"

if ! command -v uv >/dev/null; then
  echo "▶ Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$LEROBOT_DIR"
cd "$LEROBOT_DIR"

if [ ! -d .venv ]; then
  echo "▶ Creating venv (Python 3.12)"
  uv venv --python 3.12
fi

echo "▶ Installing lerobot==$LEROBOT_VERSION with GR00T + world-model + reward extras"
# groot      — GR00T N1.7 (Cosmos-Reason2-2B VLM, flow matching)
# vla-jepa   — world model: latent future prediction (Qwen3-VL-2B, free at inference)
# lingbot-va — world model: autoregressive video+action (24–32GB GPU)
# fastwam    — world model: ~5B video expert + compact action expert
# robometer  — pretrained reward model (Qwen3-VL-4B, per-frame progress)
# topreward  — zero-shot VLM log-prob rewards
# dataset/training/evaluation — LeRobotDataset, trainer, lerobot-eval benchmarks
VIRTUAL_ENV="$PWD/.venv" uv pip install \
  "lerobot[groot,vla-jepa,lingbot-va,fastwam,robometer,topreward,dataset,training,evaluation]==$LEROBOT_VERSION"

# Optional but recommended for GR00T N1.7 on Ampere+:
echo "▶ Installing flash-attention (optional, may take a while to build)"
VIRTUAL_ENV="$PWD/.venv" uv pip install flash-attn --no-build-isolation \
  || echo "⚠ flash-attn install failed — GR00T N1.7 still works without it (slower)"

echo "▶ Verifying imports"
"$PWD/.venv/bin/python" - <<'PY'
import lerobot, torch
print(f"lerobot {lerobot.__version__} | torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
for mod in [
    "lerobot.datasets.lerobot_dataset",
    "lerobot.policies.factory",
]:
    __import__(mod)
print("core imports OK — see docs/lerobot-0.6-gpu.md for policy/reward usage")
PY

echo "✓ Done. Activate with: source $LEROBOT_DIR/.venv/bin/activate"
