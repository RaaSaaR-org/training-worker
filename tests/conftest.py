"""Shared fixtures — a fake LeRobot v2 dataset and a fake Isaac-GR00T repo.

The fake repo's launch_finetune.py mimics the real one's observable
behaviour: tqdm + HF-Trainer log lines on stdout, checkpoint-<step>/
output dir, non-zero exit on failure. This lets the Gr00tTrainer's
subprocess handling, progress parsing, cancellation, and artifact
packaging run for real — without CUDA or the gr00t package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the worker modules importable when pytest runs from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


_FAKE_LAUNCH_FINETUNE = '''
import argparse
import pathlib
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--base-model-path")
p.add_argument("--dataset-path")
p.add_argument("--embodiment-tag")
p.add_argument("--modality-config-path")
p.add_argument("--num-gpus")
p.add_argument("--output-dir")
p.add_argument("--max-steps", type=int)
p.add_argument("--global-batch-size")
p.add_argument("--dataloader-num-workers")
p.add_argument("--fail", action="store_true")
args = p.parse_args()

if args.fail:
    print("Traceback: something exploded (CUDA out of memory)")
    sys.exit(1)

out = pathlib.Path(args.output_dir)
for step in range(1, args.max_steps + 1):
    pct = step * 100 // args.max_steps
    print(f" {pct}%|██        | {step}/{args.max_steps} [00:01<00:01,  2.00it/s]", flush=True)
    if step % 2 == 0:
        loss = round(1.0 / step, 4)
        print({"loss": loss, "grad_norm": 1.0,
               "learning_rate": 5e-05, "epoch": step / args.max_steps}, flush=True)
    time.sleep(0.02)

ck = out / f"checkpoint-{args.max_steps}"
ck.mkdir(parents=True, exist_ok=True)
(ck / "model.safetensors").write_bytes(b"fake-weights")
(ck / "config.json").write_text("{}")
print("Training completed", flush=True)
'''


@pytest.fixture
def dataset_v2(tmp_path: Path) -> Path:
    """Minimal GR00T-compatible LeRobot v2.1 dataset (Unitree-G1-shaped)."""
    root = tmp_path / "dataset-src"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [23]},
            "action": {"dtype": "float32", "shape": [23]},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist_left": {"dtype": "video", "shape": [480, 640, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"PAR1")
    return root


@pytest.fixture
def fake_gr00t_repo(tmp_path: Path) -> Path:
    """Directory that passes the trainer's Isaac-GR00T clone checks."""
    repo = tmp_path / "Isaac-GR00T"
    script = repo / "gr00t" / "experiment" / "launch_finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text(_FAKE_LAUNCH_FINETUNE)
    converter = repo / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("raise SystemExit(0)\n")
    return repo


@pytest.fixture
def gr00t_env(monkeypatch: pytest.MonkeyPatch, fake_gr00t_repo: Path) -> Path:
    """Point the trainer at the fake repo, run scripts with this python."""
    monkeypatch.setenv("GR00T_REPO_DIR", str(fake_gr00t_repo))
    monkeypatch.setenv("GR00T_PYTHON", sys.executable)
    monkeypatch.setenv("GR00T_ALLOW_NON_CUDA", "1")
    monkeypatch.setenv("GR00T_PROGRESS_POLL_SEC", "0.1")
    return fake_gr00t_repo
