# CLAUDE.md

NeoDEM training worker — extracted from `robot-management-system/training-worker/`.

## What This Is

Python worker that polls the NeoDEM server for training jobs, runs fine-tuning (SmolVLA LoRA in-process, GR00T N1.x via Isaac-GR00T subprocess), and streams progress back over HTTP. Designed for GPU machines (CUDA) or Mac (MPS; SmolVLA only).

## Key Files

- `worker.py` — main polling loop; picks the trainer per job from `baseModel`
- `trainers/` — SmolVLA LoRA trainer, GR00T N1.x trainer (`gr00t_n1.py`), stub trainer
- `stats_worker.py` — dataset stats computation (NATS consumer)
- `storage.py` — S3/RustFS client
- `callbacks.py` — HTTP callbacks to server
- `config.py` — environment config
- `scripts/test-e2e.sh` — E2E test script
- `scripts/setup-gr00t.sh` — clone + sync Isaac-GR00T on a CUDA box
- `docs/gr00t-unitree-g1.md` — how to train/deploy GR00T on a real Unitree G1

## Dev Commands

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
TRAINER_STUB=true uv run python worker.py   # no GPU needed
pytest
ruff check .
```

## Conventions

- Talks to NeoDEM server on port 3001 and RustFS on port 9000
- Config via env vars (`NEODEM_SERVER_URL`, `TRAINING_DEVICE`, `TRAINER_STUB`, etc.)
- LeRobot must be installed from source for real SmolVLA training
- GR00T jobs (`baseModel` contains `gr00t`) shell out to an Isaac-GR00T clone
  (`GR00T_REPO_DIR`, own uv env with Python 3.10) — CUDA only, never imported in-process
- Two entry points: `worker.py` (training) and `stats_worker.py` (dataset stats)
- Tests in `tests/` run without GPU/ML deps (fake Isaac-GR00T repo fixture)
