# CLAUDE.md

NeoDEM training worker — extracted from `robot-management-system/training-worker/`.

## What This Is

Python worker that polls the NeoDEM server for training jobs, runs SmolVLA LoRA fine-tuning, and streams progress back over HTTP. Designed for GPU machines (CUDA) or Mac (MPS).

## Key Files

- `worker.py` — main polling loop
- `trainers/` — SmolVLA LoRA trainer + stub trainer
- `stats_worker.py` — dataset stats computation (NATS consumer)
- `storage.py` — S3/RustFS client
- `callbacks.py` — HTTP callbacks to server
- `config.py` — environment config
- `scripts/test-e2e.sh` — E2E test script

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
- LeRobot must be installed from source for real training
- Two entry points: `worker.py` (training) and `stats_worker.py` (dataset stats)
