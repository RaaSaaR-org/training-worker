# Testing the GR00T trainer with a real Unitree G1

End-to-end procedure: record data on the G1 → train via this worker → deploy for inference.

## 0. Prerequisites

| What | Requirement |
|---|---|
| GPU box | NVIDIA, CUDA 12.8, 40 GB+ VRAM recommended (4090 works with small batches) |
| Isaac-GR00T | `./scripts/setup-gr00t.sh ~/Isaac-GR00T` on the GPU box |
| Worker `.env` | `GR00T_REPO_DIR=~/Isaac-GR00T`, `TRAINING_DEVICE=cuda`, `TRAINER_STUB=false` |
| NeoDEM server + RustFS | reachable on ports 3001 / 9000 |
| ffmpeg | on the GPU box (dataset video decode) |

Smoke-test the wiring first (no GPU needed): `pytest tests/` in this repo runs the full
worker loop against a fake Isaac-GR00T repo.

## 1. Choose the embodiment path

There are two supported ways to run GR00T on a G1 — pick based on what the dataset controls:

**A. Manipulation / upper body (recommended first test)**
Dataset records arm + hand joints (teleop, e.g. Apple Vision Pro or [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot)).
Use `embodiment_tag: NEW_EMBODIMENT` (the default). The worker auto-generates
`modality.json` + modality config from the dataset. Works with any LeRobot v2/v3 G1 dataset.

**B. Whole-body loco-manipulation**
GR00T N1.7 has a dedicated `UNITREE_G1_SONIC` embodiment: the model emits latent action
tokens that the GEAR-SONIC whole-body controller decodes into full-body joint commands
(legs, arms, hands). Data collection and deployment follow NVIDIA's
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) workflow.
Submit jobs with:

```json
"hyperparameters": {
  "embodiment_tag": "UNITREE_G1_SONIC",
  "modality_config_path": "/home/user/Isaac-GR00T/examples/GR00TWholeBodyControl/<g1_config>.py"
}
```

(Point `modality_config_path` at the matching config in the Isaac-GR00T examples — the
auto-generated NEW_EMBODIMENT config is not valid for the SONIC latent-action space.)

## 2. Record and import a G1 dataset

1. Record episodes in LeRobot format (v2 or v3). For teleop the Unitree toolchain outputs
   LeRobot-compatible datasets directly. Aim for **≥ 50 episodes** of one task to start.
2. Make sure each episode has a task description (GR00T conditions on language).
3. Import into NeoDEM (UI → Datasets → Import, or HF import API). The worker downloads it
   from RustFS at training time; v3 datasets are converted to the GR00T v2 flavor
   automatically.
4. Optional but recommended: add a hand-written `meta/modality.json` splitting the state and
   action vectors into named parts (`left_arm`, `right_arm`, `left_hand`, …) instead of the
   auto-generated single `robot` block. See Isaac-GR00T
   `getting_started/finetune_new_embodiment.md`.

## 3. Submit the training job

Via UI (`/training` → New Training Job) or API:

```bash
curl -X POST http://<server>:3001/api/training/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "datasetId": "<dataset-id>",
    "baseModel": "gr00t-n1.7",
    "fineTuneMethod": "finetune",
    "hyperparameters": {
      "max_steps": 2000,
      "global_batch_size": 32,
      "action_horizon": 16
    }
  }'
```

Notes:
- `baseModel` anything containing `gr00t` routes to the GR00T trainer; `gr00t-n1.7`
  resolves to `nvidia/GR00T-N1.7-3B` (first run downloads ~7 GB into `HF_CACHE_DIR`).
- On a 24 GB card start with `"global_batch_size": 4` and raise until OOM.
- A quick pipeline check: `"max_steps": 50` finishes in minutes and still produces an
  artifact.
- Progress (step/loss) streams to the UI; cancel from the UI kills the fine-tune process.

Result: `s3://model-checkpoints/<jobId>/gr00t_finetune.tar.gz` containing the newest
`checkpoint-<step>` (full HF checkpoint, several GB).

## 4. Deploy to the G1

1. Download + untar the artifact on the inference machine (16 GB+ VRAM, or Jetson Thor on
   the robot).
2. Serve it with Isaac-GR00T's inference service, pointing at the checkpoint dir and the
   same modality config used for training:
   `uv run python gr00t/eval/run_gr00t_server.py --model-path <checkpoint-dir> ...`
   (see Isaac-GR00T `getting_started` deployment docs for the exact runner in your
   version).
3. Path A (manipulation): bridge server actions to the G1 with the Unitree SDK2 /
   unitree_IL_lerobot client — joint commands come back in the dataset's action order.
4. Path B (whole-body): deploy through GR00T-WholeBodyControl, which runs the SONIC
   controller on the robot and consumes the model's latent actions.

**Safety**: first run with the G1 in a harness / e-stop within reach; verify action
scaling on a replayed episode before closed-loop control.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Job fails `GR00T_REPO_DIR is not set` | Set it in the worker's `.env` on the GPU box |
| `requires CUDA, but TRAINING_DEVICE=...` | `TRAINING_DEVICE=cuda` — GR00T needs flash-attn, no MPS/CPU |
| CUDA OOM in the log tail | lower `global_batch_size`, reduce `action_horizon` |
| `Cannot auto-generate modality.json` | dataset lacks `observation.state`/`action` — ship a `meta/modality.json` |
| v3→v2 conversion fails | check the job error log tail; converter needs `ffmpeg` |
| Loss flat / actions wrong at deploy | write per-limb `modality.json` + config instead of the auto single-block one; check `action_representation` |
