# LeRobot 0.6.0 on the GPU machine — GR00T N1.7 + world models

Setup for the CUDA box: `./scripts/setup-lerobot-gpu.sh` (creates `~/lerobot-gpu/.venv`).
Release notes: <https://huggingface.co/blog/lerobot-release-v060>

## Why 0.6.0

- **On PyPI** — no more source installs. Lightweight base; everything else via extras.
- **GR00T N1.7 native** (`lerobot[groot]`): Cosmos-Reason2-2B VLM + flow matching,
  runs *in-process* in LeRobot — no Isaac-GR00T clone, no ZMQ PolicyServer.
- **World-model policies** — predict future states during training:

  | Policy | Extra | Idea | VRAM |
  |---|---|---|---|
  | VLA-JEPA | `vla-jepa` | Latent future prediction (Qwen3-VL-2B); world model drops away at deployment → zero inference cost | small |
  | LingBot-VA | `lingbot-va` | Autoregressive video+action prediction | 24–32 GB |
  | FastWAM | `fastwam` | ~5B video expert + compact action expert, direct action denoising (no test-time dreaming) | large |

- **Reward models** (`lerobot.rewards`): Robometer (pretrained Qwen3-VL-4B, per-frame
  task-progress from video+language) and TOPReward (zero-shot VLM log-probs).
  Both give progress curves per episode → useful for the NeoDEM Evaluate stage
  and dataset curation.

## Relation to the existing GR00T flow

- `trainers/gr00t_n1.py` shells out to a local **Isaac-GR00T** clone
  (`scripts/setup-gr00t.sh`, `GR00T_REPO_DIR`, Python 3.10 env) — **unaffected**
  by this upgrade; keep using it until the native path is validated.
- vla-server's GR00T backend speaks ZMQ to the Isaac-GR00T PolicyServer —
  also unaffected.
- The native path may eventually retire both (and the v3→v2 dataset
  converter, since LeRobot-native GR00T consumes LeRobotDataset directly).

## Breaking changes to know

- **GR00T N1.5 removed** from LeRobot; N1.7 replaces it (pin `lerobot==0.5.1`
  if N1.5 is ever needed). Isaac-GR00T (separate repo) is not affected.
- Base install no longer ships `datasets` etc. — use extras
  (`lerobot[dataset]`, `[training]`, …). Our Mac venvs (vla-server,
  training-worker) run `lerobot[smolvla,dataset]==0.6.0` since 2026-07-07.
- Requires torch 2.7–2.11; Linux wheels pinned to CUDA 12.8.

## Quick usage sketches

```bash
source ~/lerobot-gpu/.venv/bin/activate

# GR00T N1.7 fine-tune on a LeRobotDataset (no v3→v2 conversion needed)
lerobot-train --policy.type=groot_n1_7 \
  --dataset.repo_id=<org>/<dataset> \
  --policy.dtype=bfloat16            # mixed precision via Accelerate

# Reward-model progress curves over an episode (dataset inspection / eval)
python -c "from lerobot.rewards import ...  # Robometer / TOPReward — see release notes"

# Benchmarks (LIBERO-plus, RoboTwin 2.0, RoboCasa365, ...) — unified CLI
lerobot-eval --help
```

Also relevant for us later: `lerobot-rollout` (sentry/highlight/dagger deployment
strategies), FSDP multi-GPU training, depth-video support in datasets, and
`lerobot-annotate` (VLM-generated subtask/VQA annotations).
