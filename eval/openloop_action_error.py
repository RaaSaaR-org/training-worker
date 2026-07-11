#!/usr/bin/env python3
"""
@file openloop_action_error.py
@description Open-loop action-error evaluation for a served SmolVLA policy.

Closes the train -> serve -> evaluate loop WITHOUT a physics sim: it replays
recorded (held-out) frames from a LeRobot dataset through the deployed policy
(vla-server ``/predict``) and compares the policy's predicted action against the
logged ground-truth action. This measures whether the trained policy reproduces
the demonstrated behaviour (per-dimension L1 / MSE). It is a smoke signal, not a
task-success metric — that would need a manipulation sim (see the README).

Usage:
    python eval/openloop_action_error.py \
        --dataset-root /path/to/local/lerobot/dataset \
        --server http://localhost:8000 \
        --episodes 0 1 --max-frames 40 --task "Pick up the apple"
"""
from __future__ import annotations

import argparse
import base64
import io
import sys

import numpy as np
import requests

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("Pillow is required: uv pip install pillow", file=sys.stderr)
    raise

# LeRobot 0.4+ path, with a pre-0.4 fallback (mirrors the trainer's import guard).
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from trainers.base import default_video_backend


def _to_uint8_hwc(img) -> np.ndarray:
    """Normalize a LeRobot image item to an (H, W, 3) uint8 array.

    LeRobot returns decoded frames as torch tensors shaped (C, H, W), float in
    [0, 1]. Some paths yield (H, W, C) uint8 already. Handle both.
    """
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 + 1e-6 else arr / 255.0
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _jpeg_b64(arr_hwc_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_hwc_uint8).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _short_cam_name(feature_key: str) -> str:
    """`observation.images.cam_left_high` -> `cam_left_high`."""
    return feature_key.split(".")[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, help="Local dir of the v3.0 LeRobot dataset")
    ap.add_argument("--dataset-revision", default="v3.0")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--auth-token", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--max-frames", type=int, default=40, help="Frames sampled across the episodes")
    ap.add_argument("--task", default=None,
                    help="Override the language instruction; default uses each episode's own task")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.auth_token}"} if args.auth_token else {}

    # --- Load the dataset (from a local copy of the RustFS-hosted dataset) -----
    ds = LeRobotDataset(
        "local/dex3-eval",
        root=args.dataset_root,
        revision=args.dataset_revision,
        video_backend=default_video_backend(),
    )
    feats = ds.meta.features
    image_keys = [k for k, v in feats.items() if v.get("dtype") == "video" or v.get("dtype") == "image"]
    if not image_keys:
        image_keys = [k for k in feats if k.startswith("observation.images.")]
    state_key = "observation.state"
    action_key = "action"
    print(f"dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"image features: {image_keys}")
    print(f"state dim: {feats[state_key]['shape']}  action dim: {feats[action_key]['shape']}")

    # --- Ask the server what it expects -------------------------------------
    cfg = requests.get(f"{args.server}/config", headers=headers, timeout=10).json()
    server_cams = cfg.get("cameras", [])
    action_dim = cfg.get("action_dim")
    print(f"server: action_dim={action_dim} chunk={cfg.get('chunk_size')} cameras={server_cams}")

    # Map each server-expected camera name to a dataset image key by suffix.
    def match_key(cam: str) -> str | None:
        for k in image_keys:
            if _short_cam_name(k) == _short_cam_name(cam) or k == cam:
                return k
        return image_keys[0] if image_keys else None

    cam_to_key = {cam: match_key(cam) for cam in server_cams} if server_cams else {
        _short_cam_name(k): k for k in image_keys
    }

    # --- Build the frame index set (spread across requested episodes) --------
    # LeRobot 0.5 exposes per-episode frame ranges + language on ds.meta.episodes
    # (a HF Dataset with columns dataset_from_index / dataset_to_index / tasks).
    ep_table = ds.meta.episodes
    per_ep = max(1, args.max_frames // max(1, len(args.episodes)))
    frames: list[tuple[int, str]] = []  # (frame_id, task)
    for ep in args.episodes:
        if ep >= ds.num_episodes:
            continue
        row = ep_table[ep]
        f, t = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        ep_tasks = row.get("tasks") or []
        # Prefer the episode's own language instruction; --task overrides if given.
        task = args.task or (ep_tasks[0] if ep_tasks else "Do the task.")
        step = max(1, (t - f) // per_ep)
        for fid in range(f, t, step):
            frames.append((fid, task))
    frames = frames[: args.max_frames]
    tasks_seen = sorted({tk for _, tk in frames})
    print(f"evaluating {len(frames)} frames from episodes {args.episodes}")
    print(f"tasks: {tasks_seen}\n")

    # --- Replay through the policy ------------------------------------------
    l1_all, mse_all = [], []
    for n, (fid, task) in enumerate(frames):
        item = ds[fid]
        images = {cam: _jpeg_b64(_to_uint8_hwc(item[key])) for cam, key in cam_to_key.items()}
        state = np.asarray(item[state_key]).astype(float).ravel().tolist()
        gt = np.asarray(item[action_key]).astype(float).ravel()

        body = {"images": images, "state": state, "task": task}
        r = requests.post(f"{args.server}/predict", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        pred = np.asarray(r.json()["actions"][0], dtype=float)  # first action of the chunk

        m = min(len(pred), len(gt))
        l1 = float(np.mean(np.abs(pred[:m] - gt[:m])))
        mse = float(np.mean((pred[:m] - gt[:m]) ** 2))
        l1_all.append(l1)
        mse_all.append(mse)
        if n < 5 or n % 10 == 0:
            print(f"  frame {fid:>6}: L1={l1:.4f}  MSE={mse:.4f}")

    print("\n=== open-loop action error (predicted vs. logged) ===")
    print(f"frames evaluated : {len(l1_all)}")
    print(f"mean L1          : {np.mean(l1_all):.4f}")
    print(f"mean MSE         : {np.mean(mse_all):.4f}")
    print("(lower = policy better reproduces the demonstrations; this is a smoke")
    print(" signal — task success needs a manipulation sim, not open-loop replay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
