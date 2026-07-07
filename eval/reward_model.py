"""Reward-model evaluation runner — Robometer / TOPReward episode scoring.

Runs a LeRobot 0.6.0 reward model (``lerobot.rewards``) over the episodes
of a LeRobot v3 dataset and produces per-episode progress curves
(TASK-179 Phase 1 — the replacement evaluation signal after the Cosmos3
NO-GO in TASK-176).

Job contract (kind=``reward_model``):
  - ``baseModel``: ``"robometer"`` | ``"topreward"`` (mirrors rewardType)
  - ``hyperparameters``: ``{ rewardType, episodes?, task?, imageKey?, maxFrames? }``
  - ``finalMetrics``: ``{ kind: 'reward_model', rewardType,
        rewards: [{ episodeIndex, score, success, curve, fps }] }``

Curve semantics: for each episode we evaluate the reward model on growing
frame prefixes at up to ``curvePoints`` (default 16) evenly spaced
endpoints; each prefix is subsampled to at most ``maxFrames`` frames. The
per-entry ``fps`` is the *curve's* sampling rate (curve points per second
of episode time), so ``t(curve[j]) ≈ (j + 1) / fps``. ``score`` is the
final curve value; ``success`` compares it to ``successThreshold``
(default 0.5).

Requires the ``lerobot[robometer,topreward]`` model extras (see the
pyproject `rewards` extra and scripts/setup-lerobot-gpu.sh). Imports are
lazy; a missing stack fails the job with an actionable error.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from trainers.base import (
    BaseTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    TrainerContext,
    TrainerResult,
)

log = logging.getLogger(__name__)

_SUPPORTED_REWARD_TYPES = ("robometer", "topreward")

_MISSING_REWARDS_HINT = (
    "Reward-model evaluation needs the lerobot[robometer,topreward] extras. "
    "Install with `uv pip install -e '.[rewards]'` in training-worker/ "
    "(or run scripts/setup-lerobot-gpu.sh on the GPU machine)."
)

# Default frame budget per reward-model window (mirrors the lerobot
# config defaults: RobometerConfig.max_frames=8, TOPRewardConfig.max_frames=16).
_DEFAULT_MAX_FRAMES = {"robometer": 8, "topreward": 16}


class RewardModelRunner(BaseTrainer):
    """Score dataset episodes with a VLM reward model (progress curves)."""

    def train(
        self,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
    ) -> TrainerResult:
        started = time.monotonic()
        hp = ctx.hyperparameters

        reward_type = str(
            hp.get("rewardType") or hp.get("reward_type") or ctx.base_model or ""
        ).lower()
        if reward_type not in _SUPPORTED_REWARD_TYPES:
            raise RuntimeError(
                f"Unsupported rewardType {reward_type!r} — expected one of "
                f"{list(_SUPPORTED_REWARD_TYPES)}"
            )

        # 1) Dataset — download from RustFS + load via LeRobot (v3, local).
        dataset_dir = ctx.work_dir / "dataset"
        self._download_dataset(ctx, dataset_dir)
        dataset = self._load_lerobot_dataset(ctx, dataset_dir)

        episodes = self._resolve_episodes(hp, dataset)
        image_key = self._resolve_image_key(hp, dataset)
        max_frames = int(
            hp.get("maxFrames")
            or hp.get("max_frames")
            or _DEFAULT_MAX_FRAMES[reward_type]
        )
        curve_points = int(hp.get("curvePoints") or hp.get("curve_points") or 16)
        threshold = float(hp.get("successThreshold") or hp.get("success_threshold") or 0.5)
        task_override = str(hp.get("task") or "").strip() or None

        log.info(
            "[RewardModel] job=%s type=%s episodes=%s imageKey=%s maxFrames=%d points=%d",
            ctx.job_id,
            reward_type,
            episodes,
            image_key,
            max_frames,
            curve_points,
        )

        # 2) Reward model + preprocessing pipeline via the lerobot factories.
        model, preprocess = self._load_reward_model(
            ctx, reward_type, image_key=image_key, max_frames=max_frames
        )

        # 3) Score each episode: progress curve over growing frame prefixes.
        rewards: list[dict[str, Any]] = []
        for i, ep in enumerate(episodes):
            task = task_override or self._episode_task(dataset, ep)
            curve, curve_fps = self._score_episode(
                dataset,
                ep,
                model=model,
                preprocess=preprocess,
                image_key=image_key,
                task=task,
                max_frames=max_frames,
                curve_points=curve_points,
            )
            score = curve[-1] if curve else 0.0
            rewards.append(
                {
                    "episodeIndex": int(ep),
                    "score": round(score, 6),
                    "success": bool(score >= threshold) if curve else None,
                    "curve": [round(v, 6) for v in curve],
                    "fps": curve_fps,
                }
            )
            event = ProgressEvent(
                step=i + 1,
                total_steps=len(episodes),
                epoch=1,
                total_epochs=1,
                # No training loss here — surface the episode score instead
                # so the UI progress stream shows a live signal.
                loss=round(score, 6),
                learning_rate=0.0,
            )
            if not on_progress(event):
                raise CancelledError(f"cancelled after episode {ep}")

        # 4) rewards.json artifact + final metrics (contract shape).
        final_metrics = {
            "kind": "reward_model",
            "rewardType": reward_type,
            "rewards": rewards,
        }
        artifact_path = ctx.work_dir / "rewards.json"
        artifact_path.write_text(json.dumps(final_metrics, indent=2))
        log.info(
            "[RewardModel] scored %d episode(s) in %.1fs — artifact %s",
            len(rewards),
            time.monotonic() - started,
            artifact_path,
        )
        return TrainerResult(artifact_path=artifact_path, final_metrics=final_metrics)

    # ====================================================================
    # INTERNALS
    # ====================================================================

    def _download_dataset(self, ctx: TrainerContext, dest: Path) -> None:
        """Pull dataset Parquet + meta/ from RustFS (same as smolvla_lora)."""
        from storage import StorageClient

        storage = StorageClient(
            endpoint=os.environ.get("RUSTFS_ENDPOINT", "http://localhost:9000"),
            access_key=os.environ.get("RUSTFS_ACCESS_KEY", "rustfsadmin"),
            secret_key=os.environ.get("RUSTFS_SECRET_KEY", "rustfsadmin"),
            dataset_bucket=os.environ.get("RUSTFS_BUCKET_DATASETS", "datasets"),
            model_bucket=os.environ.get("RUSTFS_BUCKET_MODELS", "models"),
        )
        storage.download_dataset(ctx.dataset_storage_path, dest)

    def _locate_dataset_root(self, base: Path) -> Path:
        if (base / "meta" / "info.json").exists():
            return base
        candidates = sorted(p.parent.parent for p in base.rglob("meta/info.json"))
        if not candidates:
            raise FileNotFoundError(f"LeRobot meta/info.json not found under {base}")
        return candidates[0]

    def _load_lerobot_dataset(self, ctx: TrainerContext, dataset_dir: Path) -> Any:
        """Load the LeRobot v3 dataset from the downloaded local directory."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_REWARDS_HINT} (import failed: {e})") from e

        root = self._locate_dataset_root(dataset_dir)
        revision = ctx.dataset_lerobot_version or "v3.0"
        return LeRobotDataset(
            repo_id=f"local/{ctx.dataset_id}",
            root=str(root),
            revision=revision,
            download_videos=True,
        )

    def _resolve_episodes(self, hp: dict[str, Any], dataset: Any) -> list[int]:
        """hyperparameters.episodes, or all episodes when absent/empty."""
        total = int(dataset.meta.total_episodes)
        episodes = [int(e) for e in (hp.get("episodes") or [])]
        if not episodes:
            return list(range(total))
        bad = [e for e in episodes if e < 0 or e >= total]
        if bad:
            raise RuntimeError(
                f"episodes {bad} out of range — dataset has {total} episode(s) "
                f"(valid indices 0..{total - 1})"
            )
        return episodes

    def _resolve_image_key(self, hp: dict[str, Any], dataset: Any) -> str:
        """hyperparameters.imageKey, or the dataset's first camera key."""
        image_key = str(hp.get("imageKey") or hp.get("image_key") or "").strip()
        if image_key:
            return image_key
        camera_keys = list(dataset.meta.camera_keys)
        if not camera_keys:
            raise RuntimeError("Dataset has no camera keys — cannot run a VLM reward model")
        return camera_keys[0]

    def _episode_task(self, dataset: Any, ep: int) -> str:
        """The episode's own language instruction from meta/episodes, else ''."""
        try:
            row = dataset.meta.episodes[ep]
            tasks = row.get("tasks") or []
            if tasks:
                return str(tasks[0])
        except Exception:  # noqa: BLE001 — task metadata is best-effort
            pass
        return ""

    def _episode_frame_range(self, dataset: Any, ep: int) -> tuple[int, int]:
        """Absolute [from, to) frame indices for one episode (v3 meta)."""
        row = dataset.meta.episodes[ep]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])

    def _load_reward_model(
        self,
        ctx: TrainerContext,
        reward_type: str,
        image_key: str,
        max_frames: int,
    ) -> tuple[Any, Any]:
        """Build (model, pre-processing pipeline) via lerobot.rewards.factory.

        Uses ``make_reward_model_config`` / ``make_reward_model`` /
        ``make_reward_pre_post_processors``. Missing model deps (the
        robometer/topreward extras are not in the base install) fail with
        a clear install hint instead of a bare ImportError.
        """
        try:
            from lerobot.rewards.factory import (
                make_reward_model,
                make_reward_model_config,
                make_reward_pre_post_processors,
            )
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_REWARDS_HINT} (import failed: {e})") from e

        os.environ.setdefault("HF_HOME", str(ctx.hf_cache_dir))
        try:
            cfg = make_reward_model_config(
                reward_type,
                device=ctx.device,
                image_key=image_key,
                task_key="task",
                max_frames=max_frames,
            )
            model = make_reward_model(cfg)
            preprocess, _postprocess = make_reward_pre_post_processors(cfg)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(f"{_MISSING_REWARDS_HINT} (import failed: {e})") from e
        model.eval()
        return model, preprocess

    def _score_episode(
        self,
        dataset: Any,
        ep: int,
        *,
        model: Any,
        preprocess: Any,
        image_key: str,
        task: str,
        max_frames: int,
        curve_points: int,
    ) -> tuple[list[float], float | None]:
        """Progress curve for one episode: reward over growing frame prefixes.

        Evaluates at up to ``curve_points`` evenly spaced prefix endpoints;
        each prefix window is subsampled to at most ``max_frames`` frames.
        Frames are fetched once and cached across windows. Returns
        ``(curve, curve_fps)`` where curve_fps maps curve indices to
        episode time (points per second).
        """
        import torch

        start, end = self._episode_frame_range(dataset, ep)
        n_frames = end - start
        if n_frames <= 0:
            return [], None

        points = max(1, min(curve_points, n_frames))
        # Prefix endpoints: last frame of each evenly spaced prefix.
        endpoints = sorted(
            {start + max(0, round((j + 1) * n_frames / points) - 1) for j in range(points)}
        )

        frame_cache: dict[int, Any] = {}

        def frame(fid: int) -> Any:
            if fid not in frame_cache:
                frame_cache[fid] = torch.as_tensor(dataset[fid][image_key])
            return frame_cache[fid]

        curve: list[float] = []
        with torch.no_grad():
            for endpoint in endpoints:
                window = self._sample_window(start, endpoint, max_frames)
                # (1, T, C, H, W): ONE trajectory of T frames. The lerobot
                # reward encoders treat 4-D input as (B, C, H, W) — a batch
                # of single-frame trajectories — so the batch dim must be
                # explicit here (AddBatchDimensionProcessorStep only
                # unsqueezes 3-D image tensors).
                frames = torch.stack([frame(fid) for fid in window]).unsqueeze(0)
                batch = {image_key: frames, "task": task}
                scores = model.compute_reward(preprocess(batch))
                curve.append(float(scores.reshape(-1)[-1].item()))

        duration_s = n_frames / float(dataset.fps) if dataset.fps else 0.0
        curve_fps = round(len(curve) / duration_s, 4) if duration_s > 0 else None
        return curve, curve_fps

    @staticmethod
    def _sample_window(start: int, endpoint: int, max_frames: int) -> list[int]:
        """Up to ``max_frames`` evenly spaced frame ids in [start, endpoint]."""
        span = endpoint - start + 1
        count = min(max_frames, span)
        if count <= 1:
            return [endpoint]
        step = (span - 1) / (count - 1)
        window = sorted({start + round(i * step) for i in range(count)})
        # Always end the window on the prefix endpoint (progress is read there).
        if window[-1] != endpoint:
            window.append(endpoint)
        return window
