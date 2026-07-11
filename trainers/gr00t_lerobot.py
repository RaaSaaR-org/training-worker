"""GR00T N1.7 fine-tuning trainer — native, in-process LeRobot 0.6.0 path.

Unlike ``trainers/gr00t_n1.py`` (which shells out to a local Isaac-GR00T
clone and needs a LeRobot v3 → v2 dataset conversion), this trainer runs
lerobot's own GR00T port in-process:

  - consumes LeRobotDataset **v3 directly** (no conversion step),
  - builds the policy via lerobot's factory (policy type ``"groot"``,
    config class ``GrootConfig``, base model ``nvidia/GR00T-N1.7-3B``),
  - reuses the SmolVLA trainer's download/progress/cancel plumbing.

Selected in ``worker._pick_trainer`` for gr00t/groot base models when the
env var ``GR00T_BACKEND=lerobot`` is set (default stays ``isaac``).

Requires the ``lerobot[groot,dataset]`` extras (see pyproject `groot`
extra and scripts/setup-lerobot-gpu.sh) — they are NOT part of the base
install; imports are lazy and fail with an actionable error.
"""

from __future__ import annotations

import logging
import os
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .base import (
    BaseTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    TrainerContext,
    TrainerResult,
    default_video_backend,
    make_dataloader_kwargs,
)
from .gr00t_n1 import resolve_base_model

log = logging.getLogger(__name__)

_MISSING_GROOT_HINT = (
    "The native GR00T N1.7 trainer needs the lerobot[groot] extras. "
    "Install with `uv pip install -e '.[groot]'` in training-worker/ "
    "(or run scripts/setup-lerobot-gpu.sh on the GPU machine)."
)


class Gr00tLerobotTrainer(BaseTrainer):
    """Fine-tune GR00T N1.7 on a LeRobot v3 dataset via lerobot[groot]."""

    def train(
        self,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
    ) -> TrainerResult:
        started = time.monotonic()

        hp = ctx.hyperparameters
        lr = float(hp.get("learning_rate", 1e-4))
        batch_size = int(hp.get("batch_size", 4))
        epochs = int(hp.get("epochs", 1))
        max_steps = int(hp.get("max_steps", 0))  # 0 = no cap

        log.info(
            "[GR00T-lerobot] job=%s device=%s epochs=%d batch=%d lr=%g",
            ctx.job_id,
            ctx.device,
            epochs,
            batch_size,
            lr,
        )

        # Lazy heavy imports — on machines without lerobot[groot] this must
        # fail the job with a clear message, not crash the worker at import.
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as e:
            raise RuntimeError(
                f"GR00T lerobot trainer needs torch ({e}). "
                "Run `uv pip install -e .` in training-worker/ first."
            ) from e

        # 1) Dataset — download from RustFS + load via LeRobot (v3 directly,
        # no v3→v2 conversion as with the Isaac-GR00T backend).
        dataset_dir = ctx.work_dir / "dataset"
        self._download_dataset(ctx, dataset_dir)
        dataset = self._load_lerobot_dataset(ctx, dataset_dir)

        # 2) Policy config + policy via lerobot's factory. GrootConfig's
        # action_delta_indices (range(chunk_size), N1.7 default 40) drive
        # the dataset reload so actions come as [B, chunk, action_dim].
        cfg = self._make_groot_config(ctx, hp)
        dataset = self._reload_with_deltas(ctx, dataset_dir, dataset, cfg)
        policy, preprocessor = self._make_policy(ctx, cfg, dataset)
        device = torch.device(ctx.device)
        policy.to(device)

        # 3) DataLoader — shared CUDA/MPS worker policy (TASK-179 §9).
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            **make_dataloader_kwargs(ctx.device, hp),
        )
        steps_per_epoch = len(loader)
        total_steps = epochs * steps_per_epoch
        if max_steps > 0:
            total_steps = min(total_steps, max_steps)
        log.info(
            "[GR00T-lerobot] dataset ready: %d frames, steps_per_epoch=%d, total=%d",
            len(dataset),
            steps_per_epoch,
            total_steps,
        )

        # 4) Optimizer over the params GR00T's tune_* flags left trainable
        # (projector + diffusion head by default; backbone frozen).
        trainable = [p for p in policy.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "GR00T policy has no trainable parameters — check the tune_* flags"
            )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=lr,
            weight_decay=float(hp.get("weight_decay", 1e-5)),
        )

        policy.train()
        final_loss = float("nan")
        step_idx = 0
        for epoch in range(1, epochs + 1):
            for batch in loader:
                step_idx += 1
                if max_steps > 0 and step_idx > max_steps:
                    break

                batch = preprocessor(batch)
                batch = self._move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with self._amp_context(ctx.device):
                    result = policy(batch)
                    if isinstance(result, tuple):
                        loss, _info = result
                    elif isinstance(result, dict):
                        loss = result["loss"]
                    else:
                        loss = result
                loss.backward()
                optimizer.step()

                final_loss = loss.detach().float().item()
                event = ProgressEvent(
                    step=step_idx,
                    total_steps=total_steps,
                    epoch=epoch,
                    total_epochs=epochs,
                    loss=round(final_loss, 6),
                    learning_rate=lr,
                )
                if not on_progress(event):
                    raise CancelledError(f"cancelled at step {step_idx}")

            if max_steps > 0 and step_idx >= max_steps:
                break

        # 5) Save checkpoint + pack into a tarball. Save in bf16: the N1.7
        # recipe trains and serves in bf16 (see _amp_context), and an fp32
        # dump doubles the artifact to ~12 GB for no fidelity the policy
        # ever uses.
        checkpoint_dir = ctx.work_dir / "gr00t_checkpoint"
        policy.to(torch.bfloat16).save_pretrained(str(checkpoint_dir))

        artifact_path = ctx.work_dir / "gr00t_lerobot.tar.gz"
        # compresslevel=1: the tar is dominated by safetensors, which are
        # incompressible — level 9 burns tens of minutes on a 3B model for
        # no size gain (same lesson as the Isaac-GR00T servable tar).
        with tarfile.open(artifact_path, "w:gz", compresslevel=1) as tf:
            tf.add(checkpoint_dir, arcname="gr00t_checkpoint")
        log.info("[GR00T-lerobot] wrote artifact %s", artifact_path)

        duration_s = time.monotonic() - started
        return TrainerResult(
            artifact_path=artifact_path,
            final_metrics={
                "finalLoss": round(final_loss, 6),
                "trainingTimeSeconds": round(duration_s, 1),
                "totalSteps": step_idx,
                "baseModel": resolve_base_model(ctx.base_model),
            },
        )

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
        """Load a LeRobot v3 dataset from the downloaded local directory."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_GROOT_HINT} (import failed: {e})") from e

        root = self._locate_dataset_root(dataset_dir)
        revision = ctx.dataset_lerobot_version or "v3.0"
        return LeRobotDataset(
            repo_id=f"local/{ctx.dataset_id}",
            root=str(root),
            revision=revision,
            download_videos=True,
            video_backend=default_video_backend(),
        )

    def _reload_with_deltas(
        self,
        ctx: TrainerContext,
        dataset_dir: Path,
        dataset: Any,
        cfg: Any,
    ) -> Any:
        """Reload with GR00T's action delta indices so actions come chunked."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_GROOT_HINT} (import failed: {e})") from e

        fps = dataset.fps
        delta_timestamps = {
            "action": [i / fps for i in cfg.action_delta_indices],
        }
        root = self._locate_dataset_root(dataset_dir)
        revision = ctx.dataset_lerobot_version or "v3.0"
        return LeRobotDataset(
            repo_id=f"local/{ctx.dataset_id}",
            root=str(root),
            revision=revision,
            download_videos=True,
            delta_timestamps=delta_timestamps,
            video_backend=default_video_backend(),
        )

    def _make_groot_config(self, ctx: TrainerContext, hp: dict[str, Any]) -> Any:
        """Build a GrootConfig pointing at the N1.7 base model.

        `base_model_path` (NOT `pretrained_path`) is how lerobot 0.6.0 loads
        a raw NVIDIA GR00T checkpoint — `pretrained_path` is reserved for
        LeRobot-saved checkpoints whose config.json carries a `type` field.
        """
        try:
            from lerobot.policies.groot.configuration_groot import GrootConfig
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_GROOT_HINT} (import failed: {e})") from e

        os.environ.setdefault("HF_HOME", str(ctx.hf_cache_dir))
        kwargs: dict[str, Any] = {
            "base_model_path": resolve_base_model(ctx.base_model),
            "device": ctx.device,
            "embodiment_tag": str(hp.get("embodiment_tag", "new_embodiment")),
        }
        if hp.get("chunk_size"):
            kwargs["chunk_size"] = int(hp["chunk_size"])
            kwargs["n_action_steps"] = int(hp["chunk_size"])
        for flag in ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model"):
            if flag in hp:
                kwargs[flag] = bool(hp[flag])
        return GrootConfig(**kwargs)

    def _make_policy(self, ctx: TrainerContext, cfg: Any, dataset: Any) -> tuple[Any, Any]:
        """Instantiate the groot policy + preprocessor via lerobot's factory."""
        try:
            from lerobot.policies.factory import make_policy, make_pre_post_processors
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_GROOT_HINT} (import failed: {e})") from e

        log.info(
            "[GR00T-lerobot] loading %s (HF cache=%s)", cfg.base_model_path, ctx.hf_cache_dir
        )
        try:
            policy = make_policy(cfg, ds_meta=dataset.meta)
            preprocessor, _postprocessor = make_pre_post_processors(
                policy_cfg=cfg,
                dataset_stats=dataset.meta.stats,
            )
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(f"{_MISSING_GROOT_HINT} (import failed: {e})") from e
        return policy, preprocessor

    def _move_batch(self, batch: Any, device: Any) -> Any:
        """Move tensors in a batch dict to `device`, leave other types alone."""
        import torch

        if isinstance(batch, dict):
            return {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(device) if isinstance(v, torch.Tensor) else v for v in batch
            )
        return batch

    @contextmanager
    def _amp_context(self, device: str):
        """BF16 autocast on CUDA (the N1.7 recipe); no-op elsewhere."""
        if device == "cuda":
            import torch

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield
