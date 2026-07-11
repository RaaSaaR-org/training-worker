"""Abstract trainer interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


@lru_cache(maxsize=1)
def default_video_backend() -> str:
    """Video decode backend for LeRobotDataset: "torchcodec" or "pyav".

    lerobot 0.6.0 defaults to torchcodec, whose native library needs FFmpeg
    *shared* DLLs — typically absent on Windows, where the pip install
    succeeds but the first decode raises "Could not load libtorchcodec".
    Probe once and fall back to PyAV (self-contained FFmpeg). Override with
    LEROBOT_VIDEO_BACKEND.
    """
    backend = os.environ.get("LEROBOT_VIDEO_BACKEND", "").strip()
    if backend:
        return backend
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return "torchcodec"
    except Exception:
        return "pyav"


@dataclass
class TrainerContext:
    """Everything a trainer needs to run a job."""

    job_id: str
    dataset_id: str
    dataset_storage_path: str  # RustFS prefix (UUID without trailing slash)
    dataset_lerobot_version: str | None
    base_model: str
    fine_tune_method: str
    hyperparameters: dict[str, Any]
    device: str  # "mps" | "cuda" | "cpu"
    work_dir: Path  # scratch directory for this job
    hf_cache_dir: Path


@dataclass
class ProgressEvent:
    """One training progress snapshot."""

    step: int
    total_steps: int
    epoch: int
    total_epochs: int
    loss: float
    learning_rate: float
    accuracy: float | None = None


@dataclass
class TrainerResult:
    """What a trainer returns when training finishes successfully."""

    artifact_path: Path
    final_metrics: dict[str, Any]


class CancelledError(Exception):
    """Raised by a trainer when it detects a user-initiated cancel.
    Distinct from generic failures — the server has already marked
    the job as cancelled, so the worker should NOT POST /failed."""


# A progress callback the trainer calls every step; returns True to continue,
# False to stop (for cancellations sent via heartbeat).
ProgressCallback = Callable[[ProgressEvent], bool]


def make_dataloader_kwargs(device: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Shared torch DataLoader worker/pinning kwargs (TASK-179 §9).

    On CUDA, parallel decode + pinned memory speeds up multi-camera video
    datasets considerably: `dataloader_num_workers` hyperparameter (default
    4) with persistent workers and prefetching.

    On MPS/CPU this stays at num_workers=0 — multi-worker loading sometimes
    deadlocks on Apple Silicon and Raspberry Pi.
    """
    if device == "cuda":
        num_workers = int(hyperparameters.get("dataloader_num_workers", 4))
        kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs
    return {"num_workers": 0, "pin_memory": False}


class BaseTrainer(ABC):
    """All trainers implement this interface."""

    @abstractmethod
    def train(
        self,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
    ) -> TrainerResult:
        """Run the training loop.

        - Call ``on_progress`` after each step. If it returns False, stop early.
        - Return a ``TrainerResult`` with the path to the final artifact.
        - Raise any exception on failure — caller will POST /failed.
        """
        raise NotImplementedError
