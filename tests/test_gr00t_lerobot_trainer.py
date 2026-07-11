"""Gr00tLerobotTrainer — native in-process GR00T N1.7 path (TASK-179).

The heavy lerobot[groot] stack is not installed on dev Macs, so the three
modules the trainer lazily imports (dataset, GrootConfig, policy factory)
are replaced with fakes in sys.modules. Everything else — the training
loop, progress events, cancellation, artifact packaging — runs for real
on a tiny torch model.
"""

from __future__ import annotations

import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest

from trainers.base import CancelledError, ProgressEvent, TrainerContext
from trainers.gr00t_lerobot import Gr00tLerobotTrainer
from trainers.gr00t_n1 import resolve_base_model


def make_ctx(tmp_path: Path, **overrides) -> TrainerContext:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    defaults = dict(
        job_id="job-groot-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v3.0",
        base_model="groot_n1_7",
        fine_tune_method="finetune",
        hyperparameters={},
        device="cpu",
        work_dir=work_dir,
        hf_cache_dir=tmp_path / "hf-cache",
    )
    defaults.update(overrides)
    return TrainerContext(**defaults)


# ------------------------------------------------------------- base model map
def test_groot_n1_7_alias_resolves_to_n17():
    assert resolve_base_model("groot_n1_7") == "nvidia/GR00T-N1.7-3B"
    assert resolve_base_model("groot") == "nvidia/GR00T-N1.7-3B"


# ------------------------------------------------------------ fake lerobot env
@pytest.fixture
def fake_groot_modules(monkeypatch) -> dict[str, Any]:
    """Install fake lerobot dataset/config/factory modules in sys.modules."""
    import torch

    seen: dict[str, Any] = {"datasets": [], "policies": [], "processors": []}

    class FakeMeta:
        stats: dict = {}

    class FakeLeRobotDataset:
        def __init__(
            self,
            repo_id: str,
            root: str,
            revision: str,
            download_videos: bool = True,
            delta_timestamps: dict | None = None,
            video_backend: str | None = None,
        ) -> None:
            self.repo_id = repo_id
            self.root = root
            self.revision = revision
            self.delta_timestamps = delta_timestamps
            self.video_backend = video_backend
            self.fps = 10
            self.meta = FakeMeta()
            seen["datasets"].append(self)

        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int) -> dict[str, Any]:
            g = torch.Generator().manual_seed(idx)
            return {
                "observation.state": torch.randn(4, generator=g),
                "action": torch.randn(2, generator=g),
            }

    class FakeGrootConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.base_model_path = kwargs.get("base_model_path") or "nvidia/GR00T-N1.7-3B"
            self.device = kwargs.get("device", "cpu")
            self.embodiment_tag = kwargs.get("embodiment_tag", "new_embodiment")
            self.chunk_size = int(kwargs.get("chunk_size", 40))
            self.kwargs = kwargs

        @property
        def action_delta_indices(self) -> list[int]:
            return list(range(self.chunk_size))

    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(4, 2)

        def forward(self, batch: dict[str, Any]):
            pred = self.net(batch["observation.state"])
            loss = (pred - batch["action"]).pow(2).mean()
            return loss, {"loss": loss}

        def save_pretrained(self, path: str) -> None:
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")
            (out / "model.safetensors").write_bytes(b"fake-groot-weights")

    def fake_make_policy(cfg: Any, ds_meta: Any = None, **_kw: Any) -> FakePolicy:
        seen["policies"].append((cfg, ds_meta))
        return FakePolicy()

    def fake_make_pre_post_processors(policy_cfg: Any = None, dataset_stats: Any = None, **_kw):
        seen["processors"].append((policy_cfg, dataset_stats))
        return (lambda batch: batch), (lambda batch: batch)

    ds_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    ds_mod.LeRobotDataset = FakeLeRobotDataset
    cfg_mod = types.ModuleType("lerobot.policies.groot.configuration_groot")
    cfg_mod.GrootConfig = FakeGrootConfig
    factory_mod = types.ModuleType("lerobot.policies.factory")
    factory_mod.make_policy = fake_make_policy
    factory_mod.make_pre_post_processors = fake_make_pre_post_processors

    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", ds_mod)
    monkeypatch.setitem(sys.modules, "lerobot.policies.groot.configuration_groot", cfg_mod)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_mod)
    return seen


@pytest.fixture
def patched_download(monkeypatch):
    """Fake RustFS download: create a minimal v3 dataset layout."""

    def fake_download(self: Gr00tLerobotTrainer, ctx: TrainerContext, dest: Path) -> None:
        (dest / "meta").mkdir(parents=True, exist_ok=True)
        (dest / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')

    monkeypatch.setattr(Gr00tLerobotTrainer, "_download_dataset", fake_download)


# ------------------------------------------------------------ full train() run
def test_train_end_to_end(tmp_path, fake_groot_modules, patched_download):
    ctx = make_ctx(
        tmp_path,
        hyperparameters={"epochs": 1, "batch_size": 4, "max_steps": 2, "chunk_size": 8},
    )
    events: list[ProgressEvent] = []

    result = Gr00tLerobotTrainer().train(ctx, lambda ev: events.append(ev) or True)

    # Artifact: tar.gz with the saved checkpoint
    assert result.artifact_path.name == "gr00t_lerobot.tar.gz"
    with tarfile.open(result.artifact_path) as tf:
        names = tf.getnames()
    assert "gr00t_checkpoint/model.safetensors" in names

    metrics = result.final_metrics
    assert metrics["totalSteps"] == 2
    assert metrics["baseModel"] == "nvidia/GR00T-N1.7-3B"
    assert isinstance(metrics["finalLoss"], float)
    assert metrics["trainingTimeSeconds"] >= 0

    assert [ev.step for ev in events] == [1, 2]
    assert events[-1].total_steps == 2

    # v3 consumed directly: two dataset loads (plain + delta reload), NO conversion
    loads = fake_groot_modules["datasets"]
    assert len(loads) == 2
    assert loads[0].delta_timestamps is None
    # Delta reload uses the config's action delta indices (chunk_size=8 @ 10fps)
    assert loads[1].delta_timestamps == {"action": [i / 10 for i in range(8)]}
    assert loads[1].revision == "v3.0"

    # Policy built via the factory from the groot config + dataset meta
    cfg, ds_meta = fake_groot_modules["policies"][0]
    assert cfg.base_model_path == "nvidia/GR00T-N1.7-3B"
    assert ds_meta is loads[1].meta


def test_cancellation_stops_training(tmp_path, fake_groot_modules, patched_download):
    ctx = make_ctx(tmp_path, hyperparameters={"epochs": 2, "batch_size": 2})
    with pytest.raises(CancelledError):
        Gr00tLerobotTrainer().train(ctx, lambda ev: False)


# --------------------------------------------------------------------- errors
def test_missing_groot_stack_raises_actionable_error(tmp_path, patched_download, monkeypatch):
    """No lerobot[groot] → RuntimeError naming the extra, not a bare ImportError."""
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)
    ctx = make_ctx(tmp_path)
    with pytest.raises(RuntimeError, match=r"lerobot\[groot\]"):
        Gr00tLerobotTrainer().train(ctx, lambda ev: True)
