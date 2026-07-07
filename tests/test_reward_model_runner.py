"""RewardModelRunner — progress curves via mocked lerobot.rewards factories.

The lerobot.rewards *factory* module is real (installed with 0.6.0), but the
model-weight extras (robometer/topreward) are not — so the factory functions
are monkeypatched with fakes and the runner's own logic (episode iteration,
prefix windows, curve shape, contract metrics, cancellation) runs for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from eval.reward_model import RewardModelRunner
from trainers.base import CancelledError, ProgressEvent, TrainerContext


def make_ctx(tmp_path: Path, **overrides) -> TrainerContext:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    defaults = dict(
        job_id="job-reward-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v3.0",
        base_model="robometer",
        fine_tune_method="none",
        hyperparameters={"rewardType": "robometer"},
        device="cpu",
        work_dir=work_dir,
        hf_cache_dir=tmp_path / "hf-cache",
    )
    defaults.update(overrides)
    return TrainerContext(**defaults)


class FakeMeta:
    def __init__(self, episodes: list[dict], camera_keys: list[str]) -> None:
        self.episodes = episodes
        self.camera_keys = camera_keys
        self.total_episodes = len(episodes)
        self.stats: dict = {}


class FakeDataset:
    """Tiny LeRobot-shaped dataset: pixel values ramp up over each episode."""

    def __init__(self, n_eps: int = 2, frames_per_ep: int = 10, fps: int = 10) -> None:
        self.fps = fps
        self._frames_per_ep = frames_per_ep
        episodes = [
            {
                "dataset_from_index": i * frames_per_ep,
                "dataset_to_index": (i + 1) * frames_per_ep,
                "tasks": [f"task-{i}"],
            }
            for i in range(n_eps)
        ]
        self.meta = FakeMeta(episodes, ["observation.images.top"])

    def __len__(self) -> int:
        return self.meta.total_episodes * self._frames_per_ep

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        # Frame value grows within each episode → monotonic progress curve.
        within = idx % self._frames_per_ep
        value = (within + 1) / self._frames_per_ep
        return {"observation.images.top": torch.full((3, 4, 4), value)}


@pytest.fixture
def fake_dataset(monkeypatch) -> FakeDataset:
    dataset = FakeDataset()
    monkeypatch.setattr(RewardModelRunner, "_download_dataset", lambda self, ctx, dest: None)
    monkeypatch.setattr(
        RewardModelRunner, "_load_lerobot_dataset", lambda self, ctx, d: dataset
    )
    return dataset


@pytest.fixture
def fake_reward_factory(monkeypatch) -> dict[str, Any]:
    """Monkeypatch the real lerobot.rewards.factory functions with fakes."""
    factory = pytest.importorskip("lerobot.rewards.factory")
    calls: dict[str, Any] = {"configs": [], "models": [], "tasks": []}

    class FakeCfg:
        def __init__(self, reward_type: str, kwargs: dict) -> None:
            self.type = reward_type
            self.kwargs = kwargs

    class FakeModel:
        def eval(self):
            return self

        def compute_reward(self, batch):
            import torch

            # Mirrors the real robometer/topreward contract: frames arrive
            # as (B, T, C, H, W) — ONE trajectory of T frames per batch item
            # (a 4-D tensor would be misread as B single-frame trajectories)
            # — and one reward per batch item comes back. Progress = mean
            # pixel value of the trajectory's last frame.
            frames = batch["frames"]
            assert frames.dim() == 5, (
                f"expected (B, T, C, H, W) trajectory frames, got {tuple(frames.shape)}"
            )
            return torch.tensor([float(traj[-1].mean()) for traj in frames])

    def fake_make_config(reward_type: str, **kwargs):
        calls["configs"].append((reward_type, kwargs))
        return FakeCfg(reward_type, kwargs)

    def fake_make_model(cfg, **_kw):
        calls["models"].append(cfg)
        return FakeModel()

    def fake_make_processors(cfg, **_kw):
        def pre(batch: dict) -> dict:
            calls["tasks"].append(batch["task"])
            return {"frames": batch[cfg.kwargs["image_key"]], "task": batch["task"]}

        return pre, lambda x: x

    monkeypatch.setattr(factory, "make_reward_model_config", fake_make_config)
    monkeypatch.setattr(factory, "make_reward_model", fake_make_model)
    monkeypatch.setattr(factory, "make_reward_pre_post_processors", fake_make_processors)
    return calls


# ------------------------------------------------------------------ happy path
def test_scores_all_episodes_with_contract_shape(tmp_path, fake_dataset, fake_reward_factory):
    ctx = make_ctx(tmp_path, hyperparameters={"rewardType": "robometer", "curvePoints": 5})
    events: list[ProgressEvent] = []

    result = RewardModelRunner().train(ctx, lambda ev: events.append(ev) or True)

    metrics = result.final_metrics
    assert metrics["kind"] == "reward_model"
    assert metrics["rewardType"] == "robometer"
    assert [r["episodeIndex"] for r in metrics["rewards"]] == [0, 1]
    for reward in metrics["rewards"]:
        assert set(reward) == {"episodeIndex", "score", "success", "curve", "fps"}
        assert len(reward["curve"]) == 5
        # Frames ramp up within the episode → non-decreasing progress curve
        assert reward["curve"] == sorted(reward["curve"])
        assert reward["score"] == reward["curve"][-1]
        assert reward["success"] is True  # final value 1.0 >= 0.5
        # 5 curve points over a 1s episode (10 frames @ 10 fps)
        assert reward["fps"] == pytest.approx(5.0)

    # One ProgressEvent per episode, surfacing the score as the live signal
    assert [ev.step for ev in events] == [1, 2]
    assert events[-1].total_steps == 2

    # rewards.json artifact mirrors finalMetrics
    assert result.artifact_path.name == "rewards.json"
    assert json.loads(result.artifact_path.read_text()) == metrics

    # Factory got the resolved defaults: first camera key + robometer's 8 frames
    reward_type, kwargs = fake_reward_factory["configs"][0]
    assert reward_type == "robometer"
    assert kwargs["image_key"] == "observation.images.top"
    assert kwargs["max_frames"] == 8
    assert kwargs["device"] == "cpu"

    # Task fell back to the episodes' own metadata tasks
    assert "task-0" in fake_reward_factory["tasks"]
    assert "task-1" in fake_reward_factory["tasks"]


def test_episode_subset_and_task_override(tmp_path, fake_dataset, fake_reward_factory):
    ctx = make_ctx(
        tmp_path,
        hyperparameters={
            "rewardType": "topreward",
            "episodes": [1],
            "imageKey": "observation.images.top",
            "task": "stack the cubes",
            "maxFrames": 4,
        },
    )

    result = RewardModelRunner().train(ctx, lambda ev: True)

    rewards = result.final_metrics["rewards"]
    assert [r["episodeIndex"] for r in rewards] == [1]
    assert set(fake_reward_factory["tasks"]) == {"stack the cubes"}
    _, kwargs = fake_reward_factory["configs"][0]
    assert kwargs["max_frames"] == 4


def test_short_episode_caps_curve_points(tmp_path, monkeypatch, fake_reward_factory):
    dataset = FakeDataset(n_eps=1, frames_per_ep=3)
    monkeypatch.setattr(RewardModelRunner, "_download_dataset", lambda self, ctx, dest: None)
    monkeypatch.setattr(RewardModelRunner, "_load_lerobot_dataset", lambda self, ctx, d: dataset)
    ctx = make_ctx(tmp_path, hyperparameters={"rewardType": "robometer", "curvePoints": 16})

    result = RewardModelRunner().train(ctx, lambda ev: True)

    assert len(result.final_metrics["rewards"][0]["curve"]) == 3


# ---------------------------------------------------------------- cancellation
def test_cancellation_between_episodes(tmp_path, fake_dataset, fake_reward_factory):
    ctx = make_ctx(tmp_path)
    with pytest.raises(CancelledError):
        RewardModelRunner().train(ctx, lambda ev: False)


# --------------------------------------------------------------------- errors
def test_unsupported_reward_type_raises(tmp_path):
    ctx = make_ctx(tmp_path, base_model="", hyperparameters={"rewardType": "bogus"})
    with pytest.raises(RuntimeError, match="rewardType"):
        RewardModelRunner().train(ctx, lambda ev: True)


def test_episode_index_out_of_range_raises(tmp_path, fake_dataset, fake_reward_factory):
    ctx = make_ctx(tmp_path, hyperparameters={"rewardType": "robometer", "episodes": [0, 7]})
    with pytest.raises(RuntimeError, match=r"episodes \[7\] out of range"):
        RewardModelRunner().train(ctx, lambda ev: True)


def test_missing_rewards_stack_raises_actionable_error(tmp_path, fake_dataset, monkeypatch):
    """No lerobot[robometer,topreward] → RuntimeError naming the extra."""
    monkeypatch.setitem(sys.modules, "lerobot.rewards.factory", None)
    ctx = make_ctx(tmp_path)
    with pytest.raises(RuntimeError, match=r"robometer,topreward"):
        RewardModelRunner().train(ctx, lambda ev: True)


# ---------------------------------------------------------- full worker loop
def test_reward_job_runs_through_full_worker_loop(
    monkeypatch, fake_dataset, fake_reward_factory
):
    """kind=reward_model from claim payload → dispatch → /complete metrics."""
    import worker as worker_mod
    from callbacks import ClaimedJob

    from test_worker_dispatch import make_config
    from test_worker_gr00t_loop import FakeServer

    monkeypatch.setattr(
        worker_mod,
        "_upload_artifact",
        lambda cfg, job_id, path: f"s3://models/{job_id}/{path.name}",
    )

    cfg = make_config()
    server = FakeServer()
    job = ClaimedJob.from_api(
        {
            "job": {
                "id": "job-reward-loop-1",
                "kind": "reward_model",
                "datasetId": "ds-1",
                "baseModel": "robometer",
                "hyperparameters": {"rewardType": "robometer", "episodes": [0]},
            },
            "dataset": {"storagePath": "ds-1/", "lerobotVersion": "v3.0"},
        }
    )

    worker_mod._run_one_job(cfg, server, job)

    assert server.failures == []
    assert server.completed is not None
    job_id, artifact_uri, metrics = server.completed
    assert job_id == "job-reward-loop-1"
    assert artifact_uri == "s3://models/job-reward-loop-1/rewards.json"
    assert metrics["kind"] == "reward_model"
    assert [r["episodeIndex"] for r in metrics["rewards"]] == [0]
    assert server.progress_calls, "no progress was streamed to the server"
