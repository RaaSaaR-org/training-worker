"""Trainer dispatch — the worker picks the runner per job kind + base model."""

from __future__ import annotations

import pytest

from callbacks import ClaimedJob
from config import Config
from trainers import StubTrainer
from trainers.gr00t_n1 import Gr00tTrainer
from worker import _pick_trainer


def make_config(stub: bool = False) -> Config:
    return Config(
        server_url="http://localhost:3001",
        worker_id="worker-test",
        poll_interval_sec=5.0,
        rustfs_endpoint="http://localhost:9000",
        rustfs_access_key="x",
        rustfs_secret_key="x",
        rustfs_bucket_datasets="datasets",
        rustfs_bucket_models="models",
        device="cuda",
        hf_cache_dir="/tmp/hf",
        checkpoint_interval_steps=100,
        nats_servers="nats://localhost:4222",
        stub_mode=stub,
        heartbeat_interval_sec=30.0,
    )


def make_job(base_model: str, kind: str = "supervised") -> ClaimedJob:
    return ClaimedJob(
        id="job-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version=None,
        base_model=base_model,
        fine_tune_method="finetune",
        hyperparameters={},
        status="running",
        kind=kind,
    )


def test_stub_mode_wins_over_base_model():
    trainer = _pick_trainer(make_config(stub=True), make_job("gr00t-n1.7"))
    assert isinstance(trainer, StubTrainer)


@pytest.mark.parametrize(
    "base_model",
    ["gr00t-n1.7", "GR00T-N1.7", "nvidia/GR00T-N1.7-3B", "groot", "gr00t-n1.5"],
)
def test_gr00t_base_models_get_gr00t_trainer(base_model: str):
    trainer = _pick_trainer(make_config(), make_job(base_model))
    assert isinstance(trainer, Gr00tTrainer)


def test_smolvla_base_model_gets_smolvla_trainer():
    from trainers.smolvla_lora import SmolVLALoraTrainer

    trainer = _pick_trainer(make_config(), make_job("smolvla"))
    assert isinstance(trainer, SmolVLALoraTrainer)


# ------------------------------------------------------------- job kinds (179)
def test_reward_model_kind_gets_reward_runner():
    from eval.reward_model import RewardModelRunner

    trainer = _pick_trainer(make_config(), make_job("robometer", kind="reward_model"))
    assert isinstance(trainer, RewardModelRunner)


def test_annotate_kind_gets_annotate_runner():
    from eval.annotate import AnnotateRunner

    trainer = _pick_trainer(make_config(), make_job("lerobot-annotate", kind="annotate"))
    assert isinstance(trainer, AnnotateRunner)


def test_stub_mode_wins_over_job_kind():
    trainer = _pick_trainer(make_config(stub=True), make_job("robometer", kind="reward_model"))
    assert isinstance(trainer, StubTrainer)


def test_unknown_kind_raises_instead_of_supervised_fallthrough():
    """A sim_rl (or future) kind must fail the job, not train SmolVLA on it."""
    with pytest.raises(RuntimeError, match="Unsupported job kind 'sim_rl'"):
        _pick_trainer(make_config(), make_job("smolvla", kind="sim_rl"))


# ---------------------------------------------------------- GR00T backend (179)
def test_gr00t_backend_lerobot_gets_native_trainer(monkeypatch):
    from trainers.gr00t_lerobot import Gr00tLerobotTrainer

    monkeypatch.setenv("GR00T_BACKEND", "lerobot")
    trainer = _pick_trainer(make_config(), make_job("groot_n1_7"))
    assert isinstance(trainer, Gr00tLerobotTrainer)


def test_gr00t_backend_isaac_explicit_keeps_subprocess_trainer(monkeypatch):
    monkeypatch.setenv("GR00T_BACKEND", "isaac")
    trainer = _pick_trainer(make_config(), make_job("gr00t-n1.7"))
    assert isinstance(trainer, Gr00tTrainer)


def test_gr00t_backend_default_is_isaac(monkeypatch):
    monkeypatch.delenv("GR00T_BACKEND", raising=False)
    trainer = _pick_trainer(make_config(), make_job("groot_n1_7"))
    assert isinstance(trainer, Gr00tTrainer)


def test_gr00t_backend_unknown_raises(monkeypatch):
    monkeypatch.setenv("GR00T_BACKEND", "bogus")
    with pytest.raises(RuntimeError, match="GR00T_BACKEND"):
        _pick_trainer(make_config(), make_job("gr00t-n1.7"))
