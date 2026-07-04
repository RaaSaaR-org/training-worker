"""Trainer dispatch — the worker must pick the trainer per job's base model."""

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


def make_job(base_model: str) -> ClaimedJob:
    return ClaimedJob(
        id="job-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version=None,
        base_model=base_model,
        fine_tune_method="finetune",
        hyperparameters={},
        status="running",
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
