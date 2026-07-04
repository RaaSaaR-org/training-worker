"""Full worker-loop integration: a GR00T job from claim to /complete.

Uses the fake Isaac-GR00T repo (real subprocess) with a fake server and
storage — everything else (dispatch, heartbeat, progress, packaging) runs
the production code path.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import worker as worker_mod
from callbacks import ClaimedJob
from trainers.gr00t_n1 import Gr00tTrainer

from test_worker_dispatch import make_config


class FakeServer:
    def __init__(self) -> None:
        self.progress_calls: list[dict[str, Any]] = []
        self.completed: tuple | None = None
        self.failures: list[str] = []

    def heartbeat(self, job_id: str, **_: Any) -> str:
        return "continue"

    def progress(self, **kwargs: Any) -> dict[str, Any]:
        self.progress_calls.append(kwargs)
        return {"status": "continue"}

    def complete(self, job_id: str, artifact_uri: str, final_metrics: dict) -> dict:
        self.completed = (job_id, artifact_uri, final_metrics)
        return {}

    def failed(self, job_id: str, error_message: str, last_checkpoint: str | None = None) -> None:
        self.failures.append(error_message)


def test_gr00t_job_runs_through_full_worker_loop(monkeypatch, gr00t_env, dataset_v2):
    def fake_download(self: Gr00tTrainer, ctx, dest: Path) -> None:
        shutil.copytree(dataset_v2, dest)

    monkeypatch.setattr(Gr00tTrainer, "_download_dataset", fake_download)
    monkeypatch.setattr(
        worker_mod,
        "_upload_artifact",
        lambda cfg, job_id, path: f"s3://models/{job_id}/{path.name}",
    )

    cfg = make_config()
    server = FakeServer()
    job = ClaimedJob(
        id="job-loop-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v2.1",
        base_model="gr00t-n1.7",
        fine_tune_method="finetune",
        hyperparameters={"max_steps": 6, "global_batch_size": 2},
        status="running",
    )

    worker_mod._run_one_job(cfg, server, job)

    assert server.failures == []
    assert server.completed is not None
    job_id, artifact_uri, metrics = server.completed
    assert job_id == "job-loop-1"
    assert artifact_uri == "s3://models/job-loop-1/gr00t_finetune.tar.gz"
    assert metrics["totalSteps"] == 6
    assert server.progress_calls, "no progress was streamed to the server"


def test_gr00t_job_setup_failure_posts_failed(monkeypatch):
    """No GR00T_REPO_DIR → the job fails cleanly, the worker survives."""
    monkeypatch.delenv("GR00T_REPO_DIR", raising=False)
    cfg = make_config()
    server = FakeServer()
    job = ClaimedJob(
        id="job-loop-2",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v2.1",
        base_model="gr00t-n1.7",
        fine_tune_method="finetune",
        hyperparameters={},
        status="running",
    )

    worker_mod._run_one_job(cfg, server, job)

    assert server.completed is None
    assert len(server.failures) == 1
    assert "GR00T_REPO_DIR" in server.failures[0]
