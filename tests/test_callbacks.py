"""Claim protocol — worker kinds in the claim body + job kind parsing (TASK-179)."""

from __future__ import annotations

from typing import Any

import config as config_mod
from callbacks import ClaimedJob, ServerClient
from config import Config


class _Resp:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code: int = 204, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


def _capture_post(client: ServerClient, monkeypatch, response: _Resp) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_post(path: str, body: dict[str, Any]) -> _Resp:
        captured["path"] = path
        captured["body"] = body
        return response

    monkeypatch.setattr(client, "_post", fake_post)
    return captured


# ------------------------------------------------------------------ claim body
def test_claim_body_includes_kinds(monkeypatch):
    client = ServerClient(
        "http://x", "w1", device="mps", kinds=["supervised", "reward_model", "annotate"]
    )
    captured = _capture_post(client, monkeypatch, _Resp(204))

    assert client.claim_next_job() is None
    assert captured["path"] == "/api/training/workers/claim"
    assert captured["body"] == {
        "workerId": "w1",
        "device": "mps",
        "kinds": ["supervised", "reward_model", "annotate"],
    }


def test_claim_body_omits_kinds_when_unset(monkeypatch):
    client = ServerClient("http://x", "w1", device="cuda")
    captured = _capture_post(client, monkeypatch, _Resp(204))

    assert client.claim_next_job() is None
    assert "kinds" not in captured["body"]


# --------------------------------------------------------------- kind parsing
def _payload(job_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    job = {
        "id": "job-1",
        "datasetId": "ds-1",
        "baseModel": "robometer",
        "hyperparameters": {"rewardType": "robometer"},
        **(job_extra or {}),
    }
    return {"job": job, "dataset": {"storagePath": "uuid-1/", "lerobotVersion": "v3.0"}}


def test_claimed_job_parses_kind():
    job = ClaimedJob.from_api(_payload({"kind": "reward_model"}))
    assert job.kind == "reward_model"
    assert job.dataset_storage_path == "uuid-1"


def test_claimed_job_kind_defaults_to_supervised():
    assert ClaimedJob.from_api(_payload()).kind == "supervised"
    assert ClaimedJob.from_api(_payload({"kind": None})).kind == "supervised"


def test_claim_next_job_returns_job_with_kind(monkeypatch):
    client = ServerClient("http://x", "w1", kinds=["annotate"])
    _capture_post(client, monkeypatch, _Resp(200, _payload({"kind": "annotate"})))

    job = client.claim_next_job()
    assert job is not None
    assert job.kind == "annotate"


# ------------------------------------------------------------ WORKER_KINDS env
def test_config_worker_kinds_default(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_env_file", lambda path: None)
    monkeypatch.delenv("WORKER_KINDS", raising=False)
    cfg = Config.from_env()
    assert cfg.worker_kinds == ["supervised", "reward_model", "annotate"]


def test_config_worker_kinds_custom(monkeypatch):
    monkeypatch.setattr(config_mod, "_load_env_file", lambda path: None)
    monkeypatch.setenv("WORKER_KINDS", "supervised, sim_rl ,")
    cfg = Config.from_env()
    assert cfg.worker_kinds == ["supervised", "sim_rl"]
