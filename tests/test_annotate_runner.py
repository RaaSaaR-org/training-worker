"""AnnotateRunner — fake lerobot-annotate CLI (real subprocess, no VLM).

The fake CLI mimics the real pipeline's observable behaviour: it writes
per-episode staging JSONL files (plan.jsonl subtask rows, vqa.jsonl
question/answer pairs) under --staging_dir and exits non-zero on --fail.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from eval.annotate import AnnotateRunner
from trainers.base import CancelledError, ProgressEvent, TrainerContext

_FAKE_ANNOTATE_CLI = f'''#!{sys.executable}
import argparse
import json
import pathlib
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--root")
p.add_argument("--staging_dir")
p.add_argument("--only_episodes", default=None)
p.add_argument("--fail", action="store_true")
p.add_argument("--sleep", type=float, default=0.0)
args, _unknown = p.parse_known_args()

if args.fail:
    print("Traceback: VLM server unreachable")
    sys.exit(1)

episodes = json.loads(args.only_episodes) if args.only_episodes else [0]
staging = pathlib.Path(args.staging_dir)
for ep in episodes:
    d = staging / f"episode_{{ep:06d}}"
    d.mkdir(parents=True, exist_ok=True)
    plan_rows = [
        {{"role": "assistant", "content": "pick up the cube", "style": "subtask",
          "timestamp": 0.0, "tool_calls": None}},
        {{"role": "assistant", "content": "1. place it in the box", "style": "plan",
          "timestamp": 0.0, "tool_calls": None}},
        {{"role": "assistant", "content": "place it in the box", "style": "subtask",
          "timestamp": 2.5, "tool_calls": None}},
        {{"role": "assistant", "content": "done so far: pick", "style": "memory",
          "timestamp": 5.0, "tool_calls": None}},
    ]
    (d / "plan.jsonl").write_text("\\n".join(json.dumps(r) for r in plan_rows))
    vqa_rows = [
        {{"role": "user", "content": "What color is the cube?", "style": "vqa",
          "timestamp": 1.0, "camera": "front", "tool_calls": None}},
        {{"role": "assistant", "content": "{{\\\"color\\\": \\\"red\\\"}}", "style": "vqa",
          "timestamp": 1.0, "camera": "front", "tool_calls": None}},
    ]
    (d / "vqa.jsonl").write_text("\\n".join(json.dumps(r) for r in vqa_rows))
    time.sleep(args.sleep)
print("annotate: wrote %d shard(s)" % len(episodes), flush=True)
'''


@pytest.fixture
def fake_annotate_cli(tmp_path: Path, monkeypatch) -> Path:
    script = tmp_path / "fake-lerobot-annotate"
    script.write_text(_FAKE_ANNOTATE_CLI)
    script.chmod(0o755)
    monkeypatch.setenv("LEROBOT_ANNOTATE_BIN", str(script))
    monkeypatch.setenv("ANNOTATE_PROGRESS_POLL_SEC", "0.05")
    return script


def make_ctx(tmp_path: Path, **overrides) -> TrainerContext:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    defaults = dict(
        job_id="job-annotate-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v3.0",
        base_model="lerobot-annotate",
        fine_tune_method="none",
        hyperparameters={},
        device="cpu",
        work_dir=work_dir,
        hf_cache_dir=tmp_path / "hf-cache",
    )
    defaults.update(overrides)
    return TrainerContext(**defaults)


@pytest.fixture
def patched_download(monkeypatch, dataset_v2: Path):
    def fake_download(self: AnnotateRunner, ctx: TrainerContext, dest: Path) -> None:
        shutil.copytree(dataset_v2, dest)

    monkeypatch.setattr(AnnotateRunner, "_download_dataset", fake_download)


# ------------------------------------------------------------------ happy path
def test_annotate_end_to_end(tmp_path, fake_annotate_cli, patched_download):
    ctx = make_ctx(tmp_path, hyperparameters={"episodes": [0, 2]})
    events: list[ProgressEvent] = []

    result = AnnotateRunner().train(ctx, lambda ev: events.append(ev) or True)

    metrics = result.final_metrics
    assert metrics["kind"] == "annotate"
    assert [a["episodeIndex"] for a in metrics["annotations"]] == [0, 2]

    entry: dict[str, Any] = metrics["annotations"][0]
    # subtask spans: startS from the row, endS = next subtask start / max ts
    assert entry["subtasks"] == [
        {"startS": 0.0, "endS": 2.5, "text": "pick up the cube"},
        {"startS": 2.5, "endS": 5.0, "text": "place it in the box"},
    ]
    assert entry["vqa"] == [
        {"question": "What color is the cube?", "answer": '{"color": "red"}'}
    ]

    # annotations.json artifact mirrors finalMetrics
    assert result.artifact_path.name == "annotations.json"
    assert json.loads(result.artifact_path.read_text()) == metrics

    assert events, "no progress was emitted while the CLI ran"


def test_annotate_all_episodes_when_unset(tmp_path, fake_annotate_cli, patched_download):
    ctx = make_ctx(tmp_path)  # no episodes hyperparameter → CLI default set
    result = AnnotateRunner().train(ctx, lambda ev: True)
    assert [a["episodeIndex"] for a in result.final_metrics["annotations"]] == [0]


# --------------------------------------------------------------------- errors
def test_missing_cli_raises_actionable_error(tmp_path, monkeypatch, patched_download):
    monkeypatch.delenv("LEROBOT_ANNOTATE_BIN", raising=False)
    # Hide both the venv sibling and anything on PATH
    monkeypatch.setattr("eval.annotate.sys.executable", str(tmp_path / "nope" / "python"))
    monkeypatch.setattr("eval.annotate.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"lerobot\[annotations"):
        AnnotateRunner().train(make_ctx(tmp_path), lambda ev: True)


def test_bad_cli_override_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_ANNOTATE_BIN", str(tmp_path / "missing-bin"))
    with pytest.raises(RuntimeError, match="LEROBOT_ANNOTATE_BIN"):
        AnnotateRunner()._resolve_cli()


def test_cli_failure_raises_with_log_tail(tmp_path, fake_annotate_cli, patched_download):
    ctx = make_ctx(tmp_path, hyperparameters={"extra_args": ["--fail"]})
    with pytest.raises(RuntimeError, match="VLM server unreachable"):
        AnnotateRunner().train(ctx, lambda ev: True)


# ---------------------------------------------------------------- cancellation
def test_cancellation_kills_cli(tmp_path, fake_annotate_cli, patched_download):
    ctx = make_ctx(
        tmp_path,
        hyperparameters={"episodes": [0, 1, 2, 3], "extra_args": ["--sleep=0.5"]},
    )
    calls = {"n": 0}

    def cancel_after_two(ev: ProgressEvent) -> bool:
        calls["n"] += 1
        return calls["n"] < 3

    start = time.monotonic()
    with pytest.raises(CancelledError):
        AnnotateRunner().train(ctx, cancel_after_two)
    assert time.monotonic() - start < 8, "cancellation did not terminate the CLI"
