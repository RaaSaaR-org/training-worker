"""Dataset annotation runner — shells out to the ``lerobot-annotate`` CLI.

Runs LeRobot 0.6.0's VLM annotation pipeline over a downloaded dataset and
converts its output into the server contract (TASK-179 Phase 4):

  ``finalMetrics = { kind: 'annotate', annotations: [
      { episodeIndex, subtasks: [{ startS, endS, text }],
        vqa?: [{ question, answer }] } ] }``

Output-format assumption (verified against the installed lerobot 0.6.0):
``lerobot-annotate`` has no JSON-report flag — it rewrites the dataset
parquet shards in place and leaves its raw module outputs as JSONL files
under the staging dir (``--staging_dir``), one dir per episode
(``episode_{idx:06d}/{plan,interjections,vqa}.jsonl``). We point the CLI
at a staging dir inside the job work dir and parse those JSONL files:

  - ``plan.jsonl`` rows with ``style == "subtask"`` → subtask spans.
    Rows carry only a start ``timestamp``; each span's ``endS`` is the next
    subtask's start (the last span ends at the max timestamp staged for the
    episode, falling back to its own start).
  - ``vqa.jsonl`` user/assistant row pairs → ``{question, answer}``.

As a tolerant fallback (for future CLI versions that may emit a report),
an ``annotations.json`` file in the staging dir or dataset root with an
``annotations`` array matching the contract is used verbatim.

The CLI ships with base lerobot but its pipeline needs the
``lerobot[annotations]`` extras (see the pyproject `annotate` extra); a
missing binary fails the job with an actionable error.

Env vars:
  LEROBOT_ANNOTATE_BIN        override path to the lerobot-annotate binary
  ANNOTATE_PROGRESS_POLL_SEC  progress/cancel poll interval (default 5)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
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

_MISSING_ANNOTATE_HINT = (
    "lerobot-annotate CLI not found. It needs the lerobot[annotations] extras — "
    "install with `uv pip install -e '.[annotate]'` in training-worker/ "
    "(or run scripts/setup-lerobot-gpu.sh on the GPU machine)."
)

_EPISODE_DIR_RE = re.compile(r"^episode_(\d{6,})$")


class AnnotateRunner(BaseTrainer):
    """Annotate dataset episodes with VLM subtasks + VQA via lerobot-annotate."""

    def train(
        self,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
    ) -> TrainerResult:
        started = time.monotonic()
        hp = ctx.hyperparameters

        cli = self._resolve_cli()

        # 1) Dataset — download from RustFS (annotation runs on a local copy;
        # the rewritten shards stay in the scratch dir, only the parsed
        # annotations go back to the server).
        dataset_dir = ctx.work_dir / "dataset"
        self._download_dataset(ctx, dataset_dir)
        dataset_root = self._locate_dataset_root(dataset_dir)

        episodes = [int(e) for e in (hp.get("episodes") or [])]
        staging_dir = ctx.work_dir / "annotate_staging"

        # 2) Run the CLI, streaming cancellation checks + coarse progress.
        cmd = self._build_command(cli, dataset_root, staging_dir, episodes, hp)
        log.info("[Annotate] launching: %s", " ".join(cmd))
        self._run_and_poll(cmd, ctx, on_progress, staging_dir, total=len(episodes) or 1)

        # 3) Parse annotations from the staging tree (or a JSON report).
        annotations = self._parse_annotations(staging_dir, dataset_root)
        if episodes:
            wanted = set(episodes)
            annotations = [a for a in annotations if a["episodeIndex"] in wanted]

        # 4) annotations.json artifact + final metrics (contract shape).
        final_metrics = {"kind": "annotate", "annotations": annotations}
        artifact_path = ctx.work_dir / "annotations.json"
        artifact_path.write_text(json.dumps(final_metrics, indent=2))
        log.info(
            "[Annotate] annotated %d episode(s) in %.1fs — artifact %s",
            len(annotations),
            time.monotonic() - started,
            artifact_path,
        )
        return TrainerResult(artifact_path=artifact_path, final_metrics=final_metrics)

    # ====================================================================
    # ENVIRONMENT / DATASET
    # ====================================================================

    def _resolve_cli(self) -> str:
        """Locate the lerobot-annotate binary (env override → venv → PATH)."""
        override = os.environ.get("LEROBOT_ANNOTATE_BIN", "").strip()
        if override:
            path = Path(override).expanduser()
            if not path.exists():
                raise RuntimeError(
                    f"LEROBOT_ANNOTATE_BIN={override} does not exist. {_MISSING_ANNOTATE_HINT}"
                )
            return str(path)
        sibling = Path(sys.executable).parent / "lerobot-annotate"
        if sibling.exists():
            return str(sibling)
        found = shutil.which("lerobot-annotate")
        if found:
            return found
        raise RuntimeError(_MISSING_ANNOTATE_HINT)

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

    # ====================================================================
    # SUBPROCESS
    # ====================================================================

    def _build_command(
        self,
        cli: str,
        dataset_root: Path,
        staging_dir: Path,
        episodes: list[int],
        hp: dict[str, Any],
    ) -> list[str]:
        cmd = [cli, f"--root={dataset_root}", f"--staging_dir={staging_dir}"]
        if episodes:
            # draccus tuple syntax: --only_episodes=[0,1]
            cmd.append(f"--only_episodes=[{','.join(str(e) for e in episodes)}]")
        vlm_model = str(hp.get("vlmModelId") or hp.get("vlm_model_id") or "").strip()
        if vlm_model:
            cmd.append(f"--vlm.model_id={vlm_model}")
        vlm_api_base = str(hp.get("vlmApiBase") or hp.get("vlm_api_base") or "").strip()
        if vlm_api_base:
            cmd.append(f"--vlm.api_base={vlm_api_base}")
        cmd.extend(str(a) for a in hp.get("extra_args", []))
        return cmd

    def _run_and_poll(
        self,
        cmd: list[str],
        ctx: TrainerContext,
        on_progress: ProgressCallback,
        staging_dir: Path,
        total: int,
    ) -> None:
        """Run the CLI, honour cancellation, report staged-episode progress."""
        poll_sec = float(os.environ.get("ANNOTATE_PROGRESS_POLL_SEC", "5"))
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(ctx.hf_cache_dir))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        tail: deque[str] = deque(maxlen=80)
        proc = subprocess.Popen(  # noqa: S603 — command built from config, not user input
            cmd,
            cwd=str(ctx.work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Own process group so cancellation reaches spawned helpers too
            # (the pipeline may auto-serve a local VLM server).
            start_new_session=True,
        )
        assert proc.stdout is not None
        # Non-blocking byte reads so the poll loop drains output without
        # stalling on a quiet CLI (drained via _drain_output below).
        os.set_blocking(proc.stdout.fileno(), False)

        cancelled = False
        while True:
            ret = proc.poll()
            self._drain_output(proc, tail)
            done = self._count_staged_episodes(staging_dir)
            event = ProgressEvent(
                step=min(done, total),
                total_steps=max(total, done, 1),
                epoch=1,
                total_epochs=1,
                loss=0.0,
                learning_rate=0.0,
            )
            if not on_progress(event):
                cancelled = True
                break
            if ret is not None:
                break
            time.sleep(poll_sec)

        if cancelled:
            log.info("[Annotate] cancellation requested — terminating process group")
            self._signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._signal_group(proc, signal.SIGKILL)
                proc.wait(timeout=10)
            raise CancelledError("annotation cancelled")

        self._drain_output(proc, tail)
        if proc.returncode != 0:
            log_tail = "\n".join(tail)
            raise RuntimeError(
                f"lerobot-annotate exited with code {proc.returncode}. Log tail:\n{log_tail}"
            )

    @staticmethod
    def _drain_output(proc: subprocess.Popen, tail: deque[str]) -> None:
        """Non-blocking read of pending CLI output into the log tail."""
        try:
            chunk = proc.stdout.read()  # type: ignore[union-attr]
        except (OSError, ValueError):
            return
        if not chunk:  # None (no data yet) or b"" (EOF)
            return
        for raw in chunk.splitlines():
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                tail.append(line)

    @staticmethod
    def _signal_group(proc: subprocess.Popen, sig: int) -> None:
        """Signal the whole process group (same pattern as gr00t_n1)."""
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    @staticmethod
    def _count_staged_episodes(staging_dir: Path) -> int:
        if not staging_dir.exists():
            return 0
        return sum(1 for d in staging_dir.iterdir() if d.is_dir() and _EPISODE_DIR_RE.match(d.name))

    # ====================================================================
    # OUTPUT PARSING
    # ====================================================================

    def _parse_annotations(
        self, staging_dir: Path, dataset_root: Path
    ) -> list[dict[str, Any]]:
        """Parse annotations: JSON report if present, else the staging tree."""
        for report in (staging_dir / "annotations.json", dataset_root / "annotations.json"):
            if report.exists():
                data = json.loads(report.read_text())
                annotations = data.get("annotations") if isinstance(data, dict) else data
                if isinstance(annotations, list):
                    log.info("[Annotate] using JSON report %s", report)
                    return annotations

        annotations: list[dict[str, Any]] = []
        if not staging_dir.exists():
            raise RuntimeError(
                f"lerobot-annotate produced no staging output under {staging_dir} "
                "and no annotations.json report"
            )
        episode_dirs = sorted(
            d for d in staging_dir.iterdir() if d.is_dir() and _EPISODE_DIR_RE.match(d.name)
        )
        for ep_dir in episode_dirs:
            match = _EPISODE_DIR_RE.match(ep_dir.name)
            assert match is not None
            entry = self._parse_episode(int(match.group(1)), ep_dir)
            if entry is not None:
                annotations.append(entry)
        return annotations

    def _parse_episode(self, episode_index: int, ep_dir: Path) -> dict[str, Any] | None:
        plan_rows = self._read_jsonl(ep_dir / "plan.jsonl")
        vqa_rows = self._read_jsonl(ep_dir / "vqa.jsonl")
        all_rows = plan_rows + vqa_rows + self._read_jsonl(ep_dir / "interjections.jsonl")
        if not all_rows:
            return None

        timestamps = [
            float(r["timestamp"]) for r in all_rows if isinstance(r.get("timestamp"), (int, float))
        ]
        max_ts = max(timestamps) if timestamps else 0.0

        subtask_rows = sorted(
            (r for r in plan_rows if r.get("style") == "subtask"),
            key=lambda r: float(r.get("timestamp") or 0.0),
        )
        subtasks: list[dict[str, Any]] = []
        for i, row in enumerate(subtask_rows):
            start_s = float(row.get("timestamp") or 0.0)
            if i + 1 < len(subtask_rows):
                end_s = float(subtask_rows[i + 1].get("timestamp") or start_s)
            else:
                end_s = max(max_ts, start_s)
            subtasks.append(
                {"startS": start_s, "endS": end_s, "text": str(row.get("content") or "")}
            )

        entry: dict[str, Any] = {"episodeIndex": episode_index, "subtasks": subtasks}
        vqa = self._pair_vqa(vqa_rows)
        if vqa:
            entry["vqa"] = vqa
        return entry

    @staticmethod
    def _pair_vqa(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Pair consecutive user (question) / assistant (answer) vqa rows."""
        pairs: list[dict[str, str]] = []
        question: str | None = None
        for row in rows:
            if row.get("style") not in (None, "vqa"):
                continue
            role = row.get("role")
            if role == "user":
                question = str(row.get("content") or "")
            elif role == "assistant" and question is not None:
                pairs.append({"question": question, "answer": str(row.get("content") or "")})
                question = None
        return pairs

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("[Annotate] skipping malformed JSONL line in %s", path)
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
