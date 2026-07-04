"""GR00T N1.7 fine-tuning trainer — shells out to NVIDIA Isaac-GR00T.

Isaac-GR00T pins Python 3.10 + CUDA (flash-attn, TensorRT), which conflicts
with this worker's Python 3.12 environment. The trainer therefore runs
``gr00t/experiment/launch_finetune.py`` as a subprocess inside the
Isaac-GR00T repo's own uv environment and parses training progress from
its combined stdout/stderr stream.

Machine requirements (fine-tuning):
  - NVIDIA GPU, 40 GB+ VRAM recommended (H100 / L40; a 4090 works for
    small batch sizes)
  - Local clone of https://github.com/NVIDIA/Isaac-GR00T with its env
    synced: ``uv sync --python 3.10`` (see scripts/setup-gr00t.sh)

Env vars read by this trainer:
  GR00T_REPO_DIR            path to the Isaac-GR00T clone (required)
  GR00T_PYTHON              optional interpreter that already has gr00t
                            installed (e.g. NVIDIA's container) — replaces
                            the default ``uv run --project`` prefix
  GR00T_ALLOW_NON_CUDA      allow TRAINING_DEVICE != cuda (smoke tests only)
  GR00T_PROGRESS_POLL_SEC   progress/cancel poll interval (default 5)

Dataset expectations: LeRobot v2.x in GR00T flavor (meta/modality.json).
LeRobot v3 datasets are converted on the fly via Isaac-GR00T's
``scripts/lerobot_conversion/convert_v3_to_v2.py``. A missing
modality.json is auto-generated from meta/info.json (single state/action
block) — fine for first runs; hand-written configs give better results.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tarfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .base import (
    BaseTrainer,
    CancelledError,
    ProgressCallback,
    ProgressEvent,
    TrainerContext,
    TrainerResult,
)

log = logging.getLogger(__name__)

_FINETUNE_SCRIPT = Path("gr00t") / "experiment" / "launch_finetune.py"
_CONVERT_SCRIPT = Path("scripts") / "lerobot_conversion" / "convert_v3_to_v2.py"

# HF Trainer log dicts: {'loss': 0.6543, 'learning_rate': 4.9e-05, 'epoch': 0.16}
_RE_LOSS = re.compile(r"'loss':\s*([0-9.eE+-]+)")
_RE_LR = re.compile(r"'learning_rate':\s*([0-9.eE+-]+)")
# tqdm progress bars:  10%|█         | 200/2000 [01:23<12:34, 2.39it/s]
_RE_STEP = re.compile(r"(\d+)/(\d+)\s*\[")

_DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"


def resolve_base_model(base_model: str) -> str:
    """Map server-side model aliases to HuggingFace ids.

    Anything containing a "/" is passed through untouched. Bare aliases
    (gr00t, gr00t-n1.7, GR00T_N1_7, …) resolve to the N1.7 base; "n1.5"
    aliases resolve to the older N1.5 base.
    """
    name = (base_model or "").strip()
    if "/" in name:
        return name
    normalized = name.lower().replace("_", ".").replace("-", ".")
    if "n1.5" in normalized:
        return "nvidia/GR00T-N1.5-3B"
    return _DEFAULT_BASE_MODEL


class Gr00tTrainer(BaseTrainer):
    """Fine-tune GR00T N1.x on a LeRobot dataset via Isaac-GR00T."""

    def train(
        self,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
    ) -> TrainerResult:
        started = time.monotonic()
        hp = ctx.hyperparameters

        repo_dir = self._resolve_repo_dir()
        self._check_device(ctx)

        # 1) Dataset — download from RustFS, convert v3 → v2 if needed,
        # make sure the GR00T flavor extras (modality.json) exist.
        dataset_dir = ctx.work_dir / "dataset"
        self._download_dataset(ctx, dataset_dir)
        dataset_dir = self._locate_dataset_root(dataset_dir)
        if self._codebase_version(dataset_dir).startswith("v3"):
            log.info("[GR00T] dataset is LeRobot v3 — converting to v2 for Isaac-GR00T")
            self._convert_v3_to_v2(repo_dir, ctx, dataset_dir)
            dataset_dir = self._locate_dataset_root(ctx.work_dir / "dataset")
        self._ensure_modality_json(dataset_dir)

        # 2) Modality config (.py) — user-provided path or generated from
        # the dataset's modality.json.
        modality_config = self._resolve_modality_config(ctx, dataset_dir, hp)

        # 3) Launch fine-tuning subprocess and stream progress.
        output_dir = ctx.work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command(repo_dir, ctx, dataset_dir, modality_config, output_dir, hp)
        log.info("[GR00T] launching: %s", " ".join(cmd))
        final_state = self._run_and_stream(cmd, repo_dir, ctx, on_progress, hp)

        # 4) Package the newest checkpoint as the artifact.
        artifact_path = self._package_artifact(ctx, output_dir)

        duration_s = time.monotonic() - started
        return TrainerResult(
            artifact_path=artifact_path,
            final_metrics={
                "finalLoss": final_state["loss"],
                "validationLoss": final_state["loss"],  # no val split yet
                "trainingTimeSeconds": round(duration_s, 1),
                "totalSteps": final_state["step"],
                "baseModel": resolve_base_model(ctx.base_model),
            },
        )

    # ====================================================================
    # ENVIRONMENT
    # ====================================================================

    def _resolve_repo_dir(self) -> Path:
        repo = os.environ.get("GR00T_REPO_DIR", "").strip()
        if not repo:
            raise RuntimeError(
                "GR00T_REPO_DIR is not set. Clone https://github.com/NVIDIA/Isaac-GR00T "
                "on this machine, run `uv sync --python 3.10` inside it "
                "(see scripts/setup-gr00t.sh), and set GR00T_REPO_DIR to the clone path."
            )
        repo_dir = Path(repo).expanduser()
        if not (repo_dir / _FINETUNE_SCRIPT).exists():
            raise RuntimeError(
                f"GR00T_REPO_DIR={repo_dir} does not look like an Isaac-GR00T clone "
                f"({_FINETUNE_SCRIPT} missing)"
            )
        return repo_dir

    def _check_device(self, ctx: TrainerContext) -> None:
        if ctx.device == "cuda":
            return
        if os.environ.get("GR00T_ALLOW_NON_CUDA", "").lower() in ("1", "true", "yes"):
            log.warning(
                "[GR00T] device=%s allowed via GR00T_ALLOW_NON_CUDA — smoke test only",
                ctx.device,
            )
            return
        raise RuntimeError(
            f"GR00T fine-tuning requires CUDA, but TRAINING_DEVICE={ctx.device}. "
            "Isaac-GR00T depends on flash-attn — MPS/CPU are not supported. "
            "Run this worker on an NVIDIA GPU machine."
        )

    def _launch_prefix(self, repo_dir: Path) -> list[str]:
        """Command prefix that runs python inside the Isaac-GR00T env."""
        override = os.environ.get("GR00T_PYTHON", "").strip()
        if override:
            return [override]
        return ["uv", "run", "--project", str(repo_dir), "python"]

    # ====================================================================
    # DATASET
    # ====================================================================

    def _download_dataset(self, ctx: TrainerContext, dest: Path) -> None:
        """Pull dataset files from RustFS (same pattern as the SmolVLA trainer)."""
        from storage import StorageClient

        storage = StorageClient(
            endpoint=os.environ["RUSTFS_ENDPOINT"],
            access_key=os.environ["RUSTFS_ACCESS_KEY"],
            secret_key=os.environ["RUSTFS_SECRET_KEY"],
            dataset_bucket=os.environ.get("RUSTFS_BUCKET_DATASETS", "datasets"),
            model_bucket=os.environ.get("RUSTFS_BUCKET_MODELS", "models"),
        )
        storage.download_dataset(ctx.dataset_storage_path, dest)

    def _locate_dataset_root(self, base: Path) -> Path:
        """Find the directory holding meta/info.json (skipping *_v30 backups)."""
        if (base / "meta" / "info.json").exists():
            return base
        candidates = sorted(
            p.parent.parent
            for p in base.rglob("meta/info.json")
            if "_v30" not in p.parent.parent.name
        )
        if not candidates:
            raise FileNotFoundError(f"LeRobot meta/info.json not found under {base}")
        return candidates[0]

    def _codebase_version(self, dataset_dir: Path) -> str:
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        return str(info.get("codebase_version", ""))

    def _convert_v3_to_v2(self, repo_dir: Path, ctx: TrainerContext, dataset_dir: Path) -> None:
        """Run Isaac-GR00T's LeRobot v3 → v2.1 converter in its env.

        The converter renames the v3 original to *_v30 and writes the v2.1
        dataset in place.
        """
        script = repo_dir / _CONVERT_SCRIPT
        if not script.exists():
            raise RuntimeError(
                f"v3 dataset but converter missing at {script} — update the Isaac-GR00T clone"
            )
        cmd = [
            *self._launch_prefix(repo_dir),
            str(script),
            "--repo-id",
            f"local/{ctx.dataset_id}",
            "--root",
            str(dataset_dir),
        ]
        log.info("[GR00T] converting dataset: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LeRobot v3→v2 conversion failed (exit {proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )

    def _ensure_modality_json(self, dataset_dir: Path) -> Path:
        """Create meta/modality.json from info.json if the dataset lacks one.

        Generates a single state/action block spanning the full vectors —
        functional for fine-tuning, though named per-limb splits (as in
        NVIDIA's examples) are recommended for real robots like the G1.
        """
        modality_path = dataset_dir / "meta" / "modality.json"
        if modality_path.exists():
            log.info("[GR00T] dataset ships meta/modality.json — using it")
            return modality_path

        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        features: dict[str, Any] = info.get("features", {})
        state_feature = features.get("observation.state")
        action_feature = features.get("action")
        if not state_feature or not action_feature:
            raise RuntimeError(
                "Cannot auto-generate modality.json: dataset lacks observation.state/action "
                "features. Add a meta/modality.json to the dataset (see Isaac-GR00T "
                "getting_started/finetune_new_embodiment.md)."
            )
        video_keys = {
            key.rsplit(".", 1)[-1]: key
            for key, spec in features.items()
            if spec.get("dtype") in ("video", "image")
        }
        if not video_keys:
            raise RuntimeError("Cannot auto-generate modality.json: dataset has no camera keys")

        modality = {
            "state": {"robot": {"start": 0, "end": int(state_feature["shape"][0])}},
            "action": {"robot": {"start": 0, "end": int(action_feature["shape"][0])}},
            "video": {name: {"original_key": key} for name, key in video_keys.items()},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }
        modality_path.write_text(json.dumps(modality, indent=4))
        log.warning(
            "[GR00T] generated meta/modality.json (state/action as one 'robot' block, "
            "cameras: %s) — provide a hand-written one for best results",
            sorted(video_keys),
        )
        return modality_path

    # ====================================================================
    # MODALITY CONFIG (.py)
    # ====================================================================

    def _resolve_modality_config(
        self,
        ctx: TrainerContext,
        dataset_dir: Path,
        hp: dict[str, Any],
    ) -> Path:
        """Return the --modality-config-path file: user-supplied or generated."""
        user_path = str(hp.get("modality_config_path", "")).strip()
        if user_path:
            path = Path(user_path).expanduser()
            if not path.exists():
                raise RuntimeError(f"hyperparameters.modality_config_path not found: {path}")
            log.info("[GR00T] using provided modality config %s", path)
            return path
        return self._generate_modality_config(ctx, dataset_dir, hp)

    def _generate_modality_config(
        self,
        ctx: TrainerContext,
        dataset_dir: Path,
        hp: dict[str, Any],
    ) -> Path:
        """Write a modality config .py mirroring Isaac-GR00T's SO100 example,
        built from the dataset's modality.json keys."""
        modality = json.loads((dataset_dir / "meta" / "modality.json").read_text())
        video_keys = sorted(modality.get("video", {}).keys())
        state_keys = list(modality.get("state", {}).keys())
        action_keys = list(modality.get("action", {}).keys())
        if not (video_keys and state_keys and action_keys):
            raise RuntimeError(
                "modality.json is missing video/state/action blocks — cannot generate "
                "a modality config; pass hyperparameters.modality_config_path instead"
            )

        action_horizon = int(hp.get("action_horizon", 16))
        representation = str(hp.get("action_representation", "absolute")).upper()
        if representation not in ("ABSOLUTE", "RELATIVE"):
            raise RuntimeError(
                f"hyperparameters.action_representation must be absolute|relative, "
                f"got {representation!r}"
            )
        action_configs = ",\n".join(
            "            ActionConfig(\n"
            f"                rep=ActionRepresentation.{representation},\n"
            "                type=ActionType.NON_EEF,\n"
            "                format=ActionFormat.DEFAULT,\n"
            "            )"
            for _ in action_keys
        )

        config_src = f'''"""Auto-generated by NeoDEM training worker for job {ctx.job_id}.

Mirrors Isaac-GR00T examples/SO100/so100_config.py, built from the
dataset's meta/modality.json.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

neodem_config = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys={video_keys!r},
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys={state_keys!r},
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, {action_horizon})),
        modality_keys={action_keys!r},
        action_configs=[
{action_configs},
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}}

register_modality_config(neodem_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
        path = ctx.work_dir / "neodem_modality_config.py"
        path.write_text(config_src)
        log.info(
            "[GR00T] generated modality config %s (video=%s state=%s action=%s horizon=%d)",
            path,
            video_keys,
            state_keys,
            action_keys,
            action_horizon,
        )
        return path

    # ====================================================================
    # SUBPROCESS
    # ====================================================================

    def _build_command(
        self,
        repo_dir: Path,
        ctx: TrainerContext,
        dataset_dir: Path,
        modality_config: Path,
        output_dir: Path,
        hp: dict[str, Any],
    ) -> list[str]:
        max_steps = int(hp.get("max_steps", 0) or 2000)
        batch_size = int(hp.get("global_batch_size", hp.get("batch_size", 32)))
        embodiment_tag = str(hp.get("embodiment_tag", "NEW_EMBODIMENT"))
        num_workers = int(hp.get("dataloader_num_workers", 4))
        extra_args = [str(a) for a in hp.get("extra_args", [])]

        return [
            *self._launch_prefix(repo_dir),
            str(repo_dir / _FINETUNE_SCRIPT),
            "--base-model-path",
            resolve_base_model(ctx.base_model),
            "--dataset-path",
            str(dataset_dir),
            "--embodiment-tag",
            embodiment_tag,
            "--modality-config-path",
            str(modality_config),
            "--num-gpus",
            "1",
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(max_steps),
            "--global-batch-size",
            str(batch_size),
            "--dataloader-num-workers",
            str(num_workers),
            *extra_args,
        ]

    def _run_and_stream(
        self,
        cmd: list[str],
        repo_dir: Path,
        ctx: TrainerContext,
        on_progress: ProgressCallback,
        hp: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the fine-tune subprocess, parse progress, honour cancellation.

        Returns the final parsed state: {step, total, loss, lr}.
        """
        poll_sec = float(os.environ.get("GR00T_PROGRESS_POLL_SEC", "5"))
        max_steps = int(hp.get("max_steps", 0) or 2000)
        lr = float(hp.get("learning_rate", 1e-4))

        env = dict(os.environ)
        env.setdefault("HF_HOME", str(ctx.hf_cache_dir))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        state = {"step": 0, "total": max_steps, "loss": 0.0, "lr": lr}
        state_lock = threading.Lock()
        tail: deque[str] = deque(maxlen=80)

        proc = subprocess.Popen(  # noqa: S603 — command is built from config, not user input
            cmd,
            cwd=str(repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None

        def _reader() -> None:
            """Split the byte stream on \\n AND \\r so tqdm updates parse too."""
            buf = b""
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    self._parse_line(line, state, state_lock, tail)

        reader = threading.Thread(target=_reader, daemon=True, name="gr00t-log-reader")
        reader.start()

        cancelled = False
        while True:
            ret = proc.poll()
            with state_lock:
                event = ProgressEvent(
                    step=state["step"],
                    total_steps=state["total"],
                    epoch=1,
                    total_epochs=1,
                    loss=round(float(state["loss"]), 6),
                    learning_rate=float(state["lr"]),
                )
            if not on_progress(event):
                cancelled = True
                break
            if ret is not None:
                break
            time.sleep(poll_sec)

        if cancelled:
            log.info("[GR00T] cancellation requested — terminating fine-tune process")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            reader.join(timeout=5)
            raise CancelledError(f"cancelled at step {state['step']}")

        reader.join(timeout=10)
        if proc.returncode != 0:
            log_tail = "\n".join(tail)
            raise RuntimeError(
                f"GR00T fine-tuning exited with code {proc.returncode}. Log tail:\n{log_tail}"
            )
        log.info("[GR00T] fine-tuning finished — final step %d loss %.4f",
                 state["step"], state["loss"])
        return state

    def _parse_line(
        self,
        line: str,
        state: dict[str, Any],
        lock: threading.Lock,
        tail: deque[str],
    ) -> None:
        tail.append(line)
        step_match = _RE_STEP.search(line)
        loss_match = _RE_LOSS.search(line)
        lr_match = _RE_LR.search(line)
        if not (step_match or loss_match or lr_match):
            return
        with lock:
            if step_match:
                state["step"] = int(step_match.group(1))
                state["total"] = max(int(step_match.group(2)), 1)
            if loss_match:
                try:
                    state["loss"] = float(loss_match.group(1))
                except ValueError:
                    pass
            if lr_match:
                try:
                    state["lr"] = float(lr_match.group(1))
                except ValueError:
                    pass

    # ====================================================================
    # ARTIFACT
    # ====================================================================

    def _package_artifact(self, ctx: TrainerContext, output_dir: Path) -> Path:
        """Tar the newest checkpoint-* dir (or the whole output dir)."""
        checkpoints = sorted(
            (d for d in output_dir.glob("checkpoint-*") if d.is_dir()),
            key=lambda d: int(d.name.rsplit("-", 1)[-1])
            if d.name.rsplit("-", 1)[-1].isdigit()
            else -1,
        )
        source = checkpoints[-1] if checkpoints else output_dir
        if not any(source.iterdir()):
            raise RuntimeError(f"GR00T fine-tune produced no output files in {source}")

        artifact_path = ctx.work_dir / "gr00t_finetune.tar.gz"
        log.info("[GR00T] packaging %s → %s", source, artifact_path)
        with tarfile.open(artifact_path, "w:gz") as tf:
            tf.add(source, arcname="gr00t_checkpoint")
        return artifact_path
