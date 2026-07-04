"""Unit + subprocess-integration tests for the GR00T N1.x trainer."""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path

import pytest

from trainers.base import CancelledError, ProgressEvent, TrainerContext
from trainers.gr00t_n1 import Gr00tTrainer, resolve_base_model


def make_ctx(tmp_path: Path, **overrides) -> TrainerContext:
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    defaults = dict(
        job_id="job-test-1",
        dataset_id="ds-1",
        dataset_storage_path="ds-1",
        dataset_lerobot_version="v2.1",
        base_model="gr00t-n1.7",
        fine_tune_method="finetune",
        hyperparameters={},
        device="cpu",
        work_dir=work_dir,
        hf_cache_dir=tmp_path / "hf-cache",
    )
    defaults.update(overrides)
    return TrainerContext(**defaults)


def patch_download(monkeypatch: pytest.MonkeyPatch, dataset_src: Path) -> None:
    def fake_download(self: Gr00tTrainer, ctx: TrainerContext, dest: Path) -> None:
        shutil.copytree(dataset_src, dest)

    monkeypatch.setattr(Gr00tTrainer, "_download_dataset", fake_download)


# ------------------------------------------------------------- base model map
class TestResolveBaseModel:
    def test_default_alias_maps_to_n17(self):
        assert resolve_base_model("gr00t") == "nvidia/GR00T-N1.7-3B"
        assert resolve_base_model("gr00t-n1.7") == "nvidia/GR00T-N1.7-3B"
        assert resolve_base_model("GR00T_N1_7") == "nvidia/GR00T-N1.7-3B"

    def test_n15_alias(self):
        assert resolve_base_model("gr00t-n1.5") == "nvidia/GR00T-N1.5-3B"

    def test_hf_ids_pass_through(self):
        assert resolve_base_model("nvidia/GR00T-N1.7-3B") == "nvidia/GR00T-N1.7-3B"
        assert resolve_base_model("acme/my-g1-finetune") == "acme/my-g1-finetune"


# ------------------------------------------------------------- env validation
class TestEnvironmentChecks:
    def test_missing_repo_dir_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GR00T_REPO_DIR", raising=False)
        with pytest.raises(RuntimeError, match="GR00T_REPO_DIR"):
            Gr00tTrainer()._resolve_repo_dir()

    def test_invalid_repo_dir_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GR00T_REPO_DIR", str(tmp_path))
        with pytest.raises(RuntimeError, match="does not look like an Isaac-GR00T clone"):
            Gr00tTrainer()._resolve_repo_dir()

    def test_non_cuda_device_rejected(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GR00T_ALLOW_NON_CUDA", raising=False)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            Gr00tTrainer()._check_device(make_ctx(tmp_path, device="mps"))

    def test_non_cuda_device_allowed_with_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GR00T_ALLOW_NON_CUDA", "1")
        Gr00tTrainer()._check_device(make_ctx(tmp_path, device="cpu"))

    def test_cuda_device_accepted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GR00T_ALLOW_NON_CUDA", raising=False)
        Gr00tTrainer()._check_device(make_ctx(tmp_path, device="cuda"))


# ---------------------------------------------------------------- modality.json
class TestModalityJson:
    def test_generated_from_info_json(self, dataset_v2, tmp_path):
        trainer = Gr00tTrainer()
        path = trainer._ensure_modality_json(dataset_v2)
        modality = json.loads(path.read_text())
        assert modality["state"] == {"robot": {"start": 0, "end": 23}}
        assert modality["action"] == {"robot": {"start": 0, "end": 23}}
        assert modality["video"] == {
            "front": {"original_key": "observation.images.front"},
            "wrist_left": {"original_key": "observation.images.wrist_left"},
        }
        assert "human.task_description" in modality["annotation"]

    def test_existing_modality_json_untouched(self, dataset_v2):
        existing = {"state": {"custom": {"start": 0, "end": 5}}}
        (dataset_v2 / "meta" / "modality.json").write_text(json.dumps(existing))
        path = Gr00tTrainer()._ensure_modality_json(dataset_v2)
        assert json.loads(path.read_text()) == existing

    def test_missing_state_feature_raises(self, dataset_v2):
        info_path = dataset_v2 / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        del info["features"]["observation.state"]
        info_path.write_text(json.dumps(info))
        with pytest.raises(RuntimeError, match="modality.json"):
            Gr00tTrainer()._ensure_modality_json(dataset_v2)


# ------------------------------------------------------------ modality config
class TestModalityConfig:
    def test_generated_config_mirrors_so100_shape(self, dataset_v2, tmp_path):
        trainer = Gr00tTrainer()
        trainer._ensure_modality_json(dataset_v2)
        ctx = make_ctx(tmp_path, hyperparameters={"action_horizon": 8})
        path = trainer._generate_modality_config(ctx, dataset_v2, ctx.hyperparameters)
        src = path.read_text()
        assert "register_modality_config" in src
        assert "EmbodimentTag.NEW_EMBODIMENT" in src
        assert "['front', 'wrist_left']" in src
        assert "list(range(0, 8))" in src
        assert "rep=ActionRepresentation.ABSOLUTE" in src

    def test_relative_action_representation(self, dataset_v2, tmp_path):
        trainer = Gr00tTrainer()
        trainer._ensure_modality_json(dataset_v2)
        hp = {"action_representation": "relative"}
        ctx = make_ctx(tmp_path, hyperparameters=hp)
        src = trainer._generate_modality_config(ctx, dataset_v2, hp).read_text()
        assert "rep=ActionRepresentation.RELATIVE" in src

    def test_invalid_action_representation_raises(self, dataset_v2, tmp_path):
        trainer = Gr00tTrainer()
        trainer._ensure_modality_json(dataset_v2)
        hp = {"action_representation": "bogus"}
        ctx = make_ctx(tmp_path, hyperparameters=hp)
        with pytest.raises(RuntimeError, match="action_representation"):
            trainer._generate_modality_config(ctx, dataset_v2, hp)

    def test_user_supplied_config_path_wins(self, dataset_v2, tmp_path):
        custom = tmp_path / "custom_config.py"
        custom.write_text("# custom\n")
        hp = {"modality_config_path": str(custom)}
        ctx = make_ctx(tmp_path, hyperparameters=hp)
        path = Gr00tTrainer()._resolve_modality_config(ctx, dataset_v2, hp)
        assert path == custom

    def test_missing_user_config_raises(self, dataset_v2, tmp_path):
        hp = {"modality_config_path": "/nope/missing.py"}
        ctx = make_ctx(tmp_path, hyperparameters=hp)
        with pytest.raises(RuntimeError, match="modality_config_path"):
            Gr00tTrainer()._resolve_modality_config(ctx, dataset_v2, hp)


# ---------------------------------------------------------------- command build
class TestBuildCommand:
    def test_flags_and_python_override(self, gr00t_env, dataset_v2, tmp_path):
        trainer = Gr00tTrainer()
        ctx = make_ctx(tmp_path)
        hp = {"max_steps": 100, "global_batch_size": 8, "embodiment_tag": "UNITREE_G1_SONIC"}
        cmd = trainer._build_command(
            gr00t_env, ctx, dataset_v2, tmp_path / "cfg.py", tmp_path / "out", hp
        )
        joined = " ".join(cmd)
        assert "--base-model-path nvidia/GR00T-N1.7-3B" in joined
        assert "--embodiment-tag UNITREE_G1_SONIC" in joined
        assert "--max-steps 100" in joined
        assert "--global-batch-size 8" in joined
        assert "launch_finetune.py" in joined
        assert "uv run" not in joined  # GR00T_PYTHON override active

    def test_uv_prefix_without_python_override(self, monkeypatch, gr00t_env, dataset_v2, tmp_path):
        monkeypatch.delenv("GR00T_PYTHON")
        cmd = Gr00tTrainer()._build_command(
            gr00t_env, make_ctx(tmp_path), dataset_v2, tmp_path / "c.py", tmp_path / "o", {}
        )
        assert cmd[:3] == ["uv", "run", "--project"]


# ------------------------------------------------------------ full train() run
class TestTrainEndToEnd:
    def test_happy_path(self, monkeypatch, gr00t_env, dataset_v2, tmp_path):
        patch_download(monkeypatch, dataset_v2)
        trainer = Gr00tTrainer()
        ctx = make_ctx(tmp_path, hyperparameters={"max_steps": 6, "global_batch_size": 2})

        events: list[ProgressEvent] = []

        def on_progress(ev: ProgressEvent) -> bool:
            events.append(ev)
            return True

        result = trainer.train(ctx, on_progress)

        assert result.artifact_path.exists()
        with tarfile.open(result.artifact_path) as tf:
            names = tf.getnames()
        assert "gr00t_checkpoint/model.safetensors" in names
        assert result.final_metrics["totalSteps"] == 6
        assert result.final_metrics["finalLoss"] == pytest.approx(1.0 / 6, abs=1e-3)
        assert result.final_metrics["baseModel"] == "nvidia/GR00T-N1.7-3B"
        assert events, "progress callback never fired"
        # Generated helper files landed in the work dir
        assert (ctx.work_dir / "neodem_modality_config.py").exists()
        assert (ctx.work_dir / "dataset" / "meta" / "modality.json").exists()

    def test_cancellation_kills_subprocess(self, monkeypatch, gr00t_env, dataset_v2, tmp_path):
        patch_download(monkeypatch, dataset_v2)
        trainer = Gr00tTrainer()
        # 500 steps ≈ 10s if left running — cancellation must cut this short
        ctx = make_ctx(tmp_path, hyperparameters={"max_steps": 500})

        calls = {"n": 0}

        def cancel_after_two(ev: ProgressEvent) -> bool:
            calls["n"] += 1
            return calls["n"] < 3

        start = time.monotonic()
        with pytest.raises(CancelledError):
            trainer.train(ctx, cancel_after_two)
        assert time.monotonic() - start < 8, "cancellation did not terminate the subprocess"

    def test_subprocess_failure_raises_with_log_tail(
        self, monkeypatch, gr00t_env, dataset_v2, tmp_path
    ):
        patch_download(monkeypatch, dataset_v2)
        trainer = Gr00tTrainer()
        ctx = make_ctx(tmp_path, hyperparameters={"max_steps": 5, "extra_args": ["--fail"]})
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            trainer.train(ctx, lambda ev: True)


# ------------------------------------------------------------- dataset helpers
class TestDatasetHelpers:
    def test_locate_dataset_root_nested(self, dataset_v2, tmp_path):
        nested = tmp_path / "download"
        shutil.copytree(dataset_v2, nested / "some-uuid")
        assert Gr00tTrainer()._locate_dataset_root(nested) == nested / "some-uuid"

    def test_locate_dataset_root_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Gr00tTrainer()._locate_dataset_root(tmp_path)

    def test_codebase_version(self, dataset_v2):
        assert Gr00tTrainer()._codebase_version(dataset_v2) == "v2.1"
