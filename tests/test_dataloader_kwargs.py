"""make_dataloader_kwargs — CUDA gets parallel decode, MPS/CPU stay safe (TASK-179)."""

from __future__ import annotations

from trainers.base import make_dataloader_kwargs


def test_cuda_defaults_enable_parallel_loading():
    kwargs = make_dataloader_kwargs("cuda", {})
    assert kwargs == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_cuda_honours_dataloader_num_workers_hyperparameter():
    kwargs = make_dataloader_kwargs("cuda", {"dataloader_num_workers": 2})
    assert kwargs["num_workers"] == 2
    assert kwargs["persistent_workers"] is True


def test_cuda_zero_workers_drops_worker_only_options():
    kwargs = make_dataloader_kwargs("cuda", {"dataloader_num_workers": 0})
    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_mps_and_cpu_stay_single_worker():
    for device in ("mps", "cpu"):
        assert make_dataloader_kwargs(device, {"dataloader_num_workers": 8}) == {
            "num_workers": 0,
            "pin_memory": False,
        }
