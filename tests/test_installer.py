from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import instalar


def test_conda_environment_is_accepted() -> None:
    with patch.dict(os.environ, {"CONDA_PREFIX": sys.prefix}):
        assert instalar.is_virtual_environment() is True


def test_auto_huggingface_selects_nvidia_and_cuda_index() -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    with (
        patch.object(instalar, "nvidia_is_visible", return_value=True),
        patch.object(instalar, "run", side_effect=fake_run),
        patch.object(
            instalar,
            "torch_runtime_info",
            return_value={
                "version": "2.13.0+cu130",
                "compiled_cuda": "13.0",
                "cuda_available": True,
                "device_name": "NVIDIA GPU",
            },
        ),
    ):
        instalar.install_huggingface(
            [sys.executable, "-m", "pip"],
            sys.executable,
            requested_profile="auto",
            run_diagnostics=False,
            model_id="test/model",
        )

    torch_commands = [cmd for cmd in commands if "torch" in cmd]
    assert torch_commands
    assert instalar.PYTORCH_CUDA_INDEX in torch_commands[-1]


def test_cuda_runtime_rejects_cpu_only_torch() -> None:
    with (
        patch.object(instalar, "run", return_value=subprocess.CompletedProcess([], 0)),
        patch.object(
            instalar,
            "torch_runtime_info",
            return_value={
                "version": "2.13.0+cpu",
                "compiled_cuda": None,
                "cuda_available": False,
                "device_name": None,
            },
        ),
    ):
        try:
            instalar.install_pytorch_runtime(
                [sys.executable, "-m", "pip"],
                sys.executable,
                selected_profile="nvidia",
            )
        except RuntimeError as exc:
            assert "degradación silenciosa" in str(exc)
        else:
            raise AssertionError("Debió rechazar una rueda CPU-only")


def test_huggingface_diagnostic_loads_model() -> None:
    command = instalar.diagnostic_command(
        sys.executable,
        detector="huggingface",
        require_cuda=True,
        model_id="test/model",
        load_model=True,
    )
    assert "--require-cuda" in command
    assert "--load-model" in command
