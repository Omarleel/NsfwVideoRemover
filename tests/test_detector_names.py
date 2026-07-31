from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

if "progress.bar" not in sys.modules:
    progress_module = types.ModuleType("progress")
    progress_bar_module = types.ModuleType("progress.bar")

    class _ChargingBar:
        def __init__(self, *_args, max=0, **_kwargs):
            self.max = int(max or 0)
            self.index = 0

        def next(self):
            self.index += 1

        def finish(self):
            return None

    progress_bar_module.ChargingBar = _ChargingBar
    progress_module.bar = progress_bar_module
    sys.modules["progress"] = progress_module
    sys.modules["progress.bar"] = progress_bar_module

import NsfwVideoRemover
import diagnostico
import instalar
from applications.constants import (
    SUPPORTED_DETECTOR_NAMES,
    normalize_detector_backend,
)
from applications.detectors.config import DetectorConfig
from applications.detectors.factory import DetectorFactory


def test_only_canonical_detector_names_are_public() -> None:
    assert SUPPORTED_DETECTOR_NAMES == ("nudenet", "falconsai", "freepik")
    assert "huggingface" not in SUPPORTED_DETECTOR_NAMES


def test_normalization_does_not_translate_huggingface() -> None:
    assert normalize_detector_backend(" Falconsai ") == "falconsai"
    assert normalize_detector_backend("huggingface") == "huggingface"


def test_main_cli_rejects_huggingface_as_detector() -> None:
    parser = NsfwVideoRemover.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["video.mp4", "--detector", "huggingface"])
    assert exc_info.value.code == 2


def test_installer_rejects_huggingface_as_detector() -> None:
    with patch.object(sys, "argv", ["instalar.py", "--detector", "huggingface"]):
        with pytest.raises(SystemExit) as exc_info:
            instalar.main()
    assert exc_info.value.code == 2


def test_diagnostic_rejects_huggingface_as_detector() -> None:
    with patch.object(sys, "argv", ["diagnostico.py", "--detector", "huggingface"]):
        with pytest.raises(SystemExit) as exc_info:
            diagnostico.parse_args()
    assert exc_info.value.code == 2


def test_factory_does_not_register_huggingface_detector() -> None:
    assert "huggingface" not in DetectorFactory.available()
    with pytest.raises(ValueError, match="Detector desconocido"):
        DetectorFactory.create(DetectorConfig(backend="huggingface"))
