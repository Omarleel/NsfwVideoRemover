from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[1] / "applications" / "NsfwDetector.py"


class FakeSession:
    fail_cuda = False
    calls: list[list[Any]] = []

    def __init__(self, _model_path: str, *args: Any, **kwargs: Any) -> None:
        providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])
        self.calls.append(providers)
        provider_names = [item[0] if isinstance(item, tuple) else item for item in providers]
        if self.fail_cuda and "CUDAExecutionProvider" in provider_names:
            raise RuntimeError("simulated CUDA initialization failure")
        self._providers = provider_names

    def get_providers(self) -> list[str]:
        return list(self._providers)


class FakeNudeDetector:
    detections: list[dict[str, Any]] = []

    def __init__(self, providers: list[Any] | None = None) -> None:
        # Mimics NudeNet 3.4.2: accepts `providers` but does not forward it.
        import onnxruntime as ort

        self.received_providers = providers
        self.onnx_session = ort.InferenceSession("fake-model.onnx")

    def detect(self, _image: Any) -> list[dict[str, Any]]:
        return list(self.detections)


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSession.calls = []
        FakeSession.fail_cuda = False
        FakeNudeDetector.detections = []

    def _load_module(self, available_providers: list[str]):
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_ort.get_available_providers = lambda: list(available_providers)
        fake_ort.preload_dlls = lambda **_kwargs: None

        fake_nudenet = types.ModuleType("nudenet")
        fake_nudenet.NudeDetector = FakeNudeDetector

        old_ort = sys.modules.get("onnxruntime")
        old_nudenet = sys.modules.get("nudenet")
        sys.modules["onnxruntime"] = fake_ort
        sys.modules["nudenet"] = fake_nudenet

        def restore_modules() -> None:
            if old_ort is None:
                sys.modules.pop("onnxruntime", None)
            else:
                sys.modules["onnxruntime"] = old_ort
            if old_nudenet is None:
                sys.modules.pop("nudenet", None)
            else:
                sys.modules["nudenet"] = old_nudenet

        self.addCleanup(restore_modules)
        module_name = f"test_nsfw_detector_impl_{id(self)}_{len(FakeSession.calls)}"
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_auto_injects_cuda_even_when_nudenet_ignores_argument(self) -> None:
        module = self._load_module(["CUDAExecutionProvider", "CPUExecutionProvider"])
        detector = module.NsfwDetector(0.15, 0.65, device="auto")

        self.assertEqual(detector.device, "cuda")
        self.assertEqual(detector.active_providers[0], "CUDAExecutionProvider")
        first_call_names = [
            item[0] if isinstance(item, tuple) else item for item in FakeSession.calls[0]
        ]
        self.assertEqual(
            first_call_names,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_cuda_initialization_failure_falls_back_to_cpu(self) -> None:
        FakeSession.fail_cuda = True
        module = self._load_module(["CUDAExecutionProvider", "CPUExecutionProvider"])
        detector = module.NsfwDetector(0.15, 0.65, device="auto")

        self.assertEqual(detector.device, "cpu")
        self.assertIn("CUDA no pudo inicializarse", detector.fallback_reason or "")
        self.assertEqual(FakeSession.calls[-1], ["CPUExecutionProvider"])

    def test_cpu_request_never_tries_cuda(self) -> None:
        module = self._load_module(["CUDAExecutionProvider", "CPUExecutionProvider"])
        detector = module.NsfwDetector(0.15, 0.65, device="cpu")

        self.assertEqual(detector.device, "cpu")
        self.assertEqual(FakeSession.calls, [["CPUExecutionProvider"]])

    def test_thresholds_keep_original_belly_weighting_and_exclusions(self) -> None:
        module = self._load_module(["CPUExecutionProvider"])
        detector = module.NsfwDetector(0.15, 0.65, device="cpu")
        detector.nude_detector.detections = [
            {"class": "BELLY_EXPOSED", "score": 0.40},
            {"class": "FACE_FEMALE", "score": 1.00},
            {"class": "BUTTOCKS_COVERED", "score": 0.70},
        ]

        is_nsfw, _detections, exposed, covered = detector.is_nsfw(object())
        self.assertTrue(is_nsfw)
        self.assertAlmostEqual(exposed, 0.20)
        self.assertAlmostEqual(covered, 0.70)


if __name__ == "__main__":
    unittest.main()
