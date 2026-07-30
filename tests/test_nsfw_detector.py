from __future__ import annotations

import sys
import types
import unittest
from typing import Any

from applications.detectors.config import DetectorConfig
from applications.detectors.factory import DetectorFactory
from applications.detectors.huggingface import HuggingFaceImageDetector
from applications.detectors.nudenet import NudeNetDetector


class FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.execution_mode = None
        self.graph_optimization_level = None


class FakeSession:
    fail_cuda = False
    calls: list[list[Any]] = []
    kwargs_calls: list[dict[str, Any]] = []

    def __init__(self, _model_path: str, *args: Any, **kwargs: Any) -> None:
        providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])
        self.calls.append(providers)
        self.kwargs_calls.append(dict(kwargs))
        provider_names = [item[0] if isinstance(item, tuple) else item for item in providers]
        if self.fail_cuda and "CUDAExecutionProvider" in provider_names:
            raise RuntimeError("simulated CUDA initialization failure")
        self._providers = provider_names

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def disable_fallback(self) -> None:
        pass

    def enable_fallback(self) -> None:
        pass


class FakeNudeDetector:
    detections: list[dict[str, Any]] = []
    batch_detections: list[list[dict[str, Any]]] = []
    batch_calls: list[tuple[int, int]] = []

    def __init__(self, providers: list[Any] | None = None) -> None:
        import onnxruntime as ort

        self.received_providers = providers
        self.onnx_session = ort.InferenceSession("fake-model.onnx")

    def detect(self, _image: Any) -> list[dict[str, Any]]:
        return list(self.detections)

    def detect_batch(
        self, images: list[Any], batch_size: int = 4
    ) -> list[list[dict[str, Any]]]:
        self.batch_calls.append((len(images), batch_size))
        if self.batch_detections:
            return [list(items) for items in self.batch_detections]
        return [list(self.detections) for _ in images]


class NudeNetDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSession.calls = []
        FakeSession.kwargs_calls = []
        FakeSession.fail_cuda = False
        FakeNudeDetector.detections = []
        FakeNudeDetector.batch_detections = []
        FakeNudeDetector.batch_calls = []

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.InferenceSession = FakeSession
        fake_ort.SessionOptions = FakeSessionOptions
        fake_ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
        fake_ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
        fake_ort.get_available_providers = lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        fake_ort.preload_dlls = lambda **_kwargs: None

        fake_nudenet = types.ModuleType("nudenet")
        fake_nudenet.NudeDetector = FakeNudeDetector

        self.original_ort = sys.modules.get("onnxruntime")
        self.original_nudenet = sys.modules.get("nudenet")
        sys.modules["onnxruntime"] = fake_ort
        sys.modules["nudenet"] = fake_nudenet

    def tearDown(self) -> None:
        if self.original_ort is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = self.original_ort
        if self.original_nudenet is None:
            sys.modules.pop("nudenet", None)
        else:
            sys.modules["nudenet"] = self.original_nudenet

    def test_auto_injects_cuda_even_when_nudenet_ignores_argument(self) -> None:
        detector = NudeNetDetector(0.15, 0.65, device="auto")
        self.assertEqual(detector.device, "cuda")
        first_call_names = [
            item[0] if isinstance(item, tuple) else item for item in FakeSession.calls[0]
        ]
        self.assertEqual(
            first_call_names,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_cuda_initialization_failure_falls_back_to_cpu(self) -> None:
        FakeSession.fail_cuda = True
        detector = NudeNetDetector(0.15, 0.65, device="auto")
        self.assertEqual(detector.device, "cpu")
        self.assertIn("CUDA no pudo inicializarse", detector.fallback_reason or "")
        self.assertEqual(FakeSession.calls[-1], ["CPUExecutionProvider"])

    def test_cpu_thread_budget_is_forwarded(self) -> None:
        detector = NudeNetDetector(0.15, 0.65, device="cpu", intra_op_threads=3)
        self.assertEqual(detector.device, "cpu")
        options = FakeSession.kwargs_calls[-1]["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 3)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, "sequential")
        self.assertEqual(options.graph_optimization_level, "all")

    def test_max_aggregation_does_not_dilute_a_strong_detection(self) -> None:
        detector = NudeNetDetector(0.60, 0.90, device="cpu", aggregation="max")
        detector.nude_detector.detections = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.95},
            {"class": "ANUS_EXPOSED", "score": 0.05},
            {"class": "FACE_FEMALE", "score": 1.0},
        ]
        assessment = detector.is_nsfw(object())
        self.assertTrue(assessment.is_nsfw)
        self.assertAlmostEqual(assessment.exposed_score, 0.95)
        self.assertIn("FEMALE_BREAST_EXPOSED", assessment.reason or "")

    def test_mean_aggregation_preserves_legacy_option(self) -> None:
        detector = NudeNetDetector(0.60, 0.90, device="cpu", aggregation="mean")
        detector.nude_detector.detections = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.95},
            {"class": "ANUS_EXPOSED", "score": 0.05},
        ]
        assessment = detector.is_nsfw(object())
        self.assertFalse(assessment.is_nsfw)
        self.assertAlmostEqual(assessment.exposed_score, 0.50)

    def test_native_batch_returns_model_independent_assessments(self) -> None:
        detector = NudeNetDetector(0.15, 0.65, device="cpu")
        detector.nude_detector.batch_detections = [
            [{"class": "BELLY_EXPOSED", "score": 0.40}],
            [{"class": "BUTTOCKS_COVERED", "score": 0.70}],
        ]
        results = detector.analyze_batch([object(), object()], batch_size=2)
        self.assertEqual(detector.nude_detector.batch_calls, [(2, 2)])
        self.assertTrue(results[0].is_nsfw)
        self.assertAlmostEqual(results[0].exposed_score, 0.20)
        self.assertTrue(results[1].is_nsfw)
        self.assertAlmostEqual(results[1].covered_score, 0.70)
        # Old tuple-style callers still work.
        self.assertTrue(results[0][0])


class FakeClassifier:
    def __init__(self, outputs: list[list[dict[str, Any]]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, int, Any]] = []

    def __call__(self, images: list[Any], *, batch_size: int, top_k: Any):
        self.calls.append((len(images), batch_size, top_k))
        return self.outputs


class HuggingFaceDetectorTests(unittest.TestCase):
    def test_binary_predictions_are_normalized(self) -> None:
        classifier = FakeClassifier(
            [
                [
                    {"label": "normal", "score": 0.1},
                    {"label": "nsfw", "score": 0.9},
                ],
                [
                    {"label": "normal", "score": 0.8},
                    {"label": "nsfw", "score": 0.2},
                ],
            ]
        )
        detector = HuggingFaceImageDetector(
            "example/model",
            nsfw_threshold=0.5,
            device="cpu",
            classifier=classifier,
        )
        detector._to_pipeline_image = lambda image: image  # type: ignore[method-assign]
        results = detector.analyze_batch([object(), object()], batch_size=2)
        self.assertEqual(classifier.calls, [(2, 2, None)])
        self.assertTrue(results[0].is_nsfw)
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertFalse(results[1].is_nsfw)
        self.assertEqual(results[0].model_name, "example/model")

    def test_threshold_is_inclusive_for_classifier_score(self) -> None:
        detector = HuggingFaceImageDetector(
            "example/model",
            nsfw_threshold=0.5,
            device="cpu",
            classifier=FakeClassifier([]),
        )
        assessment = detector._assessment([{"label": "NSFW", "score": 0.5}])
        self.assertTrue(assessment.is_nsfw)


class FactoryTests(unittest.TestCase):
    def test_custom_detector_can_be_registered_without_processor_changes(self) -> None:
        class CustomDetector:
            name = "custom"
            device = "cpu"

            def analyze_batch(self, images, batch_size=None):
                return []

            def provider_summary(self) -> str:
                return "custom"

        DetectorFactory.register(
            "custom-test",
            lambda _config: CustomDetector(),
            replace=True,
        )
        detector = DetectorFactory.create(DetectorConfig(backend="custom-test"))
        self.assertEqual(detector.name, "custom")


if __name__ == "__main__":
    unittest.main()
