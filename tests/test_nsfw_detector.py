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




class FakeTensor:
    def __init__(self, values: Any) -> None:
        import numpy as np

        self.values = np.asarray(values, dtype=float)

    @property
    def shape(self):
        return self.values.shape

    def pin_memory(self):
        return self

    def is_pinned(self) -> bool:
        return True

    def to(self, *_args: Any, **_kwargs: Any):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


class FakeTorch:
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class cuda:
        OutOfMemoryError = None

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    cuda.OutOfMemoryError = FakeOutOfMemoryError

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def softmax(tensor: FakeTensor, dim: int = -1) -> FakeTensor:
        import numpy as np

        values = tensor.values
        shifted = values - values.max(axis=dim, keepdims=True)
        exp = np.exp(shifted)
        return FakeTensor(exp / exp.sum(axis=dim, keepdims=True))


class FakeImageProcessor:
    def __call__(self, *, images: list[Any], return_tensors: str):
        self.last_count = len(images)
        self.last_images = list(images)
        self.return_tensors = return_tensors
        return {"pixel_values": FakeTensor([[index] for index in range(len(images))])}


class FakeDirectModel:
    def __init__(self, *, fail_above: int = 0) -> None:
        self.config = types.SimpleNamespace(id2label={0: "normal", 1: "nsfw"})
        self.fail_above = fail_above
        self.batch_calls: list[int] = []
        self.eval_called = False
        self.float_called = False
        self.device = None

    def eval(self):
        self.eval_called = True
        return self

    def float(self):
        self.float_called = True
        return self

    def requires_grad_(self, _enabled: bool):
        return self

    def to(self, device: str):
        self.device = device
        return self

    def __call__(self, *, pixel_values: FakeTensor):
        count = len(pixel_values.values)
        self.batch_calls.append(count)
        if self.fail_above and count > self.fail_above:
            raise RuntimeError("CUDA out of memory")
        logits = [[0.0, 2.0] if index % 2 == 0 else [2.0, 0.0] for index in range(count)]
        return types.SimpleNamespace(logits=FakeTensor(logits))


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

    def test_direct_fp32_backend_batches_without_transformers_pipeline(self) -> None:
        processor = FakeImageProcessor()
        model = FakeDirectModel()
        detector = HuggingFaceImageDetector(
            "example/model",
            nsfw_threshold=0.5,
            device="cpu",
            image_processor=processor,
            model=model,
            torch_module=FakeTorch,
        )
        images = [
            __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
            __import__("numpy").ones((8, 8, 3), dtype="uint8"),
        ]
        results = detector.analyze_batch(images, batch_size=2)
        self.assertEqual(model.batch_calls, [2])
        self.assertTrue(model.eval_called)
        self.assertTrue(model.float_called)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(detector.inference_engine, "direct_fp32")
        self.assertTrue(results[0].is_nsfw)
        self.assertFalse(results[1].is_nsfw)
        self.assertAlmostEqual(results[0].score, 0.8807970779, places=6)
        self.assertTrue(all(getattr(image, "mode", None) == "RGB" for image in processor.last_images))

    def test_read_only_ffmpeg_array_is_converted_to_rgb_pil(self) -> None:
        import numpy as np

        raw = bytes([0, 1, 2] * 16)
        read_only = np.frombuffer(raw, dtype=np.uint8).reshape((4, 4, 3))
        self.assertFalse(read_only.flags.writeable)
        converted = HuggingFaceImageDetector._to_processor_image(read_only)
        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.size, (4, 4))

    def test_runtime_batch_size_keeps_largest_successful_batch(self) -> None:
        processor = FakeImageProcessor()
        detector = HuggingFaceImageDetector(
            "example/model",
            device="cpu",
            image_processor=processor,
            model=FakeDirectModel(),
            torch_module=FakeTorch,
        )
        import numpy as np

        detector.analyze_batch([np.zeros((4, 4, 3), dtype="uint8") for _ in range(4)], batch_size=4)
        detector.analyze_batch([np.zeros((4, 4, 3), dtype="uint8")], batch_size=4)
        self.assertEqual(detector.runtime_batch_size, 4)

    def test_direct_backend_halves_batch_after_oom(self) -> None:
        model = FakeDirectModel(fail_above=2)
        detector = HuggingFaceImageDetector(
            "example/model",
            device="cpu",
            image_processor=FakeImageProcessor(),
            model=model,
            torch_module=FakeTorch,
        )
        images = [__import__("numpy").zeros((4, 4, 3), dtype="uint8") for _ in range(4)]
        results = detector.analyze_batch(images, batch_size=4)
        self.assertEqual(len(results), 4)
        self.assertEqual(model.batch_calls, [4, 2, 2])
        self.assertEqual(detector.runtime_batch_size, 2)
        self.assertEqual(detector.oom_fallback_count, 1)

    def test_production_loader_prefers_local_cache(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        class FakeAutoImageProcessor:
            @staticmethod
            def from_pretrained(_model_id: str, **kwargs: Any):
                calls.append(("processor", dict(kwargs)))
                return FakeImageProcessor()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(_model_id: str, **kwargs: Any):
                calls.append(("model", dict(kwargs)))
                return FakeDirectModel()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
        fake_transformers.AutoModelForImageClassification = FakeAutoModel
        original = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            detector = HuggingFaceImageDetector(
                "example/model",
                device="cpu",
                torch_module=FakeTorch,
            )
        finally:
            if original is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original

        self.assertEqual(detector.model_load_source, "local_cache")
        self.assertEqual([name for name, _kwargs in calls], ["processor", "model"])
        self.assertTrue(all(kwargs.get("local_files_only") for _name, kwargs in calls))


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
