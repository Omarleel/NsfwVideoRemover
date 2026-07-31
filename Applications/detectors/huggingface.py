from __future__ import annotations

import importlib
import warnings
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import numpy as np

from applications.detectors.base import DetectionAssessment
from applications.detectors.config import DetectorConfig

_GIB = 1024**3


class HuggingFaceImageDetector:
    """Optimized adapter for Hugging Face image classifiers.

    Production inference uses ``AutoImageProcessor`` and
    ``AutoModelForImageClassification`` directly instead of
    ``transformers.pipeline``. This removes the generic pipeline dispatch and
    keeps batching under the application's control while preserving the same
    model, processor, FP32 precision and classification threshold.

    The optional ``classifier`` argument remains as a compatibility seam for
    tests and external integrations that inject a pipeline-like callable.
    """

    name = "transformers"

    def __init__(
        self,
        model_id: str,
        nsfw_threshold: float = 0.5,
        device: str = "auto",
        intra_op_threads: int = 0,
        classifier: Any | None = None,
        *,
        image_processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        normalized_device = device.strip().lower()
        if normalized_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device debe ser auto, cuda o cpu.")
        if not 0.0 <= nsfw_threshold <= 1.0:
            raise ValueError("nsfw_threshold debe estar entre 0 y 1.")
        if not model_id.strip():
            raise ValueError("model_id no puede estar vacío.")
        if (image_processor is None) != (model is None):
            raise ValueError("image_processor y model deben proporcionarse juntos.")

        self.model_id = model_id.strip()
        self.nsfw_threshold = float(nsfw_threshold)
        self.requested_device = normalized_device
        self.fallback_reason: str | None = None
        self._torch: Any | None = torch_module
        self.classifier: Any | None = None
        self.image_processor: Any | None = None
        self.model: Any | None = None
        self.inference_engine = "direct_fp32"
        self.precision = "float32"
        self.recommended_batch_size = 4
        self.runtime_batch_size = 0
        self.oom_fallback_count = 0
        self.model_load_source = "injected" if model is not None else "unknown"

        if classifier is not None:
            self.classifier = classifier
            self.inference_engine = "injected_pipeline"
            self.device = "cpu" if normalized_device == "auto" else normalized_device
            self.recommended_batch_size = 4
            return

        try:
            if self._torch is None:
                self._torch = importlib.import_module("torch")
            transformers = None
            if image_processor is None:
                transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "El backend Hugging Face requiere torch, transformers y Pillow. "
                "Ejecuta instalar.py --detector falconsai o instala requirements-falconsai.txt."
            ) from exc

        if intra_op_threads > 0:
            set_num_threads = getattr(self._torch, "set_num_threads", None)
            if callable(set_num_threads):
                set_num_threads(int(intra_op_threads))

        cuda_available = bool(getattr(self._torch.cuda, "is_available", lambda: False)())
        wants_cuda = normalized_device in {"auto", "cuda"} and cuda_available
        if normalized_device == "cuda" and not cuda_available:
            self.fallback_reason = "PyTorch no detectó una GPU CUDA disponible."
            warnings.warn(
                f"Se solicitó CUDA, pero se usará CPU. {self.fallback_reason}",
                RuntimeWarning,
            )

        if image_processor is None:
            assert transformers is not None
            try:
                # Cached-first avoids a Hub round trip on every video after the
                # model has already been downloaded. The online path remains the
                # transparent fallback for the first run or a cleared cache.
                try:
                    image_processor = transformers.AutoImageProcessor.from_pretrained(
                        self.model_id,
                        trust_remote_code=False,
                        local_files_only=True,
                    )
                    model = transformers.AutoModelForImageClassification.from_pretrained(
                        self.model_id,
                        trust_remote_code=False,
                        local_files_only=True,
                    )
                    self.model_load_source = "local_cache"
                except Exception:
                    image_processor = transformers.AutoImageProcessor.from_pretrained(
                        self.model_id,
                        trust_remote_code=False,
                    )
                    model = transformers.AutoModelForImageClassification.from_pretrained(
                        self.model_id,
                        trust_remote_code=False,
                    )
                    self.model_load_source = "hub_or_cache_refresh"
            except Exception as exc:
                raise RuntimeError(
                    f"No se pudo cargar el modelo Hugging Face {self.model_id!r}: {exc}"
                ) from exc

        self.image_processor = image_processor
        self.model = model
        self._prepare_model(wants_cuda=wants_cuda, requested_device=normalized_device)

    def _prepare_model(self, *, wants_cuda: bool, requested_device: str) -> None:
        assert self.model is not None
        assert self._torch is not None

        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()
        float_method = getattr(self.model, "float", None)
        if callable(float_method):
            float_method()
        requires_grad = getattr(self.model, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(False)

        if wants_cuda:
            try:
                self.model.to("cuda")
                self.device = "cuda"
                self._configure_cuda_fp32()
                self.recommended_batch_size = self._recommend_cuda_batch_size()
                return
            except Exception as exc:  # pragma: no cover - hardware/model dependent
                self.fallback_reason = f"La carga CUDA del modelo falló: {exc}"
                if requested_device == "cuda":
                    warnings.warn(
                        f"Se solicitó CUDA, pero se usará CPU. {self.fallback_reason}",
                        RuntimeWarning,
                    )
                empty_cache = getattr(self._torch.cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()

        self.model.to("cpu")
        self.device = "cpu"
        self.recommended_batch_size = 4

    def _configure_cuda_fp32(self) -> None:
        """Enable safe fixed-shape optimizations without reduced precision."""

        assert self._torch is not None
        backends = getattr(self._torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None)
        if cudnn is not None and hasattr(cudnn, "benchmark"):
            # Input tensors always use the model's fixed image size.
            cudnn.benchmark = True
        cuda_backend = getattr(backends, "cuda", None)
        matmul = getattr(cuda_backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            # Keep the optimized backend numerically equivalent to FP32 policy.
            matmul.allow_tf32 = False

    def _recommend_cuda_batch_size(self) -> int:
        """Choose a conservative high-throughput batch from currently free VRAM.

        A runtime OOM guard halves this value automatically, so other GPU
        applications cannot make the automatic choice fatal.
        """

        assert self._torch is not None
        free_bytes = 0
        mem_get_info = getattr(self._torch.cuda, "mem_get_info", None)
        if callable(mem_get_info):
            try:
                free_bytes = int(mem_get_info()[0])
            except Exception:
                free_bytes = 0
        if free_bytes <= 0:
            get_properties = getattr(self._torch.cuda, "get_device_properties", None)
            if callable(get_properties):
                try:
                    free_bytes = int(get_properties(0).total_memory)
                except Exception:
                    free_bytes = 0

        if free_bytes >= 12 * _GIB:
            return 32
        if free_bytes >= 6 * _GIB:
            return 16
        if free_bytes >= 3 * _GIB:
            return 8
        return 4

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "HuggingFaceImageDetector":
        return cls(
            model_id=config.model_id,
            nsfw_threshold=config.nsfw_threshold,
            device=config.device,
            intra_op_threads=config.intra_op_threads,
        )

    def provider_summary(self) -> str:
        summary = (
            f"modelo={self.model_id}; backend=Hugging Face; dispositivo={self.device}; "
            f"motor={self.inference_engine}; precisión={self.precision}; "
            f"carga={self.model_load_source}; "
            f"lote_recomendado={self.recommended_batch_size}; "
            f"umbral_nsfw={self.nsfw_threshold:.3f}"
        )
        if self.fallback_reason:
            summary += f"; fallback={self.fallback_reason}"
        return summary

    def performance_metadata(self) -> dict[str, Any]:
        return {
            "detector_inference_engine": self.inference_engine,
            "detector_precision": self.precision,
            "detector_model_load_source": self.model_load_source,
            "detector_recommended_batch_size": self.recommended_batch_size,
            "detector_runtime_batch_size": self.runtime_batch_size,
            "detector_oom_fallback_count": self.oom_fallback_count,
        }

    @staticmethod
    def _to_pipeline_image(image: Any) -> Any:
        """Legacy pipeline conversion retained for injected pipeline callables."""

        try:
            pillow_image = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise RuntimeError("El backend Hugging Face requiere Pillow.") from exc

        if hasattr(image, "convert"):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("Cada imagen debe tener forma alto x ancho x 3/4 canales.")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        return pillow_image.fromarray(array, mode="RGB")

    @staticmethod
    def _to_processor_image(image: Any) -> Any:
        """Match the legacy pipeline's RGB PIL preprocessing exactly.

        Passing NumPy views directly lets recent Transformers choose a tensor
        image-processing backend. Besides warning for FFmpeg's read-only buffer,
        that path can use a slightly different resize implementation and alter
        classifier scores near the configured threshold. Converting to RGB PIL
        preserves the established Falconsai decisions while direct model
        execution still removes the expensive generic ``pipeline`` dispatch.
        """

        return HuggingFaceImageDetector._to_pipeline_image(image)

    @staticmethod
    def _normalize_batch_output(raw: Any, expected: int) -> list[list[dict[str, Any]]]:
        if expected == 1 and isinstance(raw, list) and (not raw or isinstance(raw[0], dict)):
            return [list(raw)]
        if not isinstance(raw, list):
            raise RuntimeError("El pipeline de Hugging Face devolvió un formato inesperado.")
        normalized = [list(item) for item in raw]
        if len(normalized) != expected:
            raise RuntimeError(
                "El pipeline de Hugging Face devolvió una cantidad inesperada de resultados."
            )
        return normalized

    @staticmethod
    def _is_nsfw_label(label: str) -> bool:
        normalized = label.strip().casefold().replace("-", "_").replace(" ", "_")
        return normalized in {"nsfw", "unsafe", "porn", "explicit", "adult"}

    def _assessment(self, predictions: list[dict[str, Any]]) -> DetectionAssessment:
        normalized_predictions: list[dict[str, Any]] = []
        nsfw_score = 0.0
        nsfw_label: str | None = None
        for prediction in predictions:
            label = str(prediction.get("label", prediction.get("class", "")))
            score = max(0.0, min(1.0, float(prediction.get("score", 0.0))))
            normalized_predictions.append({"class": label, "score": score})
            if self._is_nsfw_label(label) and score > nsfw_score:
                nsfw_score = score
                nsfw_label = label

        is_nsfw = nsfw_score >= self.nsfw_threshold
        reason = None
        if is_nsfw:
            reason = f"{nsfw_label or 'nsfw'} alcanzó {nsfw_score:.3f}"
        return DetectionAssessment(
            is_nsfw=is_nsfw,
            score=nsfw_score,
            detections=tuple(normalized_predictions),
            reason=reason,
            metrics={"nsfw": nsfw_score},
            model_name=self.model_id,
        )

    def _label_for_index(self, index: int) -> str:
        assert self.model is not None
        config = getattr(self.model, "config", None)
        id2label = getattr(config, "id2label", None)
        fallback_label: str | None = None
        if isinstance(id2label, Mapping):
            label = id2label.get(index, id2label.get(str(index)))
            if label is not None:
                fallback_label = str(label)
                if not fallback_label.upper().startswith("LABEL_"):
                    return fallback_label
        label_names = getattr(config, "label_names", None)
        if isinstance(label_names, (list, tuple)) and 0 <= index < len(label_names):
            return str(label_names[index])
        return fallback_label or f"LABEL_{index}"

    def _move_inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            tensor = value
            if self.device == "cuda" and hasattr(tensor, "pin_memory"):
                try:
                    if not bool(getattr(tensor, "is_pinned", lambda: False)()):
                        tensor = tensor.pin_memory()
                except Exception:
                    tensor = value
            if hasattr(tensor, "to"):
                tensor = tensor.to(self.device, non_blocking=self.device == "cuda")
            moved[key] = tensor
        return moved

    def _direct_predictions(
        self,
        converted: Sequence[Any],
        batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        assert self.image_processor is not None
        assert self.model is not None
        assert self._torch is not None

        all_predictions: list[list[dict[str, Any]]] = []
        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode if callable(inference_mode) else nullcontext

        for start in range(0, len(converted), batch_size):
            chunk = list(converted[start : start + batch_size])
            encoded = self.image_processor(images=chunk, return_tensors="pt")
            if not isinstance(encoded, Mapping):
                encoded = dict(encoded)
            device_inputs = self._move_inputs_to_device(encoded)
            with context():
                output = self.model(**device_inputs)
                logits = output.logits
                shape = tuple(getattr(logits, "shape", ()))
                if shape and shape[-1] == 1 and callable(getattr(self._torch, "sigmoid", None)):
                    probabilities = self._torch.sigmoid(logits)
                else:
                    probabilities = self._torch.softmax(logits, dim=-1)
            rows = probabilities.detach().float().cpu().tolist()
            for row in rows:
                predictions = [
                    {"label": self._label_for_index(index), "score": float(score)}
                    for index, score in enumerate(row)
                ]
                predictions.sort(key=lambda item: item["score"], reverse=True)
                all_predictions.append(predictions)
        return all_predictions

    def _is_oom_error(self, exc: BaseException) -> bool:
        if self._torch is not None:
            cuda = getattr(self._torch, "cuda", None)
            oom_type = getattr(cuda, "OutOfMemoryError", None)
            if isinstance(oom_type, type) and isinstance(exc, oom_type):
                return True
        return "out of memory" in str(exc).casefold()

    def _analyze_direct(
        self,
        images: Sequence[Any],
        requested_batch_size: int,
    ) -> list[DetectionAssessment]:
        converted = [self._to_processor_image(image) for image in images]
        current_batch = max(1, min(int(requested_batch_size), len(converted)))
        while True:
            try:
                outputs = self._direct_predictions(converted, current_batch)
                self.runtime_batch_size = max(self.runtime_batch_size, current_batch)
                return [self._assessment(predictions) for predictions in outputs]
            except RuntimeError as exc:
                if current_batch <= 1 or not self._is_oom_error(exc):
                    raise
                self.oom_fallback_count += 1
                current_batch = max(1, current_batch // 2)
                self.recommended_batch_size = min(
                    self.recommended_batch_size,
                    current_batch,
                )
                empty_cache = getattr(self._torch.cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
                warnings.warn(
                    "Memoria CUDA insuficiente para el lote solicitado; "
                    f"reintentando con lote={current_batch}.",
                    RuntimeWarning,
                )

    def analyze_batch(
        self,
        images: Sequence[Any],
        batch_size: int | None = None,
    ) -> list[DetectionAssessment]:
        if not images:
            return []
        effective_batch_size = max(
            1,
            int(batch_size or self.recommended_batch_size or len(images)),
        )

        if self.classifier is not None:
            converted = [self._to_pipeline_image(image) for image in images]
            raw = self.classifier(
                converted,
                batch_size=effective_batch_size,
                top_k=None,
            )
            outputs = self._normalize_batch_output(raw, len(converted))
            self.runtime_batch_size = max(
                self.runtime_batch_size,
                min(effective_batch_size, len(converted)),
            )
            return [self._assessment(predictions) for predictions in outputs]

        return self._analyze_direct(images, effective_batch_size)

    def is_nsfw(self, image: Any) -> DetectionAssessment:
        return self.analyze_batch([image], batch_size=1)[0]
