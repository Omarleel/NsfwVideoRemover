from __future__ import annotations

import importlib
import warnings
from typing import Any, Sequence

import numpy as np

from applications.detectors.base import DetectionAssessment
from applications.detectors.config import DetectorConfig


class HuggingFaceImageDetector:
    """Adapter for binary/multiclass Hugging Face image classifiers.

    It is intentionally independent from any specific architecture. The default
    model is Falconsai/nsfw_image_detection, but another compatible image-
    classification model can be selected through ``model_id``.
    """

    name = "huggingface"

    def __init__(
        self,
        model_id: str,
        nsfw_threshold: float = 0.5,
        device: str = "auto",
        intra_op_threads: int = 0,
        classifier: Any | None = None,
    ) -> None:
        normalized_device = device.strip().lower()
        if normalized_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device debe ser auto, cuda o cpu.")
        if not 0.0 <= nsfw_threshold <= 1.0:
            raise ValueError("nsfw_threshold debe estar entre 0 y 1.")

        self.model_id = model_id.strip()
        self.nsfw_threshold = float(nsfw_threshold)
        self.requested_device = normalized_device
        self.fallback_reason: str | None = None
        self._torch: Any | None = None

        if classifier is not None:
            self.classifier = classifier
            self.device = "cpu" if normalized_device == "auto" else normalized_device
            return

        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "El backend Hugging Face requiere torch, transformers y Pillow. "
                "Instala requirements-huggingface.txt."
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

        pipeline_factory = transformers.pipeline
        if wants_cuda:
            try:
                self.classifier = pipeline_factory(
                    "image-classification",
                    model=self.model_id,
                    device=0,
                    trust_remote_code=False,
                )
                self.device = "cuda"
                return
            except Exception as exc:  # pragma: no cover - hardware/model dependent
                self.fallback_reason = f"La carga o inferencia CUDA falló: {exc}"
                if normalized_device == "cuda":
                    warnings.warn(
                        f"Se solicitó CUDA, pero se usará CPU. {self.fallback_reason}",
                        RuntimeWarning,
                    )

        self.classifier = pipeline_factory(
            "image-classification",
            model=self.model_id,
            device=-1,
            trust_remote_code=False,
        )
        self.device = "cpu"

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
            f"umbral_nsfw={self.nsfw_threshold:.3f}"
        )
        if self.fallback_reason:
            summary += f"; fallback={self.fallback_reason}"
        return summary

    @staticmethod
    def _to_pipeline_image(image: Any) -> Any:
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

    def analyze_batch(
        self,
        images: Sequence[Any],
        batch_size: int | None = None,
    ) -> list[DetectionAssessment]:
        if not images:
            return []
        converted = [self._to_pipeline_image(image) for image in images]
        effective_batch_size = max(1, int(batch_size or len(converted)))
        raw = self.classifier(
            converted,
            batch_size=effective_batch_size,
            top_k=None,
        )
        outputs = self._normalize_batch_output(raw, len(converted))
        return [self._assessment(predictions) for predictions in outputs]

    def is_nsfw(self, image: Any) -> DetectionAssessment:
        return self.analyze_batch([image], batch_size=1)[0]
