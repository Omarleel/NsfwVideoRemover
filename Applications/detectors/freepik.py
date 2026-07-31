from __future__ import annotations

from typing import Any

from applications.constants import (
    DEFAULT_FREEPIK_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_MODEL,
    DEFAULT_FREEPIK_UNSAFE_THRESHOLD,
)
from applications.detectors.base import DetectionAssessment
from applications.detectors.config import DetectorConfig
from applications.detectors.huggingface import HuggingFaceImageDetector

_GIB = 1024**3


class FreepikImageDetector(HuggingFaceImageDetector):
    """Adapter for Freepik's four-level EVA-02 NSFW classifier.

    The model emits ``neutral``, ``low``, ``medium`` and ``high``. A frame is
    rejected when any conservative guardrail is reached:

    * low + medium + high >= unsafe_threshold
    * medium + high >= medium_high_threshold
    * high >= high_threshold

    Direct FP32 execution is intentionally retained to prioritize stable scores.
    """

    name = "freepik"
    _KNOWN_LABELS = {"neutral", "low", "medium", "high"}

    def __init__(
        self,
        model_id: str = DEFAULT_FREEPIK_MODEL,
        unsafe_threshold: float = DEFAULT_FREEPIK_UNSAFE_THRESHOLD,
        medium_high_threshold: float = DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD,
        high_threshold: float = DEFAULT_FREEPIK_HIGH_THRESHOLD,
        device: str = "auto",
        intra_op_threads: int = 0,
        classifier: Any | None = None,
        *,
        image_processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        for name, value in (
            ("unsafe_threshold", unsafe_threshold),
            ("medium_high_threshold", medium_high_threshold),
            ("high_threshold", high_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1.")

        self.unsafe_threshold = float(unsafe_threshold)
        self.medium_high_threshold = float(medium_high_threshold)
        self.high_threshold = float(high_threshold)
        super().__init__(
            model_id=model_id,
            nsfw_threshold=self.unsafe_threshold,
            device=device,
            intra_op_threads=intra_op_threads,
            classifier=classifier,
            image_processor=image_processor,
            model=model,
            torch_module=torch_module,
        )
        if self.inference_engine == "direct_fp32":
            self.inference_engine = "direct_fp32_freepik"

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "FreepikImageDetector":
        return cls(
            model_id=config.model_id,
            unsafe_threshold=config.freepik_unsafe_threshold,
            medium_high_threshold=config.freepik_medium_high_threshold,
            high_threshold=config.freepik_high_threshold,
            device=config.device,
            intra_op_threads=config.intra_op_threads,
        )

    def _recommend_cuda_batch_size(self) -> int:
        """Use smaller batches than 224px classifiers because input is 448px."""

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
            return 16
        if free_bytes >= 6 * _GIB:
            return 8
        if free_bytes >= 3 * _GIB:
            return 4
        return 2

    @staticmethod
    def _normalize_level(label: str) -> str:
        return label.strip().casefold().replace("-", "_").replace(" ", "_")

    def _assessment(self, predictions: list[dict[str, Any]]) -> DetectionAssessment:
        scores = {label: 0.0 for label in self._KNOWN_LABELS}
        normalized_predictions: list[dict[str, Any]] = []
        recognized = 0

        for prediction in predictions:
            label = str(prediction.get("label", prediction.get("class", "")))
            score = max(0.0, min(1.0, float(prediction.get("score", 0.0))))
            normalized_predictions.append({"class": label, "score": score})
            level = self._normalize_level(label)
            if level in scores:
                scores[level] = max(scores[level], score)
                recognized += 1

        if recognized == 0:
            labels = ", ".join(item["class"] for item in normalized_predictions)
            raise RuntimeError(
                "El modelo Freepik no devolvió las etiquetas neutral/low/medium/high. "
                f"Etiquetas recibidas: {labels or 'ninguna'}."
            )

        unsafe_score = min(1.0, scores["low"] + scores["medium"] + scores["high"])
        medium_high_score = min(1.0, scores["medium"] + scores["high"])
        triggers: list[str] = []
        epsilon = 1e-9
        if unsafe_score + epsilon >= self.unsafe_threshold:
            triggers.append(
                f"low+medium+high={unsafe_score:.3f} >= {self.unsafe_threshold:.3f}"
            )
        if medium_high_score + epsilon >= self.medium_high_threshold:
            triggers.append(
                "medium+high="
                f"{medium_high_score:.3f} >= {self.medium_high_threshold:.3f}"
            )
        if scores["high"] + epsilon >= self.high_threshold:
            triggers.append(f"high={scores['high']:.3f} >= {self.high_threshold:.3f}")

        is_nsfw = bool(triggers)
        reason = "Freepik: " + "; ".join(triggers) if triggers else None
        return DetectionAssessment(
            is_nsfw=is_nsfw,
            score=unsafe_score,
            detections=tuple(normalized_predictions),
            reason=reason,
            metrics={
                "neutral": scores["neutral"],
                "low": scores["low"],
                "medium": scores["medium"],
                "high": scores["high"],
                "unsafe": unsafe_score,
                "medium_high": medium_high_score,
            },
            model_name=self.model_id,
        )

    def provider_summary(self) -> str:
        summary = (
            f"modelo={self.model_id}; detector=Freepik EVA-02; proveedor=Hugging Face; "
            f"dispositivo={self.device}; "
            f"motor={self.inference_engine}; precisión={self.precision}; "
            f"carga={self.model_load_source}; lote_recomendado={self.recommended_batch_size}; "
            f"umbrales=unsafe:{self.unsafe_threshold:.3f},"
            f"medium_high:{self.medium_high_threshold:.3f},high:{self.high_threshold:.3f}"
        )
        if self.fallback_reason:
            summary += f"; fallback={self.fallback_reason}"
        return summary

    def performance_metadata(self) -> dict[str, Any]:
        metadata = super().performance_metadata()
        metadata.update(
            {
                "freepik_unsafe_threshold": self.unsafe_threshold,
                "freepik_medium_high_threshold": self.medium_high_threshold,
                "freepik_high_threshold": self.high_threshold,
                "detector_name": self.name,
                "detector_provider": "huggingface",
                "freepik_policy": "unsafe_or_medium_high_or_high",
            }
        )
        return metadata
