from __future__ import annotations

from typing import Any

from applications.detectors.huggingface import HuggingFaceImageDetector


class FalconsaiImageDetector(HuggingFaceImageDetector):
    """Falconsai binary classifier hosted on Hugging Face Hub."""

    name = "falconsai"

    def provider_summary(self) -> str:
        summary = (
            f"modelo={self.model_id}; detector=Falconsai; proveedor=Hugging Face; "
            f"dispositivo={self.device}; motor={self.inference_engine}; "
            f"precisión={self.precision}; carga={self.model_load_source}; "
            f"lote_recomendado={self.recommended_batch_size}; "
            f"umbral_nsfw={self.nsfw_threshold:.3f}"
        )
        if self.fallback_reason:
            summary += f"; fallback={self.fallback_reason}"
        return summary

    def performance_metadata(self) -> dict[str, Any]:
        metadata = super().performance_metadata()
        metadata.update(
            {
                "detector_name": self.name,
                "detector_provider": "huggingface",
            }
        )
        return metadata
