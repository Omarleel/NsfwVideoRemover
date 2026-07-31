from __future__ import annotations

from dataclasses import dataclass, replace

from applications.constants import (
    DEFAULT_COVERED_THRESHOLD,
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_EXPOSED_THRESHOLD,
    DEFAULT_FREEPIK_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_UNSAFE_THRESHOLD,
    DEFAULT_NSFW_THRESHOLD,
    default_model_for_backend,
    normalize_detector_backend,
)


@dataclass(frozen=True)
class DetectorConfig:
    """Serializable detector configuration suitable for multiprocessing workers."""

    backend: str = DEFAULT_DETECTOR_BACKEND
    device: str = "auto"
    model_id: str = ""
    nsfw_threshold: float = DEFAULT_NSFW_THRESHOLD
    exposed_threshold: float = DEFAULT_EXPOSED_THRESHOLD
    covered_threshold: float = DEFAULT_COVERED_THRESHOLD
    nudenet_aggregation: str = "max"
    freepik_unsafe_threshold: float = DEFAULT_FREEPIK_UNSAFE_THRESHOLD
    freepik_medium_high_threshold: float = DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD
    freepik_high_threshold: float = DEFAULT_FREEPIK_HIGH_THRESHOLD
    intra_op_threads: int = 0

    def __post_init__(self) -> None:
        backend = normalize_detector_backend(self.backend)
        device = self.device.strip().lower()
        aggregation = self.nudenet_aggregation.strip().lower()
        if not backend:
            raise ValueError("backend no puede estar vacío.")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device debe ser auto, cuda o cpu.")
        if not 0.0 <= self.nsfw_threshold <= 1.0:
            raise ValueError("nsfw_threshold debe estar entre 0 y 1.")
        if not 0.0 <= self.freepik_unsafe_threshold <= 1.0:
            raise ValueError("freepik_unsafe_threshold debe estar entre 0 y 1.")
        if not 0.0 <= self.freepik_medium_high_threshold <= 1.0:
            raise ValueError("freepik_medium_high_threshold debe estar entre 0 y 1.")
        if not 0.0 <= self.freepik_high_threshold <= 1.0:
            raise ValueError("freepik_high_threshold debe estar entre 0 y 1.")
        if not 0.0 <= self.exposed_threshold <= 1.0:
            raise ValueError("exposed_threshold debe estar entre 0 y 1.")
        if not 0.0 <= self.covered_threshold <= 1.0:
            raise ValueError("covered_threshold debe estar entre 0 y 1.")
        if aggregation not in {"max", "mean"}:
            raise ValueError("nudenet_aggregation debe ser max o mean.")
        if self.intra_op_threads < 0:
            raise ValueError("intra_op_threads no puede ser negativo.")

        model_id = self.model_id.strip() or default_model_for_backend(backend)

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "nudenet_aggregation", aggregation)
        object.__setattr__(self, "model_id", model_id)

    def with_runtime(self, *, device: str, intra_op_threads: int) -> "DetectorConfig":
        return replace(
            self,
            device=device,
            intra_op_threads=max(0, int(intra_op_threads)),
        )
