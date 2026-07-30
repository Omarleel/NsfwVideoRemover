from __future__ import annotations

from collections.abc import Callable

from applications.detectors.base import ContentDetector
from applications.detectors.config import DetectorConfig


DetectorBuilder = Callable[[DetectorConfig], ContentDetector]


class DetectorFactory:
    """Composition root for detector adapters.

    The processing pipeline depends only on ``ContentDetector``. A new model is
    added by implementing that protocol and registering one builder here or at
    application startup; no video-processing code needs to change.
    """

    _builders: dict[str, DetectorBuilder] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: DetectorBuilder,
        *,
        replace: bool = False,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("El nombre del detector no puede estar vacío.")
        if normalized in cls._builders and not replace:
            raise ValueError(f"El detector {normalized!r} ya está registrado.")
        cls._builders[normalized] = builder

    @classmethod
    def create(cls, config: DetectorConfig) -> ContentDetector:
        cls._ensure_builtins()
        try:
            builder = cls._builders[config.backend]
        except KeyError as exc:
            available = ", ".join(sorted(cls._builders)) or "ninguno"
            raise ValueError(
                f"Detector desconocido: {config.backend!r}. Disponibles: {available}."
            ) from exc
        detector = builder(config)
        if not isinstance(detector, ContentDetector):
            raise TypeError(
                f"El detector {config.backend!r} no implementa ContentDetector."
            )
        return detector

    @classmethod
    def available(cls) -> tuple[str, ...]:
        cls._ensure_builtins()
        return tuple(sorted(cls._builders))

    @classmethod
    def _ensure_builtins(cls) -> None:
        if "nudenet" not in cls._builders:
            from applications.detectors.nudenet import NudeNetDetector

            cls.register("nudenet", NudeNetDetector.from_config)
        if "huggingface" not in cls._builders:
            from applications.detectors.huggingface import HuggingFaceImageDetector

            cls.register("huggingface", HuggingFaceImageDetector.from_config)


def create_detector(config: DetectorConfig) -> ContentDetector:
    return DetectorFactory.create(config)
