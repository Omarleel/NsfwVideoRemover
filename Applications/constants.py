"""Shared defaults and detector naming helpers used by the application."""

from __future__ import annotations

DEFAULT_CLIP_DURATION = 1.0
DEFAULT_CUT_PADDING_SECONDS = 4.0
DEFAULT_ANALYSIS_MAX_DIMENSION = 1280
DEFAULT_EXPOSED_THRESHOLD = 0.15
DEFAULT_COVERED_THRESHOLD = 0.65
DEFAULT_NSFW_THRESHOLD = 0.50
DEFAULT_NUDENET_MODEL = "NudeNet"
DEFAULT_FALCONSAI_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_FREEPIK_MODEL = "Freepik/nsfw_image_detector"
DEFAULT_FREEPIK_UNSAFE_THRESHOLD = 0.60
DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD = 0.45
DEFAULT_FREEPIK_HIGH_THRESHOLD = 0.25
DEFAULT_DETECTOR_BACKEND = "nudenet"

CANONICAL_DETECTOR_BACKENDS = ("nudenet", "falconsai", "freepik")
SUPPORTED_DETECTOR_NAMES = CANONICAL_DETECTOR_BACKENDS


def normalize_detector_backend(value: str) -> str:
    """Normalize a detector name without translating or accepting aliases."""

    return str(value or "").strip().casefold()


def default_model_for_backend(value: str) -> str:
    """Return the built-in model identifier for a supported backend."""

    backend = normalize_detector_backend(value)
    if backend == "falconsai":
        return DEFAULT_FALCONSAI_MODEL
    if backend == "freepik":
        return DEFAULT_FREEPIK_MODEL
    if backend == "nudenet":
        return DEFAULT_NUDENET_MODEL
    return ""


def provider_for_backend(value: str) -> str:
    """Return the runtime/model provider independently from detector identity."""

    backend = normalize_detector_backend(value)
    if backend in {"falconsai", "freepik"}:
        return "huggingface"
    if backend == "nudenet":
        return "onnxruntime"
    return "custom"
