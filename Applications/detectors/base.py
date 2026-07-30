from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class DetectionAssessment:
    """Model-independent result for a single image.

    ``metrics`` may contain backend-specific diagnostics, while the fields used
    by the video pipeline remain stable for every detector implementation.
    """

    is_nsfw: bool
    score: float
    detections: tuple[dict[str, Any], ...] = ()
    reason: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    model_name: str = "unknown"

    @property
    def exposed_score(self) -> float:
        return float(self.metrics.get("exposed", 0.0))

    @property
    def covered_score(self) -> float:
        return float(self.metrics.get("covered", 0.0))

    def as_legacy_tuple(self) -> tuple[bool, list[dict[str, Any]], float, float]:
        """Keep compatibility with callers that unpacked the old NudeNet tuple."""

        return (
            self.is_nsfw,
            [dict(item) for item in self.detections],
            self.exposed_score,
            self.covered_score,
        )

    def __iter__(self):
        return iter(self.as_legacy_tuple())

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return self.as_legacy_tuple()[index]


@runtime_checkable
class ContentDetector(Protocol):
    """Stable contract consumed by the video processor (dependency inversion)."""

    name: str
    device: str

    def analyze_batch(
        self,
        images: Sequence[Any],
        batch_size: int | None = None,
    ) -> list[DetectionAssessment]: ...

    def provider_summary(self) -> str: ...
