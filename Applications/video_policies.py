from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentPlanner:
    clip_duration: float
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.clip_duration <= 0:
            raise ValueError("clip_duration debe ser mayor que cero.")

    def build(self, duration: float) -> list[dict[str, Any]]:
        """Create stable intervals without cumulative floating-point drift."""

        duration = max(0.0, float(duration))
        if duration <= self.epsilon:
            return []
        count = int(math.ceil((duration - self.epsilon) / self.clip_duration))
        segments: list[dict[str, Any]] = []
        for zero_based_index in range(count):
            start = round(zero_based_index * self.clip_duration, 12)
            end = round(
                min((zero_based_index + 1) * self.clip_duration, duration), 12
            )
            if end - start <= self.epsilon:
                continue
            segments.append(
                {
                    "orden": len(segments) + 1,
                    "intervalo": [start, end],
                    "detecciones": None,
                    "nsfw": None,
                }
            )
        return segments


@dataclass(frozen=True)
class CutIntervalPolicy:
    padding_seconds: float

    def __post_init__(self) -> None:
        if self.padding_seconds < 0:
            raise ValueError("padding_seconds no puede ser negativo.")

    def build_cut_intervals(
        self,
        results: list[dict[str, Any]],
        duration: float,
        padding_seconds: float | None = None,
    ) -> list[tuple[float, float]]:
        padding = (
            self.padding_seconds
            if padding_seconds is None
            else max(0.0, float(padding_seconds))
        )
        duration = max(0.0, float(duration))
        raw: list[tuple[float, float]] = []
        for result in results:
            if not result.get("nsfw"):
                continue
            interval_start, interval_end = result["intervalo"]
            detected_start = max(0.0, float(interval_start))
            detected_end = min(duration, max(detected_start, float(interval_end)))
            start = round(max(0.0, detected_start - padding), 12)
            end = round(min(duration, detected_end + padding), 12)
            if end > start:
                raw.append((start, end))
        if not raw:
            return []

        raw.sort(key=lambda item: item[0])
        merged = [raw[0]]
        for start, end in raw[1:]:
            previous_start, previous_end = merged[-1]
            if start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def build_allowed_intervals(
        duration: float,
        cut_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        duration = max(0.0, float(duration))
        allowed: list[tuple[float, float]] = []
        cursor = 0.0
        for cut_start, cut_end in cut_intervals:
            if cut_start > cursor:
                allowed.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < duration:
            allowed.append((cursor, duration))
        return allowed
