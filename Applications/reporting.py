from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from applications.SrtGenerator import SrtGenerator


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value).__name__} a JSON.")


class AnalysisReportWriter:
    """Writes human-readable SRT and machine-readable JSON analysis reports."""

    def __init__(self, srt_generator: SrtGenerator | None = None) -> None:
        self.srt_generator = srt_generator or SrtGenerator()

    def write(
        self,
        *,
        video_path: str,
        duration: float,
        detector_name: str,
        results: list[dict[str, Any]],
        cut_intervals: list[tuple[float, float]],
        srt_path: str,
        json_path: str,
    ) -> None:
        self.srt_generator.reset()
        for result in results:
            start_time, end_time = result["intervalo"]
            subtitle_payload = {
                "nsfw": bool(result.get("nsfw")),
                "score": float(result.get("score_nsfw", 0.0)),
                "reason": result.get("motivo"),
                "detections": result.get("detecciones") or [],
            }
            self.srt_generator.add_subtitle(start_time, end_time, subtitle_payload)
        self.srt_generator.generate_srt(srt_path)

        report = {
            "schema_version": 1,
            "video": str(Path(video_path)),
            "duration_seconds": float(duration),
            "detector": detector_name,
            "cut_intervals": [[start, end] for start, end in cut_intervals],
            "segments": results,
        }
        self._write_json_atomic(json_path, report)

    @staticmethod
    def _write_json_atomic(file_path: str, payload: dict[str, Any]) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(
                    payload, file, ensure_ascii=False, indent=2, default=_json_default
                )
                file.write("\n")
            os.replace(temporary_name, target)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
