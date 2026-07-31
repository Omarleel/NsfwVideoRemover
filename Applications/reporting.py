from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from applications.profiling import PerformanceProfiler
from applications.SrtGenerator import SrtGenerator


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value).__name__} a JSON.")


class AnalysisReportWriter:
    """Writes human-readable SRT and machine-readable JSON analysis reports."""

    def __init__(
        self,
        srt_generator: SrtGenerator | None = None,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.srt_generator = srt_generator or SrtGenerator()
        self.profiler = profiler

    def _event(
        self,
        name: str,
        wall_started: float,
        cpu_started: float,
        start_offset: float | None,
        **details: Any,
    ) -> None:
        if self.profiler is None:
            return
        self.profiler.event(
            "reporting_atomic",
            name,
            duration_seconds=time.perf_counter() - wall_started,
            cpu_seconds=time.process_time() - cpu_started,
            start_offset_seconds=start_offset,
            **details,
        )

    def write(
        self,
        *,
        video_path: str,
        duration: float,
        detector_name: str,
        detector_provider: str | None = None,
        model_id: str | None = None,
        results: list[dict[str, Any]],
        cut_intervals: list[tuple[float, float]],
        srt_path: str,
        json_path: str,
        allowed_intervals: list[tuple[float, float]] | None = None,
        rendered_intervals: list[tuple[float, float]] | None = None,
        render_mode: str | None = None,
        expected_output_duration: float | None = None,
        actual_output_duration: float | None = None,
    ) -> None:
        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        self.srt_generator.reset()
        self._event("reset_srt", started, cpu_started, offset)

        subtitle_total_started = time.perf_counter()
        subtitle_total_cpu = time.process_time()
        subtitle_total_offset = self.profiler.now_offset_seconds() if self.profiler else None
        for result in results:
            item_started = time.perf_counter()
            item_cpu = time.process_time()
            item_offset = self.profiler.now_offset_seconds() if self.profiler else None
            start_time, end_time = result["intervalo"]
            subtitle_payload = {
                "nsfw": bool(result.get("nsfw")),
                "score": float(result.get("score_nsfw", 0.0)),
                "reason": result.get("motivo"),
                "detections": result.get("detecciones") or [],
            }
            self.srt_generator.add_subtitle(start_time, end_time, subtitle_payload)
            self._event(
                "add_srt_entry",
                item_started,
                item_cpu,
                item_offset,
                segment_order=result.get("orden"),
                detection_count=len(subtitle_payload["detections"]),
            )
        self._event(
            "build_all_srt_entries",
            subtitle_total_started,
            subtitle_total_cpu,
            subtitle_total_offset,
            segment_count=len(results),
        )

        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        self.srt_generator.generate_srt(srt_path)
        self._event(
            "write_srt_file",
            started,
            cpu_started,
            offset,
            path=srt_path,
            segment_count=len(results),
        )

        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        report = {
            "schema_version": 1,
            "video": str(Path(video_path)),
            "duration_seconds": float(duration),
            "detector": detector_name,
            "provider": detector_provider,
            "model_id": model_id,
            "cut_intervals": [[start, end] for start, end in cut_intervals],
            "allowed_intervals": [
                [start, end] for start, end in (allowed_intervals or [])
            ],
            "rendered_intervals": (
                [[start, end] for start, end in rendered_intervals]
                if rendered_intervals is not None
                else None
            ),
            "render_mode": render_mode,
            "expected_output_duration_seconds": expected_output_duration,
            "actual_output_duration_seconds": actual_output_duration,
            "segments": results,
        }
        self._event(
            "build_analysis_json_payload",
            started,
            cpu_started,
            offset,
            segment_count=len(results),
            cut_interval_count=len(cut_intervals),
        )

        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        self._write_json_atomic(json_path, report)
        self._event(
            "write_analysis_json_file",
            started,
            cpu_started,
            offset,
            path=json_path,
            segment_count=len(results),
        )

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
