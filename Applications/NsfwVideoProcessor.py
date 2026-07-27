from __future__ import annotations

import atexit
import gc
import multiprocessing as mp
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import VideoFileClip, concatenate_videoclips
from progress.bar import ChargingBar

from applications.NsfwDetector import NsfwDetector
from applications.SrtGenerator import SrtGenerator


_WORKER_VIDEO: VideoFileClip | None = None
_WORKER_DETECTOR: NsfwDetector | None = None


def _subclip(video: VideoFileClip, start_time: float, end_time: float):
    """MoviePy 2.x helper kept separate to simplify future API changes."""

    return video.subclipped(start_time, end_time)


def _close_worker_resources() -> None:
    global _WORKER_VIDEO, _WORKER_DETECTOR
    if _WORKER_VIDEO is not None:
        _WORKER_VIDEO.close()
    _WORKER_VIDEO = None
    _WORKER_DETECTOR = None


def _initialize_worker(
    video_path: str,
    umbral_minimo_expuesto: float,
    umbral_minimo_cubierto: float,
    device: str,
) -> None:
    global _WORKER_VIDEO, _WORKER_DETECTOR
    _WORKER_VIDEO = VideoFileClip(video_path)
    _WORKER_DETECTOR = NsfwDetector(
        umbral_minimo_expuesto=umbral_minimo_expuesto,
        umbral_minimo_cubierto=umbral_minimo_cubierto,
        device=device,
    )
    atexit.register(_close_worker_resources)


def _analyze_segment(segment: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_VIDEO is None or _WORKER_DETECTOR is None:
        raise RuntimeError("El worker no fue inicializado correctamente.")

    start_time, _ = segment["intervalo"]
    frame = np.ascontiguousarray(_WORKER_VIDEO.get_frame(start_time))
    es_nsfw, detections, promedio_expuesto, promedio_cubierto = (
        _WORKER_DETECTOR.is_nsfw(frame)
    )

    result = dict(segment)
    result["detecciones"] = detections
    result["nsfw"] = es_nsfw
    result["promedio_expuesto"] = promedio_expuesto
    result["promedio_cubierto"] = promedio_cubierto
    return result


def _ffmpeg_encoders() -> set[str]:
    try:
        process = subprocess.run(
            [get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    encoders: set[str] = set()
    for line in process.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            encoders.add(parts[1])
    return encoders


def _nvidia_driver_is_visible() -> bool:
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "NVIDIA" in result.stdout.upper()


class NsfwVideoProcessor:
    def __init__(
        self,
        input_video_path: str,
        umbral_minimo_expuesto: float = 0.15,
        umbral_minimo_cubierto: float = 0.65,
        output_folder_path: str = "",
        clip_duration: float = 1.0,
        num_procesos: int = 0,
        device: str = "auto",
        codec: str = "auto",
        cut_padding_seconds: float = 2.0,
        padding_segments: int | None = None,
    ) -> None:
        self.video_path = str(Path(input_video_path).expanduser().resolve())
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"No existe el video: {self.video_path}")
        if clip_duration <= 0:
            raise ValueError("clip_duration debe ser mayor que cero.")
        if num_procesos < 0:
            raise ValueError("num_procesos no puede ser negativo.")
        if cut_padding_seconds < 0:
            raise ValueError("cut_padding_seconds no puede ser negativo.")
        if padding_segments is not None and padding_segments < 0:
            raise ValueError("padding_segments no puede ser negativo.")

        self.umbral_minimo_expuesto = float(umbral_minimo_expuesto)
        self.umbral_minimo_cubierto = float(umbral_minimo_cubierto)
        self.clip_duration = float(clip_duration)
        self.requested_workers = int(num_procesos)
        self.device = device
        self.codec = codec
        # ``padding_segments`` is retained only for compatibility with older
        # callers. New code should express the margin directly in seconds.
        if padding_segments is not None:
            cut_padding_seconds = float(padding_segments) * self.clip_duration
        self.cut_padding_seconds = float(cut_padding_seconds)
        self.srt_generator = SrtGenerator()

        output_folder = Path(output_folder_path).expanduser()
        if not output_folder_path:
            output_folder = Path(self.video_path).parent
        output_folder.mkdir(parents=True, exist_ok=True)

        stem = Path(self.video_path).stem
        self.output_video_path = str(output_folder / f"{stem} (no_nsfw).mp4")
        self.output_srt_path = str(output_folder / f"{stem}.srt")
        self.active_device = "cpu"

    def _build_segments(self, duration: float) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        start_time = 0.0
        index = 1
        while start_time < duration:
            end_time = min(start_time + self.clip_duration, duration)
            segments.append(
                {
                    "orden": index,
                    "intervalo": [start_time, end_time],
                    "detecciones": None,
                    "nsfw": None,
                }
            )
            index += 1
            start_time = end_time
        return segments

    def _resolve_workers(self, number_of_segments: int, active_device: str) -> int:
        if number_of_segments <= 1:
            return 1
        if self.requested_workers > 0:
            return min(self.requested_workers, number_of_segments)
        if active_device == "cuda":
            # One ONNX session per process consumes GPU memory. A single worker is
            # the safest default across cards with very different VRAM sizes.
            return 1
        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count, number_of_segments))

    def _analyze_sequentially(
        self,
        segments: list[dict[str, Any]],
        detector: NsfwDetector,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with VideoFileClip(self.video_path) as video:
            bar = ChargingBar("Analizando segmentos", max=len(segments))
            try:
                for segment in segments:
                    start_time, _ = segment["intervalo"]
                    frame = np.ascontiguousarray(video.get_frame(start_time))
                    es_nsfw, detections, promedio_expuesto, promedio_cubierto = (
                        detector.is_nsfw(frame)
                    )
                    result = dict(segment)
                    result["detecciones"] = detections
                    result["nsfw"] = es_nsfw
                    result["promedio_expuesto"] = promedio_expuesto
                    result["promedio_cubierto"] = promedio_cubierto
                    results.append(result)
                    bar.next()
            finally:
                bar.finish()
        return results

    def _analyze_in_parallel(
        self,
        segments: list[dict[str, Any]],
        workers: int,
        active_device: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        spawn_context = mp.get_context("spawn")
        bar = ChargingBar("Analizando segmentos", max=len(segments))
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_context,
                initializer=_initialize_worker,
                initargs=(
                    self.video_path,
                    self.umbral_minimo_expuesto,
                    self.umbral_minimo_cubierto,
                    active_device,
                ),
            ) as executor:
                futures = [executor.submit(_analyze_segment, segment) for segment in segments]
                for future in as_completed(futures):
                    results.append(future.result())
                    bar.next()
        finally:
            bar.finish()
        return results

    def _build_cut_intervals(
        self,
        results: list[dict[str, Any]],
        duration: float,
        padding_seconds: float | None = None,
    ) -> list[tuple[float, float]]:
        """Return merged time ranges removed around prohibited detections.

        A frame is sampled at the beginning of every analysis segment. If that
        frame is prohibited at time ``t``, the exact interval
        ``[t - padding, t + padding]`` is removed. Overlapping or touching
        intervals are merged so MoviePy never cuts the same area twice.
        """

        padding = (
            self.cut_padding_seconds
            if padding_seconds is None
            else max(0.0, float(padding_seconds))
        )
        raw_intervals: list[tuple[float, float]] = []
        for result in results:
            if not result.get("nsfw"):
                continue
            detection_time = float(result["intervalo"][0])
            start_time = max(0.0, detection_time - padding)
            end_time = min(float(duration), detection_time + padding)
            if end_time > start_time:
                raw_intervals.append((start_time, end_time))

        if not raw_intervals:
            return []

        raw_intervals.sort(key=lambda interval: interval[0])
        merged: list[list[float]] = [list(raw_intervals[0])]
        epsilon = 1e-9
        for start_time, end_time in raw_intervals[1:]:
            previous = merged[-1]
            if start_time <= previous[1] + epsilon:
                previous[1] = max(previous[1], end_time)
            else:
                merged.append([start_time, end_time])
        return [(start_time, end_time) for start_time, end_time in merged]

    @staticmethod
    def _build_allowed_intervals(
        duration: float, cut_intervals: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Return the complement of the cut intervals inside the video."""

        allowed: list[tuple[float, float]] = []
        cursor = 0.0
        for cut_start, cut_end in cut_intervals:
            if cut_start > cursor:
                allowed.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < duration:
            allowed.append((cursor, duration))
        return allowed

    def mark_nsfw(
        self, results: list[dict[str, Any]], rango: int | None = None
    ) -> list[dict[str, Any]]:
        """Legacy helper retained for callers using segment-based padding."""

        radius = 0 if rango is None else max(0, int(rango))
        indices_to_mark: set[int] = set()
        for index, result in enumerate(results):
            if result.get("nsfw"):
                start = max(0, index - radius)
                end = min(len(results), index + radius + 1)
                indices_to_mark.update(range(start, end))
        for index in indices_to_mark:
            results[index]["nsfw"] = True
        return results

    def _codec_candidates(self) -> list[str]:
        requested = self.codec.strip().lower()
        if requested != "auto":
            if requested.endswith("_nvenc"):
                return [requested, "libx264"]
            return [requested]

        encoders = _ffmpeg_encoders()
        if _nvidia_driver_is_visible() and "h264_nvenc" in encoders:
            return ["h264_nvenc", "libx264"]
        return ["libx264"]

    def _write_video_with_fallback(self, final_clip) -> str:
        errors: list[str] = []
        for codec in self._codec_candidates():
            try:
                if os.path.exists(self.output_video_path):
                    os.remove(self.output_video_path)
                print(f"Codificando con {codec}...")
                final_clip.write_videofile(
                    self.output_video_path,
                    codec=codec,
                    audio_codec="aac",
                    threads=max(1, min(16, os.cpu_count() or 1)),
                    pixel_format="yuv420p",
                )
                return codec
            except Exception as exc:  # pragma: no cover - ffmpeg/hardware dependent
                errors.append(f"{codec}: {exc}")
                if os.path.exists(self.output_video_path):
                    try:
                        os.remove(self.output_video_path)
                    except OSError:
                        pass
                print(f"Falló {codec}; probando el siguiente codec disponible.")

        raise RuntimeError(
            "No se pudo generar el video. Errores de codificación:\n- "
            + "\n- ".join(errors)
        )

    def _render_results(
        self,
        results: list[dict[str, Any]],
        duration: float,
        cut_intervals: list[tuple[float, float]],
    ) -> None:
        allowed_intervals = self._build_allowed_intervals(duration, cut_intervals)
        for result in results:
            start_time, end_time = result["intervalo"]
            self.srt_generator.add_subtitle(
                start_time, end_time, result.get("detecciones") or []
            )
        self.srt_generator.generate_srt(self.output_srt_path)
        print(f"Archivo SRT guardado en {self.output_srt_path}")

        if cut_intervals:
            print("Intervalos eliminados por detecciones prohibidas:")
            for start_time, end_time in cut_intervals:
                print(f"  - {start_time:.3f}s a {end_time:.3f}s")
        else:
            print("No se detectaron clases prohibidas; se conservará el video completo.")

        if not allowed_intervals:
            print("Todo el video quedó dentro de los cortes; no se generó salida.")
            return

        source_video = VideoFileClip(self.video_path)
        clips = []
        final_clip = None
        bar = ChargingBar("Preparando clips permitidos", max=len(allowed_intervals))
        bar_was_finished = False
        try:
            for start_time, end_time in allowed_intervals:
                clips.append(_subclip(source_video, start_time, end_time))
                bar.next()
            bar.finish()
            bar_was_finished = True

            final_clip = concatenate_videoclips(clips, method="chain")
            used_codec = self._write_video_with_fallback(final_clip)
            print(
                f"Video guardado en {self.output_video_path} "
                f"(codec: {used_codec})"
            )
        finally:
            if not bar_was_finished:
                bar.finish()
            if final_clip is not None:
                final_clip.close()
            for clip in clips:
                clip.close()
            source_video.close()

    def process_video(self) -> list[dict[str, Any]]:
        with VideoFileClip(self.video_path) as video:
            duration = float(video.duration)
        segments = self._build_segments(duration)
        if not segments:
            raise RuntimeError("El video no contiene segmentos procesables.")

        detector = NsfwDetector(
            umbral_minimo_expuesto=self.umbral_minimo_expuesto,
            umbral_minimo_cubierto=self.umbral_minimo_cubierto,
            device=self.device,
        )
        self.active_device = detector.device
        print(f"ONNX Runtime: {detector.provider_summary()}")

        workers = self._resolve_workers(len(segments), detector.device)
        if detector.device == "cuda" and workers > 1:
            print(
                "Aviso: cada proceso carga una copia del modelo en la GPU. "
                "Reduce --workers si aparece un error de memoria."
            )
        print(f"Workers de análisis: {workers}")

        if workers == 1:
            results = self._analyze_sequentially(segments, detector)
        else:
            del detector
            gc.collect()
            results = self._analyze_in_parallel(
                segments, workers, active_device=self.active_device
            )

        results.sort(key=lambda item: item["orden"])
        cut_intervals = self._build_cut_intervals(results, duration)
        self._render_results(results, duration, cut_intervals)
        return results