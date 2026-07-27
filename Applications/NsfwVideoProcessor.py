from __future__ import annotations

import gc
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import VideoFileClip, concatenate_videoclips
from progress.bar import ChargingBar

from applications.NsfwDetector import NsfwDetector
from applications.SrtGenerator import SrtGenerator


_WORKER_DETECTOR: NsfwDetector | None = None
_FRAME_SENTINEL = object()


class _PipelineFailure:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _subclip(video: VideoFileClip, start_time: float, end_time: float):
    """MoviePy 2.x helper kept separate to simplify future API changes."""

    return video.subclipped(start_time, end_time)


def _initialize_detector_worker(
    umbral_minimo_expuesto: float,
    umbral_minimo_cubierto: float,
    device: str,
    intra_op_threads: int,
) -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = NsfwDetector(
        umbral_minimo_expuesto=umbral_minimo_expuesto,
        umbral_minimo_cubierto=umbral_minimo_cubierto,
        device=device,
        intra_op_threads=intra_op_threads,
    )


def _build_analysis_result(
    segment: dict[str, Any],
    assessment: tuple[bool, list[dict[str, Any]], float, float],
) -> dict[str, Any]:
    es_nsfw, detections, promedio_expuesto, promedio_cubierto = assessment
    result = dict(segment)
    result["detecciones"] = detections
    result["nsfw"] = es_nsfw
    result["promedio_expuesto"] = promedio_expuesto
    result["promedio_cubierto"] = promedio_cubierto
    return result


def _analyze_batch(
    batch: list[tuple[dict[str, Any], np.ndarray]],
) -> list[dict[str, Any]]:
    if _WORKER_DETECTOR is None:
        raise RuntimeError("El worker de inferencia no fue inicializado correctamente.")
    assessments = _WORKER_DETECTOR.analyze_batch(
        [frame for _, frame in batch],
        batch_size=len(batch),
    )
    if len(assessments) != len(batch):
        raise RuntimeError("NudeNet devolvió una cantidad inesperada de resultados.")
    return [
        _build_analysis_result(segment, assessment)
        for (segment, _frame), assessment in zip(batch, assessments)
    ]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _FfmpegFramePipeline:
    """Decode sampled frames once and feed inference through a bounded queue.

    FFmpeg performs sequential decoding in its own process while a Python thread
    drains stdout. The consumer can therefore execute ONNX inference at the same
    time without opening one decoder per worker or seeking independently for
    every segment.
    """

    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        clip_duration: float,
        segments: list[dict[str, Any]],
        prefetch_frames: int,
        ffmpeg_threads: int,
    ) -> None:
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.clip_duration = float(clip_duration)
        self.segments = segments
        self.prefetch_frames = max(1, int(prefetch_frames))
        self.ffmpeg_threads = max(1, int(ffmpeg_threads))
        self._queue: queue.Queue[
            tuple[dict[str, Any], np.ndarray] | _PipelineFailure | object
        ] = queue.Queue(maxsize=self.prefetch_frames)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def _command(self) -> list[str]:
        sample_filter = (
            f"fps=fps=1/{self.clip_duration:.12g}:start_time=0:eof_action=pass"
        )
        return [
            get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            str(self.ffmpeg_threads),
            "-i",
            self.video_path,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            sample_filter,
            "-frames:v",
            str(len(self.segments)),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

    def _put(self, item: tuple[dict[str, Any], np.ndarray] | _PipelineFailure | object) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        frame_size = self.width * self.height * 3
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                bufsize=max(frame_size * 2, 1024 * 1024),
            )
            if self._process.stdout is None:
                raise RuntimeError("FFmpeg no expuso el flujo de frames.")

            for segment in self.segments:
                if self._stop_event.is_set():
                    return
                raw_frame = _read_exact(self._process.stdout, frame_size)
                if len(raw_frame) != frame_size:
                    return_code = self._process.wait(timeout=10)
                    stderr_file.seek(0)
                    details = stderr_file.read().decode("utf-8", errors="replace").strip()
                    message = (
                        "FFmpeg terminó antes de producir todos los frames "
                        f"(código {return_code})."
                    )
                    if details:
                        message += f" Detalle: {details}"
                    raise RuntimeError(message)

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                if not self._put((segment, frame)):
                    return

            return_code = self._process.wait(timeout=30)
            if return_code != 0 and not self._stop_event.is_set():
                stderr_file.seek(0)
                details = stderr_file.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg falló durante la decodificación (código {return_code}). "
                    f"{details}"
                )
        except BaseException as exc:
            self._put(_PipelineFailure(exc))
        finally:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            stderr_file.close()
            self._put(_FRAME_SENTINEL)

    def __enter__(self) -> "_FfmpegFramePipeline":
        self._thread = threading.Thread(
            target=self._run,
            name="ffmpeg-frame-producer",
            daemon=True,
        )
        self._thread.start()
        return self

    def __iter__(self) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
        while True:
            item = self._queue.get()
            if item is _FRAME_SENTINEL:
                break
            if isinstance(item, _PipelineFailure):
                raise RuntimeError("Falló la etapa de decodificación.") from item.error
            segment, frame = item
            yield segment, frame

    def close(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


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
        prefetch_frames: int = 0,
        batch_size: int = 0,
        fast_copy_when_unchanged: bool = True,
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
        if prefetch_frames < 0:
            raise ValueError("prefetch_frames no puede ser negativo.")
        if batch_size < 0:
            raise ValueError("batch_size no puede ser negativo.")

        self.umbral_minimo_expuesto = float(umbral_minimo_expuesto)
        self.umbral_minimo_cubierto = float(umbral_minimo_cubierto)
        self.clip_duration = float(clip_duration)
        self.requested_workers = int(num_procesos)
        self.requested_prefetch_frames = int(prefetch_frames)
        self.requested_batch_size = int(batch_size)
        self.device = device
        self.codec = codec
        self.fast_copy_when_unchanged = bool(fast_copy_when_unchanged)
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
            # The optimized GPU pipeline already overlaps CPU decoding and GPU
            # inference. More sessions usually consume VRAM without increasing
            # throughput, so one CUDA session remains the safe default.
            return 1
        cpu_count = os.cpu_count() or 1
        decode_threads = self._resolve_ffmpeg_threads("cpu")
        inference_threads = max(1, cpu_count - decode_threads)
        # Keep at least two logical threads per ONNX CPU session when possible.
        worker_capacity = max(1, inference_threads // 2)
        return max(1, min(4, worker_capacity, number_of_segments))

    def _resolve_batch_size(
        self,
        width: int,
        height: int,
        workers: int,
        active_device: str,
    ) -> int:
        if self.requested_batch_size > 0:
            return self.requested_batch_size
        frame_bytes = max(1, int(width) * int(height) * 3)
        # Local inference avoids multiprocessing serialization, so it can use a
        # larger raw-frame batch. IPC batches remain deliberately smaller.
        target_bytes = 128 * 1024 * 1024 if workers == 1 else 32 * 1024 * 1024
        memory_limited = max(1, target_bytes // frame_bytes)
        preferred = 4 if active_device == "cuda" or workers == 1 else 8
        return max(1, min(preferred, memory_limited))

    def _resolve_prefetch_frames(
        self,
        width: int,
        height: int,
        workers: int,
        batch_size: int,
    ) -> int:
        if self.requested_prefetch_frames > 0:
            return self.requested_prefetch_frames
        frame_bytes = max(1, int(width) * int(height) * 3)
        memory_limited = max(2, (256 * 1024 * 1024) // frame_bytes)
        desired = max(4, workers * batch_size * 2)
        return max(2, min(32, memory_limited, desired))

    @staticmethod
    def _resolve_ffmpeg_threads(active_device: str) -> int:
        cpu_count = os.cpu_count() or 1
        if active_device == "cuda":
            return max(1, min(8, cpu_count // 2 or 1))
        return max(1, min(4, cpu_count // 4 or 1))

    @staticmethod
    def _worker_thread_budget(
        workers: int,
        active_device: str,
        ffmpeg_threads: int = 1,
    ) -> int:
        if active_device == "cuda":
            return 1
        cpu_count = os.cpu_count() or 1
        inference_threads = max(workers, cpu_count - max(1, ffmpeg_threads))
        return max(1, inference_threads // max(1, workers))

    def _analyze_sequential_pipeline(
        self,
        frame_pipeline: _FfmpegFramePipeline,
        detector: NsfwDetector,
        batch_size: int,
        number_of_segments: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        bar = ChargingBar("Pipeline decodificación/inferencia", max=number_of_segments)

        def analyze_local_batch(
            batch: list[tuple[dict[str, Any], np.ndarray]],
        ) -> None:
            assessments = detector.analyze_batch(
                [frame for _, frame in batch],
                batch_size=len(batch),
            )
            if len(assessments) != len(batch):
                raise RuntimeError(
                    "NudeNet devolvió una cantidad inesperada de resultados."
                )
            for (segment, _frame), assessment in zip(batch, assessments):
                results.append(_build_analysis_result(segment, assessment))
                bar.next()

        try:
            batch: list[tuple[dict[str, Any], np.ndarray]] = []
            for item in frame_pipeline:
                batch.append(item)
                if len(batch) >= batch_size:
                    analyze_local_batch(batch)
                    batch = []
            if batch:
                analyze_local_batch(batch)
        finally:
            bar.finish()
        return results

    def _analyze_parallel_pipeline(
        self,
        frame_pipeline: _FfmpegFramePipeline,
        workers: int,
        active_device: str,
        batch_size: int,
        number_of_segments: int,
        ffmpeg_threads: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        pending: set[Future[list[dict[str, Any]]]] = set()
        max_pending_batches = max(2, workers * 2)
        spawn_context = mp.get_context("spawn")
        worker_threads = self._worker_thread_budget(
            workers, active_device, ffmpeg_threads
        )
        bar = ChargingBar("Pipeline decodificación/inferencia", max=number_of_segments)

        def collect_completed(block: bool) -> None:
            if not pending:
                return
            if block:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            else:
                completed = {future for future in pending if future.done()}
            for future in completed:
                pending.remove(future)
                batch_results = future.result()
                results.extend(batch_results)
                for _ in batch_results:
                    bar.next()

        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_context,
                initializer=_initialize_detector_worker,
                initargs=(
                    self.umbral_minimo_expuesto,
                    self.umbral_minimo_cubierto,
                    active_device,
                    worker_threads,
                ),
            ) as executor:
                batch: list[tuple[dict[str, Any], np.ndarray]] = []
                for item in frame_pipeline:
                    batch.append(item)
                    if len(batch) < batch_size:
                        continue
                    pending.add(executor.submit(_analyze_batch, batch))
                    batch = []
                    collect_completed(block=False)
                    while len(pending) >= max_pending_batches:
                        collect_completed(block=True)

                if batch:
                    pending.add(executor.submit(_analyze_batch, batch))
                while pending:
                    collect_completed(block=True)
        finally:
            bar.finish()
        return results

    def _analyze_with_pipeline(
        self,
        segments: list[dict[str, Any]],
        detector: NsfwDetector | None,
        width: int,
        height: int,
        workers: int,
        active_device: str,
    ) -> list[dict[str, Any]]:
        batch_size = self._resolve_batch_size(
            width, height, workers, active_device
        )
        prefetch_frames = self._resolve_prefetch_frames(
            width, height, workers, batch_size
        )
        ffmpeg_threads = self._resolve_ffmpeg_threads(active_device)
        print(
            "Pipeline: "
            f"prefetch={prefetch_frames} frames; lote={batch_size}; "
            f"threads FFmpeg={ffmpeg_threads}"
        )

        with _FfmpegFramePipeline(
            video_path=self.video_path,
            width=width,
            height=height,
            clip_duration=self.clip_duration,
            segments=segments,
            prefetch_frames=prefetch_frames,
            ffmpeg_threads=ffmpeg_threads,
        ) as frame_pipeline:
            if workers == 1:
                if detector is None:
                    raise RuntimeError("No existe una sesión local de inferencia.")
                return self._analyze_sequential_pipeline(
                    frame_pipeline,
                    detector,
                    batch_size=batch_size,
                    number_of_segments=len(segments),
                )
            return self._analyze_parallel_pipeline(
                frame_pipeline,
                workers=workers,
                active_device=active_device,
                batch_size=batch_size,
                number_of_segments=len(segments),
                ffmpeg_threads=ffmpeg_threads,
            )

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

    def _try_fast_copy(self) -> bool:
        if not self.fast_copy_when_unchanged:
            return False
        if os.path.exists(self.output_video_path):
            os.remove(self.output_video_path)
        command = [
            get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            self.video_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            self.output_video_path,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode == 0 and os.path.isfile(self.output_video_path):
            return True
        if os.path.exists(self.output_video_path):
            os.remove(self.output_video_path)
        return False

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
            if self._try_fast_copy():
                print(
                    f"Video guardado en {self.output_video_path} "
                    "sin recodificación."
                )
                return
            print("No fue posible copiar los streams; se recodificará el video.")

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
            width, height = (int(video.size[0]), int(video.size[1]))
        segments = self._build_segments(duration)
        if not segments:
            raise RuntimeError("El video no contiene segmentos procesables.")

        requested_device = self.device.strip().lower()
        local_detector: NsfwDetector | None = None
        if requested_device == "cpu":
            self.active_device = "cpu"
            workers = self._resolve_workers(len(segments), self.active_device)
            if workers == 1:
                cpu_count = os.cpu_count() or 1
                decode_threads = self._resolve_ffmpeg_threads("cpu")
                local_detector = NsfwDetector(
                    umbral_minimo_expuesto=self.umbral_minimo_expuesto,
                    umbral_minimo_cubierto=self.umbral_minimo_cubierto,
                    device="cpu",
                    intra_op_threads=max(1, cpu_count - decode_threads),
                )
                print(f"ONNX Runtime: {local_detector.provider_summary()}")
            else:
                print(
                    "ONNX Runtime: dispositivo=cpu; las sesiones se inicializarán "
                    "directamente dentro de los workers."
                )
        else:
            detector = NsfwDetector(
                umbral_minimo_expuesto=self.umbral_minimo_expuesto,
                umbral_minimo_cubierto=self.umbral_minimo_cubierto,
                device=self.device,
            )
            self.active_device = detector.device
            print(f"ONNX Runtime: {detector.provider_summary()}")
            workers = self._resolve_workers(len(segments), detector.device)
            local_detector = detector
            if workers > 1:
                del detector
                local_detector = None
                gc.collect()

        if self.active_device == "cuda" and workers > 1:
            print(
                "Aviso: cada worker carga una copia del modelo en la GPU. "
                "La canalización con --workers 1 suele ser más eficiente."
            )
        print(f"Workers de inferencia: {workers}")

        started_at = time.perf_counter()
        results = self._analyze_with_pipeline(
            segments,
            detector=local_detector,
            width=width,
            height=height,
            workers=workers,
            active_device=self.active_device,
        )
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        print(
            f"Análisis completado: {len(results)} frames en {elapsed:.2f}s "
            f"({len(results) / elapsed:.2f} frames/s)."
        )

        results.sort(key=lambda item: item["orden"])
        cut_intervals = self._build_cut_intervals(results, duration)
        self._render_results(results, duration, cut_intervals)
        return results
