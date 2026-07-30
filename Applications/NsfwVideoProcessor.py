from __future__ import annotations

import gc
import multiprocessing as mp
import os
import queue
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import VideoFileClip
from progress.bar import ChargingBar

from applications.constants import (
    DEFAULT_CLIP_DURATION,
    DEFAULT_COVERED_THRESHOLD,
    DEFAULT_CUT_PADDING_SECONDS,
    DEFAULT_EXPOSED_THRESHOLD,
    DEFAULT_NSFW_THRESHOLD,
)
from applications.detectors import (
    ContentDetector,
    DetectionAssessment,
    DetectorConfig,
    create_detector,
)
from applications.reporting import AnalysisReportWriter
from applications.SrtGenerator import SrtGenerator
from applications.video_policies import CutIntervalPolicy, SegmentPlanner
from applications.video_renderer import VideoRenderer


_WORKER_DETECTOR: ContentDetector | None = None
_FRAME_SENTINEL = object()
DetectorBuilder = Callable[[DetectorConfig], ContentDetector]


class _PipelineFailure:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _initialize_detector_worker(config: DetectorConfig) -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = create_detector(config)


def _coerce_assessment(value: Any) -> DetectionAssessment:
    if isinstance(value, DetectionAssessment):
        return value
    try:
        is_nsfw, detections, exposed, covered = value
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "El detector debe devolver DetectionAssessment o el tuple legado de 4 valores."
        ) from exc
    return DetectionAssessment(
        is_nsfw=bool(is_nsfw),
        score=max(float(exposed), float(covered)),
        detections=tuple(dict(item) for item in detections),
        metrics={"exposed": float(exposed), "covered": float(covered)},
        model_name="legacy",
    )


def _build_analysis_result(
    segment: dict[str, Any],
    assessment_value: Any,
) -> dict[str, Any]:
    assessment = _coerce_assessment(assessment_value)
    result = dict(segment)
    result["detecciones"] = [dict(item) for item in assessment.detections]
    result["nsfw"] = assessment.is_nsfw
    result["score_nsfw"] = assessment.score
    result["motivo"] = assessment.reason
    result["metricas"] = dict(assessment.metrics)
    result["modelo"] = assessment.model_name
    # Backward-compatible diagnostic fields.
    result["promedio_expuesto"] = assessment.exposed_score
    result["promedio_cubierto"] = assessment.covered_score
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
        raise RuntimeError("El detector devolvió una cantidad inesperada de resultados.")
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
    """Decode sampled frames once and feed inference through a bounded queue."""

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
        sample_filter = f"fps=fps=1/{self.clip_duration:.12g}:start_time=0:eof_action=pass"
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

    def _put(
        self,
        item: tuple[dict[str, Any], np.ndarray] | _PipelineFailure | object,
    ) -> bool:
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
                    f"FFmpeg falló durante la decodificación (código {return_code}). {details}"
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
            if process is not None and process.stdout is not None:
                process.stdout.close()
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


class NsfwVideoProcessor:
    """Coordinates analysis without depending on a concrete ML model."""

    def __init__(
        self,
        input_video_path: str,
        umbral_minimo_expuesto: float = DEFAULT_EXPOSED_THRESHOLD,
        umbral_minimo_cubierto: float = DEFAULT_COVERED_THRESHOLD,
        output_folder_path: str = "",
        clip_duration: float = DEFAULT_CLIP_DURATION,
        num_procesos: int = 0,
        device: str = "auto",
        codec: str = "auto",
        cut_padding_seconds: float = DEFAULT_CUT_PADDING_SECONDS,
        padding_segments: int | None = None,
        prefetch_frames: int = 0,
        batch_size: int = 0,
        fast_copy_when_unchanged: bool = True,
        detector_backend: str = "nudenet",
        model_id: str = "Falconsai/nsfw_image_detection",
        nsfw_threshold: float = DEFAULT_NSFW_THRESHOLD,
        nudenet_aggregation: str = "max",
        analyze_only: bool = False,
        detector_config: DetectorConfig | None = None,
        detector: ContentDetector | None = None,
        detector_factory: DetectorBuilder = create_detector,
        segment_planner: SegmentPlanner | None = None,
        cut_policy: CutIntervalPolicy | None = None,
        report_writer: AnalysisReportWriter | None = None,
        renderer: VideoRenderer | None = None,
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

        self.clip_duration = float(clip_duration)
        if padding_segments is not None:
            cut_padding_seconds = float(padding_segments) * self.clip_duration
        self.cut_padding_seconds = float(cut_padding_seconds)
        self.requested_workers = int(num_procesos)
        self.requested_prefetch_frames = int(prefetch_frames)
        self.requested_batch_size = int(batch_size)
        self.codec = codec
        self.fast_copy_when_unchanged = bool(fast_copy_when_unchanged)
        self.analyze_only = bool(analyze_only)

        self.detector_config = detector_config or DetectorConfig(
            backend=detector_backend,
            device=device,
            model_id=model_id,
            nsfw_threshold=nsfw_threshold,
            exposed_threshold=umbral_minimo_expuesto,
            covered_threshold=umbral_minimo_cubierto,
            nudenet_aggregation=nudenet_aggregation,
        )
        self.device = self.detector_config.device
        self.umbral_minimo_expuesto = self.detector_config.exposed_threshold
        self.umbral_minimo_cubierto = self.detector_config.covered_threshold
        self._injected_detector = detector
        self._detector_factory = detector_factory
        self._custom_factory = detector_factory is not create_detector
        self._external_backend = self.detector_config.backend not in {
            "nudenet",
            "huggingface",
        }

        self.segment_planner = segment_planner or SegmentPlanner(self.clip_duration)
        self.cut_policy = cut_policy or CutIntervalPolicy(self.cut_padding_seconds)
        self.report_writer = report_writer or AnalysisReportWriter(SrtGenerator())

        output_folder = Path(output_folder_path).expanduser()
        if not output_folder_path:
            output_folder = Path(self.video_path).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = Path(self.video_path).stem
        self.output_video_path = str(output_folder / f"{stem} (no_nsfw).mp4")
        self.output_srt_path = str(output_folder / f"{stem}.srt")
        self.output_report_path = str(output_folder / f"{stem}.analysis.json")
        self.renderer = renderer or VideoRenderer(
            input_path=self.video_path,
            output_path=self.output_video_path,
            codec=self.codec,
            fast_copy_when_unchanged=self.fast_copy_when_unchanged,
        )
        self.active_device = "cpu"

    def _build_segments(self, duration: float) -> list[dict[str, Any]]:
        return self.segment_planner.build(duration)

    def _resolve_workers(self, number_of_segments: int, active_device: str) -> int:
        if (
            number_of_segments <= 1
            or self._injected_detector is not None
            or self._custom_factory
            or self._external_backend
        ):
            return 1
        if self.requested_workers > 0:
            return min(self.requested_workers, number_of_segments)
        if active_device == "cuda":
            return 1
        cpu_count = os.cpu_count() or 1
        decode_threads = self._resolve_ffmpeg_threads("cpu")
        inference_threads = max(1, cpu_count - decode_threads)
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
        detector: ContentDetector,
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
                raise RuntimeError("El detector devolvió una cantidad inesperada de resultados.")
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
        worker_threads = self._worker_thread_budget(workers, active_device, ffmpeg_threads)
        worker_config = self.detector_config.with_runtime(
            device=active_device,
            intra_op_threads=worker_threads,
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
                initargs=(worker_config,),
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
        detector: ContentDetector | None,
        width: int,
        height: int,
        workers: int,
        active_device: str,
    ) -> list[dict[str, Any]]:
        batch_size = self._resolve_batch_size(width, height, workers, active_device)
        prefetch_frames = self._resolve_prefetch_frames(width, height, workers, batch_size)
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
        return self.cut_policy.build_cut_intervals(results, duration, padding_seconds)

    @staticmethod
    def _build_allowed_intervals(
        duration: float,
        cut_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return CutIntervalPolicy.build_allowed_intervals(duration, cut_intervals)

    def mark_nsfw(
        self,
        results: list[dict[str, Any]],
        rango: int | None = None,
    ) -> list[dict[str, Any]]:
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
        return self.renderer.codec_candidates()

    def _render_results(
        self,
        results: list[dict[str, Any]],
        duration: float,
        cut_intervals: list[tuple[float, float]],
        detector_name: str,
    ) -> None:
        self.report_writer.write(
            video_path=self.video_path,
            duration=duration,
            detector_name=detector_name,
            results=results,
            cut_intervals=cut_intervals,
            srt_path=self.output_srt_path,
            json_path=self.output_report_path,
        )
        print(f"Archivo SRT guardado en {self.output_srt_path}")
        print(f"Informe JSON guardado en {self.output_report_path}")

        if cut_intervals:
            print("Intervalos eliminados por detecciones prohibidas:")
            for start_time, end_time in cut_intervals:
                print(f"  - {start_time:.3f}s a {end_time:.3f}s")

        if self.analyze_only:
            self.renderer.remove_stale_output()
            print("Modo análisis: no se generó un video de salida.")
            return

        allowed_intervals = self._build_allowed_intervals(duration, cut_intervals)
        self.renderer.render(
            allowed_intervals=allowed_intervals,
            cut_intervals=cut_intervals,
        )

    def _create_detector(self, config: DetectorConfig) -> ContentDetector:
        return self._detector_factory(config)

    def process_video(self) -> list[dict[str, Any]]:
        # Never leave an old output that could be mistaken for the current run.
        self.renderer.remove_stale_output()
        with VideoFileClip(self.video_path) as video:
            duration = float(video.duration)
            width, height = int(video.size[0]), int(video.size[1])
        segments = self._build_segments(duration)
        if not segments:
            raise RuntimeError("El video no contiene segmentos procesables.")

        local_detector: ContentDetector | None = self._injected_detector
        if local_detector is not None:
            self.active_device = local_detector.device
            workers = 1
            print(f"Detector inyectado: {local_detector.provider_summary()}")
        elif self.detector_config.device == "cpu":
            self.active_device = "cpu"
            workers = self._resolve_workers(len(segments), self.active_device)
            if workers == 1:
                cpu_count = os.cpu_count() or 1
                decode_threads = self._resolve_ffmpeg_threads("cpu")
                local_config = self.detector_config.with_runtime(
                    device="cpu",
                    intra_op_threads=max(1, cpu_count - decode_threads),
                )
                local_detector = self._create_detector(local_config)
                print(f"Detector: {local_detector.provider_summary()}")
            else:
                print(
                    f"Detector={self.detector_config.backend}; dispositivo=cpu; "
                    "las sesiones se inicializarán dentro de los workers."
                )
        else:
            local_detector = self._create_detector(self.detector_config)
            self.active_device = local_detector.device
            print(f"Detector: {local_detector.provider_summary()}")
            workers = self._resolve_workers(len(segments), self.active_device)
            if workers > 1:
                del local_detector
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
        detector_name = (
            local_detector.name
            if local_detector is not None
            else self.detector_config.backend
        )
        self._render_results(results, duration, cut_intervals, detector_name)
        return results
