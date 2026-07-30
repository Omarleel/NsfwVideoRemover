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
from progress.bar import ChargingBar

from applications.constants import (
    DEFAULT_ANALYSIS_MAX_DIMENSION,
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
from applications.profiling import PerformanceProfiler, current_rss_bytes
from applications.reporting import AnalysisReportWriter
from applications.SrtGenerator import SrtGenerator
from applications.video_policies import CutIntervalPolicy, SegmentPlanner
from applications.video_renderer import VideoRenderer
from applications.video_probe import ImageioFfmpegVideoProbe, VideoProbe


_WORKER_DETECTOR: ContentDetector | None = None
_WORKER_PROFILING_ENABLED = True
_WORKER_INIT_METRICS: dict[str, Any] | None = None
_WORKER_INIT_REPORTED = False
_FRAME_SENTINEL = object()
DetectorBuilder = Callable[[DetectorConfig], ContentDetector]


class _PipelineFailure:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _initialize_detector_worker(
    config: DetectorConfig, profiling_enabled: bool = True
) -> None:
    global _WORKER_DETECTOR, _WORKER_PROFILING_ENABLED
    global _WORKER_INIT_METRICS, _WORKER_INIT_REPORTED
    _WORKER_PROFILING_ENABLED = bool(profiling_enabled)
    rss_before = current_rss_bytes()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    _WORKER_DETECTOR = create_detector(config)
    _WORKER_INIT_METRICS = {
        "category": "inference_worker",
        "name": "initialize_detector_worker",
        "duration_seconds": time.perf_counter() - wall_started,
        "cpu_seconds": time.process_time() - cpu_started,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "rss_bytes": current_rss_bytes(),
        "details": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": current_rss_bytes(),
            "backend": config.backend,
            "device": config.device,
            "intra_op_threads": config.intra_op_threads,
        },
    }
    _WORKER_INIT_REPORTED = False


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
) -> dict[str, Any]:
    global _WORKER_INIT_REPORTED
    if _WORKER_DETECTOR is None:
        raise RuntimeError("El worker de inferencia no fue inicializado correctamente.")
    pid = os.getpid()
    batch_orders = [item[0].get("orden") for item in batch]
    prepare_started = time.perf_counter()
    prepare_cpu = time.process_time()
    images = [frame for _, frame in batch]
    prepare_duration = time.perf_counter() - prepare_started
    prepare_cpu_duration = time.process_time() - prepare_cpu

    inference_started = time.perf_counter()
    inference_cpu = time.process_time()
    assessments = _WORKER_DETECTOR.analyze_batch(images, batch_size=len(batch))
    inference_duration = time.perf_counter() - inference_started
    inference_cpu_duration = time.process_time() - inference_cpu
    if len(assessments) != len(batch):
        raise RuntimeError("El detector devolvió una cantidad inesperada de resultados.")

    if not _WORKER_PROFILING_ENABLED:
        return {
            "results": [
                _build_analysis_result(segment, assessment)
                for (segment, _frame), assessment in zip(batch, assessments)
            ],
            "events": [],
            "worker_pid": pid,
        }

    results: list[dict[str, Any]] = []
    atomic_events: list[dict[str, Any]] = []
    if _WORKER_INIT_METRICS is not None and not _WORKER_INIT_REPORTED:
        atomic_events.append(dict(_WORKER_INIT_METRICS))
        _WORKER_INIT_REPORTED = True
    atomic_events.extend([
        {
            "category": "inference_worker",
            "name": "prepare_batch_images",
            "duration_seconds": prepare_duration,
            "cpu_seconds": prepare_cpu_duration,
            "pid": pid,
            "thread_id": threading.get_ident(),
            "rss_bytes": current_rss_bytes(),
            "details": {"batch_size": len(batch), "segment_orders": batch_orders},
        },
        {
            "category": "inference_worker",
            "name": "detector_batch",
            "duration_seconds": inference_duration,
            "cpu_seconds": inference_cpu_duration,
            "pid": pid,
            "thread_id": threading.get_ident(),
            "rss_bytes": current_rss_bytes(),
            "details": {
                "batch_size": len(batch),
                "segment_orders": batch_orders,
                "seconds_per_frame": inference_duration / max(1, len(batch)),
            },
        },
    ])
    build_total = 0.0
    build_cpu_total = 0.0
    for (segment, _frame), assessment in zip(batch, assessments):
        build_started = time.perf_counter()
        build_cpu = time.process_time()
        results.append(_build_analysis_result(segment, assessment))
        duration = time.perf_counter() - build_started
        cpu_duration = time.process_time() - build_cpu
        build_total += duration
        build_cpu_total += cpu_duration
        atomic_events.append(
            {
                "category": "inference_worker",
                "name": "build_frame_result",
                "duration_seconds": duration,
                "cpu_seconds": cpu_duration,
                "pid": pid,
                "thread_id": threading.get_ident(),
                "rss_bytes": current_rss_bytes(),
                "details": {"segment_order": segment.get("orden")},
            }
        )
    atomic_events.append(
        {
            "category": "inference_worker",
            "name": "build_batch_results",
            "duration_seconds": build_total,
            "cpu_seconds": build_cpu_total,
            "pid": pid,
            "thread_id": threading.get_ident(),
            "rss_bytes": current_rss_bytes(),
            "details": {"batch_size": len(batch)},
        }
    )
    return {"results": results, "events": atomic_events, "worker_pid": pid}


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
        resize_frames: bool = False,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.clip_duration = float(clip_duration)
        self.segments = segments
        self.prefetch_frames = max(1, int(prefetch_frames))
        self.ffmpeg_threads = max(1, int(ffmpeg_threads))
        self.resize_frames = bool(resize_frames)
        self.profiler = profiler
        self._queue: queue.Queue[
            tuple[dict[str, Any], np.ndarray] | _PipelineFailure | object
        ] = queue.Queue(maxsize=self.prefetch_frames)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def _command(self) -> list[str]:
        sample_filter = (
            "setpts=PTS-STARTPTS,"
            f"select=eq(n\\,0)+gte(t\\,selected_n*{self.clip_duration:.12g})"
        )
        if self.resize_frames:
            sample_filter += (
                f",scale={self.width}:{self.height}:flags=fast_bilinear"
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
            "-fps_mode",
            "passthrough",
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
            command = self._command()
            launch_started = time.perf_counter()
            launch_cpu = time.process_time()
            launch_offset = (
                self.profiler.now_offset_seconds() if self.profiler is not None else None
            )
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                bufsize=max(frame_size * 2, 1024 * 1024),
            )
            if self.profiler is not None:
                self.profiler.event(
                    "decode",
                    "ffmpeg_process_launch",
                    duration_seconds=time.perf_counter() - launch_started,
                    cpu_seconds=time.process_time() - launch_cpu,
                    start_offset_seconds=launch_offset,
                    command=command,
                    frame_size_bytes=frame_size,
                    expected_frames=len(self.segments),
                )
            if self._process.stdout is None:
                raise RuntimeError("FFmpeg no expuso el flujo de frames.")

            for segment in self.segments:
                if self._stop_event.is_set():
                    return
                segment_order = segment.get("orden")
                read_started = time.perf_counter()
                read_cpu = time.process_time()
                read_offset = (
                    self.profiler.now_offset_seconds() if self.profiler is not None else None
                )
                raw_frame = _read_exact(self._process.stdout, frame_size)
                if self.profiler is not None:
                    self.profiler.event(
                        "decode",
                        "frame_read",
                        duration_seconds=time.perf_counter() - read_started,
                        cpu_seconds=time.process_time() - read_cpu,
                        start_offset_seconds=read_offset,
                        segment_order=segment_order,
                        bytes_read=len(raw_frame),
                        expected_bytes=frame_size,
                    )
                    self.profiler.increment("decoded_frame_bytes", len(raw_frame))
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

                convert_started = time.perf_counter()
                convert_cpu = time.process_time()
                convert_offset = (
                    self.profiler.now_offset_seconds() if self.profiler is not None else None
                )
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )
                if self.profiler is not None:
                    self.profiler.event(
                        "decode",
                        "frame_numpy_view",
                        duration_seconds=time.perf_counter() - convert_started,
                        cpu_seconds=time.process_time() - convert_cpu,
                        start_offset_seconds=convert_offset,
                        segment_order=segment_order,
                        shape=[self.height, self.width, 3],
                    )
                put_started = time.perf_counter()
                put_cpu = time.process_time()
                put_offset = (
                    self.profiler.now_offset_seconds() if self.profiler is not None else None
                )
                accepted = self._put((segment, frame))
                if self.profiler is not None:
                    self.profiler.event(
                        "queue",
                        "producer_put_wait",
                        duration_seconds=time.perf_counter() - put_started,
                        cpu_seconds=time.process_time() - put_cpu,
                        start_offset_seconds=put_offset,
                        segment_order=segment_order,
                        accepted=accepted,
                        queue_size=self._queue.qsize(),
                        queue_capacity=self.prefetch_frames,
                    )
                if not accepted:
                    return

            wait_started = time.perf_counter()
            wait_cpu = time.process_time()
            wait_offset = (
                self.profiler.now_offset_seconds() if self.profiler is not None else None
            )
            return_code = self._process.wait(timeout=30)
            if self.profiler is not None:
                self.profiler.event(
                    "decode",
                    "ffmpeg_process_wait",
                    duration_seconds=time.perf_counter() - wait_started,
                    cpu_seconds=time.process_time() - wait_cpu,
                    start_offset_seconds=wait_offset,
                    return_code=return_code,
                )
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
            get_started = time.perf_counter()
            get_cpu = time.process_time()
            get_offset = (
                self.profiler.now_offset_seconds() if self.profiler is not None else None
            )
            item = self._queue.get()
            if self.profiler is not None:
                self.profiler.event(
                    "queue",
                    "consumer_get_wait",
                    duration_seconds=time.perf_counter() - get_started,
                    cpu_seconds=time.process_time() - get_cpu,
                    start_offset_seconds=get_offset,
                    queue_size_after_get=self._queue.qsize(),
                    sentinel=item is _FRAME_SENTINEL,
                )
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
        analysis_max_dimension: int = DEFAULT_ANALYSIS_MAX_DIMENSION,
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
        profile_enabled: bool = True,
        profile_output_path: str = "",
        detector_config: DetectorConfig | None = None,
        detector: ContentDetector | None = None,
        detector_factory: DetectorBuilder = create_detector,
        segment_planner: SegmentPlanner | None = None,
        cut_policy: CutIntervalPolicy | None = None,
        report_writer: AnalysisReportWriter | None = None,
        renderer: VideoRenderer | None = None,
        video_probe: VideoProbe | None = None,
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.video_path = str(Path(input_video_path).expanduser().resolve())
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"No existe el video: {self.video_path}")
        if clip_duration <= 0:
            raise ValueError("clip_duration debe ser mayor que cero.")
        if analysis_max_dimension < 0:
            raise ValueError("analysis_max_dimension no puede ser negativo.")
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
        self.analysis_max_dimension = int(analysis_max_dimension)
        if padding_segments is not None:
            cut_padding_seconds = float(padding_segments) * self.clip_duration
        self.cut_padding_seconds = float(cut_padding_seconds)
        self.requested_workers = int(num_procesos)
        self.requested_prefetch_frames = int(prefetch_frames)
        self.requested_batch_size = int(batch_size)
        self.codec = codec
        self.fast_copy_when_unchanged = bool(fast_copy_when_unchanged)
        self.analyze_only = bool(analyze_only)
        self.profile_enabled = bool(profile_enabled)

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
        self.report_writer = report_writer

        output_folder = Path(output_folder_path).expanduser()
        if not output_folder_path:
            output_folder = Path(self.video_path).parent
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = Path(self.video_path).stem
        self.output_video_path = str(output_folder / f"{stem} (no_nsfw).mp4")
        self.output_srt_path = str(output_folder / f"{stem}.srt")
        self.output_report_path = str(output_folder / f"{stem}.analysis.json")
        self.output_profile_path = str(
            Path(profile_output_path).expanduser().resolve()
            if profile_output_path
            else output_folder / f"{stem}.profile.json"
        )
        self.profiler = profiler or PerformanceProfiler(
            self.output_profile_path,
            enabled=self.profile_enabled,
            input_path=self.video_path,
        )
        self.report_writer = self.report_writer or AnalysisReportWriter(
            SrtGenerator(), profiler=self.profiler
        )
        if report_writer is not None and hasattr(self.report_writer, "profiler"):
            self.report_writer.profiler = self.profiler
        self.video_probe = video_probe or ImageioFfmpegVideoProbe()
        self.renderer = renderer or VideoRenderer(
            input_path=self.video_path,
            output_path=self.output_video_path,
            codec=self.codec,
            fast_copy_when_unchanged=self.fast_copy_when_unchanged,
            profiler=self.profiler,
        )
        if renderer is not None and hasattr(self.renderer, "profiler"):
            self.renderer.profiler = self.profiler
        self.active_device = "cpu"
        self.profiler.configure(
            clip_duration_seconds=self.clip_duration,
            analysis_max_dimension=self.analysis_max_dimension,
            cut_padding_seconds=self.cut_padding_seconds,
            requested_workers=self.requested_workers,
            requested_prefetch_frames=self.requested_prefetch_frames,
            requested_batch_size=self.requested_batch_size,
            requested_device=self.device,
            requested_codec=self.codec,
            detector_backend=self.detector_config.backend,
            model_id=self.detector_config.model_id,
            nudenet_aggregation=self.detector_config.nudenet_aggregation,
            analyze_only=self.analyze_only,
        )
        self.profiler.artifact("analysis_json", self.output_report_path)
        self.profiler.artifact("srt", self.output_srt_path)
        self.profiler.artifact("output_video", self.output_video_path)
        self.profiler.artifact("profile_json", self.output_profile_path)

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

    def _resolve_analysis_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Bound decoded frame size before Python/IPC without changing aspect ratio."""

        width = max(1, int(width))
        height = max(1, int(height))
        limit = self.analysis_max_dimension
        largest = max(width, height)
        if limit <= 0 or largest <= limit:
            return width, height
        scale = limit / largest
        return max(1, round(width * scale)), max(1, round(height * scale))

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

        batch_index = 0

        def analyze_local_batch(
            batch: list[tuple[dict[str, Any], np.ndarray]],
        ) -> None:
            nonlocal batch_index
            batch_index += 1
            segment_orders = [segment.get("orden") for segment, _frame in batch]
            prepare_started = time.perf_counter()
            prepare_cpu = time.process_time()
            prepare_offset = self.profiler.now_offset_seconds()
            images = [frame for _, frame in batch]
            self.profiler.event(
                "inference",
                "prepare_batch_images",
                duration_seconds=time.perf_counter() - prepare_started,
                cpu_seconds=time.process_time() - prepare_cpu,
                start_offset_seconds=prepare_offset,
                batch_index=batch_index,
                batch_size=len(batch),
                segment_orders=segment_orders,
            )

            inference_started = time.perf_counter()
            inference_cpu = time.process_time()
            inference_offset = self.profiler.now_offset_seconds()
            assessments = detector.analyze_batch(images, batch_size=len(batch))
            inference_duration = time.perf_counter() - inference_started
            self.profiler.event(
                "inference",
                "detector_batch",
                duration_seconds=inference_duration,
                cpu_seconds=time.process_time() - inference_cpu,
                start_offset_seconds=inference_offset,
                batch_index=batch_index,
                batch_size=len(batch),
                segment_orders=segment_orders,
                seconds_per_frame=inference_duration / max(1, len(batch)),
            )
            if len(assessments) != len(batch):
                raise RuntimeError("El detector devolvió una cantidad inesperada de resultados.")
            for (segment, _frame), assessment in zip(batch, assessments):
                build_started = time.perf_counter()
                build_cpu = time.process_time()
                build_offset = self.profiler.now_offset_seconds()
                results.append(_build_analysis_result(segment, assessment))
                self.profiler.event(
                    "inference",
                    "build_frame_result",
                    duration_seconds=time.perf_counter() - build_started,
                    cpu_seconds=time.process_time() - build_cpu,
                    start_offset_seconds=build_offset,
                    batch_index=batch_index,
                    segment_order=segment.get("orden"),
                )
                bar.next()
            self.profiler.increment("inference_batches")
            self.profiler.increment("analyzed_frames", len(batch))

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
        pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        max_pending_batches = max(2, workers * 2)
        spawn_context = mp.get_context("spawn")
        worker_threads = self._worker_thread_budget(workers, active_device, ffmpeg_threads)
        worker_config = self.detector_config.with_runtime(
            device=active_device,
            intra_op_threads=worker_threads,
        )
        bar = ChargingBar("Pipeline decodificación/inferencia", max=number_of_segments)
        batch_index = 0

        def submit_batch(
            executor: ProcessPoolExecutor,
            batch: list[tuple[dict[str, Any], np.ndarray]],
        ) -> None:
            nonlocal batch_index
            batch_index += 1
            started = time.perf_counter()
            cpu_started = time.process_time()
            offset = self.profiler.now_offset_seconds()
            future = executor.submit(_analyze_batch, batch)
            duration = time.perf_counter() - started
            orders = [segment.get("orden") for segment, _frame in batch]
            pending[future] = {
                "batch_index": batch_index,
                "submitted_at": time.perf_counter(),
                "segment_orders": orders,
                "batch_size": len(batch),
            }
            self.profiler.event(
                "process_pool",
                "submit_batch",
                duration_seconds=duration,
                cpu_seconds=time.process_time() - cpu_started,
                start_offset_seconds=offset,
                batch_index=batch_index,
                batch_size=len(batch),
                segment_orders=orders,
                pending_batches=len(pending),
            )

        def collect_completed(block: bool) -> None:
            if not pending:
                return
            futures = set(pending)
            wait_started = time.perf_counter()
            wait_cpu = time.process_time()
            wait_offset = self.profiler.now_offset_seconds()
            if block:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            else:
                completed = {future for future in futures if future.done()}
            self.profiler.event(
                "process_pool",
                "wait_for_completed_batch",
                duration_seconds=time.perf_counter() - wait_started,
                cpu_seconds=time.process_time() - wait_cpu,
                start_offset_seconds=wait_offset,
                blocking=block,
                completed_count=len(completed),
                pending_count_before=len(futures),
            )
            for future in completed:
                metadata = pending.pop(future)
                collect_started = time.perf_counter()
                collect_cpu = time.process_time()
                collect_offset = self.profiler.now_offset_seconds()
                payload = future.result()
                batch_results = list(payload["results"])
                results.extend(batch_results)
                for worker_event in payload.get("events", []):
                    self.profiler.ingest_worker_event(worker_event)
                self.profiler.event(
                    "process_pool",
                    "collect_batch_result",
                    duration_seconds=time.perf_counter() - collect_started,
                    cpu_seconds=time.process_time() - collect_cpu,
                    start_offset_seconds=collect_offset,
                    batch_index=metadata["batch_index"],
                    batch_size=metadata["batch_size"],
                    segment_orders=metadata["segment_orders"],
                    worker_pid=payload.get("worker_pid"),
                    future_latency_seconds=time.perf_counter() - metadata["submitted_at"],
                    pending_batches_after=len(pending),
                )
                self.profiler.increment("inference_batches")
                self.profiler.increment("analyzed_frames", len(batch_results))
                for _ in batch_results:
                    bar.next()

        try:
            pool_started = time.perf_counter()
            pool_cpu = time.process_time()
            pool_offset = self.profiler.now_offset_seconds()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_context,
                initializer=_initialize_detector_worker,
                initargs=(worker_config, self.profiler.enabled),
            ) as executor:
                self.profiler.event(
                    "process_pool",
                    "create_executor",
                    duration_seconds=time.perf_counter() - pool_started,
                    cpu_seconds=time.process_time() - pool_cpu,
                    start_offset_seconds=pool_offset,
                    workers=workers,
                    worker_threads=worker_threads,
                )
                batch: list[tuple[dict[str, Any], np.ndarray]] = []
                for item in frame_pipeline:
                    batch.append(item)
                    if len(batch) < batch_size:
                        continue
                    submit_batch(executor, batch)
                    batch = []
                    collect_completed(block=False)
                    while len(pending) >= max_pending_batches:
                        collect_completed(block=True)
                if batch:
                    submit_batch(executor, batch)
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
        resize_frames: bool,
    ) -> list[dict[str, Any]]:
        batch_size = self._resolve_batch_size(width, height, workers, active_device)
        prefetch_frames = self._resolve_prefetch_frames(width, height, workers, batch_size)
        ffmpeg_threads = self._resolve_ffmpeg_threads(active_device)
        self.profiler.configure(
            active_device=active_device,
            resolved_workers=workers,
            resolved_batch_size=batch_size,
            resolved_prefetch_frames=prefetch_frames,
            resolved_ffmpeg_decode_threads=ffmpeg_threads,
            analysis_width=width,
            analysis_height=height,
            analysis_frame_bytes=width * height * 3,
            analysis_resize_applied=resize_frames,
        )
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
            resize_frames=resize_frames,
            profiler=self.profiler,
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
        with self.profiler.span(
            "planning", "build_allowed_intervals", cut_interval_count=len(cut_intervals)
        ):
            allowed_intervals = self._build_allowed_intervals(duration, cut_intervals)
        expected_output_duration = sum(
            end_time - start_time for start_time, end_time in allowed_intervals
        )
        self.profiler.configure(
            cut_interval_count=len(cut_intervals),
            allowed_interval_count=len(allowed_intervals),
            expected_output_duration_seconds=expected_output_duration,
            removed_duration_seconds=max(0.0, duration - expected_output_duration),
        )
        report_arguments = {
            "video_path": self.video_path,
            "duration": duration,
            "detector_name": detector_name,
            "results": results,
            "cut_intervals": cut_intervals,
            "allowed_intervals": allowed_intervals,
            "srt_path": self.output_srt_path,
            "json_path": self.output_report_path,
            "expected_output_duration": expected_output_duration,
        }
        with self.profiler.span(
            "reporting", "write_initial_reports", segment_count=len(results)
        ):
            self.report_writer.write(**report_arguments)
        print(f"Archivo SRT guardado en {self.output_srt_path}")
        print(f"Informe JSON guardado en {self.output_report_path}")
        print(
            "Duración total calculada de los intervalos sanos: "
            f"{expected_output_duration:.3f}s."
        )

        if cut_intervals:
            print("Intervalos eliminados por detecciones prohibidas:")
            for start_time, end_time in cut_intervals:
                print(f"  - {start_time:.3f}s a {end_time:.3f}s")

        if self.analyze_only:
            with self.profiler.span("render", "remove_stale_output_analyze_only"):
                self.renderer.remove_stale_output()
            print("Modo análisis: no se generó un video de salida.")
            return

        with self.profiler.span(
            "render",
            "render_video",
            expected_output_duration_seconds=expected_output_duration,
            allowed_interval_count=len(allowed_intervals),
        ):
            render_result = self.renderer.render(
                allowed_intervals=allowed_intervals,
                cut_intervals=cut_intervals,
            )
        self.profiler.configure(
            render_generated=render_result.generated,
            render_codec=render_result.codec,
            actual_output_duration_seconds=render_result.actual_duration,
            rendered_interval_count=len(render_result.rendered_intervals),
        )
        if render_result.generated:
            try:
                self.profiler.artifact(
                    "output_video_size_bytes", Path(self.output_video_path).stat().st_size
                )
            except OSError:
                self.profiler.artifact("output_video_size_bytes", None)
            with self.profiler.span(
                "reporting", "write_final_analysis_report", segment_count=len(results)
            ):
                self.report_writer.write(
                    **report_arguments,
                    rendered_intervals=list(render_result.rendered_intervals),
                    render_mode=render_result.codec,
                    actual_output_duration=render_result.actual_duration,
                )
            print(
                "Informe JSON actualizado con los intervalos realmente "
                "remultiplexados."
            )

    def _create_detector(self, config: DetectorConfig) -> ContentDetector:
        return self._detector_factory(config)

    def process_video(self) -> list[dict[str, Any]]:
        status = "failed"
        results: list[dict[str, Any]] = []
        try:
            with self.profiler.span("startup", "remove_stale_output"):
                self.renderer.remove_stale_output()

            with self.profiler.span("probe", "probe_input_video"):
                video_info = self.video_probe.probe(self.video_path)
            duration = video_info.duration
            width, height = video_info.width, video_info.height
            self.profiler.configure(
                source_duration_seconds=duration,
                source_width=width,
                source_height=height,
                source_pixels=width * height,
            )

            with self.profiler.span("planning", "resolve_analysis_dimensions"):
                analysis_width, analysis_height = self._resolve_analysis_dimensions(
                    width, height
                )
            if (analysis_width, analysis_height) != (width, height):
                print(
                    "Resolución de análisis optimizada: "
                    f"{width}x{height} -> {analysis_width}x{analysis_height}."
                )

            with self.profiler.span("planning", "build_segments"):
                segments = self._build_segments(duration)
            if not segments:
                raise RuntimeError("El video no contiene segmentos procesables.")
            self.profiler.configure(segment_count=len(segments))

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
                    with self.profiler.span(
                        "detector", "initialize_local_detector", requested_device="cpu"
                    ):
                        local_detector = self._create_detector(local_config)
                    print(f"Detector: {local_detector.provider_summary()}")
                else:
                    print(
                        f"Detector={self.detector_config.backend}; dispositivo=cpu; "
                        "las sesiones se inicializarán dentro de los workers."
                    )
            else:
                with self.profiler.span(
                    "detector",
                    "initialize_local_detector",
                    requested_device=self.detector_config.device,
                ):
                    local_detector = self._create_detector(self.detector_config)
                self.active_device = local_detector.device
                print(f"Detector: {local_detector.provider_summary()}")
                workers = self._resolve_workers(len(segments), self.active_device)
                if workers > 1:
                    with self.profiler.span("memory", "release_local_detector"):
                        del local_detector
                        local_detector = None
                        gc.collect()

            if self.active_device == "cuda" and workers > 1:
                print(
                    "Aviso: cada worker carga una copia del modelo en la GPU. "
                    "La canalización con --workers 1 suele ser más eficiente."
                )
            print(f"Workers de inferencia: {workers}")
            self.profiler.configure(
                active_device=self.active_device,
                resolved_workers=workers,
                detector_provider_summary=(
                    local_detector.provider_summary()
                    if local_detector is not None
                    else "initialized_in_workers"
                ),
            )

            started_at = time.perf_counter()
            with self.profiler.span(
                "analysis",
                "decode_and_inference_pipeline",
                expected_frames=len(segments),
            ):
                results = self._analyze_with_pipeline(
                    segments,
                    detector=local_detector,
                    width=analysis_width,
                    height=analysis_height,
                    workers=workers,
                    active_device=self.active_device,
                    resize_frames=(analysis_width, analysis_height) != (width, height),
                )
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            frames_per_second = len(results) / elapsed
            print(
                f"Análisis completado: {len(results)} frames en {elapsed:.2f}s "
                f"({frames_per_second:.2f} frames/s)."
            )
            self.profiler.configure(
                analyzed_frame_count=len(results),
                analysis_wall_seconds=elapsed,
                analysis_frames_per_second=frames_per_second,
                source_seconds_analyzed_per_wall_second=duration / elapsed,
            )

            with self.profiler.span("planning", "sort_analysis_results"):
                results.sort(key=lambda item: item["orden"])
            with self.profiler.span("planning", "build_cut_intervals"):
                cut_intervals = self._build_cut_intervals(results, duration)
            detector_name = (
                local_detector.name
                if local_detector is not None
                else self.detector_config.backend
            )
            with self.profiler.span("output", "reports_and_render"):
                self._render_results(results, duration, cut_intervals, detector_name)
            status = "completed"
            return results
        except BaseException as exc:
            self.profiler.error("run", "process_video", exc)
            raise
        finally:
            self.profiler.configure(final_rss_bytes=current_rss_bytes())
            try:
                self.profiler.write(status=status)
                if self.profiler.enabled:
                    print(f"Perfil de rendimiento guardado en {self.output_profile_path}")
            except BaseException as profile_error:
                print(
                    "Advertencia: no se pudo escribir el perfil de rendimiento: "
                    f"{profile_error}"
                )

