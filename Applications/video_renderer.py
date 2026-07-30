from __future__ import annotations

import bisect
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from imageio_ffmpeg import get_ffmpeg_exe
from progress.bar import ChargingBar

from applications.ffmpeg_capabilities import (
    FfmpegCapabilities,
    resolve_ffmpeg_executable,
)
from applications.profiling import PerformanceProfiler
from applications.video_probe import ImageioFfmpegVideoProbe


_KEYFRAME_PATTERN = re.compile(r"\bpts_time:([-+0-9.eE]+)")
_EPSILON = 1e-6


class KeyframeLocator(Protocol):
    def locate(self, input_path: str) -> list[float]:
        """Return keyframe timestamps in seconds, ordered increasingly."""


class FfmpegKeyframeLocator:
    """Find keyframes with the same FFmpeg binary used by the application.

    ``-skip_frame nokey`` asks the decoder to discard non-keyframes, avoiding a
    full frame-by-frame render while still working when a separate ffprobe
    executable is not installed.
    """

    def __init__(self, ffmpeg_executable: str | None = None) -> None:
        self.ffmpeg_executable = ffmpeg_executable or get_ffmpeg_exe()

    def locate(self, input_path: str) -> list[float]:
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-skip_frame",
            "nokey",
            "-i",
            input_path,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "showinfo",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip()
            raise RuntimeError(
                "FFmpeg no pudo localizar los keyframes."
                + (f" Detalle: {details}" if details else "")
            )

        keyframes = sorted(
            {
                max(0.0, float(match.group(1)))
                for match in _KEYFRAME_PATTERN.finditer(result.stderr)
            }
        )
        if not keyframes:
            raise RuntimeError("FFmpeg no devolvió ningún keyframe utilizable.")
        if keyframes[0] > _EPSILON:
            keyframes.insert(0, 0.0)
        return keyframes


@dataclass(frozen=True)
class IntervalAlignment:
    requested_start: float
    requested_end: float
    rendered_start: float
    rendered_end: float


@dataclass(frozen=True)
class RenderResult:
    generated: bool
    codec: str | None
    reason: str
    requested_intervals: tuple[tuple[float, float], ...] = ()
    rendered_intervals: tuple[tuple[float, float], ...] = ()
    expected_duration: float | None = None
    actual_duration: float | None = None


class VideoRenderer:
    """Render healthy intervals with exact or keyframe-safe cuts.

    ``codec=auto`` uses an exact FFmpeg trim/concat pipeline and re-encodes only
    when cuts are required. ``codec=copy`` keeps the legacy keyframe-safe mode:
    it is faster, but may discard healthy GOPs around every boundary.
    """

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        codec: str = "auto",
        fast_copy_when_unchanged: bool = True,
        keyframe_locator: KeyframeLocator | None = None,
        ffmpeg_executable: str | None = None,
        ffmpeg_capabilities: FfmpegCapabilities | None = None,
        hardware_acceleration: str = "auto",
        profiler: PerformanceProfiler | None = None,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.codec = str(codec or "auto").strip()
        self.fast_copy_when_unchanged = bool(fast_copy_when_unchanged)
        self.hardware_acceleration = str(hardware_acceleration or "auto").casefold()
        if ffmpeg_executable is None or ffmpeg_capabilities is None:
            resolved_executable, resolved_capabilities = resolve_ffmpeg_executable(
                ffmpeg_executable, prefer_hardware=self.hardware_acceleration != "none"
            )
            self.ffmpeg_executable = resolved_executable
            self.ffmpeg_capabilities = resolved_capabilities
        else:
            self.ffmpeg_executable = ffmpeg_executable
            self.ffmpeg_capabilities = ffmpeg_capabilities
        self.keyframe_locator = keyframe_locator or FfmpegKeyframeLocator(
            self.ffmpeg_executable
        )
        self.profiler = profiler
        if self.profiler is not None:
            self.profiler.configure(
                render_ffmpeg_executable=self.ffmpeg_executable,
                render_nvenc_available=self.ffmpeg_capabilities.supports_h264_nvenc,
            )

    def remove_stale_output(self) -> None:
        try:
            os.remove(self.output_path)
        except FileNotFoundError:
            pass

    def codec_candidates(self) -> list[str]:
        """Backward-compatible diagnostic helper."""

        if self.codec.casefold() == "copy":
            return ["copy"]
        if self.codec.casefold() in {"", "auto"}:
            candidates: list[str] = []
            if (
                self.hardware_acceleration != "none"
                and self.ffmpeg_capabilities.supports_h264_nvenc
            ):
                candidates.append("h264_nvenc")
            candidates.append("libx264")
            return candidates
        if self.codec.casefold() == "libx264":
            return ["libx264"]
        if self.codec.casefold() == "h264_nvenc":
            return ["h264_nvenc", "libx264"]
        return [self.codec, "libx264"]

    @staticmethod
    def _ffconcat_quote(path: str) -> str:
        normalized = Path(path).resolve().as_posix()
        return "'" + normalized.replace("'", "'\\''") + "'"

    @staticmethod
    def _parse_ffmpeg_time(value: str) -> float | None:
        try:
            hours, minutes, seconds = value.split(":", 2)
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (TypeError, ValueError):
            return None

    def _run(
        self,
        command: list[str],
        operation: str,
        *,
        expected_duration: float | None = None,
        progress_phase: str = "final_video_generation",
    ) -> None:
        runtime_command = list(command[:-1])
        runtime_command.extend(
            ["-stats_period", "0.25", "-progress", "pipe:1", "-nostats", command[-1]]
        )
        stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        bar = ChargingBar("Generación final del video", max=100)
        completed_percent = 0
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        start_offset = (
            self.profiler.now_offset_seconds() if self.profiler is not None else None
        )
        latest: dict[str, str] = {}
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                runtime_command,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if self.profiler is not None:
                self.profiler.register_child_process(
                    process.pid, role="ffmpeg_final_render", command=runtime_command
                )
            if process.stdout is None:
                raise RuntimeError("FFmpeg no expuso su canal de progreso.")

            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                latest[key] = value
                if key != "progress":
                    continue

                out_time = None
                if latest.get("out_time_us"):
                    try:
                        out_time = float(latest["out_time_us"]) / 1_000_000
                    except ValueError:
                        pass
                if out_time is None and latest.get("out_time_ms"):
                    try:
                        # FFmpeg's historical out_time_ms field is microseconds.
                        out_time = float(latest["out_time_ms"]) / 1_000_000
                    except ValueError:
                        pass
                if out_time is None and latest.get("out_time"):
                    out_time = self._parse_ffmpeg_time(latest["out_time"])

                percent = completed_percent
                if expected_duration and expected_duration > 0 and out_time is not None:
                    percent = min(99, max(0, int(out_time / expected_duration * 100)))
                if value == "end":
                    percent = 100
                while completed_percent < percent:
                    bar.next()
                    completed_percent += 1

                if self.profiler is not None:
                    def as_float(field: str) -> float | None:
                        raw = latest.get(field)
                        if raw in {None, "N/A", "nan"}:
                            return None
                        try:
                            return float(str(raw).rstrip("xkbits/s"))
                        except ValueError:
                            return None

                    self.profiler.progress_sample(
                        progress_phase,
                        operation=operation,
                        progress=value,
                        percent=percent,
                        out_time_seconds=out_time,
                        frame=as_float("frame"),
                        fps=as_float("fps"),
                        speed=as_float("speed"),
                        total_size_bytes=as_float("total_size"),
                        bitrate=latest.get("bitrate"),
                    )
                latest = {}

            return_code = process.wait()
            if return_code == 0 and os.path.isfile(self.output_path):
                if self.profiler is not None:
                    self.profiler.capture_system_sample(reason="ffmpeg_render_completed")
                while completed_percent < 100:
                    bar.next()
                    completed_percent += 1
                return

            stderr_file.seek(0)
            details = stderr_file.read().strip()
            self.remove_stale_output()
            raise RuntimeError(
                f"FFmpeg no pudo {operation}."
                + (f" Detalle: {details}" if details else "")
            )
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            bar.finish()
            stderr_file.close()
            if self.profiler is not None:
                self.profiler.event(
                    "ffmpeg",
                    "render_command",
                    duration_seconds=time.perf_counter() - wall_started,
                    cpu_seconds=time.process_time() - cpu_started,
                    start_offset_seconds=start_offset,
                    operation=operation,
                    expected_duration_seconds=expected_duration,
                    command=runtime_command,
                    output_path=self.output_path,
                )

    def _copy_complete_video(self, expected_duration: float) -> None:
        self.remove_stale_output()
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            self.input_path,
            "-map",
            "0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        self._run(
            command,
            "copiar los streams",
            expected_duration=expected_duration,
            progress_phase="full_stream_copy",
        )

    @staticmethod
    def _normalize_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        candidates: list[tuple[float, float]] = []
        for start, end in intervals:
            start_value = max(0.0, float(start))
            end_value = max(start_value, float(end))
            if end_value - start_value > _EPSILON:
                candidates.append((start_value, end_value))
        candidates.sort(key=lambda item: (item[0], item[1]))
        normalized: list[tuple[float, float]] = []
        for start, end in candidates:
            if not normalized or start > normalized[-1][1] + _EPSILON:
                normalized.append((start, end))
                continue
            previous_start, previous_end = normalized[-1]
            normalized[-1] = (previous_start, max(previous_end, end))
        return normalized

    def _detect_audio_stream(self) -> bool:
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            self.input_path,
            "-map",
            "0:a:0",
            "-frames:a",
            "1",
            "-f",
            "null",
            "-",
        ]
        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        result = subprocess.run(command, check=False, capture_output=True)
        detected = result.returncode == 0
        if self.profiler is not None:
            self.profiler.event(
                "ffmpeg",
                "detect_audio_stream",
                duration_seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu_started,
                start_offset_seconds=offset,
                return_code=result.returncode,
                audio_detected=detected,
                command=command,
            )
        return detected

    @staticmethod
    def _build_exact_filter(
        intervals: list[tuple[float, float]],
        *,
        has_audio: bool,
    ) -> str:
        selection = "+".join(
            f"gte(t,{start:.9f})*lt(t,{end:.9f})" for start, end in intervals
        )
        # The first setpts normalizes the source timeline to zero. After
        # ``select``, STARTPTS is the timestamp of the first retained frame,
        # while PTS still represents its position in the normalized source.
        # Remove every discarded gap using absolute normalized PTS values.
        #
        # Do not compare ``PTS-STARTPTS`` with ``next_start`` here: when the
        # first healthy interval starts after 0, that shifts every threshold
        # and leaves large timestamp holes in the output. Those holes inflate
        # duration and make players appear frozen between retained intervals.
        timestamp_terms: list[str] = []
        for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]):
            gap = next_start - previous_end
            timestamp_terms.append(
                "gte(PTS,"
                f"{next_start:.9f}/TB)*{gap:.9f}/TB"
            )
        timestamp_expression = "PTS-STARTPTS"
        if timestamp_terms:
            timestamp_expression += "-" + "-".join(timestamp_terms)
        lines = [
            f"[0:v:0]setpts=PTS-STARTPTS,select='{selection}',"
            f"setpts='{timestamp_expression}'[vout]"
        ]

        if not has_audio:
            return ";\n".join(lines) + "\n"

        for index, (start, end) in enumerate(intervals):
            duration = end - start
            audio_label = "aout" if len(intervals) == 1 else f"a{index}"
            lines.append(
                f"[0:a:0]asetpts=PTS-STARTPTS,"
                f"atrim=start={start:.9f}:end={end:.9f},"
                "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,"
                f"apad=pad_dur={duration:.9f},atrim=duration={duration:.9f}"
                f"[{audio_label}]"
            )

        if len(intervals) > 1:
            inputs = "".join(f"[a{index}]" for index in range(len(intervals)))
            lines.append(f"{inputs}concat=n={len(intervals)}:v=0:a=1[aout]")
        return ";\n".join(lines) + "\n"

    @staticmethod
    def _video_encoder_arguments(encoder: str) -> list[str]:
        normalized = encoder.casefold()
        if normalized == "libx264":
            return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        if normalized == "libx265":
            return ["-c:v", "libx265", "-preset", "veryfast", "-crf", "20"]
        if normalized == "h264_nvenc":
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
                "-spatial_aq",
                "1",
            ]
        return ["-c:v", encoder]

    def _build_exact_command(
        self,
        filter_script_path: str,
        *,
        encoder: str,
        has_audio: bool,
    ) -> list[str]:
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            self.input_path,
            "-filter_complex_script",
            filter_script_path,
            "-map",
            "[vout]",
        ]
        if has_audio:
            command.extend(["-map", "[aout]"])
        command.extend(self._video_encoder_arguments(encoder))
        command.extend(["-pix_fmt", "yuv420p", "-fps_mode", "vfr"])
        if has_audio:
            command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        else:
            command.append("-an")
        command.extend(
            [
                "-map_metadata",
                "0",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                self.output_path,
            ]
        )
        return command

    def _probe_output_duration(self) -> float:
        """Read output duration cheaply from FFmpeg metadata."""
        started = time.perf_counter()
        cpu_started = time.process_time()
        offset = self.profiler.now_offset_seconds() if self.profiler else None
        duration = ImageioFfmpegVideoProbe().probe(self.output_path).duration
        if self.profiler is not None:
            self.profiler.event(
                "probe",
                "probe_rendered_output_duration",
                duration_seconds=time.perf_counter() - started,
                cpu_seconds=time.process_time() - cpu_started,
                start_offset_seconds=offset,
                output_duration_seconds=duration,
            )
        return duration

    @staticmethod
    def _expected_duration(intervals: list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)

    @staticmethod
    def _duration_is_valid(expected: float, actual: float) -> bool:
        # Allow normal container/frame rounding but reject timeline holes.
        tolerance = max(0.25, min(1.0, expected * 0.001))
        return abs(actual - expected) <= tolerance

    @staticmethod
    def _build_concat_filter(
        intervals: list[tuple[float, float]],
        *,
        has_audio: bool,
    ) -> str:
        """Build a conservative trim/concat graph used only as a fallback.

        This graph resets every retained interval independently before concat,
        guaranteeing contiguous timestamps even for unusual source timebases.
        """

        lines: list[str] = []
        count = len(intervals)

        if count == 1:
            start, end = intervals[0]
            lines.append(
                f"[0:v:0]setpts=PTS-STARTPTS,"
                f"trim=start={start:.9f}:end={end:.9f},"
                "setpts=PTS-STARTPTS[vout]"
            )
        else:
            video_sources = "".join(f"[vsrc{index}]" for index in range(count))
            lines.append(
                f"[0:v:0]setpts=PTS-STARTPTS,split={count}{video_sources}"
            )
            for index, (start, end) in enumerate(intervals):
                lines.append(
                    f"[vsrc{index}]trim=start={start:.9f}:end={end:.9f},"
                    f"setpts=PTS-STARTPTS[v{index}]"
                )
            video_inputs = "".join(f"[v{index}]" for index in range(count))
            lines.append(f"{video_inputs}concat=n={count}:v=1:a=0[vout]")

        if not has_audio:
            return ";\n".join(lines) + "\n"

        if count == 1:
            start, end = intervals[0]
            duration = end - start
            lines.append(
                f"[0:a:0]asetpts=PTS-STARTPTS,"
                f"atrim=start={start:.9f}:end={end:.9f},"
                "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,"
                f"apad=pad_dur={duration:.9f},atrim=duration={duration:.9f}[aout]"
            )
            return ";\n".join(lines) + "\n"

        audio_sources = "".join(f"[asrc{index}]" for index in range(count))
        lines.append(
            f"[0:a:0]asetpts=PTS-STARTPTS,asplit={count}{audio_sources}"
        )
        for index, (start, end) in enumerate(intervals):
            duration = end - start
            lines.append(
                f"[asrc{index}]atrim=start={start:.9f}:end={end:.9f},"
                "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,"
                f"apad=pad_dur={duration:.9f},atrim=duration={duration:.9f}"
                f"[a{index}]"
            )
        audio_inputs = "".join(f"[a{index}]" for index in range(count))
        lines.append(f"{audio_inputs}concat=n={count}:v=0:a=1[aout]")
        return ";\n".join(lines) + "\n"

    def _render_exact(
        self,
        intervals: list[tuple[float, float]],
        *,
        has_audio: bool,
    ) -> tuple[str, float]:
        output_directory = str(Path(self.output_path).resolve().parent)
        with tempfile.TemporaryDirectory(
            prefix=".nsfw-exact-", dir=output_directory
        ) as directory:
            filter_script_path = str(Path(directory) / "trim-concat.ffmpeg")
            build_started = time.perf_counter()
            build_cpu = time.process_time()
            build_offset = self.profiler.now_offset_seconds() if self.profiler else None
            filter_text = self._build_exact_filter(intervals, has_audio=has_audio)
            if self.profiler is not None:
                self.profiler.event(
                    "render_atomic",
                    "build_exact_filter_graph",
                    duration_seconds=time.perf_counter() - build_started,
                    cpu_seconds=time.process_time() - build_cpu,
                    start_offset_seconds=build_offset,
                    interval_count=len(intervals),
                    has_audio=has_audio,
                    filter_length_chars=len(filter_text),
                )
            write_started = time.perf_counter()
            write_cpu = time.process_time()
            write_offset = self.profiler.now_offset_seconds() if self.profiler else None
            Path(filter_script_path).write_text(filter_text, encoding="utf-8")
            if self.profiler is not None:
                self.profiler.event(
                    "render_atomic",
                    "write_filter_script",
                    duration_seconds=time.perf_counter() - write_started,
                    cpu_seconds=time.process_time() - write_cpu,
                    start_offset_seconds=write_offset,
                    path=filter_script_path,
                    size_bytes=Path(filter_script_path).stat().st_size,
                )
            errors: list[str] = []
            expected_duration = self._expected_duration(intervals)
            encoder_candidates = self.codec_candidates()
            if self.profiler is not None:
                self.profiler.configure(render_encoder_candidates=encoder_candidates)
            for encoder in encoder_candidates:
                self.remove_stale_output()
                command = self._build_exact_command(
                    filter_script_path,
                    encoder=encoder,
                    has_audio=has_audio,
                )
                try:
                    self._run(
                        command,
                        f"generar cortes exactos con {encoder}",
                        expected_duration=expected_duration,
                        progress_phase=f"exact_render_{encoder}",
                    )
                    actual_duration = self._probe_output_duration()
                    valid_duration = self._duration_is_valid(
                        expected_duration, actual_duration
                    )
                    if self.profiler is not None:
                        self.profiler.event(
                            "render_atomic",
                            "validate_output_duration",
                            expected_duration_seconds=expected_duration,
                            actual_duration_seconds=actual_duration,
                            absolute_error_seconds=abs(actual_duration - expected_duration),
                            valid=valid_duration,
                            encoder=encoder,
                        )
                    if valid_duration:
                        if self.profiler is not None:
                            self.profiler.configure(
                                resolved_render_encoder=encoder,
                                render_used_nvenc=encoder.casefold() == "h264_nvenc",
                            )
                        return encoder, actual_duration

                    # A malformed timestamp graph must never be returned to the
                    # user. Retry with independent trim branches, which is a bit
                    # heavier but guarantees a continuous output timeline.
                    fallback_started = time.perf_counter()
                    fallback_cpu = time.process_time()
                    fallback_offset = (
                        self.profiler.now_offset_seconds() if self.profiler else None
                    )
                    fallback_filter = self._build_concat_filter(
                        intervals, has_audio=has_audio
                    )
                    Path(filter_script_path).write_text(
                        fallback_filter, encoding="utf-8"
                    )
                    if self.profiler is not None:
                        self.profiler.event(
                            "render_atomic",
                            "build_and_write_fallback_filter",
                            duration_seconds=time.perf_counter() - fallback_started,
                            cpu_seconds=time.process_time() - fallback_cpu,
                            start_offset_seconds=fallback_offset,
                            interval_count=len(intervals),
                            filter_length_chars=len(fallback_filter),
                        )
                    self.remove_stale_output()
                    self._run(
                        command,
                        f"regenerar cortes con timestamps continuos usando {encoder}",
                        expected_duration=expected_duration,
                        progress_phase=f"fallback_concat_{encoder}",
                    )
                    actual_duration = self._probe_output_duration()
                    valid_duration = self._duration_is_valid(
                        expected_duration, actual_duration
                    )
                    if self.profiler is not None:
                        self.profiler.event(
                            "render_atomic",
                            "validate_fallback_output_duration",
                            expected_duration_seconds=expected_duration,
                            actual_duration_seconds=actual_duration,
                            absolute_error_seconds=abs(actual_duration - expected_duration),
                            valid=valid_duration,
                            encoder=encoder,
                        )
                    if valid_duration:
                        if self.profiler is not None:
                            self.profiler.configure(
                                resolved_render_encoder=encoder,
                                render_used_nvenc=encoder.casefold() == "h264_nvenc",
                            )
                        return encoder, actual_duration
                    raise RuntimeError(
                        "La duración generada no coincide con la suma de los "
                        "intervalos sanos "
                        f"(esperada={expected_duration:.3f}s; "
                        f"real={actual_duration:.3f}s)."
                    )
                except RuntimeError as exc:
                    errors.append(str(exc))
            raise RuntimeError(" No se pudo usar ningún encoder. ".join(errors))

    def _plan_safe_intervals(
        self,
        allowed_intervals: list[tuple[float, float]],
        cut_intervals: list[tuple[float, float]] | None = None,
    ) -> list[IntervalAlignment]:
        normalized = self._normalize_intervals(allowed_intervals)
        if not normalized:
            return []

        keyframes = self.keyframe_locator.locate(self.input_path)
        cut_starts = [float(start) for start, _ in (cut_intervals or [])]
        plans: list[IntervalAlignment] = []
        for start, end in normalized:
            if start <= _EPSILON:
                safe_start = 0.0
            else:
                start_index = bisect.bisect_left(keyframes, start - _EPSILON)
                if start_index >= len(keyframes):
                    continue
                safe_start = keyframes[start_index]

            borders_a_cut = any(
                abs(cut_start - end) <= _EPSILON for cut_start in cut_starts
            )
            safe_end = end
            if borders_a_cut:
                # Use the keyframe strictly before the unsafe interval. This
                # discards the complete boundary GOP, preventing reordered
                # B-frames from crossing into the stream-copy result.
                end_index = bisect.bisect_left(keyframes, end - _EPSILON) - 1
                if end_index < 0:
                    continue
                safe_end = keyframes[end_index]

            if safe_end - safe_start > _EPSILON:
                plans.append(
                    IntervalAlignment(start, end, safe_start, safe_end)
                )
        return plans

    def _align_to_safe_keyframes(
        self,
        allowed_intervals: list[tuple[float, float]],
        cut_intervals: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        return [
            (plan.rendered_start, plan.rendered_end)
            for plan in self._plan_safe_intervals(
                allowed_intervals, cut_intervals=cut_intervals
            )
        ]

    def _write_concat_manifest(
        self,
        file_path: str,
        intervals: list[tuple[float, float]],
    ) -> None:
        quoted_input = self._ffconcat_quote(self.input_path)
        lines = ["ffconcat version 1.0"]
        for start, end in intervals:
            lines.extend(
                [
                    f"file {quoted_input}",
                    f"inpoint {start:.9f}",
                    f"outpoint {end:.9f}",
                ]
            )
        Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _build_concat_command(self, manifest_path: str) -> list[str]:
        return [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-fflags",
            "+genpts",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            manifest_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            self.output_path,
        ]

    def render(
        self,
        *,
        allowed_intervals: list[tuple[float, float]],
        cut_intervals: list[tuple[float, float]],
        has_audio: bool | None = None,
    ) -> RenderResult:
        normalize_started = time.perf_counter()
        normalize_cpu = time.process_time()
        normalize_offset = self.profiler.now_offset_seconds() if self.profiler else None
        requested = tuple(self._normalize_intervals(allowed_intervals))
        if self.profiler is not None:
            self.profiler.event(
                "render_atomic",
                "normalize_allowed_intervals",
                duration_seconds=time.perf_counter() - normalize_started,
                cpu_seconds=time.process_time() - normalize_cpu,
                start_offset_seconds=normalize_offset,
                input_count=len(allowed_intervals),
                output_count=len(requested),
            )
        if not cut_intervals:
            print("No se detectaron clases prohibidas; copiando el video sin recodificar.")
            started_at = time.perf_counter()
            self._copy_complete_video(self._expected_duration(list(requested)))
            elapsed = time.perf_counter() - started_at
            print(
                f"Video guardado en {self.output_path} mediante stream copy "
                f"({elapsed:.2f}s; calidad original)."
            )
            return RenderResult(
                True,
                "copy",
                "video completo copiado sin recodificación",
                requested,
                requested,
                self._expected_duration(list(requested)),
                self._expected_duration(list(requested)),
            )

        if not requested:
            self.remove_stale_output()
            print("Todo el video quedó dentro de los cortes; no se generó salida.")
            return RenderResult(
                False,
                None,
                "todo el video fue eliminado",
                requested,
                (),
            )

        if self.codec.casefold() != "copy":
            audio_present = self._detect_audio_stream() if has_audio is None else bool(has_audio)
            print(
                "Generando cortes exactos sin descartar GOP sanos "
                "(recodificación optimizada)."
            )
            started_at = time.perf_counter()
            encoder, actual_duration = self._render_exact(
                list(requested), has_audio=audio_present
            )
            elapsed = time.perf_counter() - started_at
            expected_duration = self._expected_duration(list(requested))
            print(
                f"Video guardado en {self.output_path} con {encoder} "
                f"({elapsed:.2f}s; duración sana esperada={expected_duration:.3f}s; "
                f"duración generada={actual_duration:.3f}s)."
            )
            return RenderResult(
                True,
                encoder,
                "intervalos exactos unidos mediante trim/concat",
                requested,
                requested,
                expected_duration,
                actual_duration,
            )

        print("Localizando keyframes para unir sin pérdida de calidad...")
        plan_started = time.perf_counter()
        plan_cpu = time.process_time()
        plan_offset = self.profiler.now_offset_seconds() if self.profiler else None
        plans = self._plan_safe_intervals(
            list(requested), cut_intervals=cut_intervals
        )
        if self.profiler is not None:
            self.profiler.event(
                "render_atomic",
                "locate_keyframes_and_plan_intervals",
                duration_seconds=time.perf_counter() - plan_started,
                cpu_seconds=time.process_time() - plan_cpu,
                start_offset_seconds=plan_offset,
                requested_interval_count=len(requested),
                planned_interval_count=len(plans),
            )
        rendered = [
            (plan.rendered_start, plan.rendered_end) for plan in plans
        ]
        if not rendered:
            self.remove_stale_output()
            print("No quedaron intervalos decodificables después de alinear keyframes.")
            return RenderResult(
                False,
                None,
                "ningún intervalo sano comenzó en un keyframe utilizable",
                requested,
                (),
            )

        for plan in plans:
            if plan.rendered_start - plan.requested_start > _EPSILON:
                print(
                    "  Inicio sano ajustado por seguridad: "
                    f"{plan.requested_start:.3f}s -> {plan.rendered_start:.3f}s "
                    "(siguiente keyframe)."
                )
            if plan.requested_end - plan.rendered_end > _EPSILON:
                print(
                    "  Final sano ajustado por seguridad: "
                    f"{plan.requested_end:.3f}s -> {plan.rendered_end:.3f}s "
                    "(keyframe previo al corte)."
                )

        self.remove_stale_output()
        started_at = time.perf_counter()
        output_directory = str(Path(self.output_path).resolve().parent)
        with tempfile.TemporaryDirectory(
            prefix=".nsfw-remux-", dir=output_directory
        ) as directory:
            manifest_path = str(Path(directory) / "intervals.ffconcat")
            manifest_started = time.perf_counter()
            manifest_cpu = time.process_time()
            manifest_offset = self.profiler.now_offset_seconds() if self.profiler else None
            self._write_concat_manifest(manifest_path, rendered)
            if self.profiler is not None:
                self.profiler.event(
                    "render_atomic",
                    "write_concat_manifest",
                    duration_seconds=time.perf_counter() - manifest_started,
                    cpu_seconds=time.process_time() - manifest_cpu,
                    start_offset_seconds=manifest_offset,
                    interval_count=len(rendered),
                    path=manifest_path,
                )
            command = self._build_concat_command(manifest_path)
            self._run(
                command,
                "unir los intervalos mediante stream copy",
                expected_duration=self._expected_duration(rendered),
                progress_phase="keyframe_stream_copy",
            )

        elapsed = time.perf_counter() - started_at
        print(
            f"Video guardado en {self.output_path} mediante stream copy "
            f"({elapsed:.2f}s; sin recodificación ni pérdida generacional)."
        )
        return RenderResult(
            True,
            "copy",
            "intervalos unidos sin recodificación",
            requested,
            tuple(rendered),
            self._expected_duration(list(requested)),
            self._expected_duration(rendered),
        )
