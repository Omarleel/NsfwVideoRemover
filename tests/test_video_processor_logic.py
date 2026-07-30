from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np

# The production dependency is installed by requirements-common.txt. Provide a
# tiny stand-in so these pure unit tests also run in minimal CI environments.
if "progress.bar" not in sys.modules:
    fake_progress = types.ModuleType("progress")
    fake_progress_bar = types.ModuleType("progress.bar")

    class DummyBar:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def next(self) -> None:
            pass

        def finish(self) -> None:
            pass

    fake_progress_bar.ChargingBar = DummyBar
    sys.modules.setdefault("progress", fake_progress)
    sys.modules.setdefault("progress.bar", fake_progress_bar)

from imageio_ffmpeg import get_ffmpeg_exe

from applications.NsfwVideoProcessor import NsfwVideoProcessor, _FfmpegFramePipeline
from applications.SrtGenerator import SrtGenerator
from applications.detectors.base import DetectionAssessment
from applications.reporting import AnalysisReportWriter
from applications.video_policies import CutIntervalPolicy, SegmentPlanner
from applications.video_renderer import VideoRenderer
from applications.video_probe import ImageioFfmpegVideoProbe


class DummyDetector:
    name = "dummy"
    device = "cpu"

    def analyze_batch(self, images, batch_size=None):
        return [
            DetectionAssessment(
                is_nsfw=False,
                score=0.1,
                detections=({"class": "normal", "score": 0.9},),
                model_name="dummy",
            )
            for _ in images
        ]

    def provider_summary(self) -> str:
        return "modelo=dummy; dispositivo=cpu"


class ProcessorLogicTests(unittest.TestCase):
    def _processor(self, **kwargs: Any) -> NsfwVideoProcessor:
        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp.close()
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return NsfwVideoProcessor(temp.name, **kwargs)

    def test_segment_generation_covers_tail(self) -> None:
        segments = SegmentPlanner(1.0).build(2.4)
        self.assertEqual(
            [segment["intervalo"] for segment in segments],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 2.4]],
        )

    def test_decimal_segment_generation_has_no_tiny_extra_tail(self) -> None:
        segments = SegmentPlanner(0.1).build(1.0)
        self.assertEqual(len(segments), 10)
        self.assertAlmostEqual(segments[-1]["intervalo"][1], 1.0)
        self.assertGreater(
            segments[-1]["intervalo"][1] - segments[-1]["intervalo"][0],
            0.09,
        )

    def test_worker_defaults_are_safe_and_injected_detector_forces_one(self) -> None:
        processor = self._processor(detector=DummyDetector())
        self.assertEqual(processor._resolve_workers(10, "cuda"), 1)
        self.assertEqual(processor._resolve_workers(10, "cpu"), 1)

    def test_external_registered_backend_forces_one_worker(self) -> None:
        from applications.detectors.config import DetectorConfig

        processor = self._processor(
            detector_config=DetectorConfig(backend="custom-registered")
        )
        self.assertEqual(processor._resolve_workers(10, "cpu"), 1)

    def test_auto_batch_and_prefetch_are_memory_bounded(self) -> None:
        processor = self._processor()
        hd_batch = processor._resolve_batch_size(1920, 1080, 4, "cpu")
        uhd_batch = processor._resolve_batch_size(3840, 2160, 4, "cpu")
        self.assertGreaterEqual(hd_batch, 1)
        self.assertLessEqual(hd_batch, 8)
        self.assertEqual(uhd_batch, 1)
        prefetch = processor._resolve_prefetch_frames(3840, 2160, 4, 1)
        self.assertGreaterEqual(prefetch, 2)
        self.assertLessEqual(prefetch * 3840 * 2160 * 3, 256 * 1024 * 1024)

    def test_ffmpeg_pipeline_uses_one_sequential_sampler(self) -> None:
        pipeline = _FfmpegFramePipeline(
            video_path="input.mp4",
            width=640,
            height=360,
            clip_duration=0.5,
            segments=[{"orden": 1}, {"orden": 2}, {"orden": 3}],
            prefetch_frames=4,
            ffmpeg_threads=2,
        )
        command = pipeline._command()
        self.assertEqual(command.count("-i"), 1)
        self.assertEqual(command[command.index("-frames:v") + 1], "3")
        self.assertIn(
            "setpts=PTS-STARTPTS,select=eq(n\\,0)+gte(t\\,selected_n*0.5)",
            command,
        )
        self.assertIn("passthrough", command)

    def test_ffmpeg_pipeline_scales_only_when_requested(self) -> None:
        pipeline = _FfmpegFramePipeline(
            video_path="input.mp4",
            width=1280,
            height=720,
            clip_duration=1.0,
            segments=[{"orden": 1}],
            prefetch_frames=2,
            ffmpeg_threads=1,
            resize_frames=True,
        )
        command = pipeline._command()
        self.assertIn("scale=1280:720:flags=fast_bilinear", command[command.index("-vf") + 1])

    def test_analysis_dimensions_preserve_aspect_ratio(self) -> None:
        processor = self._processor(analysis_max_dimension=1280)
        self.assertEqual(processor._resolve_analysis_dimensions(3840, 2160), (1280, 720))
        self.assertEqual(processor._resolve_analysis_dimensions(720, 1280), (720, 1280))

    def test_interval_normalization_sorts_and_merges_touching_ranges(self) -> None:
        self.assertEqual(
            VideoRenderer._normalize_intervals(
                [(4.0, 5.0), (0.0, 1.0), (1.0, 2.0), (4.5, 6.0)]
            ),
            [(0.0, 2.0), (4.0, 6.0)],
        )

    def test_cut_policy_merges_touching_intervals(self) -> None:
        policy = CutIntervalPolicy(1.0)
        results = [
            {"intervalo": [2.0, 3.0], "nsfw": True},
            {"intervalo": [4.0, 5.0], "nsfw": True},
        ]
        self.assertEqual(policy.build_cut_intervals(results, 10.0), [(1.0, 6.0)])
        self.assertEqual(
            policy.build_allowed_intervals(10.0, [(1.0, 6.0)]),
            [(0.0, 1.0), (6.0, 10.0)],
        )

    def test_zero_padding_still_removes_the_complete_detected_segment(self) -> None:
        policy = CutIntervalPolicy(0.0)
        results = [{"intervalo": [2.0, 3.0], "nsfw": True}]
        self.assertEqual(policy.build_cut_intervals(results, 10.0), [(2.0, 3.0)])

    def test_padding_marks_neighboring_segments(self) -> None:
        processor = self._processor()
        results = [{"orden": index + 1, "nsfw": index == 2} for index in range(6)]
        marked = processor.mark_nsfw(results, rango=1)
        self.assertEqual(
            [item["nsfw"] for item in marked],
            [False, True, True, True, False, False],
        )

    def test_renderer_removes_stale_output_when_everything_is_cut(self) -> None:
        source = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        source.close()
        output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output.write(b"old")
        output.close()
        self.addCleanup(lambda: Path(source.name).unlink(missing_ok=True))
        self.addCleanup(lambda: Path(output.name).unlink(missing_ok=True))
        renderer = VideoRenderer(input_path=source.name, output_path=output.name)
        result = renderer.render(allowed_intervals=[], cut_intervals=[(0.0, 1.0)])
        self.assertFalse(result.generated)
        self.assertFalse(Path(output.name).exists())


    def test_stream_copy_command_never_selects_an_encoder(self) -> None:
        source = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        source.close()
        output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output.close()
        self.addCleanup(lambda: Path(source.name).unlink(missing_ok=True))
        self.addCleanup(lambda: Path(output.name).unlink(missing_ok=True))
        renderer = VideoRenderer(input_path=source.name, output_path=output.name)
        command = renderer._build_concat_command("intervals.ffconcat")
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertNotIn("libx264", command)
        self.assertNotIn("h264_nvenc", command)

    def test_safe_intervals_start_at_next_keyframe(self) -> None:
        class FakeKeyframes:
            def locate(self, _input_path: str) -> list[float]:
                return [0.0, 2.0, 4.0, 6.0]

        source = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        source.close()
        output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output.close()
        self.addCleanup(lambda: Path(source.name).unlink(missing_ok=True))
        self.addCleanup(lambda: Path(output.name).unlink(missing_ok=True))
        renderer = VideoRenderer(
            input_path=source.name,
            output_path=output.name,
            keyframe_locator=FakeKeyframes(),
        )
        self.assertEqual(
            renderer._align_to_safe_keyframes(
                [(0.0, 1.5), (2.2, 5.5)],
                cut_intervals=[(1.5, 2.2)],
            ),
            [(4.0, 5.5)],
        )

    def test_report_writer_resets_srt_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            srt_path = str(Path(directory) / "result.srt")
            json_path = str(Path(directory) / "result.json")
            generator = SrtGenerator()
            writer = AnalysisReportWriter(generator)
            results = [
                {
                    "orden": 1,
                    "intervalo": [0.0, 1.0],
                    "nsfw": False,
                    "score_nsfw": 0.1,
                    "motivo": None,
                    "detecciones": [],
                }
            ]
            writer.write(
                video_path="input.mp4",
                duration=1.0,
                detector_name="dummy",
                results=results,
                cut_intervals=[],
                srt_path=srt_path,
                json_path=json_path,
            )
            writer.write(
                video_path="input.mp4",
                duration=1.0,
                detector_name="dummy",
                results=results,
                cut_intervals=[],
                srt_path=srt_path,
                json_path=json_path,
            )
            self.assertEqual(Path(srt_path).read_text(encoding="utf-8").count("-->"), 1)
            report = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(len(report["segments"]), 1)

    def test_srt_time_rounding_carries_to_next_second(self) -> None:
        self.assertEqual(SrtGenerator.format_time(1.9996), "00:00:02,000")


class FfmpegSamplingIntegrationTests(unittest.TestCase):
    def test_sampler_can_downscale_before_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = str(Path(directory) / "resize.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=128x64:r=30:d=1",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no incluye el filtro lavfi color.")

            segments = SegmentPlanner(1.0).build(1.0)
            with _FfmpegFramePipeline(
                video_path=video_path,
                width=64,
                height=32,
                clip_duration=1.0,
                segments=segments,
                prefetch_frames=2,
                ffmpeg_threads=1,
                resize_frames=True,
            ) as pipeline:
                frames = list(pipeline)

            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][1].shape, (32, 64, 3))

    def test_decimal_segments_match_real_ffmpeg_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = str(Path(directory) / "sample.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=30:d=1",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no incluye el filtro lavfi color.")

            segments = SegmentPlanner(0.1).build(1.0)
            with _FfmpegFramePipeline(
                video_path=video_path,
                width=64,
                height=64,
                clip_duration=0.1,
                segments=segments,
                prefetch_frames=2,
                ffmpeg_threads=1,
            ) as pipeline:
                frames = list(pipeline)
            self.assertEqual(len(frames), 10)

    def test_sampler_explicitly_returns_frame_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = str(Path(directory) / "first-frame.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=30:d=0.033333333",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=30:d=0.966666667",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video sintético.")

            segments = SegmentPlanner(1.0).build(1.0)
            with _FfmpegFramePipeline(
                video_path=video_path,
                width=64,
                height=64,
                clip_duration=1.0,
                segments=segments,
                prefetch_frames=2,
                ffmpeg_threads=1,
            ) as pipeline:
                frames = list(pipeline)

            self.assertEqual(len(frames), 1)
            center_pixel = frames[0][1][32, 32]
            self.assertGreater(int(center_pixel[0]), 200)
            self.assertLess(int(center_pixel[2]), 50)

    def test_stream_copy_renderer_avoids_unsafe_boundary_gops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "colors.mp4")
            output_path = str(Path(directory) / "safe.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=30:d=2",
                "-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-keyint_min",
                "30",
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video por colores.")

            renderer = VideoRenderer(
                input_path=source_path,
                output_path=output_path,
                codec="copy",
            )
            render_result = renderer.render(
                allowed_intervals=[(0.0, 2.0), (4.0, 6.0)],
                cut_intervals=[(2.0, 4.0)],
            )
            self.assertTrue(render_result.generated)
            self.assertEqual(render_result.codec, "copy")

            def center_pixel(timestamp: float) -> tuple[int, int, int]:
                frame_command = [
                    get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(timestamp),
                    "-i",
                    output_path,
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
                frame_result = subprocess.run(
                    frame_command, check=True, capture_output=True
                )
                frame = np.frombuffer(frame_result.stdout, dtype=np.uint8).reshape(
                    64, 64, 3
                )
                return tuple(int(value) for value in frame[32, 32])

            first = center_pixel(0.4)
            second = center_pixel(1.4)
            self.assertGreater(first[0], 180)
            self.assertLess(first[2], 80)
            self.assertGreater(second[1], 80)
            self.assertLess(second[2], 80)

    def test_exact_renderer_preserves_healthy_frames_between_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "long-gop.mp4")
            output_path = str(Path(directory) / "exact.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=30:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64:r=30:d=2",
                "-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
                "-c:v",
                "libx264",
                "-g",
                "180",
                "-keyint_min",
                "180",
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video por colores.")

            renderer = VideoRenderer(input_path=source_path, output_path=output_path)
            render_result = renderer.render(
                allowed_intervals=[(0.0, 1.5), (4.5, 6.0)],
                cut_intervals=[(1.5, 4.5)],
                has_audio=False,
            )
            self.assertTrue(render_result.generated)
            self.assertEqual(render_result.codec, "libx264")
            self.assertEqual(
                render_result.rendered_intervals,
                ((0.0, 1.5), (4.5, 6.0)),
            )

            def center_pixel(timestamp: float) -> tuple[int, int, int]:
                frame_command = [
                    get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(timestamp),
                    "-i",
                    output_path,
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
                frame_result = subprocess.run(
                    frame_command, check=True, capture_output=True
                )
                frame = np.frombuffer(frame_result.stdout, dtype=np.uint8).reshape(
                    64, 64, 3
                )
                return tuple(int(value) for value in frame[32, 32])

            first = center_pixel(1.4)
            second = center_pixel(1.6)
            self.assertGreater(first[0], 180)
            self.assertLess(first[2], 80)
            self.assertGreater(second[1], 80)
            self.assertLess(second[2], 80)

    def test_exact_renderer_keeps_audio_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "with-audio.mp4")
            output_path = str(Path(directory) / "exact-audio.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:r=30:d=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=6",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                "-shortest",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear video con audio.")

            renderer = VideoRenderer(input_path=source_path, output_path=output_path)
            render_result = renderer.render(
                allowed_intervals=[(0.0, 1.5), (4.5, 6.0)],
                cut_intervals=[(1.5, 4.5)],
            )
            self.assertTrue(render_result.generated)
            metadata = ImageioFfmpegVideoProbe().probe(output_path)
            self.assertTrue(metadata.has_audio)
            self.assertAlmostEqual(metadata.duration, 3.0, delta=0.12)

    def test_exact_renderer_handles_a_cut_at_the_beginning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "initial-cut.mp4")
            output_path = str(Path(directory) / "initial-cut-result.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=30:d=1.5",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=30:d=1.5",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video sintético.")

            renderer = VideoRenderer(input_path=source_path, output_path=output_path)
            render_result = renderer.render(
                allowed_intervals=[(1.5, 3.0)],
                cut_intervals=[(0.0, 1.5)],
                has_audio=False,
            )
            self.assertTrue(render_result.generated)
            metadata = ImageioFfmpegVideoProbe().probe(output_path)
            self.assertAlmostEqual(metadata.duration, 1.5, delta=0.12)

            frame_command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "0.2",
                "-i",
                output_path,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            frame_result = subprocess.run(frame_command, check=True, capture_output=True)
            frame = np.frombuffer(frame_result.stdout, dtype=np.uint8).reshape(64, 64, 3)
            center = tuple(int(value) for value in frame[32, 32])
            self.assertGreater(center[2], 180)
            self.assertLess(center[0], 80)

    def test_exact_renderer_removes_gaps_when_first_healthy_interval_starts_late(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "late-first-interval.mp4")
            output_path = str(Path(directory) / "late-first-result.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=160x90:r=30:d=10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video sintético.")

            renderer = VideoRenderer(input_path=source_path, output_path=output_path)
            render_result = renderer.render(
                allowed_intervals=[(2.0, 3.0), (8.0, 9.0)],
                cut_intervals=[(0.0, 2.0), (3.0, 8.0), (9.0, 10.0)],
                has_audio=False,
            )

            self.assertTrue(render_result.generated)
            self.assertAlmostEqual(render_result.expected_duration or 0.0, 2.0)
            metadata = ImageioFfmpegVideoProbe().probe(output_path)
            self.assertAlmostEqual(metadata.duration, 2.0, delta=0.12)
            self.assertGreater(metadata.fps, 25.0)

            timestamp_result = subprocess.run(
                [
                    get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "info",
                    "-i",
                    output_path,
                    "-vf",
                    "showinfo",
                    "-f",
                    "null",
                    "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            timestamps = []
            for line in timestamp_result.stderr.splitlines():
                marker = "pts_time:"
                if marker not in line:
                    continue
                value = line.split(marker, 1)[1].split()[0]
                timestamps.append(float(value))
            self.assertGreater(len(timestamps), 2)
            maximum_gap = max(
                right - left for left, right in zip(timestamps, timestamps[1:])
            )
            self.assertLess(maximum_gap, 0.1)

    def test_duration_guard_falls_back_to_independent_trim_concat(self) -> None:
        class BrokenFastRenderer(VideoRenderer):
            @staticmethod
            def _build_exact_filter(intervals, *, has_audio):
                if has_audio:
                    raise AssertionError("Esta prueba usa un video sin audio.")
                selection = "+".join(
                    f"gte(t,{start:.9f})*lt(t,{end:.9f})"
                    for start, end in intervals
                )
                first_start = intervals[0][0]
                terms = [f"{first_start:.9f}/TB"]
                for (_, previous_end), (next_start, _) in zip(
                    intervals, intervals[1:]
                ):
                    gap = next_start - previous_end
                    terms.append(
                        "gte(PTS-STARTPTS,"
                        f"{next_start:.9f}/TB)*{gap:.9f}/TB"
                    )
                expression = "PTS-" + "-".join(terms)
                return (
                    f"[0:v:0]setpts=PTS-STARTPTS,select='{selection}',"
                    f"setpts='{expression}'[vout]\n"
                )

        with tempfile.TemporaryDirectory() as directory:
            source_path = str(Path(directory) / "broken-fast-path.mp4")
            output_path = str(Path(directory) / "fallback-result.mp4")
            command = [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=160x90:r=30:d=10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                source_path,
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                self.skipTest("El FFmpeg disponible no puede crear el video sintético.")

            renderer = BrokenFastRenderer(
                input_path=source_path,
                output_path=output_path,
            )
            renderer.render(
                allowed_intervals=[(2.0, 3.0), (8.0, 9.0)],
                cut_intervals=[(0.0, 2.0), (3.0, 8.0), (9.0, 10.0)],
                has_audio=False,
            )
            metadata = ImageioFfmpegVideoProbe().probe(output_path)
            self.assertAlmostEqual(metadata.duration, 2.0, delta=0.12)


if __name__ == "__main__":
    unittest.main()
