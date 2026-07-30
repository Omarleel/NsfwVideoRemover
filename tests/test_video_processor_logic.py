from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

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
        self.assertIn("fps=fps=1/0.5:start_time=0:eof_action=pass", command)

    def test_cut_policy_merges_touching_intervals(self) -> None:
        policy = CutIntervalPolicy(1.0)
        results = [
            {"intervalo": [2.0, 3.0], "nsfw": True},
            {"intervalo": [4.0, 5.0], "nsfw": True},
        ]
        self.assertEqual(policy.build_cut_intervals(results, 10.0), [(1.0, 5.0)])
        self.assertEqual(
            policy.build_allowed_intervals(10.0, [(1.0, 5.0)]),
            [(0.0, 1.0), (5.0, 10.0)],
        )

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


if __name__ == "__main__":
    unittest.main()
