from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any

if "progress.bar" not in sys.modules:
    fake_progress = types.ModuleType("progress")
    fake_progress_bar = types.ModuleType("progress.bar")

    class DummyBar:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.count = 0

        def next(self) -> None:
            self.count += 1

        def finish(self) -> None:
            pass

    fake_progress_bar.ChargingBar = DummyBar
    sys.modules.setdefault("progress", fake_progress)
    sys.modules.setdefault("progress.bar", fake_progress_bar)

from imageio_ffmpeg import get_ffmpeg_exe

from applications.profiling import PerformanceProfiler
from applications.video_renderer import VideoRenderer


class ProfilerTests(unittest.TestCase):
    def test_profile_contains_atomic_events_percentiles_and_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.profile.json"
            profiler = PerformanceProfiler(str(path), enabled=True)
            profiler.configure(batch_size=4)
            profiler.increment("frames", 2)
            with profiler.span("test", "operation", item=1):
                time.sleep(0.001)
            profiler.event("test", "operation", duration_seconds=0.002, item=2)
            profiler.progress_sample("render", percent=50, out_time_seconds=1.0)
            profiler.write(status="completed")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["configuration"]["batch_size"], 4)
            self.assertEqual(payload["counters"]["frames"], 2.0)
            self.assertEqual(len(payload["events"]), 2)
            self.assertEqual(len(payload["ffmpeg_progress_samples"]), 1)
            summary = payload["summary"]["operations_by_total_time"][0]
            self.assertEqual(summary["count"], 2)
            self.assertIsNotNone(summary["p95_seconds"])

    def test_renderer_records_real_ffmpeg_progress(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            profile_path = root / "render.profile.json"
            ffmpeg = get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x64:rate=24:duration=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            profiler = PerformanceProfiler(str(profile_path), input_path=str(source))
            renderer = VideoRenderer(
                input_path=str(source),
                output_path=str(output),
                codec="auto",
                profiler=profiler,
            )
            result = renderer.render(
                allowed_intervals=[(0.0, 0.5), (1.0, 2.0)],
                cut_intervals=[(0.5, 1.0)],
                has_audio=False,
            )
            profiler.write(status="completed")

            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertTrue(result.generated)
            self.assertTrue(payload["ffmpeg_progress_samples"])
            self.assertEqual(payload["ffmpeg_progress_samples"][-1]["percent"], 100)
            self.assertTrue(
                any(event["name"] == "render_command" for event in payload["events"])
            )


if __name__ == "__main__":
    unittest.main()
