from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "applications"
    / "NsfwVideoProcessor.py"
)


class DummyVideoFileClip:
    pass


class DummySrtGenerator:
    pass


class DummyDetector:
    pass


class DummyBar:
    def __init__(self, *_args, **_kwargs):
        pass

    def next(self):
        pass

    def finish(self):
        pass


class ProcessorLogicTests(unittest.TestCase):
    def _load_module(self):
        fake_moviepy = types.ModuleType("moviepy")
        fake_moviepy.VideoFileClip = DummyVideoFileClip
        fake_moviepy.concatenate_videoclips = lambda clips, method=None: (clips, method)

        fake_progress = types.ModuleType("progress")
        fake_progress_bar = types.ModuleType("progress.bar")
        fake_progress_bar.ChargingBar = DummyBar

        fake_detector_module = types.ModuleType("applications.NsfwDetector")
        fake_detector_module.NsfwDetector = DummyDetector
        fake_srt_module = types.ModuleType("applications.SrtGenerator")
        fake_srt_module.SrtGenerator = DummySrtGenerator

        replacements = {
            "moviepy": fake_moviepy,
            "progress": fake_progress,
            "progress.bar": fake_progress_bar,
            "applications.NsfwDetector": fake_detector_module,
            "applications.SrtGenerator": fake_srt_module,
        }
        originals = {name: sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)

        def restore_modules() -> None:
            for name, original in originals.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        self.addCleanup(restore_modules)
        spec = importlib.util.spec_from_file_location(
            f"test_video_processor_impl_{id(self)}", MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _processor(self, module):
        temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp.close()
        self.addCleanup(lambda: Path(temp.name).unlink(missing_ok=True))
        return module.NsfwVideoProcessor(temp.name)

    def test_segment_generation_covers_tail(self) -> None:
        module = self._load_module()
        processor = self._processor(module)
        processor.clip_duration = 1.0
        segments = processor._build_segments(2.4)
        self.assertEqual(
            [segment["intervalo"] for segment in segments],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 2.4]],
        )

    def test_worker_defaults_are_safe_for_gpu_and_bounded_for_cpu(self) -> None:
        module = self._load_module()
        processor = self._processor(module)
        self.assertEqual(processor._resolve_workers(10, "cuda"), 1)
        self.assertGreaterEqual(processor._resolve_workers(10, "cpu"), 1)
        self.assertLessEqual(processor._resolve_workers(10, "cpu"), 4)

    def test_padding_marks_neighboring_segments(self) -> None:
        module = self._load_module()
        processor = self._processor(module)
        results = [
            {"orden": index + 1, "nsfw": index == 2}
            for index in range(6)
        ]
        marked = processor.mark_nsfw(results, rango=1)
        self.assertEqual(
            [item["nsfw"] for item in marked],
            [False, True, True, True, False, False],
        )

    def test_codec_auto_uses_nvenc_only_when_driver_and_encoder_exist(self) -> None:
        module = self._load_module()
        processor = self._processor(module)
        module._nvidia_driver_is_visible = lambda: True
        module._ffmpeg_encoders = lambda: {"h264_nvenc", "libx264"}
        self.assertEqual(processor._codec_candidates(), ["h264_nvenc", "libx264"])

        module._nvidia_driver_is_visible = lambda: False
        self.assertEqual(processor._codec_candidates(), ["libx264"])


if __name__ == "__main__":
    unittest.main()
