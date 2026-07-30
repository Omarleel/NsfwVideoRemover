from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from imageio_ffmpeg import get_ffmpeg_exe
from moviepy import VideoFileClip, concatenate_videoclips
from progress.bar import ChargingBar


def _subclip(video: VideoFileClip, start_time: float, end_time: float):
    return video.subclipped(start_time, end_time)


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


@dataclass(frozen=True)
class RenderResult:
    generated: bool
    codec: str | None
    reason: str


class VideoRenderer:
    """Owns video output concerns; it does not know how detections are produced."""

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        codec: str = "auto",
        fast_copy_when_unchanged: bool = True,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.codec = codec
        self.fast_copy_when_unchanged = bool(fast_copy_when_unchanged)

    def remove_stale_output(self) -> None:
        try:
            os.remove(self.output_path)
        except FileNotFoundError:
            pass

    def codec_candidates(self) -> list[str]:
        requested = self.codec.strip().lower()
        if requested != "auto":
            return [requested, "libx264"] if requested.endswith("_nvenc") else [requested]
        encoders = _ffmpeg_encoders()
        if _nvidia_driver_is_visible() and "h264_nvenc" in encoders:
            return ["h264_nvenc", "libx264"]
        return ["libx264"]

    def _try_fast_copy(self) -> bool:
        if not self.fast_copy_when_unchanged:
            return False
        self.remove_stale_output()
        command = [
            get_ffmpeg_exe(),
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
        if result.returncode == 0 and os.path.isfile(self.output_path):
            return True
        self.remove_stale_output()
        return False

    def _write_with_fallback(self, final_clip: Any) -> str:
        errors: list[str] = []
        for codec in self.codec_candidates():
            try:
                self.remove_stale_output()
                print(f"Codificando con {codec}...")
                final_clip.write_videofile(
                    self.output_path,
                    codec=codec,
                    audio_codec="aac",
                    threads=max(1, min(16, os.cpu_count() or 1)),
                    pixel_format="yuv420p",
                )
                return codec
            except Exception as exc:  # pragma: no cover - ffmpeg/hardware dependent
                errors.append(f"{codec}: {exc}")
                self.remove_stale_output()
                print(f"Falló {codec}; probando el siguiente codec disponible.")
        raise RuntimeError(
            "No se pudo generar el video. Errores de codificación:\n- "
            + "\n- ".join(errors)
        )

    def render(
        self,
        *,
        allowed_intervals: list[tuple[float, float]],
        cut_intervals: list[tuple[float, float]],
    ) -> RenderResult:
        if not cut_intervals:
            print("No se detectaron clases prohibidas; se conservará el video completo.")
            if self._try_fast_copy():
                print(f"Video guardado en {self.output_path} sin recodificación.")
                return RenderResult(True, "copy", "video sin cortes")
            print("No fue posible copiar los streams; se recodificará el video.")

        if not allowed_intervals:
            self.remove_stale_output()
            print("Todo el video quedó dentro de los cortes; no se generó salida.")
            return RenderResult(False, None, "todo el video fue eliminado")

        source_video = VideoFileClip(self.input_path)
        clips: list[Any] = []
        final_clip: Any | None = None
        bar = ChargingBar("Preparando clips permitidos", max=len(allowed_intervals))
        bar_finished = False
        try:
            for start_time, end_time in allowed_intervals:
                clips.append(_subclip(source_video, start_time, end_time))
                bar.next()
            bar.finish()
            bar_finished = True
            final_clip = concatenate_videoclips(clips, method="chain")
            used_codec = self._write_with_fallback(final_clip)
            print(f"Video guardado en {self.output_path} (codec: {used_codec})")
            return RenderResult(True, used_codec, "video renderizado")
        finally:
            if not bar_finished:
                bar.finish()
            if final_clip is not None:
                final_clip.close()
            for clip in clips:
                clip.close()
            source_video.close()
