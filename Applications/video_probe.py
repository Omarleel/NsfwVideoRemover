from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from imageio_ffmpeg import read_frames


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str | None
    audio_codec: str | None

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_codec)


class VideoProbe(Protocol):
    def probe(self, video_path: str) -> VideoInfo:
        """Read container metadata without decoding the complete video."""


class ImageioFfmpegVideoProbe:
    """Small FFmpeg-backed metadata probe with no MoviePy dependency."""

    def probe(self, video_path: str) -> VideoInfo:
        reader = read_frames(video_path)
        try:
            metadata = next(reader)
        finally:
            reader.close()

        source_size = metadata.get("source_size") or metadata.get("size")
        if not source_size or len(source_size) != 2:
            raise RuntimeError("FFmpeg no pudo determinar la resolución del video.")

        duration = float(metadata.get("duration") or 0.0)
        if duration <= 0:
            raise RuntimeError("FFmpeg no pudo determinar la duración del video.")

        return VideoInfo(
            duration=duration,
            width=int(source_size[0]),
            height=int(source_size[1]),
            fps=float(metadata.get("fps") or 0.0),
            video_codec=(str(metadata["codec"]) if metadata.get("codec") else None),
            audio_codec=(
                str(metadata["audio_codec"]) if metadata.get("audio_codec") else None
            ),
        )
