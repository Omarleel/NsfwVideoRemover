from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


@dataclass(frozen=True)
class FfmpegCapabilities:
    executable: str
    hwaccels: frozenset[str]
    filters: frozenset[str]
    encoders: frozenset[str]
    probe_errors: tuple[str, ...] = ()

    @property
    def supports_cuda_decode(self) -> bool:
        return "cuda" in self.hwaccels

    @property
    def supports_cuda_scale(self) -> bool:
        return "scale_cuda" in self.filters

    @property
    def supports_nvdec_pipeline(self) -> bool:
        return self.supports_cuda_decode and self.supports_cuda_scale

    @property
    def supports_h264_nvenc(self) -> bool:
        return "h264_nvenc" in self.encoders

    @property
    def hardware_score(self) -> int:
        return (
            (4 if self.supports_h264_nvenc else 0)
            + (3 if self.supports_cuda_decode else 0)
            + (2 if self.supports_cuda_scale else 0)
        )


def _run_listing(executable: str, flag: str) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            [executable, "-hide_banner", flag],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{flag}: {type(exc).__name__}: {exc}"
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        return output, f"{flag}: FFmpeg devolvió {result.returncode}"
    return output, None


def _parse_hwaccels(text: str) -> frozenset[str]:
    values: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().casefold()
        if not line or line.startswith("hardware acceleration"):
            continue
        if " " not in line and line.replace("_", "").isalnum():
            values.add(line)
    return frozenset(values)


def _parse_filters(text: str) -> frozenset[str]:
    values: set[str] = set()
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if (
            len(parts) >= 2
            and 1 <= len(parts[0]) <= 4
            and set(parts[0]).issubset(set(".TSC"))
        ):
            values.add(parts[1].casefold())
    return frozenset(values)


def _parse_encoders(text: str) -> frozenset[str]:
    values: set[str] = set()
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 2 and len(parts[0]) <= 8 and parts[0][0:1] in {"V", "A", "S", "."}:
            values.add(parts[1].casefold())
    return frozenset(values)


@functools.lru_cache(maxsize=8)
def probe_ffmpeg_capabilities(executable: str) -> FfmpegCapabilities:
    resolved = str(Path(executable).expanduser())
    hwaccels_text, hwaccels_error = _run_listing(resolved, "-hwaccels")
    filters_text, filters_error = _run_listing(resolved, "-filters")
    encoders_text, encoders_error = _run_listing(resolved, "-encoders")
    errors = tuple(
        error for error in (hwaccels_error, filters_error, encoders_error) if error
    )
    return FfmpegCapabilities(
        executable=resolved,
        hwaccels=_parse_hwaccels(hwaccels_text),
        filters=_parse_filters(filters_text),
        encoders=_parse_encoders(encoders_text),
        probe_errors=errors,
    )


def _candidate_executables(explicit: str | None = None) -> list[str]:
    candidates: list[str] = []
    for value in (
        explicit,
        os.environ.get("NSFW_FFMPEG"),
        shutil.which("ffmpeg"),
        get_ffmpeg_exe(),
    ):
        if not value:
            continue
        normalized = str(Path(value).expanduser())
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def resolve_ffmpeg_executable(
    explicit: str | None = None,
    *,
    prefer_hardware: bool = True,
) -> tuple[str, FfmpegCapabilities]:
    if explicit:
        resolved = str(Path(explicit).expanduser())
        return resolved, probe_ffmpeg_capabilities(resolved)
    environment_override = os.environ.get("NSFW_FFMPEG")
    if environment_override:
        resolved = str(Path(environment_override).expanduser())
        return resolved, probe_ffmpeg_capabilities(resolved)
    candidates = _candidate_executables(None)
    if not candidates:
        fallback = get_ffmpeg_exe()
        return fallback, probe_ffmpeg_capabilities(fallback)

    inspected: list[tuple[int, int, str, FfmpegCapabilities]] = []
    for order, candidate in enumerate(candidates):
        capabilities = probe_ffmpeg_capabilities(candidate)
        score = capabilities.hardware_score if prefer_hardware else 0
        # Earlier candidates win ties, preserving explicit/environment preference.
        inspected.append((score, -order, candidate, capabilities))
    _score, _order, executable, capabilities = max(inspected, key=lambda item: (item[0], item[1]))
    return executable, capabilities
