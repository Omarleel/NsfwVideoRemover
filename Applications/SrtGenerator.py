from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value).__name__} a JSON.")


class SrtGenerator:
    def __init__(self) -> None:
        self.subtitles: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.subtitles.clear()

    def add_subtitle(self, start_time: float, end_time: float, text: Any) -> None:
        self.subtitles.append(
            {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "text": text,
            }
        )

    @staticmethod
    def format_time(time_seconds: float) -> str:
        total_milliseconds = max(0, int(round(float(time_seconds) * 1000)))
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def generate_srt(self, file_path: str) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                for index, subtitle in enumerate(self.subtitles, start=1):
                    start_time = self.format_time(subtitle["start_time"])
                    end_time = self.format_time(subtitle["end_time"])
                    text = json.dumps(
                        subtitle["text"], ensure_ascii=False, default=_json_default
                    )
                    file.write(f"{index}\n")
                    file.write(f"{start_time} --> {end_time}\n")
                    file.write(f"{text}\n\n")
            os.replace(temporary_name, target)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
