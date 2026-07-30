from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _rss_bytes() -> int | None:
    """Return current process RSS without adding a third-party dependency."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else None
        except Exception:
            return None

    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            pass

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS/BSD report bytes.
        return value if sys.platform == "darwin" else value * 1024
    except Exception:
        return None


def current_rss_bytes() -> int | None:
    return _rss_bytes()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"No se puede serializar {type(value).__name__} a JSON.")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class PerformanceProfiler:
    """Low-overhead, thread-safe profiler tailored to this video pipeline.

    It records atomic events rather than Python call stacks. This makes the JSON
    directly useful for finding queue stalls, slow frame reads, weak batches,
    worker imbalance and FFmpeg throughput while keeping overhead bounded.
    """

    schema_version = 1

    def __init__(
        self,
        output_path: str,
        *,
        enabled: bool = True,
        input_path: str | None = None,
    ) -> None:
        self.output_path = str(Path(output_path))
        self.enabled = bool(enabled)
        self.input_path = input_path
        self.run_id = uuid.uuid4().hex
        self.started_at_utc = _utc_now()
        self._wall_origin_ns = time.perf_counter_ns()
        self._cpu_origin_ns = time.process_time_ns()
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._progress_samples: list[dict[str, Any]] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._configuration: dict[str, Any] = {}
        self._artifacts: dict[str, Any] = {}
        self._errors: list[dict[str, Any]] = []
        self._overhead_ns = 0
        self._sequence = 0
        self._finalized = False
        self._initial_rss = _rss_bytes()

    def now_offset_seconds(self) -> float:
        return (time.perf_counter_ns() - self._wall_origin_ns) / 1_000_000_000

    def configure(self, **values: Any) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        with self._lock:
            self._configuration.update(values)
        self._overhead_ns += time.perf_counter_ns() - started

    def artifact(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        with self._lock:
            self._artifacts[name] = value
        self._overhead_ns += time.perf_counter_ns() - started

    def increment(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        with self._lock:
            self._counters[name] += float(value)
        self._overhead_ns += time.perf_counter_ns() - started

    def event(
        self,
        category: str,
        name: str,
        *,
        duration_seconds: float = 0.0,
        cpu_seconds: float | None = None,
        start_offset_seconds: float | None = None,
        **details: Any,
    ) -> None:
        if not self.enabled:
            return
        overhead_started = time.perf_counter_ns()
        end_offset = self.now_offset_seconds()
        duration = max(0.0, float(duration_seconds))
        item: dict[str, Any] = {
            "category": str(category),
            "name": str(name),
            "start_offset_seconds": (
                max(0.0, float(start_offset_seconds))
                if start_offset_seconds is not None
                else max(0.0, end_offset - duration)
            ),
            "end_offset_seconds": end_offset,
            "duration_seconds": duration,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "rss_bytes": _rss_bytes(),
        }
        if cpu_seconds is not None:
            item["cpu_seconds"] = max(0.0, float(cpu_seconds))
        if details:
            item["details"] = details
        with self._lock:
            self._sequence += 1
            item["sequence"] = self._sequence
            self._events.append(item)
        self._overhead_ns += time.perf_counter_ns() - overhead_started

    @contextlib.contextmanager
    def span(self, category: str, name: str, **details: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        offset = (wall_started - self._wall_origin_ns) / 1_000_000_000
        try:
            yield
        except BaseException as exc:
            self.error(category, name, exc)
            raise
        finally:
            self.event(
                category,
                name,
                duration_seconds=(time.perf_counter_ns() - wall_started) / 1_000_000_000,
                cpu_seconds=(time.process_time_ns() - cpu_started) / 1_000_000_000,
                start_offset_seconds=offset,
                **details,
            )

    def progress_sample(self, phase: str, **values: Any) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        item = {
            "phase": phase,
            "offset_seconds": self.now_offset_seconds(),
            **values,
        }
        with self._lock:
            self._sequence += 1
            item["sequence"] = self._sequence
            self._progress_samples.append(item)
        self._overhead_ns += time.perf_counter_ns() - started

    def error(self, category: str, name: str, error: BaseException) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        item = {
            "category": category,
            "name": name,
            "offset_seconds": self.now_offset_seconds(),
            "type": type(error).__name__,
            "message": str(error),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
        }
        with self._lock:
            self._sequence += 1
            item["sequence"] = self._sequence
            self._errors.append(item)
        self._overhead_ns += time.perf_counter_ns() - started

    def ingest_worker_event(self, event: dict[str, Any]) -> None:
        """Store timing generated inside a ProcessPool worker."""

        if not self.enabled:
            return
        started = time.perf_counter_ns()
        normalized = dict(event)
        normalized.setdefault("category", "worker")
        normalized.setdefault("name", "worker_event")
        normalized.setdefault("rss_bytes", None)
        normalized.setdefault("start_offset_seconds", None)
        normalized.setdefault("end_offset_seconds", None)
        with self._lock:
            self._sequence += 1
            normalized["sequence"] = self._sequence
            self._events.append(normalized)
        self._overhead_ns += time.perf_counter_ns() - started

    def _summary(self) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        cpu_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for event in self._events:
            key = (str(event.get("category")), str(event.get("name")))
            grouped[key].append(float(event.get("duration_seconds", 0.0)))
            if event.get("cpu_seconds") is not None:
                cpu_grouped[key].append(float(event["cpu_seconds"]))

        operations: list[dict[str, Any]] = []
        for key in sorted(grouped):
            values = grouped[key]
            cpu_values = cpu_grouped.get(key, [])
            total = sum(values)
            operations.append(
                {
                    "category": key[0],
                    "name": key[1],
                    "count": len(values),
                    "total_seconds": total,
                    "mean_seconds": statistics.fmean(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                    "p50_seconds": _percentile(values, 0.50),
                    "p95_seconds": _percentile(values, 0.95),
                    "p99_seconds": _percentile(values, 0.99),
                    "total_cpu_seconds": sum(cpu_values) if cpu_values else None,
                }
            )
        operations.sort(key=lambda item: item["total_seconds"], reverse=True)
        return {"operations_by_total_time": operations}

    def payload(self, *, status: str) -> dict[str, Any]:
        finished_offset = self.now_offset_seconds()
        current_rss = _rss_bytes()
        input_stat: dict[str, Any] | None = None
        if self.input_path:
            try:
                stat = Path(self.input_path).stat()
                input_stat = {"path": self.input_path, "size_bytes": stat.st_size}
            except OSError:
                input_stat = {"path": self.input_path, "size_bytes": None}

        process_cpu_seconds = (
            time.process_time_ns() - self._cpu_origin_ns
        ) / 1_000_000_000
        overhead_seconds = self._overhead_ns / 1_000_000_000
        return {
            "schema_version": self.schema_version,
            "profile_level": "atomic_pipeline",
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_now(),
            "total_wall_seconds": finished_offset,
            "process_cpu_seconds": process_cpu_seconds,
            "profiler_estimated_overhead_seconds": overhead_seconds,
            "derived_efficiency": {
                "average_process_cpu_equivalent_cores": (
                    process_cpu_seconds / finished_offset if finished_offset > 0 else None
                ),
                "profiler_estimated_overhead_percent": (
                    overhead_seconds / finished_offset * 100 if finished_offset > 0 else None
                ),
                "input_megabytes_per_wall_second": (
                    input_stat["size_bytes"] / 1_000_000 / finished_offset
                    if input_stat and input_stat.get("size_bytes") and finished_offset > 0
                    else None
                ),
            },
            "memory": {
                "initial_rss_bytes": self._initial_rss,
                "final_rss_bytes": current_rss,
                "rss_delta_bytes": (
                    current_rss - self._initial_rss
                    if current_rss is not None and self._initial_rss is not None
                    else None
                ),
            },
            "runtime": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "pid": os.getpid(),
            },
            "input": input_stat,
            "configuration": self._configuration,
            "artifacts": self._artifacts,
            "counters": dict(sorted(self._counters.items())),
            "summary": self._summary(),
            "events": self._events,
            "ffmpeg_progress_samples": self._progress_samples,
            "errors": self._errors,
        }

    def write(self, *, status: str = "completed") -> None:
        if not self.enabled:
            return
        payload = self.payload(status=status)
        target = Path(self.output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
                file.write("\n")
            os.replace(temporary_name, target)
            self._finalized = True
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
