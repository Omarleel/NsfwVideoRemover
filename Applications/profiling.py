from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - exercised only in incomplete installations.
    psutil = None

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional on CPU-only installations.
    pynvml = None

_NVML_LOCK = threading.Lock()
_NVML_INITIALIZED = False

_PSUTIL_PROCESS_CACHE: dict[int, Any] = {}
_PSUTIL_PROCESS_CACHE_LOCK = threading.Lock()

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _windows_rss_bytes(pid: int | None = None) -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
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
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        if pid is None or pid == os.getpid():
            handle = kernel32.GetCurrentProcess()
            close_handle = False
        else:
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid)
            )
            close_handle = True
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), ctypes.sizeof(counters)
            )
            return int(counters.WorkingSetSize) if ok else None
        finally:
            if close_handle:
                kernel32.CloseHandle(handle)
    except Exception:
        return None


def _rss_bytes(pid: int | None = None) -> int | None:
    """Return resident memory for a process, preferring psutil on every OS."""

    target_pid = os.getpid() if pid is None else int(pid)
    if psutil is not None:
        try:
            with _PSUTIL_PROCESS_CACHE_LOCK:
                process = _PSUTIL_PROCESS_CACHE.get(target_pid)
                if process is None:
                    process = psutil.Process(target_pid)
                    _PSUTIL_PROCESS_CACHE[target_pid] = process
            return int(process.memory_info().rss)
        except (psutil.Error, OSError):
            with _PSUTIL_PROCESS_CACHE_LOCK:
                _PSUTIL_PROCESS_CACHE.pop(target_pid, None)

    if os.name == "nt":
        return _windows_rss_bytes(target_pid)

    if target_pid == os.getpid() and sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            pass

    if target_pid == os.getpid():
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value * 1024
        except Exception:
            pass
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


def _nvidia_smi_path() -> str | None:
    candidates = [shutil.which("nvidia-smi")]
    if os.name == "nt":
        candidates.extend(
            [
                os.path.join(
                    os.environ.get("ProgramFiles", r"C:\Program Files"),
                    "NVIDIA Corporation",
                    "NVSMI",
                    "nvidia-smi.exe",
                ),
                os.path.join(
                    os.environ.get("SystemRoot", r"C:\Windows"),
                    "System32",
                    "nvidia-smi.exe",
                ),
            ]
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def _parse_number(value: str, *, multiplier: float = 1.0) -> float | None:
    normalized = value.strip().replace("MiB", "").replace("W", "")
    if normalized.casefold() in {"", "n/a", "[not supported]", "-"}:
        return None
    try:
        return float(normalized) * multiplier
    except ValueError:
        return None


def _run_nvidia_csv(executable: str, query: str) -> list[list[str]]:
    try:
        result = subprocess.run(
            [executable, query, "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [row for row in csv.reader(io.StringIO(result.stdout)) if row]


def _gpu_snapshot_nvml() -> dict[str, Any] | None:
    global _NVML_INITIALIZED
    if pynvml is None:
        return None
    try:
        with _NVML_LOCK:
            if not _NVML_INITIALIZED:
                pynvml.nvmlInit()
                _NVML_INITIALIZED = True
            count = pynvml.nvmlDeviceGetCount()
            gpus: list[dict[str, Any]] = []
            processes_by_pid: dict[int, dict[str, Any]] = {}
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                uuid_value = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                if isinstance(uuid_value, bytes):
                    uuid_value = uuid_value.decode("utf-8", errors="replace")
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu: dict[str, Any] = {
                    "index": index,
                    "name": str(name),
                    "uuid": str(uuid_value),
                    "gpu_utilization_percent": float(utilization.gpu),
                    "memory_controller_utilization_percent": float(utilization.memory),
                    "vram_used_bytes": int(memory.used),
                    "vram_total_bytes": int(memory.total),
                }
                try:
                    gpu["temperature_celsius"] = float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except pynvml.NVMLError:
                    gpu["temperature_celsius"] = None
                try:
                    gpu["power_watts"] = float(
                        pynvml.nvmlDeviceGetPowerUsage(handle)
                    ) / 1000.0
                except pynvml.NVMLError:
                    gpu["power_watts"] = None
                try:
                    gpu["encoder_utilization_percent"] = float(
                        pynvml.nvmlDeviceGetEncoderUtilization(handle)[0]
                    )
                except pynvml.NVMLError:
                    gpu["encoder_utilization_percent"] = None
                try:
                    gpu["decoder_utilization_percent"] = float(
                        pynvml.nvmlDeviceGetDecoderUtilization(handle)[0]
                    )
                except pynvml.NVMLError:
                    gpu["decoder_utilization_percent"] = None
                gpus.append(gpu)

                getters = []
                for base_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    for suffix in ("_v3", "_v2", ""):
                        getter = getattr(pynvml, base_name + suffix, None)
                        if getter is not None:
                            getters.append(getter)
                            break
                for getter in getters:
                    try:
                        running = getter(handle)
                    except pynvml.NVMLError:
                        continue
                    for process in running:
                        pid = int(process.pid)
                        used = getattr(process, "usedGpuMemory", None)
                        if used in {None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", -1)}:
                            used_value = None
                        else:
                            used_value = int(used)
                        item = processes_by_pid.setdefault(
                            pid,
                            {
                                "pid": pid,
                                "process_name": None,
                                "vram_used_bytes": 0,
                                "gpu_uuid": str(uuid_value),
                                "gpu_index": index,
                            },
                        )
                        if used_value is not None:
                            item["vram_used_bytes"] = max(
                                int(item.get("vram_used_bytes") or 0), used_value
                            )
            return {"gpus": gpus, "processes": list(processes_by_pid.values())}
    except Exception:
        return None


def _gpu_snapshot(executable: str | None) -> dict[str, Any] | None:
    nvml_snapshot = _gpu_snapshot_nvml()
    if nvml_snapshot is not None:
        return nvml_snapshot
    if not executable:
        return None
    fields = [
        "index",
        "name",
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "utilization.encoder",
        "utilization.decoder",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    rows = _run_nvidia_csv(executable, "--query-gpu=" + ",".join(fields))
    if not rows:
        # Older drivers may not expose encoder/decoder utilization.
        fields = [
            "index",
            "name",
            "uuid",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
        ]
        rows = _run_nvidia_csv(executable, "--query-gpu=" + ",".join(fields))
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(fields):
            continue
        values = {field: value.strip() for field, value in zip(fields, row)}
        gpu: dict[str, Any] = {
            "index": int(values["index"]) if values["index"].isdigit() else values["index"],
            "name": values["name"],
            "uuid": values["uuid"],
            "gpu_utilization_percent": _parse_number(values["utilization.gpu"]),
            "memory_controller_utilization_percent": _parse_number(
                values["utilization.memory"]
            ),
            "vram_used_bytes": (
                int(number)
                if (number := _parse_number(values["memory.used"], multiplier=1024 * 1024))
                is not None
                else None
            ),
            "vram_total_bytes": (
                int(number)
                if (number := _parse_number(values["memory.total"], multiplier=1024 * 1024))
                is not None
                else None
            ),
            "temperature_celsius": _parse_number(values["temperature.gpu"]),
            "power_watts": _parse_number(values["power.draw"]),
        }
        if "utilization.encoder" in values:
            gpu["encoder_utilization_percent"] = _parse_number(
                values["utilization.encoder"]
            )
        if "utilization.decoder" in values:
            gpu["decoder_utilization_percent"] = _parse_number(
                values["utilization.decoder"]
            )
        gpus.append(gpu)

    process_rows = _run_nvidia_csv(
        executable,
        "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
    )
    processes: list[dict[str, Any]] = []
    for row in process_rows:
        if len(row) != 4:
            continue
        pid_raw, process_name, used_memory, gpu_uuid = [value.strip() for value in row]
        memory = _parse_number(used_memory, multiplier=1024 * 1024)
        processes.append(
            {
                "pid": int(pid_raw) if pid_raw.isdigit() else pid_raw,
                "process_name": process_name,
                "vram_used_bytes": int(memory) if memory is not None else None,
                "gpu_uuid": gpu_uuid,
            }
        )

    # pmon captures graphics/encoding/decoding processes that compute-apps may omit.
    pmon_by_pid: dict[int, dict[str, Any]] = {}
    try:
        result = subprocess.run(
            [executable, "pmon", "-c", "1", "-s", "um"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
        if result.returncode == 0:
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8 or not parts[1].isdigit():
                    continue
                pid = int(parts[1])
                pmon_by_pid[pid] = {
                    "gpu_index": int(parts[0]) if parts[0].isdigit() else parts[0],
                    "process_type": parts[2],
                    "sm_utilization_percent": _parse_number(parts[3]),
                    "memory_utilization_percent": _parse_number(parts[4]),
                    "encoder_utilization_percent": _parse_number(parts[5]),
                    "decoder_utilization_percent": _parse_number(parts[6]),
                    "process_name": " ".join(parts[7:]),
                }
    except (OSError, subprocess.SubprocessError):
        pass

    known_pids = {item.get("pid") for item in processes}
    for item in processes:
        pid = item.get("pid")
        if isinstance(pid, int) and pid in pmon_by_pid:
            item.update(pmon_by_pid[pid])
    for pid, item in pmon_by_pid.items():
        if pid not in known_pids:
            processes.append({"pid": pid, "vram_used_bytes": None, **item})

    return {"gpus": gpus, "processes": processes}


class PerformanceProfiler:
    """Atomic pipeline profiler plus process-tree, RAM and NVIDIA telemetry."""

    schema_version = 2

    def __init__(
        self,
        output_path: str,
        *,
        enabled: bool = True,
        input_path: str | None = None,
        system_sample_interval_seconds: float = 1.0,
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
        self._system_samples: list[dict[str, Any]] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._configuration: dict[str, Any] = {}
        self._artifacts: dict[str, Any] = {}
        self._errors: list[dict[str, Any]] = []
        self._registered_processes: dict[int, dict[str, Any]] = {
            os.getpid(): {"role": "python_main", "registered_at_offset_seconds": 0.0}
        }
        self._last_process_cpu: dict[int, tuple[float, float]] = {}
        self._overhead_ns = 0
        self._sampler_overhead_ns = 0
        self._sequence = 0
        self._finalized = False
        self._initial_rss = _rss_bytes()
        self._nvidia_smi = _nvidia_smi_path()
        self._last_gpu_sample_offset = -1e9
        self._system_sample_interval = max(0.25, float(system_sample_interval_seconds))
        self._sampler_stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None
        if self.enabled:
            self.capture_system_sample(reason="profiler_start")
            self._sampler_thread = threading.Thread(
                target=self._system_sampler_loop,
                name="performance-system-sampler",
                daemon=True,
            )
            self._sampler_thread.start()

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

    def register_child_process(
        self,
        pid: int,
        *,
        role: str,
        command: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        item: dict[str, Any] = {
            "role": str(role),
            "registered_at_offset_seconds": self.now_offset_seconds(),
        }
        if command:
            item["command"] = list(command)
        with self._lock:
            self._registered_processes[int(pid)] = item
        self._overhead_ns += time.perf_counter_ns() - started
        self.capture_system_sample(reason=f"process_registered:{role}")

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
        item = {"phase": phase, "offset_seconds": self.now_offset_seconds(), **values}
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
        if not self.enabled:
            return
        started = time.perf_counter_ns()
        normalized = dict(event)
        normalized.setdefault("category", "worker")
        normalized.setdefault("name", "worker_event")
        pid = normalized.get("pid")
        if normalized.get("rss_bytes") is None and isinstance(pid, int):
            normalized["rss_bytes"] = _rss_bytes(pid)
        normalized.setdefault("start_offset_seconds", None)
        normalized.setdefault("end_offset_seconds", None)
        with self._lock:
            self._sequence += 1
            normalized["sequence"] = self._sequence
            self._events.append(normalized)
            if isinstance(pid, int):
                self._registered_processes.setdefault(
                    pid,
                    {
                        "role": "inference_worker",
                        "registered_at_offset_seconds": self.now_offset_seconds(),
                    },
                )
        self._overhead_ns += time.perf_counter_ns() - started

    def _system_sampler_loop(self) -> None:
        while not self._sampler_stop.wait(self._system_sample_interval):
            self.capture_system_sample(reason="periodic")

    def _process_tree(self) -> list[Any]:
        if psutil is None:
            return []
        processes: dict[int, Any] = {}
        try:
            root = psutil.Process(os.getpid())
            processes[root.pid] = root
            for child in root.children(recursive=True):
                processes[child.pid] = child
        except psutil.Error:
            pass
        with self._lock:
            registered_pids = tuple(self._registered_processes)
        for pid in registered_pids:
            if pid in processes:
                continue
            try:
                processes[pid] = psutil.Process(pid)
            except psutil.Error:
                pass
        return list(processes.values())

    def capture_system_sample(self, *, reason: str = "manual") -> None:
        if not self.enabled:
            return
        started_ns = time.perf_counter_ns()
        offset = self.now_offset_seconds()
        sample: dict[str, Any] = {
            "offset_seconds": offset,
            "reason": reason,
            "root_pid": os.getpid(),
        }
        process_items: list[dict[str, Any]] = []
        aggregate_rss = 0
        aggregate_vms = 0
        aggregate_uss = 0
        aggregate_cpu_percent = 0.0
        aggregate_read_bytes = 0
        aggregate_write_bytes = 0
        now = time.perf_counter()

        if psutil is not None:
            try:
                virtual = psutil.virtual_memory()
                sample["system_memory"] = {
                    "total_bytes": int(virtual.total),
                    "available_bytes": int(virtual.available),
                    "used_bytes": int(virtual.used),
                    "percent": float(virtual.percent),
                }
                swap = psutil.swap_memory()
                sample["swap_memory"] = {
                    "total_bytes": int(swap.total),
                    "used_bytes": int(swap.used),
                    "percent": float(swap.percent),
                }
                sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            except psutil.Error:
                pass

            with self._lock:
                registrations = dict(self._registered_processes)
            for process in self._process_tree():
                try:
                    with process.oneshot():
                        memory = process.memory_info()
                        cpu_times = process.cpu_times()
                        cpu_total = float(cpu_times.user + cpu_times.system)
                        previous = self._last_process_cpu.get(process.pid)
                        cpu_percent = None
                        if previous is not None:
                            previous_time, previous_cpu = previous
                            elapsed = now - previous_time
                            if elapsed > 0:
                                cpu_percent = max(
                                    0.0, (cpu_total - previous_cpu) / elapsed * 100.0
                                )
                        self._last_process_cpu[process.pid] = (now, cpu_total)
                        try:
                            full_memory = process.memory_full_info()
                            uss = int(getattr(full_memory, "uss", 0) or 0)
                            private = int(
                                getattr(full_memory, "private", 0)
                                or getattr(full_memory, "private_usage", 0)
                                or 0
                            )
                        except (psutil.Error, AttributeError):
                            uss = 0
                            private = 0
                        try:
                            io_counters = process.io_counters()
                            read_bytes = int(io_counters.read_bytes)
                            write_bytes = int(io_counters.write_bytes)
                        except (psutil.Error, AttributeError):
                            read_bytes = 0
                            write_bytes = 0
                        registration = registrations.get(process.pid, {})
                        item = {
                            "pid": process.pid,
                            "ppid": process.ppid(),
                            "name": process.name(),
                            "role": registration.get("role", "child_process"),
                            "status": process.status(),
                            "rss_bytes": int(memory.rss),
                            "vms_bytes": int(memory.vms),
                            "uss_bytes": uss or None,
                            "private_bytes": private or None,
                            "cpu_percent_of_one_core": cpu_percent,
                            "cpu_time_seconds": cpu_total,
                            "thread_count": process.num_threads(),
                            "read_bytes": read_bytes,
                            "write_bytes": write_bytes,
                        }
                        process_items.append(item)
                        aggregate_rss += int(memory.rss)
                        aggregate_vms += int(memory.vms)
                        aggregate_uss += uss
                        aggregate_cpu_percent += cpu_percent or 0.0
                        aggregate_read_bytes += read_bytes
                        aggregate_write_bytes += write_bytes
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        else:
            rss = _rss_bytes()
            process_items.append(
                {
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "name": Path(sys.executable).name,
                    "role": "python_main",
                    "rss_bytes": rss,
                }
            )
            aggregate_rss = rss or 0

        sample["process_tree"] = process_items
        sample["process_tree_summary"] = {
            "process_count": len(process_items),
            "rss_bytes": aggregate_rss,
            "vms_bytes": aggregate_vms or None,
            "uss_bytes": aggregate_uss or None,
            "cpu_percent_of_one_core": aggregate_cpu_percent,
            "read_bytes": aggregate_read_bytes,
            "write_bytes": aggregate_write_bytes,
        }
        gpu_interval = 1.0 if pynvml is not None else 2.0
        gpu = None
        if offset - self._last_gpu_sample_offset >= gpu_interval:
            gpu = _gpu_snapshot(self._nvidia_smi)
            self._last_gpu_sample_offset = offset
        if gpu is not None:
            with self._lock:
                roles = {
                    pid: info.get("role")
                    for pid, info in self._registered_processes.items()
                }
            process_names = {
                item.get("pid"): item.get("name") for item in process_items
            }
            for process in gpu.get("processes", []):
                pid = process.get("pid")
                if isinstance(pid, int):
                    process["role"] = roles.get(pid, "gpu_process")
                    if not process.get("process_name"):
                        process["process_name"] = process_names.get(pid)
            sample["nvidia"] = gpu

        with self._lock:
            self._sequence += 1
            sample["sequence"] = self._sequence
            self._system_samples.append(sample)
        self._sampler_overhead_ns += time.perf_counter_ns() - started_ns

    def stop_system_sampler(self) -> None:
        if not self.enabled:
            return
        thread = self._sampler_thread
        if thread is None:
            return
        self._sampler_stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self._system_sample_interval * 2))
        self._sampler_thread = None
        self.capture_system_sample(reason="profiler_end")

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
        return {
            "operations_by_total_time": operations,
            "system_resources": self._resource_summary(),
        }

    def _resource_summary(self) -> dict[str, Any]:
        tree_rss: list[float] = []
        root_rss: list[float] = []
        cpu_values: list[float] = []
        available_ram: list[float] = []
        gpu_metrics: dict[str, list[float]] = defaultdict(list)
        process_peaks: dict[int, dict[str, Any]] = {}
        gpu_process_peaks: dict[int, dict[str, Any]] = {}

        for sample in self._system_samples:
            tree_summary = sample.get("process_tree_summary", {})
            if tree_summary.get("rss_bytes") is not None:
                tree_rss.append(float(tree_summary["rss_bytes"]))
            if tree_summary.get("cpu_percent_of_one_core") is not None:
                cpu_values.append(float(tree_summary["cpu_percent_of_one_core"]))
            memory = sample.get("system_memory", {})
            if memory.get("available_bytes") is not None:
                available_ram.append(float(memory["available_bytes"]))
            for process in sample.get("process_tree", []):
                pid = process.get("pid")
                if not isinstance(pid, int):
                    continue
                peak = process_peaks.setdefault(
                    pid,
                    {
                        "pid": pid,
                        "name": process.get("name"),
                        "role": process.get("role"),
                        "peak_rss_bytes": 0,
                        "peak_uss_bytes": 0,
                        "max_cpu_percent_of_one_core": 0.0,
                        "first_observed_cpu_time_seconds": None,
                        "last_observed_cpu_time_seconds": None,
                        "observed_cpu_seconds": 0.0,
                    },
                )
                rss = process.get("rss_bytes") or 0
                uss = process.get("uss_bytes") or 0
                cpu = process.get("cpu_percent_of_one_core") or 0.0
                cpu_time = process.get("cpu_time_seconds") or 0.0
                peak["peak_rss_bytes"] = max(peak["peak_rss_bytes"], int(rss))
                peak["peak_uss_bytes"] = max(peak["peak_uss_bytes"], int(uss))
                peak["max_cpu_percent_of_one_core"] = max(
                    peak["max_cpu_percent_of_one_core"], float(cpu)
                )
                if peak["first_observed_cpu_time_seconds"] is None:
                    peak["first_observed_cpu_time_seconds"] = float(cpu_time)
                peak["last_observed_cpu_time_seconds"] = max(
                    float(peak["last_observed_cpu_time_seconds"] or 0.0),
                    float(cpu_time),
                )
                peak["observed_cpu_seconds"] = max(
                    0.0,
                    float(peak["last_observed_cpu_time_seconds"] or 0.0)
                    - float(peak["first_observed_cpu_time_seconds"] or 0.0),
                )
                if pid == os.getpid() and rss:
                    root_rss.append(float(rss))
            nvidia = sample.get("nvidia", {})
            for gpu in nvidia.get("gpus", []):
                for field in (
                    "gpu_utilization_percent",
                    "memory_controller_utilization_percent",
                    "encoder_utilization_percent",
                    "decoder_utilization_percent",
                    "vram_used_bytes",
                    "temperature_celsius",
                    "power_watts",
                ):
                    value = gpu.get(field)
                    if value is not None:
                        gpu_metrics[field].append(float(value))
            for process in nvidia.get("processes", []):
                pid = process.get("pid")
                if not isinstance(pid, int):
                    continue
                peak = gpu_process_peaks.setdefault(
                    pid,
                    {
                        "pid": pid,
                        "process_name": process.get("process_name"),
                        "role": process.get("role"),
                        "peak_vram_used_bytes": 0,
                        "max_sm_utilization_percent": 0.0,
                        "max_encoder_utilization_percent": 0.0,
                        "max_decoder_utilization_percent": 0.0,
                    },
                )
                peak["peak_vram_used_bytes"] = max(
                    peak["peak_vram_used_bytes"], int(process.get("vram_used_bytes") or 0)
                )
                for source, destination in (
                    ("sm_utilization_percent", "max_sm_utilization_percent"),
                    ("encoder_utilization_percent", "max_encoder_utilization_percent"),
                    ("decoder_utilization_percent", "max_decoder_utilization_percent"),
                ):
                    peak[destination] = max(
                        peak[destination], float(process.get(source) or 0.0)
                    )

        gpu_summary = {
            field: {
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
            for field, values in gpu_metrics.items()
            if values
        }
        return {
            "sample_count": len(self._system_samples),
            "process_tree_peak_rss_bytes": int(max(tree_rss)) if tree_rss else None,
            "python_main_peak_rss_bytes": int(max(root_rss)) if root_rss else None,
            "minimum_available_system_ram_bytes": (
                int(min(available_ram)) if available_ram else None
            ),
            "process_tree_cpu_percent_of_one_core": {
                "mean": statistics.fmean(cpu_values) if cpu_values else None,
                "p95": _percentile(cpu_values, 0.95),
                "max": max(cpu_values) if cpu_values else None,
            },
            "process_peaks": sorted(process_peaks.values(), key=lambda item: item["pid"]),
            "gpu": gpu_summary,
            "gpu_process_peaks": sorted(
                gpu_process_peaks.values(), key=lambda item: item["pid"]
            ),
        }

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
        direct_overhead_seconds = self._overhead_ns / 1_000_000_000
        sampler_overhead_seconds = self._sampler_overhead_ns / 1_000_000_000
        overhead_seconds = direct_overhead_seconds + sampler_overhead_seconds
        resource_summary = self._resource_summary()
        process_tree_cpu_seconds = sum(
            float(item.get("observed_cpu_seconds") or 0.0)
            for item in resource_summary.get("process_peaks", [])
        )
        return {
            "schema_version": self.schema_version,
            "profile_level": "atomic_pipeline_and_system",
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_now(),
            "total_wall_seconds": finished_offset,
            "process_cpu_seconds": process_cpu_seconds,
            "observed_process_tree_cpu_seconds": process_tree_cpu_seconds,
            "profiler_estimated_overhead_seconds": overhead_seconds,
            "profiler_overhead_breakdown": {
                "atomic_events_seconds": direct_overhead_seconds,
                "system_sampler_seconds": sampler_overhead_seconds,
            },
            "derived_efficiency": {
                "average_python_process_cpu_equivalent_cores": (
                    process_cpu_seconds / finished_offset if finished_offset > 0 else None
                ),
                "average_observed_process_tree_cpu_equivalent_cores": (
                    process_tree_cpu_seconds / finished_offset if finished_offset > 0 else None
                ),
                # Kept for compatibility; explicitly refers to Python only.
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
                "python_main_peak_rss_bytes": resource_summary.get(
                    "python_main_peak_rss_bytes"
                ),
                "process_tree_peak_rss_bytes": resource_summary.get(
                    "process_tree_peak_rss_bytes"
                ),
                "minimum_available_system_ram_bytes": resource_summary.get(
                    "minimum_available_system_ram_bytes"
                ),
            },
            "runtime": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "pid": os.getpid(),
                "psutil_available": psutil is not None,
                "nvidia_smi_path": self._nvidia_smi,
                "pynvml_available": pynvml is not None,
                "system_sample_interval_seconds": self._system_sample_interval,
            },
            "input": input_stat,
            "configuration": self._configuration,
            "artifacts": self._artifacts,
            "counters": dict(sorted(self._counters.items())),
            "registered_processes": {
                str(pid): data for pid, data in sorted(self._registered_processes.items())
            },
            "summary": self._summary(),
            "events": self._events,
            "ffmpeg_progress_samples": self._progress_samples,
            "system_resource_samples": self._system_samples,
            "errors": self._errors,
        }

    def write(self, *, status: str = "completed") -> None:
        if not self.enabled:
            return
        self.stop_system_sampler()
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

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        try:
            self._sampler_stop.set()
        except Exception:
            pass
