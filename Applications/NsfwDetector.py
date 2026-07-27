from __future__ import annotations

import contextlib
import ctypes
import os
import site
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
from nudenet import NudeDetector


ProviderSpec = str | tuple[str, dict[str, Any]]

# Windows removes a directory from its DLL search path when the object returned
# by os.add_dll_directory() is closed or garbage-collected. Keep both the
# directory cookies and explicitly loaded DLLs alive for the whole process.
_DLL_DIRECTORY_HANDLES: list[Any] = []
_PRELOADED_DLL_HANDLES: list[Any] = []
_REGISTERED_DLL_DIRECTORIES: set[str] = set()


@contextlib.contextmanager
def _inject_onnx_providers(
    providers: Sequence[ProviderSpec],
    session_options: Any | None = None,
):
    """Inject providers/session options into NudeNet 3.4.2.

    That NudeNet release accepts ``providers`` but does not forward them to
    ONNX Runtime. The same interception lets each CPU worker receive a bounded
    thread budget, avoiding severe oversubscription when several sessions run.
    """

    original_inference_session = ort.InferenceSession

    def inference_session_with_providers(*args: Any, **kwargs: Any):
        kwargs.setdefault("providers", list(providers))
        if session_options is not None:
            kwargs.setdefault("sess_options", session_options)
        return original_inference_session(*args, **kwargs)

    ort.InferenceSession = inference_session_with_providers  # type: ignore[assignment]
    try:
        yield
    finally:
        ort.InferenceSession = original_inference_session  # type: ignore[assignment]


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    values: list[str] = []

    try:
        values.extend(site.getsitepackages())
    except AttributeError:  # pragma: no cover - unusual embedded Python
        pass

    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            values.append(user_site)
        else:
            values.extend(user_site)
    except (AttributeError, TypeError):  # pragma: no cover
        pass

    values.extend(value for value in sys.path if isinstance(value, str))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        values.append(str(Path(conda_prefix) / "Lib" / "site-packages"))

    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        try:
            key = str(path.resolve()).casefold()
        except OSError:
            key = str(path).casefold()
        if key not in seen and path.is_dir():
            seen.add(key)
            roots.append(path)
    return roots


def _windows_nvidia_dll_directories() -> list[Path]:
    """Find pip/Conda CUDA and cuDNN DLL folders in priority order."""

    if os.name != "nt":
        return []

    candidates: list[Path] = []

    # Prefer the NVIDIA wheels installed in this exact Python environment.
    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(
                path for path in nvidia_root.glob("*/bin") if path.is_dir()
            )

    # Then consider Conda and a system CUDA Toolkit installation.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "bin")
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_dir():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _register_windows_dll_directories() -> list[Path]:
    """Keep NVIDIA DLL directories available to cuDNN sub-library loading."""

    directories = _windows_nvidia_dll_directories()
    if not directories:
        return []

    add_dll_directory = getattr(os, "add_dll_directory", None)
    for directory in directories:
        key = str(directory).casefold()
        if key in _REGISTERED_DLL_DIRECTORIES:
            continue
        if callable(add_dll_directory):
            try:
                _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(directory)))
            except OSError:
                pass
        _REGISTERED_DLL_DIRECTORIES.add(key)

    # cuDNN 9 loads some engine DLLs dynamically at inference time. Keeping the
    # same directories at the front of PATH also covers those internal loads.
    current_path = os.environ.get("PATH", "")
    existing = {part.casefold() for part in current_path.split(os.pathsep) if part}
    additions = [str(path) for path in directories if str(path).casefold() not in existing]
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + [current_path])

    return directories


class NsfwDetector:
    """NudeNet wrapper with verified CUDA execution and automatic CPU fallback."""

    VALID_DEVICES = {"auto", "cuda", "cpu"}
    EXCLUDED_CLASSES = (
        "FACE_FEMALE",
        "FACE_MALE",
        "ARMPITS_EXPOSED",
        "ARMPITS_COVERED",
        "FEET_EXPOSED",
        "FEET_COVERED",
    )

    def __init__(
        self,
        umbral_minimo_expuesto: float,
        umbral_minimo_cubierto: float,
        device: str = "auto",
        intra_op_threads: int = 0,
    ) -> None:
        normalized_device = device.strip().lower()
        if normalized_device not in self.VALID_DEVICES:
            raise ValueError(
                f"Dispositivo no válido: {device!r}. Usa auto, cuda o cpu."
            )

        if intra_op_threads < 0:
            raise ValueError("intra_op_threads no puede ser negativo.")

        self.umbral_minimo_expuesto = float(umbral_minimo_expuesto)
        self.umbral_minimo_cubierto = float(umbral_minimo_cubierto)
        self.requested_device = normalized_device
        self.intra_op_threads = int(intra_op_threads)
        self.available_providers = list(ort.get_available_providers())
        self.fallback_reason: str | None = None

        self.nude_detector = self._create_detector()
        session = getattr(self.nude_detector, "onnx_session", None)
        self.active_providers = (
            list(session.get_providers()) if session is not None else []
        )
        self.device = (
            "cuda" if "CUDAExecutionProvider" in self.active_providers else "cpu"
        )

    @staticmethod
    def _preload_cuda_libraries() -> None:
        """Load pip/Conda CUDA and every required cuDNN 9 sub-library."""

        directories = _register_windows_dll_directories()
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                # Empty directory tells ORT to search NVIDIA packages in site-packages.
                preload(directory="")
            except TypeError:
                preload()
            except Exception as exc:  # pragma: no cover - hardware dependent
                warnings.warn(
                    f"No se pudieron precargar las bibliotecas CUDA/cuDNN: {exc}",
                    RuntimeWarning,
                )

        # cuDNN 9 is split into sub-libraries. Some versions load this engine at
        # first convolution rather than while the ONNX session is constructed.
        if os.name == "nt":
            target = "cudnn_engines_tensor_ir64_9.dll"
            for directory in directories:
                dll_path = directory / target
                if not dll_path.is_file():
                    continue
                try:
                    _PRELOADED_DLL_HANDLES.append(ctypes.WinDLL(str(dll_path)))
                except OSError as exc:  # pragma: no cover - hardware dependent
                    warnings.warn(
                        f"No se pudo cargar {dll_path.name}: {exc}", RuntimeWarning
                    )
                break

    def _session_options(self) -> Any | None:
        options_class = getattr(ort, "SessionOptions", None)
        if options_class is None:
            return None
        options = options_class()
        if self.intra_op_threads > 0:
            options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = 1

        execution_mode = getattr(ort, "ExecutionMode", None)
        if execution_mode is not None and hasattr(execution_mode, "ORT_SEQUENTIAL"):
            options.execution_mode = execution_mode.ORT_SEQUENTIAL
        graph_level = getattr(ort, "GraphOptimizationLevel", None)
        if graph_level is not None and hasattr(graph_level, "ORT_ENABLE_ALL"):
            options.graph_optimization_level = graph_level.ORT_ENABLE_ALL
        return options

    def _new_nude_detector(self, providers: Sequence[ProviderSpec]) -> NudeDetector:
        with _inject_onnx_providers(providers, self._session_options()):
            return NudeDetector(providers=list(providers))

    @staticmethod
    def _verify_cuda_inference(detector: NudeDetector) -> list[str]:
        """Run one real convolution so a merely-created CUDA session is not enough."""

        session = getattr(detector, "onnx_session", None)
        disable_fallback = getattr(session, "disable_fallback", None)
        enable_fallback = getattr(session, "enable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()
        try:
            detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        finally:
            if callable(enable_fallback):
                enable_fallback()
        return list(session.get_providers()) if session is not None else []

    def _create_detector(self) -> NudeDetector:
        cpu_providers: list[ProviderSpec] = ["CPUExecutionProvider"]

        if self.requested_device == "cpu":
            return self._new_nude_detector(cpu_providers)

        cuda_is_listed = "CUDAExecutionProvider" in self.available_providers
        if cuda_is_listed:
            self._preload_cuda_libraries()
            cuda_providers: list[ProviderSpec] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "do_copy_in_default_stream": True,
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                ),
                "CPUExecutionProvider",
            ]
            try:
                detector = self._new_nude_detector(cuda_providers)
                active = self._verify_cuda_inference(detector)
                if "CUDAExecutionProvider" in active:
                    return detector
                self.fallback_reason = (
                    "CUDA creó la sesión, pero la prueba real terminó usando CPU."
                )
            except Exception as exc:  # pragma: no cover - hardware dependent
                self.fallback_reason = f"CUDA no pudo inicializarse o ejecutar la prueba: {exc}"
        else:
            self.fallback_reason = (
                "CUDAExecutionProvider no está disponible en este entorno."
            )

        if self.requested_device == "cuda":
            warnings.warn(
                f"Se solicitó CUDA, pero se usará CPU. {self.fallback_reason}",
                RuntimeWarning,
            )

        return self._new_nude_detector(cpu_providers)

    def provider_summary(self) -> str:
        active = ", ".join(self.active_providers) or "desconocido"
        summary = f"dispositivo={self.device}; proveedores activos={active}"
        if self.fallback_reason:
            summary += f"; fallback={self.fallback_reason}"
        return summary

    def _classify_detections(self, detections: list[dict[str, Any]]):
        sumatoria_probabilidad_expuesto = 0.0
        contador_probabilidad_expuesto = 0
        sumatoria_probabilidad_cubierto = 0.0
        contador_probabilidad_cubierto = 0
        for detection in detections:
            detection_class = str(detection.get("class", ""))
            score = float(detection.get("score", 0.0))
            if any(excluded in detection_class for excluded in self.EXCLUDED_CLASSES):
                continue

            if "EXPOSED" in detection_class:
                sumatoria_probabilidad_expuesto += (
                    score / 2 if detection_class == "BELLY_EXPOSED" else score
                )
                contador_probabilidad_expuesto += 1
            elif "COVERED" in detection_class:
                sumatoria_probabilidad_cubierto += score
                contador_probabilidad_cubierto += 1

        promedio_probabilidad_expuesto = (
            sumatoria_probabilidad_expuesto / contador_probabilidad_expuesto
            if contador_probabilidad_expuesto
            else 0.0
        )
        promedio_probabilidad_cubierto = (
            sumatoria_probabilidad_cubierto / contador_probabilidad_cubierto
            if contador_probabilidad_cubierto
            else 0.0
        )

        es_nsfw = (
            promedio_probabilidad_expuesto > self.umbral_minimo_expuesto
            or promedio_probabilidad_cubierto > self.umbral_minimo_cubierto
        )
        return (
            es_nsfw,
            detections,
            promedio_probabilidad_expuesto,
            promedio_probabilidad_cubierto,
        )

    def analyze_batch(
        self,
        images: Sequence[Any],
        batch_size: int | None = None,
    ) -> list[tuple[bool, list[dict[str, Any]], float, float]]:
        """Run one native NudeNet batch and apply the existing thresholds."""

        if not images:
            return []
        effective_batch_size = max(1, int(batch_size or len(images)))
        detect_batch = getattr(self.nude_detector, "detect_batch", None)
        if callable(detect_batch):
            detections_per_image = detect_batch(
                list(images), batch_size=effective_batch_size
            )
        else:  # pragma: no cover - compatibility with unexpected NudeNet builds
            detections_per_image = [
                self.nude_detector.detect(image) for image in images
            ]
        return [
            self._classify_detections(list(detections))
            for detections in detections_per_image
        ]

    def is_nsfw(self, image: Any):
        detections = self.nude_detector.detect(image)
        return self._classify_detections(detections)

