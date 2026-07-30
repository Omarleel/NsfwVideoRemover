from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "no instalado"


def nvidia_info() -> str:
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not executable:
        return "nvidia-smi no encontrado"
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"no se pudo ejecutar nvidia-smi: {exc}"
    return result.stdout.strip() or result.stderr.strip() or "sin respuesta"


def find_cudnn_engine_dll() -> list[str]:
    if platform.system() != "Windows":
        return []
    matches: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry)
        if not root.is_dir():
            continue
        candidate = (
            root
            / "nvidia"
            / "cudnn"
            / "bin"
            / "cudnn_engines_tensor_ir64_9.dll"
        )
        if candidate.is_file():
            matches.append(str(candidate.resolve()))
    return list(dict.fromkeys(matches))


def print_common_info() -> None:
    print("=== Diagnóstico NsfwVideoRemover ===")
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8} bits)")
    print(f"Sistema: {platform.platform()}")
    print(f"NVIDIA: {nvidia_info()}")
    for package in (
        "nudenet",
        "onnxruntime",
        "onnxruntime-gpu",
        "torch",
        "transformers",
        "pillow",
        "imageio-ffmpeg",
        "opencv-python-headless",
        "psutil",
        "nvidia-ml-py",
    ):
        print(f"{package}: {package_version(package)}")
    try:
        from applications.ffmpeg_capabilities import resolve_ffmpeg_executable

        executable, capabilities = resolve_ffmpeg_executable(prefer_hardware=True)
        print(f"FFmpeg seleccionado: {executable}")
        print(
            "FFmpeg hardware: "
            f"CUDA={capabilities.supports_cuda_decode}; "
            f"scale_cuda={capabilities.supports_cuda_scale}; "
            f"h264_nvenc={capabilities.supports_h264_nvenc}"
        )
        if capabilities.probe_errors:
            print(f"Advertencias FFmpeg: {list(capabilities.probe_errors)}")
    except Exception as exc:
        print(f"ERROR diagnosticando FFmpeg: {exc}")


def diagnose_nudenet(*, require_cuda: bool) -> int:
    dlls = find_cudnn_engine_dll()
    if platform.system() == "Windows":
        print(
            f"DLL crítica cuDNN: encontrada en {dlls[0]}"
            if dlls
            else "DLL crítica cuDNN: NO ENCONTRADA"
        )

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"ERROR importando onnxruntime: {exc}")
        return 1

    print(f"Proveedores compilados: {ort.get_available_providers()}")
    try:
        from applications.detectors.config import DetectorConfig
        from applications.detectors.factory import create_detector

        detector = create_detector(DetectorConfig(backend="nudenet", device="auto"))
        print(f"Prueba real NudeNet: {detector.provider_summary()}")
    except Exception as exc:
        print(f"ERROR ejecutando la prueba real NudeNet: {exc}")
        return 1

    if detector.device == "cuda":
        print("OK: una inferencia real se ejecutó mediante CUDA.")
        return 0

    print("OK: NudeNet funciona por CPU.")
    fallback_reason = getattr(detector, "fallback_reason", None)
    if fallback_reason:
        print(f"Motivo del fallback: {fallback_reason}")
    if require_cuda:
        print("ERROR: se exigió CUDA, pero la inferencia terminó en CPU.")
        return 2
    return 0


def diagnose_huggingface(
    *,
    require_cuda: bool,
    load_model: bool,
    model_id: str,
) -> int:
    try:
        import torch
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:
        print(f"ERROR importando el backend Hugging Face: {exc}")
        return 1

    cuda_available = bool(torch.cuda.is_available())
    print(f"PyTorch detecta CUDA: {cuda_available}")
    if require_cuda and not cuda_available:
        print("ERROR: se exigió CUDA, pero PyTorch no detectó una GPU utilizable.")
        return 2

    if not load_model:
        print(
            "OK: dependencias Hugging Face disponibles. Usa --load-model para "
            "descargar/cargar el modelo y ejecutar una inferencia en blanco."
        )
        return 0

    try:
        from applications.detectors.config import DetectorConfig
        from applications.detectors.factory import create_detector

        requested_device = "cuda" if require_cuda else "auto"
        detector = create_detector(
            DetectorConfig(
                backend="huggingface",
                device=requested_device,
                model_id=model_id,
            )
        )
        assessment = detector.analyze_batch(
            [np.zeros((224, 224, 3), dtype=np.uint8)],
            batch_size=1,
        )[0]
        print(f"Prueba real Hugging Face: {detector.provider_summary()}")
        print(
            "Resultado de imagen en blanco: "
            f"nsfw={assessment.is_nsfw}; score={assessment.score:.4f}"
        )
    except Exception as exc:
        print(f"ERROR cargando o ejecutando el modelo Hugging Face: {exc}")
        return 1

    if require_cuda and detector.device != "cuda":
        print("ERROR: se exigió CUDA, pero el detector terminó en CPU.")
        return 2
    print("OK: el modelo Hugging Face ejecutó una inferencia real.")
    return 0


def main(
    *,
    detector: str,
    require_cuda: bool,
    load_model: bool,
    model_id: str,
) -> int:
    print_common_info()
    if detector == "huggingface":
        return diagnose_huggingface(
            require_cuda=require_cuda,
            load_model=load_model,
            model_id=model_id,
        )
    return diagnose_nudenet(require_cuda=require_cuda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprueba los backends y aceleradores.")
    parser.add_argument(
        "--detector",
        choices=("nudenet", "huggingface"),
        default="nudenet",
    )
    parser.add_argument(
        "--model-id",
        default="Falconsai/nsfw_image_detection",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Devuelve error si el backend no puede usar CUDA.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="En Hugging Face, descarga/carga el modelo y ejecuta una prueba real.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(
            detector=args.detector,
            require_cuda=args.require_cuda,
            load_model=args.load_model,
            model_id=args.model_id,
        )
    )
