from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


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
        candidate = root / "nvidia" / "cudnn" / "bin" / "cudnn_engines_tensor_ir64_9.dll"
        if candidate.is_file():
            matches.append(str(candidate.resolve()))
    return list(dict.fromkeys(matches))


def main(require_cuda: bool = False) -> int:
    print("=== Diagnóstico NsfwVideoRemover ===")
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8} bits)")
    print(f"Sistema: {platform.platform()}")
    print(f"NVIDIA: {nvidia_info()}")
    for package in (
        "nudenet",
        "onnxruntime",
        "onnxruntime-gpu",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "moviepy",
        "opencv-python-headless",
    ):
        print(f"{package}: {package_version(package)}")

    dlls = find_cudnn_engine_dll()
    if platform.system() == "Windows":
        if dlls:
            print(f"DLL crítica cuDNN: encontrada en {dlls[0]}")
        else:
            print("DLL crítica cuDNN: NO ENCONTRADA")

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"ERROR importando onnxruntime: {exc}")
        return 1

    print(f"Proveedores compilados: {ort.get_available_providers()}")
    try:
        from applications.NsfwDetector import NsfwDetector

        # NsfwDetector now runs a real blank-frame inference before reporting CUDA.
        detector = NsfwDetector(0.15, 0.65, device="auto")
        print(f"Prueba real NudeNet: {detector.provider_summary()}")
    except Exception as exc:
        print(f"ERROR ejecutando la prueba real NudeNet: {exc}")
        return 1

    if detector.device == "cuda":
        print("OK: una inferencia real con cuDNN se ejecutó mediante CUDA.")
        return 0

    print("OK: el proyecto funciona por CPU.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("Para NVIDIA ejecuta: python instalar.py --nvidia")
    elif detector.fallback_reason:
        print(f"Motivo del fallback: {detector.fallback_reason}")

    if require_cuda:
        print("ERROR: se exigió CUDA, pero la inferencia real terminó en CPU.")
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprueba CPU, CUDA y cuDNN.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Devuelve error si una inferencia real no se ejecuta mediante CUDA.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(require_cuda=args.require_cuda))