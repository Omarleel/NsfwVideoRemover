from __future__ import annotations

import argparse
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Versión reproducible comprobada con Python 3.10, CUDA 12 y RTX 50/Blackwell.
ORT_VERSION = "1.23.2"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


def nvidia_is_visible() -> bool:
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


def validate_environment() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Se requiere Python 3.10 o superior; se recomienda 3.11.")
    if struct.calcsize("P") * 8 != 64:
        raise SystemExit("ONNX Runtime requiere Python/SO de 64 bits en este proyecto.")


def uninstall_runtime_and_nudenet(pip: list[str]) -> None:
    # NudeNet declara el nombre de distribución `onnxruntime` aunque el módulo
    # también sea proporcionado por `onnxruntime-gpu`. Quitar NudeNet durante el
    # cambio de runtime evita el aviso rojo engañoso del resolvedor de pip.
    run(
        pip + ["uninstall", "-y", "nudenet", "onnxruntime", "onnxruntime-gpu"],
        check=False,
    )


def install_nudenet(pip: list[str]) -> None:
    # No permitimos que NudeNet vuelva a instalar onnxruntime CPU encima del GPU.
    run(pip + ["install", "--no-deps", "nudenet==3.4.2"])


def install_cpu_runtime(pip: list[str], *, clean: bool = False) -> None:
    command = pip + ["install", "--upgrade"]
    if clean:
        command += ["--force-reinstall", "--no-cache-dir"]
    command += [f"onnxruntime=={ORT_VERSION}"]
    run(command)


def install_nvidia_runtime(pip: list[str], *, clean: bool = False) -> None:
    command = pip + ["install", "--upgrade"]
    if clean:
        command += ["--force-reinstall", "--no-cache-dir"]
    command += [f"onnxruntime-gpu[cuda,cudnn]=={ORT_VERSION}"]
    run(command)


def diagnostic_command(python: str, *, require_cuda: bool) -> list[str]:
    command = [python, str(ROOT / "diagnostico.py")]
    if require_cuda:
        command.append("--require-cuda")
    return command


def repair_nvidia_installation(python: str, pip: list[str]) -> bool:
    print(
        "\nLa prueba CUDA real falló. Se hará una reparación limpia de "
        "ONNX Runtime, CUDA y cuDNN (puede descargar varios GB)."
    )
    uninstall_runtime_and_nudenet(pip)
    try:
        install_nvidia_runtime(pip, clean=True)
        install_nudenet(pip)
    except subprocess.CalledProcessError:
        return False
    return run(diagnostic_command(python, require_cuda=True), check=False).returncode == 0


def install(profile: str, run_diagnostics: bool) -> None:
    validate_environment()
    requested = profile
    selected = profile
    if profile == "auto":
        selected = "nvidia" if nvidia_is_visible() else "cpu"

    if selected == "nvidia" and platform.system() not in {"Windows", "Linux"}:
        print("El paquete CUDA de ONNX Runtime no está disponible aquí; usando CPU.")
        selected = "cpu"

    print(f"Perfil seleccionado: {selected}")
    python = sys.executable
    pip = [python, "-m", "pip"]

    run(pip + ["install", "--upgrade", "pip"])
    uninstall_runtime_and_nudenet(pip)
    run(pip + ["install", "-r", str(ROOT / "requirements-common.txt")])

    if selected == "nvidia":
        try:
            install_nvidia_runtime(pip)
        except subprocess.CalledProcessError:
            if requested == "nvidia":
                raise SystemExit(
                    "No se pudo instalar el runtime NVIDIA. Ejecuta de nuevo el "
                    "instalador después de comprobar la conexión y el espacio libre."
                )
            print(
                "No se pudo instalar el runtime NVIDIA en este entorno; "
                "se instalará el perfil CPU para conservar la funcionalidad."
            )
            uninstall_runtime_and_nudenet(pip)
            selected = "cpu"
            install_cpu_runtime(pip)
    else:
        install_cpu_runtime(pip)

    install_nudenet(pip)

    if not run_diagnostics:
        print("\nInstalación completada sin diagnóstico.")
        return

    require_cuda = selected == "nvidia"
    diagnostic_ok = (
        run(diagnostic_command(python, require_cuda=require_cuda), check=False).returncode
        == 0
    )

    if require_cuda and not diagnostic_ok:
        diagnostic_ok = repair_nvidia_installation(python, pip)

    if diagnostic_ok:
        print("\nInstalación completada y validada.")
        return

    if requested == "auto" and selected == "nvidia":
        print(
            "\nCUDA siguió fallando después de la reparación. Se instalará el "
            "perfil CPU para que el proyecto pueda ejecutarse."
        )
        uninstall_runtime_and_nudenet(pip)
        install_cpu_runtime(pip, clean=True)
        install_nudenet(pip)
        if run(diagnostic_command(python, require_cuda=False), check=False).returncode == 0:
            print("\nInstalación CPU completada y validada.")
            return

    raise SystemExit(
        "La instalación terminó, pero la prueba requerida no pasó. Revisa la "
        "salida de diagnostico.py antes de procesar videos."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instala el perfil CPU o NVIDIA sin mezclar paquetes ONNX Runtime."
    )
    parser.add_argument(
        "--auto",
        dest="profile",
        action="store_const",
        const="auto",
        help="Detecta NVIDIA mediante nvidia-smi; es el valor predeterminado.",
    )
    parser.add_argument(
        "--cpu",
        dest="profile",
        action="store_const",
        const="cpu",
        help="Instala ONNX Runtime CPU.",
    )
    parser.add_argument(
        "--nvidia",
        dest="profile",
        action="store_const",
        const="nvidia",
        help="Instala ONNX Runtime GPU con runtimes CUDA/cuDNN incluidos.",
    )
    parser.add_argument(
        "--sin-diagnostico",
        action="store_true",
        help="No ejecuta diagnostico.py al terminar (no recomendado).",
    )
    parser.set_defaults(profile="auto")
    args = parser.parse_args()
    install(args.profile, not args.sin_diagnostico)


if __name__ == "__main__":
    main()