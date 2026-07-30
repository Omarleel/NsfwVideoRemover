from __future__ import annotations

import argparse
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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


def is_virtual_environment() -> bool:
    return bool(
        getattr(sys, "real_prefix", None)
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )


def validate_environment(*, allow_global: bool) -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Se requiere Python 3.10 o superior; se recomienda 3.11.")
    if struct.calcsize("P") * 8 != 64:
        raise SystemExit("Se requiere Python y sistema operativo de 64 bits.")
    if not is_virtual_environment() and not allow_global:
        raise SystemExit(
            "Por seguridad, el instalador solo modifica un entorno virtual. "
            "Crea y activa .venv o usa --permitir-entorno-global de forma explícita."
        )


def uninstall_nudenet_runtime(pip: list[str]) -> None:
    run(
        pip + ["uninstall", "-y", "nudenet", "onnxruntime", "onnxruntime-gpu"],
        check=False,
    )


def install_nudenet_package(pip: list[str]) -> None:
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


def diagnostic_command(
    python: str,
    *,
    detector: str,
    require_cuda: bool,
    model_id: str,
) -> list[str]:
    command = [
        python,
        str(ROOT / "diagnostico.py"),
        "--detector",
        detector,
        "--model-id",
        model_id,
    ]
    if require_cuda:
        command.append("--require-cuda")
    return command


def repair_nvidia_installation(
    python: str,
    pip: list[str],
    *,
    model_id: str,
) -> bool:
    print(
        "\nLa prueba CUDA de NudeNet falló. Se hará una reparación limpia de "
        "ONNX Runtime, CUDA y cuDNN."
    )
    uninstall_nudenet_runtime(pip)
    try:
        install_nvidia_runtime(pip, clean=True)
        install_nudenet_package(pip)
    except subprocess.CalledProcessError:
        return False
    return (
        run(
            diagnostic_command(
                python,
                detector="nudenet",
                require_cuda=True,
                model_id=model_id,
            ),
            check=False,
        ).returncode
        == 0
    )


def install_huggingface(
    pip: list[str],
    python: str,
    *,
    requested_profile: str,
    run_diagnostics: bool,
    model_id: str,
) -> None:
    run(pip + ["install", "--upgrade", "-r", str(ROOT / "requirements-huggingface.txt")])
    if not run_diagnostics:
        print("\nInstalación Hugging Face completada sin diagnóstico.")
        return

    require_cuda = requested_profile == "nvidia"
    code = run(
        diagnostic_command(
            python,
            detector="huggingface",
            require_cuda=require_cuda,
            model_id=model_id,
        ),
        check=False,
    ).returncode
    if code != 0:
        raise SystemExit(
            "La instalación terminó, pero el diagnóstico Hugging Face no pasó. "
            "Revisa la salida anterior."
        )
    print(
        "\nInstalación Hugging Face validada. El modelo se descargará en la "
        "primera ejecución o con diagnostico.py --load-model."
    )


def install_nudenet(
    profile: str,
    run_diagnostics: bool,
    *,
    model_id: str,
) -> None:
    requested = profile
    selected = profile
    if profile == "auto":
        selected = "nvidia" if nvidia_is_visible() else "cpu"

    if selected == "nvidia" and platform.system() not in {"Windows", "Linux"}:
        print("El runtime CUDA de ONNX Runtime no está disponible aquí; usando CPU.")
        selected = "cpu"

    print(f"Perfil seleccionado: {selected}")
    python = sys.executable
    pip = [python, "-m", "pip"]

    uninstall_nudenet_runtime(pip)
    run(pip + ["install", "--upgrade", "-r", str(ROOT / "requirements-common.txt")])

    if selected == "nvidia":
        try:
            install_nvidia_runtime(pip)
        except subprocess.CalledProcessError:
            if requested == "nvidia":
                raise SystemExit(
                    "No se pudo instalar el runtime NVIDIA. Comprueba la conexión, "
                    "el espacio disponible y el controlador."
                )
            print("No se pudo instalar NVIDIA; se conservará funcionalidad por CPU.")
            uninstall_nudenet_runtime(pip)
            selected = "cpu"
            install_cpu_runtime(pip)
    else:
        install_cpu_runtime(pip)

    install_nudenet_package(pip)
    if not run_diagnostics:
        print("\nInstalación NudeNet completada sin diagnóstico.")
        return

    require_cuda = selected == "nvidia"
    diagnostic_ok = (
        run(
            diagnostic_command(
                python,
                detector="nudenet",
                require_cuda=require_cuda,
                model_id=model_id,
            ),
            check=False,
        ).returncode
        == 0
    )
    if require_cuda and not diagnostic_ok:
        diagnostic_ok = repair_nvidia_installation(
            python,
            pip,
            model_id=model_id,
        )
    if diagnostic_ok:
        print("\nInstalación NudeNet completada y validada.")
        return

    if requested == "auto" and selected == "nvidia":
        print("\nCUDA siguió fallando. Se instalará el perfil CPU.")
        uninstall_nudenet_runtime(pip)
        install_cpu_runtime(pip, clean=True)
        install_nudenet_package(pip)
        if (
            run(
                diagnostic_command(
                    python,
                    detector="nudenet",
                    require_cuda=False,
                    model_id=model_id,
                ),
                check=False,
            ).returncode
            == 0
        ):
            print("\nInstalación CPU completada y validada.")
            return

    raise SystemExit(
        "La instalación terminó, pero la prueba requerida no pasó. "
        "Revisa la salida de diagnostico.py."
    )


def install(
    profile: str,
    detector: str,
    run_diagnostics: bool,
    *,
    allow_global: bool,
    model_id: str,
) -> None:
    validate_environment(allow_global=allow_global)
    python = sys.executable
    pip = [python, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip"])

    if detector == "huggingface":
        install_huggingface(
            pip,
            python,
            requested_profile=profile,
            run_diagnostics=run_diagnostics,
            model_id=model_id,
        )
        return
    install_nudenet(profile, run_diagnostics, model_id=model_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instala un backend de detección dentro de un entorno virtual."
    )
    parser.add_argument(
        "--detector",
        choices=("nudenet", "huggingface"),
        default="nudenet",
        help="Backend que se instalará.",
    )
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--auto", dest="profile", action="store_const", const="auto")
    profile.add_argument("--cpu", dest="profile", action="store_const", const="cpu")
    profile.add_argument("--nvidia", dest="profile", action="store_const", const="nvidia")
    parser.add_argument(
        "--model-id",
        default="Falconsai/nsfw_image_detection",
        help="Modelo de Hugging Face usado por el diagnóstico opcional.",
    )
    parser.add_argument(
        "--sin-diagnostico",
        action="store_true",
        help="No ejecuta diagnostico.py al terminar.",
    )
    parser.add_argument(
        "--permitir-entorno-global",
        action="store_true",
        help="Permite modificar el Python global; úsalo solo de forma consciente.",
    )
    parser.set_defaults(profile="auto")
    args = parser.parse_args()
    install(
        args.profile,
        args.detector,
        not args.sin_diagnostico,
        allow_global=args.permitir_entorno_global,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
