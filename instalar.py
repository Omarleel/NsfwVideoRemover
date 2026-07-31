from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from applications.constants import (
    SUPPORTED_DETECTOR_NAMES,
    default_model_for_backend,
    normalize_detector_backend,
)


ROOT = Path(__file__).resolve().parent
ORT_VERSION = "1.23.2"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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
    # venv/virtualenv
    if getattr(sys, "real_prefix", None) or sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    # Conda does not always change base_prefix, so detect its active prefix explicitly.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    return bool(conda_prefix and Path(conda_prefix).resolve() == Path(sys.prefix).resolve())


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
    load_model: bool = False,
) -> list[str]:
    detector = normalize_detector_backend(detector)
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
    if load_model:
        command.append("--load-model")
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


def torch_runtime_info(python: str) -> dict[str, object]:
    script = (
        "import json; "
        "import torch; "
        "print(json.dumps({"
        "'version': torch.__version__, "
        "'compiled_cuda': torch.version.cuda, "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    result = subprocess.run(
        [python, "-c", script], capture_output=True, text=True, check=False, timeout=30
    )
    if result.returncode != 0:
        return {"import_error": (result.stderr or result.stdout).strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"import_error": result.stdout.strip()}


def install_pytorch_runtime(
    pip: list[str],
    python: str,
    *,
    selected_profile: str,
    clean: bool = False,
) -> None:
    index_url = PYTORCH_CUDA_INDEX if selected_profile == "nvidia" else PYTORCH_CPU_INDEX
    command = pip + ["install", "--upgrade"]
    if clean:
        command += ["--force-reinstall", "--no-cache-dir"]
    command += ["torch", "torchvision", "--index-url", index_url]
    run(command)

    info = torch_runtime_info(python)
    print(
        "PyTorch instalado: "
        f"versión={info.get('version', 'desconocida')}; "
        f"CUDA compilada={info.get('compiled_cuda')}; "
        f"CUDA disponible={info.get('cuda_available')}; "
        f"GPU={info.get('device_name')}"
    )
    if selected_profile == "nvidia" and not (
        info.get("compiled_cuda") and info.get("cuda_available")
    ):
        raise RuntimeError(
            "La rueda instalada de PyTorch no puede usar CUDA. "
            "No se aceptará una degradación silenciosa a CPU."
        )


def install_transformers_backend(
    pip: list[str],
    python: str,
    *,
    detector: str,
    requested_profile: str,
    run_diagnostics: bool,
    model_id: str,
) -> None:
    selected = requested_profile
    if selected == "auto":
        selected = "nvidia" if nvidia_is_visible() else "cpu"
    if selected == "nvidia" and platform.system() not in {"Windows", "Linux"}:
        if requested_profile == "nvidia":
            raise SystemExit("PyTorch CUDA solo se instala automáticamente en Windows o Linux.")
        selected = "cpu"

    print(f"Perfil de clasificadores Transformers seleccionado: {selected}")
    run(pip + ["install", "--upgrade", "-r", str(ROOT / "requirements-common.txt")])

    # PyTorch se instala antes de timm. De lo contrario, la dependencia transitiva
    # de timm puede hacer que pip elija una rueda CPU-only desde PyPI.
    try:
        install_pytorch_runtime(pip, python, selected_profile=selected)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        if requested_profile == "nvidia":
            raise SystemExit(f"No se pudo instalar/validar PyTorch CUDA: {exc}") from exc
        if selected == "nvidia":
            print(
                "PyTorch CUDA no quedó operativo. Se hará una reparación limpia "
                "antes de considerar CPU."
            )
            run(pip + ["uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
            try:
                install_pytorch_runtime(
                    pip, python, selected_profile="nvidia", clean=True
                )
            except (subprocess.CalledProcessError, RuntimeError) as retry_exc:
                print(f"La reparación CUDA falló: {retry_exc}")
                print("Se instalará CPU porque se solicitó --auto.")
                run(
                    pip + ["uninstall", "-y", "torch", "torchvision", "torchaudio"],
                    check=False,
                )
                selected = "cpu"
                install_pytorch_runtime(pip, python, selected_profile="cpu", clean=True)
        else:
            raise

    # Con el runtime CUDA/CPU correcto ya validado, estas dependencias pueden
    # resolverse sin sustituir PyTorch por una variante incompatible.
    run(
        pip
        + [
            "install",
            "--upgrade",
            "transformers>=4.40,<6",
            "pillow>=10,<13",
            "safetensors>=0.4,<1",
            "timm>=1.0,<2",
        ]
    )

    if not run_diagnostics:
        print(f"\nInstalación {detector} completada sin diagnóstico.")
        return

    require_cuda = selected == "nvidia"
    code = run(
        diagnostic_command(
            python,
            detector=detector,
            require_cuda=require_cuda,
            model_id=model_id,
            load_model=True,
        ),
        check=False,
    ).returncode
    if code != 0:
        raise SystemExit(
            f"La instalación terminó, pero el diagnóstico {detector} no pasó. "
            "Revisa la salida anterior."
        )
    print(
        f"\nInstalación {detector} validada. "
        f"Dispositivo seleccionado: {selected}. El modelo se descargará en la "
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

    detector = normalize_detector_backend(detector)
    if detector in {"falconsai", "freepik"}:
        install_transformers_backend(
            pip,
            python,
            detector=detector,
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
        choices=SUPPORTED_DETECTOR_NAMES,
        default="nudenet",
        help="Backend que se instalará.",
    )
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--auto", dest="profile", action="store_const", const="auto")
    profile.add_argument("--cpu", dest="profile", action="store_const", const="cpu")
    profile.add_argument("--nvidia", dest="profile", action="store_const", const="nvidia")
    parser.add_argument(
        "--model-id",
        default="",
        help=(
            "Modelo usado por el diagnóstico. Vacío selecciona Falconsai para "
            "falconsai y Freepik/nsfw_image_detector para freepik."
        ),
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
    model_id = args.model_id.strip() or default_model_for_backend(args.detector)
    install(
        args.profile,
        args.detector,
        not args.sin_diagnostico,
        allow_global=args.permitir_entorno_global,
        model_id=model_id,
    )


if __name__ == "__main__":
    main()
