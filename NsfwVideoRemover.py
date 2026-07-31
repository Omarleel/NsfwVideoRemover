from __future__ import annotations

import argparse
import multiprocessing as mp

from applications.constants import (
    DEFAULT_ANALYSIS_MAX_DIMENSION,
    DEFAULT_CLIP_DURATION,
    DEFAULT_COVERED_THRESHOLD,
    DEFAULT_CUT_PADDING_SECONDS,
    DEFAULT_EXPOSED_THRESHOLD,
    DEFAULT_FREEPIK_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD,
    DEFAULT_FREEPIK_UNSAFE_THRESHOLD,
    DEFAULT_NSFW_THRESHOLD,
    SUPPORTED_DETECTOR_NAMES,
)
from applications.NsfwVideoProcessor import NsfwVideoProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza un video con un detector intercambiable y elimina los "
            "intervalos marcados como NSFW."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="video.mp4",
        help="Ruta del video de entrada (por defecto: video.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Carpeta de salida. Por defecto usa la carpeta del video.",
    )
    parser.add_argument(
        "--detector",
        choices=SUPPORTED_DETECTOR_NAMES,
        default="nudenet",
        help="Backend de clasificación: nudenet, falconsai o freepik.",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help=(
            "Modelo compatible con image-classification. Vacío selecciona "
            "Falconsai/nsfw_image_detection para falconsai y "
            "Freepik/nsfw_image_detector para freepik."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Dispositivo de inferencia. auto intenta CUDA y cae a CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 selecciona automáticamente; en GPU el valor seguro es 1.",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=DEFAULT_CLIP_DURATION,
        help="Duración en segundos de cada segmento analizado.",
    )
    parser.add_argument(
        "--analysis-max-dimension",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_DIMENSION,
        help=(
            "Reduce los frames antes de enviarlos al detector. 0 conserva la "
            f"resolución original (por defecto: {DEFAULT_ANALYSIS_MAX_DIMENSION}px)."
        ),
    )
    parser.add_argument(
        "--exposed-threshold",
        type=float,
        default=DEFAULT_EXPOSED_THRESHOLD,
        help="Umbral NudeNet para clases EXPOSED.",
    )
    parser.add_argument(
        "--covered-threshold",
        type=float,
        default=DEFAULT_COVERED_THRESHOLD,
        help="Umbral NudeNet para clases COVERED.",
    )
    parser.add_argument(
        "--nudenet-aggregation",
        choices=("max", "mean"),
        default="max",
        help=(
            "Cómo combinar detecciones NudeNet. max evita que una detección "
            "fuerte quede diluida; mean conserva el comportamiento anterior."
        ),
    )
    parser.add_argument(
        "--nsfw-threshold",
        type=float,
        default=DEFAULT_NSFW_THRESHOLD,
        help="Umbral de la clase NSFW para detectores de clasificación.",
    )
    parser.add_argument(
        "--freepik-unsafe-threshold",
        type=float,
        default=DEFAULT_FREEPIK_UNSAFE_THRESHOLD,
        help=(
            "Freepik: corta cuando low+medium+high alcanza este valor "
            f"(por defecto: {DEFAULT_FREEPIK_UNSAFE_THRESHOLD:.2f})."
        ),
    )
    parser.add_argument(
        "--freepik-medium-high-threshold",
        type=float,
        default=DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD,
        help=(
            "Freepik: corta cuando medium+high alcanza este valor "
            f"(por defecto: {DEFAULT_FREEPIK_MEDIUM_HIGH_THRESHOLD:.2f})."
        ),
    )
    parser.add_argument(
        "--freepik-high-threshold",
        type=float,
        default=DEFAULT_FREEPIK_HIGH_THRESHOLD,
        help=(
            "Freepik: corta cuando high alcanza este valor "
            f"(por defecto: {DEFAULT_FREEPIK_HIGH_THRESHOLD:.2f})."
        ),
    )
    parser.add_argument(
        "--cut-padding",
        "--padding-seconds",
        dest="cut_padding_seconds",
        type=float,
        default=DEFAULT_CUT_PADDING_SECONDS,
        help=(
            "Segundos exactos que se cortan antes y después de cada "
            f"detección (por defecto: {DEFAULT_CUT_PADDING_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--padding-segments",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--codec",
        default="auto",
        help=(
            "Encoder de salida. 'auto' prefiere h264_nvenc cuando el FFmpeg y "
            "la GPU lo permiten, con fallback automático a libx264. 'copy' es "
            "más rápido, pero puede perder GOP sanos en los bordes."
        ),
    )
    parser.add_argument(
        "--hardware-accel",
        choices=("auto", "cuda", "none"),
        default="auto",
        help=(
            "Aceleración FFmpeg. auto intenta NVDEC/scale_cuda y NVENC; cuda "
            "lo solicita explícitamente; none fuerza la ruta por CPU."
        ),
    )
    parser.add_argument(
        "--ffmpeg",
        default="",
        help=(
            "Ruta a un FFmpeg concreto. Por defecto se elige entre NSFW_FFMPEG, "
            "ffmpeg del sistema e imageio-ffmpeg, priorizando soporte CUDA/NVENC."
        ),
    )
    parser.add_argument(
        "--prefetch-frames",
        type=int,
        default=0,
        help="Frames preparados por el decodificador; 0 calcula un valor seguro.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Frames por lote de inferencia; 0 calcula un valor seguro.",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--profile-output",
        default="",
        help=(
            "Ruta del JSON de perfilado atómico. Por defecto usa "
            "<video>.profile.json junto al video."
        ),
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Desactiva el profiler detallado para medir el rendimiento sin instrumentación.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Solo genera SRT e informe JSON. Elimina cualquier salida de video "
            "antigua para evitar confusiones."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    processor = NsfwVideoProcessor(
        input_video_path=args.video,
        umbral_minimo_expuesto=args.exposed_threshold,
        umbral_minimo_cubierto=args.covered_threshold,
        output_folder_path=args.output_dir,
        clip_duration=args.clip_duration,
        analysis_max_dimension=args.analysis_max_dimension,
        num_procesos=args.workers,
        device=args.device,
        codec=args.codec,
        cut_padding_seconds=args.cut_padding_seconds,
        padding_segments=args.padding_segments,
        prefetch_frames=args.prefetch_frames,
        batch_size=args.batch_size,
        fast_copy_when_unchanged=not args.force_reencode,
        detector_backend=args.detector,
        model_id=args.model_id,
        nsfw_threshold=args.nsfw_threshold,
        nudenet_aggregation=args.nudenet_aggregation,
        freepik_unsafe_threshold=args.freepik_unsafe_threshold,
        freepik_medium_high_threshold=args.freepik_medium_high_threshold,
        freepik_high_threshold=args.freepik_high_threshold,
        analyze_only=args.analyze_only,
        profile_enabled=not args.no_profile,
        profile_output_path=args.profile_output,
        ffmpeg_executable=args.ffmpeg,
        hardware_acceleration=args.hardware_accel,
    )
    processor.process_video()


if __name__ == "__main__":
    mp.freeze_support()
    main()
