from __future__ import annotations

import argparse
import multiprocessing as mp

from applications.constants import (
    DEFAULT_CLIP_DURATION,
    DEFAULT_COVERED_THRESHOLD,
    DEFAULT_CUT_PADDING_SECONDS,
    DEFAULT_EXPOSED_THRESHOLD,
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_NSFW_THRESHOLD,
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
        choices=("nudenet", "huggingface"),
        default="nudenet",
        help="Backend de clasificación. NudeNet se mantiene por compatibilidad.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_HUGGINGFACE_MODEL,
        help=(
            "Modelo compatible con image-classification cuando se usa "
            "--detector huggingface."
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
            "Codec de salida. auto intenta h264_nvenc con NVIDIA y usa "
            "libx264 como fallback."
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
        help="Recodifica incluso cuando no hay cortes.",
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
        analyze_only=args.analyze_only,
    )
    processor.process_video()


if __name__ == "__main__":
    mp.freeze_support()
    main()
