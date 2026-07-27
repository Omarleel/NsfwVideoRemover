from __future__ import annotations

import argparse
import multiprocessing as mp
from applications.NsfwVideoProcessor import NsfwVideoProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza un video con NudeNet y elimina los segmentos marcados "
            "como NSFW. Usa CUDA cuando está disponible y CPU como fallback."
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
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Proveedor de inferencia. auto intenta CUDA y cae a CPU.",
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
        default=1.0,
        help="Duración en segundos de cada segmento analizado.",
    )
    parser.add_argument(
        "--exposed-threshold",
        type=float,
        default=0.15,
        help="Umbral promedio para clases EXPOSED.",
    )
    parser.add_argument(
        "--covered-threshold",
        type=float,
        default=0.65,
        help="Umbral promedio para clases COVERED.",
    )
    parser.add_argument(
        "--cut-padding",
        "--padding-seconds",
        dest="cut_padding_seconds",
        type=float,
        default=4.0,
        help=(
            "Segundos exactos que se cortan antes y después de cada "
            "detección prohibida (por defecto: 4)."
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
    )
    processor.process_video()


if __name__ == "__main__":
    mp.freeze_support()
    main()