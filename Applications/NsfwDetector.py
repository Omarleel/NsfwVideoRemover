"""Backward-compatible import for the original NudeNet detector class."""

from applications.detectors.nudenet import NudeNetDetector


class NsfwDetector(NudeNetDetector):
    def __init__(
        self,
        umbral_minimo_expuesto: float,
        umbral_minimo_cubierto: float,
        device: str = "auto",
        intra_op_threads: int = 0,
        aggregation: str = "max",
    ) -> None:
        super().__init__(
            exposed_threshold=umbral_minimo_expuesto,
            covered_threshold=umbral_minimo_cubierto,
            device=device,
            intra_op_threads=intra_op_threads,
            aggregation=aggregation,
        )
