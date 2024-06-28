import torch
from nudenet import NudeDetector

class NsfwDetector:
    def __init__(self, umbral_minimo_expuesto, umbral_minimo_cubierto):
        self.umbral_minimo_expuesto = umbral_minimo_expuesto
        self.umbral_minimo_cubierto = umbral_minimo_cubierto
        self.cuda_available = torch.cuda.is_available()
        self.device = 'CUDAExecutionProvider' if self.cuda_available else 'CPUExecutionProvider'
        self.nude_detector = NudeDetector(providers=[self.device])

    def is_nsfw(self, image_path):
        detections = self.nude_detector.detect(image_path)
        sumatoria_probabilidad_expuesto = 0.0
        contador_probabilidad_expuesto = 0
        sumatoria_probabilidad_cubierto = 0.0
        contador_probabilidad_cubierto = 0
        excluded_classes = [
            'FACE_FEMALE', 
            'FACE_MALE', 
            'ARMPITS_EXPOSED', 
            'ARMPITS_COVERED',
            'FEET_EXPOSED',
            'FEET_COVERED'
        ]

        for detection in detections:
            if not any(excluded in detection['class'] for excluded in excluded_classes):
                if 'EXPOSED' in detection['class']:
                    sumatoria_probabilidad_expuesto += detection['score'] if detection['class'] != 'BELLY_EXPOSED' else detection['score'] / 2
                    contador_probabilidad_expuesto += 1
                elif 'COVERED' in detection['class']:
                    sumatoria_probabilidad_cubierto += detection['score']
                    contador_probabilidad_cubierto += 1

        promedio_probabilidad_expuesto = sumatoria_probabilidad_expuesto / contador_probabilidad_expuesto if contador_probabilidad_expuesto != 0 else 0.0
        promedio_probabilidad_cubierto = sumatoria_probabilidad_cubierto / contador_probabilidad_cubierto if contador_probabilidad_cubierto != 0 else 0.0

        if promedio_probabilidad_expuesto > self.umbral_minimo_expuesto or promedio_probabilidad_cubierto > self.umbral_minimo_cubierto:
            return (True, detections, promedio_probabilidad_expuesto, promedio_probabilidad_cubierto)

        return (False, detections, promedio_probabilidad_expuesto, promedio_probabilidad_cubierto)
