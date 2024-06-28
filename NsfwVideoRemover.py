from Applications.NsfwVideoProcessor import NsfwVideoProcessor
if __name__ == "__main__":
    ruta_video = "video.mp4"
    umbral_minimo_expuesto = 0.15
    umbral_minimo_cubierto = 0.65
    processor = NsfwVideoProcessor(
        input_video_path=ruta_video,
        umbral_minimo_expuesto=umbral_minimo_expuesto,
        umbral_minimo_cubierto=umbral_minimo_cubierto,
    )
    processor.process_video()