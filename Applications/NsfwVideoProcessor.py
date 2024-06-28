import os
import stat
import time
from progress.bar import ChargingBar
from moviepy.editor import VideoFileClip, concatenate_videoclips
from Applications.SrtGenerator import SrtGenerator
from Applications.NsfwDetector import NsfwDetector

class NsfwVideoProcessor:
    def __init__(self, input_video_path, umbral_minimo_expuesto = 0.15, umbral_minimo_cubierto = 0.65, output_folder_path="", clip_duration=1):
        self.video_path = input_video_path
        self.output_video_path = os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(input_video_path))[0]} (no_nsfw).mp4")
        self.clip_duration = clip_duration
        self.srt_generator = SrtGenerator()
        self.nsfw_detector = NsfwDetector(
            umbral_minimo_expuesto=umbral_minimo_expuesto, 
            umbral_minimo_cubierto=umbral_minimo_cubierto
        )
        self.temp_folder = "temp_frames"
        self.create_temp_folder()
        self.video = VideoFileClip(self.video_path)
        self.duracion_total = self.video.duration
        self.clips = []
        self.indices_nsfw = []

    def create_temp_folder(self):
        os.makedirs(self.temp_folder, exist_ok=True)
        os.chmod(self.temp_folder, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)  # Establecer permisos rwx

    def process_video(self):
        bar = ChargingBar('Procesando:', max=self.duracion_total)
        start_time = 0
        indice = 0

        while start_time < self.duracion_total:
            end_time = min(start_time + self.clip_duration, self.duracion_total)
            clip = self.video.subclip(start_time, end_time)
            frame_path = os.path.join(self.temp_folder, f"frame_{int(start_time)}.jpg")
            clip.save_frame(frame_path, t=0)
            
            es_nsfw, detections, promedio_probabilidad_expuesto, promedio_probabilidad_cubierto = self.nsfw_detector.is_nsfw(frame_path)
            self.srt_generator.add_subtitle(start_time, end_time, detections)
            # self.srt_generator.add_subtitle(start_time, end_time, f"Expuesto: {round(promedio_probabilidad_expuesto * 100, 2)}% | Cubierto {round(promedio_probabilidad_cubierto * 100, 2)}%")
            
            self.clips.append(clip)
            if es_nsfw:
                self.indices_nsfw.extend([indice, indice + 1, indice + 2])
                if indice == 1:
                    self.indices_nsfw.append(indice - 1)
                else:
                    self.indices_nsfw.extend([indice - 1, indice - 2])
                        
            try:
                os.remove(frame_path)
            except PermissionError:
                time.sleep(1)
                try:
                    os.remove(frame_path)
                except Exception as e:
                    pass
                    # print(f"Error al intentar eliminar {frame_path}: {e}")
                    
            start_time = end_time
            bar.next()
            indice += 1

        bar.finish()
        self.clean_up()

    def clean_up(self):
        # Eliminar clips con contenido NSFW
        indices_nsfw_filtrados = set(filter(lambda x: 0 <= x < len(self.clips), self.indices_nsfw))
        for indice_eliminar in sorted(indices_nsfw_filtrados, reverse=True):
            self.clips.pop(indice_eliminar)
        
        # Eliminar la carpeta temporal
        try:
            os.rmdir(self.temp_folder)
        except Exception as e:
            pass
            # print(f"Error al eliminar la carpeta temporal: {e}")

        # Generar archivo SRT
        self.srt_generator.generate_srt(f"{os.path.splitext(os.path.basename(self.video_path))[0]}.srt")
        print('Archivo .srt generado')

        if self.clips:
            # Concatenar los clips seguros
            final_clip = concatenate_videoclips(self.clips)
            # Guardar el video final
            final_clip.write_videofile(self.output_video_path, codec="hevc_nvenc", threads=32)
        else:
            print("No hay clips para concatenar")
