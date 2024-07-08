import os
import stat
import time
import json
import torch.multiprocessing as mp
from progress.bar import ChargingBar
from moviepy.editor import VideoFileClip, concatenate_videoclips
from Applications.SrtGenerator import SrtGenerator
from Applications.NsfwDetector import NsfwDetector

def process_func(video_path, temp_folder, lote, umbral_minimo_expuesto, umbral_minimo_cubierto, result_file_prefix, progress_queue):
    video = VideoFileClip(video_path)
    nsfw_detector = NsfwDetector(
        umbral_minimo_expuesto=umbral_minimo_expuesto, 
        umbral_minimo_cubierto=umbral_minimo_cubierto
    )

    results = []
    for segmento in lote:
        start_time, end_time = segmento['intervalo']
        clip = video.subclip(start_time, end_time)
        frame_path = os.path.join(temp_folder, f"frame_{int(start_time)}.jpg")
        clip.save_frame(frame_path, t=0)
        es_nsfw, detections, _, _ = nsfw_detector.is_nsfw(frame_path)
        segmento['detecciones'] = detections
        segmento['nsfw'] = es_nsfw
        results.append(segmento)
        try:
            os.remove(frame_path)
        except PermissionError:
            time.sleep(1)
            try:
                os.remove(frame_path)
            except Exception as e:
                print(f"Error al intentar eliminar {frame_path}: {e}")
        
        progress_queue.put(1)  # Informar que se ha procesado un segmento
    
    # Escribir resultados en un archivo temporal
    result_file_path = f"{result_file_prefix}_{mp.current_process().pid}.json"
    with open(result_file_path, 'w') as result_file:
        json.dump(results, result_file)

class NsfwVideoProcessor:
    def __init__(self, input_video_path, umbral_minimo_expuesto=0.15, umbral_minimo_cubierto=0.65, output_folder_path="", clip_duration=1, num_procesos=4):
        self.video_path = input_video_path
        self.umbral_minimo_expuesto = umbral_minimo_expuesto
        self.umbral_minimo_cubierto = umbral_minimo_cubierto
        self.output_video_path = os.path.join(output_folder_path, f"{os.path.splitext(os.path.basename(input_video_path))[0]} (no_nsfw).mp4")
        self.clip_duration = clip_duration
        self.srt_generator = SrtGenerator()
        self.temp_folder = "temp_frames"
        self.create_temp_folder()
        self.clips = []
        self.video = VideoFileClip(self.video_path)
        self.num_procesos = num_procesos

    def create_temp_folder(self):
        os.makedirs(self.temp_folder, exist_ok=True)
        os.chmod(self.temp_folder, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)  # Establecer permisos rwx

    def split_list_into_parts(self, lista, n):
        tamaño_parte = len(lista) // n
        partes = [lista[i * tamaño_parte:(i + 1) * tamaño_parte] for i in range(n)]
        partes[-1].extend(lista[n * tamaño_parte:])
        return partes
    
    def mark_nsfw(self, results, rango = 2):
        # Lista para almacenar los índices que deben ser marcados como nsfw
        indices_to_mark = set()

        # Buscar todos los índices que deben ser marcados como nsfw
        for i in range(len(results)):
            if results[i]['nsfw']:
                # Marcar el índice actual
                indices_to_mark.add(i)
                
                # Marcar `rango` índices arriba
                for j in range(max(0, i-rango), i):
                    indices_to_mark.add(j)
                
                # Marcar `rango` índices abajo
                for j in range(i+1, min(len(results), i+rango+1)):
                    indices_to_mark.add(j)

        # Actualizar los resultados originales basándonos en los índices encontrados
        for i in indices_to_mark:
            results[i]['nsfw'] = True

        return results
    
    def process_video(self):
        segmentos = []
        indice = 0
        start_time = 0
        while start_time < self.video.duration:
            indice += 1
            end_time = min(start_time + self.clip_duration, self.video.duration)
            segmentos.append({"orden": indice, "intervalo": [start_time, end_time], "detecciones": None, "nsfw": None})
            start_time = end_time
        
        progress_queue = mp.Queue()
        bar = ChargingBar('Procesando segmentos', max=len(segmentos))
        lotes = self.split_list_into_parts(segmentos, self.num_procesos)

        processes = []
        result_file_prefix = os.path.join(self.temp_folder, "resultados")
        for i in range(self.num_procesos):
            lote = lotes[i]
            p = mp.Process(
                target=process_func,
                args=(
                    self.video_path,
                    self.temp_folder,
                    lote,
                    self.umbral_minimo_expuesto,
                    self.umbral_minimo_cubierto,
                    result_file_prefix,
                    progress_queue,
                ),
                daemon=True)
            p.start()
            processes.append(p)
        
        completed_count = 0
        while completed_count < len(segmentos):
            completed_count += progress_queue.get()  # Actualizar el contador de segmentos completados
            bar.next()
        
        bar.finish()
        
        for p in processes:
            p.join()
            
        for p in processes:
            p.terminate()
        
        results = []
        for i in range(self.num_procesos):
            result_file_path = f"{result_file_prefix}_{processes[i].pid}.json"
            if os.path.exists(result_file_path):
                with open(result_file_path, 'r') as result_file:
                    results.extend(json.load(result_file))
                os.remove(result_file_path)
                
        results = self.mark_nsfw(results) 
        results = sorted(results, key=lambda x: x['orden'])  # Ordenar resultados por la clave orden
        
        bar2 = ChargingBar('Concatenando clips sin NSFW', max=len([result for result in results if not result['nsfw']]))
        for result in results:
            start_time, end_time = result['intervalo']
            self.srt_generator.add_subtitle(start_time, end_time, result['detecciones'])
            if not result['nsfw']:
                clip = self.video.subclip(start_time, end_time)
                self.clips.append(clip)
                bar2.next()
        bar2.finish()
        
        if self.clips:
            final_clip = concatenate_videoclips(self.clips)
            final_clip.write_videofile(self.output_video_path, codec="hevc_nvenc", threads=32)
            print(f"Clip guardado en {self.output_video_path}")
            self.srt_generator.generate_srt(f"{os.path.splitext(os.path.basename(self.video_path))[0]}.srt")
            print('Archivo .srt generado')
        else:
            print("No hay clips para concatenar")
