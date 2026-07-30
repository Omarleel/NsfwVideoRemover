# NsfwVideoRemover

Analiza frames de un video con un detector de contenido intercambiable, genera un informe y, opcionalmente, elimina los intervalos marcados.

---

## 🛠 Requisitos e Instalación

- **Python 3.10+** (64 bits).
- **FFmpeg** (proporcionado normalmente por `imageio-ffmpeg`).
- Espacio adicional para modelos y runtimes.

El instalador protegido exige un entorno virtual para evitar conflictos con el Python global.

**Crear entorno (recomendado):**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

**Instalar según el motor deseado:**
- **NudeNet (CPU):** `python instalar.py --detector nudenet --cpu`
- **NudeNet (NVIDIA auto):** `python instalar.py --detector nudenet --auto`
- **Hugging Face (Auto):** `python instalar.py --detector huggingface --auto`

*(Para probar la carga del modelo de Hugging Face: `python diagnostico.py --detector huggingface --load-model`)*

---

## 🚀 Uso

### 1. Usando NudeNet
`max` (por defecto) toma la detección más fuerte para no diluir el resultado. `mean` conserva el comportamiento de versiones anteriores.
```bash
python NsfwVideoRemover.py video.mp4 \
  --detector nudenet \
  --device auto \
  --nudenet-aggregation max \
  --exposed-threshold 0.15 \
  --covered-threshold 0.65 \
  --cut-padding 4
```

### 2. Usando Hugging Face (ej. Falconsai)
Usa la puntuación de la etiqueta NSFW para compararla con el umbral.
```bash
python NsfwVideoRemover.py video.mp4 \
  --detector huggingface \
  --model-id Falconsai/nsfw_image_detection \
  --device auto \
  --nsfw-threshold 0.50 \
  --cut-padding 4
```

### 3. Modo Análisis (Solo revisión)
Genera `video.srt` y `video.analysis.json` sin renderizar un nuevo video ni aplicar recodificación. Elimina videos de salida antiguos para evitar confusiones.
```bash
python NsfwVideoRemover.py video.mp4 --detector huggingface --analyze-only
```

---

## ⚙️ Optimización y Rendimiento

El sistema está diseñado para maximizar la eficiencia en la extracción de frames y el uso de memoria:
- **Canalización eficiente:** FFmpeg abre un único decodificador y produce frames RGB ordenados en una cola acotada (~256 MiB o 32 frames), evitando múltiples procesos o búsquedas independientes.
- **Workers:** Selección automática (`--workers 0`). En CUDA, usa 1 worker para no saturar la VRAM; en CPU usa hasta 4, reservando hilos para FFmpeg.
- **Lotes (`--batch-size`):** Calculado automáticamente según resolución, dispositivo y costo de IPC.
- **Segmentación estable:** El tiempo se deriva del índice matemático, eliminando errores de coma flotante (segmentos fantasma).
- **Recodificación:** `--codec auto` intenta NVENC como prioridad, cayendo a `libx264`. Se usa *fast copy* si no hay cortes (configurable con `--force-reencode`).

*Nota sobre métricas: Reducir `--clip-duration` mejora la precisión temporal pero aumenta linealmente los tiempos de inferencia. Mide siempre FPS y uso de memoria al ajustar estos parámetros.*

---

## 🏗 Arquitectura y Extensibilidad (SOLID)

La lógica del video (extracción, reportes, renderizado) está completamente separada de los modelos de IA.
```text
NsfwVideoProcessor (Orquestador)
 ├─ SegmentPlanner (Crea segmentos matemáticamente)
 ├─ ContentDetector (Interfaz / Protocolo para los modelos)
 │   ├─ NudeNetDetector
 │   └─ HuggingFaceImageDetector
 ├─ CutIntervalPolicy (Combina rangos y aplica padding)
 ├─ AnalysisReportWriter (SRT y JSON atómicos)
 └─ VideoRenderer (Copia rápida, códecs y limpieza)
```

### Cómo añadir un nuevo detector
Solo necesitas crear una clase que cumpla el contrato `ContentDetector` y registrarla en el `DetectorFactory`.

```python
from applications.detectors.base import DetectionAssessment
from applications.detectors.factory import DetectorFactory

class MiDetector:
    name = "mi-detector"
    device = "cpu"

    def analyze_batch(self, images, batch_size=None):
        return [
            DetectionAssessment(
                is_nsfw=False, score=0.0, detections=(), model_name=self.name
            ) for _ in images
        ]

    def provider_summary(self):
        return "modelo=mi-detector; dispositivo=cpu"

# Registrar en la fábrica
DetectorFactory.register("mi-detector", lambda config: MiDetector())
```

---

## 📄 Formato del Informe JSON
Todos los backends generan una estructura estable:
```json
{
  "orden": 1,
  "intervalo": [0.0, 1.0],
  "nsfw": false,
  "score_nsfw": 0.08,
  "motivo": null,
  "metricas": {"nsfw": 0.08},
  "modelo": "Falconsai/nsfw_image_detection",
  "detecciones": [
    {"class": "normal", "score": 0.92},
    {"class": "nsfw", "score": 0.08}
  ]
}
```

---

## 🧪 Pruebas y Limitaciones

**Ejecutar pruebas:**
```bash
python -m unittest discover -s tests -v
```
Las pruebas cubren fallbacks, presupuestos de ONNX, combinaciones de intervalos y reescritura atómica.

**Consideraciones importantes:**
- Ningún modelo es perfecto. Revisa el JSON/SRT antes de automatizar el borrado.
- Se analiza **un frame por segmento**; frames intermedios no se escanean individualmente.
- Cortar el video con MoviePy recodifica video y audio; es posible que se pierdan streams adicionales del archivo original.
- Cada modelo de Hugging Face usa etiquetas distintas. Lee la *Model Card* antes de cambiar `--model-id`.