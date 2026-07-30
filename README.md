# NsfwVideoRemover

Analiza frames de un video con un detector de contenido intercambiable, genera un informe y, opcionalmente, elimina los intervalos marcados.

---

## 🛠 Requisitos e Instalación

- **Python 3.10+** (64 bits).
- **FFmpeg**. El programa compara `NSFW_FFMPEG`, el FFmpeg del sistema e `imageio-ffmpeg`, y elige el que ofrezca mejor soporte CUDA/NVENC.
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
  --analysis-max-dimension 1280 \
  --exposed-threshold 0.15 \
  --covered-threshold 0.65 \
  --cut-padding 4 \
  --hardware-accel auto \
  --codec auto
```

### 2. Usando Hugging Face (ej. Falconsai)
Usa la puntuación de la etiqueta NSFW para compararla con el umbral.
```bash
python NsfwVideoRemover.py video.mp4 \
  --detector huggingface \
  --model-id Falconsai/nsfw_image_detection \
  --device auto \
  --analysis-max-dimension 1280 \
  --nsfw-threshold 0.50 \
  --cut-padding 4 \
  --hardware-accel auto \
  --codec auto
```

### 3. Modo Análisis (Solo revisión)
Genera `video.srt` y `video.analysis.json` sin renderizar un nuevo video ni aplicar recodificación. Elimina videos de salida antiguos para evitar confusiones.
```bash
python NsfwVideoRemover.py video.mp4 --detector huggingface --analyze-only
```

---

## ⚙️ Optimización y Rendimiento

El sistema está diseñado para maximizar la eficiencia en la extracción de frames y el uso de memoria:
- **Selección automática de FFmpeg:** compara el binario indicado con `--ffmpeg`, `NSFW_FFMPEG`, el FFmpeg del sistema e `imageio-ffmpeg`. En modo automático prioriza el que anuncie NVDEC, `scale_cuda` y NVENC.
- **NVDEC + `scale_cuda`:** con `--hardware-accel auto`, los frames 4K permanecen en la GPU durante decodificación, muestreo y escalado. Solo los frames seleccionados y ya reducidos cruzan a RAM. Antes de iniciar se ejecuta una prueba real; cualquier incompatibilidad cae automáticamente a la ruta CPU.
- **Canalización eficiente:** FFmpeg abre un único decodificador y produce frames RGB ordenados en una cola acotada (~256 MiB o 32 frames), evitando múltiples procesos o búsquedas independientes.
- **Reducción previa al modelo:** los frames grandes se escalan dentro de FFmpeg antes de entrar a Python. El valor predeterminado limita el lado mayor a 1280 px; `--analysis-max-dimension 0` conserva la resolución original.
- **Workers guiados por la cola:** `--workers 0` conserva un solo worker. No duplica modelos mientras el consumidor siga esperando una cola vacía. El profiler registra `queue_starved`, ocupación, bloqueos y una recomendación; un número fijo mayor solo se usa cuando se solicita explícitamente.
- **Lotes (`--batch-size`):** Calculado automáticamente según resolución, dispositivo y costo de IPC.
- **Segmentación estable:** El tiempo se deriva del índice matemático, eliminando errores de coma flotante (segmentos fantasma).
- **Sin cortes:** el video completo se copia con `-c copy`, sin recodificación ni pérdida generacional.
- **Cortes exactos por defecto:** `--codec auto` prefiere `h264_nvenc` (`p4`, VBR/CQ 19) cuando está disponible y vuelve a `libx264` (`veryfast`, CRF 18) si NVENC no puede inicializarse. No desplaza los límites a keyframes y, por tanto, no elimina GOP sanos adicionales.
- **Timeline continuo validado:** después del render se compara la duración del archivo con la suma de los intervalos sanos. Si no coincide, se regenera mediante un grafo conservador de `trim/concat`; nunca se conserva una salida con huecos de timestamps.
- **Audio sincronizado:** los intervalos de audio se recortan y concatenan con timestamps reiniciados, y se codifican en AAC.
- **Modo rápido opcional:** `--codec copy` conserva el modo antiguo de stream copy. Es más rápido, pero puede descartar contenido sano cercano a cada corte debido a los keyframes y B-frames.
- **Fotograma cero:** el muestreador fuerza explícitamente el primer frame (`n=0`) y luego toma una muestra por segmento.
- **Progreso real del render:** la generación final usa el canal `-progress` de FFmpeg y muestra una barra basada en `out_time`, no una estimación artificial. Funciona tanto en recodificación exacta como en stream copy.
- **Profiler atómico y de sistema:** cada ejecución genera `<video>.profile.json` con eventos por frame/lote, cola, workers, FFmpeg, progreso, RAM real, RSS/USS/VMS, CPU e I/O del árbol completo de procesos, VRAM y utilización GPU/NVENC/NVDEC mediante NVML.

*Nota sobre métricas: Reducir `--clip-duration` mejora la precisión temporal pero aumenta linealmente los tiempos de inferencia. Mide siempre FPS y uso de memoria al ajustar estos parámetros.*

---

### Forzar o desactivar aceleración

```bash
# Automático: NVDEC/scale_cuda + NVENC cuando funcionen
python NsfwVideoRemover.py video.mp4 --hardware-accel auto --codec auto

# FFmpeg concreto con soporte NVIDIA
python NsfwVideoRemover.py video.mp4 --ffmpeg "C:\\ffmpeg\\bin\\ffmpeg.exe"

# Comparación completamente por CPU
python NsfwVideoRemover.py video.mp4 --hardware-accel none --codec libx264
```

## 🔬 Profiler de rendimiento

El profiler está activado por defecto y escribe `video.profile.json` junto al informe de análisis. Está diseñado para comparar configuraciones como `--batch-size`, `--prefetch-frames`, `--workers`, resolución de análisis y codec.

```bash
python NsfwVideoRemover.py video.mp4 --device auto --codec auto
```

Ruta personalizada:

```bash
python NsfwVideoRemover.py video.mp4 --profile-output resultados/ejecucion.profile.json
```

Para medir una ejecución sin la instrumentación detallada:

```bash
python NsfwVideoRemover.py video.mp4 --no-profile
```

El JSON contiene:

- `configuration`: parámetros solicitados y resueltos, FFmpeg elegido, capacidades CUDA/NVENC, modo real de decodificación, encoder final y salud de la cola.
- `events`: eventos atómicos con duración de pared, CPU, PID, thread y RSS real para cada frame/lote/intervalo.
- `system_resource_samples`: snapshots periódicos de RAM del sistema y del árbol de procesos, RSS/VMS/USS/privada, CPU, threads, lecturas/escrituras, GPU, VRAM, NVENC y NVDEC.
- `summary.operations_by_total_time`: agrupación con total, promedio, mínimo, máximo y percentiles p50/p95/p99.
- `summary.system_resources`: picos de RAM por PID/rol, RAM total del árbol, CPU observada, mínimos de RAM disponible, picos de VRAM y utilización GPU.
- `ffmpeg_progress_samples`: muestras de frame, FPS, velocidad, bytes, bitrate, tiempo producido y porcentaje durante el render.
- `registered_processes`: PIDs y roles (`python_main`, `ffmpeg_analysis_decode`, `ffmpeg_final_render`, `inference_worker`).
- `derived_efficiency`: CPU equivalente de Python y del árbol observado por separado, overhead estimado y rendimiento relativo al tamaño del archivo.
- `errors`: fallos registrados incluso cuando la ejecución no termina correctamente.

Las métricas RSS usan `psutil` en Windows/Linux/macOS, con una implementación WinAPI corregida como fallback. Los tiempos de alto nivel y los eventos atómicos pueden solaparse deliberadamente; para localizar cuellos de botella usa primero el resumen y luego revisa los eventos crudos de la operación lenta.

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
 ├─ PerformanceProfiler (eventos, árbol de procesos, RAM, GPU/VRAM y progreso)
 ├─ VideoProbe (Metadatos sin decodificar el video completo)
 └─ VideoRenderer (cortes exactos o stream copy por keyframes)
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
Todos los backends generan una estructura estable. En el nivel superior, `allowed_intervals` contiene los tramos sanos solicitados y `rendered_intervals` los límites realmente usados. `expected_output_duration_seconds` es la suma de los intervalos sanos y `actual_output_duration_seconds` registra la duración comprobada del archivo final. Con `--codec auto`, `render_mode` será `h264_nvenc` cuando NVENC funcione y `libx264` como fallback. Con `--codec copy`, los límites pueden desplazarse hacia dentro y `render_mode` será `copy`:
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
Las pruebas cubren fallbacks, presupuestos de ONNX, combinaciones de intervalos, reescritura atómica, esquema del profiler y progreso real de FFmpeg.

**Consideraciones importantes:**
- Ningún modelo es perfecto. Revisa el JSON/SRT antes de automatizar el borrado.
- Se analiza **un frame por segmento**; frames intermedios no se escanean individualmente.
- Cuando existen cortes, `--codec auto` intenta NVENC automáticamente. Usa `--hardware-accel none` para forzar CPU o `--codec libx264` para impedir NVENC.
- NVDEC/NVENC requieren un driver NVIDIA funcional y un FFmpeg compilado con esos componentes. La presencia del nombre del encoder no basta: el programa realiza pruebas reales y mantiene fallbacks seguros.
- `--codec copy` evita recodificar, pero los límites se desplazan hacia dentro hasta keyframes seguros y pueden perderse partes sanas de uno o dos GOP por corte.
- La salida recortada conserva video y la primera pista de audio. No concatena subtítulos ni streams de datos incrustados; el SRT generado se entrega por separado.
- Cada segmento marcado se elimina completo. `--cut-padding` solo amplía el corte antes del inicio y después del final del segmento; con `--cut-padding 0` todavía se elimina el segmento detectado.
- Si el informe muestra cortes más amplios de lo deseado, reduce `--cut-padding`; el renderizador no añade pérdida extra en modo `auto`.
- Cada modelo de Hugging Face usa etiquetas distintas. Lee la *Model Card* antes de cambiar `--model-id`.

## Corrección 2.4.1: final de pista y fallback NVDEC

Algunos MP4 informan la duración del contenedor o del audio como duración total, aunque la pista de video termine unas décimas antes. En 2.4.1, un único frame faltante al final con salida limpia de FFmpeg se resuelve reutilizando el último frame disponible para analizar el segmento final. Los fallos reales de NVDEC reinician automáticamente el análisis con decodificación CPU.

El profiler registra estas situaciones mediante `reuse_last_frame_for_trailing_segment`, `retry_analysis_with_software_decode`, `reused_trailing_analysis_frames` y `hardware_decode_fallbacks`.
