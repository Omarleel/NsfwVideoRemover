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
- **Falconsai (Hugging Face, Auto):** `python instalar.py --detector falconsai --auto`
- **Freepik EVA-02 (Auto):** `python instalar.py --detector freepik --auto`

Pruebas reales de carga:

- `python diagnostico.py --detector falconsai --load-model`
- `python diagnostico.py --detector freepik --load-model`

### Instalación de clasificadores Transformers con GPU NVIDIA

`instalar.py --detector falconsai --auto` y `--detector freepik --auto` detectan `nvidia-smi`. Si existe una GPU NVIDIA utilizable, instala PyTorch desde el índice CUDA oficial y valida una inferencia real en GPU. En Conda ya no es necesario usar `--permitir-entorno-global` cuando `CONDA_PREFIX` coincide con el Python activo.

El instalador rechaza una rueda `+cpu` cuando se seleccionó NVIDIA y evita que `requirements-falconsai.txt` vuelva a sustituir PyTorch CUDA. Para forzar el comportamiento use `--nvidia` o `--cpu`.

---

## 🧭 Nombres de detectores

Los nombres públicos identifican el detector, no el sitio donde se aloja el modelo:

- `nudenet`: detector NudeNet mediante ONNX Runtime.
- `falconsai`: `Falconsai/nsfw_image_detection`, alojado en Hugging Face Hub.
- `freepik`: `Freepik/nsfw_image_detector`, alojado en Hugging Face Hub.

Solo se aceptan esos tres nombres. `huggingface` identifica al proveedor en los metadatos y al motor base de Transformers, pero no es un detector válido para `--detector`.

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

### 2. Usando Freepik EVA-02 (recomendado)

El backend `freepik` selecciona automáticamente `Freepik/nsfw_image_detector`, que devuelve cuatro probabilidades: `neutral`, `low`, `medium` y `high`. La política predeterminada corta cuando se cumple cualquiera de estas reglas:

- `low + medium + high >= 0.60`
- `medium + high >= 0.45`
- `high >= 0.25`

```bash
python NsfwVideoRemover.py video.mp4 \
  --detector freepik \
  --device auto \
  --clip-duration 0.5 \
  --freepik-unsafe-threshold 0.60 \
  --freepik-medium-high-threshold 0.45 \
  --freepik-high-threshold 0.25 \
  --cut-padding 4 \
  --hardware-accel auto \
  --codec auto
```

No es necesario indicar `--model-id` salvo que se quiera probar otro modelo compatible de cuatro niveles.

### 3. Usando Falconsai (modelo alojado en Hugging Face)
Usa la puntuación de la etiqueta NSFW para compararla con el umbral.
```bash
python NsfwVideoRemover.py video.mp4 \
  --detector falconsai \
  --device auto \
  --analysis-max-dimension 1280 \
  --nsfw-threshold 0.50 \
  --cut-padding 4 \
  --hardware-accel auto \
  --codec auto
```

### 4. Modo Análisis (Solo revisión)
Genera `video.srt` y `video.analysis.json` sin renderizar un nuevo video ni aplicar recodificación. Elimina videos de salida antiguos para evitar confusiones.
```bash
python NsfwVideoRemover.py video.mp4 --detector freepik --analyze-only
```

---

## 🎯 Freepik multiclase (v2.7.0)

`FreepikImageDetector` reutiliza el motor directo de Transformers, mantiene precisión FP32 y adapta la salida multiclase al contrato común del proyecto. El JSON conserva las probabilidades originales y añade:

- `unsafe = low + medium + high`
- `medium_high = medium + high`

La decisión es *fail-closed* ante etiquetas incompatibles: si el modelo no devuelve `neutral/low/medium/high`, el análisis se detiene en lugar de considerar el frame sano.

Como el modelo procesa imágenes de 448×448, el lote automático es más conservador que con Falconsai:

- 12 GiB o más libres: lote 16.
- 6 GiB o más: lote 8.
- 3 GiB o más: lote 4.
- Menos de 3 GiB: lote 2.

El fallback por falta de VRAM reduce el lote a la mitad y reintenta. El profiler registra los tres umbrales, la política aplicada y el lote máximo realmente ejecutado.

## ⚡ Falconsai optimizado (v2.6.1)

El backend `falconsai` ya no utiliza `transformers.pipeline` durante producción. Carga directamente `AutoImageProcessor` y `AutoModelForImageClassification`, ejecuta el modelo bajo `torch.inference_mode()` y conserva precisión FP32.

La selección automática de lote usa la VRAM libre después de cargar el modelo:

- 12 GiB o más libres: lote 32.
- 6 GiB o más: lote 16.
- 3 GiB o más: lote 8.
- Menos de 3 GiB: lote 4.

Si otra aplicación ocupa VRAM y el lote elegido falla, se reduce automáticamente a la mitad y se reintenta sin abortar el video. Se mantiene un solo worker para no duplicar el modelo en la GPU.

El modelo y el procesador se buscan primero en la caché local. Solo se consulta Hugging Face Hub cuando faltan archivos, por lo que las ejecuciones posteriores evitan una comprobación de red innecesaria.

Ejemplo conservador:

```bash
python NsfwVideoRemover.py video.mp4 \
  --detector falconsai \
  --device auto \
  --nsfw-threshold 0.15 \
  --clip-duration 0.5 \
  --hardware-accel auto \
  --codec auto
```

La consola informa `motor=direct_fp32`, `carga=local_cache` y el lote recomendado. `--batch-size N` sigue permitiendo fijar manualmente el lote para pruebas comparativas.

El profiler añade:

- `detector_inference_engine`
- `detector_precision`
- `detector_model_load_source`
- `detector_recommended_batch_size`
- `detector_runtime_batch_size`
- `detector_oom_fallback_count`

Estas optimizaciones no cambian el modelo, el umbral, la frecuencia de muestreo ni la resolución que recibe el preprocesador oficial.

## ⚙️ Optimización y Rendimiento

El sistema está diseñado para maximizar la eficiencia en la extracción de frames y el uso de memoria:
- **Selección automática de FFmpeg:** compara el binario indicado con `--ffmpeg`, `NSFW_FFMPEG`, el FFmpeg del sistema e `imageio-ffmpeg`. En modo automático prioriza el que anuncie NVDEC, `scale_cuda` y NVENC.
- **NVDEC + `scale_cuda`:** con `--hardware-accel auto`, los frames 4K permanecen en la GPU durante decodificación, muestreo y escalado. Solo los frames seleccionados y ya reducidos cruzan a RAM. Antes de iniciar se ejecuta una prueba real; cualquier incompatibilidad cae automáticamente a la ruta CPU.
- **Canalización eficiente:** FFmpeg abre un único decodificador y produce frames RGB ordenados en una cola acotada (hasta ~256 MiB o 48 frames), evitando múltiples procesos o búsquedas independientes.
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
 │   ├─ HuggingFaceImageDetector (motor base de Transformers)
 │   ├─ FalconsaiImageDetector
 │   └─ FreepikImageDetector
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
Todos los backends generan una estructura estable. El nivel superior separa `detector`, `provider` y `model_id`. En el nivel superior, `allowed_intervals` contiene los tramos sanos solicitados y `rendered_intervals` los límites realmente usados. `expected_output_duration_seconds` es la suma de los intervalos sanos y `actual_output_duration_seconds` registra la duración comprobada del archivo final. Con `--codec auto`, `render_mode` será `h264_nvenc` cuando NVENC funcione y `libx264` como fallback. Con `--codec copy`, los límites pueden desplazarse hacia dentro y `render_mode` será `copy`:
```json
{
  "schema_version": 1,
  "detector": "falconsai",
  "provider": "huggingface",
  "model_id": "Falconsai/nsfw_image_detection",
  "segments": [
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
  ]
}
```

Con Freepik, `metricas` conserva el detalle multiclase:

```json
{
  "nsfw": true,
  "score_nsfw": 0.31,
  "metricas": {
    "neutral": 0.69,
    "low": 0.18,
    "medium": 0.10,
    "high": 0.03,
    "unsafe": 0.31,
    "medium_high": 0.13
  },
  "modelo": "Freepik/nsfw_image_detector"
}
```

---

## 🧪 Pruebas y Limitaciones

**Ejecutar pruebas:**
```bash
python -m unittest discover -s tests -v
```
Las 63 pruebas cubren fallbacks, presupuestos de ONNX, la política multiclase de Freepik, combinaciones de intervalos, reescritura atómica, esquema del profiler y progreso real de FFmpeg.

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