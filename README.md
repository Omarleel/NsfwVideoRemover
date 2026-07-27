# NsfwVideoRemover

Analiza un frame por intervalo con NudeNet, genera un `.srt` con las detecciones y crea un video sin los rangos marcados como NSFW.

Esta versión mantiene la selección automática `CUDA -> CPU`, pero cambia la arquitectura de análisis para evitar decodificar y buscar el mismo video desde varios procesos.

## Arquitectura optimizada

```text
Video
  │
  ▼
FFmpeg: un solo decodificador secuencial (CPU)
  │
  ▼
Cola acotada de frames preparados
  │
  ├── CUDA: una sesión GPU persistente
  │          CPU y GPU trabajan solapadas
  │
  └── CPU: lotes distribuidos a procesos persistentes
             cada sesión recibe una cuota de hilos ONNX
  │
  ▼
Resultados ordenados -> SRT -> cortes -> codificación
```

### Qué se optimizó

- Un único proceso FFmpeg decodifica el video de forma secuencial.
- Se eliminaron las aperturas independientes del video y los seeks aleatorios por worker.
- Una cola con límite de memoria mantiene frames listos mientras se ejecuta la inferencia.
- Con CUDA, la CPU decodifica el siguiente frame mientras la GPU analiza el actual.
- NudeNet ejecuta inferencia ONNX nativa por lotes, no un bucle de llamadas individuales.
- En CPU, esos lotes se envían a procesos que conservan el modelo cargado.
- ONNX Runtime reparte los hilos disponibles entre workers para evitar sobresuscripción.
- Los tamaños automáticos de lote y prefetch consideran la resolución del video.
- Se informa el rendimiento real del análisis en frames por segundo.
- Cuando no hay cortes, FFmpeg intenta copiar los streams sin recodificar.
- La exportación conserva el fallback `h264_nvenc -> libx264` cuando sí es necesario recodificar.

No se divide automáticamente la inferencia entre GPU y CPU. Normalmente eso añade coordinación y deja que el dispositivo más lento determine el rendimiento. El modo CUDA ya usa ambos recursos en paralelo: CPU para decodificación y alimentación; GPU para inferencia.

## Requisitos

- Windows o Linux de 64 bits.
- Python 3.10 o superior; Python 3.11 es la opción recomendada.
- FFmpeg. MoviePy normalmente obtiene una copia mediante ImageIO.
- Para NVIDIA: controlador compatible con la versión de ONNX Runtime instalada.

## Instalación

### Detección automática

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python instalar.py --auto
```

Linux:

```bash
. .venv/bin/activate
python instalar.py --auto
```

### NVIDIA

```bash
python instalar.py --nvidia
```

También están disponibles `instalar_nvidia.bat` e `instalar_nvidia.sh`.

### Solo CPU

```bash
python instalar.py --cpu
```

O:

```bash
pip install -r requirements.txt
```

## Verificar el entorno

```bash
python diagnostico.py
```

El diagnóstico indica los proveedores compilados y si una inferencia real puede ejecutarse con CUDA. Si CUDA falla, la aplicación continúa con CPU.

## Uso

Selección automática:

```bash
python NsfwVideoRemover.py video.mp4
```

Ejemplo completo:

```bash
python NsfwVideoRemover.py video.mp4 \
  --device auto \
  --workers 0 \
  --clip-duration 1 \
  --exposed-threshold 0.15 \
  --covered-threshold 0.65 \
  --cut-padding 4 \
  --prefetch-frames 0 \
  --batch-size 0 \
  --codec auto \
  --output-dir resultados
```

En Windows CMD escribe el comando en una sola línea o usa `^` para continuarlo.

## Opciones de rendimiento

- `--device auto`: intenta CUDA y usa CPU como fallback. También acepta `cuda` o `cpu`.
- `--workers 0`: selección automática. CUDA usa 1 worker por defecto; CPU usa hasta 4.
- `--prefetch-frames 0`: calcula una cola acotada por resolución, memoria y paralelismo.
- `--batch-size 0`: calcula un lote nativo NudeNet. En CUDA prioriza lotes de hasta 4; en multiprocessing limita el tamaño de los envíos IPC.
- `--force-reencode`: desactiva la copia rápida cuando el video no necesita cortes.
- `--clip-duration`: intervalo entre frames analizados. Un valor menor aumenta cobertura y costo.
- `--cut-padding`: segundos eliminados antes y después de cada detección.
- `--codec auto`: usa NVENC cuando está disponible y recurre a `libx264` si falla.

### Perfiles sugeridos

GPU:

```bash
python NsfwVideoRemover.py video.mp4 --device auto --workers 1
```

CPU automática:

```bash
python NsfwVideoRemover.py video.mp4 --device cpu --workers 0
```

Memoria limitada:

```bash
python NsfwVideoRemover.py video.mp4 --prefetch-frames 2 --batch-size 1
```

No conviene aumentar `--workers` en CUDA sin medir. Cada worker crea otra sesión ONNX y otra copia del modelo en VRAM.

## Salida de rendimiento

Durante la ejecución se muestran los parámetros elegidos y el throughput:

```text
Workers de inferencia: 1
Pipeline: prefetch=8 frames; lote=4; threads FFmpeg=8
Análisis completado: 600 frames en 83.41s (7.19 frames/s).
```

Esos valores permiten comparar configuraciones en el mismo equipo sin cambiar el código.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas cubren selección CUDA/CPU, fallback, umbrales, segmentación, padding, codec, presupuesto de hilos ONNX y configuración del pipeline FFmpeg.

Para detalles de diseño y una guía de medición, consulta [`OPTIMIZACION.md`](OPTIMIZACION.md).

## Autores

- Omarleel — desarrollador original.
- Optimización y adaptación de compatibilidad CPU/NVIDIA sobre el proyecto proporcionado.
