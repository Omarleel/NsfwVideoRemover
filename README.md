# NsfwVideoRemover

Procesa un video por segmentos, analiza un frame de cada segmento con NudeNet y genera:

- un video nuevo sin los segmentos marcados;
- un archivo `.srt` con las detecciones realizadas.

La ejecución usa NVIDIA CUDA cuando puede inicializarse correctamente y cambia automáticamente a CPU cuando no existe una GPU compatible, el controlador es antiguo o faltan bibliotecas.

## Cambios de esta versión

- Compatibilidad con **NVIDIA RTX 50 / Blackwell**, incluida la GeForce RTX 5060 Ti, mediante ONNX Runtime compilado para CUDA 12.8.
- Fallback automático de inferencia `CUDA -> CPU`.
- Fallback automático de exportación `h264_nvenc -> libx264`.
- Las PC sin NVIDIA funcionan con el perfil CPU.
- Eliminada la dependencia innecesaria de PyTorch: ahora se usa `multiprocessing` estándar.
- Compatibilidad con MoviePy 2.2.1 (`moviepy.editor` ya no existe en MoviePy 2).
- Los workers ya no escriben frames ni JSON temporales y el proceso principal no se queda esperando indefinidamente si uno falla.
- La cantidad automática de workers es 1 con CUDA para evitar cargar varias copias del modelo en la VRAM.
- Nueva interfaz de línea de comandos, instalador por perfiles y diagnóstico del proveedor activo.

## Requisitos

- Windows o Linux de 64 bits.
- Python 3.10 o superior; **Python 3.11 es la opción recomendada**.
- FFmpeg. MoviePy normalmente obtiene una copia mediante ImageIO.
- Para NVIDIA: controlador NVIDIA reciente. El instalador GPU añade los runtimes CUDA y cuDNN mediante paquetes de Python, por lo que normalmente no hace falta instalar manualmente el CUDA Toolkit completo.

## Instalación recomendada

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

El modo `--auto` usa `nvidia-smi` para elegir el perfil NVIDIA. Sin NVIDIA selecciona CPU. Si la instalación del runtime NVIDIA no es posible en ese entorno, el instalador cambia al perfil CPU para conservar la funcionalidad.

### NVIDIA, incluida RTX 5060 Ti

Windows también puede ejecutar directamente:

```bat
instalar_nvidia.bat
```

O manualmente dentro de un entorno virtual:

```bash
python instalar.py --nvidia
```

El perfil fija `onnxruntime-gpu` en el rango `>=1.21,<1.27`. Esas versiones oficiales usan CUDA 12.8 y cuDNN 9; CUDA 12.8 fue la primera versión del toolkit con soporte completo para Blackwell/RTX 50.

### Solo CPU

```bash
python instalar.py --cpu
```

También puede usarse la instalación convencional:

```bash
pip install -r requirements.txt
```

`requirements.txt` es deliberadamente el perfil CPU universal. El perfil NVIDIA debe instalarse con `instalar.py` porque NudeNet 3.4.2 declara `onnxruntime` CPU como dependencia; instalar todo en un único `requirements` dejaría simultáneamente `onnxruntime` y `onnxruntime-gpu`, lo que causa instalaciones ambiguas o rotas.

## Verificar la instalación

```bash
python diagnostico.py
```

Resultado esperado con una RTX 5060 Ti correctamente configurada:

```text
Proveedores compilados: [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
Sesión NudeNet: dispositivo=cuda; proveedores activos=CUDAExecutionProvider, CPUExecutionProvider
OK: la inferencia CUDA está activa.
```

Si aparece `dispositivo=cpu`, el programa sigue funcionando. Actualiza el controlador NVIDIA y vuelve a ejecutar:

```bash
python instalar.py --nvidia
```

## Uso

Procesar `video.mp4` con selección automática de hardware:

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
  --padding-segments 2 \
  --codec auto \
  --output-dir resultados
```

En Windows CMD escribe el comando en una sola línea o usa `^` en lugar de `\` para continuarlo.

### Opciones principales

- `--device auto`: intenta CUDA y cae a CPU. También acepta `cuda` o `cpu`.
- `--workers 0`: selección automática. Con CUDA se elige 1 por seguridad de VRAM.
- `--codec auto`: intenta `h264_nvenc` si hay NVIDIA y FFmpeg lo ofrece; ante un error vuelve a `libx264`.
- `--clip-duration`: intervalo entre frames analizados. Un valor menor detecta cambios más rápidos, pero requiere más tiempo.
- `--padding-segments`: elimina segmentos vecinos alrededor de cada detección.

## Compatibilidad de GPU

No existe un único binario CUDA que acelere literalmente todas las tarjetas NVIDIA de todas las generaciones. Este proyecto sigue esta política:

1. Las GPU compatibles con ONNX Runtime CUDA 12.8 usan aceleración.
2. Una NVIDIA sin soporte de esa runtime, con controlador insuficiente o sin CUDA utilizable cambia a CPU.
3. La exportación NVENC es independiente de la inferencia CUDA; si la tarjeta o FFmpeg no admiten el encoder solicitado, se usa `libx264`.

Por eso el proyecto puede ejecutarse en más equipos sin hacer que la presencia de una GPU sea un requisito obligatorio.

## Notas técnicas

NudeNet 3.4.2 acepta un parámetro `providers`, pero su implementación publicada no lo reenvía a `onnxruntime.InferenceSession`. `applications/NsfwDetector.py` aplica el proveedor durante la creación de la sesión y después comprueba `session.get_providers()` para conocer el proveedor realmente activo.

Fuentes técnicas:

- ONNX Runtime CUDA Execution Provider: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- NVIDIA CUDA 12.8 y Blackwell: https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/
- Compute capability de GPU NVIDIA: https://developer.nvidia.com/cuda/gpus
- Migración de MoviePy 1 a 2: https://zulko.github.io/moviepy/getting_started/updating_to_v2.html

## Pruebas

Las pruebas unitarias no necesitan una GPU real y comprueban la selección CUDA/CPU, el fallback cuando CUDA falla, los umbrales, la segmentación, el padding y la selección de codec:

```bash
python -m unittest discover -s tests -v
```

## Autores

- Omarleel — desarrollador original.
- Adaptación de compatibilidad CPU/NVIDIA realizada sobre el proyecto proporcionado.
