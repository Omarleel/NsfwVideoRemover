# NsfwVideoRemover

## Descripción
Este programa procesa un video en busca de contenido NSFW (Not Safe For Work) utilizando detección de imágenes. Divide el video en clips de 1 segundo, analiza cada frame para detectar contenido explícito y genera un nuevo video libre de contenido NSFW y un archivo srt (subitítulos) con información sobre la detección.

## Requisitos
- Python 3.10
- CUDA 11.8 y cuDNN 8.9.2
- Paquetes Python: [nudenet](https://github.com/notAI-tech/NudeNet), moviepy, progress

## Configuración
Asegúrate de tener CUDA y cuDNN instalados para aceleración GPU si está disponible. Puedes configurar los umbrales `umbral_minimo_expuesto` y `umbral_minimo_cubierto` (porcentual) según tus necesidades de detección.

## Pruebas
Para correr el programa, puedes ejecutar el siguiente conjunto de comandos:
```bash
# Clonar el repositorio y acceder a la carpeta del programa
git clone https://github.com/Omarleel/NsfwVideoRemover
# Accede al proyecto
cd NsfwVideoRemover
# Instala los requerimientos
pip install -r requirements.txt
# Ejecutar el script
py NsfwVideoRemover.py
```

## Autores
- [Omarleel](https://github.com/Omarleel) - Desarrollador principal