# Memorias del Proyecto - Convolución y Correlación de Señales

## Información General
- **Nombre del Proyecto:** Convolution and Correlation of Signals
- **Curso:** MN2-CC-Proyecto Final
- **Autor:** Enzo Peralta
- **Fecha de Exploración:** 2025-12-04

## Estructura del Proyecto

```
convolution_correlation_signal/
├── .git/                          # Git repository metadata
├── Código.ipynb                   # Main Python notebook (254 lines)
├── README.md                      # Project documentation
├── prompt.txt                     # Archivo de instrucciones (vacío actualmente)
└── MEMORIAS_PROYECTO.md          # Este archivo
```

## Archivos Principales

### 1. Código.ipynb (1.05 MB)
- **Tipo:** Jupyter Notebook (Python)
- **Propósito:** Archivo principal de implementación del proyecto
- **Componentes Clave:**
  - Integración con Google Drive para acceso a archivos
  - Carga de archivos de audio usando librería `librosa`
  - Cálculo de correlación cruzada usando FFT (Fast Fourier Transform)
  - Visualización y análisis de señales
  - Detección de género musical mediante coincidencia de patrones

### 2. README.md
- **Contenido:** Título básico del proyecto
- **Propósito:** Documentación básica del proyecto

### 3. prompt.txt
- **Estado:** Vacío (0 bytes)
- **Propósito:** Pendiente de instrucciones

## Tecnologías y Dependencias

### Lenguajes de Programación
- Python 3
- Jupyter Notebook (.ipynb)

### Bibliotecas Python
- `numpy` - Cálculos numéricos
- `librosa` - Procesamiento y análisis de audio
- `matplotlib` - Visualización de datos
- `google.colab` - Integración con Google Colab

## Configuración Técnica

### Parámetros del Proyecto
- **Frecuencia de Muestreo:** 44,100 Hz (estándar para audio)
- **Umbral de Correlación (TAU):** 0.01
- **Archivos de Audio en Google Drive:**
  - Canción: `/content/drive/MyDrive/MN2-CC-Proyecto Final/cancion_no_reggae_3.wav`
  - Patrón: `/content/drive/MyDrive/MN2-CC-Proyecto Final/patron.wav`

## Repositorio Git

- **URL Remoto:** https://github.com/EnzoPeralta96/convolution_correlation_signal.git
- **Rama Actual:** main
- **Estado:** Limpio (sin cambios sin confirmar)
- **Commits Recientes:**
  - `198ee54` - "Creado con Colab"
  - `23408de` - "primer commit"

## Propósito y Funcionalidad del Proyecto

Este es un proyecto académico que implementa procesamiento de señales de audio usando conceptos de convolución y correlación.

### Objetivos Principales:
1. **Carga de Audio:** Cargar y procesar archivos de audio (canciones y patrones)
2. **Correlación Rápida:** Implementar correlación cruzada usando FFT para eficiencia
3. **Visualización de Señales:** Graficar formas de onda y espectrogramas
4. **Coincidencia de Patrones:** Detectar si una canción coincide con un patrón/género específico
5. **Normalización:** Normalizar el resultado de correlación para compararlo con un umbral
6. **Clasificación:** Determinar si una canción pertenece a un género específico

## Algoritmos Implementados

- **Fast Fourier Transform (FFT)** para cálculo eficiente de correlación
- **Correlación Cruzada Normalizada** para coincidencia de patrones
- **Análisis Espectral** usando Short-Time Fourier Transform (STFT)

## Entorno de Desarrollo

- **Plataforma:** Windows (win32)
- **Ejecución:** Google Colab (entorno Jupyter basado en la nube)
- **IDE:** Jupyter Notebook

## Notas Adicionales

- El proyecto se centra en la aplicación de conceptos de procesamiento de señales al análisis de audio
- Utiliza técnicas de coincidencia de patrones basadas en correlación para clasificación de género musical
- Diseñado para ejecutarse en Google Colab con archivos almacenados en Google Drive

---

**Última actualización:** 2025-12-04
