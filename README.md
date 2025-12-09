# Convolución y Correlación de Señales

## Resumen

El presente trabajo aborda el estudio de la **convolución** y la **correlación** de señales discretas, analizadas tanto en el dominio del tiempo como en el dominio de la frecuencia, con el objetivo de comprender su relación teórica y su aplicación práctica en el procesamiento digital de señales.

Se implementa un sistema experimental de análisis de audio que evalúa si una canción pertenece a un determinado género musical mediante la comparación con un patrón de referencia que representa un beat característico del género. Para ello, se calcula la **Correlación Cruzada** entre la señal de entrada y el patrón utilizando **Correlación Rápida por FFT** para reducir el costo computacional, validando la eficiencia del enfoque frecuencial frente al temporal.

### Correlación Cruzada Normalizada

La correlación normalizada se calcula dividiendo la correlación cruzada por la energía de las señales:

```
r_norm[m] = |r_xy[m]| / sqrt(E_x · E_p)
```

Donde:
- `E_x = Σ x²[n]` es la energía de la señal x
- `E_p = Σ p²[n]` es la energía del patrón p

Esta normalización permite obtener valores entre 0 y 1, facilitando la comparación con un umbral.

### Correlación Rápida por FFT

La correlación cruzada en el dominio del tiempo tiene complejidad `O(N²)`. Utilizando FFT, se reduce a `O(N log N)`:

```
r_xy[m] = IFFT( FFT(x) · conj(FFT(p)) )
```

Donde `conj()` denota el conjugado complejo.

---

## Estructura del Proyecto

```
convolution_correlation_signal/
├── README.md                    # Documentación del proyecto
├── experiments.ipynb            # Notebook con experimentos y evaluaciones
├── lib/                         # Librería de módulos reutilizables
│   ├── __init__.py             # Inicializador del paquete
│   ├── audio_io.py             # Módulo de carga de audio
│   ├── signal_processing.py    # Procesamiento de señales (correlación FFT)
│   ├── visualization.py        # Visualización de resultados
│   └── evaluation.py           # Evaluación y clasificación
└── data/                        # Archivos de audio para análisis (~630 MB)
    ├── patron.wav              # Patrón de referencia (ritmo reggaeton)
    ├── cancion_1.wav           # Canción reggaeton 1
    ├── cancion_2.wav           # Canción reggaeton 2
    └── cancion_no_reggae_*.wav # Canciones no-reggaeton (10 archivos)
```

## Descripción del Código

El sistema está organizado en una librería modular (`lib/`) y un notebook de experimentos (`experiments.ipynb`):

### 1. Configuración Inicial

```python
TASA_MUESTREO = 44100  # Tasa de muestreo estándar (44.1kHz)
TAU = 0.01             # Threshold para la correlación
```

- **Tasa de muestreo:** 44,100 Hz es el estándar para audio CD
- **Umbral (TAU):** Define el límite de similitud para clasificar la canción

### 2. Librería `lib/`

#### 2.1. Módulo `audio_io.py`

Este módulo proporciona funciones para cargar archivos de audio.

##### Función: `cargar_audio(audio_path, sr)`

```python
def cargar_audio(audio_path, sr):
    """
    Carga un archivo de audio.

    Args:
        audio_path (str): Ruta al archivo de audio
        sr (int): Sample rate (frecuencia de muestreo)

    Returns:
        numpy.ndarray: Señal de audio como array NumPy, o None si hay error
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
        print(f"Audio cargado: {audio_path} ({len(audio)} muestras)")
        return audio
    except Exception as e:
        print(f"Error al cargar archivo de audio: {e}")
        return None
```

**Descripción:**
- Carga un solo archivo de audio en formato WAV
- Remuestrea a la frecuencia especificada (44.1 kHz por defecto)
- Retorna la señal como array NumPy unidimensional
- Maneja errores de carga de forma robusta

**Uso típico:**
```python
patron = cargar_audio('data/patron.wav', 44100)
cancion = cargar_audio('data/cancion_1.wav', 44100)
```

##### Función: `cargar_audio_y_patron(cancion_path, patron_path, sr)`

```python
def cargar_audio_y_patron(cancion_path, patron_path, sr):
    """
    Carga archivos de audio para la canción y el patrón a detectar.

    Returns:
        tuple: (x, p) donde x es la señal de la canción y p es el patrón
    """
```

**Descripción:**
- Carga simultáneamente una canción y un patrón
- Útil para análisis standalone o scripts simples
- Retorna ambas señales como tupla (canción, patrón)

#### 2.2. Módulo `signal_processing.py`

**Función:** `correlacion_rapida_fft(x, p)`

```python
def correlacion_rapida_fft(x, p):
  N_x = len(x)
  N_p = len(p)
  N = N_x + N_p - 1

  # Transformadas de Fourier con zero-padding
  X = np.fft.fft(x, N)
  P = np.fft.fft(p, N)

  # Correlación en el dominio de la frecuencia
  r = np.fft.ifft(X * np.conjugate(P))
  r = np.real(r)

  # Centrar la correlación
  r = np.fft.fftshift(r)

  # Normalización por energía
  energy_x = np.sum(x**2)
  energy_p = np.sum(p**2)
  denom = np.sqrt(energy_x * energy_p)
  r_norm = np.abs(r / denom)

  # Crear vector de desplazamientos
  lags = np.arange(-(N_p - 1), N_x)
  return r_norm, lags
```

**Descripción paso a paso:**

1. **Dimensionamiento:** Calcula `N = N_x + N_p - 1` para evitar aliasing circular
2. **FFT:** Aplica transformada de Fourier a ambas señales con zero-padding
3. **Producto en frecuencia:** Multiplica `X` por el conjugado de `P`
4. **IFFT:** Transforma el resultado de vuelta al dominio del tiempo
5. **Centrado:** Usa `fftshift` para centrar la correlación en cero
6. **Normalización:** Divide por la raíz de las energías para obtener valores entre 0 y 1
7. **Lags:** Genera el vector de desplazamientos temporales

**Ventajas:**
- Complejidad computacional: `O(N log N)` vs `O(N²)` del método directo
- Resultados matemáticamente equivalentes al método temporal
- Escalable para señales de gran tamaño

#### 2.3. Módulo `visualization.py`

##### Función: `plot_espectros(x, p, sr)`

```python
def plot_espectros(x, p, sr):
    plt.figure(figsize=(15, 5))

    # Espectrograma de la canción
    plt.subplot(1, 2, 1)
    Sx = np.abs(librosa.stft(x))
    Sx_db = librosa.amplitude_to_db(Sx)
    librosa.display.specshow(Sx_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dBFS")
    plt.title("Espectrograma Canción")

    # Espectrograma del patrón
    plt.subplot(1, 2, 2)
    Sp = np.abs(librosa.stft(p))
    Sp_db = librosa.amplitude_to_db(Sp)
    librosa.display.specshow(Sp_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dBFS")
    plt.title("Espectrograma Patrón")

    plt.tight_layout()
    plt.show()
```

**Descripción:**
- Calcula la **STFT (Short-Time Fourier Transform)** de ambas señales
- Convierte las magnitudes a escala de **decibelios (dB)**
- Visualiza la evolución temporal del contenido frecuencial
- Usa escala logarítmica para el eje de frecuencias
- Mapa de color: `magma` para mejor visualización

**Interpretación:**
- Permite identificar patrones frecuenciales característicos
- El eje Y muestra frecuencias (Hz) en escala logarítmica
- El eje X muestra el tiempo (segundos)
- Los colores representan intensidad en dBFS

##### Función: `plot_canciones(x, p, sr)`

```python
def plot_canciones(x, p, sr):
  t_x = np.arange(len(x)) / sr
  t_p = np.arange(len(p)) / sr

  plt.figure(figsize=(14, 8))

  plt.subplot(2, 1, 1)
  plt.plot(t_x, x, color='black')
  plt.title("Señal de la Canción")
  plt.xlabel("Tiempo [s]")
  plt.ylabel("Amplitud")
  plt.grid(True)

  plt.subplot(2, 1, 2)
  plt.plot(t_p, p, color='green')
  plt.title("Patrón a detectar")
  plt.xlabel("Tiempo [s]")
  plt.ylabel("Amplitud")
  plt.grid(True)

  plt.tight_layout()
  plt.show()
```

**Descripción:**
- Convierte índices de muestras a tiempo (segundos)
- Grafica ambas señales en el dominio del tiempo
- Permite visualizar la forma de onda y duración relativa

##### Función: `plot_resultado(r_xp_norm, lags, tau, sr)`

```python
def plot_resultado(r_xp_norm, lags, tau, sr):
  t = lags / sr

  idx_max = np.argmax(r_xp_norm)
  R_max = r_xp_norm[idx_max]
  t_peak = t[idx_max]

  plt.figure(figsize=(14,6))
  plt.plot(t, r_xp_norm, color='red', label='Correlación normalizada')
  plt.axhline(tau, color='blue', linestyle='--', label='Umbral')
  plt.plot(t_peak, R_max, 'go', label=f'Pico: {t_peak:.4f}s')

  plt.xlabel("Desplazamiento temporal (s)")
  plt.ylabel("Similitud normalizada")
  plt.title("Correlación cruzada normalizada")
  plt.grid()
  plt.legend()
  plt.show()

  return R_max, t_peak
```

**Descripción:**
- Encuentra el **pico máximo de correlación** (`R_max`)
- Identifica el **desplazamiento temporal** donde ocurre el pico
- Grafica la correlación normalizada vs tiempo
- Muestra el umbral de decisión (`TAU`)
- Retorna el valor máximo y su posición temporal

**Interpretación:**
- Si `R_max > TAU`: Alta similitud, la canción pertenece al género
- Si `R_max ≤ TAU`: Baja similitud, la canción no pertenece al género

#### 2.4. Módulo `evaluation.py`

##### Función: `evaluar_deteccion(r_max, tau, es_positivo_esperado=True)`

Clasifica el resultado en una matriz de confusión:
- **VP (Verdadero Positivo)**: Reggaeton detectado correctamente (R_max > tau)
- **VN (Verdadero Negativo)**: No-reggaeton rechazado correctamente (R_max ≤ tau)
- **FP (Falso Positivo)**: No-reggaeton pero detectado como reggaeton
- **FN (Falso Negativo)**: Reggaeton pero no detectado

Retorna un diccionario con el tipo de resultado y mensaje descriptivo.

##### Función: `imprimir_resultado(resultado)`

Formatea e imprime los resultados de forma legible, mostrando R_max, tipo de resultado y mensaje.

### 3. Notebook `experiments.ipynb`

El notebook realiza experimentos sistemáticos para evaluar el sistema:

**Configuración Global:**
```python
TASA_MUESTREO = 44100  # Tasa de muestreo estándar (44.1kHz)
TAU = 0.01             # Umbral para la correlación
PATRON_FILE = os.path.join(os.getcwd(), 'data', 'patron.wav')
```

**Flujo de Experimentos:**
```python
# 1. Cargar patrón (una sola vez)
p = cargar_audio(PATRON_FILE, TASA_MUESTREO)

# 2. Para cada canción de prueba:
x = cargar_audio(CANCION_FILE, TASA_MUESTREO)

# 3. Calcular correlación
r, lags = correlacion_rapida_fft(x, p)

# 4. Visualizar señales y resultados
plot_canciones(x, p, TASA_MUESTREO)
plot_espectros(x, p, TASA_MUESTREO)
R_max, t_peak = plot_resultado(r, lags, TAU, TASA_MUESTREO)

# 5. Evaluar y clasificar
resultado = evaluar_deteccion(R_max, TAU, es_positivo_esperado=False)
imprimir_resultado(resultado)
```

**Casos de Prueba:**
- **Verdaderos Negativos (VN)**: 10 canciones no-reggaeton que deben ser rechazadas
- **Verdaderos Positivos (VP)**: 2 canciones reggaeton que deben ser detectadas

---

## Instalación y Uso

### Requisitos

```bash
pip install os numpy librosa matplotlib ipykernel jupyter
```

### Bibliotecas Utilizadas

- **NumPy:** Operaciones numéricas y arrays
- **Librosa:** Carga y análisis de audio
- **Matplotlib:** Visualización de datos y gráficos
- **IPykernel/Jupyter:** Ejecución de notebooks

### Instalación Paso a Paso

1. **Clonar el repositorio:**
```bash
git clone https://github.com/EnzoPeralta96/convolution_correlation_signal.git
cd convolution_correlation_signal
```

2. **Instalar dependencias:**
```bash
pip install numpy librosa matplotlib ipykernel jupyter
```



3. **Ejecutar el notebook de experimentos:**
```bash
jupyter notebook experiments.ipynb
```

O abrirlo directamente en VS Code con la extensión de Jupyter.

### Uso de la Librería

#### Opción 1: Cargar archivos individualmente (recomendado)

```python
import os
from lib import (
    cargar_audio,
    correlacion_rapida_fft,
    plot_espectros,
    plot_canciones,
    plot_resultado,
    evaluar_deteccion,
    imprimir_resultado
)

# Configuración
TASA_MUESTREO = 44100
TAU = 0.01

# Cargar patrón y canción por separado
patron_path = os.path.join('data', 'patron.wav')
cancion_path = os.path.join('data', 'cancion_1.wav')

p = cargar_audio(patron_path, TASA_MUESTREO)
x = cargar_audio(cancion_path, TASA_MUESTREO)

# Calcular correlación
r, lags = correlacion_rapida_fft(x, p)

# Visualizar y evaluar
plot_canciones(x, p, TASA_MUESTREO)
plot_espectros(x, p, TASA_MUESTREO)
R_max, t_peak = plot_resultado(r, lags, TAU, TASA_MUESTREO)

# Clasificar resultado
resultado = evaluar_deteccion(R_max, TAU, es_positivo_esperado=True)
imprimir_resultado(resultado)
```

#### Opción 2: Cargar ambos archivos simultáneamente

```python
from lib import cargar_audio_y_patron, correlacion_rapida_fft, ...

# Cargar patrón y canción en un solo paso
x, p = cargar_audio_y_patron(cancion_path, patron_path, TASA_MUESTREO)

# Continuar con el análisis...
```

---

---

## Autor

**Iñaki Poch - Javier Naccio - Enzo Peralta**
Proyecto Final - Computación Cientifica (CC) - Métodos Numéricos 2 (MN2-CC)
2025

---

## Licencia

Este proyecto es de uso académico y educativo.
