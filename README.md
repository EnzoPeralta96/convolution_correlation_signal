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

## Descripción del Código

El código implementado en `Código.ipynb` realiza el siguiente flujo de trabajo:

### 1. Configuración Inicial

```python
TASA_MUESTREO = 44100  # Tasa de muestreo estándar (44.1kHz)
TAU = 0.01             # Threshold para la correlación
```

- **Tasa de muestreo:** 44,100 Hz es el estándar para audio CD
- **Umbral (TAU):** Define el límite de similitud para clasificar la canción

### 2. Carga de Archivos de Audio

**Función:** `cargar_audio_y_patron(cancion_path, patron_path, sr)`

```python
def cargar_audio_y_patron(cancion_path, patron_path, sr):
  try:
    x, _ = librosa.load(cancion_path, sr=sr)
    p, _ = librosa.load(patron_path, sr=sr)
    print(f"Canción cargada: {cancion_path} ({len(x)} muestras)")
    print(f"Patrón cargado: {patron_path} ({len(p)} muestras)")
    return x, p
  except Exception as e:
    print(f"Error al cargar archivos de audio: {e}")
    return None, None
```

**Descripción:**
- Utiliza `librosa.load()` para cargar archivos de audio en formato WAV
- Los archivos se remuestrean a la tasa especificada (44.1 kHz)
- Retorna las señales como arrays NumPy unidimensionales
- Maneja errores de carga de forma robusta

### 3. Correlación Rápida por FFT

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

### 4. Visualización de Espectrogramas

**Función:** `plot_espectros(x, p, sr)`

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

### 5. Visualización de Señales en el Tiempo

**Función:** `plot_canciones(x, p, sr)`

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

### 6. Visualización del Resultado de Correlación

**Función:** `plot_resultado(r_xp_norm, lags, tau, sr)`

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

### 7. Flujo Principal de Ejecución

```python
# Cargar señales
x, p = cargar_audio_y_patron(CANCION_FILE, PATRON_FILE, TASA_MUESTREO)

# Calcular correlación
r, lags = correlacion_rapida_fft(x, p)

# Visualizar señales
plot_canciones(x, p, TASA_MUESTREO)
plot_espectros(x, p, TASA_MUESTREO)

# Visualizar correlación y obtener resultado
R_max, t_peak = plot_resultado(r, lags, TAU, TASA_MUESTREO)

# Clasificación
print("\n--- Resultados ---")
print(f"Pico Máximo de Correlación (R_max): {R_max:.4f}")

if R_max > TAU:
  print("\nConclusión: La canción pertenece al género")
else:
  print("\nConclusión: La canción no pertenece al género")
```

---

## Instalación y Uso

### Requisitos

```bash
pip install numpy librosa matplotlib
```

### Bibliotecas Utilizadas

- **NumPy:** Operaciones numéricas y arrays
- **Librosa:** Carga y análisis de audio
- **Matplotlib:** Visualización de datos

### Ejecución

1. Montar Google Drive (si se usa en Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Modificar las rutas de los archivos de audio:
```python
CANCION_FILE = 'ruta/a/tu/cancion.wav'
PATRON_FILE = 'ruta/a/tu/patron.wav'
```

3. Ejecutar todas las celdas del notebook `Código.ipynb`

---

## Resultados y Análisis

### Ventajas del Método Implementado

1. **Eficiencia Computacional:** Uso de FFT reduce complejidad de `O(N²)` a `O(N log N)`
2. **Normalización:** Valores entre 0 y 1 facilitan la interpretación
3. **Visualización Completa:** Espectrogramas y señales temporales ayudan a entender el análisis
4. **Robustez:** Manejo de errores en carga de archivos

### Limitaciones

1. **Falsos Positivos:** El enfoque puede generar falsos positivos dependiendo del tamaño del problema y la calidad del patrón de referencia
2. **Sensibilidad al Umbral:** El valor de `TAU` debe ajustarse según el caso de uso
3. **Patrones Complejos:** Géneros con variabilidad rítmica pueden ser difíciles de clasificar con un solo patrón

### Aplicaciones Extendidas

Este mismo enfoque puede aplicarse a:
- Detección de eventos en señales biomédicas (ECG, EEG)
- Reconocimiento de voz y comandos
- Sincronización de señales de audio/video
- Identificación de huellas acústicas (audio fingerprinting)

---

## Conclusiones

Los resultados obtenidos validan que:

1. La **correlación cruzada** es una herramienta efectiva para detectar similitudes temporales entre señales
2. El uso de **FFT** permite realizar estos cálculos de forma eficiente incluso en señales de gran tamaño
3. La **normalización** de la correlación facilita la comparación con umbrales establecidos
4. El enfoque implementado es aplicable a problemas reales de clasificación de audio

Sin embargo, se debe considerar que:
- El método puede generar **falsos positivos** según el patrón elegido
- Se requiere un **ajuste cuidadoso del umbral** para cada aplicación
- Para clasificación robusta, se recomienda combinar con otras técnicas (características espectrales, aprendizaje automático)

---

## Referencias

- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*. Pearson.
- McFee, B., et al. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy.
- Cooley, J. W., & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series*. Mathematics of Computation.

---

## Autor

**Enzo Peralta**
Proyecto Final - Métodos Numéricos 2 (MN2-CC)
2025

---

## Licencia

Este proyecto es de uso académico y educativo.
