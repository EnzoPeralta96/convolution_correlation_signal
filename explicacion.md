# Explicación Detallada del Código

## Explicación Paso a Paso del Proyecto de Correlación de Señales

---

## 📦 **1. Importaciones y Configuración Inicial**

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt

TASA_MUESTREO = 44100
TAU = 0.01
```

**¿Qué hace?**
- **NumPy:** Es la biblioteca fundamental para hacer operaciones matemáticas con arrays (vectores/matrices)
- **Librosa:** Especializada en procesamiento de audio, permite cargar archivos WAV/MP3 y hacer análisis espectral
- **Matplotlib:** Para crear gráficos y visualizaciones

**Parámetros:**
- **TASA_MUESTREO = 44100 Hz:** Es la frecuencia estándar de audio CD. Significa que se toman 44,100 muestras por segundo
- **TAU = 0.01:** Es el **umbral de decisión**. Si la correlación es mayor a 0.01, consideramos que la canción pertenece al género

---

## 🎵 **2. Función `cargar_audio_y_patron()`**

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

**¿Qué hace?**
1. **`librosa.load()`** lee el archivo de audio y lo convierte en un array de números (la señal digital)
2. El parámetro `sr=sr` le dice que queremos remuestrear a 44,100 Hz
3. Devuelve **dos valores**: la señal (`x` o `p`) y la tasa de muestreo (que ignoramos con `_`)
4. Si hay un error (archivo no existe, formato incorrecto), lo captura y devuelve `None`

**Resultado:**
- `x`: Array NumPy con las muestras de la canción (puede tener millones de valores)
- `p`: Array NumPy con las muestras del patrón de referencia

---

## 🔬 **3. Función `correlacion_rapida_fft()` - EL CORAZÓN DEL PROYECTO**

Esta es la función más importante. Vamos a descomponerla paso a paso:

### **Paso 1: Determinar el tamaño de salida**
```python
N_x = len(x)      # ej: 10,000,000 muestras
N_p = len(p)      # ej: 100,000 muestras
N = N_x + N_p - 1 # ej: 10,099,999
```

**¿Por qué `N_x + N_p - 1`?**
- Cuando correlacionas dos señales, el resultado tiene un tamaño específico
- Imagina que deslizas el patrón `p` sobre la canción `x` desde el inicio hasta el final
- La cantidad total de posiciones es `N_x + N_p - 1`
- Este tamaño evita el "aliasing circular" de la FFT

### **Paso 2: Aplicar FFT con zero-padding**
```python
X = np.fft.fft(x, N)
P = np.fft.fft(p, N)
```

**¿Qué está pasando aquí?**
- `np.fft.fft(señal, N)` calcula la **Transformada Rápida de Fourier**
- Convierte la señal del **dominio del tiempo** al **dominio de la frecuencia**
- El parámetro `N` hace **zero-padding** (rellena con ceros) para evitar efectos circulares
- `X` y `P` ahora son arrays de **números complejos** que representan el espectro

**Ejemplo conceptual:**
```
Tiempo:      [1, 2, 3, 4, 5]
FFT →        [15+0j, -2.5+3.4j, -2.5+0.8j, -2.5-0.8j, -2.5-3.4j]
Frecuencia:  Componentes de diferentes frecuencias
```

### **Paso 3: Correlación en el dominio de la frecuencia**
```python
r = np.fft.ifft(X * np.conjugate(P))
r = np.real(r)
```

**¡Aquí está la magia! El Teorema de la Convolución:**

**En el dominio del tiempo:**
```
Correlación = Σ x[k] · p[m+k]  →  Complejidad O(N²)
```

**En el dominio de la frecuencia:**
```
Correlación = IFFT( FFT(x) · conj(FFT(p)) )  →  Complejidad O(N log N)
```

**¿Qué significa `conjugate`?**
- Dado un número complejo `a + bi`, su conjugado es `a - bi`
- Esto es necesario matemáticamente para que la correlación funcione correctamente en frecuencia

**¿Por qué `np.real()`?**
- La `IFFT` puede devolver valores con componentes imaginarias pequeñas por errores numéricos
- `np.real()` toma solo la parte real

### **Paso 4: Centrar la correlación**
```python
r = np.fft.fftshift(r)
```

**¿Por qué centrar?**
- La FFT devuelve los resultados en un orden específico (de 0 a N-1)
- `fftshift` reorganiza para que el desplazamiento cero esté en el centro
- Facilita la interpretación: negativos a la izquierda, positivos a la derecha

### **Paso 5: Normalización por energía**
```python
energy_x = np.sum(x**2)
energy_p = np.sum(p**2)
denom = np.sqrt(energy_x * energy_p)
r_norm = np.abs(r / denom)
```

**¿Por qué normalizar?**
- Sin normalización, la correlación depende de la amplitud de las señales
- Una canción más "fuerte" (mayor amplitud) tendría correlación mayor sin ser más similar
- La normalización convierte el resultado a un rango **[0, 1]** donde:
  - **1 = perfecta similitud**
  - **0 = sin similitud**

**Fórmula:**
```
r_normalizada = |r| / sqrt(Energía_x · Energía_p)
```

### **Paso 6: Crear vector de desplazamientos (lags)**
```python
lags = np.arange(-(N_p - 1), N_x)
return r_norm, lags
```

**¿Qué son los "lags"?**
- Son los **desplazamientos temporales** en número de muestras
- Van desde `-(N_p - 1)` (patrón totalmente a la izquierda) hasta `N_x` (totalmente a la derecha)
- Cada `lag` representa una posición donde el patrón se alinea con la canción

---

## 📊 **4. Función `plot_espectros()`**

```python
def plot_espectros(x, p, sr):
    Sx = np.abs(librosa.stft(x))
    Sx_db = librosa.amplitude_to_db(Sx)
    librosa.display.specshow(Sx_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
```

**¿Qué hace?**
1. **`librosa.stft(x)`:** Calcula la **STFT (Short-Time Fourier Transform)**
   - Divide la señal en ventanas pequeñas
   - Aplica FFT a cada ventana
   - Resultado: matriz donde cada columna es un espectro en un momento del tiempo

2. **`amplitude_to_db()`:** Convierte a **decibelios (dB)**
   - Los humanos percibimos el volumen en escala logarítmica
   - dB hace más visible el contenido de baja amplitud

3. **`specshow()`:** Grafica el espectrograma
   - **Eje X:** Tiempo (segundos)
   - **Eje Y:** Frecuencia (Hz, escala logarítmica)
   - **Color:** Intensidad (dBFS)

**Interpretación:**
- Zonas brillantes = frecuencias con mucha energía
- Zonas oscuras = frecuencias con poca energía
- Permite ver visualmente el "beat" y patrones rítmicos

---

## 📈 **5. Función `plot_canciones()`**

```python
t_x = np.arange(len(x)) / sr
t_p = np.arange(len(p)) / sr
```

**¿Qué hace?**
- Convierte índices de muestras a **tiempo en segundos**
- Si hay 44,100 muestras y sr=44,100, entonces la duración es 1 segundo
- Luego grafica la **forma de onda** (amplitud vs tiempo)

**Ejemplo:**
```
Muestras:  [0, 44100, 88200, 132300, ...]
Tiempo:    [0s, 1s, 2s, 3s, ...]
```

---

## 🎯 **6. Función `plot_resultado()`**

```python
idx_max = np.argmax(r_xp_norm)
R_max = r_xp_norm[idx_max]
t_peak = t[idx_max]
```

**¿Qué hace?**
1. **`np.argmax()`:** Encuentra el **índice** del valor máximo de correlación
2. **`R_max`:** El **valor máximo** de similitud (entre 0 y 1)
3. **`t_peak`:** El **momento en el tiempo** donde ocurre la máxima similitud

**Interpretación:**
- Si `R_max = 0.85` → Hay 85% de similitud entre la canción y el patrón
- Si `t_peak = 23.5s` → La máxima similitud ocurre a los 23.5 segundos de la canción

**Visualización:**
- Grafica la correlación normalizada en función del desplazamiento temporal
- Marca el umbral `TAU` con una línea horizontal azul
- Marca el pico máximo con un punto verde

---

## 🚀 **7. Flujo Principal de Ejecución**

```python
x, p = cargar_audio_y_patron(CANCION_FILE, PATRON_FILE, TASA_MUESTREO)
r, lags = correlacion_rapida_fft(x, p)
R_max, t_peak = plot_resultado(r, lags, TAU, TASA_MUESTREO)

if R_max > TAU:
  print("La canción pertenece al género")
else:
  print("La canción no pertenece al género")
```

**Flujo completo:**
1. **Carga** los archivos de audio (canción y patrón)
2. **Calcula** la correlación rápida usando FFT
3. **Visualiza** las señales en el tiempo y sus espectrogramas
4. **Encuentra** el pico máximo de correlación
5. **Clasifica:** Compara `R_max` con el umbral `TAU` para decidir

---

## 🧮 **Ejemplo Numérico Simplificado**

Para entender mejor, veamos un ejemplo con números pequeños:

Imagina que tienes:
- Canción `x = [1, 2, 3, 4, 5, 2, 1]`
- Patrón `p = [3, 4, 5]`

**Correlación manual (sin normalizar):**
```
Desplazamiento 0: 1·3 + 2·4 + 3·5 = 3 + 8 + 15 = 26
Desplazamiento 1: 2·3 + 3·4 + 4·5 = 6 + 12 + 20 = 38
Desplazamiento 2: 3·3 + 4·4 + 5·5 = 9 + 16 + 25 = 50  ← Máximo!
Desplazamiento 3: 4·3 + 5·4 + 2·5 = 12 + 20 + 10 = 42
Desplazamiento 4: 5·3 + 2·4 + 1·5 = 15 + 8 + 5 = 28
```

El **pico máximo** ocurre en desplazamiento 2, donde el patrón `[3,4,5]` se alinea perfectamente con `[3,4,5]` de la canción.

Con **FFT**, este cálculo se hace muchísimo más rápido para señales de millones de muestras.

---

## ⚡ **Ventajas del Método FFT**

### Comparación de Complejidad Computacional

| Método | Complejidad | Tiempo (para N=1,000,000) |
|--------|-------------|---------------------------|
| Correlación directa | O(N²) | ~277 horas |
| Correlación con FFT | O(N log N) | ~20 segundos |

**¡La FFT es aproximadamente 50,000 veces más rápida!**

### ¿Por qué es tan rápida?

**Método directo:**
- Para cada posición del patrón (N posiciones)
- Multiplicas y sumas N valores
- Total: N × N = N² operaciones

**Método FFT:**
- FFT de x: N log N operaciones
- FFT de p: N log N operaciones
- Multiplicación: N operaciones
- IFFT: N log N operaciones
- Total: ≈ 3N log N operaciones

**Ejemplo numérico:**
```
N = 1,000,000 muestras

Método directo:
  1,000,000² = 1,000,000,000,000 operaciones (1 billón)

Método FFT:
  3 × 1,000,000 × log₂(1,000,000) ≈ 60,000,000 operaciones (60 millones)

Ganancia: 1,000,000,000,000 / 60,000,000 ≈ 16,666 veces más rápido
```

---

## 🎓 **Conceptos Teóricos Fundamentales**

### 1. ¿Qué es la Correlación Cruzada?

La correlación mide **qué tan similar** es una señal a otra cuando la desplazas en el tiempo.

**Matemáticamente:**
```
r_xy[m] = Σ(k=-∞ to ∞) x[k] · y[m + k]
```

**En palabras simples:**
- Tomas el patrón `y`
- Lo desplazas `m` posiciones
- Multiplicas punto a punto con `x`
- Sumas todos los productos
- El resultado es qué tan bien "encajan" en ese desplazamiento

### 2. ¿Por qué funciona la FFT para calcular correlación?

**Teorema de la Correlación:**
```
Correlación(x, p) en tiempo ↔ FFT(x) · conj(FFT(p)) en frecuencia
```

**Analogía:**
- Es como si en lugar de verificar cada posición una por una (método directo)
- Analizaras todas las frecuencias simultáneamente (método FFT)
- Y la multiplicación en frecuencia te da el mismo resultado que la suma en tiempo

### 3. ¿Qué significa el resultado?

**Valor de R_max:**
- **0.0 a 0.3:** Muy baja similitud - No pertenece al género
- **0.3 a 0.6:** Similitud moderada - Dudoso
- **0.6 a 0.8:** Alta similitud - Probablemente pertenece
- **0.8 a 1.0:** Muy alta similitud - Definitivamente pertenece

**Posición del pico (t_peak):**
- Te dice **en qué momento** de la canción aparece el patrón más similar
- Útil para detectar el inicio del beat característico

---

## 🔍 **Detalles Técnicos Importantes**

### Zero-Padding

```python
X = np.fft.fft(x, N)  # N > len(x)
```

**¿Por qué agregar ceros?**
- La FFT asume que la señal es **periódica** (se repite infinitamente)
- Sin zero-padding, el final de `x` se "conecta" con el inicio → aliasing circular
- Con zero-padding de tamaño `N = N_x + N_p - 1`, evitamos este problema

### Conjugado Complejo

```python
r = np.fft.ifft(X * np.conjugate(P))
```

**¿Por qué conjugar?**
- La correlación requiere **invertir** el patrón en el dominio del tiempo
- En el dominio de la frecuencia, invertir en tiempo = conjugar
- Esto es matemáticamente equivalente a la definición de correlación

### FFT Shift

```python
r = np.fft.fftshift(r)
```

**Antes del shift:**
```
[r[0], r[1], ..., r[N/2], r[N/2+1], ..., r[N-1]]
 (positivo)            (negativo)
```

**Después del shift:**
```
[r[N/2+1], ..., r[N-1], r[0], r[1], ..., r[N/2]]
      (negativo)              (positivo)
```

Ahora el índice central corresponde a desplazamiento cero.

---

## 🎯 **Aplicaciones Prácticas**

Este mismo código puede adaptarse para:

### 1. Detección de Beats en Música
- Patrón: Un beat de batería
- Señal: Una canción completa
- Resultado: Ubicación de todos los beats

### 2. Reconocimiento de Voz
- Patrón: "Hola" grabado
- Señal: Audio de una conversación
- Resultado: Detectar cuándo se dijo "Hola"

### 3. Análisis de ECG (Electrocardiograma)
- Patrón: Latido normal
- Señal: ECG de un paciente
- Resultado: Detectar latidos anormales

### 4. Sincronización de Audio/Video
- Patrón: Audio de referencia
- Señal: Audio grabado con delay
- Resultado: Cuánto tiempo de retraso hay

---

## 📊 **Interpretación de los Gráficos**

### Espectrograma
```
Tiempo (s) →
↑
Frecuencia (Hz)

Color: Intensidad (dB)
```

**¿Qué buscar?**
- Líneas horizontales brillantes = tonos constantes (notas musicales)
- Bandas verticales = eventos percusivos (golpes de batería)
- Patrones repetitivos = ritmo/beat del género

### Señal en el Tiempo
```
Tiempo (s) →
↑
Amplitud

```

**¿Qué buscar?**
- Amplitud alta = sonido fuerte
- Patrones periódicos = ritmo
- Duración de la señal

### Correlación Normalizada
```
Desplazamiento (s) →
↑
Similitud (0 a 1)

Pico = mejor alineamiento
```

**¿Qué buscar?**
- El pico más alto = mayor similitud
- La posición del pico = dónde está el patrón en la canción
- Múltiples picos = el patrón se repite varias veces

---

## 🛠️ **Posibles Mejoras al Código**

### 1. Múltiples Patrones
```python
patrones = ['patron_reggae.wav', 'patron_rock.wav', 'patron_jazz.wav']
for patron in patrones:
    p, _ = librosa.load(patron, sr=TASA_MUESTREO)
    r, lags = correlacion_rapida_fft(x, p)
    # Clasificar según el máximo de todos
```

### 2. Ajuste Automático del Umbral
```python
# En lugar de TAU fijo, calcular estadísticamente
TAU = np.mean(r_norm) + 2 * np.std(r_norm)
```

### 3. Detección de Múltiples Ocurrencias
```python
# Encontrar todos los picos, no solo el máximo
from scipy.signal import find_peaks
peaks, _ = find_peaks(r_norm, height=TAU)
print(f"Patrón encontrado en {len(peaks)} posiciones")
```

### 4. Análisis de Características Adicionales
```python
# Combinar correlación con características espectrales
tempo, _ = librosa.beat.beat_track(x, sr=TASA_MUESTREO)
spectral_centroid = librosa.feature.spectral_centroid(x, sr=TASA_MUESTREO)
# Usar todo junto para mejor clasificación
```

---

## 📚 **Resumen de Conceptos Clave**

1. **Correlación:** Mide similitud entre señales desplazadas en el tiempo
2. **FFT:** Convierte tiempo → frecuencia (rápido)
3. **IFFT:** Convierte frecuencia → tiempo (rápido)
4. **Teorema de la Correlación:** Permite calcular correlación en dominio de frecuencia
5. **Normalización:** Hace que el resultado sea independiente de la amplitud
6. **Zero-padding:** Evita aliasing circular en la FFT
7. **Conjugado:** Necesario matemáticamente para la correlación en frecuencia
8. **Lags:** Desplazamientos temporales donde se evalúa la similitud

---

## 💡 **Consejos para el Análisis**

### Elegir un Buen Patrón
- **Duración:** No muy largo (< 5 segundos) ni muy corto (> 0.5 segundos)
- **Representativo:** Debe capturar la esencia del género
- **Limpio:** Sin ruido de fondo, solo el beat característico
- **Energético:** Con suficiente amplitud en las frecuencias de interés

### Ajustar el Umbral TAU
- **TAU muy bajo (ej: 0.001):** Muchos falsos positivos
- **TAU muy alto (ej: 0.5):** Muchos falsos negativos
- **TAU óptimo:** Depende del corpus de canciones
- **Recomendación:** Probar con varias canciones conocidas y ajustar

### Interpretar los Resultados
- **R_max alto, pero en posición extraña:** Podría ser una coincidencia
- **Múltiples picos similares:** El patrón se repite → buena señal
- **Pico muy estrecho:** Coincidencia muy específica
- **Pico ancho:** Similitud sostenida en el tiempo

---

## 🎓 **Conclusión**

Este código demuestra de forma práctica y eficiente:

✅ Cómo aplicar **teoría de señales** a problemas reales
✅ La importancia de la **FFT** para eficiencia computacional
✅ Técnicas de **visualización** para análisis de audio
✅ **Clasificación automática** basada en similitud de patrones

La implementación es:
- **Matemáticamente correcta:** Usa las fórmulas estándar de correlación
- **Computacionalmente eficiente:** Aprovecha FFT para reducir complejidad
- **Prácticamente útil:** Se puede aplicar a clasificación de audio real
- **Educativamente valiosa:** Ilustra conceptos fundamentales de DSP

---

**¡Excelente trabajo implementando este proyecto!** 🎉
