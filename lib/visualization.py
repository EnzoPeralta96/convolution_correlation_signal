"""
Visualization module for plotting signals, spectrograms, and correlation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_espectros(x, p, sr):
    """
    Grafica los espectrogramas de la canción y el patrón.

    Args:
        x (array): Señal de la canción
        p (array): Patrón a detectar
        sr (int): Sample rate (frecuencia de muestreo)
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    Sx = np.abs(librosa.stft(x))
    Sx_db = librosa.amplitude_to_db(Sx)

    librosa.display.specshow(
        Sx_db,
        sr=sr,
        x_axis='time',
        y_axis='log',
        cmap='magma'
    )

    plt.colorbar(format="%+2.0f dBFS")
    plt.title("Espectrograma Canción")

    plt.subplot(1, 2, 2)
    Sp = np.abs(librosa.stft(p))
    Sp_db = librosa.amplitude_to_db(Sp)

    librosa.display.specshow(
        Sp_db,
        sr=sr,
        x_axis='time',
        y_axis='log',
        cmap='magma'
    )

    plt.colorbar(format="%+2.0f dBFS")
    plt.title("Espectrograma Patrón")

    plt.tight_layout()
    plt.show()


def plot_canciones(x, p, sr):
    """
    Grafica las formas de onda de la canción y el patrón en el dominio del tiempo.

    Args:
        x (array): Señal de la canción
        p (array): Patrón a detectar
        sr (int): Sample rate (frecuencia de muestreo)
    """
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


def plot_resultado(r_xp_norm, lags, tau, sr):
    """
    Grafica el resultado de la correlación cruzada con el umbral de detección.

    Args:
        r_xp_norm (array): Correlación cruzada normalizada
        lags (array): Desplazamientos
        tau (float): Umbral de detección
        sr (int): Sample rate (frecuencia de muestreo)

    Returns:
        tuple: (R_max, t_peak) - Valor máximo de correlación y tiempo del pico
    """
    t = lags / sr

    idx_max = np.argmax(r_xp_norm)
    R_max = r_xp_norm[idx_max]
    t_peak = t[idx_max]

    plt.figure(figsize=(14, 6))
    plt.plot(t, r_xp_norm, color='red', label='Correlación normalizada')
    plt.axhline(tau, color='blue', linestyle='--', label='Umbral')
    plt.plot(t_peak, R_max, 'go', label=f'Pico: {t_peak:.4f}s')

    plt.xlabel("Desplazamiento (m)")
    plt.ylabel("Correlación")
    plt.title("Correlación cruzada")
    plt.grid()
    plt.legend()
    plt.show()

    return R_max, t_peak
