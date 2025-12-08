"""
Signal processing module for correlation and FFT operations.
"""

import numpy as np


def correlacion_rapida_fft(x, p):
    """
    Calcula la correlación cruzada normalizada entre dos señales usando FFT.

    Args:
        x (array): Señal de la canción
        p (array): Patrón a buscar

    Returns:
        tuple: (r_norm, lags) donde:
            - r_norm: Correlación cruzada normalizada
            - lags: Desplazamientos correspondientes a cada valor de correlación
    """
    N_x = len(x)
    N_p = len(p)
    N = N_x + N_p - 1

    X = np.fft.fft(x, N)
    P = np.fft.fft(p, N)

    r = np.fft.ifft(X * np.conjugate(P))
    r = np.real(r)

    r = np.fft.fftshift(r)

    energy_x = np.sum(x**2)
    energy_p = np.sum(p**2)

    denom = np.sqrt(energy_x * energy_p)
    r_norm = np.abs(r / denom)

    lags = np.arange(-(N_p - 1), N_x)
    return r_norm, lags
