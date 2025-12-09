"""
Audio I/O module for loading audio files.
"""

import librosa


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


def cargar_audio_y_patron(cancion_path, patron_path, sr):
    """
    Carga archivos de audio para la canción y el patrón a detectar.

    Args:
        cancion_path (str): Ruta al archivo de audio de la canción
        patron_path (str): Ruta al archivo de audio del patrón
        sr (int): Sample rate (frecuencia de muestreo)

    Returns:
        tuple: (x, p) donde x es la señal de la canción y p es el patrón,
               o (None, None) si hay error
    """
    try:
        x, _ = librosa.load(cancion_path, sr=sr)
        p, _ = librosa.load(patron_path, sr=sr)
        print(f"Canción cargada: {cancion_path} ({len(x)} muestras)")
        print(f"Patrón cargado: {patron_path} ({len(p)} muestras)")
        return x, p
    except Exception as e:
        print(f"Error al cargar archivos de audio: {e}")
        return None, None
