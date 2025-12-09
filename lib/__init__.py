"""
Library for audio signal processing and pattern detection.

Modules:
    - audio_io: Audio file loading and handling
    - signal_processing: Signal processing operations (correlation, FFT)
    - visualization: Plotting and visualization functions
    - evaluation: Pattern detection evaluation and classification
"""
from .audio_io import cargar_audio, cargar_audio_y_patron
from .signal_processing import correlacion_rapida_fft
from .visualization import plot_espectros, plot_canciones, plot_resultado
from .evaluation import evaluar_deteccion, imprimir_resultado

__all__ = [
    'cargar_audio',
    'cargar_audio_y_patron',
    'correlacion_rapida_fft',
    'plot_espectros',
    'plot_canciones',
    'plot_resultado',
    'evaluar_deteccion',
    'imprimir_resultado'
]
