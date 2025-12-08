"""
Evaluation module for pattern detection results.
"""


def evaluar_deteccion(r_max, tau, es_positivo_esperado=True):
    """
    Evalúa si la detección fue correcta según el umbral y el resultado esperado.

    Args:
        r_max (float): Valor máximo de correlación normalizada
        tau (float): Umbral de detección
        es_positivo_esperado (bool): True si esperamos detectar el patrón (canción reggae),
                                     False si no esperamos detectarlo (canción no-reggae)

    Returns:
        dict: Diccionario con las siguientes claves:
            - detectado (bool): True si r_max > tau
            - tipo (str): Tipo de resultado: "VP", "VN", "FP", "FN"
            - mensaje (str): Mensaje descriptivo del resultado
            - r_max (float): Valor de correlación máxima

    Ejemplos:
        >>> # Canción reggae correctamente detectada
        >>> evaluar_deteccion(0.85, 0.01, es_positivo_esperado=True)
        {'detectado': True, 'tipo': 'VP', 'mensaje': '✓ DETECTADO (Verdadero Positivo)', 'r_max': 0.85}

        >>> # Canción no-reggae correctamente rechazada
        >>> evaluar_deteccion(0.005, 0.01, es_positivo_esperado=False)
        {'detectado': False, 'tipo': 'VN', 'mensaje': '✓ RECHAZADO (Verdadero Negativo)', 'r_max': 0.005}
    """
    detectado = r_max > tau

    if es_positivo_esperado and detectado:
        tipo = "VP"
        mensaje = "✓ DETECTADO (Verdadero Positivo)"
    elif es_positivo_esperado and not detectado:
        tipo = "FN"
        mensaje = "✗ NO DETECTADO (Falso Negativo)"
    elif not es_positivo_esperado and not detectado:
        tipo = "VN"
        mensaje = "✓ RECHAZADO (Verdadero Negativo)"
    else:  # not es_positivo_esperado and detectado
        tipo = "FP"
        mensaje = "✗ FALSA DETECCIÓN (Falso Positivo)"

    return {
        'detectado': detectado,
        'tipo': tipo,
        'mensaje': mensaje,
        'r_max': r_max
    }


def imprimir_resultado(resultado):
    """
    Imprime de forma formateada el resultado de la evaluación.

    Args:
        resultado (dict): Diccionario retornado por evaluar_deteccion()
    """
    print(f"\n--- Resultados ---")
    print(f"Pico Máximo de Correlación (R_max): {resultado['r_max']:.4f}")
    print(f"Tipo: {resultado['tipo']}")
    print(f"{resultado['mensaje']}")
