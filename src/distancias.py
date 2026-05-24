import numpy as np


def distancia_euclidea(p1: np.ndarray, p2: np.ndarray):
    d = np.linalg.norm(p1 - p2)
    return d


def distancia_manhattan(p1: np.ndarray, p2: np.ndarray):
    d = np.sum(np.abs(p1 - p2))
    return d


DISTANCIAS = {
    "euclidea": distancia_euclidea,
    "manhattan": distancia_manhattan,
}


def obtener_distancia(nombre: str):
    nombre = nombre.lower().strip()
    if nombre not in DISTANCIAS:
        disponibles = ", ".join(DISTANCIAS.keys())
        raise ValueError(f"Distancia no valida: {nombre}. Opciones: {disponibles}")
    return DISTANCIAS[nombre]
