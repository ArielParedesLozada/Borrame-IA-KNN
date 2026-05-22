import numpy as np


def distancia_euclidea(p1: np.ndarray, p2: np.ndarray):
    d = np.linalg.norm(p1 - p2)
    return d
