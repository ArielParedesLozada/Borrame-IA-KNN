import numpy as np
from collections import Counter
from src.distancias import distancia_euclidea


def knn(x_train, y_train, nuevo_dato: np.ndarray, k=5, f_distancia=distancia_euclidea):
    distancias = []  # Guarda (distancia, etiqueta)
    for i in range(len(x_train)):
        distancia = f_distancia(x_train[i], nuevo_dato)
        distancias.append((distancia, y_train[i]))
    distancias.sort(key=lambda x: x[0])
    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(distancias[i][1])
    return Counter(nearest_neighbors).most_common(1)[0][0]


def knn_fast(x_train, y_train, nuevo_dato, k=5, f_distancia=distancia_euclidea):
    distancias = []
    etiquetas = []
    for i in range(len(x_train)):
        d = f_distancia(x_train[i], nuevo_dato)
        distancias.append(d)
        etiquetas.append(y_train[i])
    k = min(k, len(distancias))
    indices = np.argpartition(distancias, k - 1)[:k]
    nearest = [etiquetas[i] for i in indices]
    return Counter(nearest).most_common(1)[0][0]


def predecir_masa_knn(x_test, x_train, y_train, k=5, f_distancia=distancia_euclidea):
    predicciones = []
    for dato in x_test:
        pred = knn(x_train, y_train, dato, k, f_distancia)
        predicciones.append(pred)
    return np.array(predicciones)


def predecir_masa_knn_fast(
    x_test, x_train, y_train, k=5, f_distancia=distancia_euclidea
):
    predicciones = []
    for dato in x_test:
        pred = knn_fast(x_train, y_train, dato, k, f_distancia)
        predicciones.append(pred)
    return np.array(predicciones)
