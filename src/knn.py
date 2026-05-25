import numpy as np
from collections import Counter
from src.distancias import distancia_euclidea, distancia_manhattan


def _aplanar_float(datos: np.ndarray):
    return np.asarray(datos).reshape(len(datos), -1).astype(np.float32, copy=False)


def _votar(etiquetas):
    conteo = Counter(etiquetas)
    max_votos = max(conteo.values())
    for etiqueta in etiquetas:
        if conteo[etiqueta] == max_votos:
            return etiqueta
    return etiquetas[0]


def _predecir_por_distancias(distancias, y_train, k):
    k = min(k, len(distancias))
    indices = np.argpartition(distancias, k - 1)[:k]
    indices = sorted(indices, key=lambda i: distancias[i])
    vecinos = [y_train[i] for i in indices]
    return _votar(vecinos)


def _actualizar_top_k(mejores_distancias, mejores_indices, nuevas_distancias, inicio, k):
    indices_nuevos = np.arange(inicio, inicio + nuevas_distancias.shape[0])
    indices_nuevos = np.broadcast_to(indices_nuevos[:, None], nuevas_distancias.shape)
    distancias_combinadas = np.vstack((mejores_distancias, nuevas_distancias))
    indices_combinados = np.vstack((mejores_indices, indices_nuevos))
    pos = np.argpartition(distancias_combinadas, k - 1, axis=0)[:k]
    columnas = np.arange(distancias_combinadas.shape[1])
    mejores_distancias = distancias_combinadas[pos, columnas]
    mejores_indices = indices_combinados[pos, columnas]
    orden = np.argsort(mejores_distancias, axis=0)
    mejores_distancias = np.take_along_axis(mejores_distancias, orden, axis=0)
    mejores_indices = np.take_along_axis(mejores_indices, orden, axis=0)
    return mejores_distancias, mejores_indices


def knn(x_train, y_train, nuevo_dato: np.ndarray, k=5, f_distancia=distancia_euclidea):
    distancias = []  # Guarda (distancia, etiqueta)
    for i in range(len(x_train)):
        distancia = f_distancia(x_train[i], nuevo_dato)
        distancias.append((distancia, y_train[i]))
    distancias.sort(key=lambda x: x[0])
    k = min(k, len(distancias))
    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(distancias[i][1])
    return _votar(nearest_neighbors)


def knn_fast(x_train, y_train, nuevo_dato, k=5, f_distancia=distancia_euclidea):
    distancias = []
    etiquetas = []
    for i in range(len(x_train)):
        d = f_distancia(x_train[i], nuevo_dato)
        distancias.append(d)
        etiquetas.append(y_train[i])
    k = min(k, len(distancias))
    indices = np.argpartition(distancias, k - 1)[:k]
    indices = sorted(indices, key=lambda i: distancias[i])
    nearest = [etiquetas[i] for i in indices]
    return _votar(nearest)


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


def predecir_masa_knn_vectorizado(
    x_test,
    x_train,
    y_train,
    k=5,
    f_distancia=distancia_euclidea,
    batch_size=16,
    train_chunk_size=2048,
):
    x_train_2d = _aplanar_float(x_train)
    x_test_2d = _aplanar_float(x_test)
    y_train = np.asarray(y_train)
    predicciones = []

    if f_distancia is distancia_euclidea:
        train_normas = np.einsum("ij,ij->i", x_train_2d, x_train_2d)
        for inicio in range(0, len(x_test_2d), batch_size):
            lote = x_test_2d[inicio : inicio + batch_size]
            lote_normas = np.einsum("ij,ij->i", lote, lote)
            distancias = train_normas[:, None] - 2 * x_train_2d @ lote.T
            distancias += lote_normas[None, :]
            np.maximum(distancias, 0, out=distancias)
            for columna in range(distancias.shape[1]):
                predicciones.append(
                    _predecir_por_distancias(distancias[:, columna], y_train, k)
                )
        return np.array(predicciones)

    if f_distancia is distancia_manhattan:
        k = min(k, len(x_train_2d))
        for inicio in range(0, len(x_test_2d), batch_size):
            lote = x_test_2d[inicio : inicio + batch_size]
            mejores_distancias = np.full((k, len(lote)), np.inf, dtype=np.float32)
            mejores_indices = np.full((k, len(lote)), -1, dtype=np.int64)
            for inicio_train in range(0, len(x_train_2d), train_chunk_size):
                fin_train = inicio_train + train_chunk_size
                bloque = x_train_2d[inicio_train:fin_train]
                distancias = np.abs(bloque[:, None, :] - lote[None, :, :]).sum(axis=2)
                mejores_distancias, mejores_indices = _actualizar_top_k(
                    mejores_distancias,
                    mejores_indices,
                    distancias,
                    inicio_train,
                    k,
                )
            for columna in range(mejores_indices.shape[1]):
                vecinos = [y_train[i] for i in mejores_indices[:, columna]]
                predicciones.append(_votar(vecinos))
        return np.array(predicciones)

    return predecir_masa_knn_fast(x_test, x_train, y_train, k, f_distancia)
