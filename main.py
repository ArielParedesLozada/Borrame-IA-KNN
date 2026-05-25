from src.knn import predecir_masa_knn_vectorizado
from src.distancias import obtener_distancia
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import time

DISTANCIA = "manhattan"
N_TRAIN = 3000
N_TEST = 200

if __name__ == "__main__":
    f_distancia = obtener_distancia(DISTANCIA)

    (ds_train, ds_test), ds_info = tfds.load(
        "emnist/digits", split=["train", "test"], as_supervised=True, with_info=True
    )

    train_size = ds_info.splits["train"].num_examples
    test_size = ds_info.splits["test"].num_examples

    x_train, y_train = next(iter(tfds.as_numpy(ds_train.batch(train_size))))
    x_test, y_test = next(iter(tfds.as_numpy(ds_test.batch(test_size))))
    if N_TRAIN != -1:
        x_train, y_train = x_train[:N_TRAIN], y_train[:N_TRAIN]
    if N_TEST != -1:
        x_test, y_test = x_test[:N_TEST], y_test[:N_TEST]
    print(f"Entrenamiento: {len(x_train)}")
    print(f"Prueba: {len(x_test)}")
    start = time.time()
    y_pred = predecir_masa_knn_vectorizado(
        x_test,
        x_train,
        y_train,
        f_distancia=f_distancia,
        batch_size=16,
        train_chunk_size=2048,
    )
    accuracy = accuracy_score(y_test, y_pred)
    end = time.time()
    print(f"Distancia: {DISTANCIA}")
    print(f"Accuracy: {accuracy}")
    print(f"Tiempo: {end - start}, Promedio={(end - start) / len(x_test)}")
    # class_report = classification_report(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # num_classes = ds_info.features["label"].num_classes
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm, display_labels=range(num_classes)
    # )
    # disp.plot(cmap="Blues", values_format="d")
    # plt.title("Matriz de confusión")
    # plt.show()
