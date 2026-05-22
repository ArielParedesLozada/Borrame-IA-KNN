from src.knn import knn, predecir_masa_knn, predecir_masa_knn_fast
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import time

if __name__ == "__main__":
    (ds_train, ds_test), ds_info = tfds.load(
        "emnist/digits", split=["train", "test"], as_supervised=True, with_info=True
    )

    train_size = ds_info.splits["train"].num_examples
    test_size = ds_info.splits["test"].num_examples

    x_train, y_train = next(iter(tfds.as_numpy(ds_train.batch(train_size))))
    x_test, y_test = next(iter(tfds.as_numpy(ds_test.batch(test_size))))
    x_test, y_test = x_test[:100], y_test[:10]
    print(len(x_test))
    start1 = time.time()
    y_pred = predecir_masa_knn_fast(x_test, x_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    end1 = time.time()
    start2 = time.time()
    y_pred = predecir_masa_knn(x_test, x_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    end2 = time.time()
    print(f"Tiempo fast: {end1 - start1}, Promedio={(end1 - start1) / 10}")
    print(f"Tiempo fast: {end2 - start2}, Promedio={(end2 - start2) / 10}")
    # class_report = classification_report(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # num_classes = ds_info.features["label"].num_classes
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm, display_labels=range(num_classes)
    # )
    # disp.plot(cmap="Blues", values_format="d")
    # plt.title("Matriz de confusión")
    # plt.show()
