"""Model evaluation helpers."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
) -> dict:
    """Return loss, accuracy, confusion matrix and a classification report."""
    loss, accuracy = model.evaluate(images, labels, verbose=0)
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(labels, predictions),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }
