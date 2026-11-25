"""Plotting helpers for workshop notebooks and training scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def show_image_grid(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
    number_of_images: int = 12,
) -> None:
    """Display a small grid of images with their class labels."""
    number_of_images = min(number_of_images, len(images))
    columns = 4
    rows = int(np.ceil(number_of_images / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < number_of_images:
            axis.imshow(images[index])
            axis.set_title(class_names[int(labels[index])])

    figure.tight_layout()
    plt.show()


def save_training_curves(history: tf.keras.callbacks.History, output_path: str | Path) -> None:
    """Save training and validation accuracy/loss curves."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
