"""Dataset loading and preprocessing helpers."""

from typing import Tuple

import numpy as np
import tensorflow as tf


CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 and return normalized images with integer labels."""
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.cifar10.load_data()
    )

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    return train_images, train_labels.flatten(), test_images, test_labels.flatten()


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Convert image arrays to float32 values in the [0, 1] range."""
    return images.astype("float32") / 255.0
