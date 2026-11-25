"""Prediction helpers for saved image-classification models."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(image_path: str | Path, image_size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load an image, resize it and return a normalized batch."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: tuple[str, ...],
) -> tuple[str, float]:
    """Predict the most likely class and return its confidence."""
    image_batch = load_image(image_path)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return class_names[predicted_index], float(probabilities[predicted_index])
