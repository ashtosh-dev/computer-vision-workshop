import numpy as np
import tensorflow as tf

from ml_cv_workshop.models.cnn import build_cnn_model


def test_model_can_complete_a_small_training_step():
    tf.keras.utils.set_random_seed(42)
    model = build_cnn_model()
    sample_images = np.random.default_rng(42).random((4, 32, 32, 3), dtype=np.float32)
    sample_labels = np.array([0, 1, 2, 3])

    history = model.fit(sample_images, sample_labels, epochs=1, verbose=0)

    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][0])
