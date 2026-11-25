import numpy as np

from ml_cv_workshop.models.cnn import build_cnn_model


def test_cnn_returns_probabilities_for_ten_classes():
    model = build_cnn_model()
    sample_images = np.random.default_rng(42).random((2, 32, 32, 3), dtype=np.float32)

    predictions = model.predict(sample_images, verbose=0)

    assert predictions.shape == (2, 10)
    np.testing.assert_allclose(predictions.sum(axis=1), [1.0, 1.0], atol=1e-5)
