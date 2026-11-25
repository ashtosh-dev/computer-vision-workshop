import numpy as np

from ml_cv_workshop.data.cifar10 import normalize_images


def test_normalize_images_returns_float_values_between_zero_and_one():
    images = np.array([[[[0, 127, 255]]]], dtype=np.uint8)

    normalized_images = normalize_images(images)

    assert normalized_images.dtype == np.float32
    assert normalized_images.min() >= 0.0
    assert normalized_images.max() <= 1.0
    np.testing.assert_allclose(normalized_images[0, 0, 0], [0.0, 127 / 255, 1.0])
