import numpy as np
from PIL import Image

from ml_cv_workshop.inference.predict import load_image


def test_load_image_returns_normalized_batch(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((40, 40, 3), 255, dtype=np.uint8)).save(image_path)

    image_batch = load_image(image_path)

    assert image_batch.shape == (1, 32, 32, 3)
    assert image_batch.dtype == np.float32
    assert np.all(image_batch == 1.0)
