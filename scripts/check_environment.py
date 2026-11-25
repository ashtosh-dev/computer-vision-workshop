"""Check whether the main workshop dependencies are importable."""

import cv2
import numpy as np
import tensorflow as tf

print(f"Python packages imported successfully.")
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")
