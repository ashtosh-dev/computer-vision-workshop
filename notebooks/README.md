# ML & Computer Vision Workshop — Notebook Set

Aligned with the 40-slide workshop. No OpenCV practical is included.

## Recommended order
1. `01_cv_and_dl_fundamentals.ipynb` — image tensors, dataset structure, normalization and parameter calculations.
2. `02_cnn_cifar10_training.ipynb` — CNN layers, tensor shapes, parameter counts, training and evaluation.
3. `03_model_inference.ipynb` — loading a trained model, predictions and error inspection.
4. `04_transfer_learning_mobilenetv2.ipynb` — pretrained MobileNetV2, feature extraction, freezing and fine-tuning.
5. `05_edge_ai_tensorflow_lite.ipynb` — TFLite conversion, verification, Netron and edge deployment considerations.

## Environment
Python 3.10/3.11 recommended. Install:
`python -m pip install tensorflow numpy matplotlib jupyterlab`

Run notebooks in order. The first CIFAR-10 and ImageNet-weight downloads require internet. Training time and accuracy vary by hardware. Keep a tested model/export ready for the live session.

## Notes
- Notebook 02 creates `cifar10_cnn.keras`.
- Notebook 03 and Notebook 05 use that saved model.
- Notebook 04 uses ImageNet pretrained MobileNetV2 weights and a small balanced CIFAR-10 subset for a manageable demonstration.
- Notebook 05 is deliberately last: Edge AI follows training, inference, and transfer learning.
