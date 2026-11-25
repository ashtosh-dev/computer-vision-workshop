# Machine Learning & Computer Vision Workshop

A beginner-friendly, hands-on project covering image processing, CNN image classification, model evaluation, YOLO object detection and Raspberry Pi camera inference.

## Learning path

1. Explore images and datasets.
2. Process images with OpenCV.
3. Train a CNN on CIFAR-10.
4. Evaluate the trained classifier.
5. Run inference on a custom image.
6. Explore YOLO object detection.
7. Run live detection on a Raspberry Pi camera.

The CNN classifies an image into one class. YOLO detects and localizes one or more objects in an image. They are different tasks and are demonstrated separately.

## Project structure

```text
ml-cv-workshop/
├── notebooks/                 # Guided teaching notebooks
├── src/ml_cv_workshop/         # Reusable Python implementation
├── examples/
│   ├── opencv/                 # Image-processing example
│   └── object_detection/       # YOLO webcam / Pi camera demo
├── tests/                      # Automated tests
├── configs/
├── data/
├── outputs/
├── requirements.txt
├── requirements-pi.txt
└── pyproject.toml
```

## 1. Laptop setup

Use **Python 3.12** for the workshop. Python 3.10 and 3.11 are also supported, but all participants should use the same version where possible to reduce dependency and compatibility issues.

Create a virtual environment **with the selected Python version**. Activating a virtual environment does not change the Python version used to create it. Keep your system Python untouched; the workshop environment is isolated inside `.venv/`.

### Linux / Ubuntu / WSL

First, make sure Python 3.12 is installed. On Ubuntu or WSL:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

Then create and activate the environment from the repository root:

```bash
git clone https://github.com/ashtosh-dev/computer-vision-workshop.git
cd computer-vision-workshop

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

On Fedora, install the matching Python 3.12 packages using DNF (package availability can depend on your Fedora release):

```bash
sudo dnf install python3.12 python3.12-devel
```

Then create the environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`.

### macOS

After installing Python 3.12 (for example, with the Python installer or Homebrew):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

After installing Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, see the [official Python documentation on virtual environments](https://docs.python.org/3/library/venv.html) for alternatives. Do not change or uninstall the system Python just to set up this project.

For WSL, keep the repository in the Linux filesystem (for example, `~/projects/computer-vision-workshop`) rather than under `/mnt/c/` for better file performance.

### Verify the environment

Run these commands after activation:

```bash
python --version
python -c "import sys; print(sys.executable)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

Confirm that Python reports **3.12.x** and that the executable path points inside the repository's `.venv` directory. If it reports your system Python version, the environment may not be activated.

To leave the environment later, run `deactivate`. To use it again, return to the repository and activate it with the command for your operating system.

## 2. Run the notebooks

```bash
jupyter lab
```

Open the notebooks in `notebooks/` and run cells in order.

The CNN notebook downloads CIFAR-10 through Keras on first use. An internet connection is required for the initial download.

## 3. Train from the command line

```bash
python -m ml_cv_workshop.training.train --epochs 5 --batch-size 64
```

For a quick smoke run with a smaller subset:

```bash
python -m ml_cv_workshop.training.train --epochs 1 --batch-size 32 --sample-limit 500
```

The model and training curves are saved in `outputs/`. Training uses a validation split from the training data; the held-out CIFAR-10 test set is reserved for final evaluation.

## 4. Run the OpenCV example

```bash
python examples/opencv/image_processing.py path/to/image.jpg
```

The example opens windows for the original image, grayscale image, blurred image and Canny edges. In WSL, GUI windows require a working WSLg / display setup. You can also run the notebook version.

### Evaluate on the held-out test set

After training, run:

```bash
python -m ml_cv_workshop.evaluation.evaluate
```

This evaluates the saved model on the separate CIFAR-10 test set.

## 5. Run automated tests

```bash
pytest -q
```

The tests use small synthetic inputs. They do not train on the full CIFAR-10 dataset.

## 6. Raspberry Pi + camera object detection

This is a separate deployment example. It uses a pretrained YOLO11n model; it does not train YOLO during the workshop.

### Hardware

- Raspberry Pi 4 or 5 recommended.
- Raspberry Pi Camera Module connected and enabled, or a USB webcam.
- Raspberry Pi OS 64-bit.
- Adequate power supply and cooling for sustained workloads.

### Install camera support

On Raspberry Pi OS, install Picamera2 through apt:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv
```

Create a virtual environment that can access the system Picamera2 package:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Check that Picamera2 is available:

```bash
python -c "from picamera2 import Picamera2; print('Picamera2 available')"
```

### Install the YOLO runtime

Ultralytics depends on PyTorch. PyTorch wheel availability and installation steps vary by Raspberry Pi OS, Python version and hardware. Follow the current Ultralytics Raspberry Pi guide and PyTorch installation instructions for your exact setup before installing the runtime.

The optional dependency is listed in `requirements-pi.txt`:

```bash
pip install -r requirements-pi.txt
```

If this attempts to replace or break a working Raspberry Pi camera/OpenCV environment, stop and use the current vendor installation instructions. Do not blindly force-install packages.

### Test the camera

Use the Raspberry Pi camera preview / `rpicam-hello` utility if available:

```bash
rpicam-hello --timeout 5000
```

### Run person detection with the Pi Camera Module

From the repository root:

```bash
python examples/object_detection/detect_objects.py --camera picamera2 --people-only
```

### Run general object detection with the Pi Camera Module

```bash
python examples/object_detection/detect_objects.py --camera picamera2
```

### Run with a USB webcam

```bash
python examples/object_detection/detect_objects.py --camera webcam --camera-index 0
```

Press `q` in the display window to stop.

The first run may download `yolo11n.pt`, so internet access is needed. Download the weights before the live session and test the camera/model combination in advance.

### Performance notes

- Raspberry Pi inference is CPU-based in this example.
- Start with a 416-pixel inference size; reduce it if performance is poor.
- FPS depends on the Pi model, cooling, camera settings, model runtime and scene.
- Do not promise a particular FPS before measuring it on the actual device.
- The demo draws detections and an approximate FPS counter.
- Person-only mode filters detections to the COCO `person` class.

### Camera troubleshooting

- Confirm the ribbon cable is correctly seated and oriented.
- Check camera detection using `rpicam-hello`.
- Confirm Picamera2 works in the same Python environment used to run the script.
- If using a USB webcam, check `ls /dev/video*` and try another camera index.
- If the display is unavailable, run from a desktop session or add a headless output workflow before the workshop.

## 7. Google Colab

The notebooks are designed to be portable, but Colab is not the primary development environment. Upload or open the notebook in Colab, install the package dependencies in a setup cell if needed, and run the cells from top to bottom.

The Raspberry Pi camera example must run on the Pi itself; Colab cannot directly access a locally attached Pi camera without a separate streaming setup.

## Code quality

- Clear names and small functions.
- PEP 8 style and type hints where useful.
- Concise comments that explain non-obvious decisions.
- Docstrings for reusable public functions.
- Automated tests with pytest.
- No unnecessary abstractions.

## Important notes

- CIFAR-10 is used for the teaching classification exercise.
- The YOLO demo uses pretrained weights for object detection.
- Do not commit downloaded datasets, model weights, virtual environments or generated training outputs.
- Check dataset and model licenses before using them in a public or commercial application.
