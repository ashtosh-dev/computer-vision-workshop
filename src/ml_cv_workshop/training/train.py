
"""Training entry point for the CIFAR-10 CNN."""

import argparse
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from ml_cv_workshop.data.cifar10 import load_cifar10
from ml_cv_workshop.models.cnn import build_cnn_model
from ml_cv_workshop.utils.visualization import save_training_curves


def train_model(
    epochs: int = 20,
    batch_size: int = 64,
    output_directory: str | Path = "outputs",
    sample_limit: int | None = None,
) -> tf.keras.callbacks.History:
    """Train the CNN and save its model and learning curves."""

    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if sample_limit is not None and sample_limit < 20:
        raise ValueError("Sample limit must be at least 20.")

    tf.keras.utils.set_random_seed(42)

    print("Loading CIFAR-10 dataset...")
    train_images, train_labels, _, _ = load_cifar10()

    if sample_limit is not None:
        train_images = train_images[:sample_limit]
        train_labels = train_labels[:sample_limit]

    print(f"Training samples available: {len(train_images)}")

    training_images, validation_images, training_labels, validation_labels = (
        train_test_split(
            train_images,
            train_labels,
            test_size=0.1,
            random_state=42,
            stratify=train_labels,
        )
    )

    print(f"Training samples: {len(training_images)}")
    print(f"Validation samples: {len(validation_images)}")

    model = build_cnn_model()

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting CNN training...")
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    history = model.fit(
        training_images,
        training_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = output_path / "cifar10_cnn.keras"
    curves_path = output_path / "training_curves.png"

    model.save(model_path)
    save_training_curves(history, curves_path)

    print("\nTraining completed.")
    print(f"Model saved to: {model_path}")
    print(f"Training curves saved to: {curves_path}")

    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=lambda index: history.history["val_loss"][index],
    )

    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best validation loss: {history.history['val_loss'][best_epoch]:.4f}")
    print(
        "Validation accuracy at best epoch: "
        f"{history.history['val_accuracy'][best_epoch]:.4f}"
    )

    return history


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description="Train a CNN on the CIFAR-10 dataset."
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-directory", default="outputs")
    parser.add_argument("--sample-limit", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    train_model(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        output_directory=arguments.output_directory,
        sample_limit=arguments.sample_limit,
    )
