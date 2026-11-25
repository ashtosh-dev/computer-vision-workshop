
"""Evaluate a saved CNN on the held-out CIFAR-10 test set."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

from ml_cv_workshop.data.cifar10 import CLASS_NAMES, load_cifar10
from ml_cv_workshop.evaluation.metrics import evaluate_classifier


def print_classification_report(report: dict) -> None:
    """Display precision, recall and F1-score for each class."""

    print("\nClassification Report")
    print("-" * 76)
    print(
        f"{'Class':<18}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )
    print("-" * 76)

    for class_name in CLASS_NAMES:
        metrics = report[class_name]

        print(
            f"{class_name:<18}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1-score']:>12.4f}"
            f"{int(metrics['support']):>12}"
        )

    print("-" * 76)

    for average in ("macro avg", "weighted avg"):
        metrics = report[average]

        print(
            f"{average:<18}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1-score']:>12.4f}"
            f"{int(metrics['support']):>12}"
        )


def save_classification_report(report: dict, output_path: Path) -> None:
    """Save per-class evaluation metrics as a CSV file."""

    fieldnames = ["class", "precision", "recall", "f1-score", "support"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for class_name in (*CLASS_NAMES, "macro avg", "weighted avg"):
            writer.writerow(
                {
                    "class": class_name,
                    **report[class_name],
                }
            )


def save_confusion_matrix(matrix, output_path: Path) -> None:
    """Generate and save the confusion matrix visualization."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title("CIFAR-10 Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the model, evaluate it and save evaluation results."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved CIFAR-10 CNN."
    )

    parser.add_argument(
        "--model",
        default="outputs/cifar10_cnn.keras",
        help="Path to the trained Keras model.",
    )

    parser.add_argument(
        "--output-directory",
        default="outputs/evaluation",
        help="Directory for evaluation results.",
    )

    arguments = parser.parse_args()

    model_path = Path(arguments.model)
    output_directory = Path(arguments.output_directory)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train it first."
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10 test dataset...")
    _, _, test_images, test_labels = load_cifar10()

    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model...")
    results = evaluate_classifier(
        model,
        test_images,
        test_labels,
        CLASS_NAMES,
    )

    print("\nTest Results")
    print("=" * 40)
    print(f"Test loss:     {results['loss']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.2%}")

    print_classification_report(results["classification_report"])

    report_path = output_directory / "classification_report.csv"
    matrix_path = output_directory / "confusion_matrix.png"

    save_classification_report(
        results["classification_report"],
        report_path,
    )

    save_confusion_matrix(
        results["confusion_matrix"],
        matrix_path,
    )

    print("\nEvaluation artifacts saved:")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix:      {matrix_path}")


if __name__ == "__main__":
    main()
