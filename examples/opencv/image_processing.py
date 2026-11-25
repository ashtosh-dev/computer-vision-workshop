
"""OpenCV image-processing fundamentals."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def process_image(image_path: str | Path) -> tuple:
    """Load an image and apply basic image-processing operations."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edge_image = cv2.Canny(blurred_image, 50, 150)

    return image, grayscale_image, blurred_image, edge_image


def create_comparison(image, grayscale, blurred, edges):
    """Arrange processing results in a single comparison image."""

    height, width = image.shape[:2]

    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    images = [
        ("Original", image),
        ("Grayscale", grayscale),
        ("Gaussian Blur", blurred),
        ("Canny Edges", edges),
    ]

    panels = []

    for title, processed_image in images:
        resized = cv2.resize(processed_image, (width, height))
        cv2.putText(
            resized,
            title,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        panels.append(resized)

    top_row = np.hstack((panels[0], panels[1]))
    bottom_row = np.hstack((panels[2], panels[3]))

    return np.vstack((top_row, bottom_row))


def main() -> None:
    """Run image processing and display the comparison."""

    parser = argparse.ArgumentParser(
        description="Run basic OpenCV image processing."
    )
    parser.add_argument("image_path", help="Path to an image file")
    parser.add_argument(
        "--output",
        default="outputs/opencv_comparison.png",
        help="Path to save the comparison image",
    )

    arguments = parser.parse_args()

    image, grayscale, blurred, edges = process_image(arguments.image_path)

    comparison = create_comparison(image, grayscale, blurred, edges)

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Could not save comparison image: {output_path}")

    print(f"Comparison saved to: {output_path}")

    cv2.imshow("OpenCV Image Processing", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
