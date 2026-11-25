"""Run YOLO object detection with a USB webcam or Raspberry Pi camera."""

import argparse
import time

import cv2
from ultralytics import YOLO


def open_camera(camera_type: str, camera_index: int):
    """Open a USB webcam or Picamera2 camera and return a frame reader."""
    if camera_type == "picamera2":
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is not installed. Install it with Raspberry Pi OS apt."
            ) from error

        camera = Picamera2()
        camera.configure(
            camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
        )
        camera.start()
        time.sleep(2)

        def read_frame():
            frame = camera.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return camera, read_frame

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return camera, camera.read


def parse_arguments() -> argparse.Namespace:
    """Read detection options from the command line."""
    parser = argparse.ArgumentParser(description="Run live YOLO object detection.")
    parser.add_argument("--camera", choices=["webcam", "picamera2"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--people-only", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    model = YOLO(arguments.model)
    camera, read_frame = open_camera(arguments.camera, arguments.camera_index)

    class_filter = [0] if arguments.people_only else None
    previous_time = time.perf_counter()

    try:
        while True:
            frame_available, frame = read_frame()
            if not frame_available:
                print("Could not read a frame from the camera.")
                break

            results = model.predict(
                frame,
                imgsz=arguments.image_size,
                conf=arguments.confidence,
                classes=class_filter,
                device="cpu",
                verbose=False,
            )

            annotated_frame = results[0].plot()
            current_time = time.perf_counter()
            frames_per_second = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                annotated_frame,
                f"FPS: {frames_per_second:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if not arguments.no_display:
                cv2.imshow("Live object detection - press q to quit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if arguments.camera == "picamera2":
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
