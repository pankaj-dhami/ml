import argparse
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency check
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CONFIDENCE = 0.35
DEFAULT_HISTORY = 12


HEAVY_TRAFFIC_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PEDESTRIAN_CLASSES = {"person"}
ROAD_CONTROL_CLASSES = {"traffic light", "stop sign"}


@dataclass
class DetectionSummary:
    label: str
    counts: Dict[str, int]
    text: str


def load_model(model_name: str):
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from _ULTRALYTICS_IMPORT_ERROR
    return YOLO(model_name)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Make sure webcam access is available.")
    return cap


def summarize_detections(class_names: List[str]) -> DetectionSummary:
    counts = Counter(class_names)
    car_like = sum(counts.get(name, 0) for name in HEAVY_TRAFFIC_CLASSES)
    pedestrians = counts.get("person", 0)
    controls = sum(counts.get(name, 0) for name in ROAD_CONTROL_CLASSES)

    if car_like >= 6:
        label = "heavy"
        text = f"Heavy traffic with {car_like} vehicle-like objects detected."
    elif car_like >= 3:
        label = "moderate"
        text = f"Moderate traffic with {car_like} vehicle-like objects detected."
    elif car_like > 0:
        label = "light"
        text = f"Light traffic with {car_like} vehicle-like objects detected."
    else:
        label = "clear"
        text = "Clear road ahead with few or no vehicles detected."

    extras = []
    if pedestrians > 0:
        extras.append(f"{pedestrians} pedestrian(s) present")
    if controls > 0:
        extras.append(f"{controls} road control object(s) visible")
    if extras:
        text += " " + "; ".join(extras) + "."

    return DetectionSummary(label=label, counts=dict(counts), text=text)


def draw_detections(frame, results) -> List[str]:
    names = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names.get(cls_id, str(cls_id))
            names.append(class_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if class_name == "person" else (255, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return names


def overlay_status(frame, summary: DetectionSummary, fps: float) -> None:
    overlay_lines = [
        f"Traffic: {summary.label}",
        f"FPS: {fps:.1f}",
        summary.text,
        "Press Q or ESC to quit",
    ]

    y = 24
    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24


def run_realtime_yolo(
    model_name: str = DEFAULT_MODEL,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    confidence: float = DEFAULT_CONFIDENCE,
    history_size: int = DEFAULT_HISTORY,
):
    model = load_model(model_name)
    cap = open_camera(camera_index)
    recent_labels = deque(maxlen=history_size)
    prev_time = time.time()

    try:
        print("Starting YOLO road-scene detector. Press Q or ESC to stop.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.", file=sys.stderr)
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            results = model.predict(frame, conf=confidence, verbose=False)
            names = draw_detections(frame, results)
            recent_labels.extend(names)

            summary = summarize_detections(list(recent_labels))

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            overlay_status(frame, summary, fps)
            cv2.imshow("Road AI YOLO", frame)

            print(f"\r{summary.text}                                ", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped YOLO road-scene detector.")


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight road-scene detector using YOLOv8n")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model name or path")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX, help="Webcam index")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY, help="Number of recent frames to use for summaries")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_yolo(
        model_name=args.model,
        camera_index=args.camera_index,
        confidence=args.confidence,
        history_size=args.history_size,
    )