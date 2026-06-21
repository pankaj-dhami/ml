import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import supervision as sv

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
PEDESTRIAN_CLASSES = {"person"}
CONTROL_CLASSES = {"traffic light", "stop sign"}

CLASS_COLORS = {
    "person": (0, 255, 255),
    "bicycle": (255, 220, 0),
    "car": (0, 180, 255),
    "motorcycle": (255, 120, 0),
    "bus": (0, 80, 255),
    "truck": (0, 80, 255),
    "traffic light": (0, 255, 0),
    "stop sign": (0, 0, 255),
}

MAP_WIDTH = 640
MAP_HEIGHT = 760
MAX_DISTANCE_M = 45.0
MAX_LATERAL_M = 12.0

# store recent BEV positions for trajectory prediction
track_history: Dict[int, List[Tuple[float, float]]] = {}


@dataclass
class PerceptionObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance_m: float
    lateral_m: float
    track_id: Optional[int] = None


@dataclass
class Track:
    track_id: int
    label: str
    lateral_m: float
    distance_m: float
    last_seen: int
    lost: int = 0


class LightweightTracker:
    """
    Small SORT-like centroid tracker.

    This avoids another dependency while still giving stable object IDs for the
    bird-eye visualization. It matches detections by class and nearest BEV point.
    """

    def __init__(self, max_lost: int = 10, max_match_distance: float = 7.0):
        self.max_lost = max_lost
        self.max_match_distance = max_match_distance
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_index = 0

    def update(self, objects: List[PerceptionObject]) -> List[PerceptionObject]:
        self.frame_index += 1
        unmatched_track_ids = set(self.tracks.keys())

        for obj in objects:
            best_track_id = None
            best_distance = float("inf")

            for track_id in list(unmatched_track_ids):
                track = self.tracks[track_id]
                if track.label != obj.label:
                    continue

                distance = math.hypot(
                    track.lateral_m - obj.lateral_m,
                    track.distance_m - obj.distance_m,
                )

                if distance < best_distance and distance <= self.max_match_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_id
                self.next_id += 1

            self.tracks[best_track_id] = Track(
                track_id=best_track_id,
                label=obj.label,
                lateral_m=obj.lateral_m,
                distance_m=obj.distance_m,
                last_seen=self.frame_index,
                lost=0,
            )
            obj.track_id = best_track_id
            unmatched_track_ids.discard(best_track_id)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.lost += 1
            if track.lost > self.max_lost:
                del self.tracks[track_id]

        return objects


class MidasDepthEstimator:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self.fallback_mode = False

        try:
            print("Loading MiDaS Small depth model...")
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
        except Exception as exc:
            self.fallback_mode = True
            print(
                f"MiDaS could not be initialized ({exc}). Falling back to a lightweight heuristic depth map.",
                file=sys.stderr,
            )

    def estimate(self, frame_bgr) -> np.ndarray:
        if self.fallback_mode or self.model is None or self.transform is None:
            # Lightweight heuristic depth proxy:
            # - lower image area is treated as closer
            # - brighter regions are slightly closer
            h, w = frame_bgr.shape[:2]
            y_ramp = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
            y_ramp = np.repeat(y_ramp, w, axis=1)

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            brightness = 1.0 - gray

            depth = 0.75 * y_ramp + 0.25 * brightness
            return depth.astype(np.float32)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.detach().cpu().numpy().astype(np.float32)
        return depth


def load_detector(model_path: str):
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from _ULTRALYTICS_IMPORT_ERROR

    print(f"Loading detector: {model_path}")
    return YOLO(model_path)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check webcam permissions and camera index.")
    return cap


def resize_width(frame, width: int):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = int(h * width / w)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    valid = depth_map[np.isfinite(depth_map)]
    if valid.size == 0:
        return np.zeros_like(depth_map, dtype=np.float32)

    low = np.percentile(valid, 5)
    high = np.percentile(valid, 95)
    denom = max(high - low, 1e-6)

    normalized = (depth_map - low) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def sample_object_depth(depth_norm: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    h, w = depth_norm.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.5

    roi = depth_norm[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.5

    # MiDaS is relative inverse depth in many scenes: higher often means closer.
    return float(np.median(roi))


def depth_to_distance_m(relative_depth: float, min_distance: float = 3.0, max_distance: float = MAX_DISTANCE_M) -> float:
    closeness = np.clip(relative_depth, 0.0, 1.0)
    return float(max_distance - closeness * (max_distance - min_distance))


def image_to_lateral_m(center_x: float, image_width: int, distance_m: float) -> float:
    normalized_x = (center_x / max(image_width, 1) - 0.5) * 2.0

    # Wider spread for far objects and narrower spread near the ego vehicle.
    field_width_m = 5.0 + (distance_m / MAX_DISTANCE_M) * 16.0
    lateral_m = normalized_x * field_width_m / 2.0
    return float(np.clip(lateral_m, -MAX_LATERAL_M, MAX_LATERAL_M))


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )

    lane_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # avoid division by zero
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # filter out near-horizontal lines (noise)
            if abs(slope) < 0.4:
                continue

            # keep only strong left/right lane candidates
            cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return cv2.addWeighted(frame, 1, lane_img, 0.7, 0)

def detect_objects(
    detector,
    frame_bgr,
    depth_norm: np.ndarray,
    confidence: float,
) -> List[PerceptionObject]:
    results = detector.predict(frame_bgr, conf=confidence, verbose=False)
    objects: List[PerceptionObject] = []
    h, w = frame_bgr.shape[:2]

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            label = result.names.get(class_id, str(class_id))
            if label not in RELEVANT_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            relative_depth = sample_object_depth(depth_norm, (x1, y1, x2, y2))
            distance_m = depth_to_distance_m(relative_depth)
            center_x = (x1 + x2) / 2.0
            lateral_m = image_to_lateral_m(center_x, w, distance_m)

            objects.append(
                PerceptionObject(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    distance_m=distance_m,
                    lateral_m=lateral_m,
                )
            )

    return objects


def draw_camera_view(frame_bgr, objects: List[PerceptionObject], fps: float) -> None:
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        color = CLASS_COLORS.get(obj.label, (255, 255, 255))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        track = f"#{obj.track_id}" if obj.track_id is not None else "#?"
        label = f"{obj.label} {track} {obj.distance_m:.1f}m"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame_bgr,
        f"Perception FPS: {fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def world_to_map(lateral_m: float, distance_m: float) -> Tuple[int, int]:
    x_norm = (lateral_m + MAX_LATERAL_M) / (2.0 * MAX_LATERAL_M)
    y_norm = np.clip(distance_m / MAX_DISTANCE_M, 0.0, 1.0)

    x = int(x_norm * MAP_WIDTH)
    # Ego car is near bottom. Far distance is near top.
    y = int(MAP_HEIGHT - 90 - y_norm * (MAP_HEIGHT - 150))
    return x, y


def draw_road(canvas) -> None:
    canvas[:] = (18, 18, 18)

    road_left_bottom = int(MAP_WIDTH * 0.18)
    road_right_bottom = int(MAP_WIDTH * 0.82)
    road_left_top = int(MAP_WIDTH * 0.38)
    road_right_top = int(MAP_WIDTH * 0.62)

    road_poly = np.array(
        [
            [road_left_bottom, MAP_HEIGHT],
            [road_right_bottom, MAP_HEIGHT],
            [road_right_top, 70],
            [road_left_top, 70],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(canvas, [road_poly], (45, 45, 45))
    cv2.polylines(canvas, [road_poly], True, (95, 95, 95), 2)

    # Lane lines.
    for offset in (-0.18, 0.0, 0.18):
        bottom_x = int(MAP_WIDTH * (0.5 + offset))
        top_x = int(MAP_WIDTH * (0.5 + offset * 0.25))
        cv2.line(canvas, (bottom_x, MAP_HEIGHT), (top_x, 70), (80, 80, 80), 1)

    # Distance grid.
    for distance in range(10, int(MAX_DISTANCE_M) + 1, 10):
        _, y = world_to_map(0, distance)
        cv2.line(canvas, (80, y), (MAP_WIDTH - 80, y), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            f"{distance}m",
            (18, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )


def draw_ego_vehicle(canvas) -> None:
    ego_x = MAP_WIDTH // 2
    ego_y = MAP_HEIGHT - 55

    body = np.array(
        [
            [ego_x, ego_y - 36],
            [ego_x - 24, ego_y + 26],
            [ego_x + 24, ego_y + 26],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(canvas, [body], (40, 160, 255))
    cv2.polylines(canvas, [body], True, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "YOUR CAR",
        (ego_x - 42, ego_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_predicted_trajectory(canvas, obj: PerceptionObject) -> None:
    if obj.track_id is None:
        return
    history = track_history.get(obj.track_id, [])
    if len(history) < 2:
        return

    (x1, y1), (x2, y2) = history[-2], history[-1]
    vx = x2 - x1
    vy = y2 - y1

    # simple future prediction
    for step in range(1, 6):
        fx = x2 + vx * step
        fy = y2 + vy * step
        cv2.circle(canvas, (int(fx), int(fy)), 4, (120, 255, 120), -1)

def draw_map_object(canvas, obj: PerceptionObject) -> None:
    x, y = world_to_map(obj.lateral_m, obj.distance_m)

    # update trajectory history
    if obj.track_id is not None:
        history = track_history.setdefault(obj.track_id, [])
        history.append((x, y))
        if len(history) > 20:
            history.pop(0)
    color = CLASS_COLORS.get(obj.label, (220, 220, 220))

    if obj.label in VEHICLE_CLASSES:
        width, height = 34, 50
        cv2.rectangle(
            canvas,
            (x - width // 2, y - height // 2),
            (x + width // 2, y + height // 2),
            color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (x - width // 2, y - height // 2),
            (x + width // 2, y + height // 2),
            (255, 255, 255),
            2,
        )
    elif obj.label in PEDESTRIAN_CLASSES:
        cv2.circle(canvas, (x, y), 15, color, -1)
        cv2.circle(canvas, (x, y), 15, (255, 255, 255), 2)
    elif obj.label == "traffic light":
        cv2.circle(canvas, (x, y), 13, (0, 255, 0), -1)
        cv2.circle(canvas, (x, y), 13, (255, 255, 255), 2)
    elif obj.label == "stop sign":
        cv2.rectangle(canvas, (x - 14, y - 14), (x + 14, y + 14), (0, 0, 255), -1)
        cv2.rectangle(canvas, (x - 14, y - 14), (x + 14, y + 14), (255, 255, 255), 2)
    else:
        cv2.circle(canvas, (x, y), 10, color, -1)

    draw_predicted_trajectory(canvas, obj)

    track = f"#{obj.track_id}" if obj.track_id is not None else ""
    label = f"{obj.label} {track}"
    cv2.putText(
        canvas,
        label,
        (x + 12, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def build_scene_summary(objects: List[PerceptionObject]) -> str:
    vehicles = sum(1 for obj in objects if obj.label in VEHICLE_CLASSES)
    people = sum(1 for obj in objects if obj.label in PEDESTRIAN_CLASSES)
    controls = sum(1 for obj in objects if obj.label in CONTROL_CLASSES)

    if vehicles >= 6:
        traffic = "heavy traffic"
    elif vehicles >= 3:
        traffic = "moderate traffic"
    elif vehicles > 0:
        traffic = "light traffic"
    else:
        traffic = "clear road"

    extras = []
    if people:
        extras.append(f"{people} pedestrian(s)")
    if controls:
        extras.append(f"{controls} traffic control object(s)")

    if extras:
        return f"{traffic}; " + "; ".join(extras)
    return traffic


def render_bev_map(objects: List[PerceptionObject], fps: float) -> np.ndarray:
    canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    draw_road(canvas)

    # Draw far objects first, near objects last.
    for obj in sorted(objects, key=lambda item: item.distance_m, reverse=True):
        draw_map_object(canvas, obj)

    draw_ego_vehicle(canvas)

    summary = build_scene_summary(objects)
    cv2.putText(
        canvas,
        "Tesla-style 2D Perception Map",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{summary} | FPS {fps:.1f}",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return canvas


def colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)


def run_perception(
    camera_index: int,
    detector_model: str,
    confidence: float,
    frame_width: int,
    depth_every: int,
    device: str,
):
    detector = load_detector(detector_model)
    depth_estimator = MidasDepthEstimator(device=device)
    tracker = sv.ByteTrack()

    cap = open_camera(camera_index)
    last_depth_norm = None
    frame_index = 0
    previous_time = time.time()

    try:
        print("Starting road perception stack.")
        print("Windows: Camera Perception + Bird-Eye 2D Map")
        print("Press Q or ESC in either OpenCV window to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.", file=sys.stderr)
                time.sleep(0.05)
                continue

            frame = resize_width(frame, frame_width)
            frame_index += 1

            if last_depth_norm is None or frame_index % max(depth_every, 1) == 0:
                depth_raw = depth_estimator.estimate(frame)
                last_depth_norm = normalize_depth(depth_raw)

            objects = detect_objects(
                detector=detector,
                frame_bgr=frame,
                depth_norm=last_depth_norm,
                confidence=confidence,
            )
            if len(objects) > 0:
                boxes = np.array([obj.bbox for obj in objects], dtype=float)
                confs = np.array([obj.confidence for obj in objects], dtype=float)
                class_ids = np.arange(len(objects))

                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=confs,
                    class_id=class_ids
                )

                tracked = tracker.update_with_detections(detections)

                for obj, tid in zip(objects, tracked.tracker_id):
                    obj.track_id = int(tid) if tid is not None else None

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            camera_view = detect_lanes(frame.copy())
            draw_camera_view(camera_view, objects, fps)
            bev_map = render_bev_map(objects, fps)

            cv2.imshow("Camera Perception", camera_view)
            cv2.imshow("Bird-Eye 2D Map", bev_map)

            print(f"\r{build_scene_summary(objects):80}", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped road perception stack.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-camera road perception stack: YOLO + MiDaS + tracking + BEV map"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--detector-model", default="yolov8x.pt", help="YOLO model path/name (use larger model for better distant detection)")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--frame-width", type=int, default=640, help="Resize camera frame to this width")
    parser.add_argument(
        "--depth-every",
        type=int,
        default=3,
        help="Run MiDaS depth every N frames. Higher is faster; lower is smoother.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_perception(
        camera_index=args.camera_index,
        detector_model=args.detector_model,
        confidence=args.confidence,
        frame_width=args.frame_width,
        depth_every=args.depth_every,
        device=args.device,
    )