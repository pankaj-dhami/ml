from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
import cv2
import torch
from io import BytesIO
import uvicorn
from collections import deque, Counter
from typing import List, Optional
import argparse
import time
from datetime import datetime
import sys
import threading
import queue

app = FastAPI(title="HuggingFace Image Captioning API")

# A lightweight CPU-friendly image captioning model.
model_name = "Salesforce/blip-image-captioning-base"

captioner = pipeline(
    task="image-to-text",
    model=model_name,
    device=0 if torch.cuda.is_available() else -1,
)

MAX_FRAME_WIDTH = 384


def summarize_road_scene(captions: List[str]) -> str:
    """
    Create a simple road-condition summary from recent frame captions.
    This is not object detection; it is a scene-level explanation layer.
    """
    if not captions:
        return "No frames were analyzed."

    joined = " ".join(captions).lower()

    traffic_keywords = {
        "heavy": ["traffic jam", "crowded", "busy street", "many cars", "congested"],
        "moderate": ["cars", "vehicle", "street", "road"],
        "clear": ["empty road", "open road", "few cars", "clear road", "road ahead"],
    }

    condition_keywords = {
        "pedestrian_present": ["person", "pedestrian", "people walking"],
        "intersection": ["intersection", "junction", "crossing"],
        "night": ["night", "dark", "street lights"],
        "rainy_or_wet": ["rain", "wet", "water", "puddle"],
    }

    traffic_state = "moderate"
    if any(keyword in joined for keyword in traffic_keywords["heavy"]):
        traffic_state = "heavy"
    elif any(keyword in joined for keyword in traffic_keywords["clear"]):
        traffic_state = "clear"

    flags = [name.replace("_", " ") for name, keywords in condition_keywords.items() if any(keyword in joined for keyword in keywords)]

    most_common = Counter(captions).most_common(1)[0][0]

    explanation_parts = [f"Traffic appears {traffic_state}.", f"Most common scene description: {most_common}."]
    if flags:
        explanation_parts.append("Detected scene cues: " + ", ".join(flags) + ".")

    return " ".join(explanation_parts)


def resize_frame(frame, max_width: int = MAX_FRAME_WIDTH):
    """Resize a frame to a smaller width while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def caption_frame(frame) -> str:
    """Caption a single RGB frame with the shared image-to-text pipeline."""
    resized = resize_frame(frame)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    result = captioner(image)
    return result[0]["generated_text"] if result else ""


def capture_road_scene(camera_index: int = 0, warmup_frames: int = 8) -> dict:
    """Capture and caption a single webcam frame."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Make sure the webcam is available.")

    try:
        for _ in range(warmup_frames):
            cap.read()

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")

        caption = caption_frame(frame)
    finally:
        cap.release()

    captions = [caption] if caption else []
    return {
        "captions": captions,
        "summary": summarize_road_scene(captions),
        "frames_analyzed": len(captions),
    }


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera. Make sure the webcam is available.")
    return cap


def warmup_camera(cap: cv2.VideoCapture, warmup_frames: int = 8) -> None:
    for _ in range(warmup_frames):
        cap.read()


def _drop_old_queue_item(frame_queue: queue.Queue) -> None:
    try:
        while True:
            frame_queue.get_nowait()
    except queue.Empty:
        return


def _frame_capture_worker(
    cap: cv2.VideoCapture,
    frame_queue: queue.Queue,
    latest_frame_holder,
    latest_frame_lock: threading.Lock,
    stop_event: threading.Event,
    capture_interval: float,
) -> None:
    """Continuously capture frames and keep only the newest one in the queue."""
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(capture_interval)
            continue

        try:
            if frame_queue.full():
                _drop_old_queue_item(frame_queue)
            frame_queue.put_nowait(frame)
            with latest_frame_lock:
                latest_frame_holder["frame"] = frame
        except queue.Full:
            pass

        time.sleep(capture_interval)


def _caption_worker(
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Caption the latest available frame without blocking camera capture."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            caption = caption_frame(frame)
            if caption:
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                result_queue.put_nowait(caption)
        except Exception as exc:
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put_nowait(f"__error__:{exc}")


def run_continuous_road_feedback(
    camera_index: int = 0,
    warmup_frames: int = 8,
    interval: float = 0.1,
):
    """Continuously capture webcam frames and print road-scene feedback until interrupted."""
    print("Starting continuous road-scene feedback. Press Ctrl+C to stop.")

    cap: Optional[cv2.VideoCapture] = None
    stop_event = threading.Event()
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    recent_captions = deque(maxlen=12)
    capture_thread: Optional[threading.Thread] = None
    caption_thread: Optional[threading.Thread] = None
    latest_frame_lock = threading.Lock()
    latest_frame_holder = {"frame": None}

    try:
        cap = open_camera(camera_index)
        warmup_camera(cap, warmup_frames=warmup_frames)

        capture_thread = threading.Thread(
            target=_frame_capture_worker,
            args=(cap, frame_queue, latest_frame_holder, latest_frame_lock, stop_event, interval),
            daemon=True,
        )
        caption_thread = threading.Thread(
            target=_caption_worker,
            args=(frame_queue, result_queue, stop_event),
            daemon=True,
        )

        capture_thread.start()
        caption_thread.start()

        last_printed_caption = None

        cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)

        while not stop_event.is_set():
            with latest_frame_lock:
                frame_to_show = latest_frame_holder["frame"]

            if frame_to_show is not None:
                display_frame = frame_to_show.copy()
                cv2.putText(
                    display_frame,
                    "Press Q or ESC to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Live Camera", display_frame)

            try:
                message = result_queue.get(timeout=0.2)
            except queue.Empty:
                message = None

            if message is not None:
                if message.startswith("__error__:"):
                    print(message.replace("__error__:", "Captioning error: "), file=sys.stderr)
                else:
                    recent_captions.append(message)
                    captions = list(recent_captions)
                    summary = summarize_road_scene(captions)

                    if message != last_printed_caption:
                        last_printed_caption = message
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n[{timestamp}] {summary}")
                        print(f"Frames analyzed: {len(captions)}")
                        print(f"Recent captions: {captions[-3:]}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\nStopped continuous feedback.")
                break

    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped continuous feedback.")
    finally:
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
        if caption_thread is not None:
            caption_thread.join(timeout=1.0)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    result = captioner(image)
    caption = result[0]["generated_text"] if result else ""

    return {
        "filename": file.filename,
        "model": model_name,
        "caption": caption,
    }


@app.get("/road-health")
def road_health():
    return {
        "status": "ok",
        "mode": "road-scene-captioning",
        "model": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "note": "This endpoint is for the live webcam demo added in the companion script."
    }


@app.get("/road-demo")
def road_demo(camera_index: int = 0):
    try:
        result = capture_road_scene(camera_index=camera_index)
        return {
            "model": model_name,
            **result,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace road-scene captioning service")
    parser.add_argument("--continuous", action="store_true", help="Run continuous webcam feedback in the terminal")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to use")
    parser.add_argument("--warmup-frames", type=int, default=8, help="Number of frames to discard before analysis")
    parser.add_argument("--interval", type=float, default=0.25, help="Seconds to wait between feedback cycles")
    parser.add_argument("--host", default="0.0.0.0", help="Host for FastAPI server mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.continuous:
        run_continuous_road_feedback(
            camera_index=args.camera_index,
            warmup_frames=args.warmup_frames,
            interval=args.interval,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)