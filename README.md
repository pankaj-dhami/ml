# HuggingFace Road Scene Captioning FastAPI Service

This project exposes a local FastAPI API for image captioning using the Hugging Face model:

- **Model**: `Salesforce/blip-image-captioning-base`
- **API framework**: FastAPI
- **Server**: Uvicorn

## What this can and cannot do

This project now supports two modes:

- **Single image captioning** via `/caption`
- **Webcam road-scene summarization** via `/road-demo`

Important: `Salesforce/blip-image-captioning-base` is a **captioning model**, not a dedicated traffic detector. It can describe what it sees in each frame and produce a useful summary of the road scene, but it is **not the best model for precise vehicle counting or object-level traffic analytics**.

### Preferred model for true real-time traffic detection

If you want accurate live traffic/road-condition detection, the preferred model family is:

- **YOLOv8n / YOLOv8s** for fast real-time detection of vehicles, pedestrians, and obstacles
- Add **tracking** such as ByteTrack or DeepSORT for traffic flow analysis
- Add a **segmentation model** like SegFormer or DeepLabV3 for road-surface condition detection

In short:

- **Use BLIP** if you want a quick natural-language explanation of a scene
- **Use YOLOv8** if you want true real-time traffic detection

## Project Files

- `hf-api.py` — main API application
- `requirements.txt` — Python dependencies
- `README.md` — setup and run instructions

## Prerequisites

- Python 3.10+ (your machine is using Python 3.13)
- Internet connection (first run downloads model weights)
- Enough disk space for model cache

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python hf-api.py
```

The API starts on:

- `http://0.0.0.0:8000`
- Local test URL: `http://127.0.0.1:8000`

## Test Endpoints

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Image Captioning

Use `curl` with multipart form data:

```bash
curl -X POST "http://127.0.0.1:8000/caption" ^
  -F "file=@path\\to\\your-image.jpg"
```

Or open the interactive docs at:

```text
http://127.0.0.1:8000/docs
```

### Live Road Scene Demo

This endpoint captures a short burst of webcam frames, captions them, and returns a road-condition summary:

```bash
curl.exe "http://127.0.0.1:8000/road-demo?camera_index=0&frames=12"
```

If you are using PowerShell and want to avoid command parsing issues, you can also use:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/road-demo?camera_index=0&frames=12"
```

You can also check the health endpoint for the road-scene mode:

```bash
curl.exe http://127.0.0.1:8000/road-health
```

## Run Continuous Scene Feedback

If you want the camera to keep running and print road-scene feedback continuously in the terminal, use:

```bash
python hf-api.py --continuous
```

Optional arguments:

```bash
python hf-api.py --continuous --camera-index 0 --frames 12 --interval 2
```

What this does:

- Opens your webcam
- Captures frames in a loop
- Generates captions for the visible scene
- Prints a fresh driving/road summary every cycle

Stop it with `Ctrl+C`.

## Notes

- First startup may take a little time while the model is downloaded.
- This setup is designed to return a **natural-language caption** instead of class labels.
- If you run into image upload issues, make sure the file is a valid JPEG or PNG.
- For a real production traffic system, replace the captioning model with YOLOv8-based detection.

---

# Road AI Perception Stack (road-ai-perception.py)

This repository now also contains an experimental **single‑camera autonomous driving perception stack** implemented in:

```
road-ai-perception.py
```

The goal is to simulate a simplified version of the perception system used in autonomous vehicles.

It processes a webcam stream and produces:

• object detection  
• depth estimation  
• multi‑object tracking  
• lane visualization  
• trajectory prediction  
• a Tesla‑style bird‑eye world model

---

# Current Perception Pipeline

The real‑time pipeline works as follows:

Camera  
→ YOLOv8 object detection  
→ MiDaS monocular depth estimation  
→ ByteTrack multi‑object tracking  
→ lane detection overlay  
→ trajectory prediction  
→ Bird‑Eye View (BEV) world map

Two visualization windows are produced:

Camera Perception  
Bird‑Eye 2D Map

---

# Implemented Components

## 1. YOLO Object Detection

The system detects road participants using YOLOv8.

Supported classes include:

- cars
- trucks
- buses
- motorcycles
- bicycles
- pedestrians
- traffic lights
- stop signs

Each detection includes:

• bounding box  
• confidence score  
• estimated distance  
• lateral road position

---

## 2. Monocular Depth Estimation (MiDaS)

Depth is estimated from a single camera using the **MiDaS Small model**.

Pipeline:

frame → MiDaS → depth map → normalized depth

Object distance is estimated by sampling the depth values inside each bounding box.

If MiDaS cannot load, the system falls back to a **lightweight heuristic depth estimator**.

---

## 3. Multi‑Object Tracking (ByteTrack)

Tracking is implemented using **ByteTrack via the supervision library**.

Benefits:

• stable object IDs across frames  
• reduced ID switching  
• smoother object motion tracking

Each detected object now receives a persistent **track ID**.

---

## 4. Lane Detection

Lane visualization is currently implemented using classical computer vision:

- Canny edge detection
- Hough line transform

This overlays lane markers directly on the camera view.

---

## 5. Bird‑Eye View World Model

A Tesla‑style **2D perception map** is generated.

The BEV map shows:

• ego vehicle  
• road boundaries  
• detected vehicles  
• pedestrians  
• traffic control objects  
• distance grid

Objects are projected into map coordinates using:

distance estimation  
lateral position estimation.

---

## 6. Trajectory Prediction

The system predicts short‑term object motion.

Implementation:

Each tracked object stores recent BEV positions:

```
track_history[track_id] → [(x,y), (x,y), ...]
```

Velocity is estimated using the last two positions.

Future positions are extrapolated and rendered on the map as **green prediction points**.

---

# Current State of the Project

Completed modules:

✅ YOLO perception stack  
✅ MiDaS depth estimation  
✅ ByteTrack multi‑object tracking  
✅ Lane visualization  
✅ BEV world map  
✅ Trajectory prediction

The system now behaves like a **mini autonomous‑driving perception prototype**.

---

# Remaining Roadmap

The following upgrades are planned:

Improve depth estimation reliability  
Traffic light state classification (red / yellow / green)  
Lane geometry projection into BEV space  
Professional driving dashboard visualization

These upgrades would move the system closer to a **research‑grade autonomous perception stack**.

---

# Running the Perception Demo

Run the perception system directly:

```
python road-ai-perception.py --detector-model yolov8x.pt
```

Optional arguments:

```
--camera-index
--confidence
--frame-width
--depth-every
--device
```

Press **Q** or **ESC** to stop the demo.

---
