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