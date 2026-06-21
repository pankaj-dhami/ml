# Road AI Perception Stack

# to run
python road-ai-perception.py --detector-model yolov8n.pt


python road-ai-perception.py for 8x model


This document describes the perception system implemented in:

road-ai-perception.py

The goal of this module is to simulate a simplified **autonomous driving perception stack** using a single camera.

The system performs real‑time perception and generates a **bird‑eye‑view (BEV) world model** similar to the visualization used in modern autonomous driving systems.

--------------------------------------------------

PROJECT OVERVIEW

The system processes a webcam stream and performs multiple perception tasks:

• object detection  
• monocular depth estimation  
• multi‑object tracking  
• lane detection  
• trajectory prediction  
• bird‑eye‑view world visualization

Two windows are produced during runtime:

Camera Perception  
Bird‑Eye 2D Map

--------------------------------------------------

PERCEPTION PIPELINE

The full perception pipeline currently works as follows:

Camera Frame  
→ YOLOv8 Object Detection  
→ MiDaS Depth Estimation  
→ ByteTrack Object Tracking  
→ Lane Detection Overlay  
→ Trajectory Prediction  
→ Bird‑Eye View Map Rendering

--------------------------------------------------

IMPLEMENTED COMPONENTS

1. YOLOv8 Object Detection

YOLOv8 is used to detect road participants.

Detected classes include:

- car
- truck
- bus
- motorcycle
- bicycle
- pedestrian
- traffic light
- stop sign

Each detection provides:

• bounding box  
• confidence score  
• estimated distance  
• lateral position on the road

--------------------------------------------------

2. Monocular Depth Estimation (MiDaS)

Depth estimation is implemented using the **MiDaS Small model**.

Pipeline:

frame → MiDaS → depth map → normalized depth

Object distance is estimated by sampling the depth values inside each bounding box.

A fallback heuristic depth estimator is used if MiDaS fails to load.

--------------------------------------------------

3. Multi‑Object Tracking (ByteTrack)

Object tracking is implemented using **ByteTrack via the supervision library**.

Benefits:

• stable object IDs  
• smoother motion tracking  
• reduced identity switching

Each detected object is assigned a persistent **track_id** across frames.

--------------------------------------------------

4. Lane Detection

Lane visualization is implemented using classical computer vision techniques:

• Canny edge detection  
• Hough line transform

Detected lane lines are drawn on the camera view.

--------------------------------------------------

5. Bird‑Eye View World Model

A 2D bird‑eye perception map is rendered.

The map shows:

• ego vehicle  
• road boundaries  
• detected vehicles  
• pedestrians  
• traffic control objects  
• distance grid

Objects are projected into the map using estimated:

distance  
lateral offset

--------------------------------------------------

6. Trajectory Prediction

Short‑term motion prediction is implemented.

Each tracked object stores recent BEV positions:

track_history[track_id] → [(x,y), (x,y), ...]

Velocity is estimated from the last two positions.

Future positions are extrapolated and drawn on the BEV map as **green prediction points**.

--------------------------------------------------

CURRENT STATE OF THE PROJECT

Completed components:

✓ YOLO perception stack  
✓ MiDaS monocular depth estimation  
✓ ByteTrack object tracking  
✓ Lane detection overlay  
✓ Bird‑Eye View world model  
✓ Trajectory prediction visualization

The system now functions as a **mini autonomous driving perception prototype**.

--------------------------------------------------

PENDING FEATURES / ROADMAP

The following improvements are still planned:

1. Improve depth estimation reliability  
   - temporal smoothing  
   - object-level depth filtering

2. Traffic light state classification  
   - detect red / yellow / green states

3. Lane geometry projection into BEV  
   - convert lane detections into map coordinates

4. Professional visualization dashboard  
   - smoother trajectory lines  
   - velocity arrows  
   - richer driving visualization

These upgrades will move the system closer to a **research‑grade perception stack**.

--------------------------------------------------

RUNNING THE PERCEPTION SYSTEM

Run the demo:

python road-ai-perception.py --detector-model yolov8x.pt

Optional arguments:

--camera-index  
--confidence  
--frame-width  
--depth-every  
--device

Press Q or ESC to stop the program.

--------------------------------------------------