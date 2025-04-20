# Object_detection_plates_lisence

# red light violation detection system.

---

# 🚦 Object Detection & License Plate Recognition for Red Light Violation

## A computer vision system for detecting red light violations and extracting license plates from surveillance video using YOLO, DeepSORT, and multiple OCR services.

## 🔍 System Pipeline Overview

## The system processes traffic video frames to detect violations and retrieve license plate data through the following logic:

### 1. 🎥 Input

- Video footage or real-time camera feed from an intersection.

---

### 2. 🛑 Crosswalk Detection & Virtual Stop Line

- Detected using **`ObjectDetector`**:
  - Extracts crosswalk bounding box from each frame.
  - Uses `create_stop_line_from_crosswalk()` to define a **virtual stop line**:
    - Placed **above the crosswalk** by a fixed offset (default `10px`).
    - Enforces a minimum line length if crosswalk is narrow.

---

### 3. 🚗 Vehicle Detection & Tracking

#### Detection:

- Handled by **`ObjectDetector`**:
  - Detects vehicles, traffic lights, crosswalks, and license plates using YOLO.
  - Filters and categorizes objects based on classes.

#### Tracking:

- Managed by **`TrackerObject`** using **DeepSORT**:
  - Assigns unique IDs to each vehicle and updates position across frames.

#### Visualization:

- Done by **`HandleTrackVehicles`**:
  - Draws bounding boxes and tracking IDs for vehicles.
  - Records vehicle ID, position, and confirmation status.

#### Color Encoding:

- Provided by **`Color_Pala`**:
  - Random color assigned to each vehicle ID.
  - Used for consistent visualization of tracked vehicles.

---

### 4. 🚦 Red Light Violation Detection

- Triggered **only when traffic light is red**:
  - For each tracked vehicle, use `is_vehicle_crossed_stop_line()`:
    - Checks if the bottom-right point of the vehicle bounding box **crosses the stop line**.
    - Must fall within the x-range of the virtual line.

---

### 5. 🔎 License Plate Detection & Association

- License plate detection handled by YOLO via `ObjectDetector`.

- Tracked separately using DeepSORT in **`HandleTracks`**:
  - Associates license plates to violating vehicles using:
    - `is_valid_plate_flexible()`:
      - Validates plate proximity to the vehicle bounding box.
      - Uses **IoU, side margin, and vertical offset** to confirm a match.

---

### 6. 🧠 OCR with Multi-AI Integration

- Cropped license plate images sent to multiple AI OCR services:

  - ✅ Plate Recognition API
  - ✅ Azure Computer Vision
  - ✅ Claude (Anthropic)
  - ✅ Gemini (Google)

- Outputs are **compared, combined, or filtered** to return the **most reliable result**.

---

### 7. 📤 Output

- For each violation:

  - Save **cropped plate image**.
  - Save **violation frame** with:
    - Stop line
    - Vehicle bounding box
    - Plate box
  - Save **timestamp, vehicle ID, and recognized license plate text**.

- Optional:
  - Send real-time alert
  - Export violation logs or analytics

---

## 🧱 Core Class Structure

| Class                 | Responsibility                                                             |
| --------------------- | -------------------------------------------------------------------------- |
| `ObjectDetector`      | Detects objects (vehicles, plates, lights, crosswalks) via YOLO.           |
| `TrackerObject`       | Tracks objects using DeepSORT and returns ID, position, and class info.    |
| `HandleTrackVehicles` | Draws vehicle boxes, maintains vehicle info, confirms tracked objects.     |
| `HandleTracks`        | Tracks license plates, associates them with vehicles, validates placement. |
| `Color_Pala`          | Assigns unique colors to vehicles for visual clarity during tracking.      |

---
