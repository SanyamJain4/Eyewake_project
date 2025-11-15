import uvicorn
import cv2
import numpy as np
import time
import os
import urllib.request
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import mediapipe as mp  # <--- FIX 1: ADDED IMPORT

# MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================
# 1. MODEL DOWNLOAD & SETUP
# ==========================
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading MediaPipe face_landmarker model to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise

# Create the Face Landmarker (load it ONCE)
try:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    # This landmarker will be shared by all connections
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("FaceLandmarker initialized successfully.")
except Exception as e:
    print(f"Failed to initialize MediaPipe FaceLandmarker: {e}")
    raise

# ==========================
# 2. UTILITY FUNCTIONS
# ==========================
def median(arr):
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 == 1 else float((s[mid - 1] + s[mid]) / 2.0)

def decode_base64_image(base64_string):
    """Decode a base64 string into an OpenCV image"""
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    
    try:
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

# ==========================
# 3. DROWSINESS PROCESSOR CLASS
# ==========================
class DrowsinessProcessor:
    # CONSTANTS
    CONSEC_FRAMES_RED = 25
    ALERT_INTERVAL = 5.0
    CALIBRATION_TIME = 5.0
    YAWN_PHYS_PIXELS = 20
    YAWN_MIN_DURATION = 4.0
    YAWN_COOLDOWN = 2.0
    EYE_HISTORY_LEN = 15
    DEFAULT_EYE_THRESH = 12.0

    # Face landmark indices
    L_EYE_T, L_EYE_B = 159, 145
    R_EYE_T, R_EYE_B = 386, 374
    MOUTH_TOP, MOUTH_BOTTOM = 13, 14

    def __init__(self):
        # State variables
        self.status = "INIT"
        self.yawns = 0
        self.alerts = 0
        self.baseline = self.DEFAULT_EYE_THRESH
        self.eye_thresh = self.DEFAULT_EYE_THRESH * 0.6
        self.calibrating = False

        self.eye_hist = []
        self.mouth_hist = []
        self.closed_frames = 0
        self.yawning = False
        self.yawn_start = 0.0
        self.last_yawn = 0.0
        self.last_alert = 0.0
        self.calib_start = None
        self.calib_samples = []
        
        print("New DrowsinessProcessor initialized.")

    def process(self, img: np.ndarray):
        """Process a single frame and update the state."""
        if img is None:
            print("Processor received None image.")
            return self.get_state()
            
        h, w = img.shape[:2]

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # <--- FIX 2: Changed 'vision.Image' to 'mp.Image'
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = face_landmarker.detect(mp_img)
        except Exception as e:
            print(f"FaceLandmarker error: {e}")
            self.status = "LANDMARKER ERROR"
            return self.get_state()

        eye_pixels = 0.0
        mouth_pixels = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            def xy(idx):
                p = lm[idx]
                return int(p.x * w), int(p.y * h)

            lt, lb = xy(self.L_EYE_T), xy(self.L_EYE_B)
            rt, rb = xy(self.R_EYE_T), xy(self.R_EYE_B)
            eye_pixels = (abs(lt[1]-lb[1]) + abs(rt[1]-rb[1])) / 2.0

            mt, mb = xy(self.MOUTH_TOP), xy(self.MOUTH_BOTTOM)
            mouth_pixels = abs(mb[1]-mt[1])
        
        # Smooth with history
        self.eye_hist.append(eye_pixels)
        if len(self.eye_hist) > self.EYE_HISTORY_LEN: self.eye_hist.pop(0)
        self.mouth_hist.append(mouth_pixels)
        if len(self.mouth_hist) > self.EYE_HISTORY_LEN: self.mouth_hist.pop(0)

        eye_avg = float(np.mean(self.eye_hist)) if self.eye_hist else 0.0
        mouth_avg = float(np.mean(self.mouth_hist)) if self.mouth_hist else 0.0

        # --- Calibration Logic ---
        if self.calibrating:
            if self.calib_start is None:
                self.calib_start = time.time()
                self.calib_samples = []
            
            elapsed = time.time() - self.calib_start
            if eye_pixels > 0: # Only add valid samples
                self.calib_samples.append(eye_pixels)

            if elapsed >= self.CALIBRATION_TIME:
                base = median(self.calib_samples) if self.calib_samples else self.DEFAULT_EYE_THRESH
                self.baseline = float(base)
                self.eye_thresh = float(base * 0.6)
                self.calibrating = False
                self.calib_start = None
                self.calib_samples = []

        # --- Yawn Detection ---
        if mouth_avg > self.YAWN_PHYS_PIXELS:
            if not self.yawning:
                self.yawning = True
                self.yawn_start = time.time()
            else:
                if time.time() - self.yawn_start >= self.YAWN_MIN_DURATION:
                    if time.time() - self.last_yawn >= self.YAWN_COOLDOWN:
                        self.yawns += 1
                        self.last_yawn = time.time()
                    self.yawning = False
        else:
            self.yawning = False

        # --- Eye Closure / Drowsiness ---
        if eye_avg < self.eye_thresh and eye_avg > 0: # eye_avg > 0 filters out bad frames
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        now = time.time()
        if self.closed_frames >= self.CONSEC_FRAMES_RED:
            if now - self.last_alert >= self.ALERT_INTERVAL:
                self.alerts += 1
                self.status = "DROWSY"
                self.last_alert = now
        else:
            if self.calibrating:
                self.status = "CALIBRATING"
            elif mouth_avg > self.YAWN_PHYS_PIXELS:
                self.status = "YAWNING"
            elif eye_avg <= 0 and self.status != "INIT": # Handle case where face is lost
                self.status = "NO FACE"
            elif not self.calibrating:
                self.status = "ACTIVE"
        
        # Return the current state as a dictionary
        return {
            "status": self.status,
            "yawns": self.yawns,
            "alerts": self.alerts,
            "eye_avg": round(eye_avg, 2),
            "mouth_avg": round(mouth_avg, 2),
            "baseline": round(self.baseline, 2),
            "eye_thresh": round(self.eye_thresh, 2)
        }

    def start_calibration(self):
        self.calibrating = True
        self.calib_start = None
        self.calib_samples = []

    def reset_counts(self):
        self.yawns = 0
        self.alerts = 0

    def get_state(self):
        """Returns the current state without processing."""
        return {
            "status": self.status,
            "yawns": self.yawns,
            "alerts": self.alerts,
            "eye_avg": 0.0,
            "mouth_avg": 0.0,
            "baseline": round(self.baseline, 2),
            "eye_thresh": round(self.eye_thresh, 2)
        }

# ==========================
# 4. FASTAPI APP & WEBSOCKET
# ==========================
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
    # (This includes the encoding="utf-8" fix)
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """The main WebSocket endpoint for video processing."""
    await websocket.accept()
    
    processor = DrowsinessProcessor() 
    
    try:
        while True:
            # (This includes the try/except fix for 0.0 values)
            try:
                data = await websocket.receive_text()
                
                if data == "CALIBRATE":
                    processor.start_calibration()
                    continue
                if data == "RESET":
                    processor.reset_counts()
                    continue

                img = decode_base64_image(data)
                
                if img is None:
                    print("Warning: Received a bad or empty frame, skipping...")
                    continue

                result = processor.process(img)
                await websocket.send_json(result)
            
            except Exception as e:
                # Log the error but keep the loop running
                print(f"Error processing frame: {e}")
                await websocket.send_json({"status": "PROCESS_ERROR"})
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Cleaning up processor for disconnected client.")
        del processor

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access the app at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)