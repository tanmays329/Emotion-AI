import os
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'emotion_model_v2.h5')

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FACE_PADDING = 0.05

app = FastAPI(title="Emotion Recognition API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model + face detector once at startup ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)


def get_padded_box(box, w, h, padding=FACE_PADDING):
    bw = box.width * w
    bh = box.height * h
    x = box.xmin * w
    y = box.ymin * h
    pad_x = bw * padding
    pad_y = bh * padding
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(w, int(x + bw + pad_x))
    y2 = min(h, int(y + bh + pad_y))
    return x1, y1, x2, y2


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_array = face_resized.astype('float32')
    face_array = tf.keras.applications.mobilenet_v2.preprocess_input(face_array)
    return np.expand_dims(face_array, axis=0)

@app.get("/")
def root():
    return {"status": "ok", "message": "Emotion Recognition API is running. POST an image to /predict."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if not results.detections:
        return JSONResponse(
            status_code=200,
            content={"faces_detected": 0, "results": []}
        )

    h, w, _ = frame.shape
    response_results = []

    for detection in results.detections:
        box = detection.location_data.relative_bounding_box
        x1, y1, x2, y2 = get_padded_box(box, w, h)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        input_tensor = preprocess_face(face_crop)
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        response_results.append({
            "emotion": CLASS_NAMES[pred_idx],
            "confidence": float(preds[pred_idx]),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "all_probabilities": {
                CLASS_NAMES[i]: float(p) for i, p in enumerate(preds)
            }
        })

    return {"faces_detected": len(response_results), "results": response_results}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

@app.get("/demo")
def demo_page():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")