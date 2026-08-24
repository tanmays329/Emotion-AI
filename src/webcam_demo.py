import os
import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'emotion_model_v2.h5')

CAMERA_INDEX = 1  # DroidCam

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CONFIDENCE_THRESHOLD = 0.30  # below this, show "uncertain" instead of committing
FACE_PADDING = 0.05          # 5% extra margin around detected face box

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

mp_face = mp.solutions.face_detection

history = deque(maxlen=18)  # increased smoothing window


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_array = face_resized.astype('float32')
    face_array = tf.keras.applications.mobilenet_v2.preprocess_input(face_array)
    return np.expand_dims(face_array, axis=0)


def get_padded_box(box, w, h, padding=FACE_PADDING):
    """Expand the MediaPipe bounding box by `padding` fraction on each side, clamped to frame."""
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


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}")
        return

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        print("Webcam demo running. Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                h, w, _ = frame.shape
                best_detection = max(
                    results.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width
                )
                box = best_detection.location_data.relative_bounding_box
                x1, y1, x2, y2 = get_padded_box(box, w, h)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    input_tensor = preprocess_face(face_crop)
                    preds = model.predict(input_tensor, verbose=0)[0]
                    pred_idx = np.argmax(preds)
                    confidence = float(preds[pred_idx])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        emotion = CLASS_NAMES[pred_idx]
                    else:
                        emotion = "uncertain"

                    history.append(emotion)
                    smoothed_emotion = Counter(history).most_common(1)[0][0]

                    box_color = (0, 255, 0) if smoothed_emotion != "uncertain" else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{smoothed_emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            cv2.imshow('Emotion Detection - Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()