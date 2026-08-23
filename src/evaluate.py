import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_generators

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load model ---
model_path = os.path.join(MODELS_DIR, 'emotion_model_v2.h5')
model = tf.keras.models.load_model(model_path)

# --- Load val/test data (no shuffle so predictions align with labels) ---
_, val_gen = get_generators(
    os.path.expanduser('~/emotion-ai-data/data/raw/train'),
    os.path.expanduser('~/emotion-ai-data/data/raw/test'),
    backbone='mobilenet',
    batch_size=32
)
class_names = list(val_gen.class_indices.keys())

# --- Predict on entire validation set ---
print("Running predictions on validation set...")
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

# --- Classification report ---
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("\n" + report)

with open(os.path.join(RESULTS_DIR, 'classification_report_v2.txt'), 'w') as f:
    f.write(report)

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix — Emotion Recognition')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_v2.png'), dpi=150)
print(f"\nConfusion matrix saved to {os.path.join(RESULTS_DIR, 'confusion_matrix_v2.png')}")

# --- Normalized confusion matrix (shows per-class % — easier to read with imbalance) ---
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix (row = true class %)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_normalized_v2.png'), dpi=150)
print(f"Normalized confusion matrix saved to {os.path.join(RESULTS_DIR, 'confusion_matrix_normalized_v2.png')}")