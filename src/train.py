import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from model import build_transfer_model, unfreeze_top_layers
from data_loader import get_generators

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Data ---
train_gen, val_gen = get_generators(
    os.path.expanduser('~/emotion-ai-data/data/raw/train'),
    os.path.expanduser('~/emotion-ai-data/data/raw/test'),
    backbone='mobilenet',
    batch_size=32
)
class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)

# --- Class weights (FER2013 is imbalanced, esp. 'disgust') ---
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_gen.classes), y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# --- Build model ---
model, base = build_transfer_model(num_classes=len(class_names), backbone='mobilenet')

# ============ PHASE 1: Train head only (backbone frozen) ============
print("\n=== PHASE 1: Training classification head ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase1_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'phase1_best.h5'), save_best_only=True
    )
]

history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=phase1_callbacks
)

# ============ PHASE 2: Fine-tune top layers of backbone ============
print("\n=== PHASE 2: Fine-tuning backbone ===")
unfreeze_top_layers(base, num_layers=20)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

phase2_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'phase2_best.h5'), save_best_only=True
    )
]

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=phase2_callbacks
)

# --- Save final model ---
final_path = os.path.join(MODELS_DIR, 'emotion_model_final.h5')
model.save(final_path)
print(f"\nFinal model saved to {final_path}")