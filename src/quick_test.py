import os
import sys

# Add project root to path and anchor all data paths to it
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model import build_transfer_model
from data_loader import get_generators

model, base = build_transfer_model(backbone='mobilenet')
model.summary()

train_gen, val_gen = get_generators(
    os.path.join(PROJECT_ROOT, 'data/raw/train'),
    os.path.join(PROJECT_ROOT, 'data/raw/test'),
    backbone='mobilenet'
)
print("Classes:", train_gen.class_indices)
print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)