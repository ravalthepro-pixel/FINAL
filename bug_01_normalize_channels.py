"""
BUG CASE 1 — Normalize: 3-Channel List → Single Scalar
Student: Jasraj "Jay" Raval
Component: Data Preprocessing (train.py)
GenAI Label: Good

WHAT IS BROKEN:
    transforms.Normalize() receives mean=[0.5] and std=[0.5] (1-element lists)
    but the image tensors have 3 channels (RGB).
    PyTorch raises a RuntimeError on the first batch.

EXPECTED ERROR:
    RuntimeError: input tensor and mean/std must have the same number of channels

TO FIX: Change mean=[0.5] and std=[0.5] back to mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ── Buggy transform ───────────────────────────────────────────────────────────
# BUG: mean and std are 1-element lists, but RGB images have 3 channels
buggy_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),   # <-- BUG: should be [0.5, 0.5, 0.5]
])

# ── Simulate a batch of RGB images and trigger the bug ────────────────────────
print("Bug Case 1: Normalize channel mismatch")
print("=" * 50)
print("Creating a fake RGB image tensor [3, 64, 64]...")

# Create a fake PIL image (RGB, 64x64)
fake_image = Image.fromarray(
    (np.random.rand(64, 64, 3) * 255).astype(np.uint8), mode="RGB"
)

print("Applying buggy transform (Normalize with 1-channel mean/std on RGB image)...")
print("This will crash:\n")

# This line triggers the RuntimeError
buggy_tensor = buggy_transform(fake_image)

print("You should never see this line — the bug should crash above.")
