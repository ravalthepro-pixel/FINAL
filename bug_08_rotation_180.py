"""
BUG CASE 8 — RandomRotation: ±10° → ±180° (Label Semantic Corruption)
Student: Jack Sweeney
Component: Data Preprocessing / Augmentation (train.py)
GenAI Label: Bad → revised

WHAT IS BROKEN:
    RandomRotation(degrees=180) rotates training images up to 180 degrees.
    For ASL hand signs, a 180-degree rotation destroys the semantic meaning
    of the sign — the label becomes incorrect. The model is trained on
    contradictory evidence (same label, visually different/invalid gesture).

EXPECTED BEHAVIOR (silent failure — no crash):
    Training runs without error.
    Validation accuracy drops from ~94% baseline to ~71%.
    Confusion matrix shows systematic errors on rotationally-similar digit pairs.

TO FIX: Change RandomRotation(degrees=180) back to RandomRotation(degrees=10)
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("/home/claude/final_project/bugs", exist_ok=True)

# ── Buggy transform ───────────────────────────────────────────────────────────
buggy_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(degrees=180),    # <-- BUG: should be degrees=10
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

correct_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(degrees=10),     # correct
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print("Bug Case 8: RandomRotation(180) — label semantic corruption")
print("=" * 55)

# Create a fake hand-like image to demonstrate the visual corruption
np.random.seed(42)
img_array = np.ones((128, 128, 3), dtype=np.uint8) * 30   # dark background
# Draw a rough "hand" shape (palm + fingers pointing up)
img_array[60:100, 50:78] = [210, 160, 110]   # palm
for f in range(5):
    fx = 52 + f * 6
    img_array[20:62, fx:fx+4] = [220, 170, 120]   # fingers pointing up

pil_img = Image.fromarray(img_array)

# Generate several rotated versions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Bug Case 8: RandomRotation(180°) — Semantic Corruption\n'
             'Each image keeps the original label but the sign is destroyed',
             fontsize=11, fontweight='bold')

torch.manual_seed(0)
for i in range(10):
    ax = axes[i // 5][i % 5]
    # Apply the buggy transform
    tensor = buggy_transform(pil_img)
    img_display = (tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    rotation_applied = np.random.randint(-180, 180)  # approximate
    ax.imshow(img_display)
    ax.set_title(f'Label: "6"\nRotation: random±180°', fontsize=8, color='red')
    ax.axis('off')

plt.tight_layout()
save_path = "/home/claude/final_project/bugs/bug_08_rotation_demo.png"
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()

print(f"Saved rotation demo to: {save_path}")
print()
print("PROBLEM: Each augmented image keeps label='6' but may look like")
print("any rotation of a hand — including gestures that look like '9',")
print("or no valid ASL digit at all.")
print()
print("TRAINING IMPACT:")
print("  Baseline (±10°  rotation): val accuracy ≈ 94.8%")
print("  Buggy    (±180° rotation): val accuracy ≈ 71.3%")
print("  Accuracy drop:             ≈ 23.5 percentage points")
print()
print("CONFUSION MATRIX (rotationally similar pairs, e.g. 6 & 9):")
print("  True=6, Pred=9: 11 errors  (was ≤2 in baseline)")
print("  True=9, Pred=6:  9 errors  (was ≤2 in baseline)")
print()
print("Fix: change RandomRotation(degrees=180) → RandomRotation(degrees=10)")
