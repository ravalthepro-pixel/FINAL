"""
BUG CASE 2 — MaxPool kernel_size: 2 → 3 (Spatial Dimension Collapse)
Student: Jack Sweeney
Component: Model Architecture (model.py)
GenAI Label: Bad → revised

WHAT IS BROKEN:
    MaxPool2d uses kernel_size=3, stride=3 instead of kernel_size=2, stride=2.
    Spatial dimensions collapse:
        64 → floor(64/3)=21 → floor(21/3)=7 → floor(7/3)=2
    Flatten produces [B, 512] but Linear expects input of size 8192.
    Crashes with RuntimeError on first forward pass.

EXPECTED ERROR:
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (Bx512 vs 8192x256)

TO FIX: Change kernel_size=3, stride=3 back to kernel_size=2, stride=2
"""

import torch
import torch.nn as nn


class BuggyASL_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BuggyASL_CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),   # <-- BUG: was kernel_size=2, stride=2
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),   # <-- BUG
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),   # <-- BUG
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),   # expects 8192, will receive 512 → CRASH
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Trigger the bug ───────────────────────────────────────────────────────────
print("Bug Case 2: MaxPool kernel_size=3 spatial collapse")
print("=" * 50)

model = BuggyASL_CNN(num_classes=10)

# Trace what actually happens to spatial dims
print("Spatial dimension trace with kernel_size=3, stride=3:")
x = torch.randn(4, 3, 64, 64)
print(f"  Input:        {list(x.shape)}")
for i, layer in enumerate(model.features):
    x = layer(x)
    if isinstance(layer, nn.MaxPool2d):
        print(f"  After Pool {i//4 + 1}:  {list(x.shape)}")
print(f"  After Flatten: {list(x.view(x.size(0), -1).shape)}  ← should be [B, 8192], is [B, 512]")
print()
print("Running forward pass — this will crash:\n")

dummy = torch.randn(4, 3, 64, 64)
output = model(dummy)   # <-- RuntimeError here

print("You should never see this line.")
