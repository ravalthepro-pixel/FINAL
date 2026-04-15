"""
BUG CASE 6 — FC Hidden Layer: 256 → 512 (Downstream Dimension Mismatch)
Student: Jack Sweeney
Component: Model Architecture (model.py)
GenAI Label: Good

WHAT IS BROKEN:
    FC1 output changed from 256 → 512, but FC2 input still expects 256.
    nn.Linear(128*8*8, 512) outputs [B, 512]
    nn.Linear(256, 10) expects input [B, 256]
    Matrix multiplication is impossible: [B, 512] × [256, 10]
    Crashes with RuntimeError on first forward pass.

EXPECTED ERROR:
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (Bx512 vs 256x10)

TO FIX: Either revert FC1 to nn.Linear(8192, 256),
        or update FC2 to nn.Linear(512, 10) to match the new FC1 output.
"""

import torch
import torch.nn as nn


class BuggyClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(128 * 8 * 8, 512)   # <-- BUG: output changed 256→512
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(256, 10)             # <-- still expects 256, not 512

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)      # outputs [B, 512]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)      # expects [B, 256] — CRASH
        return x


print("Bug Case 6: FC layer dimension mismatch (FC1→512, FC2 expects 256)")
print("=" * 55)

model = BuggyClassifierHead()

# Simulate the flattened output of conv blocks: [B, 128, 8, 8]
fake_conv_output = torch.randn(4, 128, 8, 8)

print(f"Conv output shape:   {list(fake_conv_output.shape)}")
print(f"After flatten:       [4, {128*8*8}]")
print(f"After FC1 (→512):    [4, 512]")
print(f"FC2 weight shape:    [10, 256]  ← expects input of 256, not 512")
print()
print("Running forward pass — this will crash:\n")

output = model(fake_conv_output)   # <-- RuntimeError here

print("You should never see this line.")
