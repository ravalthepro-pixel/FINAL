"""
BUG CASE 4 — Dropout: p=0.5 → p=1.0 (Complete Neuron Zeroing)
Student: Jack Sweeney
Component: Model Architecture (model.py)
GenAI Label: Good

WHAT IS BROKEN:
    Dropout(p=1.0) zeros ALL neuron outputs on every forward pass during training.
    The final Linear layer receives an all-zero input every batch.
    Gradients through the weight matrix of the final layer are always zero.
    The model never learns — accuracy stays at ~10% (random chance) for all epochs.
    Loss stays at ln(10) ≈ 2.3026 (cross-entropy of a uniform distribution).

EXPECTED BEHAVIOR (silent failure — no crash):
    Training runs all epochs without error.
    Loss stays ≈ 2.3026 every epoch.
    Accuracy stays ≈ 10% every epoch.
    best_model.pth is saved — but contains only random-init weights.

TO FIX: Change p=1.0 back to p=0.5
"""

import torch
import torch.nn as nn
import math


# ── Buggy classifier head ─────────────────────────────────────────────────────
class BuggyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(256, 256)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=1.0)   # <-- BUG: should be p=0.5
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)               # zeros out ALL activations in training mode
        x = self.fc2(x)
        return x


# ── Demonstrate the bug ───────────────────────────────────────────────────────
torch.manual_seed(42)
model     = BuggyClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Bug Case 4: Dropout(p=1.0) — complete neuron zeroing")
print("=" * 55)
print(f"Expected loss at random chance: ln(10) = {math.log(10):.4f}")
print()
print(f"{'Epoch':>6} | {'Loss':>8} | {'Acc':>7} | {'Note'}")
print("-" * 60)

model.train()
for epoch in range(1, 6):
    inputs = torch.randn(64, 256)
    labels = torch.randint(0, 10, (64,))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss    = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    acc = (outputs.argmax(1) == labels).float().mean().item()

    # Show that activations after dropout are all zero
    with torch.no_grad():
        after_relu    = model.relu(model.fc1(inputs))
        after_dropout = model.dropout(after_relu)
        all_zero      = (after_dropout == 0).all().item()

    print(f"{epoch:>6} | {loss.item():>8.4f} | {acc:>6.1%} | "
          f"dropout output all-zero: {all_zero}")

print()
print(f"Loss is stuck at ≈ {math.log(10):.4f} (ln(10)) — model outputs uniform distribution.")
print("No learning occurred. Weights are effectively unchanged from initialization.")
print("Fix: change Dropout(p=1.0) → Dropout(p=0.5)")
