"""
BUG CASE 3 — Learning Rate: 0.001 → 10.0 (Gradient Explosion / NaN Loss)
Student: Jasraj "Jay" Raval
Component: Training Configuration (train.py)
GenAI Label: Good

WHAT IS BROKEN:
    Adam optimizer is initialized with lr=10.0 instead of lr=0.001.
    Weight updates are 10,000x too large. Activations overflow to inf,
    then propagate to NaN. Loss becomes NaN within 1-2 epochs.
    The model is unrecoverable without restarting.

EXPECTED BEHAVIOR:
    Loss → NaN at epoch 1 or 2.
    Validation accuracy stays at ~10% (random chance) for all epochs.

TO FIX: Change lr=10.0 back to lr=0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ── Minimal model to demonstrate the bug ──────────────────────────────────────
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)


# ── Simulate training with buggy LR ───────────────────────────────────────────
print("Bug Case 3: Learning rate = 10.0 (gradient explosion)")
print("=" * 50)

torch.manual_seed(42)
model     = TinyModel()
criterion = nn.CrossEntropyLoss()

# BUG: lr=10.0 instead of lr=0.001
optimizer = optim.Adam(model.parameters(), lr=10.0)   # <-- BUG

print("Training with lr=10.0 (should be 0.001)...")
print(f"{'Epoch':>6} | {'Loss':>12} | {'Status'}")
print("-" * 40)

for epoch in range(1, 6):
    # Fake batch: 32 samples, 64 features, 10 classes
    inputs = torch.randn(32, 64)
    labels = torch.randint(0, 10, (32,))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss    = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    status   = "NaN — model destroyed" if torch.isnan(loss) else "unstable"
    print(f"{epoch:>6} | {loss_val:>12.4f} | {status}")

    if torch.isnan(loss):
        print()
        print("Loss is NaN. Training is unrecoverable.")
        print("All subsequent weight updates will also produce NaN.")
        print("Restart training with lr=0.001 to fix.")
        break
