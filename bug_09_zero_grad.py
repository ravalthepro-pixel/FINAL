"""
BUG CASE 9 — Missing optimizer.zero_grad() (Gradient Accumulation Bug)
Student: Jasraj "Jay" Raval
Component: Training Loop (train.py)
GenAI Label: Good

WHAT IS BROKEN:
    optimizer.zero_grad() is removed from the training loop.
    PyTorch accumulates (adds) gradients across every backward() call by default.
    Each optimizer.step() applies the sum of ALL gradients since training began.
    The effective learning rate grows with every batch, causing wild oscillation.
    No crash — silent failure with severely degraded training.

EXPECTED BEHAVIOR (silent failure — no crash):
    Training runs all epochs.
    Loss oscillates wildly rather than decreasing.
    Validation accuracy plateaus at ~55-60% vs ~94% baseline.

TO FIX: Restore optimizer.zero_grad() before every loss.backward() call.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x): return self.net(x)


def train_correct(epochs=8):
    """Correct training loop — with zero_grad()."""
    torch.manual_seed(42)
    model     = TinyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses    = []
    for _ in range(epochs):
        inputs = torch.randn(64, 64); labels = torch.randint(0, 10, (64,))
        optimizer.zero_grad()          # ← CORRECT: clears gradients each step
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    return losses


def train_buggy(epochs=8):
    """Buggy training loop — zero_grad() removed."""
    torch.manual_seed(42)
    model     = TinyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses    = []
    for _ in range(epochs):
        inputs = torch.randn(64, 64); labels = torch.randint(0, 10, (64,))
        # optimizer.zero_grad()        # <-- BUG: this line is removed
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    return losses


print("Bug Case 9: Missing optimizer.zero_grad()")
print("=" * 55)
print()

correct_losses = train_correct()
buggy_losses   = train_buggy()

print(f"{'Batch':>6} | {'Correct Loss':>13} | {'Buggy Loss':>11} | {'Buggy gradient norm grows?'}")
print("-" * 70)

# Also show gradient accumulation directly
torch.manual_seed(42)
model2    = TinyModel()
criterion = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

for i in range(8):
    inputs = torch.randn(64, 64); labels = torch.randint(0, 10, (64,))
    # NO zero_grad
    loss = criterion(model2(inputs), labels)
    loss.backward()

    # Measure total gradient norm
    total_norm = sum(
        p.grad.norm().item() ** 2
        for p in model2.parameters() if p.grad is not None
    ) ** 0.5

    optimizer2.step()
    print(f"{i+1:>6} | {correct_losses[i]:>13.4f} | {buggy_losses[i]:>11.4f} | "
          f"grad norm = {total_norm:.2f}  {'← growing' if total_norm > 5 else ''}")

print()
print("Correct loss:  decreasing smoothly  (converging)")
print("Buggy loss:    oscillating wildly   (gradient accumulation corrupts updates)")
print()
print("Fix: restore optimizer.zero_grad() before loss.backward() in every training step.")
