"""
BUG CASE 7 — EPOCHS: 20 → 0 (No Training Occurs — Silent Failure)
Student: Jasraj "Jay" Raval
Component: Training Configuration (train.py)
GenAI Label: Good

WHAT IS BROKEN:
    EPOCHS = 0 means the training loop body never executes.
    The model stays at random initialization.
    No error is raised. best_model.pth is saved containing only random weights.
    Validation accuracy ≈ 10% (random chance on 10-class balanced problem).
    Loss stays at ln(10) ≈ 2.3026 — cross-entropy of a uniform distribution.

EXPECTED BEHAVIOR (silent failure — no crash):
    Script completes immediately.
    best_val_acc = 0.0 (no epoch ever ran to update it).
    Saved model performs at random chance.

TO FIX: Change EPOCHS = 0 back to EPOCHS = 20
"""

import torch
import torch.nn as nn
import math

# BUG: EPOCHS set to 0 — training loop never runs
EPOCHS = 0   # <-- BUG: should be 20


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)


torch.manual_seed(42)
model     = TinyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Bug Case 7: EPOCHS = 0 — silent failure (no training occurs)")
print("=" * 55)
print(f"EPOCHS = {EPOCHS}  (should be 20)")
print()

best_val_acc = 0.0

# Training loop — never executes because EPOCHS = 0
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch} running...")  # never printed
    best_val_acc = 0.99                 # never updated

print(f"Training loop executed 0 times.")
print(f"best_val_acc = {best_val_acc}  (never updated from 0.0)")
print()

# Evaluate on fake validation data — will show random-chance accuracy
print("Evaluating model that was never trained...")
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for _ in range(10):
        inputs = torch.randn(32, 3, 64, 64)
        labels = torch.randint(0, 10, (32,))
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

accuracy = correct / total
print(f"Validation accuracy: {accuracy:.2%}  (expected ≈ 10% random chance)")
print(f"Cross-entropy loss:  {math.log(10):.4f}  (= ln(10), uniform distribution)")
print()
print("No error was raised. Script appears to have succeeded.")
print("This is a silent failure — only inspection of accuracy reveals the bug.")
print("Fix: change EPOCHS = 0 → EPOCHS = 20")
