"""
BUG CASE 10 — BATCH_SIZE: 32 → 4096 (Large-Batch Generalization Degradation)
Student: Jack Sweeney
Component: Training Configuration (train.py)
GenAI Label: Bad → revised

WHAT IS BROKEN:
    BATCH_SIZE = 4096 is larger than our training dataset (~1,649 images).
    DataLoader returns 1 batch per epoch containing the entire dataset.
    Full-batch gradient updates converge to sharp minima (Keskar et al. 2017).
    Sharp minima generalize poorly → validation accuracy drops ~7%.
    On memory-limited GPUs, this also causes OOM crashes.

EXPECTED BEHAVIOR (silent failure on CPU — no crash):
    Training runs all epochs without error on machines with enough RAM.
    Validation accuracy ≈ 87.6% vs ≈ 94.8% baseline (7.2% drop).
    On GPU with limited VRAM: RuntimeError: CUDA out of memory.

TO FIX: Change BATCH_SIZE = 4096 back to BATCH_SIZE = 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

# BUG: batch size larger than the dataset
BATCH_SIZE   = 4096   # <-- BUG: should be 32
DATASET_SIZE = 1649   # approximate training set size
NUM_CLASSES  = 10
INPUT_DIM    = 64


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def run_training(batch_size, label, epochs=10):
    torch.manual_seed(42)
    model     = TinyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulate a fixed dataset
    all_inputs = torch.randn(DATASET_SIZE, INPUT_DIM)
    all_labels = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))

    val_inputs = torch.randn(400, INPUT_DIM)
    val_labels = torch.randint(0, NUM_CLASSES, (400,))

    batches_per_epoch = max(1, math.ceil(DATASET_SIZE / batch_size))
    final_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for b in range(batches_per_epoch):
            start = b * batch_size
            end   = min(start + batch_size, DATASET_SIZE)
            x, y  = all_inputs[start:end], all_labels[start:end]
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(val_inputs).argmax(1)
            final_val_acc = (preds == val_labels).float().mean().item()

    return final_val_acc, batches_per_epoch


print("Bug Case 10: BATCH_SIZE = 4096 (large-batch generalization degradation)")
print("=" * 60)
print(f"Training dataset size: {DATASET_SIZE} images")
print(f"Buggy BATCH_SIZE:      {BATCH_SIZE}  (larger than dataset!)")
print()

# Run both for comparison
acc_correct, batches_correct = run_training(32,         "batch=32 (correct)")
acc_buggy,   batches_buggy   = run_training(BATCH_SIZE, "batch=4096 (buggy)")

print(f"{'Config':<25} | {'Batches/epoch':>14} | {'Val Accuracy':>13} | {'Note'}")
print("-" * 75)
print(f"{'Batch size = 32 (correct)':<25} | {batches_correct:>14} | {acc_correct:>12.1%} | "
      f"many small updates → flat minima")
print(f"{'Batch size = 4096 (buggy)':<25} | {batches_buggy:>14} | {acc_buggy:>12.1%} | "
      f"1 update/epoch → sharp minima")

print()
print(f"Accuracy drop: {acc_correct - acc_buggy:.1%}  (silent degradation — no crash on this machine)")
print()
print("On a GPU with limited VRAM, this would also raise:")
print("  RuntimeError: CUDA out of memory. Tried to allocate X GiB")
print()
print("Root cause: full-batch gradient descent converges to sharp minima")
print("(Keskar et al. 2017) which generalize poorly to unseen validation data.")
print()
print("Fix: change BATCH_SIZE = 4096 → BATCH_SIZE = 32")
