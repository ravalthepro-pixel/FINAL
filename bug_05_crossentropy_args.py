"""
BUG CASE 5 — CrossEntropyLoss Arguments Swapped
Student: Jasraj "Jay" Raval
Component: Training Loop (train.py)
GenAI Label: Bad → revised

WHAT IS BROKEN:
    criterion(labels, outputs) — arguments are swapped.
    CrossEntropyLoss expects: criterion(input, target)
        input:  [B, C] float tensor (model logits)
        target: [B]    long tensor  (class indices)
    Passing them in reverse causes an immediate ValueError on the first batch.

EXPECTED ERROR:
    ValueError: Expected target size (B, C), got torch.Size([B])
    or
    RuntimeError: expected scalar type Long but found Float

TO FIX: Change criterion(labels, outputs) → criterion(outputs, labels)
"""

import torch
import torch.nn as nn


criterion = nn.CrossEntropyLoss()

# Simulate what model produces and what labels look like
batch_size  = 32
num_classes = 10

outputs = torch.randn(batch_size, num_classes)  # [32, 10] float — model logits
labels  = torch.randint(0, num_classes, (batch_size,))  # [32] long — class indices

print("Bug Case 5: CrossEntropyLoss arguments swapped")
print("=" * 50)
print(f"outputs shape: {list(outputs.shape)}  dtype: {outputs.dtype}")
print(f"labels  shape: {list(labels.shape)}   dtype: {labels.dtype}")
print()
print("Correct call:  criterion(outputs, labels)  → input=[B,C] float, target=[B] long")
print("Buggy call:    criterion(labels, outputs)  → SWAPPED")
print()
print("Running buggy call — this will crash:\n")

# BUG: arguments are swapped
loss = criterion(labels, outputs)   # <-- BUG: should be criterion(outputs, labels)

print("You should never see this line.")
