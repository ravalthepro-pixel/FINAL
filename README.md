# AI 100 — Final Project: AI System Building with GenAI as a Cognitive Partner

**Authors:** Jasraj "Jay" Raval & Jack Sweeney  
**Course:** AI 100 — Deep Learning  
**Date:** April 2026

---

## Overview

This repository extends our midterm ASL hand sign digit classifier CNN for the AI 100 final project. We intentionally introduced **10 bugs** into the system — modifying dataset preprocessing, model architecture, training configuration, and the training loop — and documented each case using the structured self-reflection + GenAI evaluation process required by the course.

## Repository Structure

```
├── model.py                    # ASL_CNN architecture (fixed/working version)
├── train.py                    # Training script (fixed/working version, bug comments inline)
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── bug_cases.xlsx              # 10 bug cases — full table with reflections & GenAI labels
├── final_report.pdf            # Written report: lessons learned about GenAI & AI systems
│
├── bugs/
│   └── README.md               # Summary of all 10 bugs and which files they modified
│
├── results/
│   ├── training_curves.png     # Loss & accuracy curves
│   ├── confusion_matrix.png    # Per-class confusion matrix heatmap
│   ├── sample_predictions.png  # 10 sample validation predictions
│   └── classification_report.txt
│
├── AI 100 Midterm Project.pdf  # Original midterm report
└── Final Project Rubric.pdf    # Course rubric
```

## Bug Cases Summary

| Case | Student | Component Modified | GenAI Label |
|---|---|---|---|
| 1 | Jasraj Raval | Normalize: 3-element → 1-element list | Good |
| 2 | Jack Sweeney | MaxPool kernel_size: 2 → 3 | Bad |
| 3 | Jasraj Raval | Learning rate: 0.001 → 10.0 | Good |
| 4 | Jack Sweeney | Dropout: p=0.5 → p=1.0 | Good |
| 5 | Jasraj Raval | CrossEntropyLoss args swapped | Bad |
| 6 | Jack Sweeney | FC hidden size: 256 → 512 (mismatch) | Good |
| 7 | Jasraj Raval | EPOCHS: 20 → 0 | Good |
| 8 | Jack Sweeney | RandomRotation: 10° → 180° | Bad |
| 9 | Jasraj Raval | optimizer.zero_grad() removed | Good |
| 10 | Jack Sweeney | BATCH_SIZE: 32 → 4096 | Bad |

## Setup & Run

```bash
pip install -r requirements.txt
# Download dataset from Kaggle into data/ folder (one subfolder per digit 0-9)
python train.py
```

## Contributions
- **Jasraj "Jay" Raval:** Cases 1, 3, 5, 7, 9 (preprocessing, optimizer, training loop)
- **Jack Sweeney:** Cases 2, 4, 6, 8, 10 (architecture, augmentation, batch config)
