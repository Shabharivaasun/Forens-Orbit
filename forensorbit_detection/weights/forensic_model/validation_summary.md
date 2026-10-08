# Forensic Model Validation Summary

## Overview
- **Model Architecture:** YOLOv8n (nano)
- **Input Resolution:** 416x416
- **Format:** NCNN FP16 Bundle
- **Classes (7):** Handgun, Knife, Missile, Rifle, Shotgun, Sword, Tank
- **Total Model Size:** 11.53 MB

## Validation Metrics (imgsz=416, 135 Validation Images)

| Metric | Overall |
|---|---|
| **mAP50** | **0.5662** |
| **mAP50-95** | **0.3514** |
| **Precision (P)** | **0.7067** |
| **Recall (R)** | **0.4520** |

### Per-Class Validation Performance

| Class ID | Class Name | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 0 | Handgun | 20 | 20 | 0.907 | 0.450 | 0.657 | 0.423 |
| 1 | Knife | 20 | 29 | 0.686 | 0.552 | 0.682 | 0.480 |
| 2 | Missile | 16 | 24 | 0.861 | 0.458 | 0.564 | 0.290 |
| 3 | Rifle | 22 | 30 | 0.492 | 0.333 | 0.424 | 0.195 |
| 4 | Shotgun | 15 | 19 | 0.536 | 0.305 | 0.323 | 0.133 |
| 5 | Sword | 23 | 26 | 0.727 | 0.538 | 0.629 | 0.518 |
| 6 | Tank | 19 | 32 | 0.738 | 0.528 | 0.685 | 0.420 |

## Verification on Laptop NCNN Runtime
- model.param and model.bin load cleanly with NCNN C++/Python runtime.
- Input tensor: in0, Output tensor: out0 (shape: [1, 11, 3549]).
