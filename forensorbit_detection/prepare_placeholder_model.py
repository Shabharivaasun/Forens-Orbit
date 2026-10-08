#!/usr/bin/env python3
"""Download and export the stock YOLOv8n placeholder to NCNN once.

The .pt file is used only for conversion.  Pi inference in native_pipeline.py
accepts the resulting NCNN directory and rejects .pt files.
"""
from pathlib import Path

from ultralytics import YOLO


target = Path(__file__).resolve().parent / "weights" / "yolov8n_ncnn_model"
target.parent.mkdir(parents=True, exist_ok=True)
result = YOLO("yolov8n.pt").export(format="ncnn")
result_path = Path(result)
if result_path.resolve() != target.resolve():
    if target.exists():
        raise RuntimeError(f"Target already exists: {target}")
    result_path.rename(target)
print(target)
