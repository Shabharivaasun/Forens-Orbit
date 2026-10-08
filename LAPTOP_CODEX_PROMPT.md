# Prompt for Laptop Codex

Copy everything below into Codex on the laptop.

```text
I have a ForensOrbit Raspberry Pi 4 project. The source package is in the current workspace at:

  ./forensorbit_detection

The Raspberry Pi uses a USB webcam and runs native Python CPU inference only.
Do NOT install Docker, ROS 2, CUDA, TensorRT, NVIDIA packages, or PyTorch on the Raspberry Pi.
Training and NCNN export happen on this laptop only.

The Pi-side source entry point is:

  forensorbit_detection/forensorbit_detection/native_pipeline.py

The Pi will run NumPy, OpenCV, NCNN, and ReportLab. It records untouched MP4 videos,
samples one frame every 1–2 seconds, and produces report.json, a forensic_report.pdf,
and annotated positive frames.

My labelled YOLO-format forensic dataset is at:

  DATASET_PATH

First inspect that path. Do not claim that a forensic model was trained if the dataset is
missing, invalid, empty, or lacks annotations. Tell me the exact problem if so.

Goal: train a small forensic detector on this laptop and export a Raspberry-Pi-safe NCNN bundle.

Requirements:

1. Inspect data.yaml, image/label splits, class names, image counts, and label validity.
2. Train only on the laptop, using a small detector such as YOLOv8n.
3. Prefer 416x416 input. Use 320x320 if the model must be made smaller/faster.
4. Use the dataset's real class names and class-ID order. Expected classes may include knife,
   shell_casing, drug_paraphernalia, and blood_stain, but do not invent classes absent from data.yaml.
5. Evaluate the best model and report precision, recall, mAP50, and per-class results.
6. Export the best model to NCNN.
7. Create this exact output folder:

   ./forensorbit_detection/weights/forensic_model/
   ├── model.param
   ├── model.bin
   ├── labels.txt
   ├── model_info.json
   ├── validation_summary.md
   └── test_images/

8. labels.txt must contain one class per line in exact class-ID order.
9. model_info.json must document: model architecture; training and export package versions;
   input width/height; input tensor name; output tensor name(s); BGR/RGB preprocessing;
   scaling/normalisation; letterboxing; output layout; box-decoding method; recommended
   confidence threshold; NMS IoU threshold; class list; model file SHA-256 hashes; and model size.
10. Keep the final NCNN model under 50 MB if possible. Prefer a reliable FP16 or INT8 export,
    but do not use quantisation if it breaks validation or inference.
11. Put 5–10 non-sensitive validation images in test_images with expected detections documented.
12. Verify model.param and model.bin are non-empty and loadable with an NCNN-compatible runtime
    on the laptop if available.
13. Do not modify the Pi and do not create fake detections or placeholder forensic labels.

At the end, give me the complete forensic_model folder, its size, class list, validation metrics,
known limitations, and an rsync command that copies only the model folder back to the Pi.
```

Replace `DATASET_PATH` with the actual absolute path of the labelled forensic dataset on the laptop.
