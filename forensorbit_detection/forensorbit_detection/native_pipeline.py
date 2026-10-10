#!/usr/bin/env python3
"""Native CSI-camera evidence pipeline for a Raspberry Pi 4.

This is the deployment entry point.  It deliberately keeps the raw MP4
unchanged, samples it sparsely, and runs an *NCNN* YOLO model only on those
samples.  The ROS 2 nodes remain in this package for ROS demonstrations, but
are not required by this production camera path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:  # Supports both `python native_pipeline.py` and package execution.
    from .pdf_generator import generate_forensic_pdf_report
except ImportError:
    from pdf_generator import generate_forensic_pdf_report


# native_pipeline.py lives in <project>/forensorbit_detection/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def capture_video(output: Path, seconds: float, width: int, height: int, fps: int,
                  camera_source: str, camera_device: str) -> None:
    """Record an untouched MP4 from the selected camera source."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if camera_source == "usb":
        if not Path(camera_device).exists():
            raise RuntimeError(
                f"USB webcam device {camera_device} was not found. "
                "Run 'v4l2-ctl --list-devices' and pass its capture device with --camera-device."
            )
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed; install it with the Raspberry Pi OS package manager.")
        # Copy the webcam's MJPEG stream directly into MP4. No frames are decoded,
        # resized, annotated, or otherwise modified during evidence capture.
        command = [
            "ffmpeg", "-hide_banner", "-y", "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", camera_device, "-t", str(seconds), "-c:v", "copy", str(output),
        ]
    else:
        if shutil.which("rpicam-vid") is None:
            raise RuntimeError("rpicam-vid is not installed; install Raspberry Pi camera apps first.")
        command = [
            "rpicam-vid", "--timeout", str(max(1, round(seconds * 1000))),
            "--nopreview", "--width", str(width), "--height", str(height),
            "--framerate", str(fps), "--codec", "libav", "--output", str(output),
        ]
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Camera did not create a usable video: {output}")


def load_ncnn_model(model_path: Path):
    """Load the direct NCNN CPU runtime (no PyTorch or CUDA required)."""
    if model_path.suffix == ".pt":
        raise ValueError("Raw .pt inference is disabled on the Pi. Provide an NCNN export directory instead.")
    try:
        import ncnn
    except ImportError as exc:
        raise RuntimeError("NCNN is missing. Install the 'ncnn' package in the project venv.") from exc

    # Determine model files
    if model_path.is_dir() and (model_path / "model.param").is_file() and (model_path / "model.bin").is_file():
        param = model_path / "model.param"
        binary = model_path / "model.bin"
        labels_file = model_path / "labels.txt"
        class_names = {}
        if labels_file.is_file():
            for idx, line in enumerate(labels_file.read_text(encoding="utf-8").strip().splitlines()):
                if line.strip():
                    class_names[idx] = line.strip()
    elif model_path.is_dir() and (model_path / "model.ncnn.param").is_file() and (model_path / "model.ncnn.bin").is_file():
        param = model_path / "model.ncnn.param"
        binary = model_path / "model.ncnn.bin"
        labels_file = model_path / "labels.txt"
        class_names = {}
        if labels_file.is_file():
            for idx, line in enumerate(labels_file.read_text(encoding="utf-8").strip().splitlines()):
                if line.strip():
                    class_names[idx] = line.strip()
    else:
        # Fall back to maintained YOLOv8 NCNN assets installed locally during setup.
        asset_dir = Path.home() / ".ncnn" / "models"
        param, binary = asset_dir / "yolov8s.param", asset_dir / "yolov8s.bin"
        class_names = {0: "person", 43: "knife"}
        if not param.is_file() or not binary.is_file():
            raise FileNotFoundError(f"Model files not found in {model_path} or default {asset_dir}")

    class DirectYoloV8:
        def __init__(self, c_names):
            self.class_names = c_names
            self.net = ncnn.Net()
            self.net.opt.num_threads = 4
            self.net.load_param(str(param))
            self.net.load_model(str(binary))

        def __call__(self, image):
            h, w = image.shape[:2]
            target_size = 416
            scale = min(target_size / w, target_size / h)
            nw, nh = int(w * scale), int(h * scale)
            mat = ncnn.Mat.from_pixels_resize(image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, nw, nh)
            left, right = (target_size - nw) // 2, target_size - nw - (target_size - nw) // 2
            top, bottom = (target_size - nh) // 2, target_size - nh - (target_size - nh) // 2
            mat = ncnn.copy_make_border(mat, top, bottom, left, right, ncnn.BorderType.BORDER_CONSTANT, 114.0)
            mat.substract_mean_normalize([], [1 / 255.0] * 3)
            ex = self.net.create_extractor()
            ex.input("in0", mat)
            _, out = ex.extract("out0")
            cols = np.array(out).T
            boxes, scores, labels = [], [], []
            for col in cols:
                cx, cy, bw, bh = col[:4]
                cls_scores = col[4:]
                label = int(np.argmax(cls_scores))
                score = float(cls_scores[label])
                if score < 0.25:
                    continue
                x1 = (cx - bw / 2.0 - left) / scale
                y1 = (cy - bh / 2.0 - top) / scale
                w_orig = bw / scale
                h_orig = bh / scale
                boxes.append([float(x1), float(y1), float(w_orig), float(h_orig)])
                scores.append(score)
                labels.append(label)
            keep = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45) if boxes else []
            return [
                type(
                    "Detection",
                    (),
                    {
                        "label": labels[int(i)],
                        "prob": scores[int(i)],
                        "rect": type(
                            "Rect",
                            (),
                            {
                                "x": boxes[int(i)][0],
                                "y": boxes[int(i)][1],
                                "w": boxes[int(i)][2],
                                "h": boxes[int(i)][3],
                            },
                        )(),
                    },
                )()
                for i in np.array(keep).reshape(-1)
            ]

    return DirectYoloV8(class_names)


def annotate_and_report(video_path: Path, report_dir: Path, model_path: Path | None,
                        sample_period: float, confidence: float, no_detect: bool) -> Path:
    """Sample a video at wall-clock intervals and create JSON plus annotated positives."""
    if sample_period <= 0:
        raise ValueError("sample_period must be greater than zero.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open recorded video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    report_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = report_dir / "annotated"
    model = None if no_detect else load_ncnn_model(model_path or Path())

    detections: list[dict] = []
    sampled_frames: list[dict] = []
    frame_index = 0
    next_sample_time = 0.0
    positive_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / fps
        if timestamp + 1e-9 < next_sample_time:
            frame_index += 1
            continue
        next_sample_time += sample_period
        sample = {"frame_number": frame_index, "timestamp_seconds": round(timestamp, 3)}
        sampled_frames.append(sample)

        frame_detections: list[dict] = []
        if model is not None:
            for box in model(frame):
                x1, y1 = float(box.rect.x), float(box.rect.y)
                x2, y2 = x1 + float(box.rect.w), y1 + float(box.rect.h)
                class_id = int(box.label)
                score = float(box.prob)
                frame_detections.append({
                    "class_id": class_id,
                    "class": str(model.class_names.get(class_id, f"coco_class_{class_id}")),
                    "confidence": round(score, 4),
                    "bounding_box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "frame_number": frame_index,
                    "timestamp_seconds": round(timestamp, 3),
                })
        if frame_detections:
            annotated_dir.mkdir(exist_ok=True)
            image_name = f"frame_{frame_index:06d}_{timestamp:.3f}s.jpg"
            annotated = frame.copy()
            for detection in frame_detections:
                x1, y1, x2, y2 = (int(value) for value in detection["bounding_box_xyxy"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                label = f'{detection["class"]} {detection["confidence"]:.2f}'
                cv2.putText(annotated, label, (x1, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
                detection["annotated_image"] = str(Path("annotated") / image_name)
            if not cv2.imwrite(str(annotated_dir / image_name), annotated):
                raise RuntimeError(f"Could not write annotated image: {image_name}")
            positive_frames += 1
            detections.extend(frame_detections)
        frame_index += 1
    cap.release()

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "AI findings are advisory and require investigator verification.",
        "source_video": str(video_path.resolve()),
        "video": {"fps": round(fps, 3), "frame_count": frame_total, "width": width, "height": height},
        "sampling": {"period_seconds": sample_period, "sampled_frame_count": len(sampled_frames), "sampled_frames": sampled_frames},
        "model": "not_run" if no_detect else str(model_path.resolve()),
        "detections": detections,
        "positive_frame_count": positive_frames,
    }
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pdf_path = generate_forensic_pdf_report(str(report_dir), report)
    if pdf_path:
        report["pdf_report"] = Path(pdf_path).name
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForensOrbit Pi evidence capture and NCNN detection")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-seconds", type=float, help="Record a new CSI-camera video for this duration")
    source.add_argument("--video", type=Path, help="Analyse an existing raw MP4 without modifying it")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "weights" / "yolov8n_ncnn_model")
    parser.add_argument("--sample-period", type=float, default=1.5, help="Seconds between inference samples (default: 1.5)")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--camera-source", choices=("usb", "rpi"), default="usb",
                        help="Capture source for --record-seconds (default: usb)")
    parser.add_argument("--camera-device", default="/dev/video1",
                        help="V4L2 webcam device for --camera-source usb (default: /dev/video1)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--no-detect", action="store_true", help="Test capture, extraction, and JSON reporting without a model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.expanduser().resolve()
    if args.record_seconds is not None:
        video_path = output_root / "videos" / f"forensorbit_{run_id}.mp4"
        capture_video(video_path, args.record_seconds, args.width, args.height, args.fps,
                      args.camera_source, args.camera_device)
    else:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
    report_dir = output_root / "reports" / video_path.stem
    report_path = annotate_and_report(video_path, report_dir, args.model.expanduser(), args.sample_period, args.confidence, args.no_detect)
    print(f"Raw video: {video_path}")
    print(f"Evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ForensOrbit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
